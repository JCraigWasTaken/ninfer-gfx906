#pragma once

// ninfer::ops - split-KV GQA small-T attention, INT8 KV-cache partial kernel,
// gfx906 wave64 SIMT edition (port-order step 7).
//
// Drop-in replacement for gqa_attention_decode_i8_tiled_kernel under
// NINFER_GFX906_COMPAT: same grid meaning (KVHeads x splits x batch) and the
// same partial_acc/partial_m/partial_l contract, so the existing mma-free
// split reducer consumes its output unchanged (reducer instantiated with
// Int8=true; the active-split policy here matches via
// gqa_small_t_active_splits<Geometry, true>).
//
// Structure is the validated stage-4/5 bf16 SIMT decode kernel
// (gqa_attention_decode_bf16_gfx906.cuh) with the int8-KV codec fused in,
// keeping the NVIDIA int8 kernel's numerical scheme:
//   * QK is integer: Q is quantized on-chip per (row, 64-group) with an FP32
//     scale; K codes are read straight from the cache and contracted with
//     v_dot4_i32_i8 (__builtin_amdgcn_sdot4, the llama.cpp-gfx906 mmvq
//     idiom), then rescaled by qs[row,g] * ks[key,g] per 64-group.
//   * PV dequantizes V codes inline (float(code) * vscale) into the FP32
//     accumulator - the bf16 kernel's probability-broadcast loop with the
//     dequant folded into the FMA.
//   * Like the NVIDIA int8 kernel (and unlike the bf16 SIMT kernel), ALL
//     keys - history and the newly appended tokens - are read back from the
//     quantized cache: the owning split's fused append quantizes and writes
//     the new rows first and the CTA-wide barrier before the key loop orders
//     the in-block readback. No from_new special-casing, so new and cached
//     keys see identical quantization error.
// Append quantization math (per-token, per-64-group absmax -> FP16-rounded
// scale, RNE symmetric-clamp codes) is copied from
// gqa_attention_prefill_fill_i8_kernel so decode-appended rows are
// bit-identical to prefill-filled rows.
//
// Correctness-first: DPP ladders, K-code LDS staging, and half-wave pairing
// remain stage-10 tuning items.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops {

// v_dot4_i32_i8: 4-way int8 dot product with int32 accumulate (gfx906 sdot4).
__device__ __forceinline__ int gqa_gfx906_sdot4(int a, int b, int c) {
#if defined(__gfx906__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const std::int8_t* pa = reinterpret_cast<const std::int8_t*>(&a);
    const std::int8_t* pb = reinterpret_cast<const std::int8_t*>(&b);
#pragma unroll
    for (int i = 0; i < 4; ++i) { c += static_cast<int>(pa[i]) * static_cast<int>(pb[i]); }
    return c;
#endif
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128) __global__ void gqa_attention_small_t_simt_partial_i8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k,
    std::int8_t* cache_v, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Bc      = 32; // one key per logical lane
    constexpr int D       = kGqaHeadDim;
    constexpr int Threads = Wc * 32;
    // The GQA Op's 262144-key maximum envelope spans at most 49 pages in one 27B split.
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int MaxRows       = TokenTile * Geometry::GroupSize;
    constexpr int DPerLane      = D / 32;           // 8 output elems / lane
    constexpr int Groups        = kGqaKvQuantGroups;
    constexpr int GroupDwords   = kGqaKvQuantGroup / 4; // 16 dwords per 64-group
    constexpr int QStride       = D / 4 + 1;            // 65 dwords: bank-staggered rows

    static_assert(D == kGqaKvQuantHeadDim);

    __shared__ __align__(16) int q_i8_s[MaxRows * QStride];
    __shared__ float q_scale_s[MaxRows * Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > MaxRows || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    // Int8=true: must agree with the reducer's active-split policy.
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // Fused append: the owning split quantizes and writes each new row
        // (codes + per-64-group scales), with the exact
        // gqa_attention_prefill_fill_i8_kernel math. One warp per
        // (token, group) unit; a lane covers dims d0 and d0+32 of the group.
        const int units = valid_tokens * Groups;
        for (int unit = warp; unit < units; unit += Wc) {
            const int token = unit / Groups;
            const int group = unit - token * Groups;
            const int p_tok = pos[token];
            const bool owned =
                p_tok >= split_start && p_tok < split_end && p_tok >= 0 && p_tok < logical_capacity;
            if (!owned) { continue; }
            const int d0            = group * kGqaKvQuantGroup + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            const float k0          = __bfloat162float(input.k[src0]);
            const float k1          = __bfloat162float(input.k[src1]);
            const float v0          = __bfloat162float(input.v[src0]);
            const float v1          = __bfloat162float(input.v[src1]);

            float k_abs = fmaxf(fabsf(k0), fabsf(k1));
            float v_abs = fmaxf(fabsf(v0), fabsf(v1));
            k_abs       = warp_max(k_abs, FullMask);
            v_abs       = warp_max(v_abs, FullMask);

            const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 127.0f : 0.0f);
            const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / 127.0f : 0.0f);
            const float ks   = __half2float(ksh);
            const float vs   = __half2float(vsh);
            const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
            const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;

            int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
            physical_page     = __shfl_sync(FullMask, physical_page, 0);
            const int page_off = p_tok & kPagedKVPageMask;

            const std::int64_t code_base = gqa_kv_quant_code_index<Geometry>(
                physical_page, kv_head, group * kGqaKvQuantGroup, page_off);
            cache_k[code_base + lane]      = gqa_kv_quant_code(k0, kinv);
            cache_k[code_base + lane + 32] = gqa_kv_quant_code(k1, kinv);
            cache_v[code_base + lane]      = gqa_kv_quant_code(v0, vinv);
            cache_v[code_base + lane + 32] = gqa_kv_quant_code(v1, vinv);
            if (lane == 0) {
                const std::int64_t scale_off =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, group, page_off);
                cache_k_scale[scale_off] = ksh;
                cache_v_scale[scale_off] = vsh;
            }
        }
    }

    // Quantize the CTA's Q rows per (row, 64-group) with FP32 scales into the
    // bank-staggered LDS code tile (byte order matches the cache codes, so
    // sdot4 contracts matching dims). One warp per unit; invalid heads write
    // zero codes and a zero scale exactly as the bf16 kernel staged zero rows.
    {
        const int q_units = row_count * Groups;
        for (int unit = warp; unit < q_units; unit += Wc) {
            const int row   = unit / Groups;
            const int group = unit - row * Groups;
            int q_head      = 0;
            int token       = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            const int d0 = group * kGqaKvQuantGroup + lane;
            float x0     = 0.0f;
            float x1     = 0.0f;
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                x0 = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0, token)]);
                x1 = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0 + 32, token)]);
            }
            const float amax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs   = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float qinv = qs > 0.0f ? 1.0f / qs : 0.0f;
            if (lane == 0) { q_scale_s[row * Groups + group] = qs; }
            std::int8_t* row_codes = reinterpret_cast<std::int8_t*>(q_i8_s) +
                                     (static_cast<std::int64_t>(row) * QStride +
                                      static_cast<std::int64_t>(group) * GroupDwords) *
                                         4;
            row_codes[lane]      = gqa_kv_quant_code(x0, qinv);
            row_codes[lane + 32] = gqa_kv_quant_code(x1, qinv);
        }
    }
    __syncthreads(); // pages, appended cache rows, and Q codes all visible

    // One logical 32-lane warp walks one (q_head, token) row over the whole
    // split range; rows are assigned round-robin across the CTA's warps.
    for (int row = warp; row < row_count; row += Wc) {
        int q_head = 0;
        int token  = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        const int qabs        = pos[token];
        const int* q_dw       = &q_i8_s[row * QStride];
        const float* q_sc     = &q_scale_s[row * Groups];

        float m = -CUDART_INF_F;
        float l = 0.0f;
        float acc[DPerLane];
#pragma unroll
        for (int e = 0; e < DPerLane; ++e) { acc[e] = 0.0f; }

        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0            = first_tile + kb * Bc;
            const int physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
            const int key           = k0 + lane;
            const bool in_range     = key >= split_start && key < split_end && key <= qabs;

            // Phase A: lane -> key integer dot product. sdot4 over 4-code
            // dwords per 64-group, rescaled by qs[row,g] * ks[key,g].
            float score = -CUDART_INF_F;
            if (in_range) {
                const int page_off = key & kPagedKVPageMask;
                const int* k_dw    = reinterpret_cast<const int*>(
                    cache_k + gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, 0, page_off));
                const __half* k_sc =
                    cache_k_scale +
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, 0, page_off);
                float s = 0.0f;
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    int ig = 0;
#pragma unroll
                    for (int i = 0; i < GroupDwords; ++i) {
                        ig = gqa_gfx906_sdot4(k_dw[g * GroupDwords + i], q_dw[g * GroupDwords + i],
                                              ig);
                    }
                    s = fmaf(static_cast<float>(ig), q_sc[g] * __half2float(k_sc[g]), s);
                }
                score = s * scale;
            }

            // Phase B: per-row online softmax over the 32 lane scores
            // (bf16 SIMT kernel arithmetic verbatim).
            const float bm    = warp_max(score, FullMask);
            const float nm    = fmaxf(m, bm);
            const float alpha = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
            const float p     = (nm > -CUDART_INF_F && score > -CUDART_INF_F)
                                    ? exp2_approx((score - nm) * Log2E)
                                    : 0.0f;
            const float bl    = warp_sum(p, FullMask);
            l                 = l * alpha + bl;
            m                 = nm;
#pragma unroll
            for (int e = 0; e < DPerLane; ++e) { acc[e] *= alpha; }

            // Phase C: P*V. Broadcast each lane's probability; the 32 lanes
            // read that key's V codes side by side (8 codes per lane, one
            // 64-group per 8 lanes) and dequantize into the FMA.
            for (int j = 0; j < Bc; ++j) {
                const float pj = __shfl_sync(FullMask, p, j);
                if (pj == 0.0f) { continue; }
                const int keyj     = k0 + j;
                const int page_off = keyj & kPagedKVPageMask;
                const std::int64_t v_base =
                    gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, 0, page_off);
                const float vs = __half2float(
                    cache_v_scale[gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head,
                                                                     lane >> 3, page_off)]);
                const int2 raw       = load_vec<int2>(cache_v + v_base + lane * DPerLane);
                const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
                const float pv       = pj * vs;
#pragma unroll
                for (int e = 0; e < DPerLane; ++e) {
                    acc[e] = fmaf(pv, static_cast<float>(c[e]), acc[e]);
                }
            }
        }

        if (lane == 0) {
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l;
        }
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const int d0 = lane * DPerLane;
#pragma unroll
            for (int e = 0; e < DPerLane; ++e) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d0 + e, token, split, tokens)] =
                    __float2bfloat16(acc[e]);
            }
        }
    }
}

} // namespace ninfer::ops
