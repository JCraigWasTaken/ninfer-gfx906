#pragma once

// ninfer::ops - split-KV GQA small-T attention, BF16 KV-cache partial kernel,
// gfx906 wave64 SIMT edition (stage-4 DRAFT).
//
// *** UNVALIDATED: this kernel COMPILES for gfx906 but has never executed on
// *** hardware. Numerical correctness is untested until an MI50 is available.
//
// Drop-in replacement for gqa_attention_small_t_tc_partial_bf16_kernel
// (gqa_attention_decode_bf16.cuh): same template parameters, same argument
// list, same grid geometry (KVHeads x splits x batch, WarpsPerCta*32 threads),
// same partial_acc/partial_m/partial_l contract, so the existing mma-free
// split reducer (gqa_attention_small_t_reduce_output_kernel) consumes its
// output unchanged.
//
// Kernel structure follows the llama.cpp-gfx906 flash-decode pattern
// (iacopPBK fork, fattn-q8.cuh / gfx906-common.cuh): one lane owns one key
// per 32-key tile and computes a serial dot product; per-row online softmax
// uses logical-32-lane warp reductions (the compat __shfl shims subdivide the
// 64-lane wave, so every helper keeps CUDA semantics); the P*V accumulation
// broadcasts each lane's probability and reads V rows coalesced (32 lanes x
// 8 bf16 = one 512B line). No ldmatrix, no mma, no cross-half-wave traffic.
// DPP-ladder reductions and half-wave pairing are deliberately left to the
// stage-10 tuning pass; this draft optimizes for a verifiable 1:1 mapping to
// the tensor-core kernel's semantics:
//   - identical split/window arithmetic and active-split policy
//   - identical cache-append responsibility (owning split writes new rows)
//   - identical new-vs-cached K/V source selection and causal/range masks
//   - identical neutral-partial writes for empty/invalid splits
//   - identical exp2_approx softmax basis (Log2E), bf16-rounded partial acc

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128) __global__ void gqa_attention_small_t_simt_partial_bf16_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
    __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
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
    constexpr int DPerLane      = D / 32; // 8 bf16 output elements per lane

    __shared__ __align__(16) __nv_bfloat16 q_s[MaxRows * D];
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
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
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
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page     = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
    }

    // Stage the Q rows for this KV head un-swizzled; softmax dot products read
    // them as full-warp broadcasts (conflict-free on LDS).
    for (int idx = tid; idx < row_count * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        q_s[row * D + d] = value;
    }
    __syncthreads();

    // Resolve the K / V source row for one absolute key position, honoring
    // the new-tokens-from-input rule of the tensor-core kernel. The input
    // planes are only named inside the constexpr-guarded branch because
    // GqaCachedInput has no k/v members.
    const auto resolve_k = [&](int key, int physical_page) -> const __nv_bfloat16* {
        if constexpr (CacheInput::writes_cache) {
            const int new_token = key - first_pos;
            if (new_token >= 0 && new_token < valid_tokens) {
                return input.k + gqa_kv_new_index<Geometry>(kv_head, 0, new_token);
            }
        }
        return cache_k +
               gqa_cache_index<Geometry>(physical_page, kv_head, 0, key & kPagedKVPageMask);
    };
    const auto resolve_v = [&](int key, int physical_page) -> const __nv_bfloat16* {
        if constexpr (CacheInput::writes_cache) {
            const int new_token = key - first_pos;
            if (new_token >= 0 && new_token < valid_tokens) {
                return input.v + gqa_kv_new_index<Geometry>(kv_head, 0, new_token);
            }
        }
        return cache_v +
               gqa_cache_index<Geometry>(physical_page, kv_head, 0, key & kPagedKVPageMask);
    };

    // One logical 32-lane warp walks one (q_head, token) row over the whole
    // split range; rows are assigned round-robin across the CTA's warps.
    for (int row = warp; row < row_count; row += Wc) {
        int q_head = 0;
        int token  = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        const int qabs                = pos[token];
        const __nv_bfloat16* q_row    = &q_s[row * D];
        const __nv_bfloat162* q_row_2 = reinterpret_cast<const __nv_bfloat162*>(q_row);

        float m = -CUDART_INF_F;
        float l = 0.0f;
        float acc[DPerLane];
#pragma unroll
        for (int e = 0; e < DPerLane; ++e) { acc[e] = 0.0f; }

        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0            = first_tile + kb * Bc;
            const int physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
            const int key           = k0 + lane;
            const bool in_range = key >= split_start && key < split_end && key <= qabs;

            // Phase A: lane -> key dot product (serial over D, bf16x2 steps).
            float score = -CUDART_INF_F;
            if (in_range) {
                const __nv_bfloat16* k_row    = resolve_k(key, physical_page);
                const __nv_bfloat162* k_row_2 = reinterpret_cast<const __nv_bfloat162*>(k_row);
                float s                       = 0.0f;
#pragma unroll 8
                for (int p2 = 0; p2 < D / 2; ++p2) {
                    const float2 qf = __bfloat1622float2(q_row_2[p2]);
                    const float2 kf = __bfloat1622float2(k_row_2[p2]);
                    s               = fmaf(qf.x, kf.x, s);
                    s               = fmaf(qf.y, kf.y, s);
                }
                score = s * scale;
            }

            // Phase B: per-row online softmax over the 32 lane scores.
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

            // Phase C: P*V. Broadcast each lane's probability; all 32 lanes
            // then read that key's V row coalesced (lane*8 bf16 apiece).
            for (int j = 0; j < Bc; ++j) {
                const float pj = __shfl_sync(FullMask, p, j);
                if (pj == 0.0f) { continue; }
                const int keyj             = k0 + j;
                const __nv_bfloat16* v_row = resolve_v(keyj, physical_page);
                const __nv_bfloat162* v_pair =
                    reinterpret_cast<const __nv_bfloat162*>(v_row + lane * DPerLane);
#pragma unroll
                for (int e2 = 0; e2 < DPerLane / 2; ++e2) {
                    const float2 vf = __bfloat1622float2(v_pair[e2]);
                    acc[2 * e2]     = fmaf(pj, vf.x, acc[2 * e2]);
                    acc[2 * e2 + 1] = fmaf(pj, vf.y, acc[2 * e2 + 1]);
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
