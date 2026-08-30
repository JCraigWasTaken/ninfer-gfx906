#pragma once

// ninfer::ops - GQA prompt (prefill) attention, INT8 KV-cache kernel, gfx906
// wave64 SIMT edition (port-order step 7).
//
// Replaces gqa_attention_prefill_i8_kernel under NINFER_GFX906_COMPAT with the
// same launcher contract as the stage-4 bf16 SIMT prefill kernel (Br = 32
// query rows per CTA, 128 threads, no dynamic smem). The KV cache holds int8
// codes with per-(token, 64-group) FP16 scales, already written by the
// (mma-free, unchanged) i8 fill kernels before this kernel runs.
//
// Deviations from the validated bf16 SIMT prefill kernel
// (gqa_attention_prefill_bf16_gfx906.cuh), which this file otherwise copies
// phase-for-phase:
//   * K tiles are staged into LDS as raw int8 codes (bank-staggered 65-dword
//     rows) plus their FP16 scales; Q rows are quantized once per CTA to int8
//     per (row, 64-group) with FP32 scales. Phase A contracts codes with
//     v_dot4_i32_i8 (__builtin_amdgcn_sdot4) and rescales per 64-group -
//     the NVIDIA int8 prefill kernel's numerical scheme on SIMT.
//   * V tiles are dequantized to bf16 during staging (gqa_kv_dequant_i8x8_from)
//     into the same padded LDS layout the bf16 kernel uses, so phase C and the
//     epilogue are unchanged.
// Static LDS total ~50 KiB - under the 64 KiB/CU limit (one CTA per CU, same
// occupancy as the bf16 edition).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode_i8_gfx906.cuh" // gqa_gfx906_sdot4
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_bf16_gfx906.cuh" // kGqaPrefillSimt* constants
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, typename Metadata>
__launch_bounds__(kGqaPrefillSimtThreads) __global__
    void gqa_attention_prefill_simt_i8_kernel(
        const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
        const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
        const __half* __restrict__ cache_v_scale, Metadata metadata,
        const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
        std::int32_t width) {
    constexpr int D           = kGqaPrefillHeadDim;     // 256
    constexpr int Br          = kGqaPrefillSimtBr;      // 32
    constexpr int Bc          = kGqaPrefillSimtBc;      // 32
    constexpr int Threads     = kGqaPrefillSimtThreads; // 128
    constexpr int Warps       = Threads / 32;           // 4
    constexpr int RowsPerWarp = Br / Warps;             // 8
    constexpr int DP          = D + kGqaPrefillSimtPad; // padded bf16 LDS row stride
    constexpr int DPerLane    = D / 32;                 // 8 output elems / lane
    constexpr int Groups      = kGqaKvQuantGroups;      // 4
    constexpr int GroupDwords = kGqaKvQuantGroup / 4;   // 16 dwords per 64-group
    constexpr int CStride     = D / 4 + 1;              // 65 dwords: bank-staggered code rows
    constexpr float Log2E     = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Br % Warps == 0);
    static_assert(D == kGqaKvQuantHeadDim);
    // A Bc-tile must sit inside one KV page so a single page lookup covers it.
    static_assert((1 << kPagedKVPageShift) % Bc == 0);

    __shared__ __align__(16) __nv_bfloat16 q_s[Br * DP]; // bf16 staging for quantization
    __shared__ __align__(16) int q_i8_s[Br * CStride];
    __shared__ float q_scale_s[Br * Groups];
    __shared__ __align__(16) int k_i8_s[Bc * CStride];
    __shared__ __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __nv_bfloat16 v_s[Bc * DP]; // dequantized at staging

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int window        = max_query_abs + 1; // keys [0, window)
    const int n_tiles       = (window + Bc - 1) / Bc;

    // Stage this CTA's Q rows (4-byte granularity, as the bf16 kernel).
    for (int idx = tid; idx < tile_rows * (D / 2); idx += Threads) {
        const int row = idx / (D / 2);
        const int d   = (idx - row * (D / 2)) * 2;
        *reinterpret_cast<__nv_bfloat162*>(&q_s[row * DP + d]) =
            *reinterpret_cast<const __nv_bfloat162*>(
                &q[gqa_prefill_q_index<Geometry>(q_head, d, q0 + row)]);
    }
    __syncthreads(); // q_s visible to the cross-thread quantization below

    // Quantize the staged Q rows per (row, 64-group): FP32 scale, RNE codes,
    // byte order matching the cache codes. One warp per unit; a lane covers
    // dims d0 and d0+32 of the group.
    for (int unit = warp; unit < tile_rows * Groups; unit += Warps) {
        const int row   = unit / Groups;
        const int group = unit - row * Groups;
        const int d0    = group * kGqaKvQuantGroup + lane;
        const float x0  = __bfloat162float(q_s[row * DP + d0]);
        const float x1  = __bfloat162float(q_s[row * DP + d0 + 32]);
        const float amax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
        const float qs   = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float qinv = qs > 0.0f ? 1.0f / qs : 0.0f;
        if (lane == 0) { q_scale_s[row * Groups + group] = qs; }
        std::int8_t* row_codes =
            reinterpret_cast<std::int8_t*>(q_i8_s) + (row * CStride + group * GroupDwords) * 4;
        row_codes[lane]      = gqa_kv_quant_code(x0, qinv);
        row_codes[lane + 32] = gqa_kv_quant_code(x1, qinv);
    }

    // Per-warp persistent state for its round-robin rows r = r_local*Warps+warp.
    float m[RowsPerWarp];
    float l[RowsPerWarp];
    float acc[RowsPerWarp][DPerLane];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        m[r] = -CUDART_INF_F;
        l[r] = 0.0f;
#pragma unroll
        for (int e = 0; e < DPerLane; ++e) { acc[r][e] = 0.0f; }
    }

    for (int kb = 0; kb < n_tiles; ++kb) {
        const int k0            = kb * Bc;
        const int keys_in_tile  = min(Bc, window - k0);
        const int physical_page = paged_kv_physical_page(block_table, k0);
        const int page_off      = k0 & kPagedKVPageMask;
        const std::int64_t code_base =
            gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, 0, page_off);
        const std::int64_t scale_base =
            gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, 0, page_off);

        __syncthreads(); // previous tile's phase A/C done reading k_i8_s/v_s
                         // (first iteration: Q codes visible before phase A)

        // Stage K codes (64 dwords per key, bank-staggered rows) + K scales.
        for (int idx = tid; idx < keys_in_tile * (D / 4); idx += Threads) {
            const int key_l = idx / (D / 4);
            const int dw    = idx - key_l * (D / 4);
            k_i8_s[key_l * CStride + dw] = reinterpret_cast<const int*>(
                cache_k + code_base)[key_l * (D / 4) + dw];
        }
        for (int idx = tid; idx < keys_in_tile * Groups; idx += Threads) {
            const int key_l = idx / Groups;
            const int g     = idx - key_l * Groups;
            k_scale_s[key_l * Groups + g] = cache_k_scale[scale_base + key_l * Groups + g];
        }
        // Stage V dequantized to bf16 (8 codes per unit, one 64-bit load).
        for (int idx = tid; idx < keys_in_tile * (D / 8); idx += Threads) {
            const int key_l = idx / (D / 8);
            const int d0    = (idx - key_l * (D / 8)) * 8;
            const float vs  = __half2float(cache_v_scale[scale_base + key_l * Groups + (d0 >> 6)]);
            const int4 dq   = gqa_kv_dequant_i8x8_from(
                cache_v + code_base + static_cast<std::int64_t>(key_l) * D + d0, vs);
            int* dst = reinterpret_cast<int*>(&v_s[key_l * DP + d0]);
            dst[0]   = dq.x;
            dst[1]   = dq.y;
            dst[2]   = dq.z;
            dst[3]   = dq.w;
        }
        __syncthreads();

#pragma unroll
        for (int r_local = 0; r_local < RowsPerWarp; ++r_local) {
            const int r = r_local * Warps + warp;
            if (r >= tile_rows) { continue; }
            const int qabs = base_pos + q0 + r;
            if (k0 > qabs) { continue; } // tile fully beyond this row's causal limit

            // Phase A: lane -> key integer dot product from the LDS code
            // tiles (sdot4 per 4-code dword, rescale per 64-group).
            const int key       = k0 + lane;
            const bool in_range = lane < keys_in_tile && key <= qabs;
            float score         = -CUDART_INF_F;
            if (in_range) {
                const int* q_dw   = &q_i8_s[r * CStride];
                const int* k_dw   = &k_i8_s[lane * CStride];
                const float* q_sc = &q_scale_s[r * Groups];
                const __half* k_sc = &k_scale_s[lane * Groups];
                float s           = 0.0f;
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

            // Phase B: online softmax (bf16 SIMT kernel arithmetic verbatim).
            const float bm    = warp_max(score, FullMask);
            const float nm    = fmaxf(m[r_local], bm);
            const float alpha = (m[r_local] == -CUDART_INF_F)
                                    ? 0.0f
                                    : exp2_approx((m[r_local] - nm) * Log2E);
            const float p     = (nm > -CUDART_INF_F && score > -CUDART_INF_F)
                                    ? exp2_approx((score - nm) * Log2E)
                                    : 0.0f;
            const float bl    = warp_sum(p, FullMask);
            l[r_local]        = l[r_local] * alpha + bl;
            m[r_local]        = nm;
#pragma unroll
            for (int e = 0; e < DPerLane; ++e) { acc[r_local][e] *= alpha; }

            // Phase C: P*V from the dequantized LDS V tile (bf16 kernel
            // verbatim).
            for (int j = 0; j < Bc; ++j) {
                const float pj = __shfl_sync(FullMask, p, j);
                if (pj == 0.0f) { continue; }
                const __nv_bfloat162* v_pair =
                    reinterpret_cast<const __nv_bfloat162*>(&v_s[j * DP + lane * DPerLane]);
#pragma unroll
                for (int e2 = 0; e2 < DPerLane / 2; ++e2) {
                    const float2 vf          = __bfloat1622float2(v_pair[e2]);
                    acc[r_local][2 * e2]     = fmaf(pj, vf.x, acc[r_local][2 * e2]);
                    acc[r_local][2 * e2 + 1] = fmaf(pj, vf.y, acc[r_local][2 * e2 + 1]);
                }
            }
        }
    }

    // Epilogue: normalize and write this warp's rows (bf16 kernel verbatim).
#pragma unroll
    for (int r_local = 0; r_local < RowsPerWarp; ++r_local) {
        const int r = r_local * Warps + warp;
        if (r >= tile_rows) { continue; }
        const int qrow    = q0 + r;
        const float inv_l = (l[r_local] > 0.0f) ? 1.0f / l[r_local] : 0.0f;
        const int d0      = lane * DPerLane;
#pragma unroll
        for (int e = 0; e < DPerLane; ++e) {
            out[gqa_prefill_q_index<Geometry>(q_head, d0 + e, qrow)] =
                __float2bfloat16(acc[r_local][e] * inv_l);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Threads);
}

} // namespace ninfer::ops
