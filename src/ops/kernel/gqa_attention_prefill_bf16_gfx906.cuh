#pragma once

// ninfer::ops - GQA prompt (prefill) attention, BF16 KV-cache kernel, gfx906
// wave64 SIMT edition (stage 4).
//
// Replaces gqa_attention_prefill_bf16_kernel (gqa_attention_prefill_bf16.cuh)
// under NINFER_GFX906_COMPAT: same template surface and argument list, but the
// launcher swaps the grid geometry (Br = 32 query rows per CTA instead of 64)
// and passes no dynamic shared memory — the tensor-core kernel's 96 KiB
// dynamic-smem arena exceeds gfx906's 64 KiB/CU LDS limit, so this kernel owns
// a static ~48 KiB arena instead.
//
// Structure (llama.cpp-gfx906 flash tile pattern, simplified to SIMT; same
// conventions as the stage-4 decode draft in gqa_attention_decode_bf16_gfx906
// .cuh): FlashAttention outer loop over 32-key tiles; K and V tiles staged
// cooperatively (coalesced) into LDS, Q rows staged once per CTA. Within a
// tile, each logical 32-lane warp owns up to 8 query rows round-robin;
// per row: one key per lane with a serial bf16x2 dot product (phase A),
// online softmax via logical-warp shuffle reductions (phase B, exp2_approx /
// Log2E basis exactly as the decode draft), probability-broadcast P*V read
// from the LDS V tile (phase C). Causal masking uses the same bottom-right
// alignment as the tensor-core kernel: query row i attends keys
// [0, base_pos + i]. No ldmatrix, no mma, no cp.async. Correctness-first;
// DPP ladders, half-wave pairing and K-reuse across the q-head group are
// stage-10 tuning items.
//
// LDS layout: rows are padded by 2 bf16 (row stride 258 elements = 129
// dwords) so phase A's per-lane row walks touch 32 distinct banks; staging
// and all LDS accesses use 4-byte (__nv_bfloat162) granularity because the
// 516-byte row stride keeps only 4-byte alignment. A 32-key tile never
// straddles a KV page (page size 64, tiles are 32-aligned), so one physical
// page lookup per tile suffices.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillSimtBr      = 32;  // query rows per CTA
inline constexpr int kGqaPrefillSimtBc      = 32;  // key columns per tile
inline constexpr int kGqaPrefillSimtThreads = 128; // 4 logical 32-lane warps
inline constexpr int kGqaPrefillSimtPad     = 2;   // bf16 row padding (banks)

template <typename Geometry, typename Metadata>
__launch_bounds__(kGqaPrefillSimtThreads) __global__
    void gqa_attention_prefill_simt_bf16_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache_k,
        const __half* __restrict__ cache_v, Metadata metadata,
        const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
        std::int32_t width) {
    constexpr int D           = kGqaPrefillHeadDim;     // 256
    constexpr int Br          = kGqaPrefillSimtBr;      // 32
    constexpr int Bc          = kGqaPrefillSimtBc;      // 32
    constexpr int Threads     = kGqaPrefillSimtThreads; // 128
    constexpr int Warps       = Threads / 32;           // 4
    constexpr int RowsPerWarp = Br / Warps;             // 8
    constexpr int DP          = D + kGqaPrefillSimtPad; // padded LDS row stride
    constexpr int DPerLane    = D / 32;                 // 8 output elems / lane
    constexpr float Log2E     = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Br % Warps == 0);
    // A Bc-tile must sit inside one KV page so a single page lookup covers it.
    static_assert((1 << kPagedKVPageShift) % Bc == 0);

    __shared__ __align__(16) __nv_bfloat16 q_s[Br * DP];
    __shared__ __align__(16) __nv_bfloat16 k_s[Bc * DP];
    __shared__ __align__(16) __half v_s[Bc * DP];

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

    // Stage this CTA's Q rows once (4-byte granularity; see header comment).
    for (int idx = tid; idx < tile_rows * (D / 2); idx += Threads) {
        const int row = idx / (D / 2);
        const int d   = (idx - row * (D / 2)) * 2;
        *reinterpret_cast<__nv_bfloat162*>(&q_s[row * DP + d]) =
            *reinterpret_cast<const __nv_bfloat162*>(
                &q[gqa_prefill_q_index<Geometry>(q_head, d, q0 + row)]);
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
        const std::int64_t tile_base = paged_kv_element_offset<D, Geometry::KVHeads>(
            physical_page, kv_head, k0 & kPagedKVPageMask, 0);

        __syncthreads(); // previous tile's phase C is done reading k_s/v_s

        for (int idx = tid; idx < keys_in_tile * (D / 2); idx += Threads) {
            const int key_l = idx / (D / 2);
            const int d     = (idx - key_l * (D / 2)) * 2;
            const std::int64_t src = tile_base + static_cast<std::int64_t>(key_l) * D + d;
            *reinterpret_cast<__nv_bfloat162*>(&k_s[key_l * DP + d]) =
                *reinterpret_cast<const __nv_bfloat162*>(&cache_k[src]);
            *reinterpret_cast<__half2*>(&v_s[key_l * DP + d]) =
                *reinterpret_cast<const __half2*>(&cache_v[src]);
        }
        __syncthreads();

#pragma unroll
        for (int r_local = 0; r_local < RowsPerWarp; ++r_local) {
            const int r = r_local * Warps + warp;
            if (r >= tile_rows) { continue; }
            const int qabs = base_pos + q0 + r;
            if (k0 > qabs) { continue; } // tile fully beyond this row's causal limit

            // Phase A: lane -> key serial dot product from the LDS tiles.
            const int key       = k0 + lane;
            const bool in_range = lane < keys_in_tile && key <= qabs;
            float score         = -CUDART_INF_F;
            if (in_range) {
                const __nv_bfloat162* q_row_2 =
                    reinterpret_cast<const __nv_bfloat162*>(&q_s[r * DP]);
                const __nv_bfloat162* k_row_2 =
                    reinterpret_cast<const __nv_bfloat162*>(&k_s[lane * DP]);
                float s = 0.0f;
#pragma unroll 8
                for (int p2 = 0; p2 < D / 2; ++p2) {
                    const float2 qf = __bfloat1622float2(q_row_2[p2]);
                    const float2 kf = __bfloat1622float2(k_row_2[p2]);
                    s               = fmaf(qf.x, kf.x, s);
                    s               = fmaf(qf.y, kf.y, s);
                }
                score = s * scale;
            }

            // Phase B: online softmax over the 32 lane scores (decode-draft
            // arithmetic verbatim).
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

            // Phase C: P*V from the LDS V tile. Broadcast each lane's
            // probability; the 32 lanes read that key's V row side by side.
            for (int j = 0; j < Bc; ++j) {
                const float pj = __shfl_sync(FullMask, p, j);
                if (pj == 0.0f) { continue; }
                const __half2* v_pair =
                    reinterpret_cast<const __half2*>(&v_s[j * DP + lane * DPerLane]);
#pragma unroll
                for (int e2 = 0; e2 < DPerLane / 2; ++e2) {
                    const float2 vf     = __half22float2(v_pair[e2]);
                    acc[r_local][2 * e2]     = fmaf(pj, vf.x, acc[r_local][2 * e2]);
                    acc[r_local][2 * e2 + 1] = fmaf(pj, vf.y, acc[r_local][2 * e2 + 1]);
                }
            }
        }
    }

    // Epilogue: normalize and write this warp's rows.
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
