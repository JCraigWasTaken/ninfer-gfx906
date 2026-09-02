#pragma once

// gfx906 register-resident T=1 GEMV for RowSplit grouped-int weights (pass 2).
//
// out[row] (+)= W[row, :] . x[:]
//
// Why a separate kernel: the upstream q5_rowsplit_gemv relies on a cp.async
// pipeline that is a synchronous no-op under compat (memory.cuh pipe_copy),
// stages every weight tile through LDS (occupancy-throttled), dequantizes with
// ~6 scalar VALU ops per value and reduces with two width-32 ds_bpermute
// ladders per wave64. On the MI50 it reaches 20-35 % of DRAM bandwidth
// (docs/gfx906/PASS2-DESIGN.md).
//
// Geometry (donor mmvq geometry, ninfer numerics):
//  - 256 threads = 4 wave64; 32 lanes per row, 2 rows per wave, 8 rows per
//    block. Grid = kN / 8.
//  - A lane owns one 16-byte nibble chunk per step (32 values = half a Q5G64
//    group) plus the matching 4 bytes of the high plane and the group's fp16
//    scale. Loads go straight to registers; kDepth chunks are in flight per
//    lane (64 B nibbles/lane, 4 KB per wave) and the loads for step s+kDepth
//    are issued before step s is decoded (stage-8 register-prefetch pattern).
//  - x is converted BF16 -> FP16 (clamped +-65504) once per block into LDS
//    (kK / 2 dwords; 12 KB at K=6144). Weight reads stay 512 B contiguous per
//    half-wave step.
//  - Decode = Q5TileAtomGfx906::decode_quarter (bias trick, __hmul2 scale)
//    verbatim, twice per chunk; dot = v_dot2_f32_f16 (exact fp16 products,
//    fp32 accumulate); reduce = gfx906_reduce_sum32 (DPP ladder, both 32-lane
//    halves at once).
//  - Epilogue contract is the upstream one: Epilogue::operator()<SplitOutput,
//    SplitRow>(out, out_tail, row, acc) from half-lane 0, residual read from
//    out[row] first when kResidual, so Q5GemvStoreEpilogue / the residual path
//    / Q5GdnDecodeEpilogue plug in unchanged. PDL hooks preserved (no-ops on
//    gfx906).

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Gfx906GemvGeometry {
    static constexpr int kThreads       = 256;
    static constexpr int kLanesPerRow   = 32;
    static constexpr int kRowsPerWave   = 2;
    static constexpr int kRowsPerBlock  = kThreads / kLanesPerRow; // 8
    static constexpr int kValuesPerLane = 32;                      // 16 B of nibbles
    static constexpr int kValuesPerStep = kLanesPerRow * kValuesPerLane; // 1024
};

// One Q5G64 row chunk per lane per step: 16 B nibbles (values [k0, k0+32)),
// 4 B of the high plane, the group's fp16 scale. Lane hl of a half-wave owns
// group step*16 + (hl >> 1), half (hl & 1).
struct Q5GemvChunkGfx906 {
    using Storage = Q5RowSplitStorage;

    struct Raw {
        uint4 words;
        std::uint32_t plane32;
        std::uint16_t scale_bits;
    };

    __device__ static __forceinline__ Raw load(const std::uint8_t* __restrict__ code_row,
                                               const std::uint8_t* __restrict__ high_row,
                                               const std::uint8_t* __restrict__ scale_row,
                                               int step, int hl) {
        const int group = step * (Gfx906GemvGeometry::kLanesPerRow / 2) + (hl >> 1);
        const int half  = hl & 1;
        Raw raw;
        raw.words = *reinterpret_cast<const uint4*>(code_row + group * Storage::kCodeBytesPerGroup +
                                                    16 * half);
        raw.plane32 = *reinterpret_cast<const std::uint32_t*>(
            high_row + group * Storage::kHighBytesPerGroup + 4 * half);
        raw.scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scale_row + group * Storage::kScaleBytesPerGroup);
        return raw;
    }

    // 32 values -> 16 fp16-pair dwords, dst[j] = (value 2j, value 2j+1) scaled.
    __device__ static __forceinline__ void decode(const Raw& raw, std::uint32_t (&dst)[16]) {
        Q5TileAtomGfx906::Raw q;
        q.scale_bits = raw.scale_bits;
        q.words      = make_uint2(raw.words.x, raw.words.y);
        q.plane16    = raw.plane32 & 0xffffu;
        Q5TileAtomGfx906::decode_quarter(q, *reinterpret_cast<std::uint32_t(*)[8]>(&dst[0]));
        q.words   = make_uint2(raw.words.z, raw.words.w);
        q.plane16 = raw.plane32 >> 16;
        Q5TileAtomGfx906::decode_quarter(q, *reinterpret_cast<std::uint32_t(*)[8]>(&dst[8]));
    }
};

template <int kN, int kK, bool kResidual, bool kSplitOutput = false, int kSplitRow = 0,
          class Epilogue = Q5GemvStoreEpilogue, bool TriggerPdl = false, bool JoinPdl = false,
          int kDepth = 2>
__global__ void __launch_bounds__(Gfx906GemvGeometry::kThreads)
q5_gemv_gfx906_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                      const std::uint8_t* __restrict__ high_bits,
                      const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                      __nv_bfloat16* __restrict__ out_tail, Epilogue epilogue = {}) {
    using G     = Gfx906GemvGeometry;
    using Chunk = Q5GemvChunkGfx906;
    constexpr int kGroups   = kK / Chunk::Storage::kGroupK;
    constexpr int kSteps    = kK / G::kValuesPerStep;
    constexpr int kInFlight = kDepth < kSteps ? kDepth : kSteps;
    static_assert(kK % G::kValuesPerStep == 0, "K must be a multiple of 1024");
    static_assert(kN % G::kRowsPerBlock == 0, "N must be a multiple of 8 rows");
    static_assert(kDepth >= 1, "need at least one chunk in flight");
    static_assert(!kSplitOutput || (kSplitRow > 0 && kSplitRow < kN),
                  "split-output GEMV requires an interior compile-time seam");
    static_assert(!kResidual || !kSplitOutput, "the residual GEMV epilogue is contiguous-only");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    // x as fp16 pairs (kK / 2 dwords), 16-byte aligned for uint4 traffic.
    __shared__ __align__(16) std::uint32_t x_sh[kK / 2];

    const int tid    = static_cast<int>(threadIdx.x);
    const int wave   = tid >> 6;
    const int lane64 = tid & 63;
    const int half   = lane64 >> 5;
    const int hl     = lane64 & 31;
    const int row    = static_cast<int>(blockIdx.x) * G::kRowsPerBlock + wave * G::kRowsPerWave + half;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kCodeBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kHighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kScaleBytesPerGroup;

    // Issue the first weight chunk loads before staging x so DRAM requests are
    // in flight during the block's x conversion + barrier.
    typename Chunk::Raw raw[kInFlight];
#pragma unroll
    for (int s = 0; s < kInFlight; ++s) { raw[s] = Chunk::load(code_row, high_row, scale_row, s, hl); }

    {
        // x is 16-byte aligned (activations come from the 256-byte-aligned
        // workspace arena; the tests allocate with cudaMalloc).
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        auto* x_s       = reinterpret_cast<uint4*>(x_sh);
        for (int i = tid; i < kK / 8; i += G::kThreads) {
            const uint4 v = x_g[i];
            uint4 h;
            h.x    = gfx906_bf16x2_to_half2(v.x);
            h.y    = gfx906_bf16x2_to_half2(v.y);
            h.z    = gfx906_bf16x2_to_half2(v.z);
            h.w    = gfx906_bf16x2_to_half2(v.w);
            x_s[i] = h;
        }
    }
    __syncthreads();

    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        const int slot               = s % kInFlight;
        const typename Chunk::Raw cur = raw[slot];
        if (s + kInFlight < kSteps) {
            raw[slot] = Chunk::load(code_row, high_row, scale_row, s + kInFlight, hl);
        }
        std::uint32_t w[16];
        Chunk::decode(cur, w);

        // fp16 x pairs for values [k0, k0+32): 16 dwords = 4 uint4 at dword k0/2.
        const int k0        = (s * (G::kLanesPerRow / 2) + (hl >> 1)) * Chunk::Storage::kGroupK +
                       G::kValuesPerLane * (hl & 1);
        const uint4* x_s4   = reinterpret_cast<const uint4*>(x_sh) + (k0 >> 3);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 xv = x_s4[v];
            acc            = gfx906_fdot2(w[4 * v + 0], xv.x, acc);
            acc            = gfx906_fdot2(w[4 * v + 1], xv.y, acc);
            acc            = gfx906_fdot2(w[4 * v + 2], xv.z, acc);
            acc            = gfx906_fdot2(w[4 * v + 3], xv.w, acc);
        }
    }

    acc = gfx906_reduce_sum32(acc);
    if (hl == 0) {
        if constexpr (kResidual) { acc = __bfloat162float(out[row]) + acc; }
        epilogue.template operator()<kSplitOutput, kSplitRow>(out, out_tail, row, acc);
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

template <int kN, int kK>
inline void q5_gemv_gfx906_launch(const __nv_bfloat16* x, const std::uint8_t* codes,
                                  const std::uint8_t* high_bits, const std::uint8_t* scales,
                                  __nv_bfloat16* out, cudaStream_t stream) {
    constexpr int kGrid = kN / Gfx906GemvGeometry::kRowsPerBlock;
    q5_gemv_gfx906_kernel<kN, kK, false>
        <<<kGrid, Gfx906GemvGeometry::kThreads, 0, stream>>>(x, codes, high_bits, scales, out,
                                                             nullptr);
}

template <int kN, int kK>
inline void q5_gemv_gfx906_residual_launch(const __nv_bfloat16* x, const std::uint8_t* codes,
                                           const std::uint8_t* high_bits,
                                           const std::uint8_t* scales,
                                           __nv_bfloat16* residual_out, cudaStream_t stream) {
    constexpr int kGrid = kN / Gfx906GemvGeometry::kRowsPerBlock;
    q5_gemv_gfx906_kernel<kN, kK, true>
        <<<kGrid, Gfx906GemvGeometry::kThreads, 0, stream>>>(x, codes, high_bits, scales,
                                                             residual_out, nullptr);
}

} // namespace ninfer::ops::detail
