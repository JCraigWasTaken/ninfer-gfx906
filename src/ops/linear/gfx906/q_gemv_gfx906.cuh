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
//  - kStageX=false (pass 2c, K=17408): x is NOT staged in LDS and there is no
//    block barrier; each lane reads its own 64 B of x per step straight from
//    global (L2 hits after the first block) and converts BF16 -> FP16 in
//    registers. Removes the 34 KB LDS occupancy limit at K=17408.

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

// Pass 2g: LDS layout of the staged x in the T=1 kernels. Natural layout has
// lane hl reading 4 uint4 at a 64 B lane stride (banks 0/16 only: a 16-way
// ds_read_b128 conflict, the pass-2e finding). true = the small-T
// [step][v][lane] layout (gfx906_stage_x_slab at T=1): uint4 slot
// step * 128 + v * 32 + hl, so every ds_read_b128 is wave-wide contiguous.
constexpr bool kGfx906GemvXSwizzle = true;

// Natural uint4 index i = step * 128 + hl * 4 + v -> swizzled slot.
__device__ __forceinline__ int gfx906_swizzle_x_slot(int i) {
    return (i & ~127) | ((i & 3) << 5) | ((i >> 2) & 31);
}

template <int kN, int kK, bool kResidual, bool kSplitOutput = false, int kSplitRow = 0,
          class Epilogue = Q5GemvStoreEpilogue, bool TriggerPdl = false, bool JoinPdl = false,
          int kDepth = 2, bool kStageX = true, int kThreads = Gfx906GemvGeometry::kThreads,
          bool kSwizzleX = kGfx906GemvXSwizzle>
__global__ void __launch_bounds__(kThreads)
q5_gemv_gfx906_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                      const std::uint8_t* __restrict__ high_bits,
                      const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                      __nv_bfloat16* __restrict__ out_tail, Epilogue epilogue = {}) {
    using G     = Gfx906GemvGeometry;
    using Chunk = Q5GemvChunkGfx906;
    constexpr int kGroups   = kK / Chunk::Storage::kGroupK;
    constexpr int kSteps    = kK / G::kValuesPerStep;
    constexpr int kInFlight = kDepth < kSteps ? kDepth : kSteps;
    // Rows per block follow the block size (kThreads / 32): 256 threads = 8
    // rows; 1024 threads = 16 waves = 32 rows sharing one LDS copy of x.
    constexpr int kRowsPerBlock = kThreads / G::kLanesPerRow;
    static_assert(kThreads % 64 == 0 && kThreads >= 64 && kThreads <= 1024, "1..16 wave64 per block");
    static_assert(kK % G::kValuesPerStep == 0, "K must be a multiple of 1024");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of the rows per block");
    static_assert(kDepth >= 1, "need at least one chunk in flight");
    static_assert(!kSplitOutput || (kSplitRow > 0 && kSplitRow < kN),
                  "split-output GEMV requires an interior compile-time seam");
    static_assert(!kResidual || !kSplitOutput, "the residual GEMV epilogue is contiguous-only");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    // x as fp16 pairs (kK / 2 dwords), 16-byte aligned for uint4 traffic
    // (unused 16 B placeholder when x is read through L2).
    __shared__ __align__(16) std::uint32_t x_sh[kStageX ? kK / 2 : 4];

    const int tid    = static_cast<int>(threadIdx.x);
    const int wave   = tid >> 6;
    const int lane64 = tid & 63;
    const int half   = lane64 >> 5;
    const int hl     = lane64 & 31;
    const int row    = static_cast<int>(blockIdx.x) * kRowsPerBlock + wave * G::kRowsPerWave + half;

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

    if constexpr (kStageX) {
        // x is 16-byte aligned (activations come from the 256-byte-aligned
        // workspace arena; the tests allocate with cudaMalloc).
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        auto* x_s       = reinterpret_cast<uint4*>(x_sh);
        for (int i = tid; i < kK / 8; i += kThreads) {
            const uint4 v = x_g[i];
            uint4 h;
            h.x    = gfx906_bf16x2_to_half2(v.x);
            h.y    = gfx906_bf16x2_to_half2(v.y);
            h.z    = gfx906_bf16x2_to_half2(v.z);
            h.w    = gfx906_bf16x2_to_half2(v.w);
            x_s[kSwizzleX ? gfx906_swizzle_x_slot(i) : i] = h;
        }
        __syncthreads();
    }

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
        const int k0 = (s * (G::kLanesPerRow / 2) + (hl >> 1)) * Chunk::Storage::kGroupK +
                       G::kValuesPerLane * (hl & 1);
        if constexpr (kStageX) {
            // swizzled: slots s * 128 + v * 32 + hl; natural: 4 uint4 at k0 / 8.
            constexpr int kVStride = kSwizzleX ? G::kLanesPerRow : 1;
            const uint4* x_s4      = reinterpret_cast<const uint4*>(x_sh) +
                                (kSwizzleX ? s * (G::kValuesPerStep / 8) + hl : (k0 >> 3));
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const uint4 xv = x_s4[v * kVStride];
                acc            = gfx906_fdot2(w[4 * v + 0], xv.x, acc);
                acc            = gfx906_fdot2(w[4 * v + 1], xv.y, acc);
                acc            = gfx906_fdot2(w[4 * v + 2], xv.z, acc);
                acc            = gfx906_fdot2(w[4 * v + 3], xv.w, acc);
            }
        } else {
            // Same 32 values straight from global as bf16 (64 B contiguous per
            // lane, 16-byte aligned), converted to fp16 pairs in registers.
            const uint4* x_g4 = reinterpret_cast<const uint4*>(x + k0);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const uint4 xb = x_g4[v];
                acc            = gfx906_fdot2(w[4 * v + 0], gfx906_bf16x2_to_half2(xb.x), acc);
                acc            = gfx906_fdot2(w[4 * v + 1], gfx906_bf16x2_to_half2(xb.y), acc);
                acc            = gfx906_fdot2(w[4 * v + 2], gfx906_bf16x2_to_half2(xb.z), acc);
                acc            = gfx906_fdot2(w[4 * v + 3], gfx906_bf16x2_to_half2(xb.w), acc);
            }
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

template <int kN, int kK, int kDepth = 2, bool kStageX = true,
          int kThreads = Gfx906GemvGeometry::kThreads>
inline void q5_gemv_gfx906_residual_launch(const __nv_bfloat16* x, const std::uint8_t* codes,
                                           const std::uint8_t* high_bits,
                                           const std::uint8_t* scales,
                                           __nv_bfloat16* residual_out, cudaStream_t stream) {
    constexpr int kGrid = kN / (kThreads / Gfx906GemvGeometry::kLanesPerRow);
    q5_gemv_gfx906_kernel<kN, kK, true, false, 0, Q5GemvStoreEpilogue, false, false, kDepth, kStageX,
                          kThreads><<<kGrid, kThreads, 0, stream>>>(x, codes, high_bits, scales,
                                                                     residual_out, nullptr);
}


// ---------------------------------------------------------------------------
// Pass 2, shape 2: Q4 SwiGLU gate/up pair (q4_linear_swiglu_gemv.cu row
// pairing: gate row r, up row r + kIntermediate).
//
//   out[r] = epilogue(Wg[r, :] . x, Wu[r, :] . x)      (silu(gate) * up)
//
// Same geometry as q5_gemv_gfx906_kernel (256 threads, 32 lanes per row pair,
// 2 pairs per wave, 8 pairs per block, grid kIntermediate / 8) but each
// half-wave streams BOTH rows of its pair: two Q4 chunk streams (16 B nibbles
// + fp16 scale each, no high plane) and two fp32 accumulators per lane, so the
// swiglu epilogue needs no cross-wave exchange. x is staged once per block as
// fp16 in LDS (10 KB at K=5120) with the first chunk loads of both streams
// issued before the barrier. Decode = Q4TileAtomGfx906::decode_quarter (bias
// trick) twice per chunk; dot = v_dot2_f32_f16; reduce = gfx906_reduce_sum32.

struct Q4GemvChunkGfx906 {
    using Storage = Q4RowSplitStorage;

    struct Raw {
        uint4 words;
        std::uint16_t scale_bits;
    };

    // Lane hl of a half-wave owns group step*16 + (hl >> 1), half (hl & 1):
    // values [group*64 + 32*half, +32).
    __device__ static __forceinline__ Raw load(const std::uint8_t* __restrict__ code_row,
                                               const std::uint8_t* __restrict__ scale_row,
                                               int step, int hl) {
        const int group = step * (Gfx906GemvGeometry::kLanesPerRow / 2) + (hl >> 1);
        const int half  = hl & 1;
        Raw raw;
        raw.words = *reinterpret_cast<const uint4*>(code_row + group * Storage::kCodeBytesPerGroup +
                                                    16 * half);
        raw.scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scale_row + group * Storage::kScaleBytesPerGroup);
        return raw;
    }

    // 32 values -> 16 fp16-pair dwords, dst[j] = (value 2j, value 2j+1) scaled.
    __device__ static __forceinline__ void decode(const Raw& raw, std::uint32_t (&dst)[16]) {
        Q4TileAtomGfx906::Raw q;
        q.scale_bits = raw.scale_bits;
        q.words      = make_uint2(raw.words.x, raw.words.y);
        Q4TileAtomGfx906::decode_quarter(q, *reinterpret_cast<std::uint32_t(*)[8]>(&dst[0]));
        q.words = make_uint2(raw.words.z, raw.words.w);
        Q4TileAtomGfx906::decode_quarter(q, *reinterpret_cast<std::uint32_t(*)[8]>(&dst[8]));
    }
};

// Epilogue contract: epilogue(out, row, gate_acc, up_acc) from half-lane 0.
template <int kIntermediate, int kK, class Epilogue, int kDepth = 2,
          bool kSwizzleX = kGfx906GemvXSwizzle>
__global__ void __launch_bounds__(Gfx906GemvGeometry::kThreads)
q4_swiglu_pair_gemv_gfx906_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, Epilogue epilogue) {
    using G     = Gfx906GemvGeometry;
    using Chunk = Q4GemvChunkGfx906;
    constexpr int kGroups   = kK / Chunk::Storage::kGroupK;
    constexpr int kSteps    = kK / G::kValuesPerStep;
    constexpr int kInFlight = kDepth < kSteps ? kDepth : kSteps;
    static_assert(kK % G::kValuesPerStep == 0, "K must be a multiple of 1024");
    static_assert(kIntermediate % G::kRowsPerBlock == 0, "N/2 must be a multiple of 8 rows");
    static_assert(kDepth >= 1, "need at least one chunk in flight");

    __shared__ __align__(16) std::uint32_t x_sh[kK / 2];

    const int tid    = static_cast<int>(threadIdx.x);
    const int wave   = tid >> 6;
    const int lane64 = tid & 63;
    const int half   = lane64 >> 5;
    const int hl     = lane64 & 31;
    const int row    = static_cast<int>(blockIdx.x) * G::kRowsPerBlock + wave * G::kRowsPerWave + half;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kCodeBytesPerGroup;
    const std::uint8_t* gate_scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kScaleBytesPerGroup;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(row + kIntermediate) * kGroups *
                    Chunk::Storage::kCodeBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(row + kIntermediate) * kGroups *
                     Chunk::Storage::kScaleBytesPerGroup;

    // Both streams' first chunks in flight before the x staging barrier.
    typename Chunk::Raw raw_g[kInFlight];
    typename Chunk::Raw raw_u[kInFlight];
#pragma unroll
    for (int s = 0; s < kInFlight; ++s) {
        raw_g[s] = Chunk::load(gate_code_row, gate_scale_row, s, hl);
        raw_u[s] = Chunk::load(up_code_row, up_scale_row, s, hl);
    }

    {
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        auto* x_s       = reinterpret_cast<uint4*>(x_sh);
        for (int i = tid; i < kK / 8; i += G::kThreads) {
            const uint4 v = x_g[i];
            uint4 h;
            h.x    = gfx906_bf16x2_to_half2(v.x);
            h.y    = gfx906_bf16x2_to_half2(v.y);
            h.z    = gfx906_bf16x2_to_half2(v.z);
            h.w    = gfx906_bf16x2_to_half2(v.w);
            x_s[kSwizzleX ? gfx906_swizzle_x_slot(i) : i] = h;
        }
    }
    __syncthreads();

    float acc_g = 0.0f;
    float acc_u = 0.0f;
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        const int slot                  = s % kInFlight;
        const typename Chunk::Raw cur_g = raw_g[slot];
        const typename Chunk::Raw cur_u = raw_u[slot];
        if (s + kInFlight < kSteps) {
            raw_g[slot] = Chunk::load(gate_code_row, gate_scale_row, s + kInFlight, hl);
            raw_u[slot] = Chunk::load(up_code_row, up_scale_row, s + kInFlight, hl);
        }
        std::uint32_t wg[16];
        std::uint32_t wu[16];
        Chunk::decode(cur_g, wg);
        Chunk::decode(cur_u, wu);

        const int k0      = (s * (G::kLanesPerRow / 2) + (hl >> 1)) * Chunk::Storage::kGroupK +
                       G::kValuesPerLane * (hl & 1);
        constexpr int kVStride = kSwizzleX ? G::kLanesPerRow : 1;
        const uint4* x_s4      = reinterpret_cast<const uint4*>(x_sh) +
                            (kSwizzleX ? s * (G::kValuesPerStep / 8) + hl : (k0 >> 3));
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 xv = x_s4[v * kVStride];
            acc_g          = gfx906_fdot2(wg[4 * v + 0], xv.x, acc_g);
            acc_u          = gfx906_fdot2(wu[4 * v + 0], xv.x, acc_u);
            acc_g          = gfx906_fdot2(wg[4 * v + 1], xv.y, acc_g);
            acc_u          = gfx906_fdot2(wu[4 * v + 1], xv.y, acc_u);
            acc_g          = gfx906_fdot2(wg[4 * v + 2], xv.z, acc_g);
            acc_u          = gfx906_fdot2(wu[4 * v + 2], xv.z, acc_u);
            acc_g          = gfx906_fdot2(wg[4 * v + 3], xv.w, acc_g);
            acc_u          = gfx906_fdot2(wu[4 * v + 3], xv.w, acc_u);
        }
    }

    acc_g = gfx906_reduce_sum32(acc_g);
    acc_u = gfx906_reduce_sum32(acc_u);
    if (hl == 0) { epilogue(out, row, acc_g, acc_u); }
}

// ---------------------------------------------------------------------------
// Pass 2e: small-T (2..5) variants of the two decode GEMV families above.
//
// The weight chunk streams are IDENTICAL to T=1 (every weight byte is read
// exactly once, kDepth 1, same 32-lanes-per-row geometry); what changes:
//  - x is T vectors staged in LDS as fp16, per K-slab: the slab is sized so
//    the block's LDS stays <= kLdsCapBytes (32 KB -> 2 blocks/CU at 512
//    threads = 16 wave64/CU, the pass-2c occupancy lesson). K=5120 at T<=3 is
//    one slab; K=17408 at T=5 is 6 slabs of 3 steps (2 barriers per slab).
//  - kT fp32 accumulators per lane (2 kT for the swiglu pair), kT dot2
//    sequences per decoded chunk, the DPP reduction runs kT times.
//  - Layout is the one the stage-3 simt / stage-8 tiled routes use: x is
//    token-major [T][x_ld] bf16, out is [T][out_ld] bf16, the residual is
//    read from the same (t, row) slot.

template <int kK, int kT, int kLdsCapBytes>
struct Gfx906SmallTSlab {
    static constexpr int kSteps         = kK / Gfx906GemvGeometry::kValuesPerStep;
    static constexpr int kBytesPerStepT = kT * Gfx906GemvGeometry::kValuesPerStep * 2; // fp16
    static constexpr int kCapSteps      = kLdsCapBytes / kBytesPerStepT;
    static constexpr int kSlabSteps =
        kCapSteps < 1 ? 1 : (kCapSteps < kSteps ? kCapSteps : kSteps);
    static constexpr int kSlabs     = (kSteps + kSlabSteps - 1) / kSlabSteps;
    static constexpr int kSlabK     = kSlabSteps * Gfx906GemvGeometry::kValuesPerStep;
    static constexpr int kLdsDwords = kT * kSlabK / 2;
};

// Stage kT bf16 x rows [k_base, k_base + kSlabK) into LDS as fp16 pairs.
// Layout is [t][step][v][lane] uint4 (128 uint4 per 1024-value step): lane hl
// of a half-wave owns values [32 hl, 32 hl + 32) of every step = 4 uint4, and
// storing its v-th uint4 at step * 128 + v * 32 + hl makes each ds_read_b128
// wave-wide contiguous (conflict-free) instead of 64 B-strided (banks 0/16
// only, 16-way conflict). Caller owns the barriers.
template <int kT, int kSlabK, int kThreads>
__device__ __forceinline__ void gfx906_stage_x_slab(std::uint32_t* __restrict__ x_sh,
                                                    const __nv_bfloat16* __restrict__ x, int x_ld,
                                                    int k_base, int tid) {
    constexpr int kVecsPerT = kSlabK / 8;
    auto* x_s               = reinterpret_cast<uint4*>(x_sh);
#pragma unroll 4
    for (int i = tid; i < kT * kVecsPerT; i += kThreads) {
        const int t    = i / kVecsPerT;
        const int pos  = i - t * kVecsPerT; // swizzled slot within the token slab
        const int step = pos >> 7;
        const int v    = (pos >> 5) & 3;
        const int hl   = pos & 31;
        const int nat  = step * 128 + hl * 4 + v; // natural uint4 index
        const uint4 b  = *reinterpret_cast<const uint4*>(
            x + static_cast<std::int64_t>(t) * x_ld + k_base + nat * 8);
        uint4 h;
        h.x    = gfx906_bf16x2_to_half2(b.x);
        h.y    = gfx906_bf16x2_to_half2(b.y);
        h.z    = gfx906_bf16x2_to_half2(b.z);
        h.w    = gfx906_bf16x2_to_half2(b.w);
        x_s[i] = h;
    }
}

template <int kN, int kK, int kT, bool kResidual, int kThreads = 512, int kLdsCapBytes = 32768>
__global__ void __launch_bounds__(kThreads)
q5_gemv_smallt_gfx906_kernel(const __nv_bfloat16* __restrict__ x, int x_ld,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ high_bits,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, int out_ld) {
    using G     = Gfx906GemvGeometry;
    using Chunk = Q5GemvChunkGfx906;
    using Slab  = Gfx906SmallTSlab<kK, kT, kLdsCapBytes>;
    constexpr int kGroups       = kK / Chunk::Storage::kGroupK;
    constexpr int kSteps        = Slab::kSteps;
    constexpr int kRowsPerBlock = kThreads / G::kLanesPerRow;
    static_assert(kT >= 2 && kT <= 8, "small-T kernel covers T=2..8");
    static_assert(kThreads % 64 == 0 && kThreads >= 64 && kThreads <= 1024, "1..16 wave64 per block");
    static_assert(kK % G::kValuesPerStep == 0, "K must be a multiple of 1024");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of the rows per block");

    __shared__ __align__(16) std::uint32_t x_sh[Slab::kLdsDwords];

    const int tid    = static_cast<int>(threadIdx.x);
    const int wave   = tid >> 6;
    const int lane64 = tid & 63;
    const int half   = lane64 >> 5;
    const int hl     = lane64 & 31;
    const int row    = static_cast<int>(blockIdx.x) * kRowsPerBlock + wave * G::kRowsPerWave + half;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kCodeBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kHighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kScaleBytesPerGroup;

    // First chunk in flight before the first x staging barrier (kDepth 1).
    typename Chunk::Raw raw = Chunk::load(code_row, high_row, scale_row, 0, hl);

    float acc[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { acc[t] = 0.0f; }

#pragma unroll
    for (int slab = 0; slab < Slab::kSlabs; ++slab) {
        if (slab > 0) { __syncthreads(); } // previous slab's LDS reads retired
        gfx906_stage_x_slab<kT, Slab::kSlabK, kThreads>(x_sh, x, x_ld, slab * Slab::kSlabK, tid);
        __syncthreads();
#pragma unroll
        for (int ss = 0; ss < Slab::kSlabSteps; ++ss) {
            const int s = slab * Slab::kSlabSteps + ss;
            if (s < kSteps) {
                const typename Chunk::Raw cur = raw;
                if (s + 1 < kSteps) { raw = Chunk::load(code_row, high_row, scale_row, s + 1, hl); }
                std::uint32_t w[16];
                Chunk::decode(cur, w);
                // fp16 x pairs for values [32 hl, 32 hl + 32) of slab-step ss: swizzled
                // slots ss * 128 + v * 32 + hl (see gfx906_stage_x_slab).
#pragma unroll
                for (int t = 0; t < kT; ++t) {
                    const uint4* x_s4 = reinterpret_cast<const uint4*>(x_sh) +
                                        t * (Slab::kSlabK / 8) + ss * 128 + hl;
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const uint4 xv = x_s4[v * 32];
                        acc[t]         = gfx906_fdot2(w[4 * v + 0], xv.x, acc[t]);
                        acc[t]         = gfx906_fdot2(w[4 * v + 1], xv.y, acc[t]);
                        acc[t]         = gfx906_fdot2(w[4 * v + 2], xv.z, acc[t]);
                        acc[t]         = gfx906_fdot2(w[4 * v + 3], xv.w, acc[t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        const float r = gfx906_reduce_sum32(acc[t]);
        if (hl == 0) {
            __nv_bfloat16* o = out + static_cast<std::int64_t>(t) * out_ld + row;
            float v          = r;
            if constexpr (kResidual) { v = __bfloat162float(*o) + v; }
            *o = __float2bfloat16_rn(v);
        }
    }
}

template <int kN, int kK, int kT, bool kResidual, int kThreads = 512, int kLdsCapBytes = 32768>
inline void q5_gemv_smallt_gfx906_launch(const __nv_bfloat16* x, int x_ld, const std::uint8_t* codes,
                                         const std::uint8_t* high_bits, const std::uint8_t* scales,
                                         __nv_bfloat16* out, int out_ld, cudaStream_t stream) {
    constexpr int kGrid = kN / (kThreads / Gfx906GemvGeometry::kLanesPerRow);
    q5_gemv_smallt_gfx906_kernel<kN, kK, kT, kResidual, kThreads, kLdsCapBytes>
        <<<kGrid, kThreads, 0, stream>>>(x, x_ld, codes, high_bits, scales, out, out_ld);
}

// Runtime T -> instantiation (T=2..5). Returns false when T is outside the
// small-T domain so the caller keeps its existing route.
template <int kN, int kK, bool kResidual, int kThreads = 512, int kLdsCapBytes = 32768>
inline bool q5_gemv_smallt_gfx906_dispatch(int t, const __nv_bfloat16* x, int x_ld,
                                           const std::uint8_t* codes, const std::uint8_t* high_bits,
                                           const std::uint8_t* scales, __nv_bfloat16* out,
                                           int out_ld, cudaStream_t stream) {
    switch (t) {
    case 2:
        q5_gemv_smallt_gfx906_launch<kN, kK, 2, kResidual, kThreads, kLdsCapBytes>(
            x, x_ld, codes, high_bits, scales, out, out_ld, stream);
        return true;
    case 3:
        q5_gemv_smallt_gfx906_launch<kN, kK, 3, kResidual, kThreads, kLdsCapBytes>(
            x, x_ld, codes, high_bits, scales, out, out_ld, stream);
        return true;
    case 4:
        q5_gemv_smallt_gfx906_launch<kN, kK, 4, kResidual, kThreads, kLdsCapBytes>(
            x, x_ld, codes, high_bits, scales, out, out_ld, stream);
        return true;
    case 5:
        q5_gemv_smallt_gfx906_launch<kN, kK, 5, kResidual, kThreads, kLdsCapBytes>(
            x, x_ld, codes, high_bits, scales, out, out_ld, stream);
        return true;
    default:
        return false;
    }
}

// Q4 swiglu gate/up pair, small T. Epilogue contract: epilogue(out_t, row,
// gate, up) from half-lane 0 with out_t = out + t * out_ld.
template <int kIntermediate, int kK, int kT, class Epilogue, int kThreads = 512,
          int kLdsCapBytes = 32768>
__global__ void __launch_bounds__(kThreads)
q4_swiglu_pair_gemv_smallt_gfx906_kernel(const __nv_bfloat16* __restrict__ x, int x_ld,
                                         const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ scales,
                                         __nv_bfloat16* __restrict__ out, int out_ld,
                                         Epilogue epilogue) {
    using G     = Gfx906GemvGeometry;
    using Chunk = Q4GemvChunkGfx906;
    using Slab  = Gfx906SmallTSlab<kK, kT, kLdsCapBytes>;
    constexpr int kGroups       = kK / Chunk::Storage::kGroupK;
    constexpr int kSteps        = Slab::kSteps;
    constexpr int kRowsPerBlock = kThreads / G::kLanesPerRow;
    static_assert(kT >= 2 && kT <= 8, "small-T kernel covers T=2..8");
    static_assert(kK % G::kValuesPerStep == 0, "K must be a multiple of 1024");
    static_assert(kIntermediate % kRowsPerBlock == 0, "N/2 must be a multiple of the rows per block");

    __shared__ __align__(16) std::uint32_t x_sh[Slab::kLdsDwords];

    const int tid    = static_cast<int>(threadIdx.x);
    const int wave   = tid >> 6;
    const int lane64 = tid & 63;
    const int half   = lane64 >> 5;
    const int hl     = lane64 & 31;
    const int row    = static_cast<int>(blockIdx.x) * kRowsPerBlock + wave * G::kRowsPerWave + half;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kCodeBytesPerGroup;
    const std::uint8_t* gate_scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * Chunk::Storage::kScaleBytesPerGroup;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(row + kIntermediate) * kGroups *
                    Chunk::Storage::kCodeBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(row + kIntermediate) * kGroups *
                     Chunk::Storage::kScaleBytesPerGroup;

    typename Chunk::Raw raw_g = Chunk::load(gate_code_row, gate_scale_row, 0, hl);
    typename Chunk::Raw raw_u = Chunk::load(up_code_row, up_scale_row, 0, hl);

    float acc_g[kT];
    float acc_u[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        acc_g[t] = 0.0f;
        acc_u[t] = 0.0f;
    }

#pragma unroll
    for (int slab = 0; slab < Slab::kSlabs; ++slab) {
        if (slab > 0) { __syncthreads(); }
        gfx906_stage_x_slab<kT, Slab::kSlabK, kThreads>(x_sh, x, x_ld, slab * Slab::kSlabK, tid);
        __syncthreads();
#pragma unroll
        for (int ss = 0; ss < Slab::kSlabSteps; ++ss) {
            const int s = slab * Slab::kSlabSteps + ss;
            if (s < kSteps) {
                const typename Chunk::Raw cur_g = raw_g;
                const typename Chunk::Raw cur_u = raw_u;
                if (s + 1 < kSteps) {
                    raw_g = Chunk::load(gate_code_row, gate_scale_row, s + 1, hl);
                    raw_u = Chunk::load(up_code_row, up_scale_row, s + 1, hl);
                }
                std::uint32_t wg[16];
                std::uint32_t wu[16];
                Chunk::decode(cur_g, wg);
                Chunk::decode(cur_u, wu);
#pragma unroll
                for (int t = 0; t < kT; ++t) {
                    const uint4* x_s4 = reinterpret_cast<const uint4*>(x_sh) +
                                        t * (Slab::kSlabK / 8) + ss * 128 + hl;
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const uint4 xv = x_s4[v * 32];
                        acc_g[t]       = gfx906_fdot2(wg[4 * v + 0], xv.x, acc_g[t]);
                        acc_u[t]       = gfx906_fdot2(wu[4 * v + 0], xv.x, acc_u[t]);
                        acc_g[t]       = gfx906_fdot2(wg[4 * v + 1], xv.y, acc_g[t]);
                        acc_u[t]       = gfx906_fdot2(wu[4 * v + 1], xv.y, acc_u[t]);
                        acc_g[t]       = gfx906_fdot2(wg[4 * v + 2], xv.z, acc_g[t]);
                        acc_u[t]       = gfx906_fdot2(wu[4 * v + 2], xv.z, acc_u[t]);
                        acc_g[t]       = gfx906_fdot2(wg[4 * v + 3], xv.w, acc_g[t]);
                        acc_u[t]       = gfx906_fdot2(wu[4 * v + 3], xv.w, acc_u[t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        const float g = gfx906_reduce_sum32(acc_g[t]);
        const float u = gfx906_reduce_sum32(acc_u[t]);
        if (hl == 0) { epilogue(out + static_cast<std::int64_t>(t) * out_ld, row, g, u); }
    }
}

template <int kIntermediate, int kK, int kT, class Epilogue, int kThreads = 512,
          int kLdsCapBytes = 32768>
inline void q4_swiglu_pair_gemv_smallt_gfx906_launch(const __nv_bfloat16* x, int x_ld,
                                                     const std::uint8_t* codes,
                                                     const std::uint8_t* scales,
                                                     __nv_bfloat16* out, int out_ld,
                                                     Epilogue epilogue, cudaStream_t stream) {
    constexpr int kGrid = kIntermediate / (kThreads / Gfx906GemvGeometry::kLanesPerRow);
    q4_swiglu_pair_gemv_smallt_gfx906_kernel<kIntermediate, kK, kT, Epilogue, kThreads, kLdsCapBytes>
        <<<kGrid, kThreads, 0, stream>>>(x, x_ld, codes, scales, out, out_ld, epilogue);
}

template <int kIntermediate, int kK, class Epilogue, int kThreads = 512, int kLdsCapBytes = 32768>
inline bool q4_swiglu_pair_gemv_smallt_gfx906_dispatch(int t, const __nv_bfloat16* x, int x_ld,
                                                       const std::uint8_t* codes,
                                                       const std::uint8_t* scales,
                                                       __nv_bfloat16* out, int out_ld,
                                                       Epilogue epilogue, cudaStream_t stream) {
    switch (t) {
    case 2:
        q4_swiglu_pair_gemv_smallt_gfx906_launch<kIntermediate, kK, 2, Epilogue, kThreads,
                                                 kLdsCapBytes>(x, x_ld, codes, scales, out, out_ld,
                                                               epilogue, stream);
        return true;
    case 3:
        q4_swiglu_pair_gemv_smallt_gfx906_launch<kIntermediate, kK, 3, Epilogue, kThreads,
                                                 kLdsCapBytes>(x, x_ld, codes, scales, out, out_ld,
                                                               epilogue, stream);
        return true;
    case 4:
        q4_swiglu_pair_gemv_smallt_gfx906_launch<kIntermediate, kK, 4, Epilogue, kThreads,
                                                 kLdsCapBytes>(x, x_ld, codes, scales, out, out_ld,
                                                               epilogue, stream);
        return true;
    case 5:
        q4_swiglu_pair_gemv_smallt_gfx906_launch<kIntermediate, kK, 5, Epilogue, kThreads,
                                                 kLdsCapBytes>(x, x_ld, codes, scales, out, out_ld,
                                                               epilogue, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninfer::ops::detail
