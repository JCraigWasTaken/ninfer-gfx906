#pragma once

// gfx906 wave64 LDS-tiled GEMM for RowSplit grouped-int weights (stage 8).
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One 256-thread CTA (4 wave64) owns a 64-row x ColsPerTile output tile.
// Each 64-value K stage is cooperatively dequantized to FP16 in LDS
// (weights) and converted BF16 -> FP16 in LDS (activations); the compute
// loop contracts them with v_dot2_f32_f16 (packed FP16 dot with FP32
// accumulate -- exact products, FP32 sums). Weights leave global memory
// exactly once per (row-tile, col-tile) pass, i.e. ceil(T / ColsPerTile)
// times per token batch, vs ceil(T / 8) for the stage-3 SIMT fallback.
// That weight-re-read factor is what made prefill and MTP-verify GEMMs the
// dominant cost on the MI50 (STAGE5-7-LOG.md).
//
// Conventions inherited from the validated stage-4/7 gfx906 kernels:
//  - FP16 staging and packed-FP16 arithmetic (gfx906 has no bf16 VALU);
//    activations are clamped to +-65504 before the BF16->FP16 convert.
//  - Static LDS only, well under the 64 KiB/CU limit (17.4 KiB at C64).
//  - LDS rows padded to 34 dwords: rows stay 8-byte aligned for
//    ds_read_b64 and per-instruction bank pairs never collide.
//  - No ldmatrix/mma/cp.async; plain global loads (the compat cp.async shim
//    is synchronous on gfx906 anyway) with multi-CTA/CU latency hiding.
//
// K handling: K must be even. A K that is not a multiple of 64 is walked in
// whole 64-value stages against the padded code/scale planes; weight LDS
// lanes past K are zeroed after decode (padding codes are never trusted)
// and activation LDS lanes past K are zero-filled.

#include "core/tensor.h"
#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// gfx906 packed-FP16 primitives
// ---------------------------------------------------------------------------

// v_dot2_f32_f16: acc += a.lo*b.lo + a.hi*b.hi with FP32 accumulation (the
// llama.cpp-gfx906 idiom). Operands are FP16 pairs carried as dwords.
__device__ __forceinline__ float gfx906_fdot2(std::uint32_t a, std::uint32_t b, float acc) {
#if defined(NINFER_GFX906_COMPAT) && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
#else
    const __half2 av = *reinterpret_cast<const __half2*>(&a);
    const __half2 bv = *reinterpret_cast<const __half2*>(&b);
    const float2 af  = __half22float2(av);
    const float2 bf  = __half22float2(bv);
    return acc + af.x * bf.x + af.y * bf.y;
#endif
}

__device__ __forceinline__ std::uint32_t gfx906_half2_bits(__half2 value) {
    return *reinterpret_cast<const std::uint32_t*>(&value);
}

__device__ __forceinline__ __half2 gfx906_bits_half2(std::uint32_t bits) {
    return *reinterpret_cast<const __half2*>(&bits);
}

// BF16 pair (as dword) -> FP16 pair (as dword), clamped to the FP16 range.
__device__ __forceinline__ std::uint32_t gfx906_bf16x2_to_half2(std::uint32_t bits) {
    constexpr float kF16Max = 65504.0f;
    float lo               = __uint_as_float(bits << 16);
    float hi               = __uint_as_float(bits & 0xffff0000u);
    lo                     = fminf(fmaxf(lo, -kF16Max), kF16Max);
    hi                     = fminf(fmaxf(hi, -kF16Max), kF16Max);
    return gfx906_half2_bits(__floats2half2_rn(lo, hi));
}

// ---------------------------------------------------------------------------
// Tile geometry
// ---------------------------------------------------------------------------

struct Gfx906TiledGemmGeometry {
    static constexpr int kRowsPerCta = 64;
    static constexpr int kThreads    = 256;
    static constexpr int kStageK     = 64;              // K values per stage
    static constexpr int kStageK2    = kStageK / 2;     // FP16 pairs per stage row
    static constexpr int kLdsStride  = kStageK2 + 2;    // dwords; even => b64-aligned rows
};

// ---------------------------------------------------------------------------
// Weight tile atoms: cooperative one-stage dequant into LDS.
//
// Thread assignment (all atoms): row = tid >> 2 (64 rows), quarter q = tid & 3
// covers the row's K sub-range [16q, 16q+16) of the stage, written as 8
// FP16-pair dwords at s_w[row * kLdsStride + 8q .. +8].
// ---------------------------------------------------------------------------

// Each atom stages one thread-quarter of a 64-value K stage: the 16 values
// at K [stage_k0 + 16q, stage_k0 + 16q + 16) of one weight row, decoded and
// scaled into 8 FP16-pair dwords. The load (global -> registers) and decode
// (registers -> LDS) phases are split so the kernel can prefetch the next
// stage's raw bytes over the current stage's compute loop.

struct Q4TileAtomGfx906 {
    using Storage = Q4RowSplitStorage;
    static constexpr bool kHasHigh = false;

    struct Raw {
        uint2 words;
        std::uint16_t scale_bits;
    };

    // Q4G64: byte j of a 32-byte group = values (2j, 2j+1) as low/high nibble.
    __device__ static __forceinline__ Raw load_quarter(const std::uint8_t* code_row,
                                                       const std::uint8_t* /*high_row*/,
                                                       const std::uint8_t* scale_row,
                                                       int stage_k0, int q) {
        const std::int64_t group = stage_k0 / Storage::kGroupK;
        Raw raw;
        raw.words = *reinterpret_cast<const uint2*>(code_row +
                                                    group * Storage::kCodeBytesPerGroup + 8 * q);
        raw.scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scale_row + group * Storage::kScaleBytesPerGroup);
        return raw;
    }

    __device__ static __forceinline__ void decode_quarter(const Raw& raw,
                                                          std::uint32_t (&dst)[8]) {
        const __half2 scale2 = __half2half2(__ushort_as_half(raw.scale_bits));
        const __half2 bias   = __half2half2(__ushort_as_half(0x6408)); // 1032.0 = 1024 + 8
        // ^ 0x88888888: 4-bit two's complement -> biased so that
        // (nibble ^ 8) + 1024 - 1032 = sign-extended value.
        const std::uint32_t w[2] = {raw.words.x ^ 0x88888888u, raw.words.y ^ 0x88888888u};
#pragma unroll
        for (int d = 0; d < 2; ++d) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const std::uint32_t t = w[d] >> (8 * p);
                std::uint32_t bits    = (t & 0x0fu) | ((t & 0xf0u) << 12);
                bits |= 0x64006400u;
                dst[4 * d + p] = gfx906_half2_bits(
                    __hmul2(__hsub2(gfx906_bits_half2(bits), bias), scale2));
            }
        }
    }
};

struct Q5TileAtomGfx906 {
    using Storage = Q5RowSplitStorage;
    static constexpr bool kHasHigh = true;

    struct Raw {
        uint2 words;
        std::uint32_t plane16;
        std::uint16_t scale_bits;
    };

    // Q5G64: Q4 layout plus an 8-byte high plane per group; code byte j draws
    // its two high bits from plane byte j>>2, bit pair (j&3)*2.
    __device__ static __forceinline__ Raw load_quarter(const std::uint8_t* code_row,
                                                       const std::uint8_t* high_row,
                                                       const std::uint8_t* scale_row,
                                                       int stage_k0, int q) {
        const std::int64_t group = stage_k0 / Storage::kGroupK;
        Raw raw;
        raw.words = *reinterpret_cast<const uint2*>(code_row +
                                                    group * Storage::kCodeBytesPerGroup + 8 * q);
        // Plane bytes for code bytes [8q, 8q+8): bytes 2q and 2q+1 of the plane.
        raw.plane16 = *reinterpret_cast<const std::uint16_t*>(
            high_row + group * Storage::kHighBytesPerGroup + 2 * q);
        raw.scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scale_row + group * Storage::kScaleBytesPerGroup);
        return raw;
    }

    __device__ static __forceinline__ void decode_quarter(const Raw& raw,
                                                          std::uint32_t (&dst)[8]) {
        const __half2 scale2 = __half2half2(__ushort_as_half(raw.scale_bits));
        const __half2 bias   = __half2half2(__ushort_as_half(0x6410)); // 1040.0 = 1024 + 16
        // ^ 0xffff: 5-bit two's complement flips the high bit, so that
        // (nibble + 16 * (hi ^ 1)) + 1024 - 1040 = sign-extended value.
        const std::uint32_t plane16 = raw.plane16 ^ 0xffffu;
        const std::uint32_t w[2]    = {raw.words.x, raw.words.y};
#pragma unroll
        for (int d = 0; d < 2; ++d) {
            const std::uint32_t plane = plane16 >> (8 * d);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const std::uint32_t t = w[d] >> (8 * p);
                std::uint32_t bits    = (t & 0x0fu) | ((t & 0xf0u) << 12);
                bits |= ((plane >> (2 * p)) & 1u) << 4;
                bits |= ((plane >> (2 * p + 1)) & 1u) << 20;
                bits |= 0x64006400u;
                dst[4 * d + p] = gfx906_half2_bits(
                    __hmul2(__hsub2(gfx906_bits_half2(bits), bias), scale2));
            }
        }
    }
};

struct W8TileAtomGfx906 {
    using Storage = W8RowSplitStorage;
    static constexpr bool kHasHigh = false;

    struct Raw {
        uint4 words;
        std::uint16_t scale_bits;
    };

    // W8G32: one int8 code byte per value, 32-byte groups; a 16-value quarter
    // is the half-group at byte offset 16 * (q & 1) of group k0/32 + (q >> 1).
    __device__ static __forceinline__ Raw load_quarter(const std::uint8_t* code_row,
                                                       const std::uint8_t* /*high_row*/,
                                                       const std::uint8_t* scale_row,
                                                       int stage_k0, int q) {
        const std::int64_t group = stage_k0 / Storage::kGroupK + (q >> 1);
        Raw raw;
        raw.words = *reinterpret_cast<const uint4*>(code_row +
                                                    group * Storage::kCodeBytesPerGroup +
                                                    16 * (q & 1));
        raw.scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scale_row + group * Storage::kScaleBytesPerGroup);
        return raw;
    }

    __device__ static __forceinline__ void decode_quarter(const Raw& raw,
                                                          std::uint32_t (&dst)[8]) {
        const __half2 scale2 = __half2half2(__ushort_as_half(raw.scale_bits));
        const std::uint32_t w[4] = {raw.words.x, raw.words.y, raw.words.z, raw.words.w};
#pragma unroll
        for (int d = 0; d < 4; ++d) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int b0 = static_cast<int>(static_cast<std::int8_t>(w[d] >> (16 * p)));
                const int b1 = static_cast<int>(static_cast<std::int8_t>(w[d] >> (16 * p + 8)));
                const __half2 decoded =
                    __floats2half2_rn(static_cast<float>(b0), static_cast<float>(b1));
                dst[2 * d + p] = gfx906_half2_bits(__hmul2(decoded, scale2));
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Epilogues (scalar per output element)
// ---------------------------------------------------------------------------

struct Gfx906TiledStoreEpilogue {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void operator()(std::int32_t row, std::int32_t col,
                                               float value) const {
        out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16(value);
    }
};

// residual_out[row, col] += gemm(row, col) -- the fused LinearAdd contract.
struct Gfx906TiledResidualAddEpilogue {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void operator()(std::int32_t row, std::int32_t col,
                                               float value) const {
        const std::int64_t index = static_cast<std::int64_t>(col) * out_ld + row;
        out[index] = __float2bfloat16(__bfloat162float(out[index]) + value);
    }
};

// Split-row store: rows [0, split) -> out_a, rows [split, ...) -> out_b.
// Covers the fused pair/proj contracts that carve one weight into planes.
struct Gfx906TiledSplitStoreEpilogue {
    __nv_bfloat16* out_a;
    __nv_bfloat16* out_b;
    std::int32_t ld_a;
    std::int32_t ld_b;
    std::int32_t split;

    __device__ __forceinline__ void operator()(std::int32_t row, std::int32_t col,
                                               float value) const {
        if (row < split) {
            out_a[static_cast<std::int64_t>(col) * ld_a + row] = __float2bfloat16(value);
        } else {
            out_b[static_cast<std::int64_t>(col) * ld_b + (row - split)] =
                __float2bfloat16(value);
        }
    }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <class Atom, int ColsPerTile, class Epilogue>
__global__ __launch_bounds__(Gfx906TiledGemmGeometry::kThreads, 2) void
rowsplit_tiled_gemm_gfx906_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ high,
                                  const std::uint8_t* __restrict__ scales, std::int32_t rows,
                                  std::int32_t k, std::int32_t cols, std::int32_t padded_k,
                                  Epilogue epilogue) {
    using Geo     = Gfx906TiledGemmGeometry;
    using Storage = typename Atom::Storage;

    static_assert(ColsPerTile == 16 || ColsPerTile == 32 || ColsPerTile == 64);
    constexpr int kColGroups     = 16; // (tid >> 4) selects the column group
    constexpr int kColsPerThread = ColsPerTile / kColGroups;
    static_assert(kColsPerThread >= 1);
    constexpr int kRowsPerThread = 4;
    // X staging: threads per column and dwords (FP16 pairs) per thread.
    constexpr int kXThreadsPerCol = Geo::kThreads / ColsPerTile;
    constexpr int kXDwordsPerThread = Geo::kStageK2 / kXThreadsPerCol;
    static_assert(kXDwordsPerThread * kXThreadsPerCol == Geo::kStageK2);
    static_assert(kXDwordsPerThread % 2 == 0, "X staging loads 8-byte packs");
    // Weight staging: groups per stage (Q4/Q5: 1 group of 64; W8: 2 of 32).
    constexpr int kGroupsPerStage = Geo::kStageK / Storage::kGroupK;
    static_assert(kGroupsPerStage * Storage::kGroupK == Geo::kStageK);

    __shared__ std::uint32_t s_w[Geo::kRowsPerCta * Geo::kLdsStride];
    __shared__ std::uint32_t s_x[ColsPerTile * Geo::kLdsStride];

    const int tid  = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * Geo::kRowsPerCta;
    const int col0 = static_cast<int>(blockIdx.y) * ColsPerTile;

    // Staging assignments.
    const int wq        = tid & 3;        // weight quarter: K sub-range [16*wq, 16*wq+16)
    const int w_row     = tid >> 2;       // 0..63
    const int x_col     = tid / kXThreadsPerCol;
    const int x_sub     = tid % kXThreadsPerCol;
    const bool w_active = (row0 + w_row) < rows;
    const bool x_active = (col0 + x_col) < cols;

    const int padded_groups = padded_k / Storage::kGroupK;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row0 + w_row) * padded_groups *
                    Storage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row0 + w_row) * padded_groups *
                     Storage::kScaleBytesPerGroup;
    const std::uint8_t* high_row = nullptr;
    if constexpr (Atom::kHasHigh) {
        high_row = high + static_cast<std::int64_t>(row0 + w_row) * padded_groups *
                              Storage::kHighBytesPerGroup;
    }
    const __nv_bfloat16* x_col_base =
        x + static_cast<std::int64_t>(col0 + x_col) * k; // guarded by x_active

    // Compute assignments: rows (tid & 15) + 16j, cols (tid >> 4) + 16i.
    const int cr = tid & 15;
    const int cc = tid >> 4;

    float acc[kRowsPerThread][kColsPerThread];
#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
#pragma unroll
        for (int i = 0; i < kColsPerThread; ++i) { acc[j][i] = 0.0f; }
    }

    const int stages = (k + Geo::kStageK - 1) / Geo::kStageK;

    // Raw next-stage bytes held in registers: the load for stage s+1 is
    // issued before stage s's compute loop, so global latency overlaps the
    // FP16 dot-product work (the compat cp.async shim is synchronous, so
    // this register prefetch is the only intra-wave overlap available).
    typename Atom::Raw w_raw = {};
    std::uint32_t x_raw[kXDwordsPerThread];
#pragma unroll
    for (int d = 0; d < kXDwordsPerThread; ++d) { x_raw[d] = 0u; }

    const auto load_stage = [&](int stage) {
        const int k0 = stage * Geo::kStageK;
        const int kq = k0 + 16 * wq;
        if (w_active && kq < k) {
            w_raw = Atom::load_quarter(code_row, high_row, scale_row, k0, wq);
        }
        const int xk0 = k0 + x_sub * 2 * kXDwordsPerThread;
        if (x_active && (xk0 + 2 * kXDwordsPerThread) <= k) {
#pragma unroll
            for (int d = 0; d < kXDwordsPerThread; d += 2) {
                const uint2 pair = *reinterpret_cast<const uint2*>(x_col_base + xk0 + 2 * d);
                x_raw[d]         = pair.x;
                x_raw[d + 1]     = pair.y;
            }
        } else if (x_active) {
            // K tail: element-wise guarded loads (BF16 zero bits are 0).
            const std::uint16_t* xb = reinterpret_cast<const std::uint16_t*>(x_col_base);
#pragma unroll
            for (int d = 0; d < kXDwordsPerThread; ++d) {
                const int ka          = xk0 + 2 * d;
                const int kb          = ka + 1;
                const std::uint32_t a = ka < k ? xb[ka] : 0u;
                const std::uint32_t b = kb < k ? xb[kb] : 0u;
                x_raw[d]              = a | (b << 16);
            }
        }
    };

    load_stage(0);

    for (int stage = 0; stage < stages; ++stage) {
        const int k0 = stage * Geo::kStageK;

        __syncthreads(); // the previous stage's compute is done; LDS is free

        // -- Stage weights: this thread covers K [k0 + 16*wq, k0 + 16*wq + 16).
        {
            std::uint32_t decoded[8];
            const int kq = k0 + 16 * wq;
            if (w_active && kq < k) {
                Atom::decode_quarter(w_raw, decoded);
                if (kq + 16 > k) {
                    // Zero pairs at or past K (padding planes are untrusted).
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        if (kq + 2 * p >= k) { decoded[p] = 0u; }
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 8; ++p) { decoded[p] = 0u; }
            }
            std::uint32_t* dst = &s_w[w_row * Geo::kLdsStride + 8 * wq];
#pragma unroll
            for (int p = 0; p < 8; ++p) { dst[p] = decoded[p]; }
        }

        // -- Stage activations: this thread covers K
        //    [k0 + x_sub * 2*kXDwordsPerThread, ... + 2*kXDwordsPerThread).
        {
            std::uint32_t* dst = &s_x[x_col * Geo::kLdsStride + x_sub * kXDwordsPerThread];
#pragma unroll
            for (int d = 0; d < kXDwordsPerThread; ++d) {
                dst[d] = gfx906_bf16x2_to_half2(x_raw[d]);
            }
        }

        __syncthreads();

        // Issue the next stage's global loads before the compute loop.
        if (stage + 1 < stages) { load_stage(stage + 1); }

        // -- Contract the stage: 32 FP16 pairs per row/col, b64 LDS reads.
#pragma unroll 4
        for (int k2 = 0; k2 < Geo::kStageK2; k2 += 2) {
            uint2 wv[kRowsPerThread];
            uint2 xv[kColsPerThread];
#pragma unroll
            for (int j = 0; j < kRowsPerThread; ++j) {
                wv[j] = *reinterpret_cast<const uint2*>(&s_w[(cr + 16 * j) * Geo::kLdsStride + k2]);
            }
#pragma unroll
            for (int i = 0; i < kColsPerThread; ++i) {
                xv[i] = *reinterpret_cast<const uint2*>(&s_x[(cc + 16 * i) * Geo::kLdsStride + k2]);
            }
#pragma unroll
            for (int j = 0; j < kRowsPerThread; ++j) {
#pragma unroll
                for (int i = 0; i < kColsPerThread; ++i) {
                    acc[j][i] = gfx906_fdot2(wv[j].x, xv[i].x, acc[j][i]);
                    acc[j][i] = gfx906_fdot2(wv[j].y, xv[i].y, acc[j][i]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
        const int row = row0 + cr + 16 * j;
        if (row >= rows) { continue; }
#pragma unroll
        for (int i = 0; i < kColsPerThread; ++i) {
            const int col = col0 + cc + 16 * i;
            if (col < cols) { epilogue(row, col, acc[j][i]); }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared launch helper
// ---------------------------------------------------------------------------

template <class Atom, int ColsPerTile, class Epilogue>
inline void launch_rowsplit_tiled_gemm_gfx906(const Tensor& x, const Weight& w,
                                              std::int32_t rows, Epilogue epilogue,
                                              cudaStream_t stream) {
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(
        static_cast<unsigned>((rows + Gfx906TiledGemmGeometry::kRowsPerCta - 1) /
                              Gfx906TiledGemmGeometry::kRowsPerCta),
        static_cast<unsigned>((cols + ColsPerTile - 1) / ColsPerTile), 1u);

    rowsplit_tiled_gemm_gfx906_kernel<Atom, ColsPerTile, Epilogue>
        <<<grid, Gfx906TiledGemmGeometry::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint8_t*>(w.scales), rows, k, cols, padded_k, epilogue);
}

} // namespace ninfer::ops::detail
