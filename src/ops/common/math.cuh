#pragma once

#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplus(float x) { return (x > 20.0f) ? x : log1pf(expf(x)); }

#if defined(NINFER_GFX906_COMPAT)

// gfx906: v_exp_f32 via the amdgcn builtin (same fast-approximate contract as
// PTX ex2.approx).
__device__ __forceinline__ float exp2_approx(float x) { return __builtin_amdgcn_exp2f(x); }

// gfx906 has no bf16 VALU; round-to-nearest-even convert through the HIP bf16
// helpers and pack the two 16-bit patterns manually.
__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    const __hip_bfloat16_raw lo_raw = static_cast<__hip_bfloat16_raw>(__float2bfloat16(lo));
    const __hip_bfloat16_raw hi_raw = static_cast<__hip_bfloat16_raw>(__float2bfloat16(hi));
    return static_cast<std::uint32_t>(lo_raw.x) |
           (static_cast<std::uint32_t>(hi_raw.x) << 16);
}

#else // !NINFER_GFX906_COMPAT

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    std::uint32_t out;
    const std::uint32_t lo_bits = __float_as_uint(lo);
    const std::uint32_t hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}

#endif // NINFER_GFX906_COMPAT

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t bits) {
    return bf16x2_to_float2(load_vec<__nv_bfloat162>(&bits));
}

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    return load_vec<__half2>(&bits);
}

} // namespace ninfer::ops
