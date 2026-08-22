#pragma once
#include "core/hip_compat.h"

#if defined(__HIPCC__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// ROCm's hip_bf16.h mirrors the CUDA bf16 API on __hip_bfloat16 types.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;
typedef __hip_bfloat162_raw __nv_bfloat162_raw;

// Present in CUDA but absent from ROCm 6.4's header: explicit
// round-to-nearest-even variants. The HIP base conversions round RNE.
__device__ __host__ inline __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

__device__ __host__ inline __hip_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return __float22bfloat162_rn(float2{x, y});
}

__device__ inline __hip_bfloat162 __hsub2_rn(__hip_bfloat162 a, __hip_bfloat162 b) {
    return __hsub2(a, b);
}

#else // host-only translation unit

// hip_bf16.h cannot be parsed outside HIP compilation (it reaches for amdgcn
// builtins), but host code only moves bf16 data around. Provide
// layout-compatible stand-ins; all arithmetic stays device-side.
struct __nv_bfloat16 {
    unsigned short x;
};
struct __nv_bfloat162 {
    __nv_bfloat16 x, y;
};
struct __nv_bfloat16_raw {
    unsigned short x;
};
struct __nv_bfloat162_raw {
    unsigned short x, y;
};

#endif // __HIPCC__
