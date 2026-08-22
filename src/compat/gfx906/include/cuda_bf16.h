#pragma once
#include "core/hip_compat.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// ROCm's hip_bf16.h mirrors the CUDA bf16 API on __hip_bfloat16 types.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;
typedef __hip_bfloat162_raw __nv_bfloat162_raw;

#if defined(__HIPCC__)
// Present in CUDA but absent from ROCm 6.4's header: explicit
// round-to-nearest-even float -> bf16. HIP's __float2bfloat16 rounds RNE.
__device__ __host__ inline __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}
#endif
