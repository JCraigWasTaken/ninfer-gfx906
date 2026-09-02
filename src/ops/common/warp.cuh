#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops {

// gfx906 (wave64) note: kWarpSize stays 32 as the *logical* subgroup width.
// core/hip_compat.h maps __shfl_*_sync(mask, ...) to HIP's unmasked shuffles;
// a width-32 shuffle subdivides a 64-lane wave into two independent 32-lane
// groups, so every helper below keeps its CUDA semantics. kFullWarpMask is
// accepted and ignored by the compat shims.
inline constexpr int kWarpSize          = 32;
inline constexpr unsigned kFullWarpMask = 0xffffffffu;

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_reduce_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize>
__device__ __forceinline__ float warp_max(float x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(mask, x, offset, Width));
    }
    return x;
}

#if defined(NINFER_GFX906_COMPAT)
// Wave64 DPP reduction (pass 2, donor llama.cpp-gfx906 warp_reduce_amd_f32
// ladder): sums x across each 32-lane half of a wave64 independently, both
// halves at once, using row-local DPP moves instead of ds_bpermute shuffles.
// Every lane of a half ends with that half's total.
//   xor1 / xor2 : quad_perm(1,0,3,2) = 0xB1, quad_perm(2,3,0,1) = 0x4E
//   +4 / +8     : row_ror:4 = 0x124, row_ror:8 = 0x128 (after the quad steps
//                 every lane of a quad holds the quad sum, so rotating by 4 then
//                 8 within the 16-lane row gives the row total in every lane)
//   xor16       : ds_swizzle bit mode (and 0x1f, or 0, xor 0x10) = 0x401F; the
//                 swizzle never crosses a 32-lane group, which is the point.
// Used by the pass-2 GEMV only; warp_reduce_sum users are unchanged.
__device__ __forceinline__ float gfx906_reduce_sum32(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    // row_mask = bank_mask = 0xf (all rows/banks), bound_ctrl = true.
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x124, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x128, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x401F));
#else
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        x += __shfl_xor_sync(kFullWarpMask, x, offset, 32);
    }
#endif
    return x;
}
#endif // NINFER_GFX906_COMPAT

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float x, float* sums) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int Warps = BlockSize / kWarpSize;

    x = warp_reduce_sum(x);
    if constexpr (Warps == 1) { return x; }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) { sums[warp] = x; }
    __syncthreads();

    x = threadIdx.x < Warps ? sums[lane] : 0.0f;
    if (warp == 0) { x = warp_reduce_sum<Warps>(x); }
    return x;
}

} // namespace ninfer::ops
