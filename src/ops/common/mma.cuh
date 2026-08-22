#pragma once

#include "ops/common/memory.cuh"

// Marks a tensor-core kernel body that is not yet rewritten for gfx906. The
// early trap lets the AMD backend dead-code-eliminate the register tiles that
// only fit thanks to ldmatrix/mma on NVIDIA (they exceed the 64KiB scratch
// limit if materialized); dispatch must never route to such a kernel.
#if defined(NINFER_GFX906_COMPAT)
#define NINFER_GFX906_UNPORTED_KERNEL() \
    {                                   \
        __builtin_trap();               \
        return;                         \
    }
#else
#define NINFER_GFX906_UNPORTED_KERNEL()
#endif

namespace ninfer::ops {

#if defined(NINFER_GFX906_COMPAT)

// gfx906 has no tensor cores and no ldmatrix. Every kernel that reaches these
// helpers is REWRITE scope (dequant -> packed-FP16 FMA); until each family is
// rewritten, the stubs below keep the mechanical translation units compiling
// and abort the wave if executed. Dispatch must never route here on gfx906.
__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned) {
    r0 = r1 = 0u;
    __builtin_trap();
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned) {
    r0 = r1 = r2 = r3 = 0u;
    __builtin_trap();
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned) {
    r0 = r1 = 0u;
    __builtin_trap();
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned) {
    r0 = r1 = r2 = r3 = 0u;
    __builtin_trap();
}

__device__ __forceinline__ void mma_bf16(float&, float&, float&, float&, unsigned, unsigned,
                                         unsigned, unsigned, unsigned, unsigned) {
    __builtin_trap();
}

__device__ __forceinline__ void mma_f16(float&, float&, float&, float&, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, unsigned) {
    __builtin_trap();
}

__device__ __forceinline__ void mma_s8(int&, int&, int&, int&, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned) {
    __builtin_trap();
}

__device__ __forceinline__ void mma_fp8_e4m3(float&, float&, float&, float&, unsigned, unsigned,
                                             unsigned, unsigned, unsigned, unsigned) {
    __builtin_trap();
}

__device__ __forceinline__ void mma_tf32_bits(float&, float&, float&, float&, unsigned, unsigned,
                                              unsigned, unsigned, unsigned, unsigned) {
    __builtin_trap();
}

__device__ __forceinline__ void mma_tf32(float&, float&, float&, float&, float, float, float, float,
                                         float, float) {
    __builtin_trap();
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float&, float&, float&, float&, unsigned, unsigned,
                                               unsigned, unsigned, unsigned, unsigned, unsigned,
                                               unsigned) {
    __builtin_trap();
}

#else // !NINFER_GFX906_COMPAT

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

#endif // NINFER_GFX906_COMPAT

} // namespace ninfer::ops
