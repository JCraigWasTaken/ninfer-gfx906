#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"
#if defined(NINFER_GFX906_COMPAT)
#include "ops/linear/gfx906/q_gemv_gfx906.cuh"
#include "ops/linear/gfx906/stage8_route.h"
#endif

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

#if defined(NINFER_GFX906_COMPAT)
// Pass 2c tuning for the down projection (5120x17408): chunks in flight per
// lane, whether x is staged in LDS (34 KB) or read through L2, and threads
// per block (rows per block = threads / 32). Screened 2026-09-02: LDS x at
// 256 threads is capped at 4 waves/CU by the 34 KB (229 us); x through L2
// serialises on its own loads (192 us); a 1024-thread block sharing one LDS
// copy of x (16 waves/CU, grid 160) is the winner at kDepth 1 (139.5 us).
constexpr int kQ5DownDepthGfx906   = 1;
constexpr bool kQ5DownStageXGfx906 = true;
constexpr int kQ5DownThreadsGfx906 = 1024;

// TP2 slice 7: the row-parallel shard halves (o_proj / gdn/output K = 3072,
// mlp/down K = 8704). K = 8704 is the K-tail instantiation (8 x 1024 + 512,
// q_gemv_gfx906.cuh); x in LDS is 18 KB (tail step padded), so 512 threads
// (16 rows per block, grid 320) keeps 3 blocks = 24 waves per CU at kDepth 1.
constexpr int kQ5DownShardDepthGfx906   = 1;
constexpr int kQ5DownShardThreadsGfx906 = 512;
#endif

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);

    if (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) {
#if defined(NINFER_GFX906_COMPAT)
        // Pass 2: register-resident wave64 GEMV (NINFER_GFX906_PASS2=0 reverts).
        if (gfx906_pass2_gemv_enabled()) {
            q5_gemv_gfx906_residual_launch<5120, 6144>(xp, codes, high, scales, out, stream);
        } else
#endif
        {
            q5_rowsplit_gemv_residual_launch_kernel<5120, 6144, 16, 2, true>(
                xp, codes, high, scales, out, stream);
        }
    } else if (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408) {
#if defined(NINFER_GFX906_COMPAT)
        // Pass 2c: same register-resident GEMV, K=17408 (NINFER_GFX906_PASS2=0 reverts).
        if (gfx906_pass2_gemv_enabled()) {
            q5_gemv_gfx906_residual_launch<5120, 17408, kQ5DownDepthGfx906, kQ5DownStageXGfx906,
                                           kQ5DownThreadsGfx906>(xp, codes, high, scales, out,
                                                                 stream);
        } else
#endif
        {
            q5_rowsplit_gemv_residual_launch_kernel<5120, 17408, 16, 2, false>(
                xp, codes, high, scales, out, stream);
        }
#if defined(NINFER_GFX906_COMPAT)
    } else if (w.n == 5120 && w.k == 3072 && w.padded_shape[1] == 3072 &&
               gfx906_pass2_gemv_enabled()) {
        // TP2 slice 7: o_proj / gdn/output row-parallel shard (pass-2 route only;
        // the plan keeps the tiled residual GEMM when NINFER_GFX906_PASS2=0).
        q5_gemv_gfx906_residual_launch<5120, 3072>(xp, codes, high, scales, out, stream);
    } else if (w.n == 5120 && w.k == 8704 && w.padded_shape[1] == 8704 &&
               gfx906_pass2_gemv_enabled()) {
        // TP2 slice 7: mlp/down row-parallel shard, K-tail instantiation.
        q5_gemv_gfx906_residual_launch<5120, 8704, kQ5DownShardDepthGfx906, true,
                                       kQ5DownShardThreadsGfx906>(xp, codes, high, scales, out,
                                                                  stream);
#endif
    } else {
        throw std::invalid_argument("q5 linear_add GEMV: unsupported exact shape");
    }
    CUDA_CHECK(cudaGetLastError());
}

#if defined(NINFER_GFX906_COMPAT)
// Pass 2e: small-T (2..5) variant of the two residual GEMVs above. Same chunk
// streams as T=1; x is T token rows staged per K-slab in LDS (<= 32 KB), T
// accumulators per lane, residual read from out[t * out_ld + row].
constexpr int kQ5AddSmallTThreadsGfx906 = 512;
constexpr int kQ5AddSmallTLdsCapGfx906  = 32768;

bool q5_linear_add_gemv_smallt_gfx906_launch(const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    const int t = x.ne[1];
    if (t < 2 || t > 5) { return false; }
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);
    const int x_ld     = static_cast<int>(x.nb[1] / sizeof(__nv_bfloat16));
    const int out_ld   = static_cast<int>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    bool launched      = false;
    if (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) {
        launched = q5_gemv_smallt_gfx906_dispatch<5120, 6144, true, kQ5AddSmallTThreadsGfx906,
                                                  kQ5AddSmallTLdsCapGfx906>(
            t, xp, x_ld, codes, high, scales, out, out_ld, stream);
    } else if (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408) {
        launched = q5_gemv_smallt_gfx906_dispatch<5120, 17408, true, kQ5AddSmallTThreadsGfx906,
                                                  kQ5AddSmallTLdsCapGfx906>(
            t, xp, x_ld, codes, high, scales, out, out_ld, stream);
    } else if (w.n == 5120 && w.k == 3072 && w.padded_shape[1] == 3072) {
        // TP2 slice 7: row-parallel shard halves (K = 3072 clean, K = 8704 K-tail).
        launched = q5_gemv_smallt_gfx906_dispatch<5120, 3072, true, kQ5AddSmallTThreadsGfx906,
                                                  kQ5AddSmallTLdsCapGfx906>(
            t, xp, x_ld, codes, high, scales, out, out_ld, stream);
    } else if (w.n == 5120 && w.k == 8704 && w.padded_shape[1] == 8704) {
        launched = q5_gemv_smallt_gfx906_dispatch<5120, 8704, true, kQ5AddSmallTThreadsGfx906,
                                                  kQ5AddSmallTLdsCapGfx906>(
            t, xp, x_ld, codes, high, scales, out, out_ld, stream);
    }
    if (launched) { CUDA_CHECK(cudaGetLastError()); }
    return launched;
}
#endif

} // namespace ninfer::ops::detail
