#include "ops/linear/q4/q4_dispatch.h"

#if defined(NINFER_GFX906_COMPAT)
#include "ops/linear/gfx906/stage8_route.h"
#endif

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

#if defined(NINFER_GFX906_COMPAT)
namespace {

// gfx906 stage-3 reachability registry: the complete set of Q4 launches the
// compat selector may return. GEMV kernels are T=1-only and reached only via
// the t==1 whitelist rows; the SIMT GEMM kernels are shape-generic. Everything
// else (small-T mma, rowsplit mma tiles) traps on gfx906 and is rerouted.
constexpr Q4Launch kGfx906ReachableQ4Launches[] = {
    launch_q4_gemv_r1_w8_direct,  // T=1 rows (table returns it only at t==1)
    launch_q4_gemv_r4_w1_direct,  // T=1 rows (table returns it only at t==1)
    launch_q4_simt_r8_c4,         // generic (n, k, t)
    launch_q4_simt_r8_c8,         // generic (n, k, t)
};

Q4Launch gfx906_reroute(Q4Launch launch, std::int32_t k, std::int32_t t) {
    // Stage-8 tiled route: wave64 LDS-tiled GEMM for multi-token (prefill /
    // verify) shapes. T=1 stays on the tuned GEMV whitelist rows; T in [2,5]
    // stays on the stage-3 SIMT fallback, which still wins there (the tiled
    // kernel is latency-flat across T<=16). Thresholds set by the Tier-1
    // microbenchmarks in docs/gfx906/STAGE8-9-LOG.md.
    if (gfx906_stage8_tiled_enabled() && t >= 6 && k % 4 == 0) {
        if (launch != launch_q4_gemv_r1_w8_direct && launch != launch_q4_gemv_r4_w1_direct) {
            if (t <= 16) { return launch_q4_tiled_c16; }
            if (t <= 32) { return launch_q4_tiled_c32; }
            return launch_q4_tiled_c64;
        }
    }
    for (const Q4Launch safe : kGfx906ReachableQ4Launches) {
        if (launch == safe) { return launch; }
    }
    return t <= 4 ? launch_q4_simt_r8_c4 : launch_q4_simt_r8_c8;
}

} // namespace
#endif // NINFER_GFX906_COMPAT

// TP2 shard geometries. See the block comment in q5_dispatch.cpp for the two rules every
// groupwise family follows here: shard entries are additive and never collide with a tp1 extent,
// and a shard uses the family's generic runtime-dimension launchers rather than the parent's
// compile-time exact instantiations (Q4's draft-head small-T table is exact in both N and K).
// Returns nullptr when (n, k) is not a registered shard extent.
Q4Launch select_q4_tp2_shard_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool column_shard = (k == 5120 && (n == 512 ||    // 1024  / 2
                                             n == 2048 ||   // 4096  / 2 (gdn/query_key)
                                             n == 3072 ||   // 6144  / 2
                                             n == 3584 ||   // 7168  / 2 (attention/query_key)
                                             n == 17408 ||  // 34816 / 2 (mlp/gate_up)
                                             n == 65536))|| // 131072/ 2 (draft_head)
                              (k == 2048 && n == 65536);    // 131072/ 2 (draft_head, short K)
    if (!column_shard) { return nullptr; }
    if (t == 1) { return n >= 65536 ? launch_q4_gemv_r4_w1_direct : launch_q4_gemv_r1_w8_direct; }
    if (t <= 4) { return launch_q4_simt_r8_c4; }
    if (t <= 16) { return launch_q4_simt_r8_c8; }
    return launch_q4_mma_r64_c128;
}

// The tp1 table, exactly as it was: returns nullptr rather than throwing so the caller
// can fall back to the tp2 shard table. It is consulted FIRST, so a geometry that is
// both registered here and listed as a shard extent keeps its tuned tp1 launcher.
Q4Launch select_q4_a16_registered(std::int32_t n, std::int32_t k, std::int32_t t) {
    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 4096:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 34816:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 131072:
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 8) { return launch_q4_draft_head_small_t; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    case 2048:
        if (n == 131072) {
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 20) { return launch_q4_draft_head_small_t; }
            if (t <= 32) { return launch_q4_mma_r64_c32; }
            if (t <= 48) { return launch_q4_mma_r64_c48; }
            if (t <= 56) { return launch_q4_mma_r64_c56; }
            if (t <= 63) { return launch_q4_mma_r64_c72; }
            if (t == 64) { return launch_q4_mma_r64_c64_endpoint; }
            if (t <= 72) { return launch_q4_mma_r64_c72; }
            if (t <= 80) { return launch_q4_mma_r64_c80; }
            if (t <= 96) { return launch_q4_mma_r64_c96; }
            if (t <= 104) { return launch_q4_mma_r64_c104_bounded; }
            if (t <= 111) { return launch_q4_mma_r64_c112_partial; }
            if (t == 112) { return launch_q4_mma_r64_c112; }
            if (t <= 119) { return launch_q4_mma_r64_c120_partial; }
            if (t == 120) { return launch_q4_mma_r64_c120; }
            return launch_q4_mma_r64_c128;
        }
        break;
    case 1152:
        if (t < 4 || t > 131072 || (t % 4) != 0) { break; }
        switch (n) {
        case 3456:
            if (t <= 36) { return launch_q4_simt_r8_c4; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        case 4304:
            if (t == 4) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t == 12) { return launch_q4_simt_r8_c4; }
            if (t <= 24) { return launch_q4_simt_r8_c8; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    default:
        break;
    }

    return nullptr;
}

} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q4 linear: unsupported shape or T"); }
    if (const Q4Launch tp1 = select_q4_a16_registered(n, k, t); tp1 != nullptr) {
        return tp1;
    }
    if (const Q4Launch shard = select_q4_tp2_shard_launch(n, k, t); shard != nullptr) {
        return shard;
    }
    throw std::invalid_argument("q4 linear: unsupported shape or T");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
#if defined(NINFER_GFX906_COMPAT)
        // Keep the (n, k, t) whitelist as the shape gate, then force the
        // selection onto a gfx906-reachable kernel.
        return gfx906_reroute(select_q4_a16_launch(n, k, t), k, t);
#else
        return select_q4_a16_launch(n, k, t);
#endif
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q4 linear: unsupported policy");
}

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q4Launch launch = select_q4_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
