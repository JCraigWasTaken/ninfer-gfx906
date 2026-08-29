#include "ops/linear/q5/q5_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

#if defined(NINFER_GFX906_COMPAT)
namespace {

// gfx906 stage-3 reachability registry: the complete set of Q5 launches the
// compat selector may return. The GEMV kernel is T=1-only and reached only via
// the t==1 whitelist rows; the split-exact SIMT kernels are reached only on
// their tuned small-T rows; the SIMT GEMM kernels are shape-generic. The
// rowsplit mma tiles trap on gfx906 and are rerouted.
constexpr Q5Launch kGfx906ReachableQ5Launches[] = {
    launch_q5_gemv_r16_s2_x,      // T=1 rows (table returns it only at t==1)
    launch_q5_simt_split2_exact,  // small-T rows only (exact-shape kernel)
    launch_q5_simt_split4_exact,  // small-T rows only (exact-shape kernel)
    launch_q5_simt_r8_c4,         // generic (n, k, t)
    launch_q5_simt_r8_c8,         // generic (n, k, t)
};

Q5Launch gfx906_reroute(Q5Launch launch, std::int32_t t) {
    for (const Q5Launch safe : kGfx906ReachableQ5Launches) {
        if (launch == safe) { return launch; }
    }
    return t <= 4 ? launch_q5_simt_r8_c4 : launch_q5_simt_r8_c8;
}

} // namespace
#endif // NINFER_GFX906_COMPAT

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q5 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t <= 4) { return launch_q5_simt_r8_c4; }
            if (t <= 16) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            if (t <= 64) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 16) { return launch_q5_simt_r8_c4; }
            return launch_q5_mma_r64_c128;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 1152:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 76) { return launch_q5_simt_r8_c4; }
            if (t <= 636) { return launch_q5_mma_r64_c64; }
            if (t <= 700) { return launch_q5_mma_r64_c128; }
            if (t == 704) { return launch_q5_mma_r64_c64; }
            if (t <= 828) { return launch_q5_mma_r64_c128; }
            if (t == 832) { return launch_q5_mma_r64_c64; }
            if (t <= 896) { return launch_q5_mma_r64_c128; }
            if (t <= 960) { return launch_q5_mma_r64_c64; }
            if (t <= 1024) { return launch_q5_mma_r64_c128; }
            if (t <= 1088) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 4304:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 120) { return launch_q5_simt_r8_c4; }
            if (t <= 1148) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q5 linear: unsupported shape or T");
}

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
#if defined(NINFER_GFX906_COMPAT)
        // Keep the (n, k, t) whitelist as the shape gate, then force the
        // selection onto a gfx906-reachable kernel.
        return gfx906_reroute(select_q5_a16_launch(n, k, t), t);
#else
        return select_q5_a16_launch(n, k, t);
#endif
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q5 linear: unsupported policy");
}

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q5Launch launch = select_q5_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
