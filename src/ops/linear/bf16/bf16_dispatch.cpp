#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

#if defined(NINFER_GFX906_COMPAT)
namespace {

// gfx906 stage-3 rerouting. launch_bf16_mma traps on gfx906, and the safe
// small-T family only covers T in [2, 32], so prefill token counts are walked
// in 32-token slices through the small-T kernels (a leftover single token goes
// through the decode GEMV). Slow prefill, correct output; the real bf16 GEMM
// rewrite is stage-8 scope.
//
// gfx906 reachability registry for this table: launch_bf16_decode (T=1),
// launch_bf16_small_t (T in [2, 32]), launch_bf16_sliced_small_t (any T>1,
// composed exclusively of the two former launches). launch_bf16_mma is
// unreachable under NINFER_GFX906_COMPAT.
void launch_bf16_sliced_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    std::int32_t offset       = 0;
    while (offset < tokens) {
        const std::int32_t count = std::min<std::int32_t>(kBf16SmallTMaxTokens, tokens - offset);
        const Tensor x_slice     = x.slice(1, offset, count);
        Tensor out_slice         = out.slice(1, offset, count);
        if (count == 1) {
            launch_bf16_decode(x_slice, weight, out_slice, stream);
        } else {
            launch_bf16_small_t(x_slice, weight, out_slice, stream);
        }
        offset += count;
    }
}

} // namespace
#endif // NINFER_GFX906_COMPAT

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool supported_problem = (n == 14336 && k == 5120) || (n == 5120 && k == 6144);
    // TP2 shards of the two registered problems: attention/query_key_gate_value splits
    // column-parallel (14336 -> 7168) and attention/output row-parallel (6144 -> 3072). BF16's
    // decode and small-T launchers are exact-geometry (see bf16_gemv.cu / bf16_small_t.cu), so a
    // shard resolves to the runtime-dimension MMA launcher at every T rather than gaining a second
    // tuned kernel set. See q5_dispatch.cpp for the rules every family follows here.
    const bool tp2_shard = (n == 7168 && k == 5120) || (n == 5120 && k == 3072);
    if ((!supported_problem && !tp2_shard) || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    if (tp2_shard) {
#if defined(NINFER_GFX906_COMPAT)
        throw std::invalid_argument(
            "bf16 linear: tp2 shard has no gfx906-reachable launcher (decode/small-T are "
            "exact-geometry, the MMA GEMM traps)");
#else
        return launch_bf16_mma;
#endif
    }
    if (t == 1) { return launch_bf16_decode; }
#if defined(NINFER_GFX906_COMPAT)
    if (t <= kBf16SmallTMaxTokens) { return launch_bf16_small_t; }
    return launch_bf16_sliced_small_t;
#else
    const std::int32_t small_t_end =
        n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
    if (t <= small_t_end) { return launch_bf16_small_t; }
    return launch_bf16_mma;
#endif
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
