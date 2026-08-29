#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include <algorithm>

namespace ninfer::ops::detail {

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
#if defined(NINFER_GFX906_COMPAT)
    // gfx906 stage-3 rerouting: the mma launch traps on gfx906. Walk every
    // token count through the small-T kernels ([2, 32] tokens per launch; a
    // leftover single token uses the decode GEMV). Slow prefill, correct
    // output.
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += kBf16AttnInputSmallTMaxTokens) {
        const std::int32_t count =
            std::min<std::int32_t>(kBf16AttnInputSmallTMaxTokens, x.ne[1] - offset);
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor gate_slice    = gate.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        if (count == 1) {
            bf16_attn_input_decode_launch(x_slice, weight, q_slice, gate_slice, k_slice, v_slice,
                                          stream);
        } else {
            bf16_attn_input_small_t_launch(x_slice, weight, q_slice, gate_slice, k_slice, v_slice,
                                           stream);
        }
    }
#else
    if (x.ne[1] <= kBf16AttnInputSmallTDispatchEnd) {
        bf16_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    bf16_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
#endif
}

} // namespace ninfer::ops::detail
