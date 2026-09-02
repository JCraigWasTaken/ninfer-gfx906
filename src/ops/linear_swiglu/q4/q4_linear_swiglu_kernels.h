#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
#if defined(NINFER_GFX906_COMPAT)
// gfx906 pass 2e: register-resident small-T (2..5) gate/up pair GEMV. Returns
// false (launches nothing) when T is outside its domain.
bool q4_linear_swiglu_gemv_pair_smallt_gfx906_launch(const Tensor& x, const Weight& w,
                                                     Tensor& out, cudaStream_t stream);
// TP2 slice 7: the column-parallel gate_up shard ([17408,5120] per rank = gate
// rows [0,8704) | up rows [8704,17408) of that rank's half) on the pass-2 pair
// GEMV at kIntermediate = 8704 (T=1) / its small-T variant (T=2..5). Returns
// false (launches nothing) when the shape, T, or the NINFER_GFX906_PASS2 /
// _SMALLT knobs are outside its domain, so the wrapper keeps its composition.
bool q4_linear_swiglu_gemv_pair_shard_gfx906_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                    cudaStream_t stream);
#endif
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
