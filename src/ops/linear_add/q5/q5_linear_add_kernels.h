#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
#if defined(NINFER_GFX906_COMPAT)
// gfx906 pass 2e: register-resident small-T (2..5) GEMV with residual. Returns
// false (launches nothing) when T or the shape is outside its domain.
bool q5_linear_add_gemv_smallt_gfx906_launch(const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream);
#endif
void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
// gfx906 stage-8 wave64 LDS-tiled GEMM with fused residual accumulate.
void q5_linear_add_tiled_gfx906_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
