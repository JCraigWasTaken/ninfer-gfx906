#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void q5_linear_add_tiled_gfx906_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("q5 linear_add tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t rows   = residual_out.ne[0];
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld =
        static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const Gfx906TiledResidualAddEpilogue epilogue{
        static_cast<__nv_bfloat16*>(residual_out.data), out_ld};
    if (cols <= 16) {
        launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, 16>(x, w, rows, epilogue, stream);
    } else if (cols <= 32) {
        launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, 32>(x, w, rows, epilogue, stream);
    } else {
        launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, 64>(x, w, rows, epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
