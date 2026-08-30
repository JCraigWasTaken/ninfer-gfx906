#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include "core/device.h"
#include "ops/linear/q5/q5_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Cols>
void launch_q5_tiled(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("q5 tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t rows   = out.ne[0];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, Cols>(
        x, w, rows, Gfx906TiledStoreEpilogue{static_cast<__nv_bfloat16*>(out.data), out_ld},
        stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q5_tiled_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q5_tiled<16>(x, w, out, stream);
}

void launch_q5_tiled_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q5_tiled<32>(x, w, out, stream);
}

void launch_q5_tiled_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q5_tiled<64>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
