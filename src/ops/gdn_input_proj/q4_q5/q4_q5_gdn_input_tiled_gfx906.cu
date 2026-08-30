#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kValueZRows = 12288;

__host__ inline std::int32_t bf16_ld(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

template <int Cols>
void launch_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    const Gfx906TiledStoreEpilogue qk_epilogue{static_cast<__nv_bfloat16*>(qk.data), bf16_ld(qk)};
    launch_rowsplit_tiled_gemm_gfx906<Q4TileAtomGfx906, Cols>(x, qk_weight, kQkRows, qk_epilogue,
                                                              stream);
    const Gfx906TiledSplitStoreEpilogue vz_epilogue{
        static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
        bf16_ld(value), bf16_ld(z), kValueRows};
    launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, Cols>(x, value_z_weight, kValueZRows,
                                                              vz_epilogue, stream);
}

} // namespace

void q4_q5_gdn_input_tiled_gfx906_launch(const Tensor& x, const Weight& qk_weight,
                                         const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                         Tensor& z, cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("gdn input tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t cols = x.ne[1];
    if (cols <= 16) {
        launch_pair<16>(x, qk_weight, value_z_weight, qk, value, z, stream);
    } else if (cols <= 32) {
        launch_pair<32>(x, qk_weight, value_z_weight, qk, value, z, stream);
    } else {
        launch_pair<64>(x, qk_weight, value_z_weight, qk, value, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
