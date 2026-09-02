#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

__host__ inline std::int32_t bf16_ld(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

// Row extents come from the output planes, never from tp1 constants: the tiled GEMM bounds its
// weight reads by the row count it is handed, and the TP2 column shard is the head-local half
// (qk [2048,5120], value_z [6144,5120] = 3072 value + 3072 z rows against the tp1 parent's
// 4096 / 12288 = 6144 + 6144). Handing the parent's 4096 / 12288 to the shard walked the weight
// reads past the shard allocation (TP2 slice 3b: "Memory access fault ... Page not present").
__host__ inline void check_rows(const Weight& w, std::int32_t rows, const char* what) {
    if (w.n != rows) {
        throw std::invalid_argument(std::string("gdn input tiled gfx906: ") + what +
                                    " weight rows do not match the output planes");
    }
}

template <int Cols>
void launch_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::int32_t qk_rows = qk.ne[0];
    check_rows(qk_weight, qk_rows, "query_key");
    const Gfx906TiledStoreEpilogue qk_epilogue{static_cast<__nv_bfloat16*>(qk.data), bf16_ld(qk)};
    launch_rowsplit_tiled_gemm_gfx906<Q4TileAtomGfx906, Cols>(x, qk_weight, qk_rows, qk_epilogue,
                                                              stream);
    const std::int32_t value_rows   = value.ne[0];
    const std::int32_t value_z_rows = value_rows + z.ne[0];
    check_rows(value_z_weight, value_z_rows, "value_z");
    const Gfx906TiledSplitStoreEpilogue vz_epilogue{
        static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
        bf16_ld(value), bf16_ld(z), value_rows};
    launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, Cols>(x, value_z_weight, value_z_rows,
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
