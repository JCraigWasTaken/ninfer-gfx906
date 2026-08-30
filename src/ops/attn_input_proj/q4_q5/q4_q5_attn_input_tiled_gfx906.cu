#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;

__host__ inline std::int32_t bf16_ld(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

template <int Cols>
void launch_pair(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                 Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Gfx906TiledSplitStoreEpilogue qk_epilogue{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data), bf16_ld(q),
        bf16_ld(k), kSplitRow};
    launch_rowsplit_tiled_gemm_gfx906<Q4TileAtomGfx906, Cols>(x, query_key_weight, kParentRows,
                                                              qk_epilogue, stream);
    const Gfx906TiledSplitStoreEpilogue gv_epilogue{
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data), bf16_ld(gate),
        bf16_ld(v), kSplitRow};
    launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, Cols>(x, gate_value_weight, kParentRows,
                                                              gv_epilogue, stream);
}

} // namespace

void q4_q5_attn_input_tiled_gfx906_launch(const Tensor& x, const Weight& query_key_weight,
                                          const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("attn input tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t cols = x.ne[1];
    if (cols <= 16) {
        launch_pair<16>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
    } else if (cols <= 32) {
        launch_pair<32>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
    } else {
        launch_pair<64>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
