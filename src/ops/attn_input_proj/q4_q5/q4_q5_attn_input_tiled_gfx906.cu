#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

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
// weight reads by the row count it is handed, and the TP2 column shard carries the same section
// order at half the heads ([3584,5120] = 3072 query/gate + 512 key/value rows against the tp1
// parent's 7168 = 6144 + 1024). Handing the parent's 7168 to a 3584-row shard walked the weight
// reads past the shard allocation (TP2 slice 3b: "Memory access fault ... Page not present").
__host__ inline std::int32_t pair_rows(const Weight& w, const Tensor& first, const Tensor& second,
                                       const char* what) {
    const std::int32_t rows = first.ne[0] + second.ne[0];
    if (w.n != rows) {
        throw std::invalid_argument(std::string("attn input tiled gfx906: ") + what +
                                    " weight rows do not match the output planes");
    }
    return rows;
}

template <int Cols>
void launch_pair(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                 Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const std::int32_t qk_rows = pair_rows(query_key_weight, q, k, "query_key");
    const Gfx906TiledSplitStoreEpilogue qk_epilogue{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data), bf16_ld(q),
        bf16_ld(k), q.ne[0]};
    launch_rowsplit_tiled_gemm_gfx906<Q4TileAtomGfx906, Cols>(x, query_key_weight, qk_rows,
                                                              qk_epilogue, stream);
    const std::int32_t gv_rows = pair_rows(gate_value_weight, gate, v, "gate_value");
    const Gfx906TiledSplitStoreEpilogue gv_epilogue{
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data), bf16_ld(gate),
        bf16_ld(v), gate.ne[0]};
    launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, Cols>(x, gate_value_weight, gv_rows,
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
