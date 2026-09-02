#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/gfx906/q_gemv_gfx906.cuh"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"
#include "ops/linear/gfx906/stage8_route.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

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

// TP2 slice 7: T=1 decode of the column shard ([3584,5120] = 3072 query/gate + 512 key/value
// rows per weight). gate_value (Q5) rides the pass-2 register-resident GEMV with the split store
// epilogue at the gate rows (kDepth 1, 512 threads, x in LDS); query_key (Q4) rides the upstream
// runtime-rows Q4 GEMV with the split at the query rows (the tp1 T=1 kernel for this weight,
// q4_q5_attn_input_small_t.cu; no plain-Q4 pass-2 kernel exists). Extents are checked against
// the planes exactly as the tiled path does. NINFER_GFX906_PASS2=0 keeps the tiled GEMM.
constexpr std::int32_t kShardHidden   = 5120;
constexpr std::int32_t kShardSplitRow = 3072;
constexpr std::int32_t kShardRows     = 3584;
constexpr int kShardQ5DepthGfx906     = 1;
constexpr int kShardQ5ThreadsGfx906   = 512;

bool launch_t1_pass2(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    if (x.ne[0] != kShardHidden || q.ne[0] != kShardSplitRow || gate.ne[0] != kShardSplitRow ||
        k.ne[0] != kShardRows - kShardSplitRow || v.ne[0] != kShardRows - kShardSplitRow ||
        query_key_weight.padded_shape[1] != kShardHidden ||
        gate_value_weight.padded_shape[1] != kShardHidden) {
        return false;
    }
    (void)pair_rows(query_key_weight, q, k, "query_key");
    (void)pair_rows(gate_value_weight, gate, v, "gate_value");
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    {
        using Schedule = Q4GemvR1W8DirectSchedule;
        const dim3 grid(static_cast<unsigned>(div_up(kShardRows, Schedule::kRowsPerCta)), 1u, 1u);
        q4_rowsplit_gemv_kernel<Schedule, true, kShardSplitRow>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                xp, static_cast<const std::uint8_t*>(query_key_weight.qdata),
                static_cast<const std::uint8_t*>(query_key_weight.scales),
                static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                kShardRows, kShardHidden);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        constexpr int kRowsPerBlock = kShardQ5ThreadsGfx906 / Gfx906GemvGeometry::kLanesPerRow;
        constexpr int kGrid         = kShardRows / kRowsPerBlock;
        q5_gemv_gfx906_kernel<kShardRows, kShardHidden, false, true, kShardSplitRow,
                              Q5GemvStoreEpilogue, false, false, kShardQ5DepthGfx906, true,
                              kShardQ5ThreadsGfx906><<<kGrid, kShardQ5ThreadsGfx906, 0, stream>>>(
            xp, static_cast<const std::uint8_t*>(gate_value_weight.qdata),
            static_cast<const std::uint8_t*>(gate_value_weight.qhigh),
            static_cast<const std::uint8_t*>(gate_value_weight.scales),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data));
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

} // namespace

void q4_q5_attn_input_tiled_gfx906_launch(const Tensor& x, const Weight& query_key_weight,
                                          const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("attn input tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t cols = x.ne[1];
    if (cols == 1 && gfx906_pass2_gemv_enabled() &&
        launch_t1_pass2(x, query_key_weight, gate_value_weight, q, gate, k, v, stream)) {
        return;
    }
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
