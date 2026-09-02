#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

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

// TP2 slice 7: T=1 decode of the column shard (qk [2048,5120], value_z [6144,5120] = 3072 value
// + 3072 z). value_z rides the pass-2 register-resident GEMV (q_gemv_gfx906.cuh) with the split
// store epilogue at the value rows (the tp1 pass-2d geometry: kDepth 1, 512 threads, x in LDS);
// query_key rides the upstream runtime-rows Q4 GEMV (the tp1 T=1 kernel for this weight,
// q4_q5_gdn_input_independent.cu; no plain-Q4 pass-2 kernel exists). Extents are checked
// against the planes exactly as the tiled path does. NINFER_GFX906_PASS2=0 keeps the tiled GEMM.
constexpr std::int32_t kShardHidden     = 5120;
constexpr std::int32_t kShardQkRows     = 2048;
constexpr std::int32_t kShardValueRows  = 3072;
constexpr std::int32_t kShardValueZRows = 6144;
constexpr int kShardQ5DepthGfx906       = 1;
constexpr int kShardQ5ThreadsGfx906     = 512;

bool launch_t1_pass2(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                     Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[0] != kShardHidden || qk.ne[0] != kShardQkRows || value.ne[0] != kShardValueRows ||
        z.ne[0] != kShardValueZRows - kShardValueRows || qk_weight.padded_shape[1] != kShardHidden ||
        value_z_weight.padded_shape[1] != kShardHidden) {
        return false;
    }
    check_rows(qk_weight, kShardQkRows, "query_key");
    check_rows(value_z_weight, kShardValueZRows, "value_z");
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    {
        using Schedule = Q4GemvR1W8DirectSchedule;
        const dim3 grid(static_cast<unsigned>(div_up(kShardQkRows, Schedule::kRowsPerCta)), 1u, 1u);
        q4_rowsplit_gemv_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
            xp, static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(qk.data), nullptr, kShardQkRows, kShardHidden);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        constexpr int kRowsPerBlock = kShardQ5ThreadsGfx906 / Gfx906GemvGeometry::kLanesPerRow;
        constexpr int kGrid         = kShardValueZRows / kRowsPerBlock;
        q5_gemv_gfx906_kernel<kShardValueZRows, kShardHidden, false, true, kShardValueRows,
                              Q5GemvStoreEpilogue, false, false, kShardQ5DepthGfx906, true,
                              kShardQ5ThreadsGfx906><<<kGrid, kShardQ5ThreadsGfx906, 0, stream>>>(
            xp, static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

} // namespace

void q4_q5_gdn_input_tiled_gfx906_launch(const Tensor& x, const Weight& qk_weight,
                                         const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                         Tensor& z, cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("gdn input tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t cols = x.ne[1];
    if (cols == 1 && gfx906_pass2_gemv_enabled() &&
        launch_t1_pass2(x, qk_weight, value_z_weight, qk, value, z, stream)) {
        return;
    }
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
