// gfx906 port: throwing stubs for the deleted FP8 / NVFP4 / sparse-MoE op
// families (pattern: ninfer-3090's nvfp4_sm86_stubs.cpp, extended to cover the
// fp8 and sparse_moe families). gfx906 has no FP8/NVFP4 hardware and the
// Qwen3.8-27B groupwise-int artifact never routes here; reaching any of these
// entry points is a dispatch bug or an unsupported artifact.

#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_fp8() {
    throw std::runtime_error("FP8 execution is not supported on gfx906");
}

[[noreturn]] void reject_nvfp4() {
    throw std::runtime_error("NVFP4 execution is not supported on gfx906");
}

[[noreturn]] void reject_moe() {
    throw std::runtime_error("sparse-MoE execution is not built for gfx906 (dense targets only)");
}

} // namespace

// --- linear/fp8 --------------------------------------------------------------

Fp8WeightGeometry validate_fp8_weight(const Weight&, const char*) { reject_fp8(); }

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                std::int32_t, std::int32_t) {
    reject_fp8();
}

void fp8_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                  cudaStream_t) {
    reject_fp8();
}

void launch_fp8_decode(const Tensor&, const Weight&, Tensor&, cudaStream_t) { reject_fp8(); }
void launch_fp8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t) { reject_fp8(); }
void launch_fp8_vocabulary_a16_mma(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_fp8();
}

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8();
}
void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8();
}

// --- linear/nvfp4 ------------------------------------------------------------

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight&, const char*) { reject_nvfp4(); }

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                    cudaStream_t) {
    reject_nvfp4();
}

void launch_nvfp4_decode(const Tensor&, const Weight&, Tensor&, cudaStream_t) { reject_nvfp4(); }
void launch_nvfp4_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t) { reject_nvfp4(); }

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4();
}
void launch_nvfp4_w4a4(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4();
}

// --- TMA launchers (deleted ninfer_nvfp4_tma archive) ------------------------

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    reject_nvfp4();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, std::int32_t, float,
                                     cudaStream_t) {
    reject_nvfp4();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    reject_nvfp4();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    reject_nvfp4();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                         std::int32_t, float, cudaStream_t) {
    reject_nvfp4();
}

// --- attn_input_proj ---------------------------------------------------------

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_fp8();
}
void fp8_attn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  cudaStream_t) {
    reject_fp8();
}
void fp8_attn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                   cudaStream_t) {
    reject_fp8();
}
void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    reject_fp8();
}
void fp8_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                             LinearPolicy, WorkspaceArena*, cudaStream_t) {
    reject_fp8();
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}
void nvfp4_attn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                    Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_attn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                               LinearPolicy, WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}

// --- gdn_input_proj ----------------------------------------------------------

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_fp8();
}
void fp8_gdn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_input_a16_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_input_a8_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, WorkspaceArena&,
                               cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                            WorkspaceArena*, cudaStream_t) {
    reject_fp8();
}

std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t,
                                                      std::int32_t) {
    reject_fp8();
}
std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t,
                                                    std::int32_t) {
    reject_fp8();
}
void fp8_gdn_snapshot_fused_launch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                   const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                   Tensor&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_record_fused_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                 Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                               LinearPolicy, WorkspaceArena&, cudaStream_t) {
    reject_fp8();
}
void fp8_gdn_record_dispatch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                             Tensor&, LinearPolicy, WorkspaceArena&, cudaStream_t) {
    reject_fp8();
}

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}
void nvfp4_gdn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Nvfp4W4a4Workspace,
                                 cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                              WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}
std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}
void nvfp4_gdn_snapshot_decode_launch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                      const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                      Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_snapshot_small_t_launch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                       const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                       Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_snapshot_post_launch(const Tensor&, const Tensor&, Tensor&, const Tensor&,
                                    const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                    cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                 Tensor&, Tensor&, LinearPolicy, WorkspaceArena&, cudaStream_t) {
    reject_nvfp4();
}

// --- linear_add --------------------------------------------------------------

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                    std::int32_t, std::int32_t) {
    reject_fp8();
}
void fp8_linear_add_decode_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_linear_add_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    reject_fp8();
}
void fp8_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena&,
                             cudaStream_t) {
    reject_fp8();
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                      std::int32_t, std::int32_t) {
    reject_nvfp4();
}
void nvfp4_linear_add_decode_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_linear_add_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                               cudaStream_t) {
    reject_nvfp4();
}

// --- linear_swiglu -----------------------------------------------------------

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_fp8();
}
void fp8_linear_swiglu_decode_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_linear_swiglu_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_fp8();
}
void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    reject_fp8();
}
void fp8_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                WorkspaceArena&, cudaStream_t) {
    reject_fp8();
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}
void nvfp4_linear_swiglu_decode_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_linear_swiglu_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    reject_nvfp4();
}
void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                  WorkspaceArena&, cudaStream_t) {
    reject_nvfp4();
}

// --- sparse_moe --------------------------------------------------------------

std::size_t sparse_moe_decode_workspace_bytes() { reject_moe(); }
SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType, QType) { reject_moe(); }
void sparse_moe_decode_launch_d3_small_t(const Tensor&, const SparseMoeWeights&, const int*, float*,
                                         std::int32_t, SparseMoeSmallTD3Schedule, cudaStream_t,
                                         const int*) {
    reject_moe();
}
void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights&, Tensor&, const int*, const float*,
                                         const float*, const float*, std::int32_t,
                                         SparseMoeSmallTD4Schedule, cudaStream_t, const int*) {
    reject_moe();
}
void sparse_moe_decode_launch(const Tensor&, const SparseMoeWeights&, Tensor&,
                              const SparseMoeDecodeWorkspace&, cudaStream_t) {
    reject_moe();
}

bool sparse_moe_uses_prefill(std::int32_t, QType, QType) noexcept { return false; }
std::size_t sparse_moe_prefill_workspace_bytes(std::int32_t) { reject_moe(); }
SparseMoePrefillPlan resolve_sparse_moe_prefill_plan(std::int32_t, QType, QType) { reject_moe(); }
void sparse_moe_prefill_launch(const Tensor&, const SparseMoeWeights&, Tensor&,
                               const SparseMoePrefillPlan&, const SparseMoePrefillWorkspace&,
                               cudaStream_t) {
    reject_moe();
}

bool sparse_moe_uses_small_t(std::int32_t) noexcept { return false; }
std::size_t sparse_moe_small_t_workspace_bytes(std::int32_t) { reject_moe(); }
SparseMoeSmallTPlan resolve_sparse_moe_small_t_plan(std::int32_t, QType, QType) { reject_moe(); }
void sparse_moe_small_t_launch(const Tensor&, const SparseMoeWeights&, Tensor&,
                               const SparseMoeSmallTPlan&, const SparseMoeSmallTWorkspace&,
                               cudaStream_t) {
    reject_moe();
}


// --- TP2 shard forms (donor tp2 plumbing; fp8/nvfp4 kernels are not built) ----

void fp8_attn_input_dispatch_shard(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                   Tensor&, LinearPolicy, WorkspaceArena*, cudaStream_t) {
    reject_fp8();
}
void nvfp4_attn_input_dispatch_shard(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, LinearPolicy, WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}
void fp8_gdn_input_dispatch_shard(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                                  WorkspaceArena*, cudaStream_t) {
    reject_fp8();
}
void nvfp4_gdn_input_dispatch_shard(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                                    WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}
std::size_t nvfp4_linear_swiglu_shard_workspace_capacity_bytes(LinearPolicy, std::int32_t,
                                                               std::int32_t) {
    reject_nvfp4();
}
void nvfp4_linear_swiglu_dispatch_shard(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                        WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}
void fp8_linear_swiglu_dispatch_shard(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                      WorkspaceArena*, cudaStream_t) {
    reject_fp8();
}

} // namespace ninfer::ops::detail
