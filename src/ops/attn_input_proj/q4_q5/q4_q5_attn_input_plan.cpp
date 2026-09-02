#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#if defined(NINFER_GFX906_COMPAT)
#include "ops/linear/gfx906/stage8_route.h"
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

#if defined(NINFER_GFX906_COMPAT)
// gfx906 stage-3 rerouting: the grouped-pair mma kernels trap on gfx906. The
// small-T kernel (ParentSplitFixed -> q4_rowsplit_gemv/simt split-output
// kernels, T in [1,16]) is the only reachable schedule; larger token counts
// are walked in 16-token slices at execute time. Slow prefill, correct output.
constexpr std::array<RouteSpec, 1> kRoutes{{
    {{1, kAnyCols}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
}};

// Stage-8 routing: one wave64 LDS-tiled GEMM per parent weight for every
// T > 1 (the Q5 half's split4/simt fallbacks lose to the tiled kernel from
// T=2 up, and by far more than the Q4 half's small-T fallback gains).
// Selected at runtime by the NINFER_GFX906_STAGE8 gate.
constexpr std::array<RouteSpec, 2> kRoutesTiled{{
    {{1, 1}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{2, kAnyCols}, Q4Q5AttnInputScheduleId::TiledGfx906},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last == kAnyCols &&
           kRoutesTiled[0].cols.first == 1 &&
           kRoutesTiled[0].cols.last + 1 == kRoutesTiled[1].cols.first &&
           kRoutesTiled[1].cols.last == kAnyCols;
}

// Token-slice width matched to the small-T launcher's T-domain [1, 16].
constexpr std::int32_t kGfx906SmallTSlice = 16;
#else
constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, 20}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3},
    {{21, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last + 1 == kRoutes[2].cols.first && kRoutes[2].cols.last == kAnyCols;
}
#endif // NINFER_GFX906_COMPAT

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::TiledGfx906:
        return "attn_input_proj.q4_q5.tiled.gfx906";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

#if defined(NINFER_GFX906_COMPAT)
    if (gfx906_stage8_tiled_enabled()) {
        for (const RouteSpec& route : kRoutesTiled) {
            if (!route.cols.contains(problem.cols)) { continue; }
            return {route.schedule};
        }
        throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
    }
#endif
    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
#if defined(NINFER_GFX906_COMPAT)
        // Walk arbitrary token counts through the [1,16]-token kernel.
        for (std::int32_t offset = 0; offset < problem.cols; offset += kGfx906SmallTSlice) {
            const std::int32_t count =
                std::min<std::int32_t>(kGfx906SmallTSlice, problem.cols - offset);
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor q_slice       = q.slice(1, offset, count);
            Tensor gate_slice    = gate.slice(1, offset, count);
            Tensor k_slice       = k.slice(1, offset, count);
            Tensor v_slice       = v.slice(1, offset, count);
            q4_q5_attn_input_small_t_launch(x_slice, query_key_weight, gate_value_weight, q_slice,
                                            gate_slice, k_slice, v_slice, stream);
        }
#else
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
#endif
        return;
    case Q4Q5AttnInputScheduleId::TiledGfx906:
        q4_q5_attn_input_tiled_gfx906_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                             stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream);
}

bool q4_q5_attn_input_admits_shard(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 3072 && problem.kv_rows == 512 &&
           problem.padded_k == 5120 && problem.cols >= 1;
}

void q4_q5_attn_input_dispatch_shard(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    if (!q4_q5_attn_input_admits_shard(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input column-parallel: exact shard problem is not admitted");
    }
#if defined(NINFER_GFX906_COMPAT)
    // The grouped-MMA kernel traps on gfx906; the runtime-dimensioned wave64 tiled GEMM serves
    // the shard at every T (no T constraint, row counts read from the tensors).
    q4_q5_attn_input_tiled_gfx906_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                         stream);
#else
    q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                   k, v, stream);
#endif
}

} // namespace ninfer::ops::detail
