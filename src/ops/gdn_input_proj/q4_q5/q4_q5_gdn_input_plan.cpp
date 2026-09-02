#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

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
    Q4Q5GdnInputScheduleId schedule;
};

#if defined(NINFER_GFX906_COMPAT)
// gfx906 stage-3 rerouting: the grouped-mixed mma kernel traps on gfx906. The
// independent launch (q4_rowsplit_gemv/simt kernels, T in [1,16]) is the only
// reachable schedule; larger token counts are walked in 16-token slices at
// execute time. Slow prefill, correct output.
constexpr std::array<RouteSpec, 1> kRoutes{{
    {{1, kAnyCols}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
}};

// Stage-8 routing: one wave64 LDS-tiled GEMM per weight for every T > 1 (the
// Q5 value/z half's split4/simt fallbacks lose to the tiled kernel from T=2
// up, by far more than the Q4 half's small-T fallback gains). Selected at
// runtime by the NINFER_GFX906_STAGE8 gate.
constexpr std::array<RouteSpec, 2> kRoutesTiled{{
    {{1, 1}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{2, kAnyCols}, Q4Q5GdnInputScheduleId::TiledGfx906},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last == kAnyCols &&
           kRoutesTiled[0].cols.first == 1 &&
           kRoutesTiled[0].cols.last + 1 == kRoutesTiled[1].cols.first &&
           kRoutesTiled[1].cols.last == kAnyCols;
}

// Token-slice width matched to the independent launcher's T-domain [1, 16].
constexpr std::int32_t kGfx906IndependentSlice = 16;
#else
constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{17, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#endif // NINFER_GFX906_COMPAT

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

// The tp2 column shard of the same parent -- each device's own head-local half
// (qk_rows=2048=8*256, value_z_rows=6144=8*768, qkv_rows=5120, z_rows=3072). Registered as a
// second exact shape rather than widening supported_shape() to a formula: a shard extent is a
// shape like any other, and a tp1 shape that later happened to equal it must keep its own tuned
// route.
bool supported_shard_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 2048 && problem.value_z_rows == 6144 &&
           problem.qkv_rows == 5120 && problem.z_rows == 3072 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::TiledGfx906:
        return "gdn_input_proj.q4_q5.tiled.gfx906";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused:
        return "gdn_input_proj_conv.q4_q5.projection_epilogue_fused";
    case Q4Q5GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q4_q5.materialized";
    }
    return "gdn_input_proj_conv.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return (supported_shape(problem) || supported_shard_shape(problem)) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    // The shard shape has no tuned small-T exact kernel (q4_q5_gdn_input_independent_launch
    // is compile-time-exact to the tp1 parent's 4096/12288 row counts). Upstream routes the
    // shard through the row-count-generic grouped-MMA kernel; on gfx906 that kernel traps, so
    // the shard rides the runtime-dimensioned wave64 tiled GEMM at every T (stage-8 kernel,
    // no T constraint). tp1 selection below is untouched.
    if (supported_shard_shape(problem)) {
#if defined(NINFER_GFX906_COMPAT)
        return {Q4Q5GdnInputScheduleId::TiledGfx906};
#else
        return {Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128};
#endif
    }

#if defined(NINFER_GFX906_COMPAT)
    if (gfx906_stage8_tiled_enabled()) {
        for (const RouteSpec& route : kRoutesTiled) {
            if (!route.cols.contains(problem.cols)) { continue; }
            return {route.schedule};
        }
        throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
    }
#endif
    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    // TP2 shard (collision 4): the fused projection epilogue bakes the tp1 row counts, so a
    // shard problem always takes the composed/Materialized route, at every T and batch.
    if (problem.qk_rows != 4096) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
    if (batch_size > 1) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
#if defined(NINFER_GFX906_COMPAT)
    // Stage-8: the fused projection epilogue's inner GEMMs are the stage-3
    // fallback kernels, so every multi-token width (MTP verify) now takes
    // the Materialized route and rides the tiled GEMMs; T=1 decode keeps
    // the fused epilogue. This is why draft-2 verify (width 3) was 2.4x
    // slower per round than draft-3 (width 4, already Materialized) before
    // this reroute — see STAGE8-9-LOG.md.
    if (gfx906_stage8_tiled_enabled() && problem.cols > 1) {
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
#endif
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
#if defined(NINFER_GFX906_COMPAT)
        // Walk arbitrary token counts through the [1,16]-token kernels. The
        // dim-0 views are taken per slice; the kernels handle their strided
        // outputs via the leading dimension exactly as in the fused route.
        for (std::int32_t offset = 0; offset < problem.cols; offset += kGfx906IndependentSlice) {
            const std::int32_t count =
                std::min<std::int32_t>(kGfx906IndependentSlice, problem.cols - offset);
            const Tensor x_slice   = x.slice(1, offset, count);
            Tensor qkv_slice       = qkv.slice(1, offset, count);
            Tensor z_slice         = z.slice(1, offset, count);
            Tensor qk_slice        = qkv_slice.slice(0, 0, problem.qk_rows);
            Tensor value_slice     = qkv_slice.slice(0, problem.qk_rows, problem.z_rows);
            q4_q5_gdn_input_independent_launch(x_slice, qk_weight, value_z_weight, qk_slice,
                                               value_slice, z_slice, stream);
        }
#else
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
#endif
        return;
    }
    case Q4Q5GdnInputScheduleId::TiledGfx906: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_tiled_gfx906_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
