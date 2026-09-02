#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include "ops/linear_add/q5/q5_linear_add_kernels.h"

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

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q5LinearAddScheduleId schedule;
};

constexpr std::array<SupportSpec, 2> kSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

#if defined(NINFER_GFX906_COMPAT)
// gfx906 stage-3 rerouting: the residual mma kernels trap on gfx906. Only the
// GEMV (T=1) and split2-exact SIMT (small-T) kernels are reachable; token
// counts beyond the split2 domain are walked in 8-token slices at execute
// time (8 is inside both k-shapes' tuned split2 ranges). Slow prefill,
// correct output.
constexpr std::array<RouteSpec, 2> kK6144Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, kAnyCols}, Q5LinearAddScheduleId::Split2ExactResidual},
}};

constexpr std::array<RouteSpec, 2> kK17408Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, kAnyCols}, Q5LinearAddScheduleId::Split2ExactResidual},
}};

// Token-slice width inside the split2-exact launcher's instantiated domain.
constexpr std::int32_t kGfx906Split2Slice = 8;

// Stage-8 routing: the wave64 LDS-tiled GEMM with fused residual accumulate
// replaces the sliced split2 walk for every T > 1 (both k shapes are
// stage-aligned). Selected at runtime by the NINFER_GFX906_STAGE8 gate.
constexpr std::array<RouteSpec, 2> kK6144RoutesTiled{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, kAnyCols}, Q5LinearAddScheduleId::TiledResidualGfx906},
}};

constexpr std::array<RouteSpec, 2> kK17408RoutesTiled{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, kAnyCols}, Q5LinearAddScheduleId::TiledResidualGfx906},
}};
#else
constexpr std::array<RouteSpec, 6> kK6144Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 13}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{14, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

constexpr std::array<RouteSpec, 6> kK17408Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{17, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};
#endif // NINFER_GFX906_COMPAT

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kK6144Routes) && catalog_is_closed(kK17408Routes),
              "Q5 LinearAdd routes must be exact, contiguous, and closed");
#if defined(NINFER_GFX906_COMPAT)
static_assert(catalog_is_closed(kK6144RoutesTiled) && catalog_is_closed(kK17408RoutesTiled),
              "Q5 LinearAdd gfx906 tiled routes must be exact, contiguous, and closed");
#endif

bool supported_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

} // namespace

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        return "linear_add.q5.gemv.residual";
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return "linear_add.q5.simt.split2.exact.residual";
    case Q5LinearAddScheduleId::TiledResidualGfx906:
        return "linear_add.q5.tiled.gfx906.residual";
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        return "linear_add.q5.mma.r64.c16.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        return "linear_add.q5.mma.r64.c24.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q5.mma.r64.c64.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual";
    }
    return "linear_add.q5.unknown";
}

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }

    const auto resolve_from = [&](const auto& routes) -> Q5LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (route.cols.contains(problem.cols)) { return {route.schedule, 0}; }
        }
        throw std::logic_error("q5 linear_add: admitted problem has no covering route");
    };
#if defined(NINFER_GFX906_COMPAT)
    if (gfx906_stage8_tiled_enabled()) {
        return problem.k == 6144 ? resolve_from(kK6144RoutesTiled)
                                 : resolve_from(kK17408RoutesTiled);
    }
#endif
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK17408Routes);
}

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q5 linear_add: invalid column interval");
    }
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, min_cols});
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, max_cols});

    return 0;
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }
    (void)ws;

    switch (plan.schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        q5_linear_add_gemv_residual_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::Split2ExactResidual:
#if defined(NINFER_GFX906_COMPAT)
        // Walk arbitrary token counts through the split2-exact SIMT kernel in
        // 8-token slices (a leftover single token uses the GEMV kernel).
        for (std::int32_t offset = 0; offset < problem.cols; offset += kGfx906Split2Slice) {
            const std::int32_t count =
                std::min<std::int32_t>(kGfx906Split2Slice, problem.cols - offset);
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            if (count == 1) {
                q5_linear_add_gemv_residual_launch(x_slice, w, residual_slice, stream);
            } else {
                q5_linear_add_split2_exact_launch(x_slice, w, residual_slice, stream);
            }
        }
#else
        q5_linear_add_split2_exact_launch(x, w, residual_out, stream);
#endif
        return;
    case Q5LinearAddScheduleId::TiledResidualGfx906:
#if defined(NINFER_GFX906_COMPAT)
        // Pass 2e: T=2..5 (the MTP verify widths) take the register-resident
        // small-T GEMV; NINFER_GFX906_PASS2_SMALLT=0 keeps the tiled GEMM.
        if (gfx906_pass2_smallt_enabled() &&
            q5_linear_add_gemv_smallt_gfx906_launch(x, w, residual_out, stream)) {
            return;
        }
#endif
        q5_linear_add_tiled_gfx906_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        q5_linear_add_mma_r64_c16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        q5_linear_add_mma_r64_c24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        q5_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        q5_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("q5 linear_add: unknown schedule");
}

void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan plan = q5_linear_add_resolve_plan(problem);
    q5_linear_add_execute_plan(plan, x, w, residual_out, ws, stream);
}

} // namespace ninfer::ops::detail
