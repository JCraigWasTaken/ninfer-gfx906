#pragma once

// Stage-8 route gate for the gfx906 wave64 tiled GEMM kernels.
//
// NINFER_GFX906_STAGE8=0 in the environment restores the stage-3 SIMT
// fallback routing (the A/B lever for the Tier-1/Tier-2 benchmarks); any
// other value, or an unset variable, keeps the tiled kernels on.

#include <cstdlib>

namespace ninfer::ops::detail {

inline bool gfx906_stage8_tiled_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GFX906_STAGE8");
        return !(value != nullptr && value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// Pass-2 route gate for the gfx906 register-resident T=1 GEMV kernels
// (q_gemv_gfx906.cuh). NINFER_GFX906_PASS2=0 restores the upstream
// q5_rowsplit_gemv route (the A/B lever); unset or any other value keeps the
// pass-2 kernels on. Separate from NINFER_GFX906_STAGE8 so the two retunes can
// be A/B'd independently.
inline bool gfx906_pass2_gemv_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GFX906_PASS2");
        return !(value != nullptr && value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

} // namespace ninfer::ops::detail
