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

} // namespace ninfer::ops::detail
