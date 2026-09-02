#pragma once

// Gate for the context-lookup MTP verification path (ported from geoffwatts/ninfer-v100
// cd6ed9ce). Upstream ships the path always on with no CLI option; this port keeps it OFF by
// default so the byte-identical donor-parity gate runs on the unchanged five-token MTP round,
// and so the second verify frame, the second GDN replay-record plane, and the second captured
// graph family cost nothing until the path is asked for.
//
// NINFER_GFX906_MTP_LOOKUP=1 turns it on; unset, 0, or any other value keeps it off.
//
// The value is read once per process. Everything downstream -- the persistent layout, the
// workspace plan, the graph allowance, and the decode-time frame selection -- must agree, so it
// must not be re-read after startup.

#include <cstdlib>

namespace ninfer::targets::qwen3_6 {

inline bool mtp_context_lookup_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GFX906_MTP_LOOKUP");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

} // namespace ninfer::targets::qwen3_6
