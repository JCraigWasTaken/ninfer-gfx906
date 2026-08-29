// libcurl-free stand-in for product/media_acquire/acquire.cpp, selected by
// -DNINFER_ENABLE_MEDIA=OFF (same pattern as media/decode/decode_stub.cpp).
// Text-only prompts never reach this entry point; image/video message parts
// report a clear configuration error instead of failing at link time.

#include "product/media_acquire/acquire.h"

#include <stdexcept>

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::runtime_error(
        "media acquisition is disabled in this build (NINFER_ENABLE_MEDIA=OFF)");
}

} // namespace ninfer::product::media_acquire
