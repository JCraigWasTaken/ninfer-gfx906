// FFmpeg-free stand-in for media/decode/decode.cpp, selected by
// -DNINFER_ENABLE_MEDIA=OFF. Text-only inference never reaches these entry
// points; vision inputs report a clear configuration error.

#include "media/decode/decode.h"

#include <stdexcept>

namespace ninfer::media::decode {

namespace {

[[noreturn]] void reject_media() {
    throw std::runtime_error(
        "media decode is disabled in this build (NINFER_ENABLE_MEDIA=OFF)");
}

} // namespace

Image decode_image(std::span<const std::uint8_t>, const Policy&) { reject_media(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    reject_media();
}

} // namespace ninfer::media::decode
