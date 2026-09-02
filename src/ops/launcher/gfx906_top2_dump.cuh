#pragma once
// gfx906 pass-2f debug probe (stage 10), shared by the sample and argmax launchers. Off by default.
// NINFER_GFX906_DUMP_TOP2=1 synchronizes the stream after every sample/argmax launch and prints, per
// column, the top-2 ids/logits (bf16 as stored) and the logits of the comma-separated
// NINFER_GFX906_WATCH_IDS to stderr. Skips launches recorded under stream capture (run the CLI with
// --no-cuda-graph to observe decode). Never set in benchmarks: the sync serializes decode.
#include "core/device.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer::ops::detail {

inline void gfx906_dump_top2(const char* tag, const Tensor& logits, const Tensor& out,
                             std::int32_t token_domain, cudaStream_t stream) {
    static const char* enabled = std::getenv("NINFER_GFX906_DUMP_TOP2");
    if (enabled == nullptr || enabled[0] == '0') { return; }
    static const std::vector<std::int32_t> watch = [] {
        std::vector<std::int32_t> ids;
        const char* spec = std::getenv("NINFER_GFX906_WATCH_IDS");
        if (spec == nullptr) { return ids; }
        std::string token;
        for (const char* c = spec;; ++c) {
            if (*c == ',' || *c == '\0') {
                if (!token.empty()) { ids.push_back(std::atoi(token.c_str())); }
                token.clear();
                if (*c == '\0') { break; }
            } else {
                token.push_back(*c);
            }
        }
        return ids;
    }();
    hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capturing) != hipSuccess ||
        capturing != hipStreamCaptureStatusNone) {
        return; // graph-captured launches cannot be observed here; run with --no-cuda-graph
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::int32_t rows          = logits.ne[0];
    const std::int32_t cols          = logits.ne[1] * logits.ne[2] * logits.ne[3];
    const std::int32_t sampled_count = out.ne[0] * out.ne[1] * out.ne[2] * out.ne[3];
    if (sampled_count < cols) {
        std::fprintf(stderr, "TOP2 %s: out has %d < %d columns, skipped\n", tag, sampled_count,
                     cols);
        return;
    }
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * cols);
    std::vector<std::int32_t> sampled(cols);
    CUDA_CHECK(cudaMemcpy(host.data(), logits.data, host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sampled.data(), out.data, sampled.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    auto to_float = [](std::uint16_t bits) {
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    };
    for (std::int32_t b = 0; b < cols; ++b) {
        const std::uint16_t* column = host.data() + static_cast<std::size_t>(b) * rows;
        std::int32_t top1 = -1, top2 = -1;
        float v1 = -INFINITY, v2 = -INFINITY;
        for (std::int32_t r = 0; r < token_domain; ++r) {
            const float v = to_float(column[r]);
            if (v > v1) {
                top2 = top1;
                v2   = v1;
                top1 = r;
                v1   = v;
            } else if (v > v2) {
                top2 = r;
                v2   = v;
            }
        }
        std::fprintf(stderr,
                     "TOP2 %s cols=%d col=%d sampled=%d top1=%d %.4f top2=%d %.4f margin=%.4f", tag,
                     cols, b, sampled[b], top1, v1, top2, v2, v1 - v2);
        for (const std::int32_t id : watch) {
            if (id >= 0 && id < token_domain) {
                std::fprintf(stderr, " id%d=%.4f", id, to_float(column[id]));
            }
        }
        std::fputc('\n', stderr);
    }
}

} // namespace ninfer::ops::detail
