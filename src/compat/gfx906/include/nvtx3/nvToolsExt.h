#pragma once
// gfx906 port: NVTX no-op stub covering the surface core/nvtx.h,
// core/nvtx_range.h, and the runtime instrumentation use. Profiling ranges
// become free no-ops; roctracer integration can replace this later.

#include <cstdint>

#define NVTX_VERSION 3
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE (static_cast<std::uint16_t>(sizeof(nvtxEventAttributes_t)))
#define NVTX_COLOR_ARGB 1
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 1
#define NVTX_MESSAGE_TYPE_REGISTERED 3

typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;

typedef union nvtxMessageValue_t {
    const char* ascii;
    const wchar_t* unicode;
    nvtxStringHandle_t registered;
} nvtxMessageValue_t;

typedef union nvtxEventPayload_t {
    std::uint64_t ullValue;
    std::int64_t llValue;
    double dValue;
} nvtxEventPayload_t;

typedef struct nvtxEventAttributes_t {
    std::uint16_t version;
    std::uint16_t size;
    std::uint32_t category;
    std::int32_t colorType;
    std::uint32_t color;
    std::int32_t payloadType;
    std::int32_t reserved0;
    nvtxEventPayload_t payload;
    std::int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_t;

inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePop() { return 0; }
inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) { return nullptr; }
inline void nvtxDomainNameCategoryA(nvtxDomainHandle_t, std::uint32_t, const char*) {}
inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char*) {
    return nullptr;
}
inline int nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) { return 0; }
inline int nvtxDomainRangePop(nvtxDomainHandle_t) { return 0; }
