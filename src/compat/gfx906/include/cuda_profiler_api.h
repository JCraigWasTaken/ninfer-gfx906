#pragma once
// gfx906 port: profiler-control shim. Collection is driven externally
// (rocprof); the in-process start/stop markers become no-ops.
#include <hip/hip_runtime.h>

static inline hipError_t cudaProfilerStart(void) { return hipSuccess; }
static inline hipError_t cudaProfilerStop(void) { return hipSuccess; }
