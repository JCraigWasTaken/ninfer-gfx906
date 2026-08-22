#pragma once
#include "core/hip_compat.h"

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#else
// Layout-compatible host stand-ins; see cuda_bf16.h.
struct __half {
    unsigned short x;
};
struct __half2 {
    __half x, y;
};
#endif
