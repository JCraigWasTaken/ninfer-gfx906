#pragma once
// gfx906 port: the CUDA driver API is used only by the excluded NVFP4 TMA
// path. Nothing is mapped; retained code must not depend on it.
#include "core/hip_compat.h"
