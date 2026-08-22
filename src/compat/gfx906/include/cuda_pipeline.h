#pragma once
// gfx906 port: cp.async / the CUDA pipeline API do not exist on gfx906.
// ops/common/memory.cuh replaces every pipeline primitive with synchronous
// loads under NINFER_GFX906_COMPAT; nothing is provided here.
#include "core/hip_compat.h"
