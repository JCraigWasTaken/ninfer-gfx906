#pragma once
// gfx906 port: cub -> hipcub.
#include "core/hip_compat.h"

#include <hipcub/block/block_merge_sort.hpp>

namespace cub {
using namespace hipcub;
} // namespace cub
