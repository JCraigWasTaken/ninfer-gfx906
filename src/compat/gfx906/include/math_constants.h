#pragma once
// gfx906 port: the CUDA math-constants ninfer uses.

#define CUDART_INF_F __builtin_huge_valf()
#define CUDART_INF __builtin_huge_val()
#define CUDART_NAN_F __builtin_nanf("")
#define CUDART_NAN __builtin_nan("")
