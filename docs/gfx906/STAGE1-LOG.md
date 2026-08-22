# gfx906 port — Stage 1 log (toolchain + mechanical sweep)

Status: **complete** (2026-08-22). Stages 1 and 2 of the port order in
[PORT-AUDIT.md](PORT-AUDIT.md): HIP toolchain, compat shims, family deletion,
and the mechanical compile sweep — including elementwise ops, sampling
(cub→hipcub), and embed_gather (FP8 branch stripped), which all live in
`ninfer_ops` and compiled as part of the sweep.

## Milestone (positive marker)

`ninfer_core`, `ninfer_artifact`, and `ninfer_ops` compile clean for gfx906
from a wiped build directory inside `rocm/dev-ubuntu-22.04:6.4.1`
(ROCm 6.4.1, hip-clang 19). Verification-run evidence:

```
[164/167] Building CXX object src/CMakeFiles/ninfer_ops.dir/ops/wrapper/sparse_moe.cpp.o
[165/167] Building CXX object src/CMakeFiles/ninfer_ops.dir/ops/wrapper/vision_attention.cpp.o
[166/167] Building CXX object src/CMakeFiles/ninfer_ops.dir/ops/gfx906_stubs.cpp.o
[167/167] Linking HIP static library src/libninfer_ops.a
=== ninja exit: 0  error-lines: 0 ===
-rw-r--r-- 1 root root   472886 Aug 22 22:27 /build/gfx906/src/libninfer_artifact.a
-rw-r--r-- 1 root root   284872 Aug 22 22:27 /build/gfx906/src/libninfer_core.a
-rw-r--r-- 1 root root 21121532 Aug 22 22:28 /build/gfx906/src/libninfer_ops.a
$ strings libninfer_ops.a | grep -c gfx906   # embedded amdgcn ISA marker
178
```

Compile-error trajectory across sweep iterations: **281 → 123 → 53 → 13 → 0**.

## Build recipe

```sh
cmake -B build-gfx906 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DNINFER_ENABLE_MEDIA=OFF -DNINFER_BUILD_APPS=OFF
cmake --build build-gfx906 --target ninfer_core ninfer_artifact ninfer_ops
```

- ROCm clang++ as the host CXX compiler is **required**: host translation
  units that include the bf16/fp16 compat shims still parse HIP headers'
  `_Float16`, which Ubuntu 22.04's g++ 11 rejects.
- `NINFER_ENABLE_MEDIA=OFF` skips FFmpeg/libcurl (Ubuntu 22.04 ships
  libavformat 58 < required 60) and builds a throwing media-decode stub;
  vision is deferred anyway (port order step 11).
- `.cu` files compile directly as HIP via the `LANGUAGE HIP` source property
  (llama.cpp-gfx906 pattern); no RDC — the old CUDA separable-compilation
  properties are gone.

## What changed (29 files, ~1070 insertions)

1. **Build**: `CMAKE_HIP_ARCHITECTURES=gfx906` pin replaces the sm_120a/CUDA
   13.1 gates; `ninfer_nvfp4_tma` deleted; `fp8/`, `nvfp4/`, `sparse_moe/`
   filtered out of `ninfer_ops`; tree-wide `NINFER_GFX906_COMPAT=1`.
2. **Compat layer**: `src/core/hip_compat.h` (CUDA→HIP macro map, graph-API
   signature adapters, wave64-safe unmasked `__shfl_*_sync` shims with logical
   32-lane subgroups) + `src/compat/gfx906/include/` shim tree so mechanical
   files keep their literal `#include <cuda_*.h>` lines (bf16/fp16/hipcub/
   cooperative-groups forwards, NVTX no-op stub, `math_constants.h`).
3. **Device shims**: `pdl.cuh` plain-launch downgrade (ninfer-3090 pattern);
   `memory.cuh` synchronous cp.async/pipeline bodies; `mma.cuh` trapping
   ldmatrix/mma stubs; `math.cuh` `v_exp` + manual bf16x2 pack; `bf16_gemv`
   plain pack loads.
4. **Host fixes**: `__HIPCC__` accepted in device guards; cooperative split-K
   launch via `hipLaunchCooperativeKernel`; `sm() != 120` gate now accepts
   HIP's gfx9 (major 9, minor 0) report.
5. **Stubs**: `src/ops/gfx906_stubs.cpp` — throwing entry points for all
   deleted families; `src/media/decode/decode_stub.cpp` for media-off builds.

## Stub inventory (throw on gfx906)

- `ops/gfx906_stubs.cpp` (~80 functions, `ninfer::ops::detail`): fp8 linear /
  attn_input / gdn_input / gdn conv-fused / linear_add / linear_swiglu
  families; nvfp4 equivalents + w4a4 + all five TMA launchers; sparse-MoE
  decode / prefill / small_t plans and launches (`sparse_moe_uses_prefill` /
  `_small_t` return false instead of throwing).
- `embed_gather_fp8_launch` throws under the compat macro (kernel compiled
  out).
- Device-side: all `mma.cuh` helpers and two whole kernel bodies
  (`w8_small_t_mma_kernel`, `w8_rowsplit_medium_t_splitk_kernel`, via
  `NINFER_GFX906_UNPORTED_KERNEL()`) trap if executed — they exist only so
  the mechanical TUs link; **dispatch must never route to them on gfx906**.

## Known-deferred items

- **Stage 3**: route dispatch tables to simt/gemv/decode variants so the
  trapping mma kernels are unreachable at runtime; GDN via the SIMT
  `recurrent` path.
- **Stage 4+**: rewrite scope untouched — GQA attention family, GDN chunked
  prefill, grouped-int/bf16 prefill GEMMs (~24 files). They compile against
  the trapping mma stubs today.
- `ninfer_engine` / `ninfer_serve` / apps not yet built (host-code sweep for
  those targets is stage 3 territory; `serve/request_log.cpp` uses
  `props.uuid`, mapped but unverified).
- `cp_wait`/`pipe_wait` are per-thread no-ops (correct: every kernel already
  pairs them with `__syncthreads`); the overlap they bought is a stage-10
  tuning concern.
- `__maxnreg__` is compiled away; gfx906 occupancy retuning is stage 10.
- Local-LDS budget: gfx906 has 64KiB LDS/CU; kernels requesting >64KiB
  dynamic shared memory will fail at *launch* (not compile) — audit when
  bringing each family up.
- Compat `__shfl_*_sync` shims drop the mask argument (all ninfer call sites
  pass full masks; audit confirmed no ballot/activemask use anywhere).
- Media decode disabled in the stage-1 build; re-enable by installing FFmpeg
  ≥ 60 dev packages and dropping `-DNINFER_ENABLE_MEDIA=OFF`.
