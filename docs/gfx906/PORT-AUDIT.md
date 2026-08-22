# ninfer → AMD gfx906 (Instinct MI50/MI60) — Port Audit

Status: **audit complete, port in progress** (2026-08-22). This document is the
engineering map produced before writing code: a full-codebase kernel census, a
diff study of the existing NVIDIA down-ports, and a catalog of donor kernels
from the gfx906 llama.cpp community fork.

Target: Qwen3.8-27B (groupwise-int artifact) on a single 32GB MI50 (gfx906 /
Vega20: HIP/ROCm, wave64 GCN, no matrix cores, no FP8/NVFP4, `v_dot4_i32_i8` +
fast packed-FP16, ~1TB/s HBM2).

## Why this is feasible

1. **Qwen3.8-27B needs no new target.** It is a weights-profile of the
   `qwen3_6_27b` target (`src/targets/registry.cpp` routes it; shared
   `TextConfig`: hidden 5120, 64 layers = 48 GatedDeltaNet + 16 GQA with
   head-dim 256, 1 MTP layer).
2. **The shipped groupwise-int artifact runs on gfx906 as-is.** The
   `qwen3_8_27b.ninfer` (groupwise-int profile) contains only
   Q4G64/Q5G64/W8G32/Q6/BF16/FP32/I32 tensors — zero FP8/NVFP4. ~19–20GB of
   weights fits a 32GB card with generous KV. Converters are pure Python/CPU.
3. **All tensor-core PTX funnels through one choke point**
   (`src/ops/common/mma.cuh`, ~105 LOC) plus the rowsplit tile loaders. The
   `fp8/`, `nvfp4/`, and `sparse_moe/` op families (~100 files) are *deleted*
   from the build (the 27B is dense) using the pattern proven by the
   ninfer-3090 fork: CMake source-list filtering + throwing stubs.
4. **No worst-case wave64 landmines**: zero uses of
   `__ballot/__activemask/__any_sync/__all_sync`. Shuffles route through
   central helpers (`src/ops/common/warp.cuh`) — one shim keeps logical
   32-lane subgroups (HIP `__shfl(x, off, 32)` subdivides a 64-lane wave).

## Key technical fact

ninfer's int-quant linear kernels are **weight-only int with BF16
activations** (dequantize → `mma_bf16`), not int8-dot. The gfx906 rewrite is
therefore dequant → packed-FP16 FMA (`v_pk_fma_f16`); gfx906 has no bf16 VALU,
so rewritten kernels should stage in fp16 (bf16x2 intrinsics compile via
float-roundtrip emulation — correct but slow). `v_dot4_i32_i8` is needed only
for the int8-KV attention QK path.

## Kernel census (`src/ops`: 224 translation units, ~36.3k LOC)

| Class | Files | Notes |
|---|---:|---|
| MECHANICAL (hipify-clean) | 150 | plain CUDA C++, shuffles via helpers, shared mem, half2/bf16x2 vector math |
| ADAPT | 14 | `cp.async`/PDL only — includes the **decode workhorses** (`q4/q5_rowsplit_gemv`, `*_simt`, `w8_k2048_decode`) |
| REWRITE in scope | **~24 (~9–10k LOC)** | GQA attention family (~2.5k), GDN chunked prefill (~1.7k), grouped-int/bf16 prefill GEMMs (~4k), misc (~1k) |
| DELETED | ~100 | `fp8/`, `nvfp4/`, `sparse_moe/` |

Infrastructure: `decode_graph.cpp` is pure stream-capture (maps 1:1 to
hipGraph — graphs are kept); `pdl.cuh` (Programmatic Dependent Launch, no HIP
equivalent) becomes no-op stubs — correctness-safe, PDL is an overlap
optimization; `cub` → `hipcub`; no cuBLAS/cuDNN/VMM/mempools anywhere.

## Donor kernels (llama.cpp gfx906 community fork)

The iacopPBK `llama.cpp-gfx906` fork (~4.4k LOC of hand-written gfx906
kernels) supplies working patterns for every rewrite family:

- **Flash-attention Q8 tiles with head-dim-256 configs** — exact match for
  ninfer's GQA (`kHeadDim = 256`)
- **Wave64 GEMV**: two 32-lane half-waves per wave, `__builtin_amdgcn_sdot4`,
  DPP reductions (~120 LOC per quant format)
- **`gfx906-common.cuh`** (~308 LOC): DPP reduction ladder, SGPR broadcast,
  `v_exp/v_rcp` asm, `v_dot2_f32_f16` — liftable nearly verbatim
- Fused RMSNorm+mul+quantize epilogues, RoPE, Q8 activation cache
- Build recipe: ROCm clang, `CMAKE_HIP_ARCHITECTURES=gfx906`,
  `GGML_HIP_NO_VMM=ON` (required on MI50), HIP graphs on, ROCm 7.1.x-tested

## Precedent (fork lineage study)

- **ninfer-3090** (sm_86): the structural playbook — arch gating, source
  filtering, `NINFER_SM8X_COMPAT` macro, PDL downgrade, schedule retunes,
  runtime SM-count query with split-K fallback ladders. ~3–4 weeks calendar,
  one author + four community contributors.
- **ninfer-cmp170hx** (sm_80): the 3090 fork plus three build-flag changes and
  two runtime fixes — a same-ISA retarget in days, agent-driven, by a
  self-described non-coder. Proof of method, not of timeline.
- **This port is strictly harder than both**: no fork has ever rewritten the
  kernel math (NVIDIA `mma.sync` int8/bf16 carried over unchanged down to
  sm_80). gfx906 has no matrix cores — the ~24-file rewrite layer is new
  territory, with the llama.cpp-gfx906 kernels as donors. Estimate: **several
  weeks**, agent-assisted.

## Port order (first token earliest)

1. Toolchain: HIP CMake (`gfx906`), hipify host code, `pdl.cuh`/`memory.cuh`/
   `warp.cuh` shims, remove the sm_120a CMake pin + the `sm() != 120` runtime
   gate (`layouts_impl.h`), drop `ninfer_nvfp4_tma`, filter fp8/nvfp4/moe.
2. Elementwise ops + sampling (hipcub) + embed_gather.
3. **Decode-shape linears only**: route dispatch tables to the simt/gemv/
   decode variants; GDN via the existing SIMT `recurrent` path for both decode
   and prefill (slow prefill, correct output).
4. One wave64 GQA decode kernel (bf16 KV, T=1) + a simple prefill attention →
   **first end-to-end token** (`--no-cuda-graph`, greedy, bf16 KV).
5. hipGraphs on → 6. MTP speculative (ops are mechanical) → 7. int8-KV
   attention (`v_dot4_i32_i8`) → 8. grouped-int prefill GEMM rewrites →
   9. GDN chunked rewrite → 10. dispatch retuning on real MI50 timings (the
   (n,k,t) whitelists are RTX 5090-tuned) → 11. vision.

## Hardest five

1. GQA attention family (paged KV, int8-KV fused append/quant,
   producer–consumer warp choreography) — full VALU rewrite.
2. GDN chunked prefill (WY-representation delta rule) — mitigated by the
   `recurrent` fallback.
3. Grouped-int prefill GEMMs + per-shape dispatch retuning.
4. The wave64 / no-bf16-VALU long tail (launch bounds, block geometry,
   LDS layouts designed for `ldmatrix`).
5. Recovering the overlap that PDL/`cp.async` provided, as tuning work.

## Expected results (from the CMP fork's measurements + gfx906 headroom)

Dense 27B decode +25–50% over llama.cpp-gfx906; prefill +15–30% (no matrix
cores = a permanent prefill handicap); MoE-per-card targets ~2× class if
added later. For reference the CMP170HX port measured ~2× llama.cpp on a
35B-A3B MoE (218 tok/s generation).

## Multi-GPU

ninfer is single-GPU per instance. One fork adds vocab-matrix offload to a
second GPU via pinned-host bounce buffers (explicitly no peer access — works
over narrow PCIe links); cherry-pickable after single-card bring-up. No true
tensor parallelism exists in the ninfer ecosystem yet.

## Credits

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the engine
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) — the
  down-port playbook this fork follows
- Ithrial/ninfer-cmp170hx — proof that agent-driven ports of this codebase work
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)
  and the gfx906 community — the donor kernel library
