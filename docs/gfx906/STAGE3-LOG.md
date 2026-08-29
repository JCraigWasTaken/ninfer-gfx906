# gfx906 port — Stage 3 log (engine/serve/apps + decode-safe dispatch)

Status: **complete** (2026-08-29). Stage 3 of the port order in
[PORT-AUDIT.md](PORT-AUDIT.md): full-tree compile (engine, serve, both apps)
and rerouting of every linear dispatch table so no reachable schedule maps to
a trapping mma stub, plus the GDN recurrent fallback, the RMSNorm convention
audit, and a stage-4 attention decode kernel DRAFT (stretch, unvalidated).

## Milestone (positive marker)

The **entire build** — `ninfer_core`, `ninfer_artifact`, `ninfer_ops`,
`ninfer_text`, `ninfer_media_decode`, `ninfer_media_acquire`,
`ninfer_product_*`, `ninfer_engine`, `ninfer_serve`, and both executables
(`ninfer`, `ninfer-serve`) — compiles and links clean for gfx906 from a wiped
build volume inside `ninfer-rocm:6.4.1` (ROCm 6.4.1, hip-clang 19):

```
[219/219] Linking HIP executable apps/ninfer-serve
=== ninja exit: 0 ===   error-lines: 0
```

Verification evidence recorded on the from-scratch rebuild after the final
commit (`.build-log-final.txt`, untracked): all five archives plus the two
app binaries present; `strings libninfer_ops.a` shows **1056** instantiations
of the new `simt_partial_bf16` attention kernel and **0** remaining
`tc_partial_bf16` (tensor-core) instantiations.

## Compile status per target

| Target | Status | Notes |
|---|---|---|
| ninfer_core / ninfer_artifact / ninfer_ops | clean (stage 1) | unchanged milestone |
| ninfer_text, ninfer_media_decode | clean | media stub build (stage 1) |
| ninfer_media_acquire | clean | **new stub**: `acquire_stub.cpp` selected when `NINFER_ENABLE_MEDIA=OFF` (container has no libcurl; apps force the library on). `acquire_bytes` throws a configuration error; text-only prompts never reach it |
| ninfer_product_prompt_input / load_progress | clean | no changes needed |
| ninfer_engine (targets/registry + qwen3_6* runtime) | clean | **zero source changes needed** — the stage-1 host sweep already covered it |
| ninfer_serve | clean | `serve/request_log.cpp` `props.uuid` compiles under HIP's `hipDeviceProp_t::uuid` — the stage-1 mapping is now verified |
| apps: ninfer (CLI), ninfer-serve | clean, link to 18/19 MB HIP executables | |

Only pre-existing warnings remain (`amdgpu-waves-per-eu` occupancy notes on
the int8 attention kernel — a stage-10 tuning item).

## Dispatch rerouting (decode-safe paths)

Design: under `NINFER_GFX906_COMPAT` every selector keeps its original
(n,k,t) **shape whitelist** as the validation gate, but the selected kernel
is forced onto a TU-local **reachability registry** of decode/GEMV/SIMT
kernels; anything else (mma tiles, split-K mma, small-T mma) reroutes to a
shape-generic safe kernel or a token-sliced walk over a safe small-T kernel.
The registries are `constexpr` arrays / single-route tables in each dispatch
TU — a unit test can iterate shapes and assert `select(...)` membership.

### Plain linear tables (registry arrays in each `*_dispatch.cpp`)

| Table | gfx906-reachable kernels | T>tuned-range fallback |
|---|---|---|
| `q4_dispatch.cpp` | `gemv_r1_w8_direct`, `gemv_r4_w1_direct` (t==1 rows), `simt_r8_c4/c8` | SIMT c4 (t≤4) / c8 |
| `q5_dispatch.cpp` | `gemv_r16_s2_x` (t==1), `simt_split2/4_exact` (tuned small-T rows), `simt_r8_c4/c8` | SIMT c4/c8 |
| `q6_dispatch.cpp` | `simt_r8_c4/c5/c6/c7/c8` | SIMT c4/c8 |
| `w8_dispatch.cpp` | `decode_r4` ([2048,16384] t==1), `simt_r8_c4/c8` | SIMT c4/c8 |
| `bf16_dispatch.cpp` | `decode` (t==1), `small_t` (t∈[2,32]), `sliced_small_t` (t>32, 32-token slices) | sliced small-T |

### Fused projection plans (compat route tables in each `*_plan.cpp`)

| Plan | 27B groupwise? | gfx906 routes |
|---|---|---|
| `q4_q5_attn_input` | **yes** (GQA layers, Q4 qk + Q5 gate/value [7168,5120]) | `ParentSplitFixed` for all T; T>16 walked in 16-token slices (kernel domain [1,16]) |
| `q4_q5_gdn_input` | **yes** (GDN prefill, Q4 [4096,5120] + Q5 [12288,5120]) | `IndependentDirectFixed` for all T; 16-token slices |
| GDN conv snapshot/record | **yes** (decode/verify) | `ProjectionEpilogueFused` (widths 1,2,3,5,6 — gemv/simt kernels, safe) or `Materialized` → routes through the compat `q4_q5_gdn_input` plan above |
| `q5_linear_add` | **yes** (o-proj [5120,6144], MLP down [5120,17408]) | `GemvResidual` (T==1), `Split2ExactResidual` in 8-token slices for all T>1 |
| `q4_linear_swiglu` | **yes** (MLP [34816,5120]) | `GemvPair` (T==1), `Materialized` for all T>1 (= `ops::linear` through the compat q4 SIMT dispatch + `silu_mul`; `SmallTExact` rides the q4 small-T **mma** kernel and is NOT safe) |
| `bf16_gdn_gating` 27B geometry | **yes** (all regimes, [48,5120]×2 BF16) | `GemvPairedRows` (T==1), `SmallTSplit10` for all T≥2 — its grid already tiles tokens (`grid.z = div_up(t, kSmallTMax)`), so no slicing needed |
| `w8_pair` k=5120 | **yes** (MTP prefill K/V [1024,5120]) | `TwoSimtR8C4/C8` for all T |
| `w8_dispatch` via `ops::linear` | **yes** (all MTP W8 projections: [5120,10240], [14336,5120], [6144,5120], [5120,6144], [34816,5120], [5120,17408]) | SIMT (table above) |
| `q6_dispatch` via `ops::linear` | **yes** (lm_head + tied embedding head [248320,5120]) | SIMT (table above) |
| `w8_attn_input` | no (35B) | `DecodeR8Direct` (T==1), `SimtR8C4` all T>1 |
| `w8_gdn_input` | no (35B) | decode kernel; T>1 one token per launch |
| `w8_linear_add` | no (35B) | `DecodeR16`/`SimtR8C4` (T==1), `SimtR8C4` all T>1 |
| `w8_linear_swiglu` | no (35B) | `DecodePairR16`; T>1 one token per launch |
| `w8_pair` k=2048 | no (35B dflash) | `DualDecodeR4` (T==1), `TwoSimt` T>1 |
| `bf16_attn_input` / `bf16_linear_add` | no (NVFP4-profile-only) | decode + small-T family in 32-token slices |
| `bf16_gdn_gating` 35B geometry | no | `SimtWarpRowC4/C8`; the fused 35B norm+gating mma route is disabled (always `Composed`) |

### GDN core

`gated_delta_net.cpp`: under compat the chunked entry point forces
`T_full = 0`, so **every** token count (decode, verify, prefill) runs the
sequential SIMT `recurrent` kernel (grid over heads, tokens looped in-kernel
— no T limit); the workspace helper mirrors the rule so no chunked scratch is
reserved. The trapping WY-representation chunked kernels are unreachable.

### Reachability guarantee (27B groupwise-int profile)

The 27B groupwise op surface was mapped call-site-by-call-site
(`qwen3_6_27b/impl/variant.cpp` + `load/bindings.cpp:216-267`; token regimes:
graph decode T=B∈[1,8], speculative verify T=W·B∈[2,48], prefill T=chunk
≤ `prefill_chunk`). With the routes above:

- **Every linear-family and GDN dispatch is mma-free for all T** — decode,
  speculative verify, and prefill. The dflash module is unreachable on the
  27B (`DFlashConfig::supported = false`), and `bf16_dispatch` is never
  entered from the groupwise text path (its two shapes are NVFP4-profile
  bindings) — both are patched anyway.
- **GQA attention decode/verify (bf16 KV, T∈[1,6])** now runs the stage-4
  SIMT draft kernel (below) — compiled, UNVALIDATED.
- **Still trapping** (documented, expected at this stage): GQA **prefill**
  attention (`gqa_attention_prefill*`, port-order step 4's second half), the
  **int8-KV** decode partial kernel (step 7), `bidirectional_gqa_attention` /
  `vision_attention` (step 11), and the 35B gating `MmaUnsplit` residual
  route beyond 524 280 tokens (unreachable: real chunks are orders of
  magnitude smaller). So end-to-end prefill first-token requires the stage-4
  prefill kernel; everything below the attention ops is already safe.

## RMSNorm convention audit (L1T finding)

**Verdict: the Gemma-style (1+weight) convention is applied at RUNTIME in the
kernel epilogue; stored norm weights are raw HF-checkpoint values. Nothing is
baked into the artifact.** Evidence:

- Kernel: `src/ops/kernel/rmsnorm.cuh:22` —
  `if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }`;
  contract in `include/ninfer/ops/rmsnorm.h:13` —
  `gain[d] = unit_offset ? 1 + weight[d] : weight[d]`.
- Routing: `src/ops/launcher/rmsnorm.cu:91-95` — `unit_offset` selects
  `RmsEpilogue::Offset` vs `Plain`; a gated call selects `Gated` (no +1).
- 27B call sites pass `unit_offset=true` for **every** non-gated norm:
  input/q/k/post/final norms (`text_context_impl.h:801,817,818,962,668,726,
  1168`) and all MTP norms (`:348,349,358,382,383,414,422,488,514,541,546`);
  the GDN pre-gating norm likewise (`bf16_gdn_gating_proj_plan.cpp:482`,
  Composed route).
- The **gated** GDN output norm uses plain weight: `ops::gated_rmsnorm`
  (`text_context_impl.h:953`) → `src/ops/wrapper/rmsnorm.cpp:98` passes
  `unit_offset=false` with the z-gate → `RmsEpilogue::Gated` multiplies
  `x*inv*weight*silu(z)` with **no** +1.
- Storage: the converter copies norm weights verbatim — no arithmetic
  transform anywhere in `tools/convert/` (e.g.
  `tools/convert/qwen3_6_27b/recipe.py:65-66` input_norm ← 
  `input_layernorm.weight`, `:97-102` q/k norm, `:162-163` gdn/norm ←
  `linear_attn.norm.weight`, `:198-199` final_norm). Artifact binding is a
  BF16 control-tensor copy (`bind_groupwise_text_layers`).
- Stage-1 impact: **none of the stage-1 shims touched norm code** — `git log`
  shows no gfx906 commit on `rmsnorm.cuh/.cu/.cpp`; the shimmed helpers they
  use (`warp.cuh` logical-32-lane shuffles, bf16 conversions) preserve CUDA
  semantics, and the `+1.0f` epilogue is scalar FP32 math with no arch
  dependence. Translated/rewritten norm kernels (and the stage-4 fused-norm
  work, if any) must keep this exact split: **+1 at runtime for `Offset`
  norms, raw weight for the gated GDN output norm.**

## Stage-4 attention decode kernel (stretch — DRAFT, UNVALIDATED)

`src/ops/kernel/gqa_attention_decode_bf16_gfx906.cuh`:
`gqa_attention_small_t_simt_partial_bf16_kernel` — a wave64 SIMT drop-in for
the tensor-core bf16-KV split-KV partial kernel (T∈[1,6], head_dim 256,
paged KV, append + cached inputs, masked + multi-batch forms). Same template
surface, grid geometry, and `partial_acc/m/l` contract; the existing mma-free
split reducer is reused unchanged. Structure follows llama.cpp-gfx906
flash-decode (fattn-q8.cuh / gfx906-common.cuh): one logical 32-lane warp
per (q_head, token) row, one key per lane with a serial bf16x2 dot product,
warp-shuffle online softmax (`exp2_approx`, same Log2E basis), and
probability-broadcast coalesced P·V. Wired in
`src/ops/launcher/gqa_attention_decode.cu` under the compat macro.

**COMPILES (1056 instantiations in `libninfer_ops.a`; 0 tensor-core bf16
partials remain) but has never executed — correctness is untestable until an
MI50 is available. Marked UNVALIDATED in the source.** DPP-ladder reductions,
half-wave pairing, and K-reuse across the q-head group are deliberately left
to stage-10 tuning; the draft optimizes for a verifiable 1:1 semantic mapping
(split arithmetic, append ownership, new-vs-cached source selection, causal
masks, neutral-partial writes all replicated line-for-line).

## Deferred items

- **First hardware validation** of the SIMT attention decode draft (needs
  the MI50 box back up — cold power cycle pending).
- **Prefill attention** (stage 4 proper): `gqa_attention_prefill*` still
  traps; blocks end-to-end first token. The decode-side scaffolding
  (reducer, invocation plumbing) is proven reusable.
- int8-KV attention decode (`v_dot4_i32_i8`, step 7); bidirectional/vision
  attention (step 11).
- All compat token-slicing fallbacks are **correctness routes**: prefill
  through 8/16/32-token slices (and the 35B one-token decode walks) is slow
  by design; the real grouped-int prefill GEMMs are steps 8–9, retuning of
  the whitelists on MI50 timings is step 10.
- Unit test that iterates each dispatch table's shape domain and asserts the
  selected launch is registry-member (registries are in place; the test
  harness hookup was not part of this stage).
- `.gfx906-build.sh` / `.build-log*.txt` remain untracked build-loop helpers.
