# gfx906 port — Stage 5–7 log (HIP graphs, MTP speculative, int8-KV attention)

Status: **stages 5 and 6 complete, stage 7 (stretch) landed** (2026-08-30).
Port-order steps 5–7 from [PORT-AUDIT.md](PORT-AUDIT.md), all validated on the
MI50 box (`forgev1`, ROCm 6.4.1 container, single 32GB MI50). Same run recipe
as [STAGE4-LOG.md](STAGE4-LOG.md): `ninfer` CLI, greedy, `--no-thinking`,
`--max-context 2048 --kv-capacity 2048`, exit 0 on every milestone run.

Probes used throughout (stage-4's two milestone prompts, a regenerated long
passage, and two generation workloads):

| # | Prompt | Notes |
|---|---|---|
| p1 | `Say OK` | stage-4 milestone |
| p2 | `What is the capital of France?` | stage-4 milestone |
| p3 | 166-token Amazon-passage + 2-part question | multi-block prefill; stage-4's passage text was not preserved verbatim, so an equivalent probe was written — its bf16 eager answer is correct and cites the passage facts |
| p4 | `Write a short story about a lighthouse keeper.` (`--max-new 128`) | prose decode workload |
| p5 | `Write a Python function that reverses a linked list.` (`--max-new 200`) | code decode workload |

## Stage 5 — HIP graphs on (drop `--no-cuda-graph`)

**Milestone: all five probes produce bit-identical greedy token ids with
graphs ON vs OFF** (`tokens generated ids` lines compared verbatim; stdout
identical), first execution of `decode_graph.cpp`'s hipified
capture/instantiate/launch path.

One bug, host-side, no kernel or capture failures:

1. **Graph allowance gate tripped** — `CUDA Graph preparation consumed
   37748736 bytes, exceeding the planned allowance of 12582912 bytes` →
   ROCm's hipGraph capture/instantiation materializes several times the
   device memory the CUDA-tuned empirical tiers in `layouts_impl.h` plan for
   (measured: 36 MiB vs the 12 MiB ordinary tier; later 51 MiB vs the 12 MiB
   MTP tier). Fixed by scaling every tier **8x** under
   `NINFER_GFX906_COMPAT` (first landed as 4x, raised when stage 6 measured
   the MTP tier). The allowance feeds the device reservation, so the margin
   costs planned VRAM only.

Capture itself was uneventful: `hipStreamBeginCapture(ThreadLocal)` /
`hipGraphInstantiateWithFlags` / `hipGraphExecUpdate` through the stage-1
`hip_compat.h` adapters worked on first contact, including the pinned-host
ingress/egress memcpy nodes inside the captured decode bodies.

## Stage 6 — MTP speculative decoding (`--spec mtp`)

**Milestone: first hardware execution of the MTP round** (target verify,
`mtp_forward_*`, accept/prepare ops) **and of the stage-3 decode draft's
verify / masked / multi-batch kernel forms** — draft windows 2 and 3, full
proposal head and `--lm-head-draft` (artifact's `text/draft_head` loads:
+0.33 GiB device weights), graphs ON. Zero traps, zero incorrect outputs.

### Greedy-invariance investigation (the one output change, root-caused)

p1/p2 reproduce the stage-5 baseline token-for-token. p3/p4/p5 diverge from
the *non-speculative* baseline at a single token each, then continue
coherently (p3: "also **called** the boto" → "also **referred to as** the
boto", same facts; p5: `while current is not None:` → `while current:`,
both correct Python). Investigated rather than waved off:

- **Graphs exonerated**: `--spec mtp` with `--no-cuda-graph` is
  **bit-identical** to `--spec mtp` with graphs on, for both diverging
  probes (p3, p4). The divergence is a property of the speculative path
  itself, not of capture/replay.
- **Root cause**: the verify pass evaluates logits through different
  kernels than the T=1 decode path — T=width linear dispatches and
  attention TokenTile instantiations, plus envelope-dependent split windows
  whose key-accumulation order differs — so logits agree only to rounding.
  Greedy argmax flips on near-ties. This is inherent to the engine's
  design (the same kernel divergence exists upstream on NVIDIA); strict
  spec-vs-nonspec bit-identity was never an engine invariant. Acceptance
  bookkeeping is not implicated: within the speculative configuration the
  output is deterministic and bit-stable across eager/graph, and two of the
  six perf runs (mtp2/p5, mtp3lm/p5) happened to match the baseline exactly.

### Acceptance and throughput (informational — see thermal note)

| Run | Accept rate | Accept len (tok/round) | Accepted by pos | Decode tok/s |
|---|---|---|---|---|
| p4 prose, draft 3 | 40.12% | 2.19 | 42,21,6 | 3.17 |
| p5 code, draft 3 | 95.00% | 3.85 | 40,38,36 | 4.74 |
| p4 prose, draft 2 | 55.00% | 2.10 | 43,23 | 3.45 |
| p5 code, draft 2 | 95.37% | 2.91 | 53,50 | 4.64 |
| p4 prose, draft 3, lm-head-draft | 42.77% | 2.27 | 43,21,7 | 2.54 |
| p5 code, draft 3, lm-head-draft | 91.27% | 3.74 | 42,39,34 | 4.10 |

Acceptance is strongly workload-dependent exactly as expected (code ≫
prose). **MTP currently decodes *slower* than plain decode** (~3–5 vs ~13
tok/s): every verify token runs the T=width correctness-route linears
(steps 8–9 rewrites pending) and each draft step pays a full proposal-head
GEMV, so a round costs far more than the ~2–3.9 tokens it licenses.
Expected to invert once the grouped-int prefill/small-T GEMMs land
(steps 8–10); recorded here as bring-up truth, not a verdict.

## Stage 7 (stretch) — int8-KV attention, wave64 SIMT + `v_dot4_i32_i8`

Two new kernels (commit `e1a7893`), built on the validated SIMT attention
structure with the NVIDIA int8 kernels' numerical scheme:

- **Decode partial** (`gqa_attention_decode_i8_gfx906.cuh`): Q quantized
  on-chip per (row, 64-group) with FP32 scales; K codes read straight from
  the cache and contracted with `__builtin_amdgcn_sdot4` (the
  llama.cpp-gfx906 mmvq idiom), rescaled by `qs·ks` per group; V dequantized
  inline in the P·V broadcast. Fused append replicates the i8 fill kernels'
  quantization (FP16-rounded absmax/127 scale, RNE clamp) bit-for-bit; all
  keys — new and cached — are read back from the quantized cache, matching
  the original kernel's contract. Active-split policy `Int8=true` to agree
  with the shared reducer.
- **Prefill** (`gqa_attention_prefill_i8_gfx906.cuh`): K tiles staged to LDS
  as raw codes (bank-staggered 65-dword rows) + FP16 scales, Q quantized
  once per CTA, sdot4 phase A; V dequantized to bf16 at staging
  (`gqa_kv_dequant_i8x8_from`) so phase C and the epilogue are the bf16
  kernel verbatim. ~50 KiB static LDS (under the 64 KiB/CU limit, one
  CTA/CU like the bf16 edition). The mma-free i8 fill kernels run as-is.

**Milestone: first `--kv-dtype int8` tokens on hardware, all four probes on
first execution** (token identity with bf16 not expected — int8 KV is lossy;
coherence + factual agreement required and observed):

- p1 → `OK`; p2 → `The capital of France is **Paris**.` (content-identical
  to bf16); p3 → same two facts as bf16, benign wording variant; p4 → fluent
  128-token story, same premise/character as the bf16 story before wording
  drift. All stop-token/output-limit finishes, zero traps.
- **VRAM**: KV payload 66 MiB vs 128 MiB bf16 at context 2048 (codes +
  per-group FP16 scales ≈ 0.52x); GPU sequence arena 371.55 vs 433.55 MiB.
- Decode 10.4–12.3 tok/s, prefill 6.3–8.4 tok/s (hot card; bf16 same-session
  ballpark 12–13 / 8–9.7).

## Perf summary (informational — thermally confounded)

The card throttles across a session: 40 °C cold → 83–91 °C junction after
4–5 consecutive loads+runs, with decode/prefill dropping ~20% between the
first and last run of an arm (e.g. graph arm decode 13.42 → 10.94 tok/s in
run order; eager arm, run earlier and cooler, held ~13). Cross-arm deltas
smaller than that are noise until the cooling work lands.

| Config (bf16 KV unless noted) | Decode tok/s | Prefill tok/s |
|---|---|---|
| eager (`--no-cuda-graph`) | 12.9–13.2 | 8.8–9.7 |
| graphs on | 13.4 (cold) … 10.9 (hot) | 8.4–9.4 |
| graphs + MTP (best, code d3) | 4.7 | 7.8 |
| graphs + MTP (worst, prose d3 lm-head) | 2.5 | 7.0 |
| graphs, int8 KV | 10.4–12.3 | 6.3–8.4 |

## Operational notes

- The stage-4 pinned-memory race did not recur; every session used
  `docker stop llama` → ≥10 s wait → run → `docker start llama`.
- One session was killed mid-run to apply the 8x allowance fix (runner PID
  + probe container killed); the GPU survived without reset.
- All GPU work in `ninfer-rocm:6.4.1` (`--device=/dev/kfd --device=/dev/dri
  --security-opt seccomp=unconfined --group-add video`, repo + artifact via
  `-v ~/ninfer-work:/work`); binary `build-gfx906/apps/ninfer`.

## Transcripts

Untracked on the box: `~/ninfer-work/stage5/` (p1–p5 × eager/graph,
`compare.sh`) and `~/ninfer-work/stage6/` (mtp3/mtp2/mtp3lm/mtp3e/kvint8
arms, `compare.sh`); every claim above is greppable from the `.out`/`.err`
pairs.

## Deferred items

- **Grouped-int prefill GEMMs (step 8) and GDN chunked prefill (step 9)** —
  now the dominant cost everywhere, and the reason MTP loses to plain decode.
- **Dispatch/attention retuning on MI50 timings (step 10)** — includes the
  int8 kernels' obvious wins (LDS-staged K codes in decode, DPP ladders,
  half-wave pairing) and re-measuring graph/MTP gains on a cooled card.
- int8-KV + MTP combined run; long-context/deep-context agreement probes vs
  llama.cpp; thinking-mode and sampling-path runs (everything here was
  greedy `--no-thinking`).
- Vision (step 11).
- The tensor-core i8 attention kernels are compiled-but-unreachable under
  compat; prune once the SIMT path is tuned.
