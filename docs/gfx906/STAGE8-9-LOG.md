# gfx906 port — Stage 8–9 log (grouped-int prefill/verify GEMM rewrites)

Status: **stage 8 complete** (2026-08-30); stage 9 (GDN chunked prefill)
**deferred** — see the assessment at the end. Port-order step 8 from
[PORT-AUDIT.md](PORT-AUDIT.md), validated on the MI50 box (`forgev1`, ROCm
6.4.1 container, single 32GB MI50 active). Run recipe unchanged from
[STAGE5-7-LOG.md](STAGE5-7-LOG.md).

## The kernel

`src/ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh`:
`rowsplit_tiled_gemm_gfx906_kernel` — one wave64 LDS-tiled GEMM template
instantiated for Q4G64/Q5G64/W8G32 RowSplit weights × {16, 32, 64}-column
tiles × pluggable scalar epilogues (plain store, fused residual accumulate,
split-row store).

- One 256-thread CTA (4 wave64) owns a 64-row × ColsPerTile output tile.
  Grid = (ceil(rows/64), ceil(T/ColsPerTile)).
- K walks in 64-value stages. Each stage is cooperatively dequantized into
  a padded LDS pane: weights decoded to FP16 pairs (the SIMT decode atoms'
  bias trick, two's-complement XOR included), activations converted
  BF16→FP16 with a ±65504 clamp. gfx906 has no bf16 VALU; FP16 staging is
  the stage-4/7 convention.
- The compute loop contracts LDS panes with `v_dot2_f32_f16` (packed-FP16
  dot, FP32 accumulate — exact products, FP32 sums; the llama.cpp-gfx906
  idiom). Per-thread register tile: 4 rows × {1,2,4} cols.
- LDS rows padded to 34 dwords: 8-byte alignment for `ds_read_b64`
  everywhere and no bank collisions inside a read instruction. ~17.4 KiB
  static LDS at C64 (≤64 KiB/CU; no dynamic smem, no cp.async, no mma).
- Raw next-stage bytes (codes + scales + BF16 activations) are
  **register-prefetched** before each compute loop — the only intra-wave
  global/compute overlap available since the compat cp.async shim is
  synchronous.
- Weights leave DRAM `ceil(T/ColsPerTile)` times per token batch instead of
  the stage-3 SIMT fallback's `ceil(T/8)` (or worse: the fused-op compat
  walks re-read full weights every 8–16 tokens).

Numerics: weight dequant to FP16 is *more* precise than the NVIDIA mma
route's BF16 dequant; BF16→FP16 activation conversion is exact (8→11
mantissa bits) within FP16 range. All conformance suites pass at the
standard criterion (relative L2 ≤ 1 BF16 ulp).

## Routing (all behind `NINFER_GFX906_STAGE8`, default on; `=0` restores stage-3 routes)

| Op | Tiled route | Fallback kept |
|---|---|---|
| linear q4 | T ≥ 6 | T=1 GEMV; T 2–5 SIMT (still wins — see Tier 1) |
| linear q5 | T ≥ 2 | T=1 GEMV (split2/split4 walks lose everywhere) |
| linear w8 | T ≥ 4 | T=1 decode/GEMV; T 2–3 SIMT |
| linear_add q5 (k=6144, 17408) | T ≥ 2 (`TiledResidualGfx906`, fused residual epilogue) | T=1 GEMV-residual |
| attn_input_proj q4_q5 | T ≥ 2 (two tiled GEMMs, split-row epilogues) | T=1 parent-split GEMV |
| gdn_input_proj q4_q5 | T ≥ 2 (two tiled GEMMs; Q5 plane split-stores value/z) | T=1 independent GEMV |
| linear_swiglu q4 | inherited: its compat `Materialized` route calls linear q4 | — |
| gdn conv-snapshot q4_q5 | cols > 1 → `Materialized` (rides the tiled proj GEMMs) | cols=1 decode keeps `ProjectionEpilogueFused` |

Column-tile pick: C16 for T ≤ 16, C32 ≤ 32, else C64.

## Tier 1 — per-kernel microbenchmarks (cold cache, `ninfer_linear_bench`, warmup 2 / repeat 10, interleaved tiled↔fallback per point)

Median µs (effective useful GB/s = (weights+x+out)/time; MI50 theoretical 1 TB/s).
Fallback = the stage-3 compat route for that T (SIMT r8c4/r8c8 or split walks).

### Q4 n=34816 k=5120 (MLP gate_up, 94.7 MB weights)

| T | tiled µs | tiled GB/s (%1TB/s) | fallback µs | fallback GB/s | speedup |
|---|---|---|---|---|---|
| 2 | 758.7 | 125.0 (12.5%) | 452.8 | 209.5 | 0.60x |
| 4 | 753.4 | 126.1 (12.6%) | 605.6 | 156.9 | 0.80x |
| 8 | 757.6 | 125.8 (12.6%) | 1194.9 | 79.8 | 1.58x |
| 16 | 756.3 | 126.9 (12.7%) | 2123.2 | 45.2 | 2.81x |
| 32 | 921.6 | 105.5 (10.6%) | 4093.4 | 23.8 | 4.44x |
| 64 | 1663.7 | 60.0 | 8253.0 | 12.1 | 4.96x |
| 128 | 3171.2 | 33.1 | 16430.1 | 6.4 | 5.18x |
| 256 | 6648.9 | 17.3 | 32715.6 | 3.5 | 4.92x |

(The falling "useful GB/s" at large T is the regime change: weights are
re-read only once per 64 columns, and the kernel becomes compute-bound at
~13.7 TFLOP/s FP16 — ~52% of the MI50's 26.5 TFLOP/s peak.)

### Q4 n=4096 k=5120

| T | tiled µs | fallback µs | speedup |
|---|---|---|---|
| 2 | 205.8 | 73.4 | 0.36x |
| 4 | 205.9 | 99.5 | 0.48x |
| 8 | 207.5 | 222.6 | 1.07x |
| 16 | 212.3 | 374.7 | 1.76x |
| 32 | 271.0 | 670.7 | 2.47x |
| 64 | 416.6 | 1107.5 | 2.66x |
| 128 | 635.7 | 2054.5 | 3.23x |
| 256 | 870.9 | 4048.1 | 4.65x |

### Q4 n=131072 k=2048 (draft head)

| T | tiled µs | fallback µs | speedup |
|---|---|---|---|
| 2 | 994.4 | 662.7 | 0.67x |
| 8 | 989.6 | 1506.6 | 1.52x |
| 16 | 993.3 | 3080.3 | 3.10x |
| 64 | 2397.7 | 13202.6 | 5.51x |
| 256 | 10074.3 | 52875.2 | 5.25x |

### Q5 n=6144 / n=7168, k=5120 (split4/split2 fallbacks)

| shape | T | tiled µs | fallback µs | speedup |
|---|---|---|---|---|
| 6144 | 2 | 227.4 | 866.6 | 3.81x |
| 6144 | 4 | 232.6* | 1581.1* | 6.80x |
| 6144 | 8 | 235.2* | 253.3* | 1.08x |
| 6144 | 16 | 237.9* | 457.1* | 1.92x |
| 6144 | 64 | 424.8* | 1504.9* | 3.54x |
| 6144 | 256 | 1199.7* | 5676.1* | 4.73x |
| 7168 | 4 | 228.3 | 1838.4 | 8.05x |
| 7168 | 64 | 451.0 | 1731.8 | 3.84x |
| 7168 | 256 | 1427.7 | 7221.8 | 5.06x |

(* from the pre-prefetch Tier-1 pass, same session ballpark; the q5 rows
were re-measured post-prefetch at T=2/4 for both shapes with matching
results.)

### W8 n=34816 k=5120 and n=5120 k=17408 (MTP layer)

| shape | T | tiled µs | tiled GB/s | fallback µs | speedup |
|---|---|---|---|---|---|
| 34816×5120 | 2 | 778.2 | 243.6 (24.4%) | 714.1 | 0.92x |
| 34816×5120 | 4 | 774.9 | 244.8 | 1060.0 | 1.37x |
| 34816×5120 | 8 | 751.0 | 253.0 (25.3%) | 1937.6 | 2.58x |
| 34816×5120 | 16 | 769.3 | 247.9 | 3812.6 | 4.96x |
| 34816×5120 | 64 | 1672.6 | 116.3 | 15114.3 | 9.04x |
| 34816×5120 | 256 | 6775.0 | 31.0 | 60328.8 | 8.90x |
| 5120×17408 | 4 | 676.6 | 140.2 | 609.1 | 0.90x |
| 5120×17408 | 8 | 649.3 | 146.4 | 1030.9 | 1.59x |
| 5120×17408 | 64 | 1212.9 | 80.5 | 7215.8 | 5.95x |
| 5120×17408 | 256 | 3670.5 | 28.9 | 28283.6 | 7.71x |

### Q6 output head (measured for the MTP round model; NOT rewritten)

n=248320 k=5120, fallback only: T=1 2870.6 µs (346 GB/s — healthy GEMV);
T=2 3814.5; T=4 5575.0; T=8 10887.6. At MTP verify widths (≤4) the head
costs ≤1.9x its T=1 read — not the inversion driver; a Q6 tile atom is a
cheap follow-up if ever needed.

### Reading of the small-T regime

The tiled kernel is latency-flat across T ≤ 16 (~same time at T=2 and
T=16): with only ceil(rows/64) CTAs and synchronous stage loads, it is
bound by the serialized K-stage walk, not bandwidth. Register prefetch
bought ~10% at n=4096; the fallback's 8x-more CTAs still win below the
per-family thresholds above. Deeper small-T work (K-split grids,
wide-GEMV forms) is stage-10 scope.

## Tier 2 — end-to-end A/B (graphs on, bf16 KV, greedy, `--no-thinking`, interleaved on/off/on/off per probe)

Stage-5 probes, one container per run, exit 0 everywhere. `on` = stage-8
tiled routes, `off` = `NINFER_GFX906_STAGE8=0` (stage-3 fallbacks).

| Probe | prefill on (run1/run2) | prefill off | **ratio** | decode on | decode off |
|---|---|---|---|---|---|
| p3 (166-tok passage) | 170.7 / 168.9 tok/s | 9.65 / 9.63 | **~17.6x** | 13.37 / 13.21 | 13.27 / 13.12 |
| p4 (prose, 128 new) | 63.6 / 62.7 | 8.58 / 8.25 | **~7.5x** | 13.10 / 12.36 | 12.74 / 11.72 |
| p5 (code, 200 new) | 60.2 / 55.2 | 6.27 / 5.37 | **~10x** | 11.09 / 9.79 | 10.26 / 9.51 |

- p4/p5 prompts are ~10 tokens — their "prefill" is a single small-T chunk
  (launch-bound), hence the lower ratio; p3 is the representative
  multi-block prefill and lands at **~170 tok/s vs the stage-5–7 baseline's
  ~9** (stage-4 logged 9–9.7 on this very probe).
- Decode is untouched by design (T=1 routes unchanged): on/off decode
  alternates inside a monotonic thermal decline (13.4 → 9.5 across the
  12-run session) with no arm-shaped separation. Absolutes are
  thermally confounded per the standing protocol; the interleaved ratios
  are the result.

**Correctness**: p3 and p5 greedy token ids are bit-identical on-vs-off in
both interleaved rounds. p4 diverges at a single token (#47:
"…clatter **that Elias had long since learned to ignore**" vs "…clatter
**that had become the heartbeat of Elias's life**") and both arms continue
fluently on the same premise — the known-acceptable near-tie class from the
stage-6 findings. Both arms are bit-stable run-to-run. p1 → `OK`, p2 →
`The capital of France is **Paris**.` (stage-4 milestones, tiled arm).

## MTP re-measurement (step 3) — the crossover

Stage-6 configs re-run with the tiled routes (graphs on, greedy,
interleaved on/off where shown). The stage-6 verdict — "MTP currently
decodes slower than plain decode" — is now inverted for code, and at
break-even-or-better for prose at draft 2:

| Config | decode tok/s stage 6 | s8 off (this session) | **s8 on** | plain decode (same session) |
|---|---|---|---|---|
| p5 code, draft 3 | 4.74 | 4.84 | **23.9** | ~11–13 |
| p5 code, draft 2 | 4.64 | — | **19.3** | ~11–13 |
| p4 prose, draft 3 | 3.17 | 2.30 | **10.8** | ~12–13 |
| p4 prose, draft 2 | 3.45 | — | **13.9** | ~12–13 |

- Acceptance bookkeeping is untouched (e.g. p5 d2: 95.37% / 2.91
  tok/round, identical to stage 6; p4 d2: 55.00% / 2.10 identical).
- **Crossover**: MTP now beats plain decode decisively on code (draft 3
  best: ~1.8–2x plain) and at draft 2 on prose (13.9 vs ~13 on a hotter
  card); prose draft 3 sits just under plain (acceptance 38% doesn't pay
  for the third draft step). Recommended default from this data:
  `--spec mtp --draft-tokens 3` for code-heavy use, `2` as the general
  default.
- Draft-2 needed one extra fix beyond the GEMMs: the GDN conv-snapshot
  route table sent verify widths 1,2,3,5,6 through
  `ProjectionEpilogueFused` (inner GEMMs = stage-3 fallbacks) while width
  4 took `Materialized`; before the reroute, draft-2 rounds cost 404 µs vs
  draft-3's 171 µs per round on the same probe (bug #5). p5 d3 is
  bit-reproducible across sessions (`generated ids` identical run-to-run).
- Absolutes thermally confounded as always (junction 71→81 °C across the
  re-measure session); the on/off pairs are interleaved.

## Bugs found (symptom → cause)

1. **Every conformance shape failed with sign-flipped garbage on first GPU
   run** → the tiled decode atoms dropped the two's-complement conversion:
   Q4 needs `word ^ 0x88888888` before the nibble+1024 bias trick (raw
   nibble 0 must decode to 0, not −8), Q5 needs the high-plane bits
   flipped (`^ 0xffff`). One-line fixes; every suite green after.
2. **`ninfer_linear_bench` linked with `undefined symbol: main`** → bench
   `.cu` sources weren't compiled as HIP (CMake ignored them);
   `ninfer_cuda_archive` now applied to bench/test targets.
3. **Bench/tests didn't build at all under compat** (`CUDA::cudart`,
   `cuda_runtime.h`, `cudaErrorNoDevice`, `cudaGraphGetNodes`,
   `cuda_profiler_api.h`) → the bench/tests trees were never HIP-ported in
   stages 1–7; compat include tree + `hip::host` + three shim additions.
4. **Input-proj tests abort in NVFP4 sections** → pre-existing: those
   backends throw/stub on gfx906; sections now SKIP under compat so the
   q4_q5/bf16/w8 sections gate the port.
5. **Draft-2 MTP rounds 2.4x the cost of draft-3 rounds after stage 8**
   (p5: 404 vs 171 µs/round; the isolated linears measured *cheaper* at
   T=3 than T=4) → GDN conv-snapshot route table: widths 1,2,3,5,6 →
   `ProjectionEpilogueFused` whose inner projections are the stage-3
   fallback GEMMs, width 4 → `Materialized` (tiled). Multi-token widths
   now take `Materialized` under the stage-8 gate; draft-2 decode went
   7.20 → 19.32 tok/s (code) and 4.88 → 13.87 (prose).

## Deferred

- `ninfer_gdn_input_proj_conv_snapshot_test` aborts with a device trap in
  a conv-snapshot variant (batched/masked) **independent of stage 8**
  (reproduced with `NINFER_GFX906_STAGE8=0`); the engine's validated
  stage-4/7 paths never hit it. Root-cause in stage 10.
- Small-T (2–5) tiled losses: K-split or wide-GEMV forms (stage 10).
- Q6 tile atom for the output head (only matters if MTP widths grow).
- The W8 gdn/attn/pair fused families (35B target's layers) keep their
  stage-3 compat walks — the 27B never routes there; port when a 35B
  artifact matters.
- **Stage 9 (GDN chunked prefill): deferred on measurement, not on
  difficulty.** With stage 8 in, the recurrent GDN tail is no longer a
  material prefill cost: `ninfer_gated_delta_net_bench --running
  --tokens 512` measures 3.13 ms/layer/512-tok (median, cold), so the
  48 GDN layers cost ~0.29 ms/token — **~5%** of the ~5.9 ms/token
  end-to-end prefill measured in Tier 2. The chunked rewrite
  (`prepare_wy_wu`/`state_passing`/`output`, ~1.9k LOC of mma kernels)
  can recover at most that 5% at these context lengths; it only becomes
  interesting after the next round of prefill work shrinks everything
  else. Revisit alongside stage-10 attention tuning.

## Transcripts

Untracked on the box under `~/ninfer-work/stage8/`: conformance logs
(`*-test.log`), Tier-1 sweeps (`t1.log` pre-prefetch, `t1b.log` final),
Tier-2 probe pairs (`p*-s8on/off-*.out/.err`, `t2.log`), MTP re-measures
(`p*-mtp*-s8*.{out,err}`, `t3.log`).
