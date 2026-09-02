# Stage 10 — re-baseline, profile, and the retune plan (gfx906 port)

Started 2026-09-02. This log is the plan that earlier logs deferred to as "stage 10" without writing it down.
Every pass below is one deliverable, commits incrementally, and runs long jobs detached on the box.

## 0. Ground truth first (pass 0, done 2026-09-02 09:37)

Hardware: forge, 2x MI50 32 GB, kernel 6.12.98 (PCIe P2P working), blowers ducted, memory temp equilibrium
~70 C under sustained load. Single-card work uses card 2 (PCI 0000:09:00.0, HIP device 1) to spare card 1,
which has two uncorrectable HBM ECC events on record. Production (llama-swap, 2-card tensor split) is stopped
for each batch and restarted after.

Toolchain: image `ninfer-rocm:6.4.1` (ROCm 6.4.1-83, rocprofv3 present; the profiled app needs `libdw1`
installed in the container), build tree `build-gfx906` from 2026-08-30 (code commit b81331b; a2cafe3 is docs
only). Weights: `qwen3_8_27b.ninfer`, groupwise-int, 17.1 GB on device.

### Batch A — engine bench matrix, plain decode, bf16 KV, HIP graphs on

| test | prefill t/s | decode t/s | notes |
|---|---|---|---|
| pp512 | 212.2 (sd 0.3) | | |
| pp2048 | 198.4 (sd 0.7) | | prefill chunk 1024 |
| tg128 | | 12.66 (sd 0.02) | mem peak 67 C |
| pp2048+tg128 | 196.2 | 12.29 | |

### Batch B — MTP (CLI, greedy, no thinking, bf16 KV, 2K ctx, card cooled)

| prompt | plain | MTP draft 2 (acc) | MTP draft 3 (acc) |
|---|---|---|---|
| code p5 | 12.57 | 18.27 (95.4 %) | 22.89 (94.3 %, 3.83 tok/round) |
| prose p4 | 12.60 | 13.19 (55.0 %) | 12.77 (38.3 %) |

Short-prompt prefill 58-64 t/s (10-50 token prompts, launch-bound). Identical to the throttled Aug-30 figures.
Cooling changed nothing on decode.

### Batch D — deep-context tool-call probe through ninfer-serve

Probe v3 (six planted facts with near-miss decoys, two tools) at 30,881 tokens, bf16 KV, MTP draft 3: PASS 7/7,
argument hash `7b51bb065511` = byte-identical to llama.cpp's on the same probe. Correctness at depth holds.
The same probe at 112K was ABORTED after 24 minutes: prefill ran at 204.8 t/s per 1024-token chunk at the start,
but chunk completions spaced out with depth (~66K tokens in 24 min, then minutes per chunk). Effective prefill
over 0-30K was 74 t/s. The naive prefill attention is quadratic-dominant past ~30-50K.

### Batch E — rocprofv3 kernel trace (22-token prompt + 64 decode tokens, graphs on)

Wall span 8.20 s, summed kernel time 5.49 s (67 % busy overall); the decode tail is 100 % busy — no launch gaps
under graphs. Decode is KERNEL-bound, not launch-bound. Per token (64 layers):

| kernel (T=1 decode) | calls | avg us | share | per token | bandwidth ideal at 700 GB/s |
|---|---|---|---|---|---|
| q4_linear_swiglu_gemv_pair (MLP gate+up, Q4) | 4160 | 573.7 | 43.5 % | 36.7 ms | ~130 us -> 4.4x slow |
| q5_rowsplit_gemv 5120x17408 (MLP down, Q5) | 4160 | 225.1 | 17.1 % | 14.4 ms | ~80 us -> 2.8x |
| q5_rowsplit_gemv 5120x6144 (attn/GDN out) | 4160 | 118.1 | 9.0 % | 7.6 ms | ~30 us -> 4x |
| q5_rowsplit_gemv 12288x5120 (GDN in-proj) | 3120 | 154.4 | 8.8 % | 7.4 ms | ~56 us -> 2.8x |
| w8_rowsplit_gemm_simt (output head, T=1) | 66 | 3143 | 3.8 % | 3.1 ms | ~2.2 ms |
| everything else (attention, GDN recurrent, norms, rope) | | | < 8 % | ~6 ms | |

Four GEMV kernels = 66 of the ~79 ms per token, running at 20-35 % of achievable bandwidth. These are the
stage-3 "correctness-first" T=1 routes the stage-8 log left untuned (tiled GEMM only wins at T >= 6).

Prediction check (pre-registered): tg128 15-17 -> 12.66 MISS low; pp512 200-250 -> 212 hit; MTP code d3 ~28-32
-> 22.9 MISS low (same cause). The "launch/latency-bound" hypothesis from the plan was WRONG for decode; the
profile replaces it with "T=1 GEMV kernels at a quarter of bandwidth".

## 1. Comparison table (the "Tier 3" table, created here)

Same box. llama.cpp rows from ~/BENCH-TABLE.md (bench2 8x1024 decode, bench-pp 8,148-token prefill); ninfer rows
from the engine bench / CLI probes. Bundle comparisons, not kernel races.

| engine | cards | weights | decode plain | decode MTP | prefill | deep probe |
|---|---|---|---|---|---|---|
| llama.cpp mxxm b10811 (production) | 2, tensor split | Q8_0 (29 GB) | ~42 | 45.5-46.5 (n-max 3) | 315 @8K, 222 @112K | 7/7 @204K |
| llama.cpp llama-ref:bench-v1 (pinned) | 1 | Q4_0 (16 GB) | n/m cooled | 17.63 (throttled 08-30) | n/m | n/m |
| ninfer gfx906 @a2cafe3 | 1 (card 2) | groupwise-int (17.1 GB) | 12.66 | 22.9 code / 12.8 prose | 212 @512, 74 effective @30K | 7/7 @30K, same hash |

## 2. Plan and predictions (revised after the profile)

| pass | scope | predicted single-card result | gate |
|---|---|---|---|
| 1 | cherry-pick portable upstream commits: RMSNorm hoist (9954867a), MTP prefill token on device (02bd904e), GDN conv writes q/k/v direct (92bb06eb), SM-count cooperative tiling (e51b585c), fp16 V storage (21a0e85f) | decode +1-3 % (norms are 2 % of the profile), prefill +1-3 %, one crash class removed | conformance suites pass, batch-A matrix no regression |
| 2 | T=1 GEMV retune for gfx906: the four kernels above (LDS/DPP reductions, wave64 lane mapping, vectorised q4/q5 unpack, K-split for tall shapes); then the W8 output head; then small-T (2-5) forms for MTP verify | plain 12.7 -> 28-36 (GEMVs at 60-75 % of bandwidth); MTP code 45-60; prefill unchanged | tier-1 microbench per shape, conformance, batch-A matrix, probe v3 7/7 at 30K |
| 3 | attention retune, PREFILL first (the quadratic slope), then decode: DPP-ladder reductions, half-wave pairing, K reuse across the q-head group, LDS-staged int8 keys | prefill at 30K: 74 -> 150+ effective; at 112K: finishes in < 15 min; decode +0-5 % short ctx | probe v3 at 112K and 204K through serve |
| 4a | TP2 cherry-pick from wamansou/ninfer-tp2-1m (3 HIP shim macros for peer access; donor benefits from P2P automatically); YaRN 1M deferred | 2-card plain 1.4-1.8x single | tp2 parity test, probe on two cards |
| 4b | 8-bit artifact (W8 route already exists) on two cards = the production-class head-to-head vs llama.cpp Q8_0 | plain 40-45, MTP code 55-80 vs production 46 | probe 7/7 at 204K, then BENCH-TABLE head-to-head |

Physical ceiling used: ~700-810 GB/s achievable HBM -> 39-45 t/s plain for 17.1 GB per token on one card;
MTP x1.05 (prose, this model at draft 2) to x1.8 (code, draft 3, measured); 2 cards ~1.4-1.8x after collective cost.

## 3. Corrections to earlier notes

- "Tier-3 llama.cpp comparison table" was referenced in the 2026-09-01 agenda but never existed; section 1 is it.
- The stage-8/9 log's "stage 10" items were scattered across STAGE3/4/5-7/8-9 logs and the port audit; section 2
  consolidates them, re-ranked by the profile.
- The laptop clone lives at `C:\Users\johnp\dev\ninfer-port\ninfer` (branch gfx906-port), remotes `box` and `fork`.
- Stale untracked draft `src/ops/linear/rowsplit_gemm_tiled_gfx906.cuh` on the laptop clone: unreferenced, deleted.
- Harness notes: `ninfer_bench` needs `--corpus` (or the repo as cwd) and `--profile-measured` needs one test with
  `-r 1`; rocprofv3 needs `libdw1` in the container; `ninfer-serve --kv-capacity auto` sized 120K tokens at 8.85 GiB.

## 4. Pass 1 results (2026-09-02, five upstream cherry-picks, all kept)

| pick | commit | gate results | tg128 | pp512 | note |
|---|---|---|---|---|---|
| 9954867a rmsnorm weight hoist | 06f2a173 | rmsnorm + gated_rmsnorm OK | 12.74 | 211.7 | prefetch-blocks literal 170 -> 60 CUs under compat |
| 02bd904e MTP prefill token on device | f00471c3 | greedy MTP identical | 12.74 | 211.7 | one textual conflict (timing hooks) |
| e51b585c GDN cooperative-launch capacity | d19adf37 | gdn_gating_proj OK (first build on this tree) | 12.76 | 210.6 | cooperative path unreached on gfx906 (route tables); CU count from occupancy API |
| 92bb06eb GDN prefill conv direct q/k/v | 52e3c864 | conv1d_silu + gdn_input_proj OK; greedy identical | 12.67 | 210.5 / 197.3 @2048 | predicted +1-3 % prefill did NOT appear; workspace peak unchanged |
| 21a0e85f fp16 V storage + PV | 35a9c5aa | gqa_attention + kv_cache_append OK (first build); greedy identical; probe v3 30K 7/7 | 12.61 (hot) | 209.1 | port SIMT attention kernels now consume fp16 V; 22 files |

Verdict: hygiene, not speed — exactly what the batch-E profile said would happen. Decode stays in the 12.6-12.8
band; prefill within noise. The port is now current with upstream on the portable runtime and kernel changes and
three gate tests build for the first time. Pass 2 (T=1 GEMV retune) is the speed lever; its design brief is
docs/gfx906/PASS2-DESIGN.md.

## 5. Pass 2 results (2026-09-02, T=1 GEMV retune, four shapes, PROMOTED)

One kernel family for all four (src/ops/linear/gfx906/q_gemv_gfx906.cuh): register-resident wave64 Q4/Q5 chunk
streams (weights read once, loads issued before the LDS barrier), x staged once per block as fp16 in LDS with the
stage-8 bias trick, v_dot2_f32_f16 into fp32, a DPP reduction ladder over each 32-lane half of a wave64
(gfx906_reduce_sum32 in warp.cuh), the upstream epilogue contract called from half-lane 0. Env knob
NINFER_GFX906_PASS2 (default on, =0 reverts every route). kDepth=1 won every depth screen: with the step loop
unrolled the compiler hoists loads anyway and deeper rings only cost VGPRs. The only per-shape choice was rows per
block, set by LDS occupancy and the grid tail on 60 CUs.

| shape | kernel before | after | speedup | GB/s (fraction of ~810) | tg128 after | commit |
|---|---|---|---|---|---|---|
| q5 5120x6144 out-proj + residual | 121 us | 53.6 us | 2.26x | 386 (48 %) | 13.32 | 6f8e9a8a, 1c8e37d6 |
| q4 swiglu gate/up pair 17408x2 x 5120 | 585 us | 151 us | 3.87x | 626 (77 %) | 19.44 | 17bd4cda, 50d18106 |
| q5 down 5120x17408 + residual | 229 us | 139 us | 1.65x | 421 (52 %) | 21.87 | 676b30c4 |
| q5 GDN v/z 12288x5120 split + conv/snapshot | 153.5 us | 96.2 us | 1.60x | ~405 (50 %) | 23.32 | 751290d7 |

Tier-2 interleaved A/B (tg128): off 12.73 / on 23.27 / off 12.64 / on 23.23. Prefill unchanged (T=1 never runs
in prefill). Greedy p4-MTP and p3-plain outputs byte-identical with the knob on and off after every shape. Probe v3
at 30,881 tokens through ninfer-serve: 7/7, argument hash identical to llama.cpp. RAS 0 throughout.

Lessons: at K=17408 the lever is waves sharing one LDS copy of x (1024-thread block), not ring depth, and never
route x through L2 per lane (the loads serialise behind vmcnt(0)). The down projection missed its 110 us target
because 5120/32 = 160 blocks leaves a tail on 60 CUs and no block size between 16 and 32 rows divides N. The GDN
conv_snapshot unit test has a pre-existing std::bad_alloc (arena undersized for the gfx906 plan) with the knob on
and off, so the T=1 snapshot path is covered by the greedy gates only.

MTP did NOT move (code d3 22.89 / prose d2 13.19, identical to pass 0): the verify step runs T=3-5 and those routes
are still the stage-3 scalar fallbacks (the stage-8 tiled GEMM only wins at T>=6). Plain decode now equals MTP on
code and beats it on prose, the stage-8 inversion again by the same mechanism. Pass 2e extends the family to
T=2..5; prediction MTP code 35-45, prose 24-28. It runs before pass 3 because MTP is the production path.
