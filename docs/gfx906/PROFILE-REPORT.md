# Where the non-linear time goes: per-token cost breakdown, ninfer gfx906

Date: 2026-09-02/03. Box `forgev1`, repo `~/ninfer-work/ninfer-gfx906` @ 7a3c18d9 (gfx906-port),
build `build-gfx906`, artifact `/work/qwen3_8_27b.ninfer`. All runs 64 new tokens on a 166-token
prompt (`stage5/p3.txt`), card 2 unless stated. llama-swap stopped for the whole GPU block and
restarted afterwards (health 200 confirmed). RAS `umc_err_count` before and after every run:
`0000:03:00.0 ue:0 ce:0 de:0` / `0000:09:00.0 ue:0 ce:0 de:0`. Peak card temp during the block 44 C.

## Headline

**There is no launch overhead to find.** At tp1 with graphs on the GPU is busy 42.914 ms out of a
43.047 ms token. The total idle time between consecutive kernels is **0.143 ms/token (0.3 %)**.
The ~11 ms you were attributing is not scheduling: it is 11 ms of kernels that are running, and
running below the memory-bandwidth floor.

Restated: of the 43.0 ms token, **37.2 ms is weight-streaming linear algebra** (the four pass-2
families 28.2 + the un-ported attention/GDN input projections 5.7 + the LM head 3.2) and only
**5.7 ms is everything else** (attention 2.6, norms 1.7, GDN recurrent 1.3, rope/sampling/copies 0.1).

The real finding is inside the linears. The one family that was fully rewritten in pass 2 (the q4
gate/up pair) runs at **92 % of the 624 GB/s floor**. Every other weight-reading kernel runs at
**32-66 %**. Closing that gap - not chasing launch overhead - is where the milliseconds are.

## Run summary

| Run | Config | Result | ms/token | kernels/token | GPU busy | idle |
|---|---|---|---:|---:|---:|---:|
| A | tp1 card 2, plain, graphs ON | 23.26 t/s | 43.047 | 698 | 42.914 | 0.143 |
| A2 | same, `--no-cuda-graph` | 22.62 t/s | 44.359 | 696 | 44.291 | 0.068 |
| B | tp1 MTP d3 + lm-head-draft, greedy | 35.46 t/s | 27.794 | 249 | 27.645 | 0.045 |
| C | tp2 eager, plain | 31.01 t/s | 32.262 | 2000 | see below | 14.098 |
| D | tp1 MTP d3 + lm-head-draft, sampled | 31.89 t/s | 30.896 | 277 | 30.742 | 0.051 |

CUDA-graph saving = A2 - A = **1.31 ms/token (3.0 %)**. It does not come from removing gaps
(eager actually has *less* total idle). It comes from each kernel running faster under replay -
e.g. rmsnorm 12.3 -> 9.0 us, GDN snapshot 19.9 -> 15.6 us. Graphs are worth keeping but they are
not hiding a launch-cost problem, because there is not one.

**Measured kernel launch cost** (gaps between consecutive dispatches on the same device, run A):
median 0.79 us, p90 2.7 us, p99 45.6 us, 702 positive gaps per token. Summed: 0.143 ms/token.
Even if every gap went to zero the token would go from 43.05 to 42.91 ms - 23.26 -> 23.33 t/s.

## Class table - run A (tp1, graphs on), per decode token

| Class | kernels/tok | ms/tok | mean us | largest us | share |
|---|---:|---:|---:|---:|---:|
| pass-2 linear GEMV (T=1) | 240 | 28.236 | 117.7 | 183.2 | 65.6 % |
| other linear: attn + GDN input projections | 80 | 5.736 | 71.7 | 118.6 | 13.3 % |
| LM head (w8 248320x5120) | 1 | 3.213 | 3212.9 | 3277.7 | 7.5 % |
| attention (gqa) | 32 | 2.641 | 82.5 | 170.7 | 6.1 % |
| norms (rmsnorm) | 209 | 1.686 | 8.1 | 13.9 | 3.9 % |
| GDN recurrent / conv / gating | 112 | 1.266 | 11.3 | 19.7 | 2.9 % |
| rope | 16 | 0.084 | 5.3 | 6.2 | 0.2 % |
| embed / sampling bookkeeping | 4 | 0.031 | 7.7 | 73.6 | 0.1 % |
| runtime copies / fills | 4.1 | 0.020 | 4.8 | 10.4 | 0.0 % |
| **GPU busy** | **698** | **42.914** | | | 99.7 % |
| idle between kernels | - | 0.143 | | | 0.3 % |
| **wall** | | **43.047** | | | |

### Top kernels by total time (run A)

| kernels/tok | ms/tok | mean us | kernel |
|---:|---:|---:|---|
| 64 | 10.523 | 164.4 | `q4_swiglu_pair_gemv_gfx906<17408,5120>` - MLP gate/up, all 64 layers |
| 64 | 9.560 | 149.4 | `q5_gemv_gfx906<5120,17408>` - MLP down + residual |
| 48 | 4.520 | 94.2 | `q5_gemv_gfx906<12288,5120>` GdnDecodeEpilogue - GDN v/z + conv + snapshot |
| 64 | 3.633 | 56.8 | `q5_gemv_gfx906<5120,6144>` - attention and GDN output projections |
| 1 | 3.213 | 3212.9 | `w8_rowsplit_gemm_simt` - LM head |
| 48 | 2.677 | 55.8 | `q4_rowsplit_gemv` grid 4096 rows - GDN q/k half of the input projection |
| 16 | 2.486 | 155.3 | `gqa_attention_small_t_simt_partial_bf16<24,4,1>` |
| 16 | 1.731 | 108.2 | `q5_rowsplit_gemv<7168,5120>` - attention input, q5 parent |
| 16 | 1.326 | 82.9 | `q4_rowsplit_gemv` grid 7168 rows - attention input, q4 parent |
| 129 | 1.165 | 9.0 | `rmsnorm_cta_bf16x2` |
| 48 | 0.747 | 15.6 | `gated_delta_net::recurrent_snapshot` (grid 48 blocks x 32 threads) |
| 80 | 0.521 | 6.5 | `rmsnorm_warp_bf16x2` |
| 48 | 0.423 | 8.8 | `bf16_gdn_gating_proj_gemv<48>` |
| 16 | 0.155 | 9.7 | `gqa_attention_small_t_reduce_output` |
| 16 | 0.096 | 6.0 | `sigmoid_gate_mul_bf16x8` |

## Against the floor

Weight bytes per call for the four pass-2 shapes and the LM head are taken from
`docs/gfx906/PASS2-DESIGN.md` (the same numbers the pass-2 design was sized against).
Floor = bytes / 624 GB/s.

| Family | calls/tok | bytes/tok | measured ms | achieved GB/s | % of 624 | floor ms |
|---|---:|---:|---:|---:|---:|---:|
| MLP gate/up (q4 pair) | 64 | 6.06 GB | 10.523 | **576** | **92 %** | 9.71 |
| MLP down (q5 + residual) | 64 | 3.52 GB | 9.560 | 368 | 59 % | 5.64 |
| GDN v/z input (q5) | 48 | 1.87 GB | 4.520 | 414 | 66 % | 3.00 |
| out-projections (q5) | 64 | 1.28 GB | 3.633 | 352 | 56 % | 2.05 |
| LM head (w8) | 1 | 1.27 GB | 3.213 | 396 | 63 % | 2.04 |
| GDN q/k input (q4) | 48 | ~0.54 GB | 2.677 | ~200 | ~32 % | ~0.86 |
| attention input (q4+q5 pair) | 32 | ~0.69 GB | 3.057 | ~224 | ~36 % | ~1.10 |
| **all weight streaming** | | **~15.2 GB** | **37.185** | **409** | **66 %** | **24.4** |

The 15.2 GB total is a sanity check against the resident footprint: the run reports 16.1 GiB of
weights on the card, and the difference is the MTP layer, the optimised draft head (131072x5120
w8, 0.67 GB) and the vision tower, none of which are read on a plain decode token. The two
attention-input numbers are marked `~` because the attention input projection is dispatched as
two separate parent matrices, both covering all 7168 rows (`q4_q5_attn_input_plan.cpp`,
"one wave64 LDS-tiled GEMM per parent weight"); I did not confirm from source whether the two
parents split the K range or genuinely store the matrix twice. The byte budget above only closes
if both parents are read in full, which suggests the latter. Worth ten minutes of someone's time
to check, because if it is the latter the attention input projection is reading twice the bytes
it needs to.

Non-linear classes against their floors:

| Class | measured ms/tok | byte floor ms/tok | ratio |
|---|---:|---:|---:|
| attention (gqa, ~230-token context) | 2.641 | ~0.03 (15 MB of KV) | **~90x** |
| norms (209 launches) | 1.686 | ~0.01 (4 MB) | ~170x |
| GDN recurrent + conv + gating | 1.266 | ~0.25 (144 MB of state) | ~5x |
| rope + sampling + copies | 0.135 | ~0 | floor |

GDN state: 48 layers x 48 value heads x 128 x 128 bf16 = 1.5 MB read + 1.5 MB written per layer,
144 MB/token. The attention floor is tiny because at 230 tokens of context the KV cache is
16 layers x 4 KV heads x 256 x 2 x 230 x 2 B = 15 MB.

## Why attention and the norms are so far off the floor: occupancy

The grid sizes in the trace say it plainly. This card has 60 CUs / 3840 lanes.

- `gqa_attention_small_t_simt_partial_bf16<24,4,1>`: grid 256 threads, workgroup 64 -> **4 blocks,
  256 lanes**. It occupies 4 of 60 CUs for 155 us, sixteen times a token. It is not moving data;
  it is one narrow serial walk over the KV cache.
- `recurrent_snapshot`: grid 1536 threads, workgroup 32 -> **48 blocks of 32 threads**, i.e. 24
  wave64 on a 60-CU part. 15.6 us x 48 layers.
- `rmsnorm`: 209 separate launches per token, 8.1 us mean, each moving ~20 KB. That is 2.5 GB/s.
  These are not bandwidth events, they are 209 short kernels that exist because the norm was not
  folded into the GEMV epilogue that precedes or follows it.

## MTP: what sampling actually costs (runs B vs D)

| | B (greedy) | D (temp 0.8 / top-p 0.95 / top-k 20, thinking on) |
|---|---:|---:|
| decode | 35.46 t/s | 31.89 t/s |
| rounds for 64 tokens | 18 | 20 |
| ms per round | 98.83 | 98.87 |
| kernels per round | 887 | 886 |
| acceptance rate | 83.3 % | 74.1 % |
| accepted by draft position | 16,15,14 | 17,14,12 |
| tokens per round | 3.56 | 3.20 |
| ms per token | 27.79 | 30.90 |

**The mechanism is not extra work.** A sampled round costs 98.87 ms and a greedy round costs
98.83 ms, with the same 886-887 kernels and the same kernel classes. The verify width is
unchanged (the T=4 kernels - `q4_swiglu_pair_gemv_smallt<17408,5120,4>` at 302 us,
`rowsplit_tiled_gemm<Q5TileAtom,16>` at 355 us - fire once per round either way). The whole loss
is acceptance: 3.56 -> 3.20 tokens per round, an 11 % drop that maps almost exactly onto the 10 %
t/s drop. The per-position numbers show where it goes: position 1 is barely affected (16/18 vs
17/20) and position 3 falls from 14/18 = 78 % to 12/20 = 60 %. Sampling breaks the deeper draft
positions, not the draft head itself.

Note: acceptance did not halve in this run - 83 % -> 74 %. Either the earlier "halves"
observation was prompt-dependent, or it came from a different sampler setting than the one used
here.

A round is 98.8 ms for 3.2-3.6 tokens. At tp1 the MTP path only wins because 4 tokens of verify
cost 98.8 ms rather than 4 x 43.0 = 172 ms - the weights are read once for four tokens. Its
kernels are correspondingly further from the floor (the T=4 shapes take 302-355 us for the same
bytes a T=1 kernel reads in 164 us), which is expected and fine.

## tp2 eager (run C): the load split is wrong

32.262 ms/token wall, 2000 kernels, but the two cards are not doing the same work:

| | card 1 (agent 1) | card 2 (agent 2) |
|---|---:|---:|
| busy | 22.866 ms | 27.603 ms |
| idle (of the 32.26 ms wall) | 9.438 ms (29 %) | 4.660 ms (14 %) |
| pass-2 GEMV kernels | 256, 13.259 ms | 128, 7.653 ms |
| `q5_rowsplit_gemm_simt` (non-pass-2) | 0 | **128, 10.44 ms** |

Card 2 is running 128 dispatches per token of `q5_rowsplit_gemm_simt_kernel` at 81.5 us, a kernel
that does not appear on card 1 and does not appear at tp1 at all. Card 1 does the same shapes
through the pass-2 route. **The pass-2 GEMV route is not being taken on rank 1.** Card 2 is the
critical path (27.6 ms busy against a 32.3 ms wall); routing rank 1 through the pass-2 kernels
should take that 10.44 ms down to roughly the 4.1 ms its rank-0 counterpart costs.

The other tp2 number: 14.1 ms/token of summed idle (9.4 on card 1, 4.7 on card 2) is the eager
cross-device synchronisation, and is exactly what the S9 one-graph-per-card design already fixes
(31.9 -> 35.5 t/s). Nothing new to do there.

## Ranked levers

"Class" is (a) launch count - fusable, (b) untuned kernel - rewrite, (c) floor - unfixable.

| # | Item | now ms/tok | class | realistic ms saved | why |
|---|---|---:|---|---:|---|
| 1 | MLP down `q5_gemv_gfx906<5120,17408>` | 9.560 | (b) rewrite | **3.0** | 59 % of floor while the sibling q4 pair hits 92 %; same pass-2 skeleton. The K=17408 x-staging in LDS is the suspect - PASS2-DESIGN.md:124 already flags the 34 KB LDS / 1-block-per-CU fallback |
| 2 | attention `gqa_attention_small_t_simt_partial` | 2.641 | (b) rewrite | **2.2** | runs on 4 blocks / 256 lanes of a 3840-lane GPU; 90x its KV byte floor. Split over the context (more partials) and it becomes a rounding error |
| 3 | attention + GDN input projections (q4/q5 rowsplit) | 5.734 | (b) rewrite | **2.0** | the only two big families never ported to pass 2; at 32-36 % of floor. Target is the 66 % the GDN v/z path already reaches |
| 4 | out-projections `q5_gemv_gfx906<5120,6144>` | 3.633 | (b) rewrite | **1.3** | 56 % of floor, smallest K of the pass-2 shapes, same fix as #1 |
| 5 | GDN v/z `q5_gemv_gfx906<12288,5120>` | 4.520 | (b) rewrite | **1.1** | 66 % of floor; the epilogue (conv + snapshot publish) is doing side-effect work inside the streaming loop |
| 6 | norms - 209 launches/token | 1.686 | (a) fusable | **1.2** | each moves 20 KB in 8 us. Fold into the adjacent GEMV epilogue; the epilogue template already exists |
| 7 | LM head `w8_rowsplit_gemm_simt` | 3.213 | (b) rewrite | **0.8** | 63 % of floor on a 1.27 GB read. A pass-2 style w8 GEMV gets it to ~2.4 ms. Quantising the head below 8-bit would save ~1.5 ms more but changes numerics |
| 8 | GDN recurrent snapshot / gating | 1.266 | (b) rewrite | **0.5** | 48 blocks x 32 threads = 24 waves on 60 CUs; ~5x its state byte floor |
| 9 | MLP gate/up q4 pair | 10.523 | (c) floor | 0.3 | already 92 % of 624 GB/s. Leave it alone |
| 10 | rope, sampling, embed, runtime copies | 0.135 | (c) floor | 0 | 0.3 % of the token |
| 11 | kernel launch / scheduling gaps | 0.143 | (c) floor | 0 | GPU is busy 99.7 % of the token. **Nothing here.** Median gap 0.79 us |
| 12 | CUDA graphs | (-1.31) | (c) keep | 0 | already on; worth 3 %, do not disable |
| - | tp2 rank-1 route (run C only) | 10.44 | (b) bug | **~6 on card 2** | rank 1 falls back to `q5_rowsplit_gemm_simt`; rank 0 does not |

### How many of the ~11 ms are real

If "the ~11 ms" means everything outside the four pass-2 families (43.05 - 28.24 = **14.81 ms**),
then rows 2, 3, 6, 7, 8 and 10 apply and the realistic recovery is **6.7 ms**, taking the token to
~36.3 ms = 27.5 t/s. Of that 14.81 ms, about 3.2 ms is genuinely at the floor
(rope/sampling/copies 0.14, the irreducible part of attention 0.03, the irreducible part of the
LM head 2.04, the irreducible norms and GDN state ~1.0).

If the question is the whole token, adding rows 1, 4 and 5 - which are *inside* the pass-2 set but
sitting at 56-66 % while their q4 sibling reaches 92 % - the recovery is **12.1 ms**, i.e.
43.0 -> ~30.9 ms = **32 t/s at tp1 plain**. That would be a bigger win than the entire tp2 effort
has produced so far, and it comes with none of tp2's correctness risk.

Order I would work in: (1) the q5 pass-2 kernels as one job, since #1, #4 and #5 are the same
kernel family and share the fix - 5.4 ms. (2) attention occupancy - 2.2 ms, a self-contained
kernel with no numerics risk. (3) the un-ported input projections - 2.0 ms. (4) norm fusion -
1.2 ms, but it touches every layer's epilogue, so last.

## Reproduction

Driver: `~/ninfer-work/stage10/profile-overhead/runall.sh`. It builds on the derived image
`ninfer-rocm:6.4.1-prof` = `ninfer-rocm:6.4.1` + `libdw1`, committed once so rocprofv3 loads.

```
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add video --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=<devs> \
  -v ~/ninfer-work/ninfer-gfx906:/src -v ~/ninfer-work:/work -w /src/build-gfx906 \
  ninfer-rocm:6.4.1-prof bash -c 'rocprofv3 --kernel-trace --stats --output-format csv \
    -d /work/stage10/profile-overhead/<NAME> -o <NAME> -- \
    /src/build-gfx906/apps/ninfer /work/qwen3_8_27b.ninfer \
    --prompt "$(cat /work/stage5/p3.txt)" --max-context 2048 --kv-capacity 2048 --max-new 64 <FLAGS>'
```

Per-run flags:

- A  `HIP_VISIBLE_DEVICES=1  --device 0 --greedy --no-thinking`
- A2 `HIP_VISIBLE_DEVICES=1  --device 0 --greedy --no-thinking --no-cuda-graph`
- B  `HIP_VISIBLE_DEVICES=1  --device 0 --greedy --no-thinking --spec mtp --draft-tokens 3 --lm-head-draft`
- C  `HIP_VISIBLE_DEVICES=0,1 --tp 2 --devices 0,1 --greedy --no-thinking --no-cuda-graph`
- D  `HIP_VISIBLE_DEVICES=1  --device 0 --spec mtp --draft-tokens 3 --lm-head-draft --temperature 0.8 --top-p 0.95 --top-k 20`

Analysis: `~/ninfer-work/stage10/profile-overhead/analyze.py <RUN> <tokens-per-step>`. It finds the
decode steps by the `sampling_group_finalize_sample_kernel` dispatches, drops the two warm-up
passes and the prefill pass by inter-marker duration, and reports only the steady-state steps
(A: 64 steps at 43.047 ms, sd 0.47; B: 19 rounds at 98.83, sd 0.88; C: 63 steps at 32.26, sd 0.30;
D: 21 rounds at 98.87, sd 0.83).

Logs and data, all under `~/ninfer-work/stage10/profile-overhead/`:

- `runall.log` - driver log with per-run RAS counts and card temps
- `{A,A2,B,C,D}/run.out`, `{A,A2,B,C,D}/run.err` - CLI output and the summary block
- `{A,A2,B,C,D}/<NAME>_kernel_trace.csv` - raw dispatch trace
- `{A,A2,B,C,D}/<NAME>_kernel_stats.csv` - rocprof per-kernel totals
- `{A,A2,B,C,D}/analysis.txt` - the tables above

## Protocol record

- `sudo systemctl stop llama-swap; sleep 12` -> `inactive`, no llama container present, before any GPU work.
- All GPU work under `nohup` with a polled log; no run needed killing; no sysfs writes; no tp2 graphs.
- `sudo systemctl start llama-swap` -> `active`, `/health` returned **200**.
- `umc_err_count` after all runs: `0000:03:00.0 ue:0 ce:0 de:0`, `0000:09:00.0 ue:0 ce:0 de:0`.
