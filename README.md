# NInfer-gfx906

AMD Instinct MI50/MI60 (gfx906, ROCm) port of [NInfer](https://github.com/Neroued/ninfer).

## Status: closed (2026-09-04)

The port works. It runs Qwen3.8-27B on one MI50 or split across two, and the two-card split
passes the donor parity test. But on this hardware it is slower than llama.cpp, and profiling
shows no path to beating it, so I have stopped work. The code, logs and profile stay here for
anyone who wants them.

## Use these instead

On MI50 / gfx906:

- [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp): a llama.cpp fork tuned for
  gfx906, with tensor split across cards and MTP drafts. My two-MI50 server runs it.
- [Niko1221/Strata](https://github.com/Niko1221/Strata): an engine for Qwen3.8-Flash-Next (MoE).
  I ran v0.1.40.1 on two MI50s; the gfx906 build needed a few small compile fixes.
- [Llaminar/llaminar](https://github.com/Llaminar/llaminar): a C++ engine with ROCm tensor
  parallel across cards, gfx906 included. Two-card dense Qwen3.8-27B numbers are in
  [PR #20](https://github.com/Llaminar/llaminar/pull/20).
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906): the hand-written
  gfx906 kernels this port borrowed from.

On NVIDIA:

- [Neroued/ninfer](https://github.com/Neroued/ninfer): upstream, RTX 5090.
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090): RTX 3090.
- [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100): V100.

## Final numbers

Qwen3.8-27B `groupwise-int` (4-bit), MI50 32 GB, ROCm 6.4.1. CLI runs, greedy, thinking off.
Decode in tokens/s; prefill on a 512-token prompt.

| Setup | Plain decode | MTP decode, code | MTP decode, prose | Prefill |
|---|---:|---:|---:|---:|
| One card | 23.3 | 36.7 | 22.2 | 211 |
| Two cards | 31.9 | 37.3 | 25.2 | 340 |
| Two cards, graph replay (opt-in) | 35.5 | 46.1 | 25.4 | 338 |

Served side by side on the same box: the same requests, 1,024 new tokens each, sampling and
thinking on, MTP drafts.

| Engine | Essay | Code |
|---|---:|---:|
| This port, two cards, graph replay | 30.9 | 30.4 |
| mx-llama.cpp, two cards, Q8_0 | 46.6 | 58.7 |

The port reached 0.52-0.66x of llama.cpp's speed while reading about half the bytes per token.

## Why I stopped

1. **There is no overhead left to cut.** A rocprofv3 trace
   ([PROFILE-REPORT.md](docs/gfx906/PROFILE-REPORT.md)) shows the GPU busy for 99.7% of each
   token. The time goes to matrix kernels, not to kernel launches.
2. **llama.cpp is already near the memory limit.** Each token reads every weight once, so decode
   speed is set by memory bandwidth. This port's best kernel reads at 92% of the card's measured
   624 GB/s; the rest run at 32-66%. mx-llama.cpp's kernels already run at 90% or more, so more
   tuning could at best draw level. At 8 bits, which my server uses for quality, the port would
   read twice the bytes; with every kernel at 92% I estimate about 33 tokens/s on two cards,
   against llama.cpp's 42.
3. **NInfer's big gains need matrix cores.** On an RTX 5090, most of NInfer's lead comes from
   speculative decoding: checking 4-8 drafted tokens costs about 1.5x one plain token, because
   the tensor cores absorb the extra math. gfx906 has no matrix cores, so a 4-token check here
   costs 2.3x a plain token. Deep drafts such as DFlash cannot pay off, and FP4/FP8 weights have
   no hardware path.

## What may be useful

- [PASS2-DESIGN.md](docs/gfx906/PASS2-DESIGN.md): a one-token matrix-vector kernel
  (register-resident, fp16 dot2, DPP reductions) that reaches 92% of the bandwidth limit on the
  MLP gate/up pair.
- Two-card graph replay on ROCm ([TP2-SLICES.md](docs/gfx906/TP2-SLICES.md), slice 9): ROCm's
  graph executor runs every node on the launch device, so one graph spanning two cards replays
  at 0.2 tokens/s. One graph per card with an in-graph flag barrier fixes it (+12.7% over
  eager).
- [PROFILE-REPORT.md](docs/gfx906/PROFILE-REPORT.md): per-kernel costs and a ranked list of the
  remaining gaps.
- The two-card split passes the donor parity test (KL 0.00045 against a 0.0018 budget) and a
  30K-token tool-call probe (7/7).

## Open problems

If you pick this up:

- With two cards, card 2 takes an untuned fallback kernel (`q5_rowsplit_gemm_simt`, 10.4 ms per
  token) where card 1 takes the tuned one. A fix should lift two-card eager decode from about 32
  to about 40 tokens/s.
- The q5 kernels (MLP down, output projections, GDN v/z) run at 56-66% of the bandwidth limit,
  and the attention kernel runs on 4 of 60 compute units.
- Two-card graph replay (`NINFER_GFX906_TP2_FLAG_SYNC=1`) is opt-in. Card 2 hung under
  `ninfer-serve` with it on. The crash bundle later showed an uncorrectable HBM error on that
  card, so the card was the likely cause, but nobody has tested it again.
- Two donor tests (`tp2_real_test`, MTP probe B) drift from the one-card output after 12-18
  tokens on synthetic prompts with int8 KV, with and without graph replay.

Findings that apply to llama.cpp went to mx-llama.cpp issues
[#7](https://github.com/mxxm-t/mx-llama.cpp/issues/7),
[#8](https://github.com/mxxm-t/mx-llama.cpp/issues/8),
[#9](https://github.com/mxxm-t/mx-llama.cpp/issues/9) and
[#10](https://github.com/mxxm-t/mx-llama.cpp/issues/10).

## Supported model

One model: the **Qwen3.8-27B `groupwise-int`** artifact in the v2 format (~17 GB, fits a 32 GB
card). Upstream has since moved to v3 artifacts, which this port cannot load, so pin the old
revision:

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer \
  --revision 18dfc887423fa5aabf3cb56fac41490e462b3fab --local-dir models
```

The file is 18,210,531,328 bytes, sha256
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.
Build recipe: [STAGE1-LOG.md](docs/gfx906/STAGE1-LOG.md). Run recipe (ROCm container flags,
CLI): [STAGE4-LOG.md](docs/gfx906/STAGE4-LOG.md).

*Naming note:* inside NInfer, Qwen3.8-27B is a weights profile of the code target named
`qwen3_6_27b`. The two generations share the same geometry, so the source tree says "qwen3_6"
even when it runs the 3.8 model.

## Logs

- [PORT-AUDIT.md](docs/gfx906/PORT-AUDIT.md): kernel census, port order, donor map
- [STAGE1](docs/gfx906/STAGE1-LOG.md) · [STAGE3](docs/gfx906/STAGE3-LOG.md) ·
  [STAGE4](docs/gfx906/STAGE4-LOG.md) · [STAGE5-7](docs/gfx906/STAGE5-7-LOG.md) ·
  [STAGE8-9](docs/gfx906/STAGE8-9-LOG.md) · [STAGE10](docs/gfx906/STAGE10-LOG.md): stage logs
- [TP2-SLICES.md](docs/gfx906/TP2-SLICES.md) and
  [TP2-S9B-RESULT.md](docs/gfx906/TP2-S9B-RESULT.md): the two-card port
- [PROFILE-REPORT.md](docs/gfx906/PROFILE-REPORT.md): where each token's time goes

## Branches

- **`gfx906-port`** (default): the port.
- `lookup-port`: context-lookup drafts ported from ninfer-v100, off by default behind
  `NINFER_GFX906_MTP_LOOKUP=1`. Never benchmarked.
- `master`: upstream at the fork point (2026-08-20).

## Credits

Upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) · the
[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) down-port playbook ·
[llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) donor kernels ·
[wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) tensor-parallel donor.
Apache-2.0, same as upstream.
