# NInfer-gfx906

AMD Instinct MI50/MI60 (gfx906, ROCm) port of [NInfer](https://github.com/Neroued/ninfer) —
the first AMD target in the NInfer ecosystem.

## Status

**Stages 1–7 complete: runs end-to-end on a real MI50** — first tokens
validated against llama.cpp on the same card (stage 4), HIP graphs on with
bit-identical greedy output (5), MTP speculative decoding working at 40–95%
acceptance (6), and a first int8-KV attention path via `v_dot4_i32_i8` (7).
Throughput is bring-up grade — the grouped-int prefill GEMM rewrites and
MI50 retuning (steps 8–10) are next.

- [`docs/gfx906/PORT-AUDIT.md`](docs/gfx906/PORT-AUDIT.md) — kernel census, port order, donor map
- [`docs/gfx906/STAGE1-LOG.md`](docs/gfx906/STAGE1-LOG.md) · [`STAGE3-LOG.md`](docs/gfx906/STAGE3-LOG.md) · [`STAGE4-LOG.md`](docs/gfx906/STAGE4-LOG.md) · [`STAGE5-7-LOG.md`](docs/gfx906/STAGE5-7-LOG.md) — stage logs

Contributions welcome.

## Supported model

One model: the **Qwen3.8-27B `groupwise-int`** artifact (~17GB — fits a 32GB
card). Other upstream artifacts are not part of this port.

Download: [`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
→ `qwen3_8_27b.ninfer` (18,210,531,328 bytes,
sha256 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`).
Build recipe: [`docs/gfx906/STAGE1-LOG.md`](docs/gfx906/STAGE1-LOG.md); run
recipe (ROCm container flags, CLI): [`docs/gfx906/STAGE4-LOG.md`](docs/gfx906/STAGE4-LOG.md).

*Naming note:* inside NInfer, Qwen3.8-27B is a weights-profile of the code
target named `qwen3_6_27b` — the two generations share identical geometry, so
upstream serves both from one engine target. That's why the source tree says
"qwen3_6" even when running the 3.8 model.

## Branches

- **`gfx906-port`** (default) — the port; all work happens here
- `master` — untouched upstream mirror at the fork point

## Engine documentation

For what NInfer is, how artifacts/serving/CLI work, and upstream's benchmark
results (measured on RTX 5090 — **not** applicable to this port), see the
[upstream README](https://github.com/Neroued/ninfer#readme) and
[upstream docs](https://github.com/Neroued/ninfer/tree/master/docs).

## Credits

Upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) · the
[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) down-port playbook ·
[llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) donor
kernels. Apache-2.0, same as upstream.
