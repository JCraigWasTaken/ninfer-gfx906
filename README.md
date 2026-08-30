# NInfer-gfx906

AMD Instinct MI50/MI60 (gfx906, ROCm) port of [NInfer](https://github.com/Neroued/ninfer) —
the first AMD target in the NInfer ecosystem.

## Status

**Stages 1–3 complete: the full tree compiles and links clean for gfx906**
(ROCm 6.4.1, `[219/219]`, from-scratch verified) with all execution paths
rerouted to tensor-core-free kernels. **Not yet hardware-validated** — no
performance numbers exist for this port yet. Next milestone: first end-to-end
token (blocked on the GQA prefill attention rewrite + real-hardware
iteration).

- [`docs/gfx906/PORT-AUDIT.md`](docs/gfx906/PORT-AUDIT.md) — kernel census, port order, donor map
- [`docs/gfx906/STAGE1-LOG.md`](docs/gfx906/STAGE1-LOG.md) · [`STAGE3-LOG.md`](docs/gfx906/STAGE3-LOG.md) — build logs

Contributions welcome.

## Supported model

One model: the **Qwen3.8-27B `groupwise-int`** artifact (~17GB — fits a 32GB
card). Other upstream artifacts are not part of this port.

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
