# gfx906 port — ninfer-serve first-request crash debug (2026-08-30)

Status: **root-caused and validated** (2026-08-30 evening session). First
hardware bring-up of `ninfer-serve` on the MI50 box (`forgev1`, ROCm 6.4.1
container, single 32GB MI50). The CLI was validated through stage 8
([STAGE8-9-LOG.md](STAGE8-9-LOG.md)); serve had never run on hardware.

## Incident (afternoon session, logs lost)

`ninfer-serve` (flags: `--host 0.0.0.0 --port 8081 --model-id qwen38
--device 0 --kv-dtype bf16 --spec mtp --draft-tokens 2 --lm-head-draft
--max-concurrency 1 --default-max-tokens 1200`) loaded the model, answered
`/v1/models`, then on the first `/v1/chat/completions` request closed the
connection with no response. The GPU ended up wedged (sysfs reads EPERM)
and needed a cold power cycle. The serve container was run `--rm`, so its
logs were lost.

## Root cause: uncorrectable HBM ECC error, then the known MI50 failed-reset wedge

The host kernel journal from the incident boot survived
(`journalctl -b -1`; full extract saved to
`~/ninfer-work/serve-debug/incident-2026-08-30-kernel.log`). The first GPU
event of the request window is a RAS hardware-error interrupt, not any
software fault:

```
16:32:32 amdgpu 0000:08:00.0: {1}uncorrectable hardware error(ERREVENT_ATHUB_INTERRUPT) detected!
16:32:32 amdgpu 0000:08:00.0: {1}1 correctable hardware errors detected in umc block
16:32:32 amdgpu 0000:08:00.0: {1}1 uncorrectable hardware errors detected in umc block
16:32:32 amdgpu: GPU reset begin!
...
16:32:49 amdgpu: BACO reset
16:32:54 amdgpu: PSP create ring failed! / PSP resume failed
16:32:54 amdgpu: GPU reset(2) failed ... ret = -62
```

Reading of the evidence:

- **The wedge trigger was an uncorrectable ECC error in the UMC (HBM2
  memory controller)** — one correctable plus one uncorrectable error at
  the same instant, i.e. a multi-bit flip in one HBM word. ECC syndromes
  are generated in hardware on write; no kernel bug can fabricate one. A
  software illegal access would instead log a `VM_L2_PROTECTION_FAULT` or
  a ring timeout — **neither appears anywhere before the RAS interrupt.**
- The RAS interrupt forces a GPU reset; on this card BACO reset completes
  but **PSP (firmware processor) resume fails** (`ret = -62`), which is
  the known MI50 wedge class: the driver gives up, every subsequent sysfs
  read returns EPERM, and only a cold power cycle recovers it (same
  terminal state as the 2026-08-15 DPM-write incident).
- The serve process died as downstream fallout (its HIP calls failed once
  the reset began), which is why the client saw the connection close with
  no response.
- MEM ECC is active on this card (`amdgpu: MEM ECC is active` at boot).
  This boot's RAS counters are clean (`umc_err_count: ue 0 ce 0 de 0`) and
  `gpu_vram_bad_pages` is empty — the failed reset prevented bad-page
  retirement, so the marginal address is *not* quarantined. If UMC UEs
  recur, that is the used card's HBM degrading; watch
  `/sys/class/drm/card0/device/ras/umc_err_count` (card0 = 0000:08:00.0).

Serve-layer software was investigated first (details below) and is
exonerated for the wedge; the timeline pins it on the hardware event.

### Non-causal finding: `init_user_pages` storms at every large model load

The incident window also shows two ~10 s and ~19 s bursts of
`amdgpu: init_user_pages: Failed to get user pages: -1` (~740/s) at
16:20:35 and 16:26:35 — the two serve load attempts. These looked like the
stage-4 incident-2 precursor, but the identical storm fired at 17:30:10 on
the current boot, which is just the **llama container's own auto-start
load**: pinning multi-GiB staging while an ~16-18 GiB model read fills the
page cache on a 30 GiB host. It recurs at every big load, the loads
complete anyway, and it precedes no fault here — environmental noise, not
a cause. The debug session's launcher still avoids it (idle-wait +
`drop_caches` + a MemAvailable >= 24 GiB gate before load,
`~/ninfer-work/serve-debug/serve_session.sh`), which produced a clean
22.8 s load with zero storm lines.

### Secondary finding: the crashed binary was stale (fixed)

`build-gfx906/apps/ninfer-serve` was last linked 2026-08-30 00:00 — before
the final stage-8 commits landed (`f758bfa`/`227ca65` 15:46, `b81331b`
16:06). The incident binary therefore contained a mid-flight stage-8 ops
tree (potentially including the pre-fix tiled kernels from stage-8 bug #1).
That cannot cause an ECC UE, but it disqualified the binary for
validation; `ninfer` and `ninfer-serve` were rebuilt from `dc9696f`
before any validation run (relink only — all op objects were already
current with the CLI binary).

## Serve-layer code review (pre-reproduction static pass)

Paths serve exercises that no CLI stage ever ran, reviewed before touching
the GPU:

- **Stochastic sampling** (all CLI stages ran `--greedy`): serve defaults
  to the model thinking preset (temp 1.0, top-k 20, top-p 0.95). The
  multi-block sampler (`sampling_partial_topk_kernel` /
  `sampling_group_finalize_sample_kernel`) and the stochastic speculative
  accept pipeline (`speculative_sampling_partial_topk_kernel` /
  `speculative_sampling_group_finalize_kernel`) had never executed on
  gfx906. Static review: wave64-clean (logical 32-lane shuffle shims,
  hipcub `BlockMergeSort`, last-arriving-block merges — no spin-waits,
  ~42 KiB worst-case static LDS < 64 KiB).
- **Thinking mode** (CLI ran `--no-thinking`): template/preset change,
  host-side.
- **Prefix reuse** (serve default on; CLI is one-shot per process): engine
  reuse path; the warmup request populates the cache.
- `warmup()` swallows failures (`warmup failed (continuing)`), so a broken
  warmup would still answer `/v1/models` — reviewed as a possible
  masking point, but the validated timeline (UE at request time) made it
  moot.
- KV sizing: serve defaults resolved to `max_context 8192 / kv-capacity
  explicit 8192` — measured startup: runtime 1.60 GiB, 14.14 GiB free
  after startup, graphs 52 MiB observed / 656 MiB allowance. No memory
  pressure on the 32 GB card.

No serve-layer code change was needed; all of the above ran correctly in
validation (below).

## Validation (rebuilt binary @ dc9696f, original full flag set)

Serve container (kept, logs saved): recipe as prior stages plus
`--ulimit memlock=-1 --ulimit core=-1 -p 8081:8081`,
`--request-log-jsonl` enabled. Load 22.8 s clean, warmup OK, `/health`
200, KV line:

```
KV capacity explicit resolved=8192 tokens pages=128/128 runtime=1.60 GiB
free-after-weights=15.15 GiB free-after-startup=14.14 GiB slack=13.55 GiB
graphs=52.00 MiB/656.00 MiB
```

**10 sequential `/v1/chat/completions`** (mixed prose/code, max_tokens
512, default sampling = thinking preset temp 1.0, MTP d2 + lm-head-draft
active — request log confirms `proposal_head: "optimized"`, backend mtp,
draft_window 2): all 10 returned HTTP 200 with a finish reason and
coherent, correct content; UMC RAS counters remained `ue 0 ce 0 de 0`
throughout; server stayed up. One request (#10, an LRU-cache prompt)
returned `finish_reason=length` with empty *content* because all 512
tokens were spent in thinking-mode reasoning — correct behavior, not a
fault. Transcript: `~/ninfer-work/serve-debug/validate-d2-run1.log`.

**Sustain runs** (`~/ninfer_sustain.py`, 8 back-to-back completions each,
temp 0.8, cooldown to mem <= 45C before each):

<!-- SUSTAIN RESULTS -->

## End state

- llama container restarted and healthy after the session.
- No GPU-wedge events during the debug session; RAS counters clean.
- Serve container stopped; full `docker logs` saved to
  `~/ninfer-work/serve-debug/`.

## Operational guidance going forward

1. **A UMC UE means cold power cycle** — the MI50 cannot warm-reset out of
   it (PSP resume fails). Do not warm-reboot-loop.
2. Run serve containers **without `--rm`** until the port is mature, and
   with `--request-log-jsonl`; the incident cost its own evidence.
3. Rebuild `ninfer-serve` whenever ops sources change — the CLI relink
   does not refresh it.
4. Watch `ras/umc_err_count` across sessions; recurring UEs = the used
   card's HBM is degrading (page retirement is not persisting because the
   post-UE reset fails before the EEPROM write).
5. The `init_user_pages` storm at load is benign but avoidable
   (`serve_session.sh` gate).
