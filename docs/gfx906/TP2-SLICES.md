# TP2 cherry-pick plan: donor 6a355d5 -> ninfer gfx906-port (f5651be2)

Produced 2026-09-02 by a read-only planning pass. Donor = wamansou/ninfer-tp2-1m, one squash commit on upstream
feaf4dd0, which is also the port's merge-base. Scope decision: TP2 only; YaRN / 1M context deferred (--rope native).
Method: PORT's object db holds feaf4dd0, so `git --git-dir=PORT/.git --work-tree=DONOR diff --stat feaf4dd0` is the
exact donor delta (195 changed, 13,798+/2,086-) plus 53 new files. "PORT-conflict" = PORT also changed the file
since feaf4dd0 (148 files).

## 1. Donor delta, partitioned

### (a) Allreduce transport + execution context (no model logic)

| File | Donor delta | PORT-conflict |
|---|---|---|
| include/ninfer/ops/allreduce.h | new 186 | no |
| src/ops/common/allreduce.cu | new 328 | no |
| src/ops/common/split_launch.h | new 115 | no |
| src/core/device.h / device.cu | +20 / +42 (ExecutionContext) | yes (port added DeviceExecutionView, compute_capability(); different hunks, trivial) |
| src/core/decode_graph.h / .cpp | +68 / +202 (DecodeGraphPeerBridge, fork/join capture) | no |
| src/ops/launcher/kernel_attr_once.h | new 79 | no (consumers conflict, see b) |
| src/CMakeLists.txt | +1 (ops/common/allreduce.cu) | yes (port rewrote HIP build; 1-line add) |
| tests/ops/test_allreduce.cpp, tests/test_device.cpp (+44), tests/test_tensor_slice.cpp | tests | tests/CMakeLists.txt conflict (port +9) |

Shim gaps (all absent from src/core/hip_compat.h): cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess,
cudaPointerGetAttributes, cudaPointerAttributes, cudaMemoryTypeDevice, cudaErrorPeerAccessAlreadyEnabled,
cudaStreamIsCapturing, cudaStreamCaptureStatus, cudaStreamCaptureStatusNone. NINE macros, not three
(cudaMemcpyPeerAsync appears only in comments/tools). hipPointerGetAttributes fills hipPointerAttribute_t whose
field is `.type` in ROCm 6.x (was `.memoryType` before 5.x); verify on the box.

### (b) Shard map, storage layouts, binding, split launch plumbing

| File | Donor delta | PORT-conflict |
|---|---|---|
| src/artifact/storage_layouts.cpp (+315), binder.{h,cpp} (+52/+97), materializer.{h,cpp} (+33/+284), typed_binding.{h,cpp}, reader.h (+47) | slice geometry, tensor_row_slice / tensor_column_slice | no |
| src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp} (+178/+633) | plan_for, shard_mapping_for, Q|K|Gate|V and Q|K|V|Z orders | no |
| src/targets/qwen3_6_27b/impl/variant.{h,cpp} (+110/+361), package.cpp (+32) | *_column_parallel / *_row_parallel per-block forms | yes (port +7/+4, execution_view plumbing) |
| src/targets/qwen3_6_35b_a3b/impl/variant.{h,cpp} (+98/+111), package.cpp | tp=1-only rejection stubs | yes (trivial) |
| src/targets/registry.{h,cpp} (+135) | construct_target(options, ExecutionContext) | no |
| include/ninfer/ops/{linear,linear_add,linear_swiglu,attn_input_proj,gdn_input_proj,gdn_gating_proj}.h | split-form declarations (+98/+58/+109/+135/+147/+50) | gdn_gating_proj.h yes (+11) |
| src/ops/linear/linear.cpp (+127), linear_dispatch.h (new 27) | linear_{column,row}_parallel | no |
| src/ops/wrapper/{linear_add,linear_swiglu,attn_input_proj,gdn_input_proj,gdn_gating_proj}.cpp (+330/+189/+206/+579/+147) | split forms; gdn composed conv-snapshot route | gdn_gating_proj.cpp yes (+27) |
| src/ops/linear/{q4,q5,q6,w8,bf16}_dispatch.cpp (+39/+53/+33/+38/+9) | additive tp2 shard-extent tables | yes, all five (port wraps select_*_a16_launch in gfx906_reroute) |
| src/ops/linear_add/q5/q5_linear_add_plan.{h,cpp} (+8/+38) | shard supports | yes (port +77, pass-2e) |
| src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp (+19), attn_input_proj/q4_q5/*_plan.{h,cpp} (+18/+16) | shard admission | yes (port +79/+65) |
| src/ops/gdn_gating_proj/bf16/*_kernels.{cu,h} (+138/+22) | shard shape | yes (port +355/+87, heavy) |
| src/ops/kernel/gqa_attention_geometry.cuh (+45), src/ops/launcher/gqa_geometry_dispatch.cuh (new 89), gqa_attention_decode.cu (+43), gqa_attention_prefill.cu (+48), gqa_attention_decode{,_bf16,_i8}.cuh, include/ninfer/ops/gqa_attention.h (+39), src/ops/wrapper/gqa_attention.cpp (+7) | Gqa27Tp2Geometry = GqaGeometry<12,2,2>, per-device ensure_func_attr_per_device | yes (port's #if NINFER_GFX906_COMPAT SIMT launch blocks in both launchers, fp16-V in wrapper) |
| src/ops/kernel/mtp_pack.cuh, launcher/mtp_pack.cu, wrapper/mtp_pack.cpp, include/.../mtp_pack.h | mtp_split_attn_in head-local | no |
| src/ops/linear_attention/gated_delta_net/{recurrent.cu,recurrent.cuh,replay.cpp} | head-count runtime | no (port's SIMT forcing is in gated_delta_net.cpp, common.cuh) |
| src/runtime/engine/kv_capacity.{h,cpp} (+22/+17) | per-device pool sizing | no |

### (c) Runtime

| File | Donor delta | PORT-conflict |
|---|---|---|
| src/targets/qwen3_6/impl/runtime/text_context_impl.h | +1570 | yes (port 3 hunks: execution_view(), causal_conv1d_silu_split, device-resident MTP ids) |
| program_impl.h (+744), program.h (+154), mtp_impl.h (+358), text_context.h (+238), schedule.h (+94), graph_impl.h (+36), decode_impl.h, api_impl.h, request_plan_impl.h, speculative_target_impl.h, text_prefill_impl.h, instance.h, dflash_impl.h, export/.../runtime.h | rank-1 program, peer ingress, egress check | no |
| layouts_impl.h (+160), layouts.h (+20), workspace_recipe.h (+93) | per-device KV layout | yes (port: 8x graph allowance, FP16 V plane, gfx9 cc check; workspace_recipe +12) |
| src/runtime/engine/engine.cpp (+152), concurrent_executor.h (+192), include/ninfer/engine.h (+33), include/ninfer/types.h (+81) | resolve_execution_device_ids, debug logit capture | no |
| src/targets/qwen3_6/impl/frontend/frontend.cpp (+101) | mostly yarn/ignore_eos; keep only tp hunks | no |

### (d) CLI / serve
apps/cli/options.{h,cpp} (+15/+98), apps/cli/main.cpp (+39), apps/serve/main.cpp, src/serve/serve_options.{h,cpp}
(+9/+103), request_log.{h,cpp}, generation_service.cpp. No port conflicts. Take --tp, --devices, --ignore-eos;
drop --rope/--yarn-*.

### (e) Tests + tools/tp2
Take: test_allreduce, test_tensor_slice, test_kv_capacity_tp2, test_shard_map, test_sharded_materialization,
test_linear_split, test_linear_add_split, test_linear_swiglu_split, test_attn_input_proj_split,
test_gdn_projections_split, test_gdn_headsplit, test_attention_headlocal, test_output_head_split, test_mtp_split,
test_concurrent_executor, test_cli_options, test_engine_tp2_real, test_engine_mtp_tp2_real, test_graph_tp2,
tools/tp2/parity.cpp, tools/tp2/{p2p,transport,capture}_probe.cu, scripts/tp1-regression.sh (as a template;
regenerate goldens locally). tests/CMakeLists.txt +147 conflicts with port +9.

### (f) DROP
All fp8/ and nvfp4/ files (25 files: *_launch_shard / *_dispatch_shard; the port filters these dirs out of the
build; gfx906_stubs.cpp must gain matching throwing stubs for any _shard symbol referenced from retained wrappers),
yarn_rope.{h,cpp}, test_rope_yarn, test_yarn_rope*, test_engine_yarn_real, test_rope_cli,
tests/core/data/yarn_ref_4x.json, rope.{h,cuh,cu} and wrapper/rope.cpp (all YaRN RopeFrequencyOverride),
tests/ops/test_gqa_attention.cpp -840 refactor + gqa_attention_fixture.h + test_gqa_attention_long_context.cpp
(1M floor criterion), gqa_attention_decode.cuh DecodeSplitScale / gridDim.y=85 tuning hunks, eval/**,
docs/performance.md, README.md, tools/tp2/{build_needle_1m_corpus,summarize_*,dump_yarn_ref,
verify_needle_1m_prompt}.py, tools/bench/* power sampling, test_gated_delta_net_replay_record.cpp (+269, 1M
replay). Vision-at-tp2 rejection stays (one throw).

## 2. Port-specific collision points

1. q4/q5/w8/q6/bf16_dispatch.cpp. Donor splits select_*_a16_launch into _registered (tp1 table) + _tp2_shard_launch
   (additive). Port wraps the same function in gfx906_reroute(select_*(n,k,t), k, t). Adaptation: apply the donor's
   split INSIDE select_*_a16_launch; the port's reroute wraps the combined result unchanged. Shard launches the donor
   picks (launch_q4_simt_r8_c4/c8, gemv_r1_w8_direct, q5_simt_r8_*) are already in kGfx906Reachable* lists, and
   stage-8 tiled kernels are runtime-dimensioned (k % 4 == 0), so shard T>1 rides the tiled path automatically.
   tp=1 selection is byte-identical because the tp1 table is consulted first.
2. Pass-2 GEMVs (q_gemv_gfx906.cuh) are compile-time <kN,kK> with static_assert(kK % 1024 == 0). Route sites gate
   on exact shape: q5_linear_add_gemv.cu:37/48 (5120x6144, 5120x17408), q4_linear_swiglu_gemv.cu:224 (throws unless
   [34816,5120]), q4_q5_gdn_input_conv_snapshot.cu:181 (kValueZRows x kHidden). Shard extents: down-proj K=8704 is
   NOT a multiple of 1024 -> no pass-2 instantiation possible; o-proj/gdn-out K=3072 is fine (<5120,3072>); swiglu
   pair N=17408 needs a second instantiation. Minimal adaptation for S3: the donor's shard routes fall to the generic
   launchers (check q5_rowsplit_gemv_residual_launch_kernel's else branch throws; if so add shard rows falling to
   simt_r8_c4). Instantiate pass-2 at <5120,3072> and swiglu <17408,5120> as a later perf slice (S7), and make the
   q4_linear_swiglu_gemv throw fall through instead.
3. rowsplit_tiled_gemm_gfx906.cuh reads N/K from the Weight: shard-safe, nothing to do.
4. GDN conv-snapshot Materialized route (b81331b). Donor doc 7.2(4): the shard always takes the composed route
   (compose_batched_snapshot in gdn_input_proj.cpp) because fused snapshot kernels bake tp1 rows. Port already
   routes T>1 to Materialized under stage-8 and T=1 through the pass-2 split-output kernel at exact kValueZRows.
   Adaptation: in q4_q5_gdn_input_conv_resolve_plan, shard problems (qk_rows != 4096) return Materialized for all
   T; the port's T=1 fused epilogue stays tp1-only. Donor's gdn_input_proj.cpp (+579) is the wrapper and does not
   touch the port's .cu.
5. fp16-V KV (35a9c5aa). Donor predates it; donor's gqa_attention.cpp +7 and layouts_impl.h assume bf16 V. Port's
   hunks (v_pages.dtype == FP16, FP16 V plane in persistent_layout) must be re-applied on top of the donor's
   per-device layout; Gqa27Tp2Geometry instantiation of the port's SIMT kernels (template <typename Geometry,...>
   in *_gfx906.cuh) is free. Port's kv_heads_for_q_heads already maps 16->2 but must add 12->2.
6. kernel_attr_once.h: port's SIMT decode/prefill branches skip cudaFuncSetAttribute (the #if block returns early),
   so the donor's ensure_func_attr_per_device hunks land only in the #else arms; harmless.
7. Graph allowance: port scales tiers 8x (kGraphAllowanceScale in layouts_impl.h); donor's tp2 capture is ~3x tp1
   nodes (1888 vs 640). Keep 8x; expect tp2 to need it; make it an env knob in S5 if boot aborts on allowance.

## 3. SLICE PLAN

Gate G0 (every slice): build clean; host ctest green; greedy md5s unchanged on a single card (HIP_VISIBLE_DEVICES=1,
card 2): p3 plain 52960e3a..., p4 MTP d3 knob-on 32ba0f7a.... Rollback for every slice: git reset --hard
<slice base>; tag each base tp2-s<N>-base.

S1 - transport + ExecutionContext (60 min). Files: (a) except decode_graph; hip_compat.h +9 macros;
  src/CMakeLists.txt +1; tests/CMakeLists.txt add ninfer_allreduce_test, test_device.cpp (tensor_slice test
  deferred to S2). Conflicts: device.h/.cu (merge both structs). Gate: G0 + ninfer_allreduce_test with both cards
  (HIP_VISIBLE_DEVICES=0,1, llama-swap stopped) + the donor's tools/tp2/{p2p,transport,capture}_probe.cu = the
  ONLY early tp=2 GPU touch, justified because it isolates transport/P2P/UVA/capture semantics on ROCm before any
  model code. Nothing in the engine references ExecutionContext yet, so tp=1 is trivially unchanged.

S2 - shard map + storage slices + binder (75 min). Files: storage_layouts.cpp, binder.*, materializer.*,
  typed_binding.*, reader.h, bindings.{h,cpp}, test_shard_map, test_tensor_slice, test_sharded_materialization,
  test_artifact_materialization (+29), test_load_plan (+9). No port conflicts. Keep the NVFP4 blockscale slice branch
  (host-only, compiles without kernels). Gate: G0 + those host tests.

S3 - split-op plumbing, groupwise formats only (90 min). Files: six include/ninfer/ops/*.h, linear.cpp,
  linear_dispatch.h, split_launch.h consumers, five wrappers, five *_dispatch.cpp, q5_linear_add_plan, q4_q5_*_plan,
  bf16_gdn_gating_proj_kernels, mtp_pack*, gated_delta_net/recurrent*, gdn_gating_proj.h/.cpp; add _shard throwing
  stubs to gfx906_stubs.cpp; collision fixes 1, 2 (fall-through only), 4. Conflicts: all the "yes" rows in (b).
  Gate: G0 + test_linear_split, test_linear_add_split, test_linear_swiglu_split, test_attn_input_proj_split,
  test_gdn_projections_split, test_gdn_headsplit, test_output_head_split, test_mtp_split on two cards; existing
  test_gdn_input_proj_conv_snapshot, test_causal_conv1d_silu, test_target_logprobs green. Split into S3a
  (linear/linear_add/swiglu) and S3b (attn/gdn/gating/mtp_pack) if the gdn_gating_proj_kernels.cu merge is ugly.

S4 - attention head-local geometry + decode_graph peer bridge (60 min). Files: gqa_attention_geometry.cuh,
  gqa_geometry_dispatch.cuh, kernel_attr_once.h, both launchers, three decode .cuh (geometry hunks only, drop
  split-scale tuning), gqa_attention.h, gqa_attention.cpp (add 12->2), decode_graph.{h,cpp},
  test_attention_headlocal, test_kv_capacity_tp2, kv_capacity.*. Conflicts: both launchers, wrapper. Gate: G0 +
  test_gqa_attention (existing), test_attention_headlocal, test_kv_capacity_tp2. Keep the port's
  test_gqa_attention.cpp (+17 fp16-V form); do not take the donor's -840 refactor.

S5 - runtime + engine + CLI (90 min, largest). Files: all of (c) and (d), registry.*, variant.* x2, package.*,
  types.h, frontend.cpp tp hunks, test_concurrent_executor, test_cli_options, test_frontend (+84, tp hunks),
  test_openai_schema / test_anthropic_schema / test_request_log (small). Conflicts: text_context_impl.h (re-apply
  the port's 3 hunks over the donor), layouts_impl.h (keep 8x scale + FP16 V + gfx9 check), workspace_recipe.h,
  variant.cpp. Drop yarn fields from EngineOptions/Options/ServeOptions. Gate: G0 strictly - tp=1 default must
  reproduce both md5s and tg128 23.55 within noise; host ctest. No tp=2 run yet.

S6 - tp2 engine bring-up + parity (90 min box, GPU-heavy). Files: test_engine_tp2_real, test_graph_tp2,
  test_engine_mtp_tp2_real, tools/tp2/parity.cpp + ctest wiring, capture_probe.cu. Gate: section 4. Rollback keeps S5.

S7 (optional perf) - pass-2 shard instantiations <5120,3072> residual, <17408,5120> swiglu pair; gate = tp2 tg128
  improves, tp1 md5s unchanged.

## 4. TP2 acceptance on the MI50 pair

Preconditions: llama-swap stopped, HIP_VISIBLE_DEVICES=0,1, no other GPU processes, artifact env var set. Order:
1. ninfer_allreduce_test (S1) then tools/tp2/p2p_probe / transport_probe / capture_probe for measured P2P + capture
   facts.
2. ninfer_qwen3_8_27b_tp2_real_test (skips 77 if <2 devices, TIMEOUT 1800), ninfer_qwen3_8_27b_graph_tp2_test.
3. ninfer_qwen3_8_27b_tp2_parity_test (TIMEOUT 5400; env NINFER_TP2_PARITY_KV, _OUTPUT, _DUMP_DIR). Bar is
   comparative: tp2 inside tp1's own prefill-chunk perturbation envelope on argmax / KL / (1-cos) - donor margins
   KL 0.0157 vs 0.0714 budget. Our tp1 control is the port's own binary; no CUDA goldens involved.
4. ninfer_qwen3_8_27b_mtp_tp2_real_test (TIMEOUT 3600).
5. Serve: --tp 2 --devices 0,1 --kv-dtype bf16, probe v3 at 30K; then tg128 and MTP d3 vs 23.55 / 36.7 single-card.
   Expect the tp2 decode gain to be smaller than the donor's 1.44x: ROCm allreduce latency per call is unmeasured
   and there are 128 reduces per token.

bf16 KV at tp2/40K: KV per device = 16 layers x 2 heads x 256 x (2 B K + 2 B fp16 V) = 16 KiB/token -> 640 MiB at
40K per card. INT8 KV was a 1M-capacity requirement only (donor 4.1); nothing in the split path depends on the KV
dtype. Fine.

## 5. Open questions and risks

- hipGraph capture of cross-device hipMemcpyAsync D2D (ROCm 6.4.1). The donor relies on UVA D2D being capturable
  and on a single fork/join capture across two devices with cudaStreamWaitEvent on a peer-device stream. HIP has
  historically rejected cross-device event waits INSIDE capture (hipErrorStreamCaptureUnsupported / ...Isolation).
  This is the single biggest risk; capture_probe.cu in S1 answers it before S5 is written. Fallback: eager tp2 (the
  donor keeps it as the identity reference) - costs the ~15% capture gain.
- P2P present here: enable_peer_access (allreduce.cu:112-135) probes both directions and enables; pull_peer
  (:53-63) is a plain cudaMemcpyAsync D2D that the driver routes directly when peer access is enabled - no code
  path change, but MI50 large-BAR P2P must be on (rocm-smi --showtopo, HSA_ENABLE_SDMA), else it is host-staged like
  the donor's 5090s.
- Graph node budget: donor's tp2 graph is 1888 nodes; the port's 8x allowance was sized against ROCm's per-node
  overhead at tp1 (36-51 MiB observed vs 12 MiB tier). tp2 could exceed 8x; make the scale an env knob in S5 if
  boot aborts with the allowance error.
- hipPointerAttribute_t field naming and hipErrorPeerAccessAlreadyEnabled existence in ROCm 6.4 headers - confirm
  in S1.
- q5_rowsplit_gemv_residual_launch and q4_linear_swiglu_gemv_pair_launch throw on non-exact shapes; the donor's
  shard registrations for Q5 linear_add rely on Split2Exact / simt routes for T=1 shards - verify q5_linear_add_plan
  kSupports gains {5120,3072,...} / {5120,8704,...} with a non-GEMV T=1 schedule under the port's kK*Routes tables.
- Vision-at-tp2 rejection, DFlash rejection: keep as in the donor (one throw each in engine.cpp).

Critical files: PORT src/core/hip_compat.h; DONOR src/ops/common/allreduce.cu; PORT src/ops/linear/q5/q5_dispatch.cpp
(pattern for all five dispatch merges); DONOR src/targets/qwen3_6/impl/runtime/text_context_impl.h; PORT
src/targets/qwen3_6/impl/runtime/layouts_impl.h.

S8 (diagnosis, 2026-09-02) - why the dual-device graph replays at 0.21 t/s. tools/tp2/replay_probe.cu (ninfer_tp2_replay_probe)
   times two-device graphs of 4..1500 nodes in seven shapes: every shape replays at 1-10 us per node, so neither per-node
   host serialisation, host-staged memcpy nodes nor a fixed launch cost explains 4.7 s. NINFER_GFX906_GRAPH_TRACE=1
   (src/core/decode_graph.cpp) on the real graph: 2000 nodes (1737 kernel, 263 memcpy), hipGraphLaunch 2-3 ms, device
   4.72 s. rocprofv3 of the same run: during replay EVERY dispatch is on device 0, on two queues; rank 1's GEMVs take
   100-300x longer there (q4_swiglu_pair_gemv 88 us vs 28 ms) because they read rank 1's weights over PCIe through the
   UVA peer mapping. ROCm 6.4.1's graph executor creates its parallel-branch streams on the LAUNCH device and ignores
   the nodes' capture device; output is correct, time is PCIe bandwidth. The probe shows the same agent split under
   rocprof (dual-* shapes 99 % on device 0), invisible at 16 KB working sets. Consequence: the single-graph
   DecodeGraphPeerBridge design cannot work on this runtime; two captures bridged by events are rejected by HIP
   (the cross-capture wait is accepted, the next cross-device memcpy/kernel is hipErrorInvalidValue, the stream is
   poisoned). Fix design for S9: two per-device graphs launched on their own streams, all-reduce synchronised on device
   memory (signal kernel P2P-writes a sequence into the peer's flag, spin-wait kernel, peer-read copy kernel, combine),
   knob NINFER_GFX906_TP2_FLAG_SYNC. Until then tp2 runs eager (--no-cuda-graph, 31.93 t/s); treat tp2 graphs-on as
   broken on ROCm 6.4.1. Full note: ~/ninfer-work/stage10/tp2-s8-diagnosis.md; RESULT TP2-S8 in ~/EXPERIMENTS.md.

S9 (flag-synced split graphs, 2026-09-02) - tp2 graphs-on works: md5 = tp1 = eager, tg128 35.53 t/s vs eager 31.75.
   Design (S8 write-up + the mxxm llama.cpp fork's tp-allreduce.cu idiom, b10811): both device streams capture at once,
   each into its OWN hipGraph (DecodeGraphSplitCapture; no cross-capture edge), launched back to back on their own
   streams by one thread (DecodeGraphExecutable::launch). The collectives (ops::allreduce_sum, allgather_rows) become
   one kernel per rank behind NINFER_GFX906_TP2_FLAG_SYNC=1: each block PCIe-writes its slice into the PEER's
   hipDeviceMallocUncached staging, thread 0 stores a per-block sequence (RELEASE/SYSTEM) into the PEER's uncached
   signal block and polls its OWN (ACQUIRE/SYSTEM + s_sleep) - the spin never crosses PCIe - then the combine reads
   local staging and a second barrier protects it from the next call. Only CAPTURED calls use it; prefill and eager
   decode keep the event transport (prefill's 5 MB all-reduces are faster as SDMA copies: pp512 339 vs 253 tok/s).
   Memory types are the whole story on gfx906 (tools/tp2/replay_probe.cu split-flag shape, CLI [signal] [staging]):
   uncached+uncached replays == eager through 20 000 replays; fine-grained signal blocks DEADLOCK (local poll served
   from L2); plain-cudaMalloc staging returns stale partials (DIFFER at 5 rounds); the first design (poll the peer's
   flag via UVA, peer-READ data) deadlocks or reads stale data because remote VRAM is mapped cacheable and gfx906 has
   no L2 writeback instruction. Probe cost 16 us/round (4 nodes) vs eager 38 and dual-memcpy 62 (which ran on one
   device); long run 100 x 20 000 replays: both cards 96-99 % busy, 57-58 W, == eager. Runtime gates: tp2 graphs-on
   p3 md5 52960e3a58e9ef2002021c8cd3904855 (35.5 tok/s in apps/ninfer; S8: 0.21), ninfer_bench tg128 graphs-on
   35.53 vs eager 31.75 (+12.7 %), tp1 p3 md5 unchanged. NOT delivered: the "cooler cards" prediction -
   both cards read 100 % busy at ~145 W with graphs on exactly as eager; decode is bandwidth-bound continuously and a
   spinning wave counts as busy. Remaining: promote the knob + graphs as the tp2 default (parity test, MTP family),
   h2h vs production, surface FlagSignal::status through the graph-trace knob, size the 16 MB staging from the plan.
   Commits 7317d9c2 (probe), 99bd9f9d (runtime), 7c02856c (capture-only + docs); RESULT TP2-S9 in ~/EXPERIMENTS.md.

S9b (graphs-on gates + flag-sync default, 2026-09-02) - the flag-sync transport is now the tp2 DEFAULT.
   Gates with NINFER_GFX906_TP2_FLAG_SYNC=1 (all two-card, stage10/tp2-s9b-*.log): parity test with phase F ON
   (128 positions x 5 prompts, 1511 s): gate PASS, KL 0.000451 vs budget 0.00181, argmax 7/7, (1-cos) 7.87e-5 vs
   8.53e-5 (byte-for-byte the S6 eager numbers: phases C/D are eager), corpus argmax 636/640, and phase F "tp2
   graphs-on vs tp2 eager" EXACT on every prompt (193/411/160/512/512 tokens). ninfer_qwen3_8_27b_graph_tp2_test OK
   (48 s: graphs == eager over 128 steps, reproducible across two graph runs). MTP at tp2 graphs-on, apps/ninfer:
   p5 code d3 200-new 46.11 tok/s @ 94.31 % (md5 c56fa74a60c73835 = eager = S6; eager tp2 37.3, tp1 36.7), p4
   prose d2 128-new 25.38 tok/s @ 55 % (md5 1a2308484a9c3797 = the plain 128-new text; eager 25.2, tp1 22.2);
   ninfer_bench --mtp-draft-tokens 3 -p 512 -n 128 -r 3: decode_output_tok_s 22.46 +- 0.04 @ acceptance 29.2 %
   (decode_path mtp_cuda_graph; tp1 pass 2e 18.00 @ 30.2 %), pp512 338.2.
   Two donor tests FAIL for reasons the graph does not own: ninfer_qwen3_8_27b_tp2_real_test "tp1 and tp2 greedy
   output diverge after 12 of 32 tokens" with graphs ON and IDENTICALLY with NINFER_TP2_TEST_GRAPHS=0 (eager, 60 s
   each) - a synthetic-prompt + Int8Group64-KV split divergence, the same verdict the S6 note anticipated;
   ninfer_qwen3_8_27b_mtp_tp2_real_test probe A passes in full (oracle 64/64 both widths, acceptance 0.90 = tp1)
   but probe B fails leg 1 ("tp2 MTP3 tracks tp1 MTP3 for 3 of 64 tokens, worse than the ordinary path's 18"): the
   ordinary split itself leaves tp1 at token 18 on that prompt, so the comparison is between two diverged texts (the
   donor's own comment on probe B records the same knife edge at token 12). The test now honours
   NINFER_TP2_TEST_GRAPHS=0 like its sibling so the eager leg can be run; see RESULT TP2-S9b for that run.
   Default flip: allreduce.cu flag_sync_requested() is true unless NINFER_GFX906_TP2_FLAG_SYNC=0 (which restores
   the S8 event-bridged single graph; --no-cuda-graph is still eager). Post-flip gates with NO env: tp2 p3 md5
   52960e3a58e9ef2002021c8cd3904855 (33.98 tok/s, transport banner "tp2 default"), tp1 p3 on card 2 md5 unchanged
   (23.13 tok/s), tp1 p4 MTP d3 on card 2 md5 32ba0f7a69b8da571d60ce25f6dbb3aa unchanged (20.26 tok/s, 37.85 %).
   RESULT TP2-S9b in ~/EXPERIMENTS.md.
