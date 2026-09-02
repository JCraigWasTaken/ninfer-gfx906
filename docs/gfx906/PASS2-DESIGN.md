# Pass 2: T=1 GEMV retune for gfx906 — design brief

Produced 2026-09-02 by a read-only planning pass over the laptop clone at 52e3c864 and the llama.cpp-gfx906 donor.
Profile inputs: rocprofv3 on card 2, decode 64 tokens, HIP graphs on; decode is 100 % GPU-busy; four T=1 GEMV
kernels = 66 of ~79 ms/token at 20-35 % of the ~700-810 GB/s the MI50 sustains.

## 1. Where the four kernels live and how they work

All four are upstream CUDA kernels compiled under compat, not port-specific. Stage 8 only replaced T>=2 routes
(docs/gfx906/STAGE8-9-LOG.md:43-56: "T=1 GEMV" is the "Fallback kept" in every row).

(a) q4_linear_swiglu_gemv_pair_kernel — src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu:114-201, launched at
:210-213 with grid = 17408/4 = 4352, 128 threads (kWarpsPerBlock=4, :29-31). One logical 32-lane warp owns one
gate/up row pair (:133). K loop: 5 tiles of 16 groups (kTiles = 80/16, :33), each tile staged through LDS by
pipe_copy<16> per lane (:101-104, 512 B gate + 512 B up per warp-tile) with a 3-stage ring (:118-122). Consume
(:175-194): each lane reads one byte from LDS per group (:182), sign-extends two nibbles, converts each to fp32,
multiplies by scale, and does 4 scalar fmaf against a bf16x2 x pair from LDS (:189-193). Reduce: warp_reduce_sum =
5-step __shfl_down width-32 ladder (src/ops/common/warp.cuh:26-33). Epilogue: lane 0 writes silu(gate)*up (:200).
Routed via q4_linear_swiglu_plan.cpp:41 ({{1,1}, GemvPair}).

(b)/(c)/(d) q5_rowsplit_gemv_kernel — src/ops/linear/q5/q5_rowsplit_gemv.cuh:107-200. 16 logical warps per block
(kRowsPerBlock=16, 512 threads), one warp per row (:148-150), grid kN/16 (:209). Tile = 16 groups: 32 uint4 nibbles
+ 8 uint4 high + 2 uint4 scales via pipe_copy (:44-50), 2-stage ring. Consume :69-80: byte-per-lane nibble + 2-bit
high extraction (:71-74), scalar fp32 fmaf. Instantiations:
- MLP down: src/ops/linear_add/q5/q5_linear_add_gemv.cu:25 <5120, 17408, 16, 2, kStageX=false> + kResidual=true
  (residual add at :193-195, x read from global because 34 KB x won't fit).
- attn/GDN out-proj: q5_linear_add_gemv.cu:22 <5120, 6144, 16, 2, true> + residual.
- GDN value/z: src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu:175-183 <12288, 5120, 16, 2, true,
  false, SplitOutput=true, kSplitRow=6144, Q5GdnDecodeEpilogue>. Epilogue (:92-106): rows < 6144 go into
  GdnConvEpilogue::store (4-tap causal conv + silu + snapshot publish, gdn_conv.cuh:66-116), rows >= 6144 are plain
  bf16 stores to z.

(e) w8_rowsplit_gemm_simt_kernel (output head, 248320x5120, w8_dispatch.cpp:74 -> launch_w8_small_t rerouted to
simt_r8_c4): 1.27 GB / 3143 us = 404 GB/s already (~55 % of achievable). Not a pass-2 target.

## 2. Why they are slow on gfx906 (ranked)

1. The cp.async pipeline is a no-op under compat. src/ops/common/memory.cuh:69-107: pipe_copy is a synchronous global
   load + LDS store; pipe_commit/pipe_wait are empty. The kernels' entire latency-hiding strategy (header comment
   q5_rowsplit_gemv.cuh:6-11) evaporates: each logical warp issues 16 B/lane, stalls on s_waitcnt vmcnt(0), writes
   LDS, then computes. Bytes in flight per wave = 1 KB (q4 pair) / ~660 B (q5).
2. Occupancy is LDS-throttled to 1-2 waves/SIMD. q4 pair: ~23 KB static per 128-thread block -> 2 blocks/CU = 4
   wave64/CU = 1 wave per SIMD. q5 K=5120: ~31 KB per 512-thread block -> 2 blocks/CU = 16 wave64/CU. With #1,
   in-flight bytes per CU are ~4-10 KB vs the ~7-10 KB/CU needed just to cover DRAM latency at 700 GB/s, and
   nothing overlaps compute.
3. Dequant on the critical path with ~6 VALU ops per weight value (byte LDS read, 2 shift/mask, 2 sign_extend,
   2 cvt_i32_f32, 2 fmul, 2 fma, plus bf16x2->f32 x convert per group). At 178 M MACs per gate/up call that is
   ~1.1 G lane-ops -> ~165 us of pure VALU on 60 CU x 64 lanes x 1.7 GHz — larger than the 130 us DRAM floor.
   Without overlap (#1, #2) it adds serially. Packed v_dot2_f32_f16 + the bias trick already used at
   rowsplit_tiled_gemm_gfx906.cuh:131-149 cuts this to ~1.5 ops/value.
4. Wave64 half-utilised on the reduction and on scale broadcasts: lane = tid & 31 means each wave64 executes two
   independent 5-step __shfl_down ladders (warp.cuh:26-33, ds_bpermute-based); minor vs #1-#3 but free to fix with
   the donor DPP ladder.
5. Coalescing is fine — per-tile loads are 512 B contiguous per logical warp along K. Grid size is also fine
   (4352 / 320 / 320 / 768 blocks).

## 3. Donor approach

llama.cpp-gfx906/ggml/src/ggml-cuda/gfx906/matmul/mmvq-q4_0.cuh:10-94: __launch_bounds__(64,1), one wave64 block,
32 lanes per row, 2 rows per wave (:25-29), grid = nrows/2 (:110). Each lane loads a 16 B q4_0 block (32 values) +
32 B of q8_1 activations with plain register loads (no LDS), does 8 dp4a (__builtin_amdgcn_sdot4, common.cuh:713)
then one fp32 fma per block; loop ib += 32 (:48). Reduction: warp_reduce_sum<32> -> warp_reduce_amd_f32 DPP ladder
(common.cuh:456-458, gfx906-common.cuh:133-142: xor1/xor2 via v_add_f32_dpp quad_perm, xor4 via row_shl/shr, xor8
row_ror:8, xor16 ds_swizzle). The donor's v_dot2_f32_f16 idiom is at common.cuh:765. No throughput numbers are
documented in the donor; it relies on activation quantisation to int8 (Q8_1), which ninfer's 1-ulp criterion cannot
afford, so we take its geometry and reduction, not its int8 path.

## 4. Design per shape

Common skeleton (q_gemv_gfx906.cuh, new, alongside rowsplit_tiled_gemm_gfx906.cuh):
- No LDS for weights. Lane loads its uint4 (16 B = 32 nibbles = half a group) straight to registers; Q5 adds a 4 B
  high-plane load + a 2 B scale. Unroll 4 group-halves deep per lane (64 B nibbles in flight per lane, 4 KB per
  wave) with the raw loads for step i+1 issued before decode of step i (the stage-8 register-prefetch pattern,
  STAGE8-9-LOG.md:30-33).
- 32 lanes per row, 2 rows per wave64, 4 waves (256 threads) per block. K=5120 is 5 steps, K=6144 is 6, K=17408
  is 17. Weight reads stay 512 B contiguous per half-wave.
- x staged in LDS as fp16 once per block (K=5120: 10 KB; K=17408: 34 KB -> 1 block/CU, acceptable; alternatively
  a 2-way K split with fp32 atomics — avoid, breaks bit-determinism). The bias trick subtracts in fp16 before the
  dot, exactly as Q4TileAtomGfx906::decode_quarter (rowsplit_tiled_gemm_gfx906.cuh:131-149) / Q5TileAtomGfx906
  (:180-202) — reuse those atoms verbatim.
- Dot: gfx906_fdot2 (rowsplit_tiled_gemm_gfx906.cuh:51-53) on fp16 pairs, fp32 accumulate. Per-group scale applied
  inside the atom via __hmul2 (same numerics stage 8 already passed conformance with).
- Reduce: donor DPP ladder warp_reduce_amd_f32<32, AddOp> ported into warp.cuh under NINFER_GFX906_COMPAT as the
  warp_reduce_sum specialisation (keeps width-32 semantics for both half-waves simultaneously).
- Occupancy target: ~12 KB LDS (x fp16) + ~40 VGPRs -> 4-5 blocks/CU = 16-20 wave64/CU, ~64-80 KB in flight per CU.

| Shape | Grid | Bytes/call | Est. fraction | Est. us | ms/token (x layers) |
|---|---|---|---|---|---|
| gate/up pair 2x17408x5120 | 4352 blocks | 94.7 MB | 0.65 | ~210 | 13.4 (vs 36.7) |
| down 5120x17408 q5 + residual | 640 blocks | 55 MB | 0.60 | ~130 | 8.3 (vs 14.4) |
| out-proj 5120x6144 q5 + residual | 640 blocks | 20 MB | 0.55 | ~52 | 3.3 (vs 7.6) |
| GDN v/z 12288x5120 q5 + conv epilogue | 1536 blocks | 39 MB | 0.60 | ~93 | 4.5 (vs 7.4) |

Sum: ~29.5 ms/token of GEMV vs 66 now; total decode ~79 ms -> ~42 ms => ~24-28 t/s plain.

Epilogues: keep the templates — Epilogue::operator()<SplitOutput, SplitRow>(out, out_tail, row, acc)
(q5_rowsplit_gemv.cuh:197) is what Q5GdnDecodeEpilogue and the residual path expect; the new kernel calls it from
half_lane == 0 with the same arguments; the swiglu pair epilogue becomes a tiny functor silu(gate)*up on two
accumulators.

## 5. Test/bench plan

- Conformance: tests/ops/linear/test_q5_a16.cpp:34-46 covers a16(1) at 5120x6144 and 5120x17408 (also 6144/7168x5120
  T=1); tests/ops/linear_add/test_q5_a16.cpp:16,23 have interior T=1 for both K; tests/ops/linear_swiglu/
  test_q4_a16.cpp:14-19 sweeps T=1 at 34816x5120; tests/ops/test_gdn_input_proj_conv_snapshot.cpp covers width=1
  with snapshot publish. Criterion: relative L2 <= 1 bf16 ulp (STAGE8-9-LOG.md:41).
- Tier 1: ninfer_linear_bench --qtype q5 --n 5120 --k 17408 --t 1; --n 5120 --k 6144 --t 1; --n 6144 --k 5120
  --t 1; ninfer_q4_linear_swiglu_bench --t-sweep 1; ninfer_gdn_input_proj_conv_snapshot_bench; ninfer_q5_linear_add_bench.
  Interleave on/off.
- Gate: add gfx906_pass2_gemv_enabled() to src/ops/linear/gfx906/stage8_route.h (same pattern as :13-19), env
  NINFER_GFX906_PASS2 default on, =0 reverts. Hook points: q5_dispatch.cpp:29-36 (t==1 branch),
  q5_linear_add_gemv.cu:21-26, q4_linear_swiglu_gemv.cu:205-215, q4_q5_gdn_input_conv_snapshot.cu:157-185.

## 6. Risks

- Accumulation order: fp16 products (exact) with fp32 dot2 accumulate in a different order than scalar fmaf; stage 8
  established this class is within 1 ulp (STAGE8-9-LOG.md:39-41) and produced one near-tie greedy divergence
  (:176-179) — expected and acceptable per the stage-6 rule.
- GDN epilogue side effects: GdnConvEpilogue::store reads state_read and calls publish (gdn_conv.cuh:76-116); it
  must still execute exactly once per row from one lane. PDL hooks (TriggerPdl/JoinPdl, q5_rowsplit_gemv.cuh:
  127-129,199) must be preserved in the new template.
- Swiglu interleave: gate row r and up row r+17408 (q4_linear_swiglu_gemv.cu:135-141) must be streamed by the same
  half-wave so the pair epilogue needs no cross-wave exchange; two accumulators double VGPR pressure — budget it.
- K=17408 x in LDS (34 KB) forces 1 block/CU; if bandwidth fraction stalls below 50 %, fall back to reading x
  through L2 (kStageX=false path exists) or a 2-way K split.

## Implementation order

1. Port DPP ladder + q_gemv_gfx906.cuh skeleton on the Q5 5120x6144 residual shape first (smallest, exercises the
   residual epilogue and the Q5 atom).
2. Q4 gate/up pair (43.5 % of GPU time — the payoff).
3. Q5 down 5120x17408 (LDS/x decision).
4. GDN 12288x5120 split-epilogue last (most side effects).
5. Tier 2 A/B with graphs, then flip the default.

## Open questions

- Is the x fp16 clamp (+/-65504) acceptable for decode activations at the down-proj input? Stage 8 already does
  this for T>=2.
- Does hipcc keep 4 unrolled uint4 loads in flight or serialise on s_waitcnt? Check ISA after first build.
- Should NINFER_GFX906_PASS2 be folded into NINFER_GFX906_STAGE8 or stay separate for A/B? (Separate, for A/B.)

Critical files: src/ops/linear/q5/q5_rowsplit_gemv.cuh, src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu,
src/ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh, src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu,
donor llama.cpp-gfx906/ggml/src/ggml-cuda/gfx906/gfx906-common.cuh.
