
## 2026-09-02 21:40 — NINFER RESULT TP2-S9b: graphs-on gates PASS with the flag-sync split graphs (parity gate identical to S6 + phase F EXACT, graph_tp2 OK, MTP md5 = eager) → flag-sync is now the tp2 DEFAULT; MTP code at tp2 graphs-on 46.11 t/s (eager 37.3, tp1 36.7), prose 25.38 (eager 25.2, tp1 22.2)
Repo ~/ninfer-work/ninfer-gfx906 branch gfx906-port, base tag tp2-s9b-base (7c02856c). Commits (not pushed): 2fe747b8
"graphs-on gates" (mtp_tp2_real_test honours NINFER_TP2_TEST_GRAPHS=0), de210ae4 "flag-sync default at tp2"
(allreduce.cu flag_sync_requested() true unless NINFER_GFX906_TP2_FLAG_SYNC=0; header comment; TP2-SLICES.md S9b).
Container ninfer-rocm:6.4.1, MI50 pair, llama-swap stopped for the two GPU windows (chain 20:46-21:22, final
21:23-21:33) and active after each; RAS ue/ce/de 0/0/0 on both cards after every step; no new core files.
All four test binaries were relinked against the S9 runtime first (they dated from 17:27-17:48, before 99bd9f9d).

GATES with NINFER_GFX906_TP2_FLAG_SYNC=1, graphs ON (stage10/tp2-s9b-chain.log + per-step logs):
  1. ninfer_qwen3_8_27b_tp2_real_test: FAIL exit 1, 60 s, "tp1 and tp2 greedy output diverge after 12 of 32 tokens"
     (memory-table checks pass). The same test EAGER (NINFER_TP2_TEST_GRAPHS=0, tp2-s9b-tp2_real_test-eager.log):
     FAIL, 60 s, the IDENTICAL message. Divergence count 12/32 both ways => not a graph-replay defect (S6's "graph
     pathology" reading of this failure is withdrawn); it is the split on the test's synthetic cyclic prompt with
     Int8Group64 KV, which is what the S6 note said an eager leg would decide. Open item, see below.
  2. ninfer_qwen3_8_27b_graph_tp2_test: OK, exit 0, 48 s: "tp2 graphs == eager over 128 greedy decode steps, and
     reproducible across two graph runs" (S6 could not run it: ~10 min/leg at 0.21 t/s).
     ninfer_qwen3_8_27b_mtp_tp2_real_test: FAIL exit 1, 443 s. Probe A passes in full: oracle agreement 64/64 (tp1
     64/64), worst disagreement gap 0; tp2-vs-tp1 64 (MTP) / 64 (ordinary); MTP-vs-off 64/64 both widths; acceptance
     0.901961 tp1 = 0.901961 tp2. Probe B fails leg 1: "tp2 MTP3 tracks tp1 MTP3 for 3 of 64 tokens, worse than the
     ordinary path's 18 on the same prompt" - the ordinary tp2 path itself leaves tp1 at token 18 on this prompt, so
     leg 1 compares two already-diverged continuations (the donor's own comment on probe B describes the same knife
     edge on its hardware, "texts that stop being the same after token 12"). Probe A's LEG 3 (graphs vs eager) sits
     after probe B and never ran. Eager re-run (new NINFER_TP2_TEST_GRAPHS=0 knob): FAIL exit 1, 441 s, the IDENTICAL output (probe A 64/64 both widths, acceptance 0.901961 = tp1; probe B "3 of 64 vs 18"). So neither donor-test failure is owned by the graph path: both reproduce eagerly, byte for byte.
     The DFlash-at-tp2 rejection line is the expected "not split-aware yet" message, not a failure.
  3. ninfer_qwen3_8_27b_tp2_parity_test, NINFER_TP2_PARITY_GRAPHS=1, POSITIONS=128, KV bf16: exit 0 "ok", 1511 s
     (S6 eager: 1543). Phases C/D are eager by construction and reproduce S6 to the digit: teacher-forced argmax
     636/640 (99.375 %), cosine min 0.990863 mean 0.999759, KL(tp1||tp2) mean 0.000615 max 0.02436; per class
     reasoning 128/128, code 128/128, prose 126/128, multilingual 126/128, long_repetition 128/128. Control 7/160
     perturbed; GATE on the perturbed basis: KL 0.000451 vs budget 0.00181 nats; argmax 7/7 = 1 vs floor 0.99;
     (1-cos) 7.87e-5 vs budget 8.53e-5 -- PASS (S6: 0.000451 / 7/7 / 7.87e-5 vs 8.53e-5). Null control (tp1 on
     device 1) 160/160 bit-identical. NEW: phase F "tp2 graphs-on vs tp2 eager": EXACT on every prompt (reasoning
     193, code 411, prose 160, multilingual 512, long_repetition 512 tokens; tp2-graph engine open 14.9 s).
     JSON stage10/tp2-s9b-parity.json.
  4. MTP at tp2 graphs-on (apps/ninfer --tp 2 --devices 0,1, greedy, bf16 KV, PASS2):
        p5 code  --spec mtp --draft-tokens 3 --max-new 200: 46.11 tok/s, acceptance 94.31 % (3.83 tok/round, by pos
                 41/39/36), md5 c56fa74a60c73835 = eager tp2 = S6/S7.   eager tp2 37.3, tp1 36.7  => +24 % / +26 %
        p4 prose --spec mtp --draft-tokens 2 --max-new 128: 25.38 tok/s, acceptance 55.00 % (2.10 tok/round),
                 md5 1a2308484a9c3797 = the plain 128-new p4 text (pass 2f).   eager tp2 25.2, tp1 22.2  => flat / +14 %
     ninfer_bench --tp 2 --devices 0,1 --kv-dtype bf16 --mtp-draft-tokens 3 -p 512 -n 128 -r 3 --warmup 1
     (tp2-s9b-bench-mtp3.json, decode_path mtp_cuda_graph): tg128 decode_output_tok_s 22.46 +- 0.04, acceptance
     29.2 % (606 drafted / 177 accepted, length 1.87, 3 fallback steps); pp512 338.2. tp1 pass-2e reference on the
     same corpus: 18.00 +- 0.03 @ 30.2 %. (The bench corpus is low-acceptance by nature; the CLI code prompt is the
     94 % case.) Plain-decode reference from S9: tg128 35.53 graphs-on vs 31.75 eager.
     Reading: graphs help MTP on code far more than plain decode (+24 % vs +12 %) because a d3 round launches
     ~4x the kernels of a plain step and the host gaps were the bottleneck; prose at 55 % acceptance is
     verify-bound and stays flat.

DEFAULT FLIP (de210ae4): flag_sync_requested() returns true unless NINFER_GFX906_TP2_FLAG_SYNC=0 (=0 restores the
S8 event-bridged single graph, correct at ~0.2 t/s; --no-cuda-graph is still eager). tp1 constructs no PeerEvents
and is code-identical. Rebuilt apps/ninfer, ninfer-serve, ninfer_bench and the four tp2 tests (tp2-s9b-build-2/3.log).
Post-flip gates with NO env (tp2-s9b-final.log): tp2 p3 md5 52960e3a58e9ef2002021c8cd3904855 = reference, 33.98
tok/s, banner "[tp2] flag-sync transport ON (tp2 default; NINFER_GFX906_TP2_FLAG_SYNC=0 disables)"; tp1 p3 on card 2
md5 52960e3a58e9ef2002021c8cd3904855 unchanged, 23.13 tok/s; tp1 p4 MTP d3 on card 2 md5
32ba0f7a69b8da571d60ce25f6dbb3aa unchanged, 20.26 tok/s @ 37.85 %.

OPEN (carried): (a) tp2_real_test / mtp probe B: the split diverges from tp1 at token 12-18 on the synthetic
Int8Group64-KV prompts while the bf16-KV parity corpus is inside the donor budget - decide whether the donor's own
tolerance applies (its comment says yes for probe B) or whether the int8 KV split needs a look; (b) FlagSignal::status
still unread (surface via the graph-trace knob); (c) the 16 MB staging constant; (d) h2h vs production with graphs
on at tp2 - the MTP code number (46.1) is now above llama-cpp's plain 45.4 on the code prompt, so the head-to-head
should be re-run with the default build.
Files: stage10/tp2-s9b-chain.sh, tp2-s9b-final.sh, tp2-s9b-patch-default.py, tp2-s9b-patch-mtptest.py,
tp2-s9b-{tp2_real_test,graph_tp2_test,mtp_tp2_real_test,parity_test}.log, tp2-s9b-tp2_real_test-eager.log,
tp2-s9b-mtp_tp2_real_test-eager.log, tp2-s9b-p5-mtp3.{txt,err}, tp2-s9b-p4-mtp2.{txt,err}, tp2-s9b-bench-mtp3.json,
tp2-s9b-parity.json, tp2-s9b-{tp2-p3,tp1-p3,tp1-p4-mtp3}.{txt,err}, tp2-s9b-build-{1,2,3}.log.
