// TP2 slice 8: per-node REPLAY cost of a graph whose nodes span two devices.
//
// tools/tp2/capture_probe.cu settled that a two-device capture replays the
// same VALUES as eager execution. It never timed the replay. On the MI50 pair
// (ROCm 6.4.1) the real tp2 decode graph (~1400 nodes) replays correctly but
// at ~4.7 s per launch, while the tp1 graph (696 nodes) replays in ~40 ms.
// This tool builds the S1 choreography at several sizes, in several shapes,
// and prints wall time and host CPU time per replay and per node so the shape
// that carries the cost can be read off a curve:
//
//   single-x2     two SINGLE-device graphs (one per rank), launched back to
//                 back on their own streams. No cross-device edge anywhere.
//                 The control: what "graphs at tp2" would cost if each rank
//                 had its own graph.
//   dual-fork     ONE graph holding both devices' kernels; the only cross-
//                 device edges are the fork and join that enroll the peer.
//   dual-edges    ONE graph, plus the four-event record/wait choreography of
//                 every all-reduce round (cross-device EDGES), but no transport
//                 nodes.
//   dual-memcpy   ONE graph, edges + the real UVA D2D pull (memcpy nodes).
//                 This is exactly what src/core/decode_graph.cpp records.
//   dual-kcopy    ONE graph, edges + a peer-read copy KERNEL instead of the
//                 memcpy node (the step-3 alternative transport).
//   split-events  TWO per-device captures live at once, bridged by event
//                 record/wait ACROSS the captures (CUDA rejects this with
//                 cudaErrorStreamCaptureMerge; HIP accepted it in S1). If HIP
//                 produces two graphs that replay to the eager values, this is
//                 the split-graph design.
//   eager         the dual-memcpy body issued eagerly, for reference.
//   split-flag    (S9) TWO per-device graphs, each capturing ONLY its own
//                 stream: no cross-capture waits, no cross-device memcpy
//                 nodes. The all-reduce is ONE kernel per rank, synchronised
//                 through DEVICE MEMORY the way the mxxm llama.cpp fork's
//                 tp-allreduce.cu does it on this hardware (b10811):
//                   push     every block PCIe-writes its slice of the local
//                            partial into the PEER's staging (posted writes,
//                            straight to the peer's HBM);
//                   start    thread 0 bumps a per-block sequence kept in the
//                            local signal block, RELEASE/SYSTEM-stores it into
//                            the PEER's signal block (the flag rides the same
//                            PCIe link behind the data, so it lands after it)
//                            and polls its OWN signal block (allocated with
//                            hipDeviceMallocUncached, so the ACQUIRE/SYSTEM
//                            load re-fetches HBM) until the peer has written
//                            the same sequence. The spin never crosses PCIe.
//                   combine  buffer += staging (the peer's contribution is
//                            already local);
//                   end      the same barrier on a second flag array: the
//                            peer must not push the NEXT round into staging
//                            before this round finished reading it.
//                 Per rank per round: scale, allreduce = 2 kernel nodes
//                 (4 per round over both graphs). The sequence numbers live
//                 in the signal block (a graph replays identical kernel
//                 arguments) and keep counting across replays; the poll has a
//                 count timeout that sets a status word, and the host joins
//                 with a timeout too. The first S9 attempt (poll the PEER's
//                 flag through UVA) deadlocked with fine-grained device flags:
//                 remote VRAM is mapped cacheable on gfx906, so the spinning
//                 load was served from a stale L2 line; host-coherent flags
//                 resolved the spin but the peer-READ data path then returned
//                 stale partials (no L2 writeback instruction on gfx906).
//                 Inverting both directions (write remote, read local) is
//                 what makes it correct.
//
// Each round is: per rank a local kernel, the exchange, per rank a combine
// kernel -> 4 kernel nodes + 2 transport nodes per round.
//
// Build: part of the CMake tree as ninfer_tp2_replay_probe (tools/tp2/CMakeLists.txt).
// Run:   ninfer_tp2_replay_probe [dev_a] [dev_b] [rounds,rounds,...] [replays] [shape,...|all]
//            [signal: uncached|finegrained|host] [staging: plain|uncached]
// Exit:  0 ran; 1 hard failure; 77 fewer than two devices.

#include <cuda_runtime.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA %s:%d: %s: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorName(e_), cudaGetErrorString(e_));                 \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr int kElems = 4096;
constexpr size_t kBytes = kElems * sizeof(float);
constexpr int kBlock = 256;
constexpr int kGrid = (kElems + kBlock - 1) / kBlock;

struct Rank {
  int device = 0;
  cudaStream_t stream = nullptr;
  float* buffer = nullptr;
  float* staging = nullptr;
  // split-flag: the flag the PEER polls (fine-grained, peer-visible), this
  // rank's sequence counter and the wait-timeout status word (ordinary device
  // memory, touched only by this device).
  struct Signal* signal = nullptr;  // this rank's signal block (peer writes into it)
  unsigned* status = nullptr;
};

// split-flag: per-block barrier flags. `start`/`end` are written by the PEER
// (indexed by the writer's rank), `seq` is this rank's own per-block counter.
struct Signal {
  unsigned start[kGrid][2];
  unsigned end[kGrid][2];
  unsigned seq[kGrid];
};

// signal block memory: 0 = hipDeviceMallocUncached, 1 = hipDeviceMallocFinegrained, 2 = coherent host
int g_signal_mem = 0;
// staging memory: 0 = plain cudaMalloc, 1 = hipDeviceMallocUncached
int g_staging_mem = 0;
constexpr unsigned long long kWaitTimeoutPolls = 1ull << 26;  // local polls, ~64 M: seconds, not forever

struct Events {
  cudaEvent_t inputs_ready[2] = {nullptr, nullptr};
  cudaEvent_t pull_done[2] = {nullptr, nullptr};
  cudaEvent_t fork = nullptr;
  cudaEvent_t join = nullptr;
};

enum class Transport { None, Memcpy, Kernel };

struct Shape {
  const char* name;
  bool one_graph;      // both devices in one capture (fork/join)
  bool round_edges;    // the four-event choreography per round
  Transport transport;
  bool split;          // two live captures bridged by cross-capture events
  bool eager;
  bool flag;           // S9: per-device graphs synchronised through device-memory flags
};

__global__ void add_into(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void scale_by(float* dst, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] *= k;
}

__global__ void peer_copy(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

// S9 flag sync (one thread). Signal: bump this rank's sequence, release-fence
// the data written by the kernels before this one, store the sequence to the
// flag the peer polls (system-scope release; the flag lives in fine-grained
// memory so the store bypasses L2 and lands where a PCIe read can see it).
// Wait: system-scope atomic loads of the PEER's flag through its UVA pointer
// (peer VRAM is mapped MTYPE_UC on gfx9, so the load is never served from a
// stale L2 line), sleeping between polls, until it reaches the same sequence.
// Both ranks signal BEFORE they wait, so the pair cannot deadlock unless one
// graph is never launched; that case trips the wall-clock timeout and is
// reported through `status` instead of hanging the queue.
// One barrier phase of the flag all-reduce (thread 0 of a block): publish
// `flag` into the peer's slot, then poll the local slot the peer writes.
// Returns false on timeout (status set), true when the peer has arrived.
__device__ __forceinline__ bool flag_barrier(unsigned* peer_slot, const unsigned* self_slot,
                                             unsigned flag, unsigned* status,
                                             unsigned long long timeout_polls) {
  __hip_atomic_store(peer_slot, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  unsigned long long polls = 0;
  while (__hip_atomic_load(self_slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag) {
    __builtin_amdgcn_s_sleep(1);
    if (++polls > timeout_polls) {
      atomicOr(status, 1u);
      status[1] = flag;
      return false;
    }
  }
  return true;
}

// The two-rank all-reduce, in place on `buffer`, one launch per rank per
// round. Block b owns elements [b*blockDim, (b+1)*blockDim).
__global__ void flag_allreduce(float* buffer, float* peer_staging, const float* staging, int n,
                               Signal* self_sg, Signal* peer_sg, int rank, unsigned* status,
                               unsigned long long timeout_polls) {
  const int b = blockIdx.x;
  const int i = b * blockDim.x + threadIdx.x;
  __shared__ int ok;
  // push this rank's slice into the peer's staging (PCIe posted writes)
  if (i < n) peer_staging[i] = buffer[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned flag = self_sg->seq[b] + 1;
    self_sg->seq[b] = flag;
    ok = flag_barrier(&peer_sg->start[b][rank], &self_sg->start[b][1 - rank], flag, status,
                      timeout_polls);
  }
  __syncthreads();
  if (!ok) return;
  if (i < n) buffer[i] += staging[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned flag = self_sg->seq[b];
    ok = flag_barrier(&peer_sg->end[b][rank], &self_sg->end[b][1 - rank], flag, status,
                      timeout_polls);
  }
  __syncthreads();
}

double now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

double cpu_us() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

void setup(Rank r[2], int dev_a, int dev_b) {
  r[0].device = dev_a;
  r[1].device = dev_b;
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    cudaError_t pe = cudaDeviceEnablePeerAccess(r[1 - i].device, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
      printf("  note: peer access %d->%d: %s\n", r[i].device, r[1 - i].device,
             cudaGetErrorName(pe));
    }
    (void)cudaGetLastError();
    CK(cudaStreamCreateWithFlags(&r[i].stream, cudaStreamNonBlocking));
    CK(cudaMalloc(&r[i].buffer, kBytes));
    if (g_staging_mem == 1) {
      CK(hipExtMallocWithFlags(reinterpret_cast<void**>(&r[i].staging), kBytes,
                               hipDeviceMallocUncached));
    } else {
      CK(cudaMalloc(&r[i].staging, kBytes));
    }
    CK(cudaMemset(r[i].staging, 0, kBytes));
    CK(cudaMalloc(&r[i].status, 256));
    CK(cudaMemset(r[i].status, 0, 256));
  }
  // The signal blocks: uncached so the local poll re-fetches HBM and sees the
  // peer's PCIe write; allocated AFTER peer access is enabled so the runtime
  // maps them into the peer's address space too.
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    void* p = nullptr;
    if (g_signal_mem == 0 || g_signal_mem == 1) {
      const unsigned flags = g_signal_mem == 0 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained;
      hipError_t fe = hipExtMallocWithFlags(&p, sizeof(Signal), flags);
      if (fe != hipSuccess) {
        printf("  note: hipExtMallocWithFlags(%s) on device %d: %s -> host coherent signal\n",
               g_signal_mem == 0 ? "uncached" : "finegrained", r[i].device, hipGetErrorName(fe));
        (void)hipGetLastError();
        g_signal_mem = 2;
      }
    }
    if (g_signal_mem == 2) {
      CK(hipHostMalloc(&p, sizeof(Signal), hipHostMallocCoherent | hipHostMallocMapped));
    }
    r[i].signal = static_cast<Signal*>(p);
    CK(cudaMemset(r[i].signal, 0, sizeof(Signal)));
    CK(cudaDeviceSynchronize());
  }
}

// S9: rank `rank`'s half of `rounds` rounds, on its own stream only.
void issue_flag(Rank r[2], int rounds, int rank) {
  Rank& me = r[rank];
  const Rank& peer = r[1 - rank];
  CK(cudaSetDevice(me.device));
  for (int round = 0; round < rounds; ++round) {
    scale_by<<<kGrid, kBlock, 0, me.stream>>>(me.buffer, 0.5f, kElems);
    CK(cudaGetLastError());
    flag_allreduce<<<kGrid, kBlock, 0, me.stream>>>(me.buffer, peer.staging, me.staging, kElems,
                                                    me.signal, peer.signal, rank, me.status,
                                                    kWaitTimeoutPolls);
    CK(cudaGetLastError());
  }
}

// Host-side join with a timeout: poll both streams instead of blocking, so a
// graph pair that never meets (one launched, one not) reports instead of
// hanging the probe. Returns false on timeout.
bool sync_both_timeout(Rank r[2], double timeout_s) {
  const double t0 = now_us();
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    for (;;) {
      cudaError_t q = hipStreamQuery(r[i].stream);
      if (q == cudaSuccess) break;
      if (q != hipErrorNotReady) CK(q);
      if (now_us() - t0 > timeout_s * 1e6) return false;
    }
  }
  return true;
}

// Reads back both ranks' status words; true when neither wait timed out.
bool flags_ok(Rank r[2]) {
  bool ok = true;
  for (int i = 0; i < 2; ++i) {
    unsigned st[2] = {0, 0};
    CK(cudaSetDevice(r[i].device));
    CK(cudaMemcpy(st, r[i].status, sizeof(st), cudaMemcpyDeviceToHost));
    if (st[0] != 0) {
      printf("  split-flag: rank %d wait TIMED OUT at sequence %u\n", i, st[1]);
      ok = false;
    }
  }
  return ok;
}

void fill(Rank r[2]) {
  std::vector<float> host(kElems);
  for (int i = 0; i < 2; ++i) {
    const float base = i == 0 ? 10.0f : 100.0f;
    for (int j = 0; j < kElems; ++j) host[j] = base + static_cast<float>(j % 7);
    CK(cudaSetDevice(r[i].device));
    CK(cudaMemcpy(r[i].buffer, host.data(), kBytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
  }
}

std::vector<float> read_back(const Rank& rank) {
  std::vector<float> host(kElems);
  CK(cudaSetDevice(rank.device));
  CK(cudaMemcpy(host.data(), rank.buffer, kBytes, cudaMemcpyDeviceToHost));
  return host;
}

// A capture that failed part way (split shapes) can leave a stream poisoned
// (EndCapture: hipErrorStreamCaptureUnjoined, then hipErrorIllegalState on
// the next BeginCapture) and its events pointing into a dead capture. Replace
// them all so the next shape starts clean.
void recreate_streams_and_events(Rank r[2], Events& ev) {
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    (void)cudaStreamDestroy(r[i].stream);
    (void)cudaEventDestroy(ev.inputs_ready[i]);
    (void)cudaEventDestroy(ev.pull_done[i]);
    (void)cudaGetLastError();
    CK(cudaStreamCreateWithFlags(&r[i].stream, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ev.inputs_ready[i], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ev.pull_done[i], cudaEventDisableTiming));
  }
  CK(cudaSetDevice(r[0].device));
  (void)cudaEventDestroy(ev.fork);
  CK(cudaEventCreateWithFlags(&ev.fork, cudaEventDisableTiming));
  CK(cudaSetDevice(r[1].device));
  (void)cudaEventDestroy(ev.join);
  CK(cudaEventCreateWithFlags(&ev.join, cudaEventDisableTiming));
  (void)cudaGetLastError();
}

void sync_both(Rank r[2]) {
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    CK(cudaStreamSynchronize(r[i].stream));
  }
}

// Issue `rounds` rounds for the ranks in `mask`.
// Returns false (instead of aborting) when `soft` and a cross-device wait is
// rejected -- the split-events shape exists to learn exactly that.
bool issue(Rank r[2], Events& ev, int rounds, int mask, bool round_edges,
           Transport transport, bool soft = false) {
  for (int round = 0; round < rounds; ++round) {
    for (int rank = 0; rank < 2; ++rank) {
      if (!(mask & (1 << rank))) continue;
      CK(cudaSetDevice(r[rank].device));
      scale_by<<<kGrid, kBlock, 0, r[rank].stream>>>(r[rank].buffer, 0.5f, kElems);
      CK(cudaGetLastError());
    }
    if (round_edges) {
      for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(r[rank].device));
        CK(cudaEventRecord(ev.inputs_ready[rank], r[rank].stream));
      }
      for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(r[rank].device));
        if (soft) {
          cudaError_t we = cudaStreamWaitEvent(r[rank].stream, ev.inputs_ready[1 - rank], 0);
          if (we != cudaSuccess) {
            printf("  split-events: cross-capture wait (rank %d waits rank %d) -> %s\n", rank,
                   1 - rank, cudaGetErrorName(we));
            (void)cudaGetLastError();
            return false;
          }
        } else {
          CK(cudaStreamWaitEvent(r[rank].stream, ev.inputs_ready[1 - rank], 0));
        }
        if (transport == Transport::Memcpy) {
          cudaError_t me = cudaMemcpyAsync(r[rank].staging, r[1 - rank].buffer, kBytes,
                                           cudaMemcpyDeviceToDevice, r[rank].stream);
          if (soft && me != cudaSuccess) {
            printf("  split: cross-capture memcpy (rank %d pulls rank %d) -> %s\n", rank,
                   1 - rank, cudaGetErrorName(me));
            (void)cudaGetLastError();
            return false;
          }
          CK(me);
        } else if (transport == Transport::Kernel) {
          peer_copy<<<kGrid, kBlock, 0, r[rank].stream>>>(r[rank].staging,
                                                          r[1 - rank].buffer, kElems);
          cudaError_t ke = cudaGetLastError();
          if (soft && ke != cudaSuccess) {
            printf("  split: peer-read kernel inside capture (rank %d reads rank %d) -> %s\n",
                   rank, 1 - rank, cudaGetErrorName(ke));
            return false;
          }
          CK(ke);
        }
        {
          cudaError_t re = cudaEventRecord(ev.pull_done[rank], r[rank].stream);
          if (soft && re != cudaSuccess) {
            printf("  split: record(pull_done) inside capture (rank %d) -> %s\n", rank,
                   cudaGetErrorName(re));
            (void)cudaGetLastError();
            return false;
          }
          CK(re);
        }
      }
      for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(r[rank].device));
        if (soft) {
          cudaError_t we = cudaStreamWaitEvent(r[rank].stream, ev.pull_done[1 - rank], 0);
          if (we != cudaSuccess) {
            printf("  split-events: cross-capture wait(pull_done) -> %s\n",
                   cudaGetErrorName(we));
            (void)cudaGetLastError();
            return false;
          }
        } else {
          CK(cudaStreamWaitEvent(r[rank].stream, ev.pull_done[1 - rank], 0));
        }
      }
    }
    for (int rank = 0; rank < 2; ++rank) {
      if (!(mask & (1 << rank))) continue;
      CK(cudaSetDevice(r[rank].device));
      add_into<<<kGrid, kBlock, 0, r[rank].stream>>>(r[rank].buffer, r[rank].staging, kElems);
      CK(cudaGetLastError());
    }
  }
  return true;
}

void histogram(cudaGraph_t graph, int& kernel, int& memcpy_n, int& other) {
  size_t count = 0;
  kernel = memcpy_n = other = 0;
  if (graph == nullptr) return;
  CK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  if (count != 0) CK(cudaGraphGetNodes(graph, nodes.data(), &count));
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    CK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeKernel) ++kernel;
    else if (type == cudaGraphNodeTypeMemcpy) ++memcpy_n;
    else ++other;
  }
}

struct Built {
  cudaGraph_t graph[2] = {nullptr, nullptr};
  cudaGraphExec_t exec[2] = {nullptr, nullptr};
  int nodes = 0, kernels = 0, memcpys = 0, others = 0;
  bool ok = true;
  std::string note;
};

Built build(Rank r[2], Events& ev, const Shape& s, int rounds) {
  Built b;
  if (s.eager) return b;
  if (s.flag) {
    // two per-device graphs, each capturing only its own stream
    for (int i = 0; i < 2; ++i) {
      CK(cudaSetDevice(r[i].device));
      CK(cudaStreamBeginCapture(r[i].stream, cudaStreamCaptureModeThreadLocal));
      issue_flag(r, rounds, i);
      CK(cudaStreamEndCapture(r[i].stream, &b.graph[i]));
      int k, m, o;
      histogram(b.graph[i], k, m, o);
      b.kernels += k; b.memcpys += m; b.others += o;
      CK(cudaGraphInstantiate(&b.exec[i], b.graph[i], 0));
    }
    b.nodes = b.kernels + b.memcpys + b.others;
    b.note = "4 nodes/round";
    return b;
  }
  if (s.split) {
    CK(cudaSetDevice(r[0].device));
    CK(cudaStreamBeginCapture(r[0].stream, cudaStreamCaptureModeThreadLocal));
    CK(cudaSetDevice(r[1].device));
    cudaError_t e = cudaStreamBeginCapture(r[1].stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
      b.ok = false;
      b.note = std::string("second BeginCapture: ") + cudaGetErrorName(e);
      cudaGraph_t d = nullptr;
      CK(cudaSetDevice(r[0].device));
      (void)cudaStreamEndCapture(r[0].stream, &d);
      (void)cudaGetLastError();
      return b;
    }
    const bool issued = issue(r, ev, rounds, 3, true, s.transport, /*soft=*/true);
    cudaError_t e0, e1;
    CK(cudaSetDevice(r[0].device));
    e0 = cudaStreamEndCapture(r[0].stream, &b.graph[0]);
    CK(cudaSetDevice(r[1].device));
    e1 = cudaStreamEndCapture(r[1].stream, &b.graph[1]);
    (void)cudaGetLastError();
    if (!issued) {
      b.ok = false;
      b.note = std::string("cross-capture wait rejected; EndCapture ") + cudaGetErrorName(e0) +
               " / " + cudaGetErrorName(e1);
      return b;
    }
    if (e0 != cudaSuccess || e1 != cudaSuccess || b.graph[0] == nullptr ||
        b.graph[1] == nullptr) {
      b.ok = false;
      b.note = std::string("EndCapture: ") + cudaGetErrorName(e0) + " / " + cudaGetErrorName(e1);
      return b;
    }
    int per[2][3];
    for (int i = 0; i < 2; ++i) {
      histogram(b.graph[i], per[i][0], per[i][1], per[i][2]);
      b.kernels += per[i][0]; b.memcpys += per[i][1]; b.others += per[i][2];
      CK(cudaSetDevice(r[i].device));
      cudaError_t ie = cudaGraphInstantiate(&b.exec[i], b.graph[i], 0);
      if (ie != cudaSuccess) {
        b.ok = false;
        b.note = std::string("Instantiate: ") + cudaGetErrorName(ie);
        return b;
      }
    }
    b.nodes = b.kernels + b.memcpys + b.others;
    b.note = "g0 " + std::to_string(per[0][0] + per[0][1] + per[0][2]) + " nodes (other " +
             std::to_string(per[0][2]) + "), g1 " +
             std::to_string(per[1][0] + per[1][1] + per[1][2]) + " nodes (other " +
             std::to_string(per[1][2]) + ")";
    return b;
  }
  if (!s.one_graph) {
    // two single-device graphs
    for (int i = 0; i < 2; ++i) {
      CK(cudaSetDevice(r[i].device));
      CK(cudaStreamBeginCapture(r[i].stream, cudaStreamCaptureModeThreadLocal));
      issue(r, ev, rounds, 1 << i, false, Transport::None);
      CK(cudaStreamEndCapture(r[i].stream, &b.graph[i]));
      int k, m, o;
      histogram(b.graph[i], k, m, o);
      b.kernels += k; b.memcpys += m; b.others += o;
      CK(cudaGraphInstantiate(&b.exec[i], b.graph[i], 0));
    }
    b.nodes = b.kernels + b.memcpys + b.others;
    return b;
  }
  // one graph, peer forked in
  CK(cudaSetDevice(r[0].device));
  CK(cudaStreamBeginCapture(r[0].stream, cudaStreamCaptureModeThreadLocal));
  CK(cudaEventRecord(ev.fork, r[0].stream));
  CK(cudaSetDevice(r[1].device));
  CK(cudaStreamWaitEvent(r[1].stream, ev.fork, 0));
  issue(r, ev, rounds, 3, s.round_edges, s.transport);
  CK(cudaSetDevice(r[1].device));
  CK(cudaEventRecord(ev.join, r[1].stream));
  CK(cudaSetDevice(r[0].device));
  CK(cudaStreamWaitEvent(r[0].stream, ev.join, 0));
  CK(cudaStreamEndCapture(r[0].stream, &b.graph[0]));
  histogram(b.graph[0], b.kernels, b.memcpys, b.others);
  b.nodes = b.kernels + b.memcpys + b.others;
  CK(cudaGraphInstantiate(&b.exec[0], b.graph[0], 0));
  return b;
}

void run_once(Rank r[2], Events& ev, const Shape& s, const Built& b, int rounds) {
  if (s.eager) {
    issue(r, ev, rounds, 3, true, Transport::Memcpy);
    return;
  }
  if (s.split || !s.one_graph || s.flag) {
    // Both launches are issued by this thread with no synchronisation in
    // between: the deadlock rule for the flag shape.
    for (int i = 0; i < 2; ++i) {
      CK(cudaSetDevice(r[i].device));
      CK(cudaGraphLaunch(b.exec[i], r[i].stream));
    }
    return;
  }
  CK(cudaSetDevice(r[0].device));
  CK(cudaGraphLaunch(b.exec[0], r[0].stream));
}

void destroy(Built& b) {
  for (int i = 0; i < 2; ++i) {
    if (b.exec[i]) (void)cudaGraphExecDestroy(b.exec[i]);
    if (b.graph[i]) (void)cudaGraphDestroy(b.graph[i]);
    b.exec[i] = nullptr;
    b.graph[i] = nullptr;
  }
  (void)cudaGetLastError();
}

}  // namespace

int main(int argc, char** argv) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
    printf("skip: replay probe needs two devices\n");
    return 77;
  }
  const int dev_a = argc > 1 ? atoi(argv[1]) : 0;
  const int dev_b = argc > 2 ? atoi(argv[2]) : 1;
  std::vector<int> round_list = {1, 5, 25, 100};
  if (argc > 3) {
    round_list.clear();
    std::string spec = argv[3];
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t next = spec.find(',', pos);
      if (next == std::string::npos) next = spec.size();
      round_list.push_back(atoi(spec.substr(pos, next - pos).c_str()));
      pos = next + 1;
    }
  }
  const int replays = argc > 4 ? atoi(argv[4]) : 20;
  std::vector<std::string> only;
  if (argc > 5 && strcmp(argv[5], "all") != 0) {
    std::string spec = argv[5];
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t next = spec.find(',', pos);
      if (next == std::string::npos) next = spec.size();
      only.push_back(spec.substr(pos, next - pos));
      pos = next + 1;
    }
  }
  if (argc > 6) {
    g_signal_mem = strcmp(argv[6], "host") == 0 ? 2 : strcmp(argv[6], "finegrained") == 0 ? 1 : 0;
  }
  if (argc > 7) g_staging_mem = strcmp(argv[7], "uncached") == 0 ? 1 : 0;
  const bool long_run = replays >= 1000;

  Rank r[2];
  setup(r, dev_a, dev_b);
  Events ev;
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    CK(cudaEventCreateWithFlags(&ev.inputs_ready[i], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ev.pull_done[i], cudaEventDisableTiming));
  }
  CK(cudaSetDevice(r[0].device));
  CK(cudaEventCreateWithFlags(&ev.fork, cudaEventDisableTiming));
  CK(cudaSetDevice(r[1].device));
  CK(cudaEventCreateWithFlags(&ev.join, cudaEventDisableTiming));

  int can_ab = 0, can_ba = 0;
  CK(cudaDeviceCanAccessPeer(&can_ab, dev_a, dev_b));
  CK(cudaDeviceCanAccessPeer(&can_ba, dev_b, dev_a));
  printf("devices %d,%d peer access %d/%d, %d timed replays per cell (2 warm-up)\n", dev_a,
         dev_b, can_ab, can_ba, replays);

  const Shape shapes[] = {
      {"single-x2", false, false, Transport::None, false, false, false},
      {"dual-fork", true, false, Transport::None, false, false, false},
      {"dual-edges", true, true, Transport::None, false, false, false},
      {"dual-memcpy", true, true, Transport::Memcpy, false, false, false},
      {"dual-kcopy", true, true, Transport::Kernel, false, false, false},
      {"split-memcpy", false, true, Transport::Memcpy, true, false, false},
      {"split-kcopy", false, true, Transport::Kernel, true, false, false},
      {"split-flag", false, false, Transport::Kernel, false, false, true},
      {"eager", false, true, Transport::Memcpy, false, true, false},
  };
  printf("split-flag: signal blocks in %s, staging %s\n",
         g_signal_mem == 0   ? "device memory, hipDeviceMallocUncached"
         : g_signal_mem == 1 ? "device memory, hipDeviceMallocFinegrained"
                             : "coherent host memory",
         g_staging_mem == 0 ? "plain cudaMalloc (peer-written)" : "hipDeviceMallocUncached (peer-written)");

  printf("\n%-13s %6s %6s %6s %6s %12s %10s %12s %8s\n", "shape", "rounds", "nodes", "kern",
         "memcpy", "wall us/rep", "us/node", "cpu us/rep", "values");
  for (int rounds : round_list) {
    // Reference values for this round count: the eager memcpy body.
    fill(r);
    issue(r, ev, rounds, 3, true, Transport::Memcpy);
    sync_both(r);
    const std::vector<float> ref0 = read_back(r[0]);
    const std::vector<float> ref1 = read_back(r[1]);

    for (const Shape& s : shapes) {
      if (!only.empty()) {
        bool want = false;
        for (const std::string& n : only) want = want || n == s.name;
        if (!want) continue;
      }
      Built b = build(r, ev, s, rounds);
      if (!b.ok) {
        printf("%-13s %6d  -- not buildable: %s\n", s.name, rounds, b.note.c_str());
        fflush(stdout);
        destroy(b);
        recreate_streams_and_events(r, ev);
        continue;
      }
      // warm-up
      bool timed_out = false;
      for (int w = 0; w < 2; ++w) {
        fill(r);
        run_once(r, ev, s, b, rounds);
        if (s.flag) timed_out = !sync_both_timeout(r, 30.0) || timed_out;
        else sync_both(r);
      }
      const bool comparable = s.eager || s.split || s.transport != Transport::None;
      std::string values = "n/a";
      if (comparable) {
        const std::vector<float> o0 = read_back(r[0]);
        const std::vector<float> o1 = read_back(r[1]);
        values = (o0 == ref0 && o1 == ref1) ? "==eager" : "DIFFER";
      }
      if (s.flag && (timed_out || !flags_ok(r))) {
        printf("%-13s %6d  -- wait timed out (host %d); aborting\n", s.name, rounds, timed_out);
        return 1;
      }
      double wall = 0, cpu = 0;
      for (int rep = 0; rep < replays; ++rep) {
        fill(r);
        const double c0 = cpu_us();
        const double t0 = now_us();
        run_once(r, ev, s, b, rounds);
        if (s.flag) {
          if (!sync_both_timeout(r, 30.0)) {
            printf("%-13s %6d  -- host join timed out at replay %d; aborting\n", s.name, rounds, rep);
            (void)flags_ok(r);
            return 1;
          }
        } else {
          sync_both(r);
        }
        wall += now_us() - t0;
        cpu += cpu_us() - c0;
        if (long_run && (rep + 1) % 500 == 0) {
          printf("  %s: replay %d/%d\n", s.name, rep + 1, replays);
          fflush(stdout);
        }
      }
      wall /= replays;
      cpu /= replays;
      if (comparable && long_run) {
        // values after the LAST replay too (the sequence numbers have advanced
        // by rounds * 2 * replays by now)
        const std::vector<float> o0 = read_back(r[0]);
        const std::vector<float> o1 = read_back(r[1]);
        values += (o0 == ref0 && o1 == ref1) ? "/last==eager" : "/lastDIFFER";
      }
      if (s.flag && !flags_ok(r)) values += "/TIMEOUT";
      const int nodes = s.eager ? rounds * 6 : b.nodes;
      printf("%-13s %6d %6d %6d %6d %12.1f %10.2f %12.1f %8s%s%s\n", s.name, rounds, nodes,
             b.kernels, b.memcpys, wall, wall / nodes, cpu, values.c_str(),
             b.note.empty() ? "" : "  ", b.note.c_str());
      fflush(stdout);
      destroy(b);
    }
    printf("\n");
  }
  return 0;
}
