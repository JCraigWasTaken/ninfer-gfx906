// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cuda_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    int forward = 0;
    int reverse = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]));
    // Asymmetric support is not a usable transport for a symmetric collective: fall back to the
    // staged path rather than enabling one direction only.
    if (forward == 0 || reverse == 0) { return false; }

    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(pair[rank]);
        const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            // Already enabled by an earlier call; clear the sticky runtime error so the next
            // cudaGetLastError() in an unrelated launcher does not observe it.
            cudaGetLastError();
            continue;
        }
        CUDA_CHECK(status);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// gfx906 flag-sync transport (TP2 slice 9). See allreduce.h and tools/tp2/replay_probe.cu
// (split-flag shape) for the design and the measurements behind the memory-type choices:
// hipDeviceMallocFinegrained signal blocks DEADLOCK (the local poll is served from L2),
// plain-cudaMalloc staging returns STALE partials (the reader's L2 keeps the previous call's
// lines); hipDeviceMallocUncached for both is the combination that replays == eager.
namespace {

constexpr int kFlagBlocks  = 64;
constexpr int kFlagThreads = 256;
// Local polls between checks; ~64 M polls is tens of seconds, so a lost peer reports through the
// status word instead of hanging the queue forever.
constexpr unsigned long long kFlagTimeoutPolls = 1ull << 26;

struct FlagSignal {
    unsigned start[kFlagBlocks][2]; // written by the PEER, indexed by the writer's rank
    unsigned end[kFlagBlocks][2];
    unsigned seq[kFlagBlocks];      // this rank's own per-block sequence
    unsigned status[2];             // [0] != 0: a wait timed out, [1] = the sequence it waited for
};

__device__ __forceinline__ bool flag_barrier(unsigned* peer_slot, const unsigned* self_slot,
                                             unsigned flag, unsigned* status) {
    __hip_atomic_store(peer_slot, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned long long polls = 0;
    while (__hip_atomic_load(self_slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag) {
        __builtin_amdgcn_s_sleep(1);
        if (++polls > kFlagTimeoutPolls) {
            atomicOr(status, 1u);
            status[1] = flag;
            return false;
        }
    }
    return true;
}

// Two-rank summing all-reduce, in place on `buffer`, BF16. Block b owns every element whose
// 256-element chunk index is congruent to b modulo gridDim.x; the push and the combine of a block
// cover the same elements, so the per-block barrier is sufficient.
__global__ void flag_allreduce_bf16_kernel(__nv_bfloat16* buffer, __nv_bfloat16* peer_staging,
                                           const __nv_bfloat16* staging, std::int64_t n,
                                           FlagSignal* self_sg, FlagSignal* peer_sg, int rank) {
    const int b                 = blockIdx.x;
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t first    = static_cast<std::int64_t>(b) * blockDim.x + threadIdx.x;
    __shared__ int ok;
    for (std::int64_t i = first; i < n; i += stride) { peer_staging[i] = buffer[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned flag = self_sg->seq[b] + 1;
        self_sg->seq[b]     = flag;
        ok = flag_barrier(&peer_sg->start[b][rank], &self_sg->start[b][1 - rank], flag,
                          self_sg->status);
    }
    __syncthreads();
    if (!ok) { return; }
    for (std::int64_t i = first; i < n; i += stride) {
        // The qualified residual_add body: FP32 accumulate, one round-to-nearest-even on store.
        buffer[i] = __float2bfloat16_rn(__bfloat162float(buffer[i]) + __bfloat162float(staging[i]));
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ok = flag_barrier(&peer_sg->end[b][rank], &self_sg->end[b][1 - rank], self_sg->seq[b],
                          self_sg->status);
    }
    __syncthreads();
}

// Two-rank row gather over bytes: push my block into the peer's staging, barrier, then write my
// own block and the peer's (now local) block into my destination, barrier.
template <typename Word>
__global__ void flag_allgather_kernel(Word* destination, const Word* own, std::int64_t own_words,
                                      std::int64_t own_offset, Word* peer_staging,
                                      const Word* staging, std::int64_t peer_words,
                                      std::int64_t peer_offset, FlagSignal* self_sg,
                                      FlagSignal* peer_sg, int rank) {
    const int b               = blockIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t first  = static_cast<std::int64_t>(b) * blockDim.x + threadIdx.x;
    __shared__ int ok;
    for (std::int64_t i = first; i < own_words; i += stride) { peer_staging[i] = own[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned flag = self_sg->seq[b] + 1;
        self_sg->seq[b]     = flag;
        ok = flag_barrier(&peer_sg->start[b][rank], &self_sg->start[b][1 - rank], flag,
                          self_sg->status);
    }
    __syncthreads();
    if (!ok) { return; }
    for (std::int64_t i = first; i < own_words; i += stride) { destination[own_offset + i] = own[i]; }
    for (std::int64_t i = first; i < peer_words; i += stride) {
        destination[peer_offset + i] = staging[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ok = flag_barrier(&peer_sg->end[b][rank], &self_sg->end[b][1 - rank], self_sg->seq[b],
                          self_sg->status);
    }
    __syncthreads();
}

bool flag_sync_requested() {
    const char* text = std::getenv("NINFER_GFX906_TP2_FLAG_SYNC");
    return text != nullptr && text[0] == '1' && text[1] == '\0';
}

// Per-rank uncached staging capacity. The decode-time collectives are the row-parallel
// projections' outputs ([hidden, B] BF16, 10 KB per row) and the logits gather ([V/2, B] BF16,
// ~150 KB per row); 16 MB covers batches into the hundreds. Larger calls (prefill) take the event
// transport, which is fine because prefill runs eagerly.
constexpr std::size_t kFlagStagingBytes = std::size_t{16} << 20;

// The flag transport is for CAPTURED collectives, where the event transport cannot be used at all
// (two live captures cannot be bridged). Outside a capture the event transport stays: prefill's
// multi-MB all-reduces are faster as SDMA copies than as a 64-block push kernel (pp512 339 vs 253
// tok/s), and eager decode keeps its byte-identical S7 path.
bool capturing(const ExecutionContext& ec) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(ec.dev[0]->stream, &status));
    return status != hipStreamCaptureStatusNone;
}

int grid_for(std::int64_t work) {
    const std::int64_t blocks = (work + kFlagThreads - 1) / kFlagThreads;
    return static_cast<int>(blocks < 1 ? 1 : blocks > kFlagBlocks ? kFlagBlocks : blocks);
}

} // namespace

PeerEvents::PeerEvents(const ExecutionContext& ec) {
    require_two_devices(ec, "PeerEvents: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    cudaEvent_t created[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int slot = 0; slot < 4; ++slot) {
        const int rank             = slot % 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    inputs_ready_ = {created[0], created[1]};
    pull_done_    = {created[2], created[3]};

    if (!flag_sync_requested()) { return; }
    // Allocated AFTER enable_peer_access() (the caller's order) so the runtime maps both blocks
    // into the peer's address space; peer writes into them are the whole transport.
    for (int rank = 0; rank < 2; ++rank) {
        const int device = ec.dev[rank]->device;
        CurrentDeviceGuard::set(device);
        void* signal  = nullptr;
        void* staging = nullptr;
        cudaError_t status =
            hipExtMallocWithFlags(&signal, sizeof(FlagSignal), hipDeviceMallocUncached);
        if (status == cudaSuccess) {
            status = hipExtMallocWithFlags(&staging, kFlagStagingBytes, hipDeviceMallocUncached);
        }
        if (status == cudaSuccess) { status = cudaMemset(signal, 0, sizeof(FlagSignal)); }
        if (status == cudaSuccess) { status = cudaDeviceSynchronize(); }
        if (status != cudaSuccess) {
            if (signal != nullptr) { (void)cudaFree(signal); }
            if (staging != nullptr) { (void)cudaFree(staging); }
            for (int done = 0; done < rank; ++done) {
                CurrentDeviceGuard::set(flag_device_[static_cast<std::size_t>(done)]);
                (void)cudaFree(flag_signal_[static_cast<std::size_t>(done)]);
                (void)cudaFree(flag_staging_[static_cast<std::size_t>(done)]);
                flag_signal_[static_cast<std::size_t>(done)]  = nullptr;
                flag_staging_[static_cast<std::size_t>(done)] = nullptr;
            }
            throw std::runtime_error(std::string("PeerEvents: flag-sync allocation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
        flag_signal_[static_cast<std::size_t>(rank)]  = signal;
        flag_staging_[static_cast<std::size_t>(rank)] = staging;
        flag_device_[static_cast<std::size_t>(rank)]  = device;
    }
    flag_capacity_ = kFlagStagingBytes;
    std::fprintf(stderr, "[tp2] flag-sync transport ON (NINFER_GFX906_TP2_FLAG_SYNC=1): uncached "
                         "signal blocks + %zu MB staging per rank\n", kFlagStagingBytes >> 20);
}

PeerEvents::~PeerEvents() {
    if (flag_sync()) {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            (void)cudaSetDevice(flag_device_[slot]);
            (void)cudaFree(flag_signal_[slot]);
            (void)cudaFree(flag_staging_[slot]);
            flag_signal_[slot]  = nullptr;
            flag_staging_[slot] = nullptr;
        }
        if (restore) { (void)cudaSetDevice(previous); }
    }
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_),
      flag_signal_(other.flag_signal_), flag_staging_(other.flag_staging_),
      flag_device_(other.flag_device_), flag_capacity_(other.flag_capacity_) {
    other.inputs_ready_ = {nullptr, nullptr};
    other.pull_done_    = {nullptr, nullptr};
    other.flag_signal_  = {nullptr, nullptr};
    other.flag_staging_ = {nullptr, nullptr};
    other.flag_capacity_ = 0;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    flag_signal_.swap(other.flag_signal_);
    flag_staging_.swap(other.flag_staging_);
    flag_device_.swap(other.flag_device_);
    std::swap(flag_capacity_, other.flag_capacity_);
    return *this;
}


void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    if (events.flag_sync() && bytes <= events.flag_capacity() && capturing(ec)) {
        // Both launches are issued by this thread with no synchronisation in between (the
        // deadlock rule: a rank's kernel waits for the peer's, which must already be enqueued or
        // guaranteed to be by the same thread).
        const std::int64_t n = buffer[0].numel();
        const int grid       = grid_for(n);
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            flag_allreduce_bf16_kernel<<<grid, kFlagThreads, 0, local.stream>>>(
                static_cast<__nv_bfloat16*>(buffer[rank].data),
                static_cast<__nv_bfloat16*>(events.flag_staging(1 - rank)),
                static_cast<const __nv_bfloat16*>(events.flag_staging(rank)), n,
                static_cast<FlagSignal*>(events.flag_signal(rank)),
                static_cast<FlagSignal*>(events.flag_signal(1 - rank)), rank);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer's operand into storage only it owns. Both inbound copies
    // are issued before either rank waits again, so the two directions overlap.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    if (events.flag_sync() && block[0] <= events.flag_capacity() &&
        block[1] <= events.flag_capacity() && capturing(ec)) {
        const auto d0 = reinterpret_cast<std::uintptr_t>(destination[0].data);
        const auto d1 = reinterpret_cast<std::uintptr_t>(destination[1].data);
        const auto p0 = reinterpret_cast<std::uintptr_t>(part[0].data);
        const auto p1 = reinterpret_cast<std::uintptr_t>(part[1].data);
        const bool words4 = ((d0 | d1 | p0 | p1 | block[0] | block[1]) & 3) == 0;
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            auto* self_sg = static_cast<FlagSignal*>(events.flag_signal(rank));
            auto* peer_sg = static_cast<FlagSignal*>(events.flag_signal(1 - rank));
            if (words4) {
                const auto own_words  = static_cast<std::int64_t>(block[rank] / 4);
                const auto peer_words = static_cast<std::int64_t>(block[1 - rank] / 4);
                flag_allgather_kernel<std::uint32_t>
                    <<<grid_for(own_words > peer_words ? own_words : peer_words), kFlagThreads, 0,
                       local.stream>>>(static_cast<std::uint32_t*>(destination[rank].data),
                                       static_cast<const std::uint32_t*>(part[rank].data),
                                       own_words, static_cast<std::int64_t>(offset[rank] / 4),
                                       static_cast<std::uint32_t*>(events.flag_staging(1 - rank)),
                                       static_cast<const std::uint32_t*>(events.flag_staging(rank)),
                                       peer_words, static_cast<std::int64_t>(offset[1 - rank] / 4),
                                       self_sg, peer_sg, rank);
            } else {
                const auto own_words  = static_cast<std::int64_t>(block[rank]);
                const auto peer_words = static_cast<std::int64_t>(block[1 - rank]);
                flag_allgather_kernel<std::uint8_t>
                    <<<grid_for(own_words > peer_words ? own_words : peer_words), kFlagThreads, 0,
                       local.stream>>>(static_cast<std::uint8_t*>(destination[rank].data),
                                       static_cast<const std::uint8_t*>(part[rank].data),
                                       own_words, static_cast<std::int64_t>(offset[rank]),
                                       static_cast<std::uint8_t*>(events.flag_staging(1 - rank)),
                                       static_cast<const std::uint8_t*>(events.flag_staging(rank)),
                                       peer_words, static_cast<std::int64_t>(offset[1 - rank]),
                                       self_sg, peer_sg, rank);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer's block, both on its own
    // stream, so destination[r] has exactly one writer.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                             part[1 - rank].data, block[1 - rank], local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

} // namespace ninfer::ops
