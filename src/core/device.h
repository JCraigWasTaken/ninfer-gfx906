#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
