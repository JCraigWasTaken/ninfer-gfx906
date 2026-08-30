#pragma once

// CUDA -> HIP compatibility surface for the gfx906 (AMD Instinct MI50/MI60)
// port. Modeled on llama.cpp's ggml-cuda/vendors/hip.h: only the API surface
// ninfer actually uses is mapped. Included through the shim headers in
// src/compat/gfx906/include (cuda_runtime.h, cuda_bf16.h, ...), so the
// ~150 mechanical translation units keep their original #includes.

#if !defined(NINFER_GFX906_COMPAT)
#error "hip_compat.h is only for NINFER_GFX906_COMPAT builds"
#endif

// ROCm 6.4 declares masked __shfl_*_sync builtins that require 64-bit wave
// masks; ninfer passes 32-bit CUDA masks and expects logical 32-lane
// subgroups. Disable the builtins and provide unmasked shims below.
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1

#include <hip/hip_runtime.h>

#if !defined(__HIP_PLATFORM_AMD__)
#error "The gfx906 port supports only AMD HIP targets"
#endif

// --- runtime API: types ------------------------------------------------------
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaUUID_t hipUUID
#define cudaMemcpyKind hipMemcpyKind
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t

#define CUDART_VERSION HIP_VERSION

// --- runtime API: enums / constants -----------------------------------------
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaGraphExecUpdateSuccess hipGraphExecUpdateSuccess
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// --- runtime API: functions --------------------------------------------------
#define cudaGetLastError hipGetLastError
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphUpload hipGraphUpload

// CUDA-12 flag-style instantiate maps to hipGraphInstantiateWithFlags; the
// legacy hipGraphInstantiate signature takes error-node/log-buffer arguments.
#define cudaGraphInstantiate hipGraphInstantiateWithFlags

// CUDA 12 reports graph-exec update results through a result-info struct; HIP
// keeps the CUDA 11 (error node + result enum) signature. Adapt.
struct ninferHipGraphExecUpdateResultInfo {
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    hipGraphNode_t errorNode        = nullptr;
};
#define cudaGraphExecUpdateResultInfo ninferHipGraphExecUpdateResultInfo
inline hipError_t ninfer_hip_graph_exec_update(hipGraphExec_t exec, hipGraph_t graph,
                                               ninferHipGraphExecUpdateResultInfo* info) {
    return hipGraphExecUpdate(exec, graph, &info->errorNode, &info->result);
}
#define cudaGraphExecUpdate ninfer_hip_graph_exec_update

// hipFuncSetAttribute takes const void*; CUDA accepts kernel pointers.
template <class F>
inline hipError_t ninfer_hip_func_set_attribute(F* func, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attribute, value);
}
#define cudaFuncSetAttribute ninfer_hip_func_set_attribute

// --- device-code attributes without a HIP equivalent -------------------------
// __grid_constant__ params are plain by-value params on HIP; __maxnreg__ is an
// NVIDIA register-budget hint (gfx906 occupancy is retuned in the rewrite
// stages instead).
#define __grid_constant__
#define __maxnreg__(n)

// --- warp shuffles -----------------------------------------------------------
// ninfer's kernels operate on logical 32-lane subgroups. HIP's unmasked
// __shfl_* with width 32 subdivides a 64-lane gfx906 wave into two independent
// 32-lane groups, which preserves the CUDA semantics of every ninfer call
// site. Masks are dropped (all call sites pass full masks).
#if defined(__HIPCC__)
template <class T>
__device__ __forceinline__ T __shfl_sync(unsigned, T var, int src_lane, int width = 32) {
    return __shfl(var, src_lane, width);
}
template <class T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned, T var, int lane_mask, int width = 32) {
    return __shfl_xor(var, lane_mask, width);
}
template <class T>
__device__ __forceinline__ T __shfl_up_sync(unsigned, T var, unsigned delta, int width = 32) {
    return __shfl_up(var, delta, width);
}
template <class T>
__device__ __forceinline__ T __shfl_down_sync(unsigned, T var, unsigned delta, int width = 32) {
    return __shfl_down(var, delta, width);
}
__device__ __forceinline__ void __syncwarp(unsigned = 0) {
    // Lanes of a gfx906 wave execute in lockstep; a wave-level sync is a no-op
    // barrier beyond making memory operations visible.
    __builtin_amdgcn_wave_barrier();
}
#endif // defined(__HIPCC__)
