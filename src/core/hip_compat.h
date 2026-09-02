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

#include <cstddef>

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
// Test tooling also uses the CUDA-11 five-argument form (error node, log buffer,
// buffer size), which is HIP's legacy hipGraphInstantiate signature.
inline hipError_t ninfer_hip_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                               unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t ninfer_hip_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                               hipGraphNode_t* error_node, char* log_buffer,
                                               std::size_t buffer_size) {
    return hipGraphInstantiate(exec, graph, error_node, log_buffer, buffer_size);
}
#define cudaGraphInstantiate ninfer_hip_graph_instantiate
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal

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

// --- TP2 slice 1: peer access, pointer attributes, capture status ------------
// Used by src/ops/common/allreduce.cu, split_launch.h and tools/tp2/*_probe.cu.
// hipPointerAttribute_t's memory-type field is `.type` on ROCm 6.x (verified
// against /opt/rocm/include/hip/hip_runtime_api.h in the 6.4.1 container), so
// the CUDA-12 `attributes.type == cudaMemoryTypeDevice` use sites port verbatim.
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaFuncAttribute hipFuncAttribute

// Probe-only surface (tools/tp2): the peer-copy control, graph node inspection,
// and the capture-info / capture-dependency forms the transport probe splices a
// memcpy node with. CUDA 12 grew edge-data arguments on cudaStreamGetCaptureInfo
// and cudaStreamUpdateCaptureDependencies; HIP keeps the CUDA 11 shapes, so the
// two overloads below accept the CUDA-12 call sites (edge data must be nullptr).
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaGraphNode_t hipGraphNode_t
#define cudaGraphNodeType hipGraphNodeType
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel
#define cudaGraphNodeTypeMemcpy hipGraphNodeTypeMemcpy
#define cudaGraphNodeGetType hipGraphNodeGetType
#define cudaGraphAddMemcpyNode hipGraphAddMemcpyNode
#define cudaMemcpy3DParms hipMemcpy3DParms
#define make_cudaPitchedPtr make_hipPitchedPtr
#define make_cudaExtent make_hipExtent
#define cudaStreamSetCaptureDependencies hipStreamSetCaptureDependencies
inline hipError_t ninfer_hip_stream_get_capture_info(hipStream_t stream,
                                                     hipStreamCaptureStatus* status,
                                                     unsigned long long* id) {
    return hipStreamGetCaptureInfo(stream, status, id);
}
inline hipError_t ninfer_hip_stream_get_capture_info(hipStream_t stream,
                                                     hipStreamCaptureStatus* status,
                                                     unsigned long long* id, hipGraph_t* graph,
                                                     const hipGraphNode_t** dependencies,
                                                     std::nullptr_t /*edge data*/,
                                                     std::size_t* dependency_count) {
    return hipStreamGetCaptureInfo_v2(stream, status, id, graph, dependencies, dependency_count);
}
#define cudaStreamGetCaptureInfo ninfer_hip_stream_get_capture_info
inline hipError_t ninfer_hip_stream_update_capture_dependencies(hipStream_t stream,
                                                                hipGraphNode_t* dependencies,
                                                                std::nullptr_t /*edge data*/,
                                                                std::size_t count,
                                                                unsigned int flags) {
    return hipStreamUpdateCaptureDependencies(stream, dependencies, count, flags);
}
#define cudaStreamUpdateCaptureDependencies ninfer_hip_stream_update_capture_dependencies
