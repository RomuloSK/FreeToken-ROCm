#pragma once

// The native extensions use the CUDA Runtime API names because PyTorch's CUDA
// and HIP builds expose the same stream/event model.  Map only the small host
// runtime surface FreeToken needs; device kernels keep their own CUDA/HIP
// compatibility headers.
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime_api.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaMemcpyKind = hipMemcpyKind;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaHostAllocPortable = hipHostMallocPortable;
constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
constexpr auto cudaHostRegisterPortable = hipHostRegisterPortable;
constexpr auto cudaHostRegisterMapped = hipHostRegisterMapped;
constexpr auto cudaMemcpyHostToHost = hipMemcpyHostToHost;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;

#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy hipMemcpy
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaLaunchHostFunc hipLaunchHostFunc

// CUDA annotates stream host callbacks with CUDART_CB (cdecl on Windows).
// HIP's callback ABI uses the ordinary host calling convention, so keeping the
// annotation empty lets the CPU-MoE extension share the same source on both
// runtimes, including Windows HIP builds.
#ifndef CUDART_CB
#define CUDART_CB
#endif

// CUDA and HIP use different enumerator names for the two properties used by
// the pinned-memory identity probe.
#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#define cudaDevAttrCanUseHostPointerForRegisteredMem \
  hipDeviceAttributeCanUseHostPointerForRegisteredMem

#else
#include <cuda_runtime_api.h>
#endif
