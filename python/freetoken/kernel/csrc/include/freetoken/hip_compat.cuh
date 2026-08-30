#pragma once

// Compatibility surface for CUDA-flavored FreeToken device kernels when they
// are compiled by hipcc.  Keep the shim deliberately small and ABI-oriented so
// it also works with the ROCm 10.x gfx906 driver stack.
#include <hip/hip_runtime.h>

#include <tuple>
#include <utility>

#ifndef __always_inline
#define __always_inline inline __attribute__((always_inline))
#endif
#ifndef __grid_constant__
#define __grid_constant__
#endif

using cudaError_t = ::hipError_t;
using cudaStream_t = ::hipStream_t;
using cudaLaunchConfig_t = ::hipLaunchConfig_t;
using cudaLaunchAttribute = ::hipLaunchAttribute;
using cudaFuncAttribute = ::hipFuncAttribute;
using cudaMemcpyKind = ::hipMemcpyKind;
using cudaGraph_t = ::hipGraph_t;
using cudaGraphExec_t = ::hipGraphExec_t;

inline constexpr auto cudaSuccess = ::hipSuccess;
inline constexpr auto cudaFuncAttributeMaxDynamicSharedMemorySize =
    ::hipFuncAttributeMaxDynamicSharedMemorySize;
inline constexpr auto cudaDevAttrUnifiedAddressing =
    ::hipDeviceAttributeUnifiedAddressing;
inline constexpr auto cudaDevAttrCanUseHostPointerForRegisteredMem =
    ::hipDeviceAttributeCanUseHostPointerForRegisteredMem;
inline constexpr auto cudaMemcpyHostToHost = ::hipMemcpyHostToHost;
inline constexpr auto cudaMemcpyHostToDevice = ::hipMemcpyHostToDevice;
inline constexpr auto cudaMemcpyDeviceToHost = ::hipMemcpyDeviceToHost;
inline constexpr auto cudaMemcpyDeviceToDevice = ::hipMemcpyDeviceToDevice;

[[nodiscard]] inline auto cudaGetErrorString(::cudaError_t error) -> const char * {
  return ::hipGetErrorString(error);
}
inline auto cudaGetLastError() -> ::cudaError_t { return ::hipGetLastError(); }

template <typename... Args>
inline auto cudaGetDevice(Args &&...args) -> ::cudaError_t {
  return ::hipGetDevice(std::forward<Args>(args)...);
}
template <typename... Args>
inline auto cudaDeviceGetAttribute(Args &&...args) -> ::cudaError_t {
  return ::hipDeviceGetAttribute(std::forward<Args>(args)...);
}
template <typename... Args>
inline auto cudaSetDevice(Args &&...args) -> ::cudaError_t {
  return ::hipSetDevice(std::forward<Args>(args)...);
}
template <typename... Args>
inline auto cudaHostGetDevicePointer(Args &&...args) -> ::cudaError_t {
  return ::hipHostGetDevicePointer(std::forward<Args>(args)...);
}

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
#define cudaLaunchHostFunc hipLaunchHostFunc

template <typename F>
inline auto cudaFuncSetAttribute(F *func, ::hipFuncAttribute attr, int value)
    -> ::cudaError_t {
  return ::hipFuncSetAttribute(reinterpret_cast<const void *>(func), attr, value);
}

template <typename F, typename... Args>
inline auto cudaLaunchKernelEx(const ::cudaLaunchConfig_t *config, F func,
                               Args &&...args) -> ::cudaError_t {
  auto storage = std::make_tuple(std::forward<Args>(args)...);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    void *params[] = {const_cast<void *>(
        static_cast<const void *>(&std::get<I>(storage)))...};
    return ::hipLaunchKernelExC(config, reinterpret_cast<const void *>(func), params);
  }(std::index_sequence_for<Args...>{});
}
