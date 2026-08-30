"""Exact-size pinned host tensors and HIP/CUDA address translation."""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import mmap
import os
from pathlib import Path
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_pinned_extension():
    """Load the packaged native extension, if this install built one.

    ROCm wheels may use the ctypes HIP fallback when a platform cannot build a
    C++ extension (for example a minimal Windows install); the fallback still
    registers and translates host memory rather than silently treating pageable
    memory as GPU-visible.
    """

    try:
        return importlib.import_module("freetoken.kernel._pinned_tensor")
    except (ImportError, OSError):
        return None


@lru_cache(maxsize=1)
def _hip_runtime():
    if not getattr(torch.version, "hip", None):
        return None
    candidates = [
        "amdhip64_10.dll",
        "amdhip64_7.dll",
        "amdhip64.dll",
        "libamdhip64.so",
        "libamdhip64.so.1",
    ]
    # Custom ROCm 10.x-compatible driver channels often install amdhip64 next
    # to the private Torch artifact instead of a system loader path.
    for root_name in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(root_name)
        if root:
            root_path = Path(root)
            for subdir in ("bin", "lib"):
                candidates.extend(
                    str(root_path / subdir / name)
                    for name in ("amdhip64.dll", "libamdhip64.so", "libamdhip64.so.1")
                )
    discovered = ctypes.util.find_library("amdhip64")
    if discovered:
        candidates.append(discovered)
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype."""

    ext = _load_pinned_extension()
    if ext is not None:
        return ext.create_pinned_tensor_like(input)
    try:
        return torch.empty_like(input, pin_memory=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "pinned host allocation is unavailable; install the ROCm/CUDA Torch "
            "build and the matching FreeToken native extension"
        ) from exc


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate exact-size mapped host storage via the native runtime."""

    ext = _load_pinned_extension()
    if ext is not None:
        return ext.alloc_pinned_tensor(list(shape), dtype)
    try:
        return torch.empty(*shape, dtype=dtype, pin_memory=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "pinned host allocation is unavailable; install the ROCm/CUDA Torch "
            "build and the matching FreeToken native extension"
        ) from exc


def host_register(addr: int, nbytes: int) -> None:
    """Register an existing host range as portable and device-mapped."""

    ext = _load_pinned_extension()
    if ext is not None:
        ext.host_register(addr, nbytes)
        return
    hip = _hip_runtime()
    if hip is not None:
        status = hip.hipHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_uint(3)
        )
        if status != 0:
            raise RuntimeError(f"hipHostRegister({nbytes} bytes) failed with hipError {status}")
        return
    if getattr(torch.version, "hip", None):
        raise RuntimeError("Torch reports ROCm/HIP but amdhip64 could not be loaded")
    # CUDA installations historically built the extension at install time.  If
    # an intentionally extension-free CUDA install reaches this path, callers
    # will use the staged-copy fallback rather than dereferencing host memory.


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    ext = _load_pinned_extension()
    if ext is not None:
        return bool(ext.host_ptr_identity())
    hip = _hip_runtime()
    if hip is None:
        return False

    # Probe the exact registration path used by mmap-backed expert banks.  A
    # hipHostMalloc probe is insufficient on Windows/WDDM because it can be
    # unified while hipHostRegister maps to a distinct device VA.
    buf = mmap.mmap(-1, 4096)
    try:
        host_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        if hip.hipHostRegister(
            ctypes.c_void_p(host_addr), ctypes.c_size_t(4096), ctypes.c_uint(3)
        ) != 0:
            return False
        device_addr = ctypes.c_void_p()
        ok = (
            hip.hipHostGetDevicePointer(
                ctypes.byref(device_addr), ctypes.c_void_p(host_addr), ctypes.c_uint(0)
            )
            == 0
        )
        hip.hipHostUnregister(ctypes.c_void_p(host_addr))
        return bool(ok and device_addr.value == host_addr)
    finally:
        buf.close()


def device_ptr(t: torch.Tensor) -> int:
    """Return the address a GPU kernel must dereference for ``t``."""

    # Device tensors already expose the address kernels use.  Host tensors go
    # through the runtime translation API even on UVA/identity systems: the
    # driver may report identity support globally while a registered allocation
    # (notably Windows/WDDM or a compatible MI50 driver) still receives a
    # distinct device alias.  ``hipHostGetDevicePointer``/the native CUDA
    # equivalent is cheap at plan-build time and returns the same value when
    # identity truly holds.
    if t.is_cuda:
        return t.data_ptr()
    ext = _load_pinned_extension()
    if ext is not None:
        return int(ext.host_device_ptr(t.data_ptr()))
    hip = _hip_runtime()
    if hip is None:
        raise RuntimeError("device_ptr requires mapped pinned memory and a HIP/CUDA runtime")
    device_addr = ctypes.c_void_p()
    status = hip.hipHostGetDevicePointer(
        ctypes.byref(device_addr), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0)
    )
    if status != 0 or not device_addr.value:
        raise RuntimeError(f"hipHostGetDevicePointer failed with hipError {status}")
    return int(device_addr.value)


__all__ = [
    "alloc_pinned_tensor",
    "copy_to_pinned_tensor",
    "create_pinned_tensor_like",
    "device_ptr",
    "host_register",
]
