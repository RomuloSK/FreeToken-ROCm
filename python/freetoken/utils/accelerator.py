"""Accelerator and device capability discovery.

PyTorch deliberately exposes ROCm devices through ``torch.cuda``.  Keeping that
implementation detail in one module lets the rest of FreeToken reason about the
actual accelerator (CUDA versus HIP) without spreading vendor checks through
the engine and kernel packages.
"""

from __future__ import annotations

import functools
import ctypes
import ctypes.util
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}


class AcceleratorKind(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceCapabilities:
    """Stable, serializable facts used by backend selection and diagnostics."""

    kind: AcceleratorKind
    index: int | None
    name: str
    architecture: str
    runtime_version: str | None
    torch_version: str | None
    warp_size: int
    supports_fp8: bool
    supports_fp4: bool
    supports_graphs: bool
    supports_pdl: bool
    supports_collectives: bool
    host_pointer_identity: bool | None = None

    @property
    def is_gpu(self) -> bool:
        return self.kind is not AcceleratorKind.CPU

    @property
    def target(self) -> str:
        """Compiler target (``sm_XX`` or ``gfxXXXX``)."""

        return self.architecture

    @property
    def host_pointer_mapping(self) -> bool | None:
        """Whether registered host allocations are directly GPU-addressable.

        ``host_pointer_identity`` is retained as the low-level spelling used
        by the pinned allocator; this alias is the accelerator-neutral public
        capability name.
        """

        return self.host_pointer_identity

    def as_dict(self) -> dict[str, Any]:
        return {
            "accelerator": self.kind.value,
            "index": self.index,
            "name": self.name,
            "architecture": self.architecture,
            "runtime_version": self.runtime_version,
            "torch_version": self.torch_version,
            "warp_size": self.warp_size,
            "supports_fp8": self.supports_fp8,
            "supports_fp4": self.supports_fp4,
            "supports_graphs": self.supports_graphs,
            "supports_pdl": self.supports_pdl,
            "supports_collectives": self.supports_collectives,
            "host_pointer_identity": self.host_pointer_identity,
            "host_pointer_mapping": self.host_pointer_mapping,
        }


def _arch_from_properties(props: Any, kind: AcceleratorKind, index: int) -> str:
    if kind is AcceleratorKind.CUDA:
        return f"sm_{int(props.major)}{int(props.minor)}"

    for attr in ("gcnArchName", "gcn_arch_name", "architecture"):
        value = getattr(props, attr, None)
        if value:
            if isinstance(value, bytes):
                value = value.decode("ascii", "replace")
            value = str(value).lower()
            if value.startswith("gfx"):
                # rocminfo and some vendor Torch builds append feature
                # modifiers (e.g. ``gfx906:sramecc+``); compiler targets use
                # the base ``gfx*`` token.
                return value.split(":", 1)[0]

    # Some ROCm Torch builds omit gcnArchName from DeviceProperties.  The
    # environment is still useful for cross-compiled or single-arch installs.
    for name in (
        "FREETOKEN_ROCM_ARCH",
        "FREETOKEN_ROCM_ARCH_LIST",
        "PYTORCH_ROCM_ARCH",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        value = os.environ.get(name)
        if value:
            value = value.replace(";", ",").split(",", 1)[0].strip().lower()
            if value.startswith("gfx"):
                return value.split(":", 1)[0]
            if name == "HSA_OVERRIDE_GFX_VERSION" and value.count(".") == 2:
                return "gfx" + value.replace(".", "")

    # Keep the report useful even when a vendor Torch build does not expose its
    # target.  Callers must treat this as unknown and avoid arch-specialized code.
    return f"gfx_unknown_{index}"


def _rocm_feature(architecture: str, feature: str) -> bool:
    """Conservative feature policy for ROCm architectures.

    The policy is intentionally narrower than the compiler's theoretical
    capabilities.  A future ROCm/driver can opt in by adding its target here,
    while unknown targets use correctness fallbacks.
    """

    arch = architecture.lower()
    if arch.startswith("gfx12"):
        return True
    if feature == "fp8" and arch.startswith(("gfx94", "gfx95")):
        return True
    if feature == "fp4" and arch.startswith(("gfx94", "gfx95")):
        return True
    return False


def _default_warp_size(kind: AcceleratorKind, architecture: str) -> int:
    if kind is AcceleratorKind.CUDA:
        return 32
    return 64 if architecture.startswith(("gfx90", "gfx94", "gfx95", "gfx906", "gfx908")) else 32


@functools.lru_cache(maxsize=8)
def detect_device_capabilities(index: int | None = None) -> DeviceCapabilities:
    """Discover capabilities for one visible PyTorch device.

    This function imports Torch lazily so CPU-only commands and package builds
    can inspect FreeToken without requiring an accelerator runtime.
    """

    try:
        import torch
    except Exception:
        return DeviceCapabilities(
            AcceleratorKind.CPU, None, "CPU", "cpu", None, None, 1,
            False, False, False, False, False,
        )

    if not torch.cuda.is_available():
        return DeviceCapabilities(
            AcceleratorKind.CPU, None, "CPU", "cpu", None, torch.__version__, 1,
            False, False, False, False, False,
        )

    if index is None:
        index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(index)
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_version = getattr(version, "cuda", None)
    kind = AcceleratorKind.ROCM if hip_version else AcceleratorKind.CUDA
    runtime_version = hip_version or cuda_version
    architecture = _arch_from_properties(props, kind, index)
    warp_size = int(getattr(props, "warp_size", 0) or _default_warp_size(kind, architecture))

    if kind is AcceleratorKind.CUDA:
        capability = (int(props.major), int(props.minor))
        supports_fp8 = capability >= (8, 9)
        supports_fp4 = capability >= (8, 0)
        supports_pdl = capability >= (9, 0)
    else:
        supports_fp8 = _rocm_feature(architecture, "fp8")
        supports_fp4 = _rocm_feature(architecture, "fp4")
        supports_pdl = False

    return DeviceCapabilities(
        kind=kind,
        index=index,
        name=str(getattr(props, "name", "Unknown GPU")),
        architecture=architecture,
        runtime_version=str(runtime_version) if runtime_version else None,
        torch_version=str(torch.__version__),
        warp_size=warp_size,
        supports_fp8=supports_fp8,
        supports_fp4=supports_fp4,
        supports_graphs=(
            hasattr(torch.cuda, "CUDAGraph")
            and os.getenv("FREETOKEN_DISABLE_GPU_GRAPHS", "").strip().lower()
            not in {"1", "true", "yes", "on"}
        ),
        supports_pdl=supports_pdl,
        supports_collectives=bool(
            getattr(
                getattr(torch, "distributed", None),
                "is_available",
                lambda: bool(getattr(torch, "distributed", None)),
            )()
        ),
    )


def active_accelerator(index: int | None = None) -> AcceleratorKind:
    return detect_device_capabilities(index).kind


def is_rocm(index: int | None = None) -> bool:
    return active_accelerator(index) is AcceleratorKind.ROCM


def is_cuda(index: int | None = None) -> bool:
    return active_accelerator(index) is AcceleratorKind.CUDA


def _probe_graph_replay(torch_module: Any, index: int | None) -> bool | None:
    """Run a tiny graph capture/replay when the runtime exposes the API.

    Attribute presence is not sufficient on every HIP driver: some compatible
    stacks expose ``CUDAGraph`` but reject capture for a particular device.  The
    probe is intentionally isolated to diagnostics (capability detection remains
    side-effect free for normal engine startup) and returns ``None`` when the
    fake/minimal Torch surface cannot be probed.
    """

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not all(
        hasattr(cuda, name) for name in ("CUDAGraph", "Stream", "stream", "graph")
    ):
        return None
    if index is None:
        try:
            index = int(cuda.current_device())
        except Exception:
            index = 0
    try:
        with cuda.device(index):
            stream = cuda.Stream()
            with cuda.stream(stream):
                value = torch_module.zeros(1, dtype=torch_module.float32, device="cuda")
                stream.synchronize()
                graph = cuda.CUDAGraph()
                with cuda.graph(graph, stream=stream):
                    value.add_(1)
                graph.replay()
                stream.synchronize()
                return bool(value.item() == 1)
    except Exception:
        return False


def accelerator_report(index: int | None = None) -> dict[str, Any]:
    """Return a diagnostic report without raising on optional components."""

    try:
        capabilities = detect_device_capabilities(index)
    except Exception as exc:
        # A partially installed driver can make even ``is_available`` raise;
        # diagnostics must still print the Python/toolchain context needed to
        # repair that installation.
        capabilities = DeviceCapabilities(
            AcceleratorKind.CPU,
            None,
            "Unknown",
            "unknown",
            None,
            None,
            1,
            False,
            False,
            False,
            False,
            False,
        )
        detection_error = f"{type(exc).__name__}: {exc}"
    else:
        detection_error = None
    report = capabilities.as_dict()
    report["rocm_version"] = (
        capabilities.runtime_version if capabilities.kind is AcceleratorKind.ROCM else None
    )
    report["cuda_version"] = (
        capabilities.runtime_version if capabilities.kind is AcceleratorKind.CUDA else None
    )
    report["graph_replay"] = {
        "supported": capabilities.supports_graphs,
        "disabled_by_env": os.getenv("FREETOKEN_DISABLE_GPU_GRAPHS", "").strip().lower()
        in _TRUE_VALUES,
        "probe": None,
    }
    if detection_error is not None:
        report["detection_error"] = detection_error
    try:
        import torch

        torch_cuda_namespace = bool(torch.cuda.is_available())
        if capabilities.is_gpu and capabilities.supports_graphs:
            report["graph_replay"]["probe"] = _probe_graph_replay(torch, capabilities.index)
    except Exception:
        torch_cuda_namespace = False
    report.update(
        {
            "nvcc": shutil.which("nvcc"),
            "hipcc": shutil.which("hipcc"),
            "rocm_path": os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME"),
            "hip_path": os.environ.get("HIP_PATH"),
            "visible_devices": {
                name: os.environ.get(name)
                for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
                if os.environ.get(name) is not None
            },
            "driver_abi": {
                "runtime": capabilities.runtime_version,
                "torch_cuda_namespace": torch_cuda_namespace,
                "runtime_library": (
                    "amdhip64"
                    if capabilities.kind is AcceleratorKind.ROCM
                    else "cudart"
                    if capabilities.kind is AcceleratorKind.CUDA
                    else None
                ),
            },
        }
    )
    # Keep ABI/library probes separate from Torch's ``cuda`` namespace.  HIP
    # intentionally reuses that namespace, so a namespace check alone cannot
    # distinguish a usable amdhip64 driver from a CPU-only Torch wheel.
    if capabilities.kind is AcceleratorKind.ROCM:
        hip_candidates = [
            "amdhip64.dll",
            "amdhip64_10.dll",
            "amdhip64_7.dll",
            "libamdhip64.so",
            "libamdhip64.so.1",
        ]
        for root_name in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
            root = os.environ.get(root_name)
            if root:
                for subdir in ("bin", "lib"):
                    hip_candidates.extend(
                        str(Path(root) / subdir / name)
                        for name in ("amdhip64.dll", "libamdhip64.so", "libamdhip64.so.1")
                    )
        hip_library = None
        for candidate in hip_candidates:
            try:
                hip_library = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        driver_version = None
        if hip_library is not None:
            try:
                version = ctypes.c_int()
                hip_library.hipDriverGetVersion(ctypes.byref(version))
                driver_version = int(version.value) or None
            except (AttributeError, OSError):
                pass
        report["driver_abi"].update(
            {
                "hip_runtime_loaded": hip_library is not None,
                "hip_driver_version": driver_version,
            }
        )
        library_candidates = {
            "amdhip64": hip_candidates,
            "rocblas": ("rocblas.dll", "librocblas.so", "librocblas.so.4"),
            "rccl": ("rccl.dll", "librccl.so", "librccl.so.1"),
            "rocdevice-libs": ("oclc_isa_version_900.bc", "oclc_isa_version_1100.bc"),
        }
        report["device_libraries"] = {
            name: any(
                bool(ctypes.util.find_library(candidate) or shutil.which(candidate))
                for candidate in candidates
            )
            for name, candidates in library_candidates.items()
        }
    else:
        cuda_library_candidates = (
            "cudart64_130.dll",
            "cudart64_12.dll",
            "libcudart.so",
            "libcudart.so.12",
        )
        report["device_libraries"] = {
            "cudart": any(
                bool(ctypes.util.find_library(candidate) or shutil.which(candidate))
                for candidate in cuda_library_candidates
            ),
        }
    try:
        from freetoken.kernel._toolchain import hip_release, nvcc_release

        report["compiler"] = {
            "hipcc": report["hipcc"],
            "hipcc_version": hip_release(report["hipcc"]) if report["hipcc"] else None,
            "nvcc": report["nvcc"],
            "nvcc_version": nvcc_release(report["nvcc"]) if report["nvcc"] else None,
        }
    except Exception as exc:
        report["compiler_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from freetoken.kernel.pinned import _host_ptr_identity

        report["host_pointer_identity"] = bool(_host_ptr_identity())
        report["host_pointer_mapping"] = report["host_pointer_identity"]
    except Exception as exc:
        report["host_pointer_identity_error"] = f"{type(exc).__name__}: {exc}"

    # Optional runtime probes are isolated so ``ft diagnose`` remains useful
    # on a fresh CPU install and on a compatible-driver build whose optional
    # extension wheels have not been installed yet.
    report["native_extensions"] = {}
    for module_name in ("freetoken.kernel._pinned_tensor", "freetoken.kernel._cpu_moe"):
        try:
            __import__(module_name)
            report["native_extensions"][module_name.rsplit(".", 1)[-1]] = True
        except Exception as exc:
            report["native_extensions"][module_name.rsplit(".", 1)[-1]] = False
            report.setdefault("native_extension_errors", {})[module_name] = f"{type(exc).__name__}: {exc}"

    report["collective_backend"] = "none"
    report["rccl_available"] = False
    try:
        import torch

        distributed = getattr(torch, "distributed", None)
        if distributed is not None and distributed.is_available():
            report["collective_backend"] = (
                "rccl"
                if capabilities.kind is AcceleratorKind.ROCM
                else "nccl"
                if capabilities.kind is AcceleratorKind.CUDA
                else "gloo"
            )
            report["rccl_available"] = bool(
                getattr(distributed, "is_nccl_available", lambda: False)()
            )
    except Exception as exc:
        report["collective_probe_error"] = f"{type(exc).__name__}: {exc}"

    report["backend_probes"] = {
        "flashinfer": False,
        "sgl_kernel": False,
        "attention_default": "triton"
        if capabilities.kind is AcceleratorKind.ROCM
        else "auto",
    }
    try:
        from freetoken.kernel.backend import is_flashinfer_installed, is_sgl_kernel_installed

        report["backend_probes"] = {
            "flashinfer": bool(is_flashinfer_installed()),
            "sgl_kernel": bool(is_sgl_kernel_installed()),
            "attention_default": "triton" if capabilities.kind is AcceleratorKind.ROCM else "auto",
        }
    except Exception as exc:
        report["backend_probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        cache = __import__("freetoken_kernel_cache")
        cache_report = {
            "version": str(getattr(cache, "__version__", "unknown")),
            "compatible": True,
            "architecture": capabilities.architecture,
        }
        try:
            from freetoken.kernel.utils import _kernel_cache_dir

            cache_dir = _kernel_cache_dir()
            cache_report["path"] = str(cache_dir) if cache_dir is not None else None
        except Exception as exc:
            cache_report["compatible"] = False
            cache_report["compatibility_error"] = f"{type(exc).__name__}: {exc}"
        report["kernel_cache"] = cache_report
    except Exception as exc:
        report["kernel_cache"] = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    report["fallback_decisions"] = {
        "attention": "triton" if capabilities.kind is AcceleratorKind.ROCM else "auto",
        "collectives": (
            "rccl"
            if capabilities.kind is AcceleratorKind.ROCM and report.get("rccl_available")
            else "rccl_unavailable"
            if capabilities.kind is AcceleratorKind.ROCM
            else report["collective_backend"]
        ),
        "pynccl": "disabled_on_rocm" if capabilities.kind is AcceleratorKind.ROCM else "eligible",
        "fp8": "native" if capabilities.supports_fp8 else "emulated_or_bf16",
        "fp4": "native" if capabilities.supports_fp4 else "triton_or_bf16",
        "expert_host_copies": "mapped_zero_copy"
        if report.get("host_pointer_mapping")
        else "staged_h2d",
        "gpu_graphs": "enabled" if capabilities.supports_graphs else "disabled",
    }
    return report


__all__ = [
    "AcceleratorKind",
    "DeviceCapabilities",
    "accelerator_report",
    "active_accelerator",
    "detect_device_capabilities",
    "is_cuda",
    "is_rocm",
]
