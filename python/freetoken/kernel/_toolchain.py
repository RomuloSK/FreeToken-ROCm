"""CUDA/HIP toolchain and Torch-runtime consistency checks.

The module is intentionally standalone: ``setup.py`` and the kernel-cache
build backend load it by path before the freetoken package is importable.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

# Keep the historical CUDA spelling as the public constant while accepting the
# accelerator-neutral name for new ROCm deployments.
ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
TOOLCHAIN_ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_TOOLCHAIN_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _torch():
    import torch

    return torch


def is_rocm_runtime() -> bool:
    try:
        version = getattr(_torch(), "version", None)
        return bool(getattr(version, "hip", None))
    except Exception:
        return False


def _rocm_root() -> Path | None:
    candidates = [
        os.environ.get("HIP_PATH"),
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
    ]
    try:
        import torch.utils.cpp_extension as cpp_extension

        candidates.append(getattr(cpp_extension, "ROCM_HOME", None))
    except Exception:
        pass
    for raw in candidates:
        if raw:
            root = Path(raw).expanduser()
            if root.exists():
                return root
    hipcc = shutil.which("hipcc")
    if hipcc:
        return Path(hipcc).resolve().parent.parent
    return None


def nvcc_path() -> str | None:
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            candidate = Path(CUDA_HOME) / "bin" / "nvcc"
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass
    return shutil.which("nvcc")


def hipcc_path() -> str | None:
    root = _rocm_root()
    if root is not None:
        for candidate in (root / "bin" / "hipcc", root / "bin" / "hipcc.exe"):
            if candidate.exists():
                return str(candidate)
    return shutil.which("hipcc")


# Private spelling retained for callers that imported the original CUDA-only
# helper while the implementation grows an accelerator-neutral toolchain API.
_nvcc_path = nvcc_path


def nvcc_release(nvcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else None


def hip_release(hipcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run([hipcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    text = proc.stdout + "\n" + proc.stderr
    match = re.search(r"HIP version:\s*(\d+)\.(\d+)", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"rocm(?: version)?\s*(\d+)\.(\d+)", text, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def torch_cuda_major() -> int | None:
    version = getattr(_torch(), "version", None)
    cuda = getattr(version, "cuda", None)
    return int(str(cuda).split(".")[0]) if cuda else None


def torch_rocm_release() -> tuple[int, int] | None:
    version = getattr(_torch(), "version", None)
    hip = getattr(version, "hip", None)
    if not hip:
        return None
    match = re.search(r"(\d+)\.(\d+)", str(hip))
    return (int(match.group(1)), int(match.group(2))) if match else None


def rocm_arch_list() -> list[str]:
    """Resolve explicit ROCm targets for an AOT build or the active device."""

    for name in (
        "TVM_FFI_ROCM_ARCH_LIST",
        "FREETOKEN_ROCM_ARCH_LIST",
        "FREETOKEN_ROCM_ARCH",
        "PYTORCH_ROCM_ARCH",
    ):
        raw = os.environ.get(name, "").replace(",", " ").replace(";", " ")
        values = [
            value.strip().lower().split(":", 1)[0]
            for value in raw.split()
            if value.strip().lower().startswith("gfx")
        ]
        if values:
            return values

    try:
        torch = _torch()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            for attr in ("gcnArchName", "gcn_arch_name", "architecture"):
                value = getattr(props, attr, None)
                if isinstance(value, bytes):
                    value = value.decode("ascii", "replace")
                if value and str(value).lower().startswith("gfx"):
                    return [str(value).lower().split(":", 1)[0]]
    except Exception:
        pass
    return []


def compiler_arch_flags() -> list[str]:
    if not is_rocm_runtime():
        return []
    return [f"--offload-arch={arch}" for arch in rocm_arch_list()]


@functools.cache
def check_toolchain_matches_torch() -> None:
    """Reject mismatched CUDA majors; require an explicit HIP compiler on ROCm."""

    if any(
        os.getenv(name, "").strip().lower() in _TRUE_VALUES
        for name in (ALLOW_MISMATCH_ENV, TOOLCHAIN_ALLOW_MISMATCH_ENV)
    ):
        return
    torch = _torch()
    if is_rocm_runtime():
        hipcc = hipcc_path()
        if hipcc is None:
            raise RuntimeError(
                "Torch was built with ROCm/HIP, but hipcc was not found. Set HIP_PATH "
                "or ROCM_PATH to the matching ROCm 7.14/10.x SDK."
            )
        torch_release = torch_rocm_release()
        compiler_release = hip_release(hipcc)
        if torch_release and compiler_release and torch_release[0] != compiler_release[0]:
            raise RuntimeError(
                f"hipcc {compiler_release[0]}.{compiler_release[1]} does not match "
                f"Torch HIP {torch_release[0]}.{torch_release[1]}. Set {ALLOW_MISMATCH_ENV}=1 "
                "only when the driver/runtime ABI is intentionally compatible."
            )
        return

    torch_major = torch_cuda_major()
    if torch_major is None:
        return
    nvcc = nvcc_path()
    if nvcc is None:
        return
    release = nvcc_release(nvcc)
    if release is None:
        return
    if release[0] != torch_major:
        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels linking "
            f"libcudart.so.{release[0]}, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda} (libcudart.so.{torch_major}). Install a CUDA "
            f"{torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )


# Backward-compatible name used by existing JIT callers.
check_nvcc_matches_torch = check_toolchain_matches_torch


def runtime_include_dirs() -> list[str]:
    if not is_rocm_runtime():
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            if CUDA_HOME:
                root = Path(CUDA_HOME)
                return [str(root / "include")]
        except Exception:
            pass
        return []
    root = _rocm_root()
    return [str(root / "include")] if root is not None and (root / "include").exists() else []


def runtime_library_dirs() -> list[str]:
    if not is_rocm_runtime():
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            if CUDA_HOME:
                root = Path(CUDA_HOME)
                return [str(path) for path in (root / "lib64", root / "lib") if path.exists()]
        except Exception:
            pass
        return []
    root = _rocm_root()
    if root is None:
        return []
    return [str(path) for path in (root / "lib", root / "lib64") if path.exists()]


def runtime_library_name() -> str:
    return "amdhip64" if is_rocm_runtime() else "cudart"


__all__ = [
    "ALLOW_MISMATCH_ENV",
    "TOOLCHAIN_ALLOW_MISMATCH_ENV",
    "check_nvcc_matches_torch",
    "check_toolchain_matches_torch",
    "compiler_arch_flags",
    "hipcc_path",
    "hip_release",
    "is_rocm_runtime",
    "nvcc_path",
    "nvcc_release",
    "rocm_arch_list",
    "runtime_include_dirs",
    "runtime_library_dirs",
    "runtime_library_name",
    "torch_cuda_major",
    "torch_rocm_release",
]
