from __future__ import annotations

import functools
from typing import Tuple

from .accelerator import AcceleratorKind, detect_device_capabilities


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def device_arch(index: int | None = None) -> str:
    """Return the normalized compiler target for the active device."""

    return detect_device_capabilities(index).architecture


def is_rocm_arch(index: int | None = None) -> bool:
    return detect_device_capabilities(index).kind is AcceleratorKind.ROCM


def is_cuda_arch(index: int | None = None) -> bool:
    return detect_device_capabilities(index).kind is AcceleratorKind.CUDA


def is_gfx_family(prefix: str, index: int | None = None) -> bool:
    return is_rocm_arch(index) and device_arch(index).startswith(prefix.lower())


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """capability >= (major, minor). Open-ended: newer archs also pass. Only use this
    for family-portable features (e.g. PDL); arch-specific kernels (sm_90a/sm_100a
    cubins) need the closed is_smXX_family checks below."""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def _is_arch_family(major: int) -> bool:
    arch = _get_torch_cuda_version()
    return arch is not None and arch[0] == major


def is_sm90_family() -> bool:
    """Exactly major 9 (Hopper). For sm_90a-only kernels (e.g. FA3)."""
    return _is_arch_family(9)


def is_sm100_family() -> bool:
    """Exactly major 10 (datacenter Blackwell). For sm_100a/103a-only kernels
    (e.g. trtllm-gen) that consumer Blackwell (sm_120/121) cannot run."""
    return _is_arch_family(10)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
