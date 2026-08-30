"""Small Triton portability shims shared by CUDA and AMD backends."""

from __future__ import annotations

import torch
import triton


_TORCH_VERSION = getattr(torch, "version", None)
if (
    getattr(_TORCH_VERSION, "hip", None) is None
    and getattr(_TORCH_VERSION, "cuda", None) is not None
):
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
else:
    # Grid dependency control is a CUDA PTX feature. Keep parser-time symbols
    # available for kernels whose constexpr PDL branch is disabled on ROCm.
    @triton.jit
    def gdc_wait():
        return

    @triton.jit
    def gdc_launch_dependents():
        return


__all__ = ["gdc_wait", "gdc_launch_dependents"]
