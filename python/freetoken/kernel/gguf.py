"""Borrowed llama.cpp GGUF dequant/GEMM GPU kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _rocm_runtime() -> bool:
    # Keep this module importable in CPU-only environments.  PyTorch ROCm
    # intentionally reports its devices through ``torch.cuda``; the HIP
    # runtime marker is the stable way to distinguish it from CUDA.
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def _extension_name() -> str:
    """Separate torch-extension build caches by accelerator/runtime/target."""

    version = getattr(getattr(torch, "version", None), "hip", None) or getattr(
        getattr(torch, "version", None), "cuda", None
    )
    runtime = re.sub(r"[^0-9a-z]+", "_", str(version or "unknown").lower()).strip("_")
    if _rocm_runtime():
        arch = (
            os.getenv("FREETOKEN_KERNEL_CACHE_ARCHES")
            or os.getenv("FREETOKEN_ROCM_ARCH")
            or os.getenv("FREETOKEN_ROCM_ARCH_LIST", "")
        )
        arch_values = arch.replace(",", " ").replace(";", " ").split()
        if not arch_values:
            try:
                from freetoken.kernel._toolchain import rocm_arch_list

                arch_values = rocm_arch_list()
            except Exception:
                arch_values = []
        arch = (arch_values or ["gfxauto"])[0].lower().split(":", 1)[0]
        return f"freetoken_gguf_kernels_rocm{runtime}_{arch}"
    return f"freetoken_gguf_kernels_cuda{runtime}"


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

@functools.cache
def _module():
    from torch.utils.cpp_extension import load
    from freetoken.kernel._toolchain import (
        check_toolchain_matches_torch,
        compiler_arch_flags,
        runtime_include_dirs,
    )

    check_toolchain_matches_torch()

    is_rocm = _rocm_runtime()
    # ``extra_cuda_cflags`` is the name used by torch's extension loader for
    # both nvcc and hipcc.  hipcc accepts the common optimisation/constexpr
    # flags and uses ``--offload-arch`` for gfx targets.
    extra_cuda_cflags = ["-O3"]
    if not is_rocm:
        extra_cuda_cflags.append("--expt-relaxed-constexpr")
    else:
        extra_cuda_cflags.extend(["-DUSE_ROCM=1", *compiler_arch_flags()])

    host_cxx = None if is_rocm else _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    include_paths = [str(_CSRC), *runtime_include_dirs()]
    return load(
        name=_extension_name(),
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
