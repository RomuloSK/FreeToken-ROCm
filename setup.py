from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME


ROOT = Path(__file__).parent
PROJECT_INCLUDE = ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include"


def _load_toolchain():
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FreeToken toolchain helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


toolchain = _load_toolchain()
is_rocm = toolchain.is_rocm_runtime()
skip_extensions = any(
    os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
    for name in ("FREETOKEN_SKIP_NATIVE_EXT", "FREETOKEN_SKIP_CUDA_EXT")
)

include_dirs = [str(PROJECT_INCLUDE), *toolchain.runtime_include_dirs()]
library_dirs = toolchain.runtime_library_dirs()
libraries = [toolchain.runtime_library_name()]

extra_compile_args: list[str] = ["-O3", "-std=c++17"]
if is_rocm:
    # The standard HIP ABI is selected by this macro; the same sources compile
    # unchanged against CUDA when it is absent.
    extra_compile_args.append("-D__HIP_PLATFORM_AMD__")

ext_modules = []
if not skip_extensions and (is_rocm or CUDA_HOME is not None):
    toolchain.check_toolchain_matches_torch()
    ext_modules = [
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=["python/freetoken/kernel/csrc/pinned_tensor.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        ),
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=["python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args
            + ([] if sys.platform == "win32" else ["-pthread"]),
        ),
    ]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
