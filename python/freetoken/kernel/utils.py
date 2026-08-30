from __future__ import annotations

import importlib
import functools
import os
import pathlib
import re
import sys
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
KERNEL_CACHE_PACKAGE = "freetoken_kernel_cache"
KERNEL_CACHE_DIR_ENV = "FREETOKEN_KERNEL_CACHE_DIR"
DISABLE_KERNEL_CACHE_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE"
DISABLE_KERNEL_CACHE_VERSION_CHECK_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK"
DISABLE_JIT_ENV = "FREETOKEN_DISABLE_JIT"
_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_HIP_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS = []


def _cuda_cflags(extra: List[str]) -> List[str]:
    """CUDA/HIP device flags for a kernel build.

    During the multi-arch AOT cache build,
    `TVM_FFI_CUDA_ARCH_LIST` (e.g. "8.6 8.9 9.0 10.0 12.0") makes tvm-ffi emit a SASS cubin
    (`-gencode ...code=sm_XX`) for each listed arch — but NO PTX. We add the PTX of the HIGHEST
    listed arch so a GPU newer than any listed one (no matching SASS) still runs via the driver's
    PTX→SASS JIT (driver-only, no CUDA toolkit). One top PTX suffices: the loader always
    JIT-forwards from the highest compatible PTX. When the env is unset (runtime JIT), this is a
    no-op and tvm-ffi targets only the local GPU."""
    from freetoken.kernel._toolchain import compiler_arch_flags, is_rocm_runtime

    if is_rocm_runtime():
        # hipcc rejects nvcc's ``--expt-relaxed-constexpr`` and ``-gencode``
        # spellings.  ``--offload-arch`` is accepted on Linux and Windows and
        # also works with the ROCm 10.x gfx906 toolchain.
        return DEFAULT_HIP_CFLAGS + compiler_arch_flags() + extra

    flags = DEFAULT_CUDA_CFLAGS + extra
    arch_list = os.getenv("TVM_FFI_CUDA_ARCH_LIST", "").split()
    if arch_list:
        def _rank(a: str) -> int:
            major, minor = a.rstrip("a").split(".")
            return int(major) * 100 + int(minor)

        cc = max(arch_list, key=_rank).rstrip("a").replace(".", "")
        flags = flags + [f"-gencode=arch=compute_{cc},code=compute_{cc}"]
    return flags
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


@functools.lru_cache(maxsize=1)
def _rocm_cache_identity() -> str:
    """Stable suffix preventing HIP JIT/AOT collisions across runtimes/targets."""

    try:
        from freetoken.kernel._toolchain import hip_release, hipcc_path, torch_rocm_release

        release = torch_rocm_release()
        runtime = f"{release[0]}.{release[1]}" if release else "unknown"
        compiler = hip_release(hipcc_path()) if hipcc_path() else None
        compiler_tag = f"hipcc{compiler[0]}.{compiler[1]}" if compiler else "hipccunknown"
    except Exception:
        runtime, compiler_tag = "unknown", "hipccunknown"
    raw_arch = (
        os.getenv("FREETOKEN_KERNEL_CACHE_ARCHES")
        or os.getenv("FREETOKEN_ROCM_ARCH")
        or os.getenv("FREETOKEN_ROCM_ARCH_LIST", "")
    )
    arch_values = raw_arch.replace(",", " ").replace(";", " ").split()
    if not arch_values:
        try:
            from freetoken.kernel._toolchain import rocm_arch_list

            arch_values = rocm_arch_list()
        except Exception:
            arch_values = []
    # A fat ROCm cache can contain several offload images.  Give it one stable
    # identity so a gfx906 runtime can discover the same artifact as gfx1102;
    # single-target JIT/AOT builds retain the concrete architecture in the key.
    normalized_arches = [
        value.lower().split(":", 1)[0]
        for value in arch_values
        if value.lower().startswith("gfx")
    ]
    arch = "gfxfat" if len(set(normalized_arches)) > 1 else (normalized_arches or ["gfxauto"])[0]
    abi = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"rocm{runtime}_{arch}_{compiler_tag}_{abi}"


def _make_name(*args: str) -> str:
    base = "_".join(str(arg) for arg in args)
    try:
        from freetoken.kernel._toolchain import is_rocm_runtime

        if is_rocm_runtime():
            return f"freetoken__{_rocm_cache_identity()}_{base}"
    except Exception:
        pass
    return f"freetoken__{base}"


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _freetoken_version() -> str:
    from freetoken.version import __version__

    return __version__


def _version_parts(version: str) -> Tuple[str, List[str]]:
    """Split a PEP 440 version string into (release, local segments):
    "0.1.1+cu130.g3f01615" -> ("0.1.1", ["cu130", "g3f01615"])."""
    base, _, local = version.partition("+")
    return base, local.split(".") if local else []


def _build_stamps(segments: List[str]) -> set[str]:
    """The `g<sha>` commit-stamp tokens of a local version segment list
    (stamped by scripts/build-release-wheels.sh)."""
    return {s for s in segments if re.fullmatch(r"g[0-9a-f]{7,40}", s)}


def _kernel_cache_version_ok(cache_version: str, runtime_version: str) -> bool:
    """Same release -- and, when both sides carry a `g<sha>` stamp, the same build.

    The cache wheel extends the runtime's version with local segments (`+cu130`,
    `.g<sha>`), so the old string-prefix test cannot pair a stamped runtime with its
    cache; and comparing the stamps rejects a runtime/cache pair from two different
    builds, which bare release numbers (both `0.1.1`) could never detect. Either side
    may lack a stamp (dev builds) -- then only the release part is compared."""
    cache_base, cache_local = _version_parts(cache_version)
    runtime_base, runtime_local = _version_parts(runtime_version)
    if cache_base != runtime_base:
        return False
    cache_stamps = _build_stamps(cache_local)
    runtime_stamps = _build_stamps(runtime_local)
    return not (cache_stamps and runtime_stamps and cache_stamps != runtime_stamps)


def _kernel_cache_dir() -> pathlib.Path | None:
    if _env_enabled(DISABLE_KERNEL_CACHE_ENV):
        return None

    override = os.getenv(KERNEL_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()

    try:
        package = importlib.import_module(KERNEL_CACHE_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name == KERNEL_CACHE_PACKAGE:
            return None
        raise

    package_version = str(getattr(package, "__version__", "0.0.0+unknown"))
    runtime_version = _freetoken_version()
    if not _env_enabled(DISABLE_KERNEL_CACHE_VERSION_CHECK_ENV):
        if not _kernel_cache_version_ok(package_version, runtime_version):
            raise RuntimeError(
                "freetoken-kernel-cache version "
                f"{package_version!r} does not match freetoken version {runtime_version!r}"
            )
        cache_cuda = re.search(r"\+cu(\d{2,})", package_version)
        if cache_cuda is not None:
            from freetoken.kernel._toolchain import torch_cuda_major

            cache_major = int(cache_cuda.group(1)[:-1])
            torch_major = torch_cuda_major()
            if torch_major is not None and cache_major != torch_major:
                raise RuntimeError(
                    f"freetoken-kernel-cache {package_version!r} was built for CUDA "
                    f"{cache_major}.x but torch runs CUDA {torch_major}.x -- install "
                    "the kernel-cache wheel matching this torch build"
                )

        cache_rocm = re.search(r"(?:^|[+.])rocm(\d+(?:\.\d+)*)", package_version)
        if cache_rocm is not None:
            from freetoken.kernel._toolchain import is_rocm_runtime, torch_rocm_release

            if not is_rocm_runtime():
                raise RuntimeError(
                    f"freetoken-kernel-cache {package_version!r} is a ROCm cache but "
                    "Torch is not running a HIP build"
                )
            torch_rocm = torch_rocm_release()
            if torch_rocm is not None:
                # A ROCm 10.x cache must never be paired with a ROCm 7.x runtime.
                expected_major = str(torch_rocm[0])
                if cache_rocm.group(1).split(".", 1)[0] != expected_major:
                    raise RuntimeError(
                        f"freetoken-kernel-cache {package_version!r} targets ROCm "
                        f"{cache_rocm.group(1)}, but Torch reports HIP {torch_rocm[0]}.{torch_rocm[1]}"
                    )
            cache_arch = re.search(r"(?:^|[+.])(gfx[0-9a-z]+)(?:$|[+.])", package_version, re.IGNORECASE)
            if cache_arch is not None and cache_arch.group(1).lower() not in {"gfxfat", "gfxauto"}:
                from freetoken.utils.arch import device_arch

                runtime_arch = device_arch()
                # A package built for a concrete target is safe to load only
                # on that target.  Unknown architecture reports are treated as
                # non-actionable (vendor Torch builds occasionally omit
                # gcnArchName), while a real mismatch fails before dlopen.
                if (
                    runtime_arch.startswith("gfx")
                    and not runtime_arch.startswith("gfx_unknown")
                    and runtime_arch.lower() != cache_arch.group(1).lower()
                ):
                    raise RuntimeError(
                        f"freetoken-kernel-cache {package_version!r} targets "
                        f"{cache_arch.group(1)}, but Torch reports {runtime_arch}"
                    )

    get_jit_cache_dir = getattr(package, "get_jit_cache_dir", None)
    if get_jit_cache_dir is None:
        raise RuntimeError(f"{KERNEL_CACHE_PACKAGE} does not expose get_jit_cache_dir()")
    return pathlib.Path(get_jit_cache_dir()).expanduser()


def _load_prebuilt(name: str) -> Module | None:
    cache_dir = _kernel_cache_dir()
    if cache_dir is None:
        if _env_enabled(DISABLE_JIT_ENV):
            raise RuntimeError(
                "JIT compilation is disabled by FREETOKEN_DISABLE_JIT, "
                f"but no prebuilt kernel cache is configured for {name!r}"
            )
        return None

    names = [name]
    if "gfx" in name:
        # Multi-architecture ROCm wheels use the ``gfxfat`` identity.  A
        # runtime with a concrete device target first tries its exact JIT/AOT
        # name, then falls back to that fat artifact.
        fat_name = re.sub(r"gfx[0-9a-z]+(?=_hipcc)", "gfxfat", name, count=1)
        if fat_name != name:
            names.append(fat_name)
    for candidate in names:
        for suffix in (".so", ".pyd", ".dll"):
            module_path = cache_dir / candidate / f"{candidate}{suffix}"
            if module_path.exists():
                import tvm_ffi

                return tvm_ffi.load_module(str(module_path))

    if _env_enabled(DISABLE_JIT_ENV):
        raise RuntimeError(
            "JIT compilation is disabled by FREETOKEN_DISABLE_JIT, "
            f"but prebuilt kernel {name!r} was not found in {cache_dir / name}"
        )
    return None


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    name = _make_name(*args)
    prebuilt = _load_prebuilt(name)
    if prebuilt is not None:
        return prebuilt

    if cuda_files:
        from freetoken.kernel._toolchain import check_toolchain_matches_torch

        check_toolchain_matches_torch()

    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(
        name,
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=_cuda_cflags(extra_cuda_cflags),
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    name = _make_name(*args)
    prebuilt = _load_prebuilt(name)
    if prebuilt is not None:
        return prebuilt

    if cuda_files or cuda_wrappers:
        from freetoken.kernel._toolchain import check_toolchain_matches_torch

        check_toolchain_matches_torch()

    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=_cuda_cflags(extra_cuda_cflags),
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
