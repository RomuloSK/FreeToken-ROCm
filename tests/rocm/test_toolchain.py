"""CPU-only tests for CUDA/HIP compiler and cache compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_hip_release_accepts_hipcc_version_output(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    monkeypatch.setattr(
        tc.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="HIP version: 10.2.0\n", stderr="", returncode=0
        ),
    )
    assert tc.hip_release("hipcc") == (10, 2)


def test_hip_release_accepts_rocm_version_output(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    monkeypatch.setattr(
        tc.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="ROCm version 7.14.1", returncode=0),
    )
    assert tc.hip_release("hipcc") == (7, 14)


def test_compiler_arch_flags_are_explicit_for_mi50(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    monkeypatch.setattr(tc, "is_rocm_runtime", lambda: True)
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH_LIST", "gfx906,gfx1102")

    assert tc.rocm_arch_list() == ["gfx906", "gfx1102"]
    assert tc.compiler_arch_flags() == ["--offload-arch=gfx906", "--offload-arch=gfx1102"]


def test_rocm_arch_environment_precedence_is_deterministic(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx950")
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH_LIST", "gfx906 gfx1102")
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx90a,gfx942")

    assert tc.rocm_arch_list() == ["gfx90a", "gfx942"]


def test_single_rocm_arch_override_is_used_for_jit(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    monkeypatch.delenv("TVM_FFI_ROCM_ARCH_LIST", raising=False)
    monkeypatch.delenv("FREETOKEN_ROCM_ARCH_LIST", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx906")

    assert tc.rocm_arch_list() == ["gfx906"]


def test_runtime_library_selection_tracks_accelerator(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    monkeypatch.setattr(tc, "is_rocm_runtime", lambda: True)
    assert tc.runtime_library_name() == "amdhip64"
    monkeypatch.setattr(tc, "is_rocm_runtime", lambda: False)
    assert tc.runtime_library_name() == "cudart"


def test_rocm_toolchain_major_mismatch_has_actionable_error(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="10.0.0", cuda=None), __version__="2.11.0+rocm10"
    )
    monkeypatch.setattr(tc, "_torch", lambda: fake_torch)
    monkeypatch.setattr(tc, "is_rocm_runtime", lambda: True)
    monkeypatch.setattr(tc, "hipcc_path", lambda: "hipcc")
    monkeypatch.setattr(tc, "hip_release", lambda path: (7, 14))
    tc.check_toolchain_matches_torch.cache_clear()

    with pytest.raises(RuntimeError, match=r"hipcc 7\.14 does not match Torch HIP 10\.0"):
        tc.check_toolchain_matches_torch()


def test_rocm_toolchain_mismatch_can_be_explicitly_overridden(monkeypatch):
    from freetoken.kernel import _toolchain as tc

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="10.0.0", cuda=None), __version__="2.11.0+rocm10"
    )
    monkeypatch.setattr(tc, "_torch", lambda: fake_torch)
    monkeypatch.setattr(tc, "is_rocm_runtime", lambda: True)
    monkeypatch.setattr(tc, "hipcc_path", lambda: "hipcc")
    monkeypatch.setattr(tc, "hip_release", lambda path: (7, 14))
    monkeypatch.setenv(tc.ALLOW_MISMATCH_ENV, "1")
    tc.check_toolchain_matches_torch.cache_clear()

    tc.check_toolchain_matches_torch()
