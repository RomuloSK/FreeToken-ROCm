"""CPU-only tests for accelerator discovery and ROCm capability policy.

The project intentionally keeps ROCm devices behind PyTorch's ``torch.cuda``
namespace.  These tests install a tiny fake torch module so the policy can be
tested on builders that do not have an AMD device (or a CUDA driver).
"""

from __future__ import annotations

import json
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest


@lru_cache(maxsize=1)
def _accelerator_module():
    """Load the standalone capability module without importing optional HF deps.

    ``freetoken.utils.__init__`` imports the Hugging Face integration for normal
    application use.  Capability discovery itself is intentionally standalone,
    so metadata tests should remain runnable in a minimal build environment.
    """

    path = Path(__file__).parents[2] / "python" / "freetoken" / "utils" / "accelerator.py"
    spec = importlib.util.spec_from_file_location("_freetoken_test_accelerator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules during
    # class creation when a module is loaded via importlib.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeCuda:
    CUDAGraph = type("CUDAGraph", (), {})

    def __init__(self, props, *, available=True):
        self._props = props
        self._available = available

    def is_available(self):
        return self._available

    def current_device(self):
        return 0

    def get_device_properties(self, index):
        return self._props[index] if isinstance(self._props, (list, tuple)) else self._props


def _fake_torch(*, hip="10.0", props=None, available=True):
    if props is None:
        props = SimpleNamespace(name="AMD Instinct MI50", gcnArchName="gfx906", warp_size=64)
    return SimpleNamespace(
        __version__="2.11.0+rocm10",
        version=SimpleNamespace(hip=hip, cuda=None),
        cuda=_FakeCuda(props, available=available),
        distributed=object(),
    )


@pytest.fixture
def fake_torch(monkeypatch):
    """Replace only the torch module imported by accelerator.py for one test."""

    accelerator = _accelerator_module()

    def install(module):
        monkeypatch.setitem(sys.modules, "torch", module)
        accelerator.detect_device_capabilities.cache_clear()

    yield install
    accelerator.detect_device_capabilities.cache_clear()


def test_cpu_capabilities_are_safe_without_an_accelerator():
    accelerator = _accelerator_module()

    accelerator.detect_device_capabilities.cache_clear()
    caps = accelerator.detect_device_capabilities()

    assert caps.kind is accelerator.AcceleratorKind.CPU
    assert caps.architecture == "cpu"
    assert caps.is_gpu is False
    assert caps.supports_fp8 is False
    assert caps.supports_graphs is False


def test_mi50_gfx906_uses_wave64_and_conservative_precision(fake_torch):
    accelerator = _accelerator_module()

    fake_torch(_fake_torch())
    caps = accelerator.detect_device_capabilities()

    assert caps.kind is accelerator.AcceleratorKind.ROCM
    assert caps.target == "gfx906"
    assert caps.warp_size == 64
    # MI50 has no native FP8/FP4 policy in FreeToken; kernels must use a
    # correctness fallback instead of assuming newer matrix instructions.
    assert caps.supports_fp8 is False
    assert caps.supports_fp4 is False
    assert caps.supports_graphs is True
    assert caps.supports_collectives is True


@pytest.mark.parametrize("architecture, expected_wave", [("gfx1102", 32), ("gfx950", 64)])
def test_rocm_architecture_and_wave_size(fake_torch, architecture, expected_wave):
    accelerator = _accelerator_module()

    props = SimpleNamespace(name="AMD GPU", gcnArchName=architecture, warp_size=0)
    fake_torch(_fake_torch(props=props))
    caps = accelerator.detect_device_capabilities()

    assert caps.architecture == architecture
    assert caps.warp_size == expected_wave
    assert caps.supports_fp8 is (architecture == "gfx950")


def test_rocm_architecture_can_use_hsa_override_when_properties_are_incomplete(fake_torch, monkeypatch):
    accelerator = _accelerator_module()

    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
    props = SimpleNamespace(name="AMD GPU", warp_size=64)
    fake_torch(_fake_torch(props=props))

    assert accelerator.detect_device_capabilities().architecture == "gfx906"


def test_rocm_architecture_uses_freetoken_override_when_properties_are_incomplete(fake_torch, monkeypatch):
    accelerator = _accelerator_module()

    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1102")
    props = SimpleNamespace(name="AMD GPU", warp_size=32)
    fake_torch(_fake_torch(props=props))

    assert accelerator.detect_device_capabilities().architecture == "gfx1102"


def test_accelerator_report_is_json_serializable(fake_torch, monkeypatch):
    accelerator = _accelerator_module()

    fake_torch(_fake_torch())
    monkeypatch.setattr(accelerator.shutil, "which", lambda name: f"/opt/bin/{name}")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    report = accelerator.accelerator_report()

    # Diagnostics are consumed by bug-report tooling, so all values must be
    # serializable even when the optional pinned extension is absent.
    encoded = json.dumps(report, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["accelerator"] == "rocm"
    assert decoded["architecture"] == "gfx906"
    assert decoded["hipcc"] == "/opt/bin/hipcc"
    assert decoded["visible_devices"] == {"HIP_VISIBLE_DEVICES": "0,1"}
