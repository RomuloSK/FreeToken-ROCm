"""Tests for ROCm-facing CLI, backend and visibility contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_diagnose_json_uses_machine_readable_report(monkeypatch, capsys):
    pytest.importorskip("transformers")
    from freetoken import diagnose

    monkeypatch.setattr(
        diagnose,
        "accelerator_report",
        lambda: {"accelerator": "rocm", "architecture": "gfx906", "runtime_version": "10.0"},
    )
    assert diagnose.main(["--json"]) == 0
    output = capsys.readouterr().out
    assert __import__("json").loads(output)["architecture"] == "gfx906"


def test_graph_max_bs_is_generic_alias_for_cuda_graph_option(tmp_path):
    pytest.importorskip("transformers")
    from freetoken.server.args import parse_args

    generic, _ = parse_args(
        ["--model-path", str(tmp_path), "--dtype", "bfloat16", "--graph-max-bs", "7"]
    )
    cuda_named, _ = parse_args(
        ["--model-path", str(tmp_path), "--dtype", "bfloat16", "--cuda-graph-max-bs", "7"]
    )
    assert generic.cuda_graph_max_bs == cuda_named.cuda_graph_max_bs == 7


def _full_model_config():
    return SimpleNamespace(
        model_type="test",
        single_stream_only=False,
        is_moe=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=2,
    )


def test_cuda_only_attention_backend_is_rejected_on_rocm(monkeypatch):
    pytest.importorskip("transformers")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import engine
    from freetoken.engine.config import EngineConfig

    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="fi",
    )
    object.__setattr__(config, "model_config", _full_model_config())

    with pytest.raises(RuntimeError, match="CUDA-only.*not available on ROCm"):
        engine._adjust_config(config)


@pytest.mark.parametrize(
    "visibility, expected",
    [("2,0", "GPU-CCC"), ("GPU-BBB,GPU-AAA", "GPU-BBB")],
)
def test_gpu_selection_honors_hip_and_rocr_visibility(monkeypatch, visibility, expected):
    from freetoken import gpu_select

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if visibility.startswith("GPU-"):
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", visibility)
        spec = "0"
    else:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", visibility)
        spec = "0"
    monkeypatch.setattr(
        gpu_select, "_nvml_uuids", lambda: ["GPU-AAA", "GPU-BBB", "GPU-CCC"]
    )

    assert gpu_select.resolve_gpu_uuids([spec]) == (expected,)
