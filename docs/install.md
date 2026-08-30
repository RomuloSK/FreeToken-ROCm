# Install

FreeToken has separate CUDA and ROCm dependency channels. Pick the channel that
matches the installed driver before installing a wheel; do not mix a CUDA Torch
wheel with a ROCm runtime.

## CUDA requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

`freetoken[cuda]` is the explicit spelling for new installations;
`freetoken[accel]` remains a backwards-compatible alias.

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## ROCm 7.14 (Linux and Windows Radeon)

The ROCm installer selects the Torch/device channel, detects the `gfx*` target,
and installs the matching FreeToken and kernel-cache artifacts. It accepts an
artifact index for private or pre-release wheels.

Linux:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
ROCM_TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.14 \
  scripts/install-rocm.sh --channel rocm7.14 --yes
source ~/.freetoken-rocm/venv/bin/activate
ft diagnose --json
```

Windows PowerShell (Windows 11 Radeon lane, Torch 2.12):

```powershell
git clone https://github.com/FlashML-org/FreeToken.git
Set-Location FreeToken
$env:ROCM_TORCH_INDEX_URL = 'https://download.pytorch.org/whl/rocm7.14'
& .\scripts\install-rocm.ps1 -Channel rocm7.14 -Yes
& "$HOME\.freetoken-rocm\venv\Scripts\ft.exe" diagnose --json
```

For a direct installation after installing the vendor Torch/device wheel into
an existing venv, use the ROCm extra and the channel constraints:

```bash
uv pip install --constraint constraints/rocm-7.14-linux.txt \
  --extra-index-url "$ROCM_TORCH_INDEX_URL" 'freetoken[rocm]'
```

The `rocm-7.14-windows.txt` constraints file pins the Windows Torch 2.12 lane.
The `rocm-7.14-linux.txt` file pins Linux Torch 2.11 and Triton 3.6.

## ROCm 10.x-compatible MI50 (Linux only)

MI50 (`gfx906`) uses the compatible ROCm 10.x driver/device artifacts supplied
by your driver channel. It is intentionally not enabled by the Windows
installer. Pass both the artifact index and Torch index so the installer cannot
silently fall back to PyPI's CUDA/CPU Torch build:

```bash
ROCM_TORCH_INDEX_URL=https://artifacts.example/rocm10-mi50/simple \
  FREETOKEN_ARTIFACT_INDEX=https://artifacts.example/freetoken/simple \
  scripts/install-rocm.sh --channel rocm10-mi50 --arch gfx906 --yes
source ~/.freetoken-rocm/venv/bin/activate
ft diagnose --json
```

The matching constraints are recorded in
`constraints/rocm-10.x-mi50-linux.txt`. The compatible driver must expose the
standard HIP ABI, `amdhip64`, and an RCCL-capable PyTorch distribution. If
`rocminfo` is unavailable, set `FREETOKEN_ROCM_ARCH=gfx906` explicitly.

This lane is currently build-ready with hardware validation pending. FreeToken
will not publish the MI50 first-class support claim until the two-card RCCL
acceptance suite passes three consecutive nightlies.

## Artifact and cache selection

`FREETOKEN_WHEEL` and `FREETOKEN_KERNEL_CACHE_WHEEL` may point at local files or
URLs when testing a build that is not yet published. The installer also accepts
`--artifact-index`; the index is expected to serve architecture-compatible
runtime and kernel-cache wheels. Run `ft diagnose --json` after installation to
verify the detected backend, GFX target, compiler, graph capabilities, and cache
compatibility before loading a model.

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
