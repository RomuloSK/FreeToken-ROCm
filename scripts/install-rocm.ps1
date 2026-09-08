<#
.SYNOPSIS
  Install FreeToken's ROCm Radeon lane on Windows.

.DESCRIPTION
  Creates a clean virtual environment, installs AMD's published ROCm 7.2.1
  Torch 2.9.1+rocm7.2.1 cp312 device wheel from repo.radeon.com, and installs
  FreeToken's ROCm extra from the local checkout plus the matching
  kernel-cache wheel. MI50/ROCm 10.x is Linux-only and is rejected here
  deliberately.

  Defaults target the verified AMD Windows index
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ (ROCm 7.2.1,
  Torch 2.9.1+rocm7.2.1, Python 3.12 only). Set ROCM_TORCH_INDEX_URL for an
  internal mirror. There is no ROCm 7.14 Windows lane on repo.radeon.com;
  7.14 stays Linux-only (scripts/install-rocm.sh).
#>
[CmdletBinding()]
param(
  [string]$Channel = $(if ($env:FREETOKEN_ROCM_CHANNEL) { $env:FREETOKEN_ROCM_CHANNEL } else { 'rocm7.2' }),
  [string]$ArtifactIndex = $(if ($env:FREETOKEN_ARTIFACT_INDEX) { $env:FREETOKEN_ARTIFACT_INDEX } else { 'https://pypi.org/simple' }),
  [string]$TorchIndex = $(if ($env:ROCM_TORCH_INDEX_URL) { $env:ROCM_TORCH_INDEX_URL } else { 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/' }),
  [string]$Arch = $(if ($env:FREETOKEN_ROCM_ARCH) { $env:FREETOKEN_ROCM_ARCH } else { '' }),
  [string]$Wheel = $(if ($env:FREETOKEN_WHEEL) { $env:FREETOKEN_WHEEL } else { '' }),
  [string]$KernelCache = $(if ($env:FREETOKEN_KERNEL_CACHE_WHEEL) { $env:FREETOKEN_KERNEL_CACHE_WHEEL } else { '' }),
  [string]$Venv = $(if ($env:FREETOKEN_VENV) { $env:FREETOKEN_VENV } else { (Join-Path $HOME '.freetoken-rocm\venv') }),
  [string]$Python = $(if ($env:FREETOKEN_PYTHON) { $env:FREETOKEN_PYTHON } else { 'py' }),
  [switch]$Yes
)

$ErrorActionPreference = 'Stop'

# A workflow may pass an empty input explicitly. Re-apply the environment or
# channel default in that case instead of accidentally falling through to
# PyPI's CUDA/CPU Torch index.
if (-not $TorchIndex) {
  if ($env:ROCM_TORCH_INDEX_URL) {
    $TorchIndex = $env:ROCM_TORCH_INDEX_URL
  } else {
    $TorchIndex = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/'
  }
}
if (-not $ArtifactIndex) {
  $ArtifactIndex = 'https://pypi.org/simple'
}

function Fail([string]$Message) {
  [Console]::Error.WriteLine("[error] $Message")
  exit 1
}

function Say([string]$Message) {
  Write-Host "==> $Message" -ForegroundColor Cyan
}

if ($Channel.ToLowerInvariant() -in @('rocm10-mi50', 'rocm10.x-mi50', 'rocm10.x-mi50-windows')) {
  Fail 'ROCm 10.x MI50/gfx906 is Linux-only; use scripts/install-rocm.sh with --channel rocm10-mi50.'
}
if ($Channel.ToLowerInvariant() -notin @('rocm7.2', 'rocm-7.2', 'rocm7.2-windows', 'rocm7.2.1', 'rocm-7.2.1', 'rocm7.2.1-windows')) {
  Fail "unsupported Windows ROCm channel '$Channel' (use rocm7.2)"
}
$Channel = 'rocm7.2-windows'
$Root = Split-Path -Parent $PSScriptRoot
$Constraints = Join-Path $Root 'constraints\rocm-7.2-windows.txt'
if (-not (Test-Path -LiteralPath $Constraints)) { Fail "missing constraints file: $Constraints" }

if (-not $Arch) {
  $rocminfo = Get-Command rocminfo -ErrorAction SilentlyContinue
  if ($rocminfo) {
    $text = (& $rocminfo.Source 2>$null) -join "`n"
    $match = [regex]::Match($text, 'Name:\s*(gfx[0-9a-z]+)')
    if ($match.Success) { $Arch = $match.Groups[1].Value }
  }
}
if (-not $Arch) {
  $hipconfig = Get-Command hipconfig -ErrorAction SilentlyContinue
  if ($hipconfig) {
    $text = (& $hipconfig.Source --amdgpu-target 2>$null) -join "`n"
    $match = [regex]::Match($text, '(gfx[0-9a-z]+)')
    if ($match.Success) { $Arch = $match.Groups[1].Value }
  }
}
if ($Arch -and $Arch -notmatch '^gfx[0-9a-z]+$') { Fail "invalid GFX target '$Arch' (expected e.g. gfx1102)" }
if (-not $Arch) {
  Write-Warning 'rocminfo/hipconfig did not report a GFX target; continuing with package installation. Set FREETOKEN_ROCM_ARCH to select an artifact explicitly.'
  $Arch = 'unknown'
}
Say "detected AMD target: $Arch"
$env:FREETOKEN_ROCM_CHANNEL = $Channel
$env:FREETOKEN_ROCM_ARCH = $Arch

$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pythonCommand) { Fail "Python launcher not found: $Python" }
Say "creating clean venv at $Venv"
& $pythonCommand.Source -m venv --clear $Venv
if ($LASTEXITCODE -ne 0) { Fail 'venv creation failed' }
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) { Fail "venv creation did not produce $VenvPython" }

$uv = Get-Command uv -ErrorAction SilentlyContinue
$IndexArgs = @('--index-url', 'https://pypi.org/simple')
if ($ArtifactIndex -and $ArtifactIndex -notmatch '^https://pypi\.org/simple/?$') {
  $IndexArgs += @('--extra-index-url', $ArtifactIndex)
}
# NOTE: $TorchIndex (repo.radeon.com flat directory) is deliberately NOT added
# as an extra index: it is not a PEP 503 simple index, so pip/uv resolution
# cannot use it. Torch/SDK artifacts are installed via direct URLs below.

function Install-Packages([string[]]$Packages) {
  if ($script:uv) {
    & $script:uv.Source pip install --python $script:VenvPython --constraint $script:Constraints @script:IndexArgs @Packages
  } else {
    & $script:VenvPython -m pip install --constraint $script:Constraints @script:IndexArgs @Packages
  }
  if ($LASTEXITCODE -ne 0) { Fail "package installation failed: $($Packages -join ' ')" }
}

& $VenvPython -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if ($LASTEXITCODE -ne 0) { Fail 'AMD ROCm 7.2.1 Torch wheels are cp312-only; recreate the venv with Python 3.12 (py -3.12)' }

# AMD native-Windows order per rocm.docs.amd.com (PyTorch via PIP on Windows):
# 1) ROCm SDK wheels + rocm metapackage tarball (provides rocm[libraries],
#    which the Torch wheel depends on), 2) Torch/torchaudio/torchvision
# wheels. Direct URLs: the AMD directory is a flat file listing, not a
# PEP 503 index, so version-spec resolution cannot see these artifacts.
# --no-cache(-dir): wheels total ~2.5GB; never duplicate them into the cache.
$TorchVersion = '2.9.1+rocm7.2.1'
$Base = $TorchIndex.TrimEnd('/')
$RocmSdk = @(
  "$Base/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
  "$Base/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
  "$Base/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
  "$Base/rocm-7.2.1.tar.gz"
)
$TorchWheels = @(
  "$Base/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
  "$Base/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
  "$Base/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
)
Say "installing ROCm 7.2.1 SDK wheels (channel $Channel)"
if ($script:uv) {
  & $script:uv.Source pip install --no-cache --python $script:VenvPython @RocmSdk
} else {
  & $script:VenvPython -m pip install --no-cache-dir @RocmSdk
}
if ($LASTEXITCODE -ne 0) { Fail 'ROCm SDK wheel installation failed' }
Say "installing Torch $TorchVersion wheels (channel $Channel)"
if ($script:uv) {
  & $script:uv.Source pip install --no-cache --python $script:VenvPython @TorchWheels
} else {
  & $script:VenvPython -m pip install --no-cache-dir @TorchWheels
}
if ($LASTEXITCODE -ne 0) { Fail 'Torch wheel installation failed' }
Install-Packages @('apache-tvm-ffi==0.1.13.post3', 'flashlib==0.3.0')
& $VenvPython -c 'import torch; assert (getattr(torch.version, "hip", None) or torch.cuda.is_available()), torch.__version__' *> $null
if ($LASTEXITCODE -ne 0) { Fail 'installed Torch has no ROCm/HIP backend (torch.version.hip empty and torch.cuda unavailable); check -TorchIndex and the Adrenalin driver' }

if ($Wheel) { $RuntimeSpec = "$Wheel[rocm]" } else { $RuntimeSpec = "$Root[rocm]" }
if ($KernelCache) { $KernelSpec = $KernelCache } else { $KernelSpec = 'freetoken-kernel-cache' }
Say "installing $RuntimeSpec for $Arch (local source; PyPI has no win_amd64 ROCm wheels)"
Install-Packages @($RuntimeSpec)
Say "installing $KernelSpec for $Arch (best effort)"
# Best effort: Install-Packages exits the script via Fail(), so a missing
# win_amd64 kernel-cache artifact must not go through it. Install directly
# and warn instead of failing.
if ($script:uv) {
  & $script:uv.Source pip install --python $script:VenvPython @script:IndexArgs @($KernelSpec)
} else {
  & $script:VenvPython -m pip install @script:IndexArgs @($KernelSpec)
}
if ($LASTEXITCODE -ne 0) { Write-Warning 'kernel-cache wheel install failed (ok on Windows without a published win_amd64 artifact)' }

$ft = Join-Path $Venv 'Scripts\ft.exe'
if (-not (Test-Path -LiteralPath $ft)) { Fail "installation finished but $ft is missing" }
& $ft --help *> $null
if ($LASTEXITCODE -ne 0) { Write-Warning 'ft --help failed; inspect the environment manually' }
& $ft diagnose --json *> $null
if ($LASTEXITCODE -eq 0) { Say 'diagnostics passed' } else { Write-Warning "ft diagnose reported a problem; run '$ft diagnose --json'" }

Write-Host "`nFreeToken ROCm installation complete.`n"
Write-Host "  channel      $Channel"
Write-Host "  GFX target   $Arch"
Write-Host "  virtualenv   $Venv"
Write-Host "  binary       $ft"
Write-Host "`nRun: & '$ft' diagnose --json"
