<#
.SYNOPSIS
  Install FreeToken's ROCm Radeon lane on Windows.

.DESCRIPTION
  Creates a clean virtual environment, installs the ROCm Torch 2.12 device
  wheel from the selected vendor/artifact index, and installs FreeToken's ROCm
  extra plus the matching kernel-cache wheel. MI50/ROCm 10.x is Linux-only and
  is rejected here deliberately.

  The default Torch URL is a placeholder for the ROCm 7.14 channel. Set
  ROCM_TORCH_INDEX_URL for AMD's published index or an internal mirror.
#>
[CmdletBinding()]
param(
  [string]$Channel = $(if ($env:FREETOKEN_ROCM_CHANNEL) { $env:FREETOKEN_ROCM_CHANNEL } else { 'rocm7.14' }),
  [string]$ArtifactIndex = $(if ($env:FREETOKEN_ARTIFACT_INDEX) { $env:FREETOKEN_ARTIFACT_INDEX } else { 'https://pypi.org/simple' }),
  [string]$TorchIndex = $(if ($env:ROCM_TORCH_INDEX_URL) { $env:ROCM_TORCH_INDEX_URL } else { 'https://download.pytorch.org/whl/rocm7.14' }),
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
    $TorchIndex = 'https://download.pytorch.org/whl/rocm7.14'
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
if ($Channel.ToLowerInvariant() -notin @('rocm7.14', 'rocm-7.14', 'rocm7.14-windows')) {
  Fail "unsupported Windows ROCm channel '$Channel' (use rocm7.14)"
}
$Channel = 'rocm7.14-windows'
$Root = Split-Path -Parent $PSScriptRoot
$Constraints = Join-Path $Root 'constraints\rocm-7.14-windows.txt'
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
if ($TorchIndex -and $TorchIndex -ne $ArtifactIndex) {
  $IndexArgs += @('--extra-index-url', $TorchIndex)
}

function Install-Packages([string[]]$Packages) {
  if ($script:uv) {
    & $script:uv.Source pip install --index-strategy unsafe-best-match --python $script:VenvPython --constraint $script:Constraints @script:IndexArgs @Packages
  } else {
    & $script:VenvPython -m pip install --constraint $script:Constraints @script:IndexArgs @Packages
  }
  if ($LASTEXITCODE -ne 0) { Fail "package installation failed: $($Packages -join ' ')" }
}

$TorchVersion = '2.12.0'
Say "installing Torch $TorchVersion from $TorchIndex (channel $Channel)"
Install-Packages @("torch==$TorchVersion", 'apache-tvm-ffi==0.1.13.post3', 'flashlib==0.3.0')

if ($Wheel) { $RuntimeSpec = "$Wheel[rocm]" } else { $RuntimeSpec = 'freetoken[rocm]' }
if ($KernelCache) { $KernelSpec = $KernelCache } else { $KernelSpec = 'freetoken-kernel-cache' }
Say "installing $RuntimeSpec and $KernelSpec for $Arch"
Install-Packages @($RuntimeSpec, $KernelSpec)

$ft = Join-Path $Venv 'Scripts\ft.exe'
if (-not (Test-Path -LiteralPath $ft)) { Fail "installation finished but $ft is missing" }
& $VenvPython -c 'import torch; assert torch.version.hip, torch.__version__' *> $null
if ($LASTEXITCODE -ne 0) { Fail 'installed Torch is not a HIP/ROCm build; check -TorchIndex' }
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
