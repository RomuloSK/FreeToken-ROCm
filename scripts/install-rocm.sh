#!/usr/bin/env bash
# FreeToken ROCm installer (Linux).
#
# This is intentionally separate from ../install.sh: the latter is the stable
# CUDA/Desktop installer and must keep its CUDA index policy. This script picks
# a ROCm channel, detects the GFX target, installs the matching Torch/device
# stack, and then installs FreeToken plus its kernel-cache wheel.
#
# Examples:
#   scripts/install-rocm.sh --channel rocm7.14 --yes
#   ROCM_TORCH_INDEX_URL=https://artifacts.example/mi50 \
#     scripts/install-rocm.sh --channel rocm10-mi50 --artifact-index https://artifacts.example/simple
#
# Environment overrides are useful for private compatible-driver artifacts:
#   FREETOKEN_ROCM_CHANNEL, FREETOKEN_ARTIFACT_INDEX, ROCM_TORCH_INDEX_URL,
#   FREETOKEN_ROCM_ARCH, FREETOKEN_WHEEL, FREETOKEN_KERNEL_CACHE_WHEEL,
#   FREETOKEN_KERNEL_CACHE_SPEC, FREETOKEN_HOME, FREETOKEN_PYTHON,
#   FREETOKEN_ASSUME_YES.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHANNEL="${FREETOKEN_ROCM_CHANNEL:-rocm7.14}"
ARTIFACT_INDEX="${FREETOKEN_ARTIFACT_INDEX:-https://pypi.org/simple}"
TORCH_INDEX="${ROCM_TORCH_INDEX_URL:-}"
PYTHON_BIN="${FREETOKEN_PYTHON:-python3}"
FT_HOME="${FREETOKEN_HOME:-$HOME/.freetoken-rocm}"
VENV="${FREETOKEN_VENV:-$FT_HOME/venv}"
ARCH="${FREETOKEN_ROCM_ARCH:-}"
WHEEL="${FREETOKEN_WHEEL:-}"
KERNEL_CACHE_WHEEL="${FREETOKEN_KERNEL_CACHE_WHEEL:-}"
KERNEL_CACHE_SPEC="${FREETOKEN_KERNEL_CACHE_SPEC:-freetoken-kernel-cache}"
ASSUME_YES="${FREETOKEN_ASSUME_YES:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/install-rocm.sh [options]

Options:
  --channel NAME          rocm7.14 (default) or rocm10-mi50
  --artifact-index URL    index serving FreeToken/ROCm artifacts
  --torch-index URL       index serving the matching ROCm Torch wheel
  --arch gfxNNNN          override GFX detection (also FREETOKEN_ROCM_ARCH)
  --wheel PATH|URL        runtime wheel; otherwise install freetoken from index
  --kernel-cache PATH|URL matching kernel-cache wheel; otherwise use package index
  --venv PATH             managed virtual environment (default ~/.freetoken-rocm/venv)
  --python PATH           Python used to create the venv (default python3)
  -y, --yes               do not prompt
  -h, --help              show this help

MI50/ROCm 10.x requires a compatible-driver Torch/device index. Pass it with
--torch-index or ROCM_TORCH_INDEX_URL; PyPI's CUDA/CPU Torch wheels are rejected.
EOF
}

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warning]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --channel) (($# >= 2)) || die "--channel needs a value"; CHANNEL="$2"; shift 2 ;;
    --artifact-index) (($# >= 2)) || die "--artifact-index needs a URL"; ARTIFACT_INDEX="$2"; shift 2 ;;
    --torch-index) (($# >= 2)) || die "--torch-index needs a URL"; TORCH_INDEX="$2"; shift 2 ;;
    --arch) (($# >= 2)) || die "--arch needs a value"; ARCH="$2"; shift 2 ;;
    --wheel) (($# >= 2)) || die "--wheel needs a path or URL"; WHEEL="$2"; shift 2 ;;
    --kernel-cache) (($# >= 2)) || die "--kernel-cache needs a path, URL, or package"; KERNEL_CACHE_WHEEL="$2"; shift 2 ;;
    --venv) (($# >= 2)) || die "--venv needs a path"; VENV="$2"; shift 2 ;;
    --python) (($# >= 2)) || die "--python needs a path"; PYTHON_BIN="$2"; shift 2 ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

case "${CHANNEL,,}" in
  rocm7.14|rocm-7.14|rocm7.14-linux)
    CHANNEL="rocm7.14-linux"
    TORCH_VERSION="2.11.0"
    CONSTRAINTS="$ROOT/constraints/rocm-7.14-linux.txt"
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.14}"
    ;;
  rocm10-mi50|rocm10.x-mi50|rocm10.x-mi50-linux|rocm10-mi50-linux)
    CHANNEL="rocm10.x-mi50-linux"
    TORCH_VERSION="2.11.0"
    CONSTRAINTS="$ROOT/constraints/rocm-10.x-mi50-linux.txt"
    # There is no public PyPI ROCm 10.x MI50 wheel. Requiring this value avoids
    # silently installing a same-version CUDA wheel that cannot load on gfx906.
    [[ -n "$TORCH_INDEX" ]] || die "ROCm 10.x MI50 needs --torch-index or ROCM_TORCH_INDEX_URL for the compatible-driver artifacts"
    ;;
  *) die "unsupported ROCm channel '$CHANNEL' (use rocm7.14 or rocm10-mi50)" ;;
esac

detect_arch() {
  [[ -n "$ARCH" ]] && return 0
  if command -v rocminfo >/dev/null 2>&1; then
    ARCH="$(rocminfo 2>/dev/null | sed -nE 's/^[[:space:]]*Name:[[:space:]]*(gfx[0-9a-z]+).*/\1/p' | head -1 || true)"
  fi
  if [[ -z "$ARCH" ]] && command -v hipconfig >/dev/null 2>&1; then
    ARCH="$(hipconfig --amdgpu-target 2>/dev/null | sed -nE 's/.*(gfx[0-9a-z]+).*/\1/p' | head -1 || true)"
  fi
}

detect_arch
if [[ -n "$ARCH" && ! "$ARCH" =~ ^gfx[0-9a-z]+$ ]]; then
  die "invalid GFX target '$ARCH' (expected e.g. gfx1102 or gfx906)"
fi
if [[ "$CHANNEL" == rocm10.x-mi50-linux && -n "$ARCH" && "$ARCH" != gfx906 ]]; then
  die "the $CHANNEL channel is for MI50/gfx906, but detected '$ARCH'; set FREETOKEN_ROCM_ARCH only when the device is really gfx906"
fi
if [[ -n "$ARCH" ]]; then
  say "detected AMD target: $ARCH"
else
  warn "rocminfo/hipconfig did not report a GFX target; continuing for package-only setup (set FREETOKEN_ROCM_ARCH for artifact selection)"
  ARCH="unknown"
fi
export FREETOKEN_ROCM_CHANNEL="$CHANNEL"
export FREETOKEN_ROCM_ARCH="$ARCH"

[[ -f "$CONSTRAINTS" ]] || die "missing constraints file: $CONSTRAINTS"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"

infer_kernel_cache() {
  [[ -n "$KERNEL_CACHE_WHEEL" ]] && return 0
  case "$WHEEL" in
    http://*|https://*|"") return 0 ;;
  esac
  [[ -e "$WHEEL" ]] || return 0
  local wheel_dir candidate found count
  wheel_dir="$(cd "$(dirname "$WHEEL")" && pwd -P)"
  found=""; count=0
  for candidate in "$wheel_dir"/freetoken_kernel_cache-*.whl "$wheel_dir"/freetoken-kernel-cache-*.whl; do
    [[ -f "$candidate" ]] || continue
    found="$candidate"; count=$((count + 1))
  done
  ((count <= 1)) || die "multiple kernel-cache wheels found next to $WHEEL; pass --kernel-cache explicitly"
  [[ "$count" == 1 ]] && KERNEL_CACHE_WHEEL="$found"
}

infer_kernel_cache
if [[ -n "$WHEEL" ]]; then
  RUNTIME_SPEC="$WHEEL[rocm]"
else
  RUNTIME_SPEC="freetoken[rocm]"
fi
if [[ -n "$KERNEL_CACHE_WHEEL" ]]; then
  KERNEL_SPEC="$KERNEL_CACHE_WHEEL"
else
  KERNEL_SPEC="$KERNEL_CACHE_SPEC"
fi

say "creating clean venv at $VENV"
"$PYTHON_BIN" -m venv --clear "$VENV"
VENV_PY="$VENV/bin/python"
[[ -x "$VENV_PY" ]] || die "venv creation did not produce $VENV_PY"

if command -v uv >/dev/null 2>&1; then
  # A custom MI50 index can carry a same-version local Torch build.  uv's
  # default first-index strategy would stop at PyPI's CUDA candidate before it
  # considers that artifact, so use the explicit multi-index strategy here.
  INSTALL=(uv pip install --index-strategy unsafe-best-match --python "$VENV")
else
  "$VENV_PY" -m pip --version >/dev/null 2>&1 || die "venv has no pip; install uv or recreate Python with ensurepip"
  INSTALL=("$VENV_PY" -m pip install)
fi

# PyPI remains the source for pure-Python dependencies. The artifact index may
# host both the FreeToken wheels and the compatible-driver wheels; the Torch
# index is added last so a custom MI50 wheel can satisfy the exact constraint.
INDEX_ARGS=(--index-url https://pypi.org/simple)
if [[ "$ARTIFACT_INDEX" != https://pypi.org/simple && "$ARTIFACT_INDEX" != https://pypi.org/simple/ ]]; then
  INDEX_ARGS+=(--extra-index-url "$ARTIFACT_INDEX")
fi
if [[ -n "$TORCH_INDEX" && "$TORCH_INDEX" != "$ARTIFACT_INDEX" ]]; then
  INDEX_ARGS+=(--extra-index-url "$TORCH_INDEX")
fi

say "installing Torch $TORCH_VERSION from $TORCH_INDEX (channel $CHANNEL)"
"${INSTALL[@]}" --constraint "$CONSTRAINTS" "${INDEX_ARGS[@]}" "torch==$TORCH_VERSION" "triton==3.6.0" "apache-tvm-ffi==0.1.13.post3"
say "installing $RUNTIME_SPEC and $KERNEL_SPEC for $ARCH"
"${INSTALL[@]}" --constraint "$CONSTRAINTS" "${INDEX_ARGS[@]}" "$RUNTIME_SPEC" "$KERNEL_SPEC"

FT_BIN="$VENV/bin/ft"
[[ -x "$FT_BIN" ]] || die "installation finished but $FT_BIN is missing"
if ! "$VENV_PY" -c 'import torch; assert torch.version.hip, torch.__version__' >/dev/null 2>&1; then
  die "installed Torch is not a HIP/ROCm build; check --torch-index (a CUDA/CPU wheel must not be used)"
fi
if ! "$FT_BIN" --help >/dev/null 2>&1; then
  warn "ft --help failed; inspect the environment with: $FT_BIN --help"
fi
if "$FT_BIN" diagnose --json >/dev/null 2>&1; then
  say "diagnostics passed"
else
  warn "ft diagnose was unavailable or reported a problem; run '$FT_BIN diagnose --json' after checking the driver"
fi

cat <<EOF

FreeToken ROCm installation complete.

  channel        $CHANNEL
  GFX target     $ARCH
  virtualenv     $VENV
  binary         $FT_BIN

Activate it with:
  source "$VENV/bin/activate"
  ft diagnose --json
  ft serve --model <path> --gpu 0
EOF
