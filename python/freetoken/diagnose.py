"""Environment diagnostics for GPU backend reports and bug submissions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


def _load_accelerator_report():
    """Load diagnostics without importing the optional HF integration.

    ``freetoken.utils`` normally imports ``transformers`` for model loading,
    but support bundles must also work immediately after a minimal runtime or
    on a broken accelerator installation.  The capability module is deliberately
    standalone, so load it directly when that package-level import is unavailable.
    """

    try:
        from freetoken.utils.accelerator import accelerator_report

        return accelerator_report
    except ModuleNotFoundError:
        path = Path(__file__).with_name("utils") / "accelerator.py"
        spec = importlib.util.spec_from_file_location("_freetoken_diagnose_accelerator", path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.accelerator_report


# Kept as a module attribute for callers/tests that inject a deterministic
# report.  The loader itself remains dependency-light as described above.
accelerator_report = _load_accelerator_report()


def main(argv: list[str] | None = None, *, prog: str = "ft diagnose") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = accelerator_report()
    report.update(
        {
            "freetoken_python": sys.version,
            "platform": platform.platform(),
            "python_executable": sys.executable,
        }
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    # Keep CPU-only support bundles usable while making a broken accelerator
    # driver visible to CI/install scripts through the process status.
    return 1 if report.get("detection_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
