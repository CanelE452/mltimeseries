"""Record the exact repository and machine state this study started from."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from . import paths


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    wanted = [
        "fev",
        "chronos-forecasting",
        "tirex-2",
        "timesfm",
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "scipy",
        "xlstm",
        "huggingface_hub",
        "accelerate",
    ]
    out = {}
    for name in wanted:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "NOT_INSTALLED"
    return out


def _gpus() -> list[dict]:
    try:
        import torch

        return [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_mb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**2
                ),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # pragma: no cover - environment dependent
        return [{"error": repr(exc)}]


def build() -> dict:
    import torch

    status = _git("status", "--short")
    tracked_dirty = [
        line for line in status.splitlines() if not line.startswith("??")
    ]
    free = shutil.disk_usage(paths.REPO).free
    return {
        "experiment": "TSFM-BENCHMARK-GAP-DISCOVERY-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": "CanelE452/mltimeseries",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "base_sha": _git("rev-parse", "HEAD"),
        "origin_main_sha": _git("rev-parse", "origin/main"),
        "dirty_before_start": {
            "tracked_modifications": tracked_dirty,
            "untracked": [
                line[3:] for line in status.splitlines() if line.startswith("??")
            ],
            "hard_stop_triggered": bool(tracked_dirty),
        },
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "gpus": _gpus(),
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "nvcc_available": shutil.which("nvcc") is not None,
            "msvc_cl_available": shutil.which("cl") is not None,
        },
        "free_disk_bytes": free,
        "free_disk_gb": round(free / 1024**3, 1),
        "packages": _package_versions(),
    }


def main() -> None:
    state = build()
    out = paths.RESULTS / "BASE_STATE.json"
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    if state["dirty_before_start"]["hard_stop_triggered"]:
        raise SystemExit(
            "HARD STOP: tracked modifications present before the study started."
        )


if __name__ == "__main__":
    main()
