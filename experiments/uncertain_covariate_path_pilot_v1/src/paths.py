"""Where each run writes.

The real run writes the filenames instruction section 30 asks for, in
`results/uncertain_covariate_path_pilot_v1/`. A dry run writes the same
filenames into a sibling `_dryrun` directory, following the convention the
earlier pilots in this repository already use, so a plumbing check can never be
mistaken for a result.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FULL = ROOT / "results/uncertain_covariate_path_pilot_v1"
DRYRUN = ROOT / "results/uncertain_covariate_path_pilot_v1_dryrun"


def out_dir(tag: str = "full") -> Path:
    path = FULL if tag == "full" else DRYRUN
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir(tag: str = "full") -> Path:
    path = ROOT / "runs/uncertain_covariate_path_pilot_v1" / tag
    path.mkdir(parents=True, exist_ok=True)
    return path
