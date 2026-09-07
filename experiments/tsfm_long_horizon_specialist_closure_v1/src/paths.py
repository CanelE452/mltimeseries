"""Canonical paths for TSFM-LONG-HORIZON-SPECIALIST-CLOSURE-v1.

The source study's directories are named here too, but only ever opened for
reading. Nothing in this package writes below `SOURCE_*`.
"""

from __future__ import annotations

import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]

EXP = REPO / "experiments" / "tsfm_long_horizon_specialist_closure_v1"
CONFIGS = EXP / "configs"
RESULTS = REPO / "results" / "tsfm_long_horizon_specialist_closure_v1"
RUNS = REPO / "runs" / "tsfm_long_horizon_specialist_closure_v1"
DATA_EXTERNAL = REPO / "data_external" / "tsfm_long_horizon_specialist_closure_v1"

PREDICTORS = RUNS / "predictors"
FORECASTS = RUNS / "forecasts"
LOGS = RUNS / "logs"
HF_CACHE = REPO / "runs" / "tsfm_benchmark_gap_discovery_v1" / "hf_cache"

# Read-only source study.
SOURCE_EXP = REPO / "experiments" / "tsfm_benchmark_gap_discovery_v1"
SOURCE_RESULTS = REPO / "results" / "tsfm_benchmark_gap_discovery_v1"
SOURCE_DATA = REPO / "data_external" / "tsfm_benchmark_gap_discovery_v1"
SOURCE_PREDICTIONS = REPO / "runs" / "tsfm_benchmark_gap_discovery_v1" / "predictions"
SOURCE_BRANCH = "tsfm-benchmark-gap-discovery-v1"

BRANCH = "tsfm-long-horizon-specialist-closure-v1"

# `datasets` names its lock files after the full cache path, so a deep cache
# directory pushes the lock past Windows' 260-character limit and every load of a
# long-named config fails with WinError 206. Model weights stay in the source
# study's cache; dataset arrow files go somewhere short.
DATASETS_CACHE = REPO / "runs" / "dscache"

for _p in (RESULTS, RUNS, DATA_EXTERNAL, PREDICTORS, FORECASTS, LOGS, DATASETS_CACHE):
    _p.mkdir(parents=True, exist_ok=True)


def configure_environment() -> None:
    """Environment this host needs before `fev` or any model is imported."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DATASETS_CACHE))
    os.environ.setdefault("CUDA_HOME", "C:/dummy" if os.name == "nt" else "/usr")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


configure_environment()
