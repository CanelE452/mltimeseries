"""Canonical paths for TSFM-BENCHMARK-GAP-DISCOVERY-v1."""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]

EXP = REPO / "experiments" / "tsfm_benchmark_gap_discovery_v1"
CONFIGS = EXP / "configs"
RESULTS = REPO / "results" / "tsfm_benchmark_gap_discovery_v1"
RUNS = REPO / "runs" / "tsfm_benchmark_gap_discovery_v1"
DATA_EXTERNAL = REPO / "data_external" / "tsfm_benchmark_gap_discovery_v1"
DOCS = REPO / "_docs" / "notes" / "tsfm_topics" / "benchmark_gap_discovery_v1"

HF_CACHE = RUNS / "hf_cache"
PRED_CACHE = RUNS / "predictions"

for _p in (RESULTS, RUNS, DATA_EXTERNAL, DOCS, HF_CACHE, PRED_CACHE):
    _p.mkdir(parents=True, exist_ok=True)
