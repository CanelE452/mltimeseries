"""The failure descriptors D1-D9, fixed before results are read (Section 15).

Only these conditions are examined. Thresholds that depend on the task pool are
tertiles of the pool or of the discovery set, both computable from metadata and
visible history alone - no test data, no model score.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from . import paths

DESCRIPTORS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")

DESCRIPTOR_LABELS = {
    "D1": "horizon ratio",
    "D2": "frequency",
    "D3": "target dimensionality",
    "D4": "future covariates",
    "D5": "past covariates",
    "D6": "zero fraction / intermittency",
    "D7": "missingness",
    "D8": "train-only distribution shift",
    "D9": "seasonal strength",
}


def _visible_history(task) -> dict[str, np.ndarray]:
    """Per-series statistics from the earliest window's visible past only.

    Using window 0 means the statistics come from the shortest history any model
    is given, and never from a horizon the model has to predict.
    """
    window = task.get_window(0, num_proc=1)
    past, _ = window.get_input_data()
    stats = {
        "zero_fraction": [],
        "missing_fraction": [],
        "shift_median": [],
        "shift_scale": [],
        "seasonal_strength": [],
        "length": [],
    }
    seasonality = int(task.seasonality)
    for column in task.target_columns:
        for values in past[column]:
            series = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(series)
            stats["length"].append(series.size)
            stats["missing_fraction"].append(1.0 - finite.mean() if series.size else 1.0)
            observed = series[finite]
            stats["zero_fraction"].append(
                float(np.mean(observed == 0.0)) if observed.size else np.nan
            )
            if observed.size >= 8:
                half = observed.size // 2
                first, second = observed[:half], observed[half:]
                scale = np.std(observed)
                scale = scale if scale > 0 else 1.0
                stats["shift_median"].append(abs(np.median(second) - np.median(first)) / scale)
                first_scale, second_scale = np.std(first), np.std(second)
                denominator = max(first_scale, 1e-12)
                stats["shift_scale"].append(abs(second_scale - first_scale) / denominator)
            else:
                stats["shift_median"].append(np.nan)
                stats["shift_scale"].append(np.nan)
            if seasonality > 1 and observed.size > 3 * seasonality:
                differenced = observed[seasonality:] - observed[:-seasonality]
                total = np.var(observed[seasonality:])
                stats["seasonal_strength"].append(
                    float(max(0.0, 1.0 - np.var(differenced) / total)) if total > 0 else np.nan
                )
            else:
                stats["seasonal_strength"].append(np.nan)
    return {key: np.asarray(value, dtype=np.float64) for key, value in stats.items()}


def build_task_metadata(splits: tuple[str, ...] = ("discovery", "confirmation")) -> pd.DataFrame:
    """task_metadata.csv: the descriptor values, computed from train-visible data."""
    import fev

    raw = yaml.safe_load(
        (paths.DATA_EXTERNAL / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv")
    selected = json.loads((paths.RESULTS / "selected_tasks.json").read_text(encoding="utf-8"))
    wanted: set[str] = set()
    for split in splits:
        wanted |= set(selected[f"{split}_tasks"])

    records = []
    for _, row in pool[pool.task_uid.isin(wanted)].iterrows():
        task = fev.Task(**raw[int(row.yaml_index)])
        task.load_full_dataset(num_proc=1)
        stats = _visible_history(task)
        with np.errstate(invalid="ignore"):
            records.append(
                {
                    "task_uid": row.task_uid,
                    "split": "discovery" if row.task_uid in set(selected["discovery_tasks"]) else "confirmation",
                    "domain": row.domain,
                    "freq": row.freq,
                    "freq_bucket": row.freq_bucket,
                    "horizon": row.horizon,
                    "seasonality": row.seasonality,
                    "n_series": row.n_series,
                    "n_targets": row.n_targets,
                    "n_known_cov": row.n_known_cov,
                    "n_past_cov": row.n_past_cov,
                    "horizon_to_median_length": row.horizon_to_median_length,
                    "visible_context_median": float(np.median(stats["length"])),
                    "horizon_to_context_ratio": float(row.horizon / np.median(stats["length"])),
                    "zero_fraction": float(np.nanmean(stats["zero_fraction"])),
                    "missing_fraction": float(np.nanmean(stats["missing_fraction"])),
                    "shift_median": float(np.nanmean(stats["shift_median"])),
                    "shift_scale": float(np.nanmean(stats["shift_scale"])),
                    "seasonal_strength": float(np.nanmean(stats["seasonal_strength"])),
                }
            )
    return pd.DataFrame(records)


def assign_buckets(metadata: pd.DataFrame, thresholds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach D1-D9 buckets. Thresholds come from the discovery split only.

    Passing `thresholds` reuses the discovery cut points on the confirmation
    split, which is what Section 21 requires: the same condition definition.
    """
    metadata = metadata.copy()
    discovery = metadata[metadata.split == "discovery"]
    if thresholds is None:
        thresholds = {}
        for column in ("horizon_to_context_ratio", "shift_median", "missing_fraction"):
            low, high = discovery[column].quantile([1 / 3, 2 / 3]).tolist()
            thresholds[column] = {"low": float(low), "high": float(high)}

    def tertile(value: float, cuts: dict) -> str:
        if not np.isfinite(value):
            return "NA"
        if value <= cuts["low"]:
            return "low"
        if value <= cuts["high"]:
            return "mid"
        return "high"

    metadata["D1_horizon_ratio"] = metadata.horizon_to_context_ratio.map(
        lambda v: {"low": "short", "mid": "medium", "high": "long", "NA": "NA"}[
            tertile(v, thresholds["horizon_to_context_ratio"])
        ]
    )
    metadata["D2_frequency"] = metadata.freq_bucket
    metadata["D3_dimensionality"] = metadata.n_targets.map(
        lambda n: "1" if n == 1 else ("2_16" if n <= 16 else "gt16")
    )
    metadata["D4_future_covariates"] = np.where(metadata.n_known_cov > 0, "available", "none")
    metadata["D5_past_covariates"] = np.where(metadata.n_past_cov > 0, "available", "none")
    metadata["D6_zero_fraction"] = metadata.zero_fraction.map(
        lambda v: "NA" if not np.isfinite(v) else ("lt0.1" if v < 0.1 else ("0.1_0.5" if v <= 0.5 else "gt0.5"))
    )
    metadata["D7_missingness"] = metadata.missing_fraction.map(
        lambda v: tertile(v, thresholds["missing_fraction"])
    )
    metadata["D8_shift"] = metadata.shift_median.map(
        lambda v: tertile(v, thresholds["shift_median"])
    )
    metadata["D9_seasonal_strength"] = metadata.seasonal_strength.map(
        lambda v: "NA" if not np.isfinite(v) else ("weak" if v < 0.3 else ("moderate" if v < 0.7 else "strong"))
    )
    return metadata, thresholds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["discovery"],
        choices=["discovery", "confirmation"],
        help="Confirmation descriptors are built only when the holdout is opened, so that "
        "nothing derived from it can reach the discovery analysis.",
    )
    args = parser.parse_args()
    metadata = build_task_metadata(tuple(args.splits))
    existing = paths.RESULTS / "descriptor_thresholds.json"
    reuse = (
        json.loads(existing.read_text(encoding="utf-8"))["cuts"]
        if existing.exists() and "discovery" not in args.splits
        else None
    )
    if reuse is not None:
        metadata, thresholds = assign_buckets(metadata, reuse)
        previous = pd.read_csv(paths.RESULTS / "task_metadata.csv")
        metadata = pd.concat(
            [previous[~previous.task_uid.isin(metadata.task_uid)], metadata], ignore_index=True
        )
    else:
        metadata, thresholds = assign_buckets(metadata)
    out = paths.RESULTS / "task_metadata.csv"
    metadata.to_csv(out, index=False)
    (paths.RESULTS / "descriptor_thresholds.json").write_text(
        json.dumps({"thresholds_from": "discovery split only", "cuts": thresholds}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}  ({len(metadata)} tasks)")
    columns = [c for c in metadata.columns if c.startswith("D")]
    print(metadata[["task_uid", "split", *columns]].to_string(index=False))
    print("\nthresholds:", json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
