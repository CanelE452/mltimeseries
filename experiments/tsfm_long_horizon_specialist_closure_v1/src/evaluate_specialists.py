"""Score specialist forecasts with the benchmark's own evaluator.

Runs in the TSFM environment. The specialist environment only produced numbers;
the metric is computed here by `fev.Task.evaluation_summary`, exactly as the
source study computed it, so a specialist score and a TSFM score are the same
quantity measured the same way.

Row order is the one thing that could silently corrupt this. The exported
`item_id` list was built in fev's own univariate order, and `tracks.to_predictions`
expects that order, so the two line up by construction; `tests/` checks it against
a re-derived order rather than trusting the comment.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.tsfm_benchmark_gap_discovery_v1.src import scoring, tracks

from . import convert_data, paths

paths.configure_environment()

FORECAST_PATTERN = re.compile(r"^(?P<label>.+?)__(?P<safe>.+)__seed(?P<seed>\d+)\.npz$")


def _windows_inputs(task, track: str = "U"):
    """Cache the per-window WindowInputs once; several labels reuse them."""
    cache = {}
    for window_idx in range(task.num_windows):
        window = task.get_window(window_idx, num_proc=1)
        cache[window_idx] = (window, tracks.build_inputs(window, track, task))
    return cache


def score_forecast_file(path, task, window_cache) -> dict:
    payload = np.load(path, allow_pickle=True)
    quantile_levels = list(task.quantile_levels)
    stored = payload["quantile_levels"].tolist()
    if [round(q, 6) for q in stored] != [round(q, 6) for q in quantile_levels]:
        raise SystemExit(f"METRIC_MISMATCH: {path.name} quantiles {stored} != task {quantile_levels}")

    median_index = quantile_levels.index(0.5)
    predictions_per_window, origin_losses, origin_keys = [], [], []
    n_missing = 0
    for window_idx in range(task.num_windows):
        window, inputs = window_cache[window_idx]
        block = payload[f"w{window_idx:02d}"].astype(np.float64)
        if block.shape[0] != inputs.n_items:
            raise SystemExit(
                f"row mismatch in {path.name} window {window_idx}: "
                f"{block.shape[0]} forecasts vs {inputs.n_items} evaluation rows"
            )
        missing = ~np.isfinite(block).all(axis=(1, 2))
        n_missing += int(missing.sum())
        if missing.any():
            # A component that could not forecast an item leaves NaN. Fill with the
            # last observed value so the row still scores; the count is reported.
            for row in np.flatnonzero(missing):
                last = inputs.targets[row][0][-1]
                block[row] = np.where(np.isfinite(block[row]), block[row], last)
        quantiles = block[:, None, :, :]
        point = quantiles[..., median_index]
        predictions = tracks.to_predictions(quantiles, point, inputs, quantile_levels)
        predictions_per_window.append(predictions)
        decomposition = scoring.per_origin_sql(
            window, predictions, quantile_levels, int(task.seasonality)
        )
        origin_losses.append(decomposition["origin_loss"])
        origin_keys.extend([f"w{window_idx}::{i}" for i in decomposition["item_ids"]])

    summary = task.evaluation_summary(predictions_per_window, model_name=path.stem)
    loss = np.concatenate(origin_losses)
    windows = np.concatenate(
        [np.full(part.size, i) for i, part in enumerate(origin_losses)]
    )
    return {
        "native_score": float(summary["test_error"]),
        "origin_loss": loss,
        "origin_window": windows,
        "origin_key": origin_keys,
        "n_missing_rows": n_missing,
        "n_origins": int(loss.size),
        "reconciliation_gap": scoring.reconcile(
            scoring.task_aggregate(loss, windows), float(summary["test_error"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--out", default="specialist_scores.csv")
    args = parser.parse_args()

    index = json.loads((paths.RESULTS / "conversion_index.json").read_text(encoding="utf-8"))
    by_safe = {t["safe_name"]: t for t in index["tasks"]}

    files = sorted(paths.FORECASTS.glob(args.pattern))
    if not files:
        raise SystemExit(f"no forecast files match {args.pattern}")

    by_task: dict[str, list] = {}
    for path in files:
        match = FORECAST_PATTERN.match(path.name)
        if not match:
            continue
        by_task.setdefault(match.group("safe"), []).append((path, match))

    rows = []
    for safe, entries in by_task.items():
        task = convert_data.load_task(by_safe[safe]["task_uid"])
        cache = _windows_inputs(task)
        for path, match in sorted(entries, key=lambda e: e[0].name):
            result = score_forecast_file(path, task, cache)
            np.savez_compressed(
                paths.RUNS / f"origins__{path.stem}.npz",
                origin_loss=result["origin_loss"].astype(np.float32),
                origin_window=result["origin_window"].astype(np.int16),
                origin_key=np.array(result["origin_key"], dtype=object),
            )
            rows.append(
                {
                    "task_uid": by_safe[safe]["task_uid"],
                    "safe_name": safe,
                    "label": match.group("label"),
                    "seed": int(match.group("seed")),
                    "native_metric": task.eval_metric,
                    "native_score": result["native_score"],
                    "n_origins": result["n_origins"],
                    "n_missing_rows": result["n_missing_rows"],
                    "reconciliation_gap": result["reconciliation_gap"],
                    "horizon": int(task.horizon),
                    "num_windows": int(task.num_windows),
                    "scored_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(
                f"{by_safe[safe]['task_uid'].split('::')[1]:24s} {match.group('label'):34s} "
                f"seed={match.group('seed')} SQL={result['native_score']:.4f} "
                f"missing={result['n_missing_rows']}",
                flush=True,
            )
    out = paths.RESULTS / args.out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
