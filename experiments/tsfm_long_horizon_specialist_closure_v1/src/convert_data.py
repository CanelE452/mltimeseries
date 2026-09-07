"""Turn a fev task into the two things a specialist needs, and nothing more.

Two exports, both TRACK U (one univariate series per row, no covariates):

`training_frame`  everything visible strictly before the *first* evaluation
                  cutoff. This is all the specialist may fit on.
`context_frames`  per evaluation window, the legal past at that window's cutoff.
                  Later windows carry newly observed target values, which the
                  frozen predictor may read as input context but never refit on.

The split point is the first window's cutoff, so no row that any window scores
can appear in the training frame. `tests/` asserts that directly by comparing
the last training timestamp against the first evaluation timestamp.

Everything is written as plain arrow/parquet-free npz plus a small JSON index, so
the specialist environment can read it without importing fev.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from . import paths

paths.configure_environment()


def load_task(task_uid: str):
    import fev

    raw = yaml.safe_load(
        (paths.SOURCE_DATA / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]
    pool = pd.read_csv(paths.SOURCE_RESULTS / "task_pool.csv").set_index("task_uid")
    task = fev.Task(**raw[int(pool.loc[task_uid, "yaml_index"])])
    task.load_full_dataset(num_proc=1)
    return task


def _univariate_window(task, window_idx: int):
    """fev's own univariate view of a window: past rows and their identifiers."""
    import fev

    window = task.get_window(window_idx, num_proc=1)
    past, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
    return window, past


def _item_ids(past, task, n_rows: int) -> list[str]:
    """Stable identifiers matching fev's row order: item-major, target-minor."""
    base = list(past[task.id_column])
    targets = list(task.target_columns)
    if len(base) != n_rows:
        raise SystemExit(f"unexpected univariate row count {len(base)} != {n_rows}")
    return [f"{item}||{targets[i % len(targets)]}" for i, item in enumerate(base)]


def export(task_uid: str) -> dict:
    """Write the training frame and every window's context for one task."""
    task = load_task(task_uid)
    safe = task_uid.replace("::", "__").replace("/", "-")
    out_dir = paths.DATA_EXTERNAL / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = []
    for window_idx in range(task.num_windows):
        window, past = _univariate_window(task, window_idx)
        targets = [np.asarray(v, dtype=np.float64) for v in past["target"]]
        timestamps = [np.asarray(v) for v in past[task.timestamp_column]]
        ids = _item_ids(past, task, len(targets))
        lengths = np.array([len(t) for t in targets], dtype=np.int64)
        np.savez_compressed(
            out_dir / f"window_{window_idx:02d}.npz",
            target=np.concatenate(targets),
            timestamp=np.concatenate(timestamps).astype("datetime64[ns]").astype("int64"),
            lengths=lengths,
            item_id=np.array(ids, dtype=object),
        )
        windows.append(
            {
                "window_idx": window_idx,
                "n_items": len(ids),
                "context_min": int(lengths.min()),
                "context_median": int(np.median(lengths)),
                "context_max": int(lengths.max()),
                "last_context_timestamp": str(
                    np.concatenate(timestamps).astype("datetime64[ns]").max()
                ),
            }
        )

    index = {
        "task_uid": task_uid,
        "safe_name": safe,
        "track": "U",
        "horizon": int(task.horizon),
        "num_windows": int(task.num_windows),
        "seasonality": int(task.seasonality),
        "freq": str(task.freq),
        "quantile_levels": list(task.quantile_levels),
        "eval_metric": task.eval_metric,
        "target_columns": list(task.target_columns),
        "training_window": 0,
        "training_note": (
            "window_00 is the training frame: it holds everything visible before the first "
            "evaluation cutoff. Later windows are inference context only."
        ),
        "windows": windows,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def read_window(safe_name: str, window_idx: int) -> pd.DataFrame:
    """Long-format frame AutoGluon can wrap in a TimeSeriesDataFrame."""
    payload = np.load(
        paths.DATA_EXTERNAL / safe_name / f"window_{window_idx:02d}.npz", allow_pickle=True
    )
    lengths = payload["lengths"]
    ids = np.repeat(np.array([str(i) for i in payload["item_id"]]), lengths)
    return pd.DataFrame(
        {
            "item_id": ids,
            "timestamp": pd.to_datetime(payload["timestamp"]),
            "target": payload["target"],
        }
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    spec = json.loads((paths.RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8"))
    tasks = args.tasks or (spec["development_tasks"] + spec["fresh_holdout_tasks"])
    summaries = []
    for task_uid in tasks:
        index = export(task_uid)
        first, last = index["windows"][0], index["windows"][-1]
        print(
            f"{task_uid.split('::')[1]:26s} h={index['horizon']:<4d} windows={index['num_windows']:<3d} "
            f"items={first['n_items']:<6d} train_ctx={first['context_median']:<7d} "
            f"last_ctx={last['context_median']}",
            flush=True,
        )
        summaries.append(index)
    (paths.RESULTS / "conversion_index.json").write_text(
        json.dumps({"tasks": summaries}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {paths.RESULTS / 'conversion_index.json'} ({len(summaries)} tasks)")


if __name__ == "__main__":
    main()
