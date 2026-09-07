"""Fit the supervised specialist suite once per task and seed, then freeze it.

Runs inside `.venv-tsfm-specialist`, which has AutoGluon and no part of the TSFM
stack in use. It reads the exported windows, fits on window 0 only, and writes
quantile forecasts for every window. Scoring happens elsewhere, in the fev
environment, so nothing here can influence the metric.

The chronology contract from Section 10, enforced by construction:

  window 0 training frame  ->  fit once  ->  predictor frozen
  window k context         ->  predict(context)          no refit, ever

AutoGluon's `predict` takes the context as an argument and does not update
parameters, so later windows' newly observed values enter as input and never as
training signal. `refit_full` is never called and the fitted model list is
written out so `verify` can confirm no foundation model appears in it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths

warnings.filterwarnings("ignore", category=FutureWarning)

SEEDS = [2026090701, 2026090702]
TIME_LIMIT_SECONDS = 20 * 60
VALIDATION_METRIC = "WQL"

# Fixed pool, no HPO. Section 7 names these; Section 14 fixes the order in which
# they are dropped if a task cannot finish inside the cap.
MODEL_POOL: dict[str, dict] = {
    "PatchTST": {},
    "TiDE": {},
    "DLinear": {},
    "DeepAR": {},
    "DirectTabular": {},
}
DROP_ORDER = ["DeepAR", "DirectTabular"]
MIN_FAMILIES = 3

FORBIDDEN = ("chronos", "toto", "timesfm", "tirex")


def read_window(safe_name: str, window_idx: int) -> tuple[pd.DataFrame, list[str]]:
    payload = np.load(
        paths.DATA_EXTERNAL / safe_name / f"window_{window_idx:02d}.npz", allow_pickle=True
    )
    lengths = payload["lengths"]
    order = [str(i) for i in payload["item_id"]]
    frame = pd.DataFrame(
        {
            "item_id": np.repeat(np.array(order), lengths),
            "timestamp": pd.to_datetime(payload["timestamp"]),
            "target": payload["target"],
        }
    )
    return frame, order


def _validation_windows(frame: pd.DataFrame, horizon: int) -> int:
    """2 rolling validation windows when the shortest series can carry them, else 1.

    Decided from series length alone, never from a score.
    """
    shortest = frame.groupby("item_id", sort=False).size().min()
    return 2 if shortest >= 3 * horizon + 2 else 1


def fit_once(task: dict, seed: int, model_pool: dict, time_limit: int):
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    safe = task["safe_name"]
    frame, order = read_window(safe, 0)
    train = TimeSeriesDataFrame.from_data_frame(frame)
    num_val_windows = _validation_windows(frame, task["horizon"])

    predictor_path = paths.PREDICTORS / f"{safe}__seed{seed}"
    if predictor_path.exists():
        shutil.rmtree(predictor_path, ignore_errors=True)

    predictor = TimeSeriesPredictor(
        prediction_length=task["horizon"],
        path=str(predictor_path),
        target="target",
        eval_metric=VALIDATION_METRIC,
        quantile_levels=list(task["quantile_levels"]),
        freq=task["freq"],
        verbosity=1,
    )
    started = time.perf_counter()
    predictor.fit(
        train,
        hyperparameters=model_pool,
        num_val_windows=num_val_windows,
        time_limit=time_limit,
        enable_ensemble=True,
        random_seed=seed,
        hyperparameter_tune_kwargs=None,
    )
    fit_seconds = time.perf_counter() - started
    return predictor, order, num_val_windows, fit_seconds


def run_task(task: dict, seed: int, model_pool: dict, time_limit: int) -> dict:
    safe = task["safe_name"]
    record = {
        "task_uid": task["task_uid"],
        "safe_name": safe,
        "seed": seed,
        "horizon": task["horizon"],
        "num_windows": task["num_windows"],
        "quantile_levels": task["quantile_levels"],
        "time_limit_seconds": time_limit,
        "requested_models": sorted(model_pool),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        predictor, order, num_val_windows, fit_seconds = fit_once(task, seed, model_pool, time_limit)
        leaderboard = predictor.leaderboard(silent=True) if hasattr(predictor, "leaderboard") else None
        fitted = list(predictor.model_names())
        contaminated = [m for m in fitted if any(f in m.lower() for f in FORBIDDEN)]
        if contaminated:
            raise SystemExit(f"HARD STOP: foundation model in the specialist suite: {contaminated}")

        # S_BEST is the strongest *single* specialist by training-only validation.
        # The ensemble is a separate comparator, so letting it be S_BEST too would
        # report one number twice (Section 12).
        singles = [m for m in fitted if "Ensemble" not in m]
        if leaderboard is not None and singles:
            ranked = leaderboard[leaderboard.model.isin(singles)].sort_values(
                "score_val", ascending=False
            )
            best = str(ranked.model.iloc[0])
        else:
            best = singles[0] if singles else predictor.model_best
        record.update(
            {
                "status": "OK",
                "num_val_windows": num_val_windows,
                "fit_seconds": round(fit_seconds, 1),
                "fitted_models": fitted,
                "model_best": best,
                "validation_scores": (
                    {r["model"]: float(r["score_val"]) for _, r in leaderboard.iterrows()}
                    if leaderboard is not None
                    else {}
                ),
            }
        )
        if leaderboard is not None:
            leaderboard.assign(task_uid=task["task_uid"], seed=seed).to_csv(
                paths.RUNS / f"leaderboard__{safe}__seed{seed}.csv", index=False
            )

        # S_ENSEMBLE: AutoGluon's own weighted ensemble when it built one.
        ensemble_name = next((m for m in fitted if "Ensemble" in m), None)
        record["ensemble_model"] = ensemble_name
        record["ensemble_fallback"] = ensemble_name is None
        if ensemble_name is not None:
            try:
                record["ensemble_weights"] = {
                    k: float(v) for k, v in predictor._trainer.load_model(ensemble_name).model_to_weight.items()
                }
            except Exception:
                record["ensemble_weights"] = {}

        if ensemble_name is None:
            # Section 31: no official ensemble, so equal-weight the top three by
            # training-only validation rank and record the fallback.
            ranked = (
                leaderboard[leaderboard.model.isin(singles)]
                .sort_values("score_val", ascending=False)
                .model.tolist()[:3]
                if leaderboard is not None
                else singles[:3]
            )
            record["equal_weight_fallback_members"] = ranked
            record["ensemble_weights"] = {m: 1 / len(ranked) for m in ranked}

        for label, model in (("S_ENSEMBLE", ensemble_name), ("S_BEST", best)):
            if label == "S_ENSEMBLE" and model is None:
                blocks = equal_weight_blocks(predictor, task, record["equal_weight_fallback_members"])
                model = "EqualWeightTop3"
            else:
                blocks = forecast_all_windows_with(predictor, task, model)
            np.savez_compressed(
                paths.FORECASTS / f"{label}__{safe}__seed{seed}.npz",
                **blocks,
                quantile_levels=np.asarray(task["quantile_levels"], dtype=np.float64),
                model_name=np.array([model], dtype=object),
            )
        # Individual components, so PatchTST alone can be reported.
        for model in fitted:
            if "Ensemble" in model:
                continue
            blocks = forecast_all_windows_with(predictor, task, model)
            np.savez_compressed(
                paths.FORECASTS / f"COMPONENT_{model}__{safe}__seed{seed}.npz",
                **blocks,
                quantile_levels=np.asarray(task["quantile_levels"], dtype=np.float64),
                model_name=np.array([model], dtype=object),
            )
    except Exception as exc:  # noqa: BLE001 - a failed task is recorded, not hidden
        record.update(
            {"status": f"SPECIALIST_FAILED_TASK: {type(exc).__name__}: {exc}",
             "traceback": traceback.format_exc()[-2000:]}
        )
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def equal_weight_blocks(predictor, task: dict, members: list[str]) -> dict[str, np.ndarray]:
    """Section 31 fallback: average the top-3 validation models' quantiles.

    This is an average of quantile curves, not a mixture distribution, and is
    labelled EQUAL_WEIGHT_FALLBACK wherever it is reported.
    """
    stacks = [forecast_all_windows_with(predictor, task, m) for m in members]
    keys = stacks[0].keys()
    return {k: np.mean(np.stack([s[k] for s in stacks], axis=0), axis=0).astype(np.float32) for k in keys}


def forecast_all_windows_with(predictor, task: dict, model: str) -> dict[str, np.ndarray]:
    from autogluon.timeseries import TimeSeriesDataFrame

    quantiles = [str(q) for q in task["quantile_levels"]]
    horizon = task["horizon"]
    blocks = {}
    for window in task["windows"]:
        index = window["window_idx"]
        frame, order = read_window(task["safe_name"], index)
        context = TimeSeriesDataFrame.from_data_frame(frame)
        prediction = predictor.predict(context, model=model)
        available = set(prediction.index.get_level_values(0).unique())
        block = np.full((len(order), horizon, len(quantiles)), np.nan)
        for row, item in enumerate(order):
            if item in available:
                block[row] = prediction.loc[item][quantiles].to_numpy()[:horizon]
        blocks[f"w{index:02d}"] = block.astype(np.float32)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "holdout", "both"], default="both")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    spec = json.loads((paths.RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8"))
    index = json.loads((paths.RESULTS / "conversion_index.json").read_text(encoding="utf-8"))
    by_uid = {t["task_uid"]: t for t in index["tasks"]}

    if args.tasks:
        wanted = args.tasks
    elif args.split == "development":
        wanted = spec["development_tasks"]
    elif args.split == "holdout":
        wanted = spec["fresh_holdout_tasks"]
    else:
        wanted = spec["development_tasks"] + spec["fresh_holdout_tasks"]

    pool = {k: v for k, v in MODEL_POOL.items() if not args.models or k in args.models}
    records = []
    for task_uid in wanted:
        for seed in args.seeds:
            safe = by_uid[task_uid]["safe_name"]
            if (paths.FORECASTS / f"S_ENSEMBLE__{safe}__seed{seed}.npz").exists():
                # Fitting one task can hold several GB. If a sweep is interrupted,
                # resume from the forecast cache rather than refitting what is done.
                print(f"cached  {task_uid} seed={seed}", flush=True)
                continue
            print(f"\n=== {task_uid} seed={seed} ===", flush=True)
            record = run_task(by_uid[task_uid], seed, pool, args.time_limit)
            print(f"  {record['status']}  best={record.get('model_best')} "
                  f"fit={record.get('fit_seconds')}s", flush=True)
            records.append(record)
    out = paths.RESULTS / f"specialist_manifest{args.tag}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(records)} rows)")


if __name__ == "__main__":
    main()
