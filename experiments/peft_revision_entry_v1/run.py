"""Causal, fixed-protocol CPU diagnostic; no foundation model is loaded."""

from __future__ import annotations

import argparse
import calendar
import csv
from datetime import date, timedelta, datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_external/alfred_revision_entry_v1/complete"
OUT = ROOT / "results/peft_revision_entry_v1"
RUN = ROOT / "runs/peft_revision_entry_v1"
PLAN = ROOT / "_docs/notes/tsfm_topics/19_peft_revision_entry_plan_20260908.md"
ARMS = ("FIRST_FIXED_X", "FIRST_BIAS", "REVISED_FIXED_X", "REVISED_CURRENT_X",
        "MATURE_ONLY", "AGE_CORRECTED")


def month(year: int, number: int) -> int:
    return year * 12 + number - 1


def start(event: int) -> date:
    year, zero_month = divmod(int(event), 12)
    return date(year, zero_month + 1, 1)


def end(event: int) -> date:
    first = start(event)
    return date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def freeze() -> None:
    previous_path = ROOT / "runs/peft_coarse_supervision_v1/study_contract.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    final_path = ROOT / "results/peft_coarse_supervision_v1/final_audit.json"
    previous_audit = json.loads(final_path.read_text(encoding="utf-8"))
    protected = {**previous["protected_hashes"], **previous["source_hashes"],
                 **previous["runtime_hashes"], **previous_audit["artifact_hashes"]}
    for path in (previous_path, final_path, Path(previous_audit["report_path"]),
                 ROOT / "_docs/notes/tsfm_topics/18_peft_problem_first_candidates_20260908.md",
                 ROOT / "_docs/reference/time_series_peft_problem_first_dossier_20260908.md",
                 ROOT / "_docs/reference/time_series_peft_problem_first_catalog_20260908.json",
                 ROOT / "_docs/notes/tsfm_topics/19a_revision_download_transport_20260908.md",
                 ROOT / "_docs/notes/tsfm_topics/19b_revision_browser_download_20260908.md"):
        protected[str(path)] = sha(path)
    for name, expected in protected.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Previous artifact changed: {name}")
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.rglob("*.py"))}
    raw = {str(p.relative_to(ROOT)): sha(p) for p in sorted(RAW.rglob("*")) if p.is_file()}
    write_json(RUN / "cpu_contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(PLAN),
        "source_hashes": sources, "input_hashes": raw, "protected_hashes": protected,
        "numpy": np.__version__, "stage": "CPU_ONLY", "gpu_runs": 0,
    })
    print(json.dumps({"contract": "frozen", "protected": len(protected), "sources": len(sources), "inputs": len(raw)}), flush=True)


def verify_contract() -> dict:
    contract = json.loads((RUN / "cpu_contract.json").read_text(encoding="utf-8"))
    if sha(PLAN) != contract["plan_sha256"]:
        raise ValueError("Plan changed")
    for section in ("source_hashes", "input_hashes", "protected_hashes"):
        for name, expected in contract[section].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Contract mismatch: {name}")
    return contract


class CausalPanel:
    def __init__(self, series):
        self.series = series
        self.events = np.arange(month(1991, 1), month(2024, 12) + 1)
        self.growth = lru_cache(maxsize=None)(series.growth)
        self.first_dates = []
        first, mature, features, age_values = [], [], [], []
        for event in self.events:
            released, value = series.first_growth(int(event))
            if released is None or released > end(int(event) + 6):
                raise ValueError(f"Missing first label: {start(event)}")
            self.first_dates.append(released)
            first.append(value)
            mature.append(self.growth(int(event), end(int(event) + 6)))
            features.append(self.features(int(event), start(event) - timedelta(days=1)))
            age_values.append([self.growth(int(event), end(int(event) + age)) for age in range(7)])
        self.first = np.asarray(first)
        self.mature = np.asarray(mature)
        self.fixed_x = np.asarray(features)
        self.age_values = np.asarray(age_values)
        self.maturity = [end(int(event) + 6) for event in self.events]
        for name, value in (("first", self.first), ("mature", self.mature), ("fixed_x", self.fixed_x)):
            if not np.isfinite(value).all():
                raise ValueError(f"Incomplete {name}; no selective row removal")
        self.scale = float(np.var(self.mature[self.events <= month(2014, 12)]))
        if self.scale <= 0:
            raise ValueError("Nonpositive historical scale")

    def features(self, event: int, asof: date) -> np.ndarray:
        return np.asarray([1.0] + [self.growth(event - lag, asof) for lag in range(2, 8)])

    def origin(self, event: int) -> dict:
        origin, cutoff = start(event), start(event) - timedelta(days=1)
        available = (self.events < event) & np.asarray([d < origin for d in self.first_dates])
        matured = available & np.asarray([d < origin for d in self.maturity])
        indices = np.flatnonzero(available)
        events = self.events[available]
        current = np.asarray([self.growth(int(self.events[i]), min(cutoff, self.maturity[i])) for i in indices])
        revised_x = np.asarray([self.features(int(t), cutoff) for t in events])
        age_corrected = current.copy()
        for j, i in enumerate(indices):
            age = event - int(self.events[i]) - 1
            if age >= 6:
                continue
            if age < 0:
                raise ValueError("Future label in training")
            history = matured & np.isfinite(self.age_values[:, age])
            if history.any():
                age_corrected[j] += float(np.mean(self.mature[history] - self.age_values[history, age]))
        first_bias = float(np.mean(self.mature[matured] - self.first[matured])) if matured.any() else 0.0
        x = self.features(event, cutoff)
        latest = max((t for t in self.series.events if t < event and np.isfinite(self.series.snapshot(t, cutoff))), default=None)
        if latest is None or latest > event - 2:
            raise ValueError(f"Unexpected publication gap: {origin}, {latest}")
        for array in (current, revised_x, age_corrected, x):
            if not np.isfinite(array).all():
                raise ValueError(f"Incomplete origin: {origin}")
        fixed = self.fixed_x[available]
        data = {
            "FIRST_FIXED_X": (fixed, self.first[available], events),
            "FIRST_BIAS": (fixed, self.first[available], events),
            "REVISED_FIXED_X": (fixed, current, events),
            "REVISED_CURRENT_X": (revised_x, current, events),
            "MATURE_ONLY": (self.fixed_x[matured], self.mature[matured], self.events[matured]),
            "AGE_CORRECTED": (fixed, age_corrected, events),
        }
        return {"data": data, "x": x, "first_bias": first_bias,
                "last": self.growth(latest, cutoff), "latest": latest,
                "origin": origin.isoformat(), "cutoff": cutoff.isoformat()}

    def revision_summary(self) -> dict:
        delta = self.mature - self.first
        lags = [(d - start(t)).days for t, d in zip(self.events, self.first_dates)]
        counts = []
        for i, event in enumerate(self.events):
            dates = [d for d in self.series.vintage_dates if self.first_dates[i] <= d <= self.maturity[i]]
            values = [self.growth(int(event), d) for d in dates]
            counts.append(sum(abs(b - a) > 1e-10 for a, b in zip(values, values[1:])))
        return {"events": len(delta), "revised_events": int(np.sum(np.abs(delta) > 1e-10)),
                "mean_revision": float(np.mean(delta)), "mean_absolute_revision": float(np.mean(np.abs(delta))),
                "median_absolute_revision": float(np.median(np.abs(delta))),
                "p95_absolute_revision": float(np.quantile(np.abs(delta), .95)),
                "maximum_absolute_revision": float(np.max(np.abs(delta))),
                "revision_rmse_over_historical_std": float(np.sqrt(np.mean(delta ** 2) / self.scale)),
                "release_lag_days_min_median_max": [min(lags), float(np.median(lags)), max(lags)],
                "revision_count_first_to_mature_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
                "historical_variance": self.scale, "units": "100 log difference; percentage points"}


def bootstrap(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(1908)
    n = differences.shape[-1]
    means = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=(n + 11) // 12)
        indices = np.concatenate([(s + np.arange(12)) % n for s in starts])[:n]
        means.append(float(differences[..., indices].mean()))
    return np.quantile(means, [.025, .975]).tolist()


def execute() -> None:
    from .data import load_series
    from .model import RidgeConfig, ridge, RevisionRidge

    started = time.perf_counter()
    verify_contract()
    panels = {name: CausalPanel(load_series(RAW, name)) for name in ("PAYEMS", "INDPRO")}
    validation = range(month(2016, 1), month(2018, 12) + 1)
    evaluation = range(month(2020, 1), month(2024, 12) + 1)
    configs = [RidgeConfig(lam, window) for lam in (.01, .1, 1., 10., 100.) for window in (None, 120)]
    qc = {name: panel.revision_summary() for name, panel in panels.items()}
    write_json(OUT / "derived_data_qc.json", qc)
    print(json.dumps({"stage": "data_qc", "series": qc}), flush=True)
    selected, scores, origin_cache, identity_rows = {}, {}, {}, []
    fit_seconds = {}
    with (OUT / "event_labels.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("series", "target_event", "first_release", "first_growth", "maturity_date", "mature_growth"))
        for name, panel in panels.items():
            for t, d, first, m, value in zip(panel.events, panel.first_dates, panel.first, panel.maturity, panel.mature):
                writer.writerow((name, start(t), d, first, m, value))
    v_predictions = {}
    for name, panel in panels.items():
        selected[name], scores[name], v_predictions[name] = {}, {}, {}
        for event in list(validation) + list(evaluation):
            origin_cache[name, event] = panel.origin(event)
        updater = RevisionRidge(n_features=7, lam=1.0)
        for event in list(validation) + list(evaluation):
            record = origin_cache[name, event]
            x_train, y, ids = record["data"]["REVISED_FIXED_X"]
            for row, target, row_id in zip(x_train, y, ids):
                updater.update(int(row_id), row, float(target))
            expected_a = x_train.T @ x_train + np.diag([0.] + [1.] * 6)
            expected_b = x_train.T @ y
            expected_coef = np.linalg.solve(expected_a, expected_b)
            check = {"n_rows": updater.n_rows, "expected_rows": len(y),
                     "a_max_abs": float(np.max(np.abs(updater.A - expected_a))),
                     "b_max_abs": float(np.max(np.abs(updater.b - expected_b))),
                     "coef_max_abs": float(np.max(np.abs(updater.coefficients() - expected_coef))),
                     "prediction_abs": abs(updater.predict(record["x"]) - float(record["x"] @ expected_coef))}
            check["passed"] = check["n_rows"] == check["expected_rows"] and max(check["coef_max_abs"], check["prediction_abs"]) <= 1e-8
            identity_rows.append({"series": name, "event": start(event).isoformat(), **check})
            if not check["passed"]:
                write_json(OUT / "ridge_identity_failure.json", identity_rows)
                raise ValueError("Ridge revision identity failed against independent current training rows")
        v_truth = panel.mature[np.isin(panel.events, list(validation))]
        for arm in ARMS:
            candidates = []
            for config in configs:
                predictions = []
                for event in validation:
                    record = origin_cache[name, event]
                    x_train, y, ids = record["data"][arm]
                    result = ridge(x_train, y, record["x"], config.lam, config.window, event, ids)
                    predictions.append(float(result["prediction"]) + (record["first_bias"] if arm == "FIRST_BIAS" else 0.0))
                mse = float(np.mean((np.asarray(predictions) - v_truth) ** 2))
                candidates.append({"lam": config.lam, "window": config.window, "mse": mse, "predictions": predictions})
            best = min(candidates, key=lambda c: (c["mse"], c["lam"], c["window"] is not None))
            selected[name][arm] = {key: best[key] for key in ("lam", "window", "mse")}
            scores[name][arm] = [{k: v for k, v in c.items() if k != "predictions"} for c in candidates]
            v_predictions[name][arm] = best["predictions"]
    write_json(OUT / "ridge_identity.json", identity_rows)
    write_json(RUN / "validation_selection.json", {"selected": selected, "all_scores": scores, "predictions": v_predictions,
                                                   "selection_cutoff": "2019-07-01", "evaluation_used_for_selection": False})
    print(json.dumps({"stage": "selection_frozen", "selected": selected}), flush=True)
    predictions, metrics, losses, origins = {}, {}, {}, []
    for name, panel in panels.items():
        fit_seconds[name] = {arm: 0.0 for arm in ARMS}
        truth = panel.mature[np.isin(panel.events, list(evaluation))]
        predictions[name] = {arm: [] for arm in (*ARMS, "ZERO_GROWTH", "LAST_KNOWN_GROWTH")}
        for event in evaluation:
            record = origin_cache[name, event]
            origins.append({"series": name, "target_event": start(event).isoformat(), "origin_date": record["origin"],
                            "vintage_cutoff": record["cutoff"], "latest_available_event": start(record["latest"]).isoformat(),
                            "event_step_gap": event - record["latest"]})
            predictions[name]["ZERO_GROWTH"].append(0.0)
            predictions[name]["LAST_KNOWN_GROWTH"].append(record["last"])
            for arm in ARMS:
                config = selected[name][arm]
                x_train, y, ids = record["data"][arm]
                fit_started = time.perf_counter()
                result = ridge(x_train, y, record["x"], config["lam"], config["window"], event, ids)
                fit_seconds[name][arm] += time.perf_counter() - fit_started
                prediction = float(result["prediction"]) + (record["first_bias"] if arm == "FIRST_BIAS" else 0.0)
                predictions[name][arm].append(prediction)
        metrics[name], losses[name] = {}, {}
        for arm, values in predictions[name].items():
            errors = np.asarray(values) - truth
            losses[name][arm] = errors ** 2 / panel.scale
            metrics[name][arm] = {"n": len(truth), "mse": float(np.mean(errors ** 2)), "mae": float(np.mean(np.abs(errors))),
                                  "normalized_mse": float(np.mean(losses[name][arm]))}
        predictions[name]["truth_six_month_mature"] = truth.tolist()
    pooled = {}
    base = np.stack([losses[n]["FIRST_FIXED_X"] for n in panels])
    for arm in (*ARMS, "ZERO_GROWTH", "LAST_KNOWN_GROWTH"):
        values = np.stack([losses[n][arm] for n in panels])
        improvement = base - values
        pooled[arm] = {"normalized_mse": float(values.mean()),
                       "improvement_percent_vs_first": float(100 * improvement.mean() / base.mean()),
                       "normalized_mse_difference_95pct_block_ci": bootstrap(improvement)}
    strong = ("REVISED_FIXED_X", "REVISED_CURRENT_X", "AGE_CORRECTED", "FIRST_BIAS", "MATURE_ONLY")
    has_revision = any(row["revised_events"] > 0 for row in qc.values())
    screen_pass = has_revision and any(pooled[arm]["improvement_percent_vs_first"] >= 1 for arm in strong)
    result = {"metrics": metrics, "pooled": pooled, "selection": selected, "revision_qc": qc,
              "cpu_wall_seconds": time.perf_counter() - started, "evaluation_ridge_fit_seconds": fit_seconds,
              "cost_scope": "Fit timers include row validation, window filtering, solve and prediction; exclude snapshots, transformation, selection and file IO. Total wall includes initial contract verification.",
              "screen": "PRIORITIZE_FM_DIAGNOSTIC" if screen_pass else "DEFER_GPU_ON_CURRENT_TWO_SERIES",
              "gpu_runs": 0, "new_method_claim": False,
              "limitations": ["Two related macro series", "60 evaluation months including COVID", "CPU ridge only; PEFT necessity untested",
                              "Six-month vintage target, not final truth", "Block CI does not remove model-selection or domain uncertainty"]}
    write_json(OUT / "evaluation_predictions.json", predictions)
    write_json(OUT / "origin_audit.json", origins)
    write_json(OUT / "cpu_results.json", result)
    verify_contract()
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "verify":
        print(json.dumps({"verified": True, "protected": len(verify_contract()["protected_hashes"])}))
    else:
        execute()


if __name__ == "__main__":
    main()
