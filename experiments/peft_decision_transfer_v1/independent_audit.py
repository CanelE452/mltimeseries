"""Independent CPU-only audit for peft_decision_transfer_v1.

This file intentionally does not import the study's analyse.py or policy.py.
It recomputes selection, losses, gates, replay timing, and chronology from
JSON/NPZ artifacts after the guarded run has completed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_decision_transfer_v1"
RESULTS = ROOT / "results/peft_decision_transfer_v1"
ACTIONS = ("STOP", "HEAD", "JOINT")
TIMED_METHODS = ("PROBE", "FULL", "FIXED", "EARLY_STOP", "HEAD", "WIDE")
ALL_METHODS = (
    "PROBE",
    "FULL",
    "HEAD",
    "FIXED",
    "EARLY_STOP",
    "WIDE",
    "CURRENT_C",
    "RANDOM",
    "HEAD_PROBE",
    "MASKED1",
    "CONSTANT",
    "SOURCE_CONSTANT",
)


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_x(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha(path)


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def assert_close(a: float, b: float, label: str, tol: float = 1e-10) -> None:
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{label}: {a!r} != {b!r} within {tol}")


def verify_plan_hashes(plan: dict[str, Any]) -> dict[str, int]:
    checked = {"source_hashes": 0, "input_hashes": 0}
    for section in ("source_hashes", "input_hashes"):
        for path, expected in plan[section].items():
            actual = sha(ROOT / path)
            if actual != expected:
                raise AssertionError(f"{section} hash mismatch for {path}: {actual} != {expected}")
            checked[section] += 1
    return checked


def raw_loss_npz(path: Path | str) -> float:
    with np.load(path, allow_pickle=False) as z:
        prediction = np.sort(z["prediction"].astype(np.float64), axis=2)
        target = z["target"].astype(np.float64)
        quantiles = z["quantiles"].astype(np.float64)
        scale = z["scale"].astype(np.float64)
        if prediction.ndim != 4 or target.ndim != 3:
            raise AssertionError(f"{path} unexpected prediction/target rank")
        if prediction.shape[0] != target.shape[0] or prediction.shape[1] != target.shape[1] or prediction.shape[3] != target.shape[2]:
            raise AssertionError(f"{path} prediction/target shape mismatch")
        if prediction.shape[2] != len(quantiles) or target.shape[1] != len(scale):
            raise AssertionError(f"{path} quantile/scale shape mismatch")
        valid = np.isfinite(target)
        counts = valid.sum(axis=(0, 2))
        if np.any(counts <= 0) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise AssertionError(f"{path} invalid target counts or scale")
        error = target[:, :, None, :] - prediction
        q = quantiles[None, None, :, None]
        loss = np.where(valid[:, :, None, :], 2.0 * np.maximum(q * error, (q - 1.0) * error), 0.0)
        per_channel_quantile = loss.sum(axis=(0, 3)) / counts[:, None] / scale[:, None]
        return float(per_channel_quantile.mean())


def bootstrap_scores_npz(path: Path | str, weights: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        prediction = np.sort(z["prediction"].astype(np.float64), axis=2)
        target = z["target"].astype(np.float64)
        quantiles = z["quantiles"].astype(np.float64)
        scale = z["scale"].astype(np.float64)
        valid = np.isfinite(target)
        error = target[:, :, None, :] - prediction
        q = quantiles[None, None, :, None]
        loss = np.where(valid[:, :, None, :], 2.0 * np.maximum(q * error, (q - 1.0) * error), 0.0)
        num = loss.sum(axis=3)
        den = valid.sum(axis=2)
        counts = weights @ den
        if np.any(counts <= 0):
            raise AssertionError(f"{path} bootstrap draw has no finite targets")
        return (np.einsum("bn,ncq->bcq", weights, num) / counts[:, :, None] / scale[None, :, None]).mean(axis=(1, 2))


def probe_choice(scores: dict[str, float], initial_score: float, margin_pct: float = 0.0) -> str:
    best = min(scores.values())
    tolerance = margin_pct * initial_score / 100.0
    for action in ACTIONS:
        if action in scores and scores[action] <= best + tolerance:
            return action
    raise AssertionError("no probe action selected")


def random_choice(key: str) -> str:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return ACTIONS[int(np.random.default_rng(seed).integers(3))]


def selected(history: list[dict[str, Any]], patience: int | None = None) -> tuple[dict[str, Any], int]:
    best = history[0]
    stale = 0
    for point in history[1:]:
        if point["V"] < best["V"]:
            best = point
            stale = 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            return best, point["step"]
    return best, history[-1]["step"]


def at_step(history: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for point in history:
        if point["step"] == step:
            return point
    raise AssertionError(f"missing step {step}")


def path_options(result: dict[str, Any], margin_pct: float = 0.0, constant: str = "STOP") -> dict[str, dict[str, Any]]:
    histories = result["histories"]
    fork = int(result["fork"])
    joint = histories["JOINT"]
    prefix = [point for point in joint if point["step"] <= fork]
    qstep = fork + int(result["probe_steps"])
    trial = {
        "STOP": at_step(joint, fork)["V"],
        "HEAD": at_step(histories["HEAD1"], qstep)["V"],
        "JOINT": at_step(joint, qstep)["V"],
    }
    action = probe_choice(trial, joint[0]["V"], margin_pct)
    key = "/".join(str(result["job"][k]) for k in ("episode", "dataset", "condition", "seed"))
    choices = {
        "PROBE": action,
        "CURRENT_C": "JOINT" if result["current_C_pct_F0"] > 0 else "HEAD",
        "RANDOM": random_choice(key),
        "HEAD_PROBE": probe_choice({name: trial[name] for name in ("STOP", "HEAD")}, joint[0]["V"], margin_pct),
        "CONSTANT": constant,
    }
    histories_for_action = {"STOP": prefix, "HEAD": histories["HEAD1"], "JOINT": joint}
    output: dict[str, dict[str, Any]] = {}
    for name, action_name in choices.items():
        option_history = histories_for_action[action_name]
        output[name] = {"action": action_name, "selected": selected(option_history)[0], "final": option_history[-1]}
    for mode in ("JOINT", "HEAD0", "HEAD1", "HEAD2", "MASKED1"):
        option_history = histories[mode]
        output[mode] = {"action": mode, "selected": selected(option_history)[0], "final": option_history[-1]}
    for patience in (1, 2, 3):
        point, stop_step = selected(joint, patience)
        output[f"ES{patience}"] = {"action": f"ES{patience}", "selected": point, "final": at_step(joint, stop_step), "stop_step": stop_step}
    output["STOP"] = {"action": "STOP", "selected": selected(prefix)[0], "final": prefix[-1]}
    return output


def method_option(fit: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if spec["family"] == "WIDE":
        history = fit["histories"]["JOINT"]
        return {"action": "WIDE", "selected": selected(history)[0], "final": history[-1]}
    return path_options(fit, float(spec.get("margin", 0.0)), spec.get("constant", "STOP"))[spec["path"]]


def result_paths(run: Path, key: str, stage: str) -> tuple[Path, Path, Path]:
    parent = run / stage / key
    return parent / "output/result.json", parent / "guard/status.json", parent / "output"


def load_records(run: Path, plan: dict[str, Any], episode: str) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    seal = read_json(run / f"{episode}_fits_sealed.json")
    records = []
    for index in seal["indices"]:
        entry = plan["jobs"][index]
        fit_result, _, _ = result_paths(run, entry["key"], "fit")
        forecast_result, _, _ = result_paths(run, entry["key"], "forecast")
        fit = read_json(fit_result)
        forecast = read_json(forecast_result)
        if not fit.get("completed") or not forecast.get("completed"):
            raise AssertionError(f"incomplete fit/forecast {entry['key']}")
        if fit["job"] != entry["job"] or forecast["job"] != entry["job"]:
            raise AssertionError(f"job mismatch {entry['key']}")
        if fit["plan_sha256"] != sha(run / "plan.json") or forecast["plan_sha256"] != sha(run / "plan.json"):
            raise AssertionError(f"plan hash mismatch {entry['key']}")
        records.append((entry, fit, forecast))
    return records


def dev_value(items: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], spec: dict[str, Any], dataset: str | None = None) -> float:
    values = []
    for entry, fit, forecast in items:
        job = entry["job"]
        if job["family"] != spec["family"] or job["recipe"] != spec["recipe"]:
            continue
        if dataset is not None and job["dataset"] != dataset:
            continue
        pick = method_option(fit, spec)["selected"]["checkpoint"]
        values.append(float(forecast["scores"][pick]) / float(forecast["scores"]["JOINT_0"]))
    expected = 2 if dataset else 4
    if len(values) != expected:
        raise AssertionError(f"development grid expected {expected} values, got {len(values)} for {spec}")
    return float(np.mean(values))


def independent_development_choice(items: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    grids: dict[str, list[dict[str, Any]]] = {}
    methods: dict[str, dict[str, Any]] = {}
    candidates = {"FULL": ["JOINT"], "HEAD": ["HEAD0"], "FIXED": ["HEAD1", "HEAD2"], "EARLY_STOP": ["ES1", "ES2", "ES3"], "PROBE": ["PROBE"]}
    for name, path_names in candidates.items():
        specs = [
            {"family": "TREE", "recipe": recipe, "path": path_name, "margin": margin}
            for recipe in (0, 1, 2)
            for path_name in path_names
            for margin in ([0.0, 0.25] if name == "PROBE" else [0.0])
        ]
        for spec in specs:
            spec["development_normalized_E"] = dev_value(items, spec)
        grids[name] = specs
        methods[name] = min(specs, key=lambda s: (s["development_normalized_E"], s["recipe"], s["path"], s["margin"]))
    wide_specs = [{"family": "WIDE", "recipe": recipe, "path": "WIDE"} for recipe in (0, 2)]
    for spec in wide_specs:
        spec["development_normalized_E"] = dev_value(items, spec)
    grids["WIDE"] = wide_specs
    methods["WIDE"] = min(wide_specs, key=lambda s: (s["development_normalized_E"], s["recipe"]))
    chosen = methods["PROBE"]
    recipe, margin = chosen["recipe"], chosen["margin"]
    for name in ("CURRENT_C", "RANDOM", "HEAD_PROBE", "MASKED1"):
        methods[name] = {"family": "TREE", "recipe": recipe, "path": name, "margin": margin}
    const_specs = [{"family": "TREE", "recipe": recipe, "path": "CONSTANT", "constant": action, "margin": margin} for action in ACTIONS]
    for spec in const_specs:
        spec["development_normalized_E"] = dev_value(items, spec)
    grids["CONSTANT"] = const_specs
    methods["CONSTANT"] = min(const_specs, key=lambda s: (s["development_normalized_E"], ACTIONS.index(s["constant"])))
    source_actions = {}
    for dataset in sorted({entry["job"]["dataset"] for entry, _, _ in items}):
        source_actions[dataset] = min(
            ACTIONS,
            key=lambda action: (
                dev_value(items, {"family": "TREE", "recipe": recipe, "path": "CONSTANT", "constant": action}, dataset),
                ACTIONS.index(action),
            ),
        )
    methods["SOURCE_CONSTANT"] = {"family": "TREE", "recipe": recipe, "path": "CONSTANT", "source_actions": source_actions, "margin": margin}
    return methods, grids


def compare_development_choice(expected: dict[str, Any], methods: dict[str, Any], grids: dict[str, Any]) -> dict[str, Any]:
    for method, spec in methods.items():
        actual = expected["methods"][method]
        for key, value in spec.items():
            if isinstance(value, float):
                assert_close(value, actual[key], f"development method {method}.{key}")
            else:
                if actual.get(key) != value:
                    raise AssertionError(f"development method {method}.{key}: {actual.get(key)!r} != {value!r}")
    for name, specs in grids.items():
        actual_specs = expected["development_grids"][name]
        if len(actual_specs) != len(specs):
            raise AssertionError(f"development grid length mismatch for {name}")
        for i, spec in enumerate(specs):
            for key, value in spec.items():
                if isinstance(value, float):
                    assert_close(value, actual_specs[i][key], f"development grid {name}[{i}].{key}")
                else:
                    if actual_specs[i].get(key) != value:
                        raise AssertionError(f"development grid {name}[{i}].{key}")
    if expected.get("test_E_used") is not False or expected.get("development_cells") != 4:
        raise AssertionError("development choice exposure markers mismatch")
    return {"methods": len(methods), "grid_candidates": {name: len(specs) for name, specs in grids.items()}}


def audit_fit_scores(run: Path, records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    max_error = 0.0
    point_count = 0
    current_c_count = 0
    branch_check_count = 0
    for entry, fit, _ in records:
        out = run / "fit" / entry["key"] / "output"
        for mode, history in fit["histories"].items():
            for point in history:
                score = raw_loss_npz(out / f"{point['checkpoint']}.npz")
                max_error = max(max_error, abs(score - float(point["V"])))
                point_count += 1
        if fit["job"]["family"] == "TREE":
            fork_point = at_step(fit["histories"]["JOINT"], int(fit["fork"]))
            off_score = raw_loss_npz(out / "current_off.npz")
            recomputed_c = 100.0 * (off_score - float(fork_point["V"])) / float(fit["histories"]["JOINT"][0]["V"])
            max_error = max(max_error, abs(recomputed_c - float(fit["current_C_pct_F0"])))
            current_c_count += 1
            branch_check_count += len(fit.get("branch_checks", []))
    if max_error > 1e-10:
        raise AssertionError(f"fit score audit max error {max_error}")
    return {"fit_point_scores": point_count, "current_c_checks": current_c_count, "branch_checks_seen": branch_check_count, "max_abs_error": max_error}


def audit_forecast_scores(run: Path, records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    max_error = 0.0
    score_count = 0
    for entry, _, forecast in records:
        out = run / "forecast" / entry["key"] / "output"
        for checkpoint, expected in forecast["scores"].items():
            actual = raw_loss_npz(out / f"{checkpoint}.npz")
            max_error = max(max_error, abs(actual - float(expected)))
            score_count += 1
    if max_error > 1e-10:
        raise AssertionError(f"forecast score audit max error {max_error}")
    return {"forecast_checkpoint_scores": score_count, "max_abs_error": max_error}


def rows_and_signals(plan: dict[str, Any], choice: dict[str, Any], test_records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (entry["job"]["dataset"], entry["job"]["condition"], entry["job"]["seed"], entry["job"]["family"], entry["job"]["recipe"]): (entry, fit, forecast)
        for entry, fit, forecast in test_records
    }
    rows: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    band = float(plan["descriptive_band_pct_F0"])
    for dataset in sorted(plan["data"]["test"]):
        for condition in ("FULL90", "SPREAD30"):
            for seed in (28002, 28003, 28004):
                for method in ALL_METHODS:
                    spec = dict(choice["methods"][method])
                    if "source_actions" in spec:
                        spec["constant"] = spec["source_actions"][dataset]
                    entry, fit, forecast = by_key[(dataset, condition, seed, spec["family"], spec["recipe"])]
                    option = method_option(fit, spec)
                    scores = forecast["scores"]
                    f0 = float(scores["JOINT_0"])
                    selected_e = float(scores[option["selected"]["checkpoint"]])
                    final_e = float(scores[option["final"]["checkpoint"]])
                    rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "seed": seed,
                            "method": method,
                            "recipe": spec["recipe"],
                            "action": option["action"],
                            "selected_step": option["selected"]["step"],
                            "E_selected": selected_e,
                            "E_final": final_e,
                            "F0": f0,
                            "normalized_E": selected_e / f0,
                            "fit_key": entry["key"],
                            "selected_checkpoint": option["selected"]["checkpoint"],
                            "selected_model_hash": option["selected"].get("model_hash"),
                        }
                    )
                    if method == "PROBE":
                        options = path_options(fit, float(spec["margin"]))
                        action_losses = {
                            "STOP": float(scores[options["STOP"]["selected"]["checkpoint"]]),
                            "HEAD": float(scores[options["HEAD1"]["selected"]["checkpoint"]]),
                            "JOINT": float(scores[options["JOINT"]["selected"]["checkpoint"]]),
                        }
                        best = min(action_losses.values())
                        acceptable = [a for a in ACTIONS if 100.0 * (action_losses[a] - best) / f0 <= band]
                        fork = int(fit["fork"])
                        qstep = fork + int(fit["probe_steps"])
                        trial_u = 100.0 * (at_step(fit["histories"]["HEAD1"], qstep)["V"] - at_step(fit["histories"]["JOINT"], qstep)["V"]) / fit["histories"]["JOINT"][0]["V"]
                        final_u = 100.0 * (
                            float(scores[options["HEAD1"]["final"]["checkpoint"]]) - float(scores[options["JOINT"]["final"]["checkpoint"]])
                        ) / f0
                        signals.append(
                            {
                                "dataset": dataset,
                                "condition": condition,
                                "seed": seed,
                                "action": option["action"],
                                "acceptable_actions": acceptable,
                                "correct_within_band": option["action"] in acceptable,
                                "selected_oracle_regret_pct_F0": 100.0 * (action_losses[option["action"]] - best) / f0,
                                "trial_U_V_pct_F0": trial_u,
                                "final_U_E_pct_F0": final_u,
                                "C_V_pct_F0": fit["current_C_pct_F0"],
                                "oracle_E": best,
                                "oracle_action_losses": action_losses,
                            }
                        )
    if len(rows) != 144 or len(signals) != 12:
        raise AssertionError(f"expected 144 rows and 12 signals, got {len(rows)} and {len(signals)}")
    return rows, signals


def aggregate_methods(plan: dict[str, Any], choice: dict[str, Any], rows: list[dict[str, Any]], signals: list[dict[str, Any]], timing_records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[str, Any] = {}
    baseline = [r for r in rows if r["method"] == "PROBE"]
    for method in ALL_METHODS:
        current = [r for r in rows if r["method"] == method]
        pairs = [
            (r, next(b for b in baseline if all(r[k] == b[k] for k in ("dataset", "condition", "seed"))))
            for r in current
        ]
        grouped[method] = {
            "mean_normalized_E": float(np.mean([r["normalized_E"] for r in current])),
            "PROBE_advantage_pct_F0": float(np.mean([100.0 * (r["E_selected"] - b["E_selected"]) / r["F0"] for r, b in pairs])),
            "per_seed_PROBE_advantage": {
                str(seed): float(np.mean([100.0 * (r["E_selected"] - b["E_selected"]) / r["F0"] for r, b in pairs if r["seed"] == seed]))
                for seed in (28002, 28003, 28004)
            },
            "per_source_PROBE_advantage": {
                dataset: float(np.mean([100.0 * (r["E_selected"] - b["E_selected"]) / r["F0"] for r, b in pairs if r["dataset"] == dataset]))
                for dataset in plan["data"]["test"]
            },
        }
    times: dict[str, Any] = {}
    for method in TIMED_METHODS:
        method_records = [r for r in timing_records if r["method"] == method]
        if len(method_records) != 12:
            raise AssertionError(f"expected 12 timing records for {method}, got {len(method_records)}")
        times[method] = {
            "mean_adaptation_seconds": float(np.mean([r["adaptation_seconds"] for r in method_records])),
            "per_seed_seconds": {
                str(seed): float(sum(r["adaptation_seconds"] for r in method_records if r["job"]["seed"] == seed))
                for seed in (28002, 28003, 28004)
            },
        }
    band = float(plan["descriptive_band_pct_F0"])
    efficiency: dict[str, Any] = {}
    for method in ("FULL", "FIXED", "EARLY_STOP", "HEAD", "WIDE"):
        savings = {str(seed): 100.0 * (1.0 - times["PROBE"]["per_seed_seconds"][str(seed)] / times[method]["per_seed_seconds"][str(seed)]) for seed in (28002, 28003, 28004)}
        gains = grouped[method]["per_seed_PROBE_advantage"]
        seed_ok = {
            str(seed): ((gains[str(seed)] >= -band and savings[str(seed)] > 5.0) or (gains[str(seed)] > band and savings[str(seed)] >= 0.0))
            for seed in (28002, 28003, 28004)
        }
        source_ok = all(value >= -band for value in grouped[method]["per_source_PROBE_advantage"].values())
        efficiency[method] = {"saving_pct_by_seed": savings, "seed_passes": seed_ok, "source_quality_guard": source_ok, "passes": all(seed_ok.values()) and source_ok}
    distinct_actions = len({signal["action"] for signal in signals})
    best_simple = min(("FULL", "HEAD", "WIDE"), key=lambda m: choice["methods"][m]["development_normalized_E"])
    gates = {
        "predictive_conditional_screen": distinct_actions >= 2 and all(grouped[m]["PROBE_advantage_pct_F0"] > band for m in ("CONSTANT", "SOURCE_CONSTANT", "RANDOM")),
        "signal_ablation_screen": all(grouped[m]["PROBE_advantage_pct_F0"] > band for m in ("CURRENT_C", "HEAD_PROBE")),
        "quality_advantage_vs_strong": {m: grouped[m]["PROBE_advantage_pct_F0"] for m in ("FULL", "HEAD", "WIDE", "FIXED", "EARLY_STOP")},
        "practical_quality_cost_screen": all(efficiency[m]["passes"] for m in {"FIXED", "EARLY_STOP", best_simple}),
        "best_development_simple_baseline": best_simple,
    }
    timing = {"replays": len(timing_records), "methods": times, "comparisons": efficiency, "single_timing_per_cell": True}
    return grouped, {"gates": gates, "timing": timing, "distinct_actions": distinct_actions}


def interval_bootstrap(run: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(28333)
    draws = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        starts = rng.integers(0, 20, size=(2000, 10))
        indices = np.stack([starts, (starts + 1) % 20], axis=-1).reshape(2000, 20)
        draws[dataset] = np.stack([np.bincount(i, minlength=20) for i in indices])
    cache: dict[tuple[str, str], np.ndarray] = {}

    def boot(row: dict[str, Any]) -> np.ndarray:
        key = (row["fit_key"], row["selected_checkpoint"])
        if key not in cache:
            path = run / "forecast" / key[0] / "output" / f"{key[1]}.npz"
            with np.load(path, allow_pickle=False) as z:
                origin_count = int(z["target"].shape[0])
            if origin_count != 20:
                raise AssertionError(f"{key} expected 20 subsampled E origins")
            cache[key] = bootstrap_scores_npz(path, draws[row["dataset"]])
        return cache[key]

    probes = [r for r in rows if r["method"] == "PROBE"]
    result = {}
    for method in sorted({r["method"] for r in rows}):
        differences = []
        for row in [r for r in rows if r["method"] == method]:
            probe = next(p for p in probes if all(p[k] == row[k] for k in ("dataset", "condition", "seed")))
            differences.append(100.0 * (boot(row) - boot(probe)) / row["F0"])
        values = np.mean(differences, axis=0)
        result[method] = np.quantile(values, [0.05, 0.95]).tolist()
    return {
        "interval": 0.9,
        "draws": 2000,
        "block_origins": 2,
        "circular": True,
        "PROBE_advantage_pct_F0": result,
        "scope": "Conditional on these two sources/windows; shared seed/condition/method draws; not population source inference.",
    }


def lr_clipping_controls(choice: dict[str, Any], test_records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    controls = []
    margin = float(choice["methods"]["PROBE"]["margin"])
    for entry, fit, forecast in test_records:
        job = entry["job"]
        if job["family"] != "TREE" or job["seed"] != 28002:
            continue
        opts = path_options(fit, margin)
        scores = forecast["scores"]
        f0 = float(scores["JOINT_0"])
        controls.append(
            {
                "dataset": job["dataset"],
                "condition": job["condition"],
                "seed": job["seed"],
                "recipe": job["recipe"],
                "selected_U_pct_F0": 100.0 * (float(scores[opts["HEAD1"]["selected"]["checkpoint"]]) - float(scores[opts["JOINT"]["selected"]["checkpoint"]])) / f0,
                "final_U_pct_F0": 100.0 * (float(scores[opts["HEAD1"]["final"]["checkpoint"]]) - float(scores[opts["JOINT"]["final"]["checkpoint"]])) / f0,
                "clipping_difference_pct_F0": 100.0 * (float(scores[opts["HEAD1"]["final"]["checkpoint"]]) - float(scores[opts["MASKED1"]["final"]["checkpoint"]])) / f0,
                "PROBE_action": opts["PROBE"]["action"],
                "PROBE_selected_E": float(scores[opts["PROBE"]["selected"]["checkpoint"]]),
            }
        )
    if len(controls) != 12:
        raise AssertionError(f"expected 12 LR/clipping controls, got {len(controls)}")
    return controls


def check_guard(path: Path) -> dict[str, Any]:
    status = read_json(path)
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons") not in ([], None):
        raise AssertionError(f"guard did not complete cleanly: {path}")
    return status


def audit_guards_and_chronology(run: Path, plan: dict[str, Any], dev_records: list[Any], test_records: list[Any], choice: dict[str, Any]) -> dict[str, Any]:
    guard_count = 0
    smoke = check_guard(run / "smoke/guard/status.json")
    guard_count += 1
    dev_seal = read_json(run / "dev_fits_sealed.json")
    test_seal = read_json(run / "test_fits_sealed.json")
    timing = read_json(run / "timing_completed.json")
    completed = read_json(run / "completed.json")
    dev_fit_finishes, dev_forecast_finishes = [], []
    test_fit_finishes, test_forecast_starts, test_forecast_finishes = [], [], []
    for entry, _, _ in dev_records:
        _, fit_guard, _ = result_paths(run, entry["key"], "fit")
        _, forecast_guard, _ = result_paths(run, entry["key"], "forecast")
        fit_status, forecast_status = check_guard(fit_guard), check_guard(forecast_guard)
        guard_count += 2
        dev_fit_finishes.append(parse_dt(fit_status["finished_at"]))
        dev_forecast_finishes.append(parse_dt(forecast_status["finished_at"]))
    for entry, _, _ in test_records:
        _, fit_guard, _ = result_paths(run, entry["key"], "fit")
        _, forecast_guard, _ = result_paths(run, entry["key"], "forecast")
        fit_status, forecast_status = check_guard(fit_guard), check_guard(forecast_guard)
        guard_count += 2
        test_fit_finishes.append(parse_dt(fit_status["finished_at"]))
        test_forecast_starts.append(parse_dt(forecast_status["started_at"]))
        test_forecast_finishes.append(parse_dt(forecast_status["finished_at"]))
    replay_starts = []
    for key in timing["keys"]:
        status = check_guard(run / key / "guard/status.json")
        guard_count += 1
        replay_starts.append(parse_dt(status["started_at"]))
    choice_time = parse_dt(choice["sealed_utc"])
    dev_seal_time = parse_dt(dev_seal["sealed_utc"])
    test_seal_time = parse_dt(test_seal["sealed_utc"])
    checks = {
        "smoke_before_dev_seal": parse_dt(smoke["finished_at"]) <= dev_seal_time,
        "dev_fits_before_dev_seal": max(dev_fit_finishes) <= dev_seal_time,
        "dev_forecasts_before_choice": max(dev_forecast_finishes) <= choice_time,
        "choice_before_test_fit_seal": choice_time <= test_seal_time,
        "test_fits_before_test_seal": max(test_fit_finishes) <= test_seal_time,
        "all_test_fits_before_any_test_forecast": max(test_fit_finishes) <= min(test_forecast_starts),
        "choice_before_all_test_forecasts": choice_time <= min(test_forecast_starts),
        "test_forecasts_before_replays": max(test_forecast_finishes) <= min(replay_starts),
        "run_completed_after_timing": parse_dt(timing["finished_utc"]) <= parse_dt(completed["finished_utc"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"chronology failure: {checks}")
    return {
        "guards_checked": guard_count,
        "dev_fit_indices": len(dev_seal["indices"]),
        "test_fit_indices": len(test_seal["indices"]),
        "timing_replays": len(timing["keys"]),
        "checks": checks,
    }


def audit_replays(run: Path, choice: dict[str, Any], timing_keys: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timing_records = []
    exact_predictions = 0
    for key in timing_keys:
        record = read_json(run / key / "output/result.json")
        if not record.get("completed") or not record.get("reference_exact") or record.get("holdout_opened") is not False:
            raise AssertionError(f"bad replay record {key}")
        method = record["method"]
        job = record["job"]
        spec = dict(choice["methods"][method])
        if "source_actions" in spec:
            spec["constant"] = spec["source_actions"][job["dataset"]]
        ref_key = f"{job['episode']}/{job['dataset']}/{job['condition']}/{job['family']}/r{job['recipe']}/s{job['seed']}"
        reference = read_json(run / "fit" / ref_key / "output/result.json")
        expected = method_option(reference, spec)["selected"]
        if record["selected_step"] != expected["step"] or record["selected_model_hash"] != expected.get("model_hash"):
            raise AssertionError(f"replay selected reference mismatch {key}")
        assert_close(record["selected_V"], expected["V"], f"replay selected V {key}")
        with np.load(run / key / "output/val_predictions.npz", allow_pickle=False) as got, np.load(run / "fit" / ref_key / "output" / f"{expected['checkpoint']}.npz", allow_pickle=False) as ref:
            np.testing.assert_array_equal(got["prediction"], ref["prediction"])
        exact_predictions += 1
        timing_records.append(record)
    if len(timing_records) != 72:
        raise AssertionError(f"expected 72 timing records, got {len(timing_records)}")
    return timing_records, {"timing_records": len(timing_records), "selected_prediction_exact": exact_predictions}


def compare_with_main_outputs(rows: list[dict[str, Any]], grouped: dict[str, Any], gates_timing: dict[str, Any], intervals: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = RESULTS / "summary.json"
    metrics_path = RESULTS / "metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("main analysis outputs are required for full independent audit")
    summary = read_json(summary_path)
    if summary.get("test_cells") != 12 or summary.get("publication_ready") is not False:
        raise AssertionError("main summary high-level markers mismatch")
    for method, stats in grouped.items():
        actual = summary["methods"][method]
        assert_close(stats["mean_normalized_E"], actual["mean_normalized_E"], f"{method}.mean_normalized_E")
        assert_close(stats["PROBE_advantage_pct_F0"], actual["PROBE_advantage_pct_F0"], f"{method}.PROBE_advantage_pct_F0")
        for seed, value in stats["per_seed_PROBE_advantage"].items():
            assert_close(value, actual["per_seed_PROBE_advantage"][seed], f"{method}.seed.{seed}")
        for dataset, value in stats["per_source_PROBE_advantage"].items():
            assert_close(value, actual["per_source_PROBE_advantage"][dataset], f"{method}.source.{dataset}")
    expected_gates = gates_timing["gates"]
    actual_gates = summary["gates"]
    for key in ("predictive_conditional_screen", "signal_ablation_screen", "practical_quality_cost_screen", "best_development_simple_baseline"):
        if expected_gates[key] != actual_gates[key]:
            raise AssertionError(f"gate mismatch {key}: {expected_gates[key]!r} != {actual_gates[key]!r}")
    for method, value in expected_gates["quality_advantage_vs_strong"].items():
        assert_close(value, actual_gates["quality_advantage_vs_strong"][method], f"gate quality {method}")
    for method, stats in gates_timing["timing"]["methods"].items():
        actual = summary["timing"]["methods"][method]
        assert_close(stats["mean_adaptation_seconds"], actual["mean_adaptation_seconds"], f"timing {method}")
        for seed, value in stats["per_seed_seconds"].items():
            assert_close(value, actual["per_seed_seconds"][seed], f"timing {method}.{seed}")
    for method, actual in summary["timing"]["comparisons"].items():
        expected = gates_timing["timing"]["comparisons"][method]
        if expected["passes"] != actual["passes"] or expected["source_quality_guard"] != actual["source_quality_guard"] or expected["seed_passes"] != actual["seed_passes"]:
            raise AssertionError(f"timing comparison boolean mismatch {method}")
        for seed, value in expected["saving_pct_by_seed"].items():
            assert_close(value, actual["saving_pct_by_seed"][seed], f"saving {method}.{seed}")
    for method, pair in intervals["PROBE_advantage_pct_F0"].items():
        assert_close(pair[0], summary["conditional_quality_intervals"]["PROBE_advantage_pct_F0"][method][0], f"interval {method}.lo")
        assert_close(pair[1], summary["conditional_quality_intervals"]["PROBE_advantage_pct_F0"][method][1], f"interval {method}.hi")
    if len(summary["LR_clipping_controls"]) != len(controls):
        raise AssertionError("LR/clipping control count mismatch")
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 144:
        raise AssertionError(f"main metrics row count {len(csv_rows)} != 144")
    expected_by_key = {
        (str(r["dataset"]), str(r["condition"]), str(r["seed"]), str(r["method"])): r
        for r in rows
    }
    for row in csv_rows:
        key = (row["dataset"], row["condition"], row["seed"], row["method"])
        if key not in expected_by_key:
            raise AssertionError(f"unexpected metrics.csv row {key}")
        expected = expected_by_key[key]
        for field in row:
            if field not in expected:
                continue
            if field in {"dataset", "condition", "method", "action", "fit_key", "selected_checkpoint"}:
                if str(expected[field]) != row[field]:
                    raise AssertionError(f"metrics.csv mismatch {key}.{field}: {row[field]!r} != {expected[field]!r}")
            else:
                assert_close(float(expected[field]), float(row[field]), f"metrics.csv {key}.{field}")
    return {"summary_sha256": sha(summary_path), "metrics_sha256": sha(metrics_path), "main_rows": len(csv_rows), "row_values_checked": len(csv_rows)}


def run_audit() -> dict[str, Any]:
    output_path = RUN / "independent_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    plan = read_json(RUN / "plan.json")
    completed = read_json(RUN / "completed.json")
    if not completed.get("completed"):
        raise AssertionError("run is not completed")
    plan_hash = sha(RUN / "plan.json")
    if completed.get("plan_sha256") != plan_hash:
        raise AssertionError("completed plan hash mismatch")
    hash_checks = verify_plan_hashes(plan)
    dev_records = load_records(RUN, plan, "dev")
    test_records = load_records(RUN, plan, "test")
    dev_methods, dev_grids = independent_development_choice(dev_records)
    choice = read_json(RUN / "development_choice.json")
    dev_compare = compare_development_choice(choice, dev_methods, dev_grids)
    fit_score_audit = audit_fit_scores(RUN, dev_records + test_records)
    forecast_score_audit = audit_forecast_scores(RUN, dev_records + test_records)
    rows, signals = rows_and_signals(plan, choice, test_records)
    timing_records, replay_audit = audit_replays(RUN, choice, read_json(RUN / "timing_completed.json")["keys"])
    grouped, gates_timing = aggregate_methods(plan, choice, rows, signals, timing_records)
    intervals = interval_bootstrap(RUN, rows)
    controls = lr_clipping_controls(choice, test_records)
    chronology = audit_guards_and_chronology(RUN, plan, dev_records, test_records, choice)
    main_compare = compare_with_main_outputs(rows, grouped, gates_timing, intervals, controls)
    payload = {
        "completed": True,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "script_sha256_before_write": sha(Path(__file__)),
        "plan_sha256": plan_hash,
        "hash_checks": hash_checks,
        "development_choice_audit": dev_compare,
        "fit_score_audit": fit_score_audit,
        "forecast_score_audit": forecast_score_audit,
        "row_metrics_count": len(rows),
        "row_metrics": rows,
        "signals_count": len(signals),
        "signals": signals,
        "method_aggregates": grouped,
        "gates": gates_timing["gates"],
        "timing": gates_timing["timing"],
        "replay_audit": replay_audit,
        "chronology": chronology,
        "conditional_quality_intervals": intervals,
        "LR_clipping_controls": controls,
        "main_output_comparison": main_compare,
        "outcome_uncertainty_limits": [
            "Bootstrap intervals are within-window, conditional on the two selected source families and periods.",
            "The three seeds share the same targets and are not independent dataset replications.",
            "BMRA channels passed an availability screen over the proposed windows; this is not a forecast-performance selection but limits fresh-source claims.",
        ],
    }
    payload["independent_audit_sha256"] = write_json_x(output_path, payload)
    return payload


def record_failure(exc: BaseException) -> None:
    target = RUN / "independent_audit_failure.json"
    if target.exists():
        index = 2
        while (RUN / f"independent_audit_failure_{index:02d}.json").exists():
            index += 1
        target = RUN / f"independent_audit_failure_{index:02d}.json"
    write_json_x(
        target,
        {
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "script_sha256": sha(Path(__file__)),
        },
    )


def self_test() -> dict[str, Any]:
    history = [{"step": 0, "V": 3.0, "checkpoint": "a"}, {"step": 1, "V": 2.0, "checkpoint": "b"}, {"step": 2, "V": 2.0, "checkpoint": "c"}]
    assert selected(history)[0]["checkpoint"] == "b"
    assert selected(history, patience=1) == (history[1], 2)
    assert probe_choice({"STOP": 1.0, "HEAD": 0.9, "JOINT": 0.89}, 1.0, 0.0) == "JOINT"
    assert probe_choice({"STOP": 1.0, "HEAD": 0.9, "JOINT": 0.899}, 1.0, 0.25) == "HEAD"
    assert random_choice("dev/jena/FULL90/28000") in ACTIONS
    return {"self_test": True, "script_sha256": sha(Path(__file__)), "actions": ACTIONS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run the full audit after main completion")
    mode.add_argument("--self-test", action="store_true", help="run CPU-only internal function checks without reading study outputs")
    args = parser.parse_args()
    try:
        payload = run_audit() if args.run else self_test()
    except BaseException as exc:
        if args.run:
            record_failure(exc)
        raise
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
