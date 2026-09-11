"""Phase 0 data gate for the Study36 dependence-based PEFT direction.

This is a CPU-only design-validity screen. It intentionally does not train a
foundation model or a PEFT adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "peft_dependence_screen_v1" / "data_gate"
PLAN = ROOT / "_docs" / "notes" / "tsfm_topics" / "07_research_direction" / "36_dependence_peft_paper_plan_20260911.md"
OLD_RHO_DOC = Path("E:/CODING/proj/covariate-trust-pilot/_docs/reference/rho_definition_synthetic_vs_real.md")

SEEDS = (36000, 36001)
LENGTH = 512
CONTEXT = 128
N_ORIGINS = 16
SPLITS = {"train": 64, "val": 32, "devtest": 128}
VALUES = np.array([-1.0, 0.0, 1.0])
P_FORWARD = np.array([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.7, 0.1, 0.2]])
P_REVERSE = P_FORWARD.T
P_BALANCED = np.array([[0.2, 0.4, 0.4], [0.4, 0.2, 0.4], [0.4, 0.4, 0.2]])


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    parameter: float | None
    transition: np.ndarray | None


CONDITIONS = (
    Condition("ar_rho_neg06", "ar1_gaussian", -0.6, None),
    Condition("ar_rho_zero", "ar1_gaussian", 0.0, None),
    Condition("ar_rho_pos06", "ar1_gaussian", 0.6, None),
    Condition("markov_forward", "three_state_markov", None, P_FORWARD),
    Condition("markov_reverse", "three_state_markov", None, P_REVERSE),
    Condition("markov_balanced", "three_state_markov", None, P_BALANCED),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origins() -> np.ndarray:
    return np.linspace(CONTEXT, LENGTH - 1, N_ORIGINS, dtype=int)


def generate_ar(rho: float, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    x = np.empty((n_paths, LENGTH), dtype=np.float64)
    x[:, 0] = rng.normal(size=n_paths)
    noise_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
    eps = rng.normal(size=(n_paths, LENGTH - 1)) * noise_scale
    for t in range(1, LENGTH):
        x[:, t] = rho * x[:, t - 1] + eps[:, t - 1]
    return x


def generate_markov(transition: np.ndarray, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    states = np.empty((n_paths, LENGTH), dtype=np.int64)
    states[:, 0] = rng.integers(0, 3, size=n_paths)
    cumulative = np.cumsum(transition, axis=1)
    draws = rng.random((n_paths, LENGTH - 1))
    for t in range(1, LENGTH):
        previous = states[:, t - 1]
        states[:, t] = (draws[:, t - 1, None] > cumulative[previous]).sum(axis=1)
    return VALUES[states]


def generate_condition(condition: Condition, condition_index: int, seed: int) -> dict[str, np.ndarray]:
    children = np.random.SeedSequence([seed, condition_index]).spawn(len(SPLITS))
    result = {}
    for child, (split, n_paths) in zip(children, SPLITS.items()):
        rng = np.random.default_rng(child)
        if condition.kind == "ar1_gaussian":
            assert condition.parameter is not None
            result[split] = generate_ar(condition.parameter, rng, n_paths)
        else:
            assert condition.transition is not None
            result[split] = generate_markov(condition.transition, rng, n_paths)
    return result


def lag1_acf(x: np.ndarray) -> float:
    left = x[..., :-1].reshape(-1)
    right = x[..., 1:].reshape(-1)
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def mean_periodogram(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, axis=1)
    return (np.abs(fft) ** 2 / x.shape[1]).mean(axis=0)


def collect_xy(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = origins()
    latest = paths[:, idx - 1].reshape(-1)
    label = paths[:, idx].reshape(-1)
    return latest, label


def fit_ols(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return coef


def mse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((y - pred) ** 2))


def markov_predictor_from_transitions(paths: np.ndarray, use_origins_only: bool) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=np.float64)
    if use_origins_only:
        idx = origins()
        previous = paths[:, idx - 1].reshape(-1)
        current = paths[:, idx].reshape(-1)
    else:
        previous = paths[:, :-1].reshape(-1)
        current = paths[:, 1:].reshape(-1)
    prev_idx = np.searchsorted(VALUES, previous)
    cur_idx = np.searchsorted(VALUES, current)
    for i, j in zip(prev_idx, cur_idx):
        counts[i, j] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.ones_like(counts) / 3.0, where=row_sums > 0)
    return probs @ VALUES


def markov_population_risk(transition: np.ndarray, pred_by_state: np.ndarray) -> float:
    total = 0.0
    for i in range(3):
        for j in range(3):
            total += transition[i, j] * (VALUES[j] - pred_by_state[i]) ** 2 / 3.0
    return float(total)


def ar_population_risk(rho: float, coef: np.ndarray, quadratic: bool) -> float:
    if quadratic:
        a, b, c = coef
        return float((rho - b) ** 2 + (1.0 - rho * rho) + a * a + 3.0 * c * c + 2.0 * a * c)
    a, b = coef
    return float((rho - b) ** 2 + (1.0 - rho * rho) + a * a)


def predictor_rows(condition: Condition, seed: int, data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    train_x, train_y = collect_xy(data["train"])
    rows = []
    train_var = float(np.var(train_y))

    mean_pred = float(np.mean(train_y))
    linear_coef = fit_ols(np.column_stack([np.ones_like(train_x), train_x]), train_y)
    quad_coef = fit_ols(np.column_stack([np.ones_like(train_x), train_x, train_x**2]), train_y)
    predictors: list[tuple[str, Callable[[np.ndarray], np.ndarray], float | None]] = [
        ("train_mean", lambda latest: np.full_like(latest, mean_pred, dtype=np.float64), None),
        ("lag1_linear", lambda latest: linear_coef[0] + linear_coef[1] * latest, None),
        ("lag1_quadratic", lambda latest: quad_coef[0] + quad_coef[1] * latest + quad_coef[2] * latest**2, None),
    ]

    if condition.kind == "ar1_gaussian":
        assert condition.parameter is not None
        oracle_risk = 1.0 - condition.parameter * condition.parameter
        predictors[1] = (
            "lag1_linear",
            predictors[1][1],
            ar_population_risk(condition.parameter, linear_coef, quadratic=False),
        )
        predictors[2] = (
            "lag1_quadratic",
            predictors[2][1],
            ar_population_risk(condition.parameter, quad_coef, quadratic=True),
        )
    else:
        assert condition.transition is not None
        oracle_pred = condition.transition @ VALUES
        oracle_risk = markov_population_risk(condition.transition, oracle_pred)
        full_transition_pred = markov_predictor_from_transitions(data["train"], use_origins_only=False)
        origin_transition_pred = markov_predictor_from_transitions(data["train"], use_origins_only=True)
        predictors.append(
            (
                "estimated_transition_full_train",
                lambda latest, p=full_transition_pred: p[np.searchsorted(VALUES, latest)],
                markov_population_risk(condition.transition, full_transition_pred),
            )
        )
        predictors.append(
            (
                "estimated_transition_origin_train",
                lambda latest, p=origin_transition_pred: p[np.searchsorted(VALUES, latest)],
                markov_population_risk(condition.transition, origin_transition_pred),
            )
        )
        predictors.append(
            (
                "true_oracle_transition",
                lambda latest, p=oracle_pred: p[np.searchsorted(VALUES, latest)],
                oracle_risk,
            )
        )
        predictors[2] = (
            "lag1_quadratic",
            predictors[2][1],
            markov_population_risk(condition.transition, quad_coef[0] + quad_coef[1] * VALUES + quad_coef[2] * VALUES**2),
        )

    for split in ("val", "devtest"):
        x, y = collect_xy(data[split])
        for name, pred_fn, population_risk in predictors:
            score = mse(y, pred_fn(x))
            rows.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "split": split,
                    "predictor": name,
                    "mse": score,
                    "normalized_mse_train_var": score / train_var,
                    "oracle_population_mse": oracle_risk,
                    "population_mse": "" if population_risk is None else population_risk,
                    "population_excess_over_oracle_pct": ""
                    if population_risk is None
                    else 100.0 * (population_risk / oracle_risk - 1.0),
                    "train_examples_origins": len(train_y),
                    "train_transitions_full": data["train"].shape[0] * (LENGTH - 1),
                }
            )
    return rows


def condition_metric_rows(condition: Condition, seed: int, data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for split, paths in data.items():
        flat = paths.reshape(-1)
        row = {
            "condition": condition.name,
            "seed": seed,
            "split": split,
            "n_paths": paths.shape[0],
            "length": paths.shape[1],
            "mean": float(np.mean(flat)),
            "variance": float(np.var(flat)),
            "lag1_acf": lag1_acf(paths),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        }
        if condition.kind == "three_state_markov":
            for value in VALUES:
                row[f"freq_value_{value:g}"] = float(np.mean(flat == value))
        rows.append(row)
    return rows


def population_audit() -> dict[str, object]:
    reflection = np.eye(3)[::-1]
    forward_mean = P_FORWARD @ VALUES
    reverse_mean = P_REVERSE @ VALUES
    features = np.column_stack([np.ones(3), VALUES, VALUES**2])
    forward_coef = np.linalg.solve(features, forward_mean)
    reverse_coef = np.linalg.solve(features, reverse_mean)

    def risk_over_horizons(transition: np.ndarray) -> list[float]:
        risks = []
        for horizon in range(1, 129):
            p_h = np.linalg.matrix_power(transition, horizon)
            pred = p_h @ VALUES
            risks.append(markov_population_risk(p_h, pred))
        return risks

    def acf_over_horizons(transition: np.ndarray) -> list[float]:
        acfs = []
        for horizon in range(1, 129):
            p_h = np.linalg.matrix_power(transition, horizon)
            acfs.append(float(np.mean(VALUES * (p_h @ VALUES))))
        return acfs

    freq = np.linspace(0, np.pi, 257)
    psd_forward = markov_population_spectrum(P_FORWARD, freq)
    psd_reverse = markov_population_spectrum(P_REVERSE, freq)
    return {
        "stationary_uniform_error": float(np.max(np.abs(np.ones(3) / 3 @ P_FORWARD - np.ones(3) / 3))),
        "reflection_conjugacy_error": float(np.max(np.abs(reflection @ P_FORWARD @ reflection - P_REVERSE))),
        "observation_sign_flip_error": float(np.max(np.abs(reflection @ VALUES + VALUES))),
        "conditional_mean_sign_conjugacy_error": float(np.max(np.abs(reverse_mean + reflection @ forward_mean))),
        "acf_difference_h1_to_128": float(np.max(np.abs(np.array(acf_over_horizons(P_FORWARD)) - np.array(acf_over_horizons(P_REVERSE))))),
        "risk_difference_h1_to_128": float(np.max(np.abs(np.array(risk_over_horizons(P_FORWARD)) - np.array(risk_over_horizons(P_REVERSE))))),
        "psd_difference_257_frequencies": float(np.max(np.abs(psd_forward - psd_reverse))),
        "risk_h1_forward": markov_population_risk(P_FORWARD, forward_mean),
        "risk_h1_reverse": markov_population_risk(P_REVERSE, reverse_mean),
        "quadratic_forward_intercept_x_x2": forward_coef.tolist(),
        "quadratic_reverse_intercept_x_x2": reverse_coef.tolist(),
        "conditional_mean_forward": forward_mean.tolist(),
        "conditional_mean_reverse": reverse_mean.tolist(),
    }


def markov_population_spectrum(transition: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    q = transition - np.ones_like(transition) / len(transition)
    identity = np.eye(len(transition))
    gamma0 = np.mean(VALUES**2)
    result = []
    for frequency in frequencies:
        z = np.exp(-1j * frequency)
        tail = np.linalg.solve(identity - z * q, z * q @ VALUES)
        result.append(float(gamma0 + 2.0 * np.real(np.mean(VALUES * tail))))
    return np.array(result)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictors(rows: list[dict[str, object]]) -> dict[str, object]:
    full_excess = []
    origin_excess = []
    quad_excess = []
    for row in rows:
        if row["condition"] in {"markov_forward", "markov_reverse"} and row["split"] == "devtest":
            value = row["population_excess_over_oracle_pct"]
            if value == "":
                continue
            if row["predictor"] == "estimated_transition_full_train":
                full_excess.append(float(value))
            if row["predictor"] == "estimated_transition_origin_train":
                origin_excess.append(float(value))
            if row["predictor"] == "lag1_quadratic":
                quad_excess.append(float(value))
    return {
        "forward_reverse_full_transition_population_excess_pct_max": max(full_excess),
        "forward_reverse_full_transition_population_excess_pct_mean": float(np.mean(full_excess)),
        "forward_reverse_origin_transition_population_excess_pct_max": max(origin_excess),
        "forward_reverse_origin_transition_population_excess_pct_mean": float(np.mean(origin_excess)),
        "forward_reverse_quadratic_population_excess_pct_max": max(quad_excess),
        "forward_reverse_quadratic_population_excess_pct_mean": float(np.mean(quad_excess)),
    }


def make_figure(pop: dict[str, object], predictor_summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    names = []
    values = []
    for key in (
        "forward_reverse_full_transition_population_excess_pct_mean",
        "forward_reverse_origin_transition_population_excess_pct_mean",
        "forward_reverse_quadratic_population_excess_pct_mean",
    ):
        names.append(key.replace("forward_reverse_", "").replace("_population_excess_pct_mean", "").replace("_", " "))
        values.append(predictor_summary[key])

    condition_order = ["markov_forward", "markov_reverse"]
    predictors = ["train_mean", "lag1_quadratic", "estimated_transition_full_train", "true_oracle_transition"]
    devtest_mse = {}
    for condition in condition_order:
        devtest_mse[condition] = []
        for predictor in predictors:
            vals = [
                float(row["mse"])
                for row in rows
                if row["condition"] == condition and row["split"] == "devtest" and row["predictor"] == predictor
            ]
            devtest_mse[condition].append(float(np.mean(vals)))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), layout="constrained")
    axes[0].bar(names, values, color=["#4C78A8", "#72B7B2", "#F58518"])
    axes[0].set_title("Simple train-only controls are near oracle")
    axes[0].set_ylabel("Population MSE excess over oracle (%)")
    axes[0].tick_params(axis="x", rotation=20)

    x = np.arange(len(predictors))
    width = 0.36
    axes[1].bar(x - width / 2, devtest_mse["markov_forward"], width, label="Forward", color="#4C78A8")
    axes[1].bar(x + width / 2, devtest_mse["markov_reverse"], width, label="Reverse", color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.replace("_", "\n") for p in predictors], fontsize=8)
    axes[1].set_ylabel("Finite-sample devtest MSE")
    axes[1].set_title("Finite-sample scores are diagnostic only")
    axes[1].legend()

    axes[2].axis("off")
    lines = [
        "Phase 0 decision: STOP before GPU",
        "",
        f"ACF/PSD/risk control error <= {max(pop['acf_difference_h1_to_128'], pop['risk_difference_h1_to_128'], pop['psd_difference_257_frequencies']):.2e}",
        "But Forward and Reverse are exact sign relabellings.",
        "The latest observed state is sufficient for the oracle.",
        "A small quadratic or transition table is a required strong control.",
        "",
        "No LoRA or foundation-model failure was measured here.",
    ]
    axes[2].text(0, 0.98, "\n".join(lines), va="top", fontsize=10.5)
    fig.suptitle("Study36 Phase 0 data gate", fontsize=14)
    fig.savefig(OUT / "02_data_gate.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, object], predictor_summary: dict[str, object]) -> None:
    report = f"""# Study36 Phase 0 Data Gate

Created: {summary['finished_utc']}

## Executive Summary

Overall decision: **{summary['overall_decision']}**.

The proposed forward/reverse Markov contrast technically controls marginal distribution, population ACF/PSD, and Bayes risk, but it does **not** isolate the intended temporal-dependence mechanism. Forward and reverse are exact sign relabellings of each other under the chosen observation values `(-1, 0, 1)`, and the latest observed state is sufficient for one-step prediction. A tiny train-only transition table or quadratic readout is therefore a mandatory strong control and already sits near the oracle in population risk.

This is a design-gate result, not a trained-model result. No LoRA, Chronos, MOMENT, Time-PEFT, or new adapter training was started.

## Domain Context

```text
domain              time-series ML / forecasting PEFT
downstream decision decide whether to spend GPU time and method-design effort
target claim        temporal-relation adapter is needed beyond frequency/difficulty/simple controls
quality bar         publication-oriented screen, zero tolerance for leakage/confounded DGP
main leakage paths  shared paths across splits, D-tuned generator changes, whole-period real descriptors
```

## Gate Results

```text
population controls forward/reverse   PASS
construct validity for PEFT mechanism FAIL
simple-control necessity              FAIL for internal-PEFT claim
real-data support from old records     HISTORICAL WARNING, not a stop reason
GPU Phase 1                            NOT STARTED
adapter Phase 2                        NOT STARTED
external Phase 3/4                     NOT STARTED
```

## Key Numbers

```text
reflection conjugacy error             {summary['population_checks']['reflection_conjugacy_error']:.3e}
observation sign-flip error            {summary['population_checks']['observation_sign_flip_error']:.3e}
ACF difference h=1..128                {summary['population_checks']['acf_difference_h1_to_128']:.3e}
PSD difference max over 257 freq        {summary['population_checks']['psd_difference_257_frequencies']:.3e}
Bayes-risk difference h=1..128          {summary['population_checks']['risk_difference_h1_to_128']:.3e}
one-step Bayes MSE forward/reverse      {summary['population_checks']['risk_h1_forward']:.6f} / {summary['population_checks']['risk_h1_reverse']:.6f}
full train transition excess risk mean  {predictor_summary['forward_reverse_full_transition_population_excess_pct_mean']:.4f}% over oracle
origin-only transition excess mean      {predictor_summary['forward_reverse_origin_transition_population_excess_pct_mean']:.4f}% over oracle
quadratic latest-value excess mean      {predictor_summary['forward_reverse_quadratic_population_excess_pct_mean']:.4f}% over oracle
```

The full train transition estimator used all adjacent transitions in the 64 train paths: `64 * 511 = 32704` transitions per seed and condition. The origin-only transition estimator used only the planned forecasting origins: `64 * 16 = 1024` transitions. Both are train-only.
These excess-risk percentages use the oracle risk as denominator, so they are not the same denominator as the later 2% model-improvement screen.

## Interpretation

The control objective partly works: forward and reverse have the same second-order statistics and the same Bayes risk. That is useful as a mathematical diagnostic.

The construct fails for the intended paper claim. Because reverse is just the reflected forward chain and reflection also flips the observed values, a model gap could be about sign handling or a simple nonlinear readout, not about a special temporal relation that needs an internal adapter. Also, this is first-order Markov: once the latest state is observed, older left/right context has no additional information for the one-step target. That makes it a weak test for a current-past interaction adapter.

The old `covariate-trust-pilot` record adds only a historical warning. It found no M5 or Favorita series with `|rho_interval| >= 0.8`; that does not prove the new direction is absent in real data, and it is not used as a stop reason here. It only means a revised design still needs a train-only real-data availability check before a paper-scale claim.

## Figure

![Phase 0 data gate](02_data_gate.png)

## Outputs

```text
results/peft_dependence_screen_v1/data_gate/summary.json
results/peft_dependence_screen_v1/data_gate/condition_metrics.csv
results/peft_dependence_screen_v1/data_gate/predictor_metrics.csv
results/peft_dependence_screen_v1/data_gate/02_data_gate.png
```

## Recommended Next Design

Do not run the 72-fit GPU package on this DGP. The next viable design should remove the sign-relabel confound and force a simple-control contest before any adapter is proposed. A better synthetic family would need at least these properties:

1. Same marginal distribution and matched second-order statistics under the intended contrast.
2. Same latest value can imply different next-step distributions depending on ordered history, so a last-state transition table is insufficient.
3. A prespecified raw-history predictor, Markov/hidden-state estimator, and same-capacity head are included from the first run.
4. Real train-only descriptors show that the relation exists in candidate datasets before model scores are opened.

"""
    (OUT / "data_gate_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    condition_rows: list[dict[str, object]] = []
    predictor_metric_rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for seed in SEEDS:
            data = generate_condition(condition, condition_index, seed)
            condition_rows.extend(condition_metric_rows(condition, seed, data))
            predictor_metric_rows.extend(predictor_rows(condition, seed, data))

    pop = population_audit()
    predictor_summary = summarize_predictors(predictor_metric_rows)
    old_real_support = {
        "source": str(OLD_RHO_DOC),
        "readable": OLD_RHO_DOC.exists(),
        "recorded_interval_support": "M5 and Favorita had zero series with |rho_I| >= 0.8 in the referenced train descriptors.",
        "recorded_magnitude_support": "Magnitude persistence had thin support; this does not rescue the interval/direction contrast.",
    }
    summary = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "source_sha256": sha256_file(Path(__file__)),
        "plan_sha256_at_execution": sha256_file(PLAN),
        "independent_population_audit_json": str(ROOT / "results/peft_dependence_screen_v1/independent_gate_audit/audit.json"),
        "domain": "time-series forecasting PEFT design gate",
        "downstream_decision": "whether to proceed to GPU Phase 1 and method development",
        "quality_bar": "publication-oriented pretraining/fine-tuning design screen",
        "splits": SPLITS,
        "length": LENGTH,
        "context": CONTEXT,
        "origins": origins().tolist(),
        "seeds": list(SEEDS),
        "rng_streams": "SeedSequence([seed, condition_index]).spawn(train,val,devtest); attempt01_shared_rng preserved separately.",
        "population_checks": pop,
        "predictor_summary": predictor_summary,
        "old_real_data_support": old_real_support,
        "overall_decision": "STOP_BEFORE_GPU_PHASE1",
        "reason_codes": [
            "FORWARD_REVERSE_IS_EXACT_SIGN_RELABELING",
            "LATEST_STATE_SUFFICIENT_FOR_ONE_STEP_ORACLE",
            "TRAIN_ONLY_TRANSITION_CONTROL_NEAR_ORACLE",
        ],
        "historical_warning_not_stop_reason": "Old M5/Favorita interval-dependence support was weak in covariate-trust-pilot records, but this was not used as a failure gate.",
        "gpu_started": False,
        "lora_or_adapter_result": "NOT_MEASURED",
    }

    write_csv(OUT / "condition_metrics.csv", condition_rows)
    write_csv(OUT / "predictor_metrics.csv", predictor_metric_rows)
    make_figure(pop, predictor_summary, predictor_metric_rows)
    write_report(summary, predictor_summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
