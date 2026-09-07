"""Turn failure-map conditions into gap candidates, then freeze their probe plan.

Section 17 forbids jumping from an observation to a cause, so each descriptor
carries at least two competing mechanism hypotheses and the one post-hoc story
that must not be assumed. Those are properties of the descriptor, written before
any result was read; what the data decides is which conditions become candidates
and how severe they are.

Section 18 requires the probe plan to be hashed before any probe runs, so that
the oracle cannot be chosen after seeing what would flatter it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths

MAX_CANDIDATES = 4

# Registered before results: per descriptor, competing explanations, the story
# that must not be assumed, and which probes are allowed to speak to it.
MECHANISMS = {
    "D1": {
        "condition": "horizon long relative to the context the model is given",
        "hypotheses": [
            "The model's effective receptive field is shorter than the seasonal structure the long horizon requires, so it extrapolates from too little context.",
            "The task itself becomes near-unpredictable at that horizon, and every estimator including a trained specialist degrades together.",
            "The probabilistic head widens too slowly with horizon, so the loss is a calibration failure rather than a point-forecast failure.",
        ],
        "must_not_assume": "that a long-horizon deficit means the model cannot use long context - context length has not been varied here.",
        "probes": ["P0", "P1", "P3", "P4", "P6"],
    },
    "D2": {
        "condition": "sampling frequency",
        "hypotheses": [
            "Pretraining corpora are unevenly dense across frequencies, so some regimes are simply under-represented.",
            "The patching or tokenisation stride interacts badly with a particular period length.",
            "High-frequency tasks in this pool differ in domain, so the effect is domain, not frequency.",
        ],
        "must_not_assume": "that a frequency effect is about frequency; in this pool frequency and domain are correlated.",
        "probes": ["P0", "P1", "P4", "P6"],
    },
    "D3": {
        "condition": "number of target variates",
        "hypotheses": [
            "Cross-series attention dilutes each variate's own history as the variate count grows.",
            "High-dimensional tasks here carry weaker per-series signal regardless of how they are modelled.",
            "The channel-independent treatment used in TRACK U discards exactly the dependence these tasks carry.",
        ],
        "must_not_assume": "that a dimensionality effect in TRACK U says anything about native multivariate modelling - TRACK M answers that.",
        "probes": ["P0", "P1", "P4", "P6"],
    },
    "D4": {
        "condition": "known-future covariates are available",
        "hypotheses": [
            "Tasks that ship known-future covariates are driven by exogenous effects that no target-only model can see.",
            "The covariate-carrying tasks in this pool are harder for unrelated reasons.",
        ],
        "must_not_assume": "that a deficit in TRACK U on covariate-carrying tasks is a covariate-utilisation failure; in TRACK U no model is given the covariates.",
        "probes": ["P0", "P1", "P4", "P5", "P6"],
    },
    "D5": {
        "condition": "past-only covariates are available",
        "hypotheses": [
            "The exogenous drivers these tasks carry are informative but unused in this information condition.",
            "These tasks are harder for reasons unrelated to their covariates.",
        ],
        "must_not_assume": "that unused covariates explain the deficit without an ablation that supplies them.",
        "probes": ["P0", "P1", "P4", "P6"],
    },
    "D6": {
        "condition": "zero-heavy or intermittent series",
        "hypotheses": [
            "A continuous-valued quantile head cannot place mass at exactly zero, so intermittency is systematically mispriced.",
            "Scaling by a seasonal error that is near zero makes the metric unstable rather than the forecast bad.",
            "Intermittent series here are short, so the deficit is a data-quantity effect.",
        ],
        "must_not_assume": "that a zero-heavy deficit is a modelling failure before checking the metric's own scaling behaviour on those tasks.",
        "probes": ["P0", "P1", "P3", "P4", "P6"],
    },
    "D7": {
        "condition": "missing observations in the visible history",
        "hypotheses": [
            "Models differ in how they impute or mask gaps, and some choices destroy the seasonal phase.",
            "Missingness co-occurs with regime changes, so the deficit is shift, not gaps.",
        ],
        "must_not_assume": "that a missingness effect is about the imputation rule without separating it from the shift descriptor.",
        "probes": ["P0", "P1", "P3", "P4", "P6"],
    },
    "D8": {
        "condition": "distribution shift within the visible history",
        "hypotheses": [
            "In-context normalisation is estimated over a window that spans the shift, biasing the level.",
            "The shift continues past the cutoff, so no amount of history fixes the level and only a calibration term can.",
            "Shifted tasks are simply noisier.",
        ],
        "must_not_assume": "that shift measured before the cutoff persists after it.",
        "probes": ["P0", "P1", "P3", "P4", "P6"],
    },
    "D9": {
        "condition": "strength of the seasonal component",
        "hypotheses": [
            "Weakly seasonal series give the model little to lock onto, so it defaults to a poor persistence forecast.",
            "The naive denominator is unusually strong when seasonality is strong, which flatters or punishes the ratio rather than the forecast.",
        ],
        "must_not_assume": "that a ratio moving with seasonal strength is a model effect; the denominator moves too.",
        "probes": ["P0", "P1", "P4", "P6"],
    },
}

PROBE_NAMES = {
    "P0": "P0_cross_model_origin_oracle",
    "P1": "P1_validation_selected_model",
    "P3": "P3_affine_calibration",
    "P4": "P4_supervised_specialist",
    "P5": "P5_covariate_ablation",
    "P6": "P6_quantile_average_ensemble",
}


def build_candidates() -> list[dict]:
    gate = pd.read_csv(paths.RESULTS / "failure_candidates_gate.csv")
    failure_map = pd.read_csv(paths.RESULTS / "failure_map.csv")
    results = pd.read_csv(paths.RESULTS / "benchmark_results.csv")
    metadata = pd.read_csv(paths.RESULTS / "task_metadata.csv")

    passing = gate[gate.passes_failure_gate].sort_values(
        ["median_condition_gap_pct", "median_regret_pct"], ascending=False
    )
    candidates = []
    for rank, (_, row) in enumerate(passing.head(MAX_CANDIDATES).iterrows(), start=1):
        descriptor = row.descriptor
        column = [c for c in metadata.columns if c.startswith(descriptor + "_")][0]
        tasks = sorted(
            metadata[(metadata.split == "discovery") & (metadata[column] == row.bucket)].task_uid
        )
        bucket_rows = failure_map[
            (failure_map.descriptor == descriptor)
            & (failure_map.bucket == row.bucket)
            & (failure_map.model_role == "primary")
        ]
        template = MECHANISMS[descriptor]
        candidates.append(
            {
                "candidate_id": f"C{rank}_{descriptor}_{row.bucket}",
                "descriptor": descriptor,
                "bucket": row.bucket,
                "observed_condition": f"{template['condition']} (bucket '{row.bucket}')",
                "affected_models": json.loads(row.affected_models),
                "affected_families": json.loads(row.affected_families),
                "affected_tasks": tasks,
                "n_affected_tasks": len(tasks),
                "severity": {
                    "median_condition_gap_pct": float(row.median_condition_gap_pct),
                    "median_regret_pct": float(row.median_regret_pct),
                    "per_model_bucket_median_relative": dict(
                        zip(bucket_rows.model, bucket_rows.median_relative_to_naive.round(4))
                    ),
                },
                "mechanism_hypotheses": template["hypotheses"],
                "post_hoc_story_that_must_not_be_assumed": template["must_not_assume"],
                "what_breaks_without_a_new_method": (
                    "If no method addresses this condition, forecasts on these tasks stay at the "
                    "measured deficit against the best available deployable estimator, and the "
                    "condition keeps costing the amount recorded in `severity`."
                ),
                "closest_known_literature_area": _literature_area(descriptor),
                "planned_probes": [PROBE_NAMES[p] for p in template["probes"] if p in PROBE_NAMES],
            }
        )
    return candidates


def _literature_area(descriptor: str) -> str:
    return {
        "D1": "long-horizon forecasting and context utilisation in pretrained time-series models",
        "D2": "frequency and resolution transfer in time-series foundation models",
        "D3": "multivariate and channel-dependence modelling",
        "D4": "covariate-informed and exogenous-variable forecasting",
        "D5": "past-covariate conditioning",
        "D6": "intermittent and count-valued demand forecasting",
        "D7": "irregular sampling and missing-data handling",
        "D8": "distribution shift, normalisation and test-time adaptation",
        "D9": "seasonality modelling and decomposition",
    }[descriptor]


def freeze(candidates: list[dict]) -> dict:
    spec = {
        "experiment": "TSFM-BENCHMARK-GAP-DISCOVERY-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "track": "U",
        "split_used_for_discovery": "discovery",
        "confirmation_untouched": True,
        "max_candidates": MAX_CANDIDATES,
        "headroom_definitions": {
            "baseline_loss": (
                "one primary foundation model, chosen once for the whole affected task set by its "
                "mean aggregate and applied unchanged to every task in it; not a per-task winner"
            ),
            "oracle_loss": "P0 per-origin minimum across the primary foundation models (uses evaluation labels; diagnostic only)",
            "simple_loss": "best deployable simple fix among P1, P3, P4 and P6",
            "H_oracle": "100 * (baseline_loss - oracle_loss) / baseline_loss",
            "R_simple": "(baseline_loss - simple_loss) / (baseline_loss - oracle_loss)",
            "H_residual": "100 * (simple_loss - oracle_loss) / baseline_loss",
        },
        "deployable_probe_window_rule": (
            "Deployable probes may use windows 0..k-1 to act on window k and are scored on "
            "windows 1..K-1; every quantity they are compared against is restricted to the same "
            "windows."
        ),
        "candidates": candidates,
    }
    return spec


def main() -> None:
    candidates = build_candidates()
    spec = freeze(candidates)
    payload = json.dumps(spec, indent=2, sort_keys=True, default=str)
    (paths.RESULTS / "candidate_probe_spec.json").write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (paths.RESULTS / "candidate_probe_spec.sha256").write_text(
        f"{digest}  candidate_probe_spec.json\n", encoding="utf-8"
    )
    (paths.RESULTS / "failure_candidates.json").write_text(
        json.dumps(candidates, indent=2, default=str), encoding="utf-8"
    )
    print(f"candidates: {len(candidates)}")
    for candidate in candidates:
        print(
            f"  {candidate['candidate_id']}: {candidate['observed_condition']} | "
            f"tasks={candidate['n_affected_tasks']} families={len(candidate['affected_families'])} "
            f"gap={candidate['severity']['median_condition_gap_pct']:.1f}% "
            f"regret={candidate['severity']['median_regret_pct']:.1f}%"
        )
    print(f"\nprobe spec sha256 {digest}")
    if not candidates:
        print("NO_STRONG_GAP_FOUND: no condition passed the failure gate, no probes to run")


if __name__ == "__main__":
    main()
