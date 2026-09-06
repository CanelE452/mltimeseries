"""Apply the pre-registered screens (instruction sections 26 and 37).

The thresholds are read from the instruction and hard-coded here; nothing in this
file looks at the numbers before deciding what the thresholds are. The verdict is
an investment decision about the next step, not a claim about publishability, and
a GO never means the method is novel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir

PATH_VALUE_RI = 2.0
ACCURACY_MC_RI = 1.0
EFFICIENCY_CRPS_RATIO = 1.005
EFFICIENCY_LATENCY_RATIO = 0.5
RMSE_TOLERANCE = 1.0  # P may not be worse than D or S by more than 1%
NO_VALUE_RI = 0.5


def load(tag: str):
    boot = json.loads((out_dir(tag) / "bootstrap_primary.json").read_text())
    contrasts = {c["contrast"]: c for c in boot["contrasts"]}
    metrics = pd.read_csv(out_dir(tag) / "metrics.csv")
    metrics = metrics[metrics["split"] == "test"]
    seeds = pd.read_csv(out_dir(tag) / "seed_effects.csv")
    latency = pd.read_csv(out_dir(tag) / "latency.csv")
    return contrasts, metrics, seeds, latency


def seed_direction(seeds: pd.DataFrame, contrast: str) -> dict:
    sub = seeds[seeds["contrast"] == contrast]
    return {"n_seeds": int(len(sub)), "n_positive": int((sub["ri_percent"] > 0).sum())}


def main(tag: str = "full") -> int:
    contrasts, metrics, seeds, latency = load(tag)
    mean_metrics = metrics.groupby("arm").mean(numeric_only=True)

    checks = {}
    for name in ("P over D", "P over S"):
        c = contrasts[name]
        d = seed_direction(seeds, name)
        checks[name] = {
            "ri_percent": c["ri_percent"],
            "ci_lower": c["ci_lower"],
            "ci_upper": c["ci_upper"],
            "meets_ri_threshold": c["ri_percent"] >= PATH_VALUE_RI,
            "ci_lower_above_zero": c["ci_lower"] > 0,
            "seeds_positive": d,
            "seed_majority_positive": d["n_positive"] >= 2,
        }

    rmse_p = float(mean_metrics.loc["P", "mean_rmse_mw"])
    rmse_guard = {
        arm: {
            "rmse_mw": float(mean_metrics.loc[arm, "mean_rmse_mw"]),
            "p_worse_by_percent": 100.0 * (rmse_p - float(mean_metrics.loc[arm, "mean_rmse_mw"]))
            / float(mean_metrics.loc[arm, "mean_rmse_mw"]),
        }
        for arm in ("D", "S")
    }
    rmse_ok = all(v["p_worse_by_percent"] <= RMSE_TOLERANCE for v in rmse_guard.values())

    path_value_go = (
        all(
            v["meets_ri_threshold"] and v["ci_lower_above_zero"] and v["seed_majority_positive"]
            for v in checks.values()
        )
        and rmse_ok
    )

    mc = contrasts.get("P over MC")
    accuracy_go = bool(
        path_value_go and mc and mc["ri_percent"] >= ACCURACY_MC_RI and mc["ci_lower"] > 0
    )

    lat = latency[latency["batch"] == 1].set_index("arm")["adapter_only_ms"]
    crps_ratio = float(mean_metrics.loc["P", "macro_scaled_crps"]) / float(
        mean_metrics.loc["MC", "macro_scaled_crps"]
    )
    latency_ratio = float(lat["P"]) / float(lat["MC"]) if "MC" in lat and "P" in lat else None
    efficiency_go = bool(
        path_value_go
        and crps_ratio <= EFFICIENCY_CRPS_RATIO
        and latency_ratio is not None
        and latency_ratio <= EFFICIENCY_LATENCY_RATIO
    )

    gains = [checks[n]["ri_percent"] for n in checks]
    no_value = all(g < NO_VALUE_RI for g in gains) and all(
        checks[n]["ci_upper"] < PATH_VALUE_RI for n in checks
    )

    if path_value_go and accuracy_go:
        token = "PATH_METHOD_EXPAND"
    elif path_value_go and efficiency_go:
        token = "EFFICIENCY_ONLY_EXPAND"
    elif no_value:
        token = "NO_INCREMENTAL_PATH_VALUE"
    else:
        token = "INCONCLUSIVE_DO_NOT_SCALE_YET"

    verdict = {
        "final_token": token,
        "screens": {
            "PATH_VALUE_GO": path_value_go,
            "ACCURACY_SCREEN_GO": accuracy_go,
            "EFFICIENCY_SCREEN_GO": efficiency_go,
            "NO_INCREMENTAL_PATH_VALUE": no_value,
        },
        "thresholds": {
            "path_value_ri_percent": PATH_VALUE_RI,
            "accuracy_mc_ri_percent": ACCURACY_MC_RI,
            "efficiency_crps_ratio": EFFICIENCY_CRPS_RATIO,
            "efficiency_latency_ratio": EFFICIENCY_LATENCY_RATIO,
            "rmse_tolerance_percent": RMSE_TOLERANCE,
            "no_value_ri_percent": NO_VALUE_RI,
        },
        "primary_contrasts": checks,
        "rmse_guard": {"passed": rmse_ok, "detail": rmse_guard},
        "mc_contrast": mc,
        "efficiency_detail": {
            "p_over_mc_crps_ratio": crps_ratio,
            "p_over_mc_latency_ratio_batch1_adapter_only": latency_ratio,
        },
        "secondary_contrasts": {
            k: contrasts[k] for k in ("P over P_BROKEN", "S over M", "M over H") if k in contrasts
        },
        "reading": (
            "This is a decision about whether to invest in the next step, not an acceptance "
            "bar. A GO says the path representation earned its keep on this pilot; it says "
            "nothing about novelty, and nothing causal about why."
        ),
    }
    (out_dir(tag) / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
