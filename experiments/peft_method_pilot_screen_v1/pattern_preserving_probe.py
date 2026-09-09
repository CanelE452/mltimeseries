"""Read-only probe: does the pattern-preserving control recover Study17's damage?

Reads the frozen Study17 prediction archive and evaluation truth, builds the
`p0 + mean(p - p0)` controls that note 18 recorded as NOT executed, and scores
them with Study17's own sealed metric function. Writes nothing into the study.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/CODING/proj/mltimeseries")
sys.path.insert(0, str(ROOT))
from experiments.peft_coarse_supervision_v1.analysis import analyze_predictions  # noqa: E402

pred = np.load(ROOT / "runs/peft_coarse_supervision_v1/forecast/predictions.npz", allow_pickle=True)
truth_npz = np.load(ROOT / "runs/peft_coarse_supervision_v1/data/evaluation_truth.npz", allow_pickle=True)
inputs = np.load(ROOT / "runs/peft_coarse_supervision_v1/data/evaluation_inputs.npz", allow_pickle=True)

ARMS = ("F0", "PROFILE", "COARSE_LIFT", "FROZEN_HEAD", "ATTN_LORA")
base = {arm: np.asarray(pred[arm], dtype=np.float64) for arm in ARMS}
horizon = np.asarray(pred["horizon"]).astype(np.int64)
scale = np.asarray(inputs["scale"], dtype=np.float64)
site = np.asarray(pred["site"]).astype(str)
target_id = np.asarray(pred["target_id"]).astype(str)
month = np.asarray(pred["month"]).astype(str)
truth = np.asarray(truth_npz["target"], dtype=np.float64)

# Truth/prediction rows must describe the same cells before anything is compared.
for name, a, b in (("site", site, np.asarray(truth_npz["site"]).astype(str)),
                   ("target_id", target_id, np.asarray(truth_npz["target_id"]).astype(str)),
                   ("month", month, np.asarray(truth_npz["month"]).astype(str)),
                   ("horizon", horizon, np.asarray(truth_npz["horizon"]).astype(np.int64))):
    if not np.array_equal(a, b):
        raise AssertionError(f"Prediction and truth rows disagree on {name}")


def level_shift(source):
    """p0 pattern with the source arm's own horizon mean; forecast.py:67 form."""
    out = base["F0"].copy()
    for index, h in enumerate(horizon):
        out[index, :h] = base["F0"][index, :h] + source[index, :h].mean() - base["F0"][index, :h].mean()
    return out


controls = {
    "PATTERN_FROM_HEAD": level_shift(base["FROZEN_HEAD"]),
    "PATTERN_FROM_LORA": level_shift(base["ATTN_LORA"]),
    "PATTERN_FROM_TRUTH_ORACLE": level_shift(np.where(np.isfinite(truth), truth, base["F0"])),
}

report = {}

# Study17 fixed its simple-arm choice on validation monthly-mean MSE before opening E.
# A level-shift control inherits its source arm's monthly mean exactly, so it inherits
# that validation score too. Carrying the ranking here keeps the comparison honest.
selection = json.loads((ROOT / "runs/peft_coarse_supervision_v1/selection.json").read_text())
report["study17_validation_rule"] = {
    "metric": "normalised monthly-mean MSE on V, lower is better",
    "frozen_before_evaluation": selection["all_choices_frozen_before_evaluation"],
    "simple_validation_scores": selection["simple_validation_scores"],
    "selected_simple_policy": selection["simple_policy"],
    "note": "PATTERN_FROM_HEAD inherits FROZEN_HEAD's validation score, PATTERN_FROM_LORA inherits"
            " ATTN_LORA's. Under the frozen rule the winner is COARSE_LIFT, not either of them.",
}

reference = analyze_predictions(base, truth, horizon, scale, site, target_id, month, "COARSE_LIFT")
report["study17_reference"] = {
    arm: {s: reference["sites"][s]["arms"][arm]["mse"] for s in reference["sites"]} for arm in ARMS
}
report["study17_reference_pattern_mse"] = {
    arm: {s: reference["sites"][s]["arms"][arm]["pattern_mse"] for s in reference["sites"]} for arm in ARMS
}

for label, values in controls.items():
    swapped = dict(base)
    swapped["PROFILE"] = values          # PROFILE slot reused as the carrier; renamed in the report
    scored = analyze_predictions(swapped, truth, horizon, scale, site, target_id, month, "COARSE_LIFT")
    report[label] = {
        "mse": {s: scored["sites"][s]["arms"]["PROFILE"]["mse"] for s in scored["sites"]},
        "level_mse": {s: scored["sites"][s]["arms"]["PROFILE"]["level_mse"] for s in scored["sites"]},
        "pattern_mse": {s: scored["sites"][s]["arms"]["PROFILE"]["pattern_mse"] for s in scored["sites"]},
    }

# The identity below is why this probe adds no new observation beyond a regrouping:
# every evaluated hour is observed, so mse(control) = pattern_mse(F0) + level_mse(source).
report["decomposition_identity"] = {
    "statement": "mse(level_shift(source)) == pattern_mse(F0) + level_mse(source)",
    "max_abs_residual": max(
        abs(report[label]["mse"][s]
            - report["study17_reference_pattern_mse"]["F0"][s]
            - report[label]["level_mse"][s])
        for label in controls for s in report[label]["mse"]
    ),
}

print(json.dumps(report, indent=1))
