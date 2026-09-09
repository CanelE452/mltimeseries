"""Post-evaluation matched-configuration decomposition; does not change study19."""

import json
from pathlib import Path
import time

import numpy as np

from .data import load_series
from .model import ridge
from .run import ROOT, RAW, OUT, RUN, CausalPanel, month, sha, write_json, verify_contract


def main():
    started = time.perf_counter()
    verify_contract()
    plan = ROOT / "_docs/notes/tsfm_topics/19c_revision_posthoc_diagnostic_20260908.md"
    write_json(RUN / "posthoc_contract.json", {
        "scope": "E-exposed matched-config diagnostic; not preregistered primary results",
        "plan_sha256": sha(plan), "source_sha256": sha(Path(__file__)),
        "original_results_sha256": sha(OUT / "cpu_results.json"),
        "original_cpu_contract_sha256": sha(RUN / "cpu_contract.json"),
    })
    selected = json.loads((RUN / "validation_selection.json").read_text(encoding="utf-8"))["selected"]
    result = {"posthoc": True, "new_selection": False, "series": {}}
    for name in ("PAYEMS", "INDPRO"):
        panel = CausalPanel(load_series(RAW, name))
        events = range(month(2020, 1), month(2024, 12) + 1)
        records = {t: panel.origin(t) for t in events}
        truth = panel.mature[np.isin(panel.events, list(events))]
        configs = sorted({(selected[name][arm]["lam"], selected[name][arm]["window"]) for arm in ("FIRST_FIXED_X", "REVISED_FIXED_X")})
        entries = []
        for lam, window in configs:
            predictions = {arm: [] for arm in ("FIRST_FIXED_X", "REVISED_FIXED_X", "FIRST_BIAS")}
            for t, record in records.items():
                for arm in predictions:
                    x, y, ids = record["data"][arm]
                    prediction = ridge(x, y, record["x"], lam, window, t, ids)["prediction"]
                    predictions[arm].append(prediction + (record["first_bias"] if arm == "FIRST_BIAS" else 0.0))
            metrics = {arm: {"mse": float(np.mean((np.asarray(p) - truth) ** 2)),
                             "mae": float(np.mean(np.abs(np.asarray(p) - truth)))} for arm, p in predictions.items()}
            entries.append({"lam": lam, "window": window, "metrics": metrics, "predictions": predictions,
                            "same_config_revision_improvement_percent": 100 * (1 - metrics["REVISED_FIXED_X"]["mse"] / metrics["FIRST_FIXED_X"]["mse"]),
                            "revision_max_prediction_difference": float(np.max(np.abs(np.asarray(predictions["REVISED_FIXED_X"]) - predictions["FIRST_FIXED_X"])))})
        result["series"][name] = entries
    result["wall_seconds"] = time.perf_counter() - started
    write_json(OUT / "posthoc_matched_config.json", result)
    verify_contract()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
