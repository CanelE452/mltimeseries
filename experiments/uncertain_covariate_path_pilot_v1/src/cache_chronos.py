"""Cache frozen Chronos-2-synth quantile forecasts for every panel example.

The backbone sees the target history and nothing else: no weather, no future
value, no calendar. The 144 half-hourly predicted steps are matched to the
twelve ENS lead timestamps by datetime, never by an assumed index (test A19).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data_external/ucp_path_pilot_v1/processed"
CACHE = ROOT / "data_external/ucp_path_pilot_v1/chronos_cache"
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"

MODEL_ID = "autogluon/chronos-2-synth"
REVISION = "3607918a9fd027d5c465d8213e46b98e2c041cea"
PREDICTION_STEPS = 144  # 72 h of half-hourly steps
STEP = pd.Timedelta(minutes=30)
LEADS_H = tuple(range(6, 73, 6))
BATCH = 64


def lead_indices(origin: pd.Timestamp) -> list[int]:
    """Positions of the twelve lead timestamps inside the 144 predicted steps."""
    predicted = [origin + (j + 1) * STEP for j in range(PREDICTION_STEPS)]
    pos = {t: j for j, t in enumerate(predicted)}
    return [pos[origin + pd.Timedelta(hours=h)] for h in LEADS_H]


def load_pipeline():
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cuda", torch_dtype=torch.float32
    )
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    pipe.model.eval()
    return pipe


def cache_split(pipe, split: str) -> dict:
    panel = np.load(PROCESSED / f"panel_{split}.npz", allow_pickle=False)
    ids = panel["example_id"]
    out = CACHE / f"chronos_{split}.npz"
    if out.exists():
        cached = np.load(out, allow_pickle=False)
        if list(cached["example_id"]) == list(ids):
            print(f"[{split}] cache already matches {len(ids)} examples", flush=True)
            return {"n_examples": int(len(ids)), "reused": True}

    if len(ids) == 0:
        np.savez_compressed(out, example_id=ids, quantiles=np.zeros((0, len(LEADS_H), 13), np.float32),
                            quantile_levels=np.array(pipe.quantiles, np.float32))
        return {"n_examples": 0, "reused": False}

    history = panel["history"]
    origins = pd.to_datetime(panel["origin"])
    levels = list(pipe.quantiles)

    selected = np.zeros((len(ids), len(LEADS_H), len(levels)), np.float32)
    with torch.no_grad():
        for start in range(0, len(ids), BATCH):
            stop = min(start + BATCH, len(ids))
            # the pipeline moves batches itself through a pinning DataLoader, so keep CPU tensors
            ctx = torch.from_numpy(history[start:stop]).unsqueeze(1)
            q, _ = pipe.predict_quantiles(
                inputs=ctx, prediction_length=PREDICTION_STEPS, quantile_levels=levels
            )
            for i, item in enumerate(q):
                idx = lead_indices(pd.Timestamp(origins[start + i]))
                selected[start + i] = item[0, idx, :].float().cpu().numpy()
            if start % (BATCH * 20) == 0:
                print(f"[{split}] {stop}/{len(ids)}", flush=True)

    if not np.isfinite(selected).all():
        raise RuntimeError(f"{split}: Chronos produced non-finite quantiles")

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        example_id=ids,
        origin=panel["origin"],
        farm_idx=panel["farm_idx"],
        quantiles=selected,
        quantile_levels=np.array(levels, np.float32),
        leads_h=np.array(LEADS_H),
    )
    print(f"[{split}] cached {len(ids)} examples", flush=True)
    return {"n_examples": int(len(ids)), "reused": False}


def main(splits: list[str]) -> int:
    import chronos
    import transformers

    pipe = load_pipeline()
    stats = {s: cache_split(pipe, s) for s in splits}
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "frozen": True,
        "fine_tuned": False,
        "lora": False,
        "chronos_forecasting_version": chronos.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "quantile_levels": [float(x) for x in pipe.quantiles],
        "prediction_steps": PREDICTION_STEPS,
        "prediction_step_minutes": 30,
        "leads_h": list(LEADS_H),
        "lead_selection": "datetime join against origin + (j+1) * 30 min, not a positional guess",
        "input": "target history only; no weather, no future target, no calendar",
        "splits": stats,
    }
    (RESULTS / "chronos_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["train", "val", "test"]))
