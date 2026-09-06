"""FlowState reference, scored on the same frozen keys as the core arms.

FlowState is zero-shot pretrained; R, M and O are trained on the target data.  The
numbers can sit in one table, but the information conditions differ and nothing
here calls it a matched training-budget contest.

The one argument that needs interpreting is `scale_factor`, and the model card
defines it outright:

    "Base Seasonality / N = 24 / 96 = 0.25", 24 being the base seasonality used in
    pretraining and N the number of steps in the data's daily cycle.

So a 10-minute grid has N = 144 and scale_factor = 24/144, and a report interval of
r base bins has N = 144/r and scale_factor = r/6.  Nothing is guessed.

The official package pins ``transformers<4.51`` and ``python<3.13``, which this
project's interpreter does not satisfy, so the run is split in two:

    python -m experiments.oa_resolution_pilot_v1.references --export
    <flowstate venv python> experiments/oa_resolution_pilot_v1/references.py --run

The export step writes the frozen inputs to runs/; the run step reads that file and
touches nothing but numpy, torch and tsfm_public.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.oa_resolution_pilot_v1.model import hourly_mean  # noqa: E402
from experiments.oa_resolution_pilot_v1.operators import (  # noqa: E402
    FORECAST_BINS,
    HISTORY_BINS,
    reconstruction_operator,
)

RESULTS = REPO / "results" / "oa_resolution_pilot_v1"
RUNS = REPO / "runs" / "oa_resolution_pilot_v1"
CACHE = RUNS / "reference_inputs.npz"

FLOWSTATE_REPO = "ibm-research/flowstate"
FLOWSTATE_REVISION = "r1.1"
BASE_SEASONALITY = 24.0
BINS_PER_DAY = 144.0

EVAL_R = (2, 3, 4, 6, 8, 12)
OPS = ("END_BIN", "INTERVAL_MEAN")
ROLE = {2: "SEEN", 4: "SEEN", 8: "SEEN", 3: "UNSEEN_INTERPOLATION",
        6: "UNSEEN_INTERPOLATION", 12: "UNSEEN_EXTRAPOLATION"}

# The native-rate variant returns predictions on its own coarse grid.  Only where
# that grid tiles the hour can the 60-minute metric be formed by the same block
# mean the core arms use; anywhere else it would take an upsampler with no official
# definition, so the condition is reported as not evaluated instead of invented.
NATIVE_HOURLY_TILING = {2: 3, 3: 2, 6: 1}


def scale_factor_for(report_interval_bins: int) -> float:
    return BASE_SEASONALITY / (BINS_PER_DAY / report_interval_bins)


def observe(history: np.ndarray, r: int, op: str) -> np.ndarray:
    n = HISTORY_BINS // r
    if op == "INTERVAL_MEAN":
        return history.reshape(history.shape[0], n, r).mean(axis=-1)
    if op == "END_BIN":
        return history[:, r - 1 :: r]
    raise ValueError(op)


# ------------------------------------------------------------------------ export


def export() -> Path:
    """Freeze the exact test-key windows the core arms were scored on."""
    from experiments.oa_resolution_pilot_v1 import data as D
    from experiments.oa_resolution_pilot_v1 import evaluate as E

    spec = json.loads((RESULTS / "execution_spec.json").read_text(encoding="utf-8"))
    payload: dict[str, np.ndarray] = {}
    meta = {"lambda_selected": spec["lambda_selected"], "datasets": {}}
    for ds, p in spec["periods"].items():
        g = D.load_dataset(ds, p["start"], p["end"])
        sp = D.chronological_split(g.n_bins)
        mu, sd = D.train_scaling(g, sp)
        ch, org = E.frozen_keys(g, sp)["test"]
        z = (g.values - mu) / sd
        idx_h = org[:, None] + np.arange(-HISTORY_BINS, 0)[None, :]
        idx_t = org[:, None] + np.arange(0, FORECAST_BINS)[None, :]
        payload[f"{ds}_history"] = z[idx_h, ch[:, None]].astype(np.float32)
        payload[f"{ds}_target"] = z[idx_t, ch[:, None]].astype(np.float32)
        meta["datasets"][ds] = {"n_keys": int(org.size), "channels": list(g.channels)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    (RUNS / "reference_inputs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)", flush=True)
    return CACHE


# --------------------------------------------------------------------------- run


def load_flowstate(device):
    from tsfm_public import FlowStateForPrediction

    model = FlowStateForPrediction.from_pretrained(
        FLOWSTATE_REPO, revision=FLOWSTATE_REVISION
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def forecast(model, context: torch.Tensor, scale_factor: float, prediction_length: int) -> torch.Tensor:
    """context: (batch, steps) -> (batch, prediction_length), the median quantile."""
    ts = context.T.unsqueeze(-1).contiguous()                  # (steps, batch, 1)
    out = model(ts, scale_factor=scale_factor,
                prediction_length=prediction_length, batch_first=False)
    y = getattr(out, "prediction_outputs", out)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = torch.as_tensor(y).float()
    if y.dim() == 4:            # documented shape [batch, quantiles, length, channels]
        y = y[:, y.shape[1] // 2, :, 0]
    elif y.dim() == 3:
        y = y[:, :, 0]
    return y


@torch.no_grad()
def resampled_variant(model, hist: torch.Tensor, targ: torch.Tensor, lam: float,
                      device, chunk: int) -> list[dict]:
    """Reference A: R's operator-aware reconstruction to the 10-minute base grid,
    then FlowState at the 10-minute scale factor."""
    sf = scale_factor_for(1)
    rows = []
    for op in OPS:
        for r in EVAL_R:
            P = torch.tensor(reconstruction_operator(r, op, lam), dtype=torch.float32, device=device)
            se10 = se60 = 0.0
            n = 0
            for s in range(0, hist.shape[0], chunk):
                h = hist[s : s + chunk].to(device)
                t = targ[s : s + chunk].to(device)
                v = torch.tensor(observe(h.cpu().numpy(), r, op), device=device)
                pred = forecast(model, v @ P.T, sf, FORECAST_BINS)
                se10 += float(((pred - t) ** 2).mean(-1).sum())
                se60 += float(((hourly_mean(pred) - hourly_mean(t)) ** 2).mean(-1).sum())
                n += h.shape[0]
            rows.append({"operation": op, "r": r, "role": ROLE[r], "n_keys": n,
                         "mse10": se10 / n, "mse60": se60 / n,
                         "primary": 0.5 * se10 / n + 0.5 * se60 / n})
            print(f"      resampled {op} r={r}: primary={rows[-1]['primary']:.5f}", flush=True)
    return rows


@torch.no_grad()
def native_variant(model, hist: torch.Tensor, targ: torch.Tensor, device, chunk: int):
    """Reference B: the coarse observed sequence at its own documented scale factor,
    scored on the 60-minute metric where the coarse grid tiles the hour."""
    rows, skipped = [], []
    for op in OPS:
        for r in EVAL_R:
            if r not in NATIVE_HOURLY_TILING:
                skipped.append({"operation": op, "r": r, "reason":
                                "the native output grid does not tile the hour, so the 60-minute "
                                "metric would need an upsampler with no official definition"})
                continue
            per_hour = NATIVE_HOURLY_TILING[r]
            steps = FORECAST_BINS // r
            sf = scale_factor_for(r)
            se60 = 0.0
            n = 0
            for s in range(0, hist.shape[0], chunk):
                h = hist[s : s + chunk].to(device)
                t = targ[s : s + chunk].to(device)
                v = torch.tensor(observe(h.cpu().numpy(), r, op), device=device)
                pred = forecast(model, v, sf, steps)
                p60 = pred.reshape(pred.shape[0], steps // per_hour, per_hour).mean(-1)
                se60 += float(((p60 - hourly_mean(t)) ** 2).mean(-1).sum())
                n += h.shape[0]
            rows.append({"operation": op, "r": r, "role": ROLE[r], "n_keys": n,
                         "scale_factor": sf, "mse60": se60 / n})
            print(f"      native    {op} r={r}: mse60={rows[-1]['mse60']:.5f}", flush=True)
    return rows, skipped


def run(chunk: int = 128) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((RUNS / "reference_inputs.json").read_text(encoding="utf-8"))
    cache = np.load(CACHE)

    out = {
        "model": FLOWSTATE_REPO, "revision": FLOWSTATE_REVISION, "license": "Apache-2.0",
        "source": "https://huggingface.co/ibm-research/flowstate",
        "package": "tsfm_public (ibm-granite/granite-tsfm, gift-flowstate branch)",
        "scale_factor_definition": "base seasonality 24 divided by the number of steps in the "
                                   "data's daily cycle, exactly as the model card states",
        "information_condition": "zero-shot pretrained. R / M / O are trained on the target data. "
                                 "Not a matched training budget, and not called one.",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "note": "the official package pins python<3.13 and transformers<4.51, so it "
                                "runs in a separate interpreter from the core experiment"},
        "datasets": {},
    }
    t0 = time.time()
    try:
        model = load_flowstate(device)
    except Exception as exc:
        out["status"] = "BLOCKED_REFERENCE_CONTRACT"
        out["error"] = f"{type(exc).__name__}: {exc}"
        (RESULTS / "flowstate_reference.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("FlowState BLOCKED:", out["error"], flush=True)
        return 0

    out["status"] = "OK"
    out["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    for ds, info in meta["datasets"].items():
        print(f"  FlowState on {ds} ({info['n_keys']} test keys)", flush=True)
        hist = torch.tensor(cache[f"{ds}_history"])
        targ = torch.tensor(cache[f"{ds}_target"])
        lam = float(meta["lambda_selected"][ds])
        cells = resampled_variant(model, hist, targ, lam, device, chunk)
        native, skipped = native_variant(model, hist, targ, device, chunk)
        out["datasets"][ds] = {
            "n_test_keys": info["n_keys"], "channels": info["channels"],
            "resampled": {"variant": "FLOWSTATE_RESAMPLED",
                          "scale_factor": scale_factor_for(1),
                          "reconstruction_lambda": lam, "cells": cells},
            "native_rate": {"variant": "FLOWSTATE_NATIVE_RATE", "cells": native,
                            "not_evaluated": skipped},
        }
    out["wall_seconds"] = round(time.time() - t0, 1)
    (RESULTS / "flowstate_reference.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("FlowState reference written", flush=True)
    return 0


def optional_references() -> None:
    """Section 39: Chronos-2 only if it is already usable here; TiRex-2 is not a
    required reference and no environment time is spent on it."""
    try:
        import chronos  # noqa: F401
        detail, status = "importable in this interpreter", "OK"
    except Exception as exc:
        detail, status = f"{type(exc).__name__}: {exc}", "CHRONOS_REFERENCE_SKIPPED_TIMEBOX"
    payload = {
        "chronos_2": {"status": status, "detail": detail,
                      "rule": "run only if already installed and usable within 15 minutes on the "
                              "same frozen key subgrid; no new dependency debugging"},
        "tirex_2": {"status": "NOT_REQUIRED",
                    "detail": "not a required reference for this pilot; no environment time spent"},
    }
    (RESULTS / "optional_references.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if "--export" in sys.argv:
        export()
        optional_references()
    elif "--run" in sys.argv:
        raise SystemExit(run())
    else:
        print(__doc__)
