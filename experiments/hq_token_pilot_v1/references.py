"""Sections 16 and 17: the dynamic-tokenization literature table and the native
foundation-model reference comparison.

Run after the core. Kept out of run.py because both stages are best-effort, are bounded
by their own wall-clock budgets, and must be able to fail without touching the verdict.

Neither stage can change the core result. A missing reference is recorded as missing;
it is never read as evidence for or against the hypothesis.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import data as D
from . import evaluate as E
from . import model as M
from .run import DATASETS, HORIZONS, PRIMARY_B, RESULTS, SEED0, fit_dir, log

# section 17: a declared subgrid, fixed before any reference number exists.
# Electricity has 321 channels and the CPU-only reference cannot cover them in budget,
# so a seed-fixed channel subsample is used and the core arms are re-scored on exactly
# the same keys, giving a like-for-like row rather than a full-grid comparison.
SUBGRID_CHANNEL_CAP = {"ETTm2": None, "weather": None, "electricity": 16}
SUBGRID_SEED = 2026090681


def subgrid(dataset: D.Dataset):
    origins = dataset.eval_origins("test")
    cap = SUBGRID_CHANNEL_CAP[dataset.dataset_id]
    if cap is None or cap >= dataset.n_channels:
        channels = np.arange(dataset.n_channels)
    else:
        rng = np.random.default_rng([SUBGRID_SEED, dataset.n_channels])
        channels = np.sort(rng.choice(dataset.n_channels, size=cap, replace=False))
    o = np.repeat(origins, channels.shape[0])
    c = np.tile(channels, origins.shape[0])
    return o, c


def raw_history(dataset: D.Dataset, origins, channels):
    """Unstandardized history. Foundation models do their own scaling internally."""
    idx = origins[:, None] + np.arange(-D.L, 0)[None, :]
    return dataset.values[idx, channels[:, None]].astype(np.float32)


def score_against_core_grid(dataset, origins, channels, horizon, pred_raw):
    """Put a raw-unit forecast into the core train-standardized space and score it."""
    mean = dataset.mean[channels][:, None]
    std = dataset.std[channels][:, None]
    pred = (pred_raw[:, :horizon] - mean) / std
    y = dataset.targets(origins, channels)[:, :horizon]
    diff = pred - y
    n = float(horizon) * origins.shape[0]
    return {"mse": float((diff**2).sum() / n), "mae": float(np.abs(diff).sum() / n)}


# ---- section 16 ------------------------------------------------------------


def literature_table(searched: dict) -> list:
    """Fields required by section 16, filled only from what was actually checked."""
    rows = []
    for key, found in searched.items():
        rows.append(
            {
                "method": key,
                "official_code_found": found["official_code_found"],
                "license": found.get("license"),
                "how_checked": found["how_checked"],
                "multi_horizon_and_same_token_budget_applicable": "NOT_ASSESSED_WITHOUT_CODE"
                if not found["official_code_found"]
                else found.get("applicable"),
                "original_purpose_and_implementation_preserved": "NOT_APPLICABLE"
                if not found["official_code_found"]
                else found.get("preserved"),
                "compared_under_the_same_contract": False,
                "status": "MISSING_OFFICIAL_BASELINE"
                if not found["official_code_found"]
                else "CODE_FOUND_NOT_INTEGRATED",
            }
        )
    return rows


# ---- section 17 ------------------------------------------------------------


def run_chronos(budget_s: float, device: str):
    t0 = time.time()
    out = {"model": "Chronos-2", "weight": "amazon/chronos-2"}
    try:
        from chronos import Chronos2Pipeline
    except Exception as e:
        out["status"] = f"BLOCKED_ENV: import failed: {e}"
        return out
    try:
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    except Exception as e:
        out["status"] = f"BLOCKED_ENV: load failed: {e}"
        return out
    out["setup_seconds"] = time.time() - t0

    quantiles = list(getattr(pipe, "quantiles", []))
    if 0.5 not in quantiles:
        out["status"] = "BLOCKED_CONTRACT: no q0.5 in pipeline.quantiles"
        out["quantiles"] = quantiles
        return out
    q50 = quantiles.index(0.5)
    out["quantiles"] = quantiles
    out["point_forecast_type"] = "median"
    out["point_forecast_note"] = (
        "The pipeline exposes quantiles, and 0.5 is the median. The core arms are "
        "MSE-trained point forecasts, whose optimum is the mean. Scoring a median with "
        "MSE is a handicap of unknown size, so this row is a reference, not a contest."
    )

    per = {}
    for name in DATASETS:
        ds = D.Dataset(name)
        o, c = subgrid(ds)
        hist = raw_history(ds, o, c)
        for h in HORIZONS:
            if time.time() - t0 > budget_s:
                out["status"] = "PARTIAL_BUDGET"
                out["per_dataset"] = per
                return out
            preds = np.empty((o.shape[0], h), dtype=np.float32)
            step = 64
            for s in range(0, o.shape[0], step):
                e = min(s + step, o.shape[0])
                ctx = [torch.from_numpy(hist[i])[None, :] for i in range(s, e)]
                res = pipe.predict(ctx, prediction_length=h)
                for k, r in enumerate(res):
                    preds[s + k] = np.asarray(r)[0, q50, :h]
            per.setdefault(name, {})[str(h)] = score_against_core_grid(ds, o, c, h, preds)
            log(f"  chronos {name} H={h}: MSE {per[name][str(h)]['mse']:.5f}")
    out["per_dataset"] = per
    out["status"] = "COMPLETE"
    out["wall_seconds"] = time.time() - t0
    return out


def run_tirex(budget_s: float):
    t0 = time.time()
    out = {
        "model": "TiRex-2",
        "weight": "NX-AI/TiRex-2",
        "device": "cpu",
        "device_note": (
            "CPU only on this machine. flashrnn compiles a fused sLSTM CUDA kernel at "
            "runtime and needs MSVC plus a CUDA Toolkit, neither of which is installed; "
            "the Triton backend exceeds the RTX 4070 shared-memory limit. No Docker, WSL "
            "or driver change was attempted, as section 17 forbids it."
        ),
    }
    try:
        from tirex2 import TimeseriesType, load_model
        model = load_model("NX-AI/TiRex-2", device="cpu")
    except Exception as e:
        out["status"] = f"BLOCKED_ENV: {e}"
        return out
    out["setup_seconds"] = time.time() - t0
    levels = [round(float(q), 4) for q in model.quantiles]
    if 0.5 not in levels:
        out["status"] = "BLOCKED_CONTRACT: no q0.5 among the returned quantiles"
        out["quantiles"] = levels
        return out
    q50 = levels.index(0.5)
    out["quantiles"] = levels
    out["point_forecast_type"] = "median"

    # TiRex-2 caps prediction_length at 320, so H=336 is outside what the released model
    # supports. Rolling it forward would be a usage the authors did not define, which
    # section 17 rules out, so that horizon is recorded as blocked rather than faked.
    probe = model.forecast(
        [TimeseriesType(target=torch.zeros(1, D.L), past_covariates=None, future_covariates=None)],
        prediction_length=max(HORIZONS),
        output_type="numpy",
    )
    max_h = int(np.asarray(probe[0]).shape[-1])
    out["max_supported_prediction_length"] = max_h
    blocked = [h for h in HORIZONS if h > max_h]
    if blocked:
        out["blocked_horizons"] = {
            str(h): f"BLOCKED_CONTRACT: model caps prediction_length at {max_h}"
            for h in blocked
        }

    per = {}
    for name in DATASETS:
        ds = D.Dataset(name)
        o, c = subgrid(ds)
        hist = raw_history(ds, o, c)
        for h in HORIZONS:
            if h > max_h:
                continue
            if time.time() - t0 > budget_s:
                out["status"] = "PARTIAL_BUDGET"
                out["per_dataset"] = per
                out["wall_seconds"] = time.time() - t0
                return out
            preds = np.empty((o.shape[0], h), dtype=np.float32)
            done = 0
            for s in range(0, o.shape[0], 32):
                if time.time() - t0 > budget_s:
                    break
                e = min(s + 32, o.shape[0])
                ts = [
                    TimeseriesType(
                        target=torch.from_numpy(hist[i])[None, :],
                        past_covariates=None,
                        future_covariates=None,
                    )
                    for i in range(s, e)
                ]
                res = model.forecast(ts, prediction_length=h, output_type="numpy")
                for k, r in enumerate(res):
                    preds[s + k] = np.asarray(r)[0, q50, :h]
                done = e
            if done < o.shape[0]:
                out["status"] = "PARTIAL_BUDGET"
                out["per_dataset"] = per
                out["covered_windows"] = int(done)
                out["planned_windows"] = int(o.shape[0])
                out["wall_seconds"] = time.time() - t0
                return out
            per.setdefault(name, {})[str(h)] = score_against_core_grid(ds, o, c, h, preds)
            log(f"  tirex {name} H={h}: MSE {per[name][str(h)]['mse']:.5f}")
    out["per_dataset"] = per
    out["status"] = "PARTIAL_HORIZON_UNSUPPORTED" if blocked else "COMPLETE"
    out["wall_seconds"] = time.time() - t0
    return out


def core_on_subgrid(device: str):
    """The trained arms re-scored on exactly the reference subgrid keys."""
    rows = {}
    for name in DATASETS:
        ds = D.Dataset(name)
        o, c = subgrid(ds)
        for arm in ("I", "C", "DENSE"):
            B = 64 if arm == "DENSE" else PRIMARY_B
            ck = os.path.join(
                fit_dir({"arm": arm, "dataset": name, "B": B, "seed": SEED0}), "best.pt"
            )
            if not os.path.exists(ck):
                continue
            net = M.build(arm, B, SEED0).to(device)
            net.load_state_dict(torch.load(ck, map_location=device))
            net.eval()
            for h in HORIZONS:
                se = 0.0
                ae = 0.0
                with torch.no_grad():
                    for s in range(0, o.shape[0], E.CHUNK):
                        e = min(s + E.CHUNK, o.shape[0])
                        x = torch.from_numpy(ds.inputs(o[s:e], c[s:e])).to(device)
                        y = torch.from_numpy(ds.targets(o[s:e], c[s:e])).to(device)[:, :h]
                        hb = torch.full((x.shape[0],), float(h), device=device)
                        d = net(x, hb, hb)[:, :h] - y
                        se += float((d**2).sum())
                        ae += float(d.abs().sum())
                n = float(h) * o.shape[0]
                rows.setdefault(name, {}).setdefault(arm, {})[str(h)] = {
                    "mse": se / n,
                    "mae": ae / n,
                }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chronos-budget-min", type=float, default=30.0)
    ap.add_argument("--tirex-budget-min", type=float, default=35.0)
    ap.add_argument(
        "--reuse-chronos",
        action="store_true",
        help="keep the Chronos-2 block already written to optional_references.json "
        "instead of running it again",
    )
    args = ap.parse_args()

    searched = {
        "LocalTokenMerging_2025_ICML": {
            "official_code_found": False,
            "how_checked": (
                "PMLR landing page carried no code link; the 21-page PDF was downloaded and "
                "its text and link annotations were scanned. The only GitHub URL in it is a "
                "citation to Lyken17/pytorch-OpCounter, a FLOP counter, not the authors' "
                "implementation. GitHub repository search on the title terms returned nothing "
                "matching."
            ),
        },
        "BPE4TS_2026_ICML": {
            "official_code_found": False,
            "how_checked": (
                "arXiv abstract page and the 32-page PDF text carried no repository link; "
                "GitHub repository search on the title terms returned nothing matching."
            ),
        },
        "TimeSqueeze": {
            "official_code_found": False,
            "how_checked": (
                "arXiv abstract page and the 21-page PDF text carried no repository link; "
                "GitHub repository search returned nothing matching."
            ),
        },
        "PATK_2026_AAAI": {
            "official_code_found": False,
            "how_checked": "AAAI OJS landing page carried no code link; GitHub search returned nothing matching.",
        },
    }
    table = literature_table(searched)
    log("section 16: no official implementation found for any of the four methods")

    log("section 17: foundation reference on the declared subgrid")
    grid = {}
    for name in DATASETS:
        ds = D.Dataset(name)
        o, c = subgrid(ds)
        grid[name] = {
            "origins": int(np.unique(o).shape[0]),
            "channels": int(np.unique(c).shape[0]),
            "windows_per_horizon": int(o.shape[0]),
            "channel_subsample": SUBGRID_CHANNEL_CAP[name] is not None,
        }
    core = core_on_subgrid(args.device)
    path = os.path.join(RESULTS, "optional_references.json")
    if args.reuse_chronos and os.path.exists(path):
        with open(path) as f:
            chronos = json.load(f)["chronos2"]
        log(f"  chronos reused from a previous invocation: {chronos['status']}")
    else:
        chronos = run_chronos(args.chronos_budget_min * 60, args.device)
        log(f"  chronos status {chronos['status']}")
    tirex = run_tirex(args.tirex_budget_min * 60)
    log(f"  tirex status {tirex['status']}")

    out = {
        "dynamic_tokenization_baselines": table,
        "missing_strong_dynamic_baseline": True,
        "missing_strong_dynamic_baseline_note": (
            "No official implementation of Local Merging, BPE for time series, TimeSqueeze or "
            "PATK was located inside the timebox, so this pilot has no strong dynamic "
            "tokenization baseline. Writing an adjacent-cosine merge heuristic here would be a "
            "LOCAL_HEURISTIC and must never be labelled a reproduction of any of those papers. "
            "Their absence says nothing about whether the core hypothesis holds."
        ),
        "reference_subgrid": grid,
        "reference_subgrid_note": (
            "Declared before any reference number existed. It is a subset of the frozen test "
            "keys, so the core arms are re-scored on exactly these keys for a like-for-like "
            "row. These numbers must not be placed beside the full-grid core table."
        ),
        "core_arms_on_subgrid": core,
        "chronos2": chronos,
        "tirex2": tirex,
        "interpretation_limits": [
            "These references were pretrained elsewhere and are evaluated zero-shot; the core "
            "arms were fitted on each target. The training conditions are not comparable.",
            "Pretraining overlap with ETTm2, weather and electricity was not checked, so no "
            "unseen zero-shot claim is made.",
            "Both references return quantiles. Scoring a median under MSE, against arms trained "
            "for the mean, is a handicap of unmeasured size.",
            "No adaptation or tokenizer surgery on these models is attempted in this run.",
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"section 16 and 17 written to {path}")


if __name__ == "__main__":
    main()
