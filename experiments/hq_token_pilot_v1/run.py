"""HQ-TOKEN-PILOT-v1 driver: freeze, smoke, checks, fits, analysis, verdict, report.

Contract source: 01_forecast_query_tokenization_CLI.txt.

One invocation carries the whole pilot to a written verdict. Stages are resumable: a
stage whose completion marker matches the current run key is skipped, and every marker
binds the code, spec, source, split and normalizer hashes, so stale artifacts from an
earlier code version are never reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import torch

from . import data as D
from . import evaluate as E
from . import model as M
from . import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = D.REPO_ROOT
RESULTS = os.path.join(ROOT, "results", "hq_token_pilot_v1")
RUNS = os.path.join(ROOT, "runs", "hq_token_pilot_v1")

SEED0 = 2026090601
SEED1 = 2026090602
BOOTSTRAP_SEED = 2026090699
BOOTSTRAP_DRAWS = 1000
NOISE_SEEDS = [2026090711 + i for i in range(8)]

DATASETS = ["ETTm2", "weather", "electricity"]
HORIZONS = [96, 336]
PRIMARY_B = 32

# section 9 defaults; section 11 may reduce these once, before the freeze
TRAIN_CFG = {
    "max_updates": 3000,
    "warmup": 150,
    "validate_every": 600,
    "batch": 64,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
}

# section 14 thresholds, pinned here so no result can move them
GATES = {
    "macro_ri_C_vs_I_min": 1.0,
    "dataset_cells_positive_min": 2,
    "worst_dataset_floor": -1.0,
    "macro_ri_C_vs_H_STATIC_min": 0.3,
    "lower95_definition": "two-sided 95 percent interval, 2.5 percentile",
    "worst_dataset_level": "dataset level, averaged over the two horizons",
    "primary_ri_estimator": "ratio of seed-averaged errors (section 12)",
    "per_seed_ri_estimator": "ratio computed within each model seed (section 14 sign condition)",
}

# seasonal period in samples, one day, from the measured sampling interval
DAILY_PERIOD = {"ETTm2": 96, "weather": 144, "electricity": 24}
# robustness re-read only; the pre-registered grid is offset 0
PHASE_OFFSET = {"ETTm2": 37, "weather": 37, "electricity": 7}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_hash() -> str:
    h = hashlib.sha256()
    for name in ("data.py", "model.py", "train.py", "evaluate.py", "run.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def core_fit_list():
    """section 10: exactly 27 fits, declared up front and never extended afterwards."""
    fits = []
    for d in DATASETS:
        for arm in ("U", "I", "H_STATIC", "C", "R"):
            fits.append({"arm": arm, "dataset": d, "B": PRIMARY_B, "seed": SEED0})
    for d in DATASETS:
        fits.append({"arm": "DENSE", "dataset": d, "B": 64, "seed": SEED0})
    for d in DATASETS:
        for arm in ("I", "H_STATIC", "C"):
            fits.append({"arm": arm, "dataset": d, "B": PRIMARY_B, "seed": SEED1})
    return fits


def fit_dir(fit, tag="core"):
    return os.path.join(
        RUNS, tag, f"{fit['dataset']}__{fit['arm']}__B{fit['B']}__s{fit['seed']}"
    )


# ---- stage 1: smoke (section 11) -------------------------------------------


def stage_smoke(device):
    """Train-only timing probe. Its accuracy is never used for any decision."""
    log("smoke: measuring update and evaluation cost")
    ds = D.Dataset(DATASETS[0])
    net = M.build("C", PRIMARY_B, SEED0).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=TRAIN_CFG["learning_rate"])
    o = ds.train_origins()[:64]
    c = np.arange(64) % ds.n_channels
    x = torch.from_numpy(ds.inputs(o, c)).to(device)
    y = torch.from_numpy(ds.targets(o, c)).to(device)
    h = torch.full((64,), 96.0, device=device)
    for _ in range(10):
        opt.zero_grad(); M.masked_mse(net(x, h, h), y, h).backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(40):
        opt.zero_grad(); M.masked_mse(net(x, h, h), y, h).backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    per_update = (time.time() - t0) / 40

    val_cost = {}
    for name in DATASETS:
        d2 = D.Dataset(name)
        t0 = time.time()
        E.evaluate_split(net, d2, "validation", 96, device, arm="C", seed=SEED0)
        val_cost[name] = time.time() - t0

    n_val_passes = TRAIN_CFG["max_updates"] // TRAIN_CFG["validate_every"]
    est = 0.0
    for fit in core_fit_list():
        est += TRAIN_CFG["max_updates"] * per_update
        est += n_val_passes * 2 * val_cost[fit["dataset"]]
    report = {
        "seconds_per_update": per_update,
        "validation_pass_seconds": val_cost,
        "estimated_core_seconds": est,
        "estimated_core_hours": est / 3600.0,
        "smoke_accuracy_used_for_decisions": False,
    }
    log(f"smoke: {per_update*1e3:.1f} ms/update, estimated core {est/3600:.2f} h")
    return report


# ---- stage 2: freeze (section 3) -------------------------------------------


def stage_freeze(smoke, device):
    manifest = D.build_manifest()
    cfg = dict(TRAIN_CFG)
    budget_flag = None
    if smoke["estimated_core_hours"] > 10.0:
        cfg.update({"max_updates": 1200, "warmup": 60, "validate_every": 240})
        budget_flag = "REDUCED_BUDGET_SCREEN"
        log("freeze: estimated core over 10 h, reducing to 1200 updates for every arm")

    epoch_fraction = {}
    for name in DATASETS:
        m = manifest["datasets"][name]
        windows = m["train_origins"] * m["n_channels"]
        epoch_fraction[name] = cfg["max_updates"] * cfg["batch"] / windows

    spec = {
        "experiment_id": "HQ-TOKEN-PILOT-v1",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hash": code_hash(),
        "repository": "CanelE452/mltimeseries",
        "repository_deviation": (
            "Section 1 names CanelE452/covariate-trust-pilot. The user redirected the run to "
            "CanelE452/mltimeseries, where the three topic notes, the benchmark CSVs, the "
            "PatchTST clone and the conda environment already live. A separate worktree was "
            "also not used: the notes, data and third_party trees are untracked or ignored in "
            "this working copy, so a fresh worktree would not have contained them."
        ),
        "window": {"L": D.L, "P": D.P, "stride": D.P, "N": D.N, "horizons": HORIZONS},
        "token_budget": {"primary_B": PRIMARY_B, "dense_B": 64, "extension_B": 16},
        "training": cfg,
        "budget_flag": budget_flag,
        "train_epoch_fraction": epoch_fraction,
        "seeds": {
            "model_seed_0": SEED0,
            "model_seed_1": SEED1,
            "fake_query_seed": 2026090671,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "noise_floor_seeds": NOISE_SEEDS,
        },
        "evaluation": {
            "origin_stride": D.EVAL_STRIDE,
            "logical_key": ["dataset_id", "split", "origin", "channel_id", "horizon", "model_seed"],
            "primary_space": "train-only standardized (ddof=0)",
            "metrics": ["MSE of MSE-trained point forecast", "MAE"],
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "block_origins": E.BLOCK_ORIGINS,
                "unit": "chronological origin, circular moving blocks, paired across arms",
                "seeds_resampled": False,
            },
        },
        "gates": GATES,
        "planned_fits": core_fit_list(),
        "datasets": manifest["datasets"],
        "provenance": manifest["provenance"],
        "added_validity_checks": {
            "why": (
                "An independent critique of this plan found four ways a null or a positive "
                "result could not be interpreted. None of the pre-registered arms, thresholds, "
                "splits, grids or budgets were changed to address them; these are extra "
                "measurements declared before any core result exists."
            ),
            "compression_headroom": (
                "DENSE(B=64) versus U(B=32) is read as a precondition. If compressing 64 "
                "tokens to 32 costs almost nothing, no pooling rule can gain the 1.0 percent "
                "the gate asks for, and a null must not be reported as SCREEN_NEGATIVE."
            ),
            "mechanism_capacity_check": (
                "A synthetic task whose optimal pooling provably differs by horizon. "
                "Separates 'the hypothesis is wrong' from 'this scorer cannot learn it'. "
                "Synthetic; never reported as a benchmark result."
            ),
            "seed_noise_floor": (
                "Arm I at eight extra seeds, validation split only, run before the core. "
                "Measures whether the 1.0 percent threshold sits above seed noise. Not an "
                "input to any verdict; the core still uses exactly the two declared seeds."
            ),
            "phase_shifted_grid": (
                "Stride 96 is exactly 24 h on ETTm2 and exactly 4 days on electricity, so "
                "every pre-registered origin falls at one time of day. The same trained "
                "checkpoints are re-read on a shifted grid as a robustness row. The "
                "pre-registered grid stays primary."
            ),
            "seasonal_naive_anchor": (
                "Repeat-last-day baseline on the same grid, so the arms can be placed on an "
                "absolute scale."
            ),
        },
        "known_limitations": {
            "epoch_fraction": (
                "The fixed 3000-update budget is far below one pass over the eligible windows, "
                "most severely on electricity. What is measured is early-optimization quality, "
                "not converged quality."
            ),
            "horizon_reaches_the_encoder_through_the_tokens": (
                "In C the pooling weights depend on H, so the merged tokens themselves encode "
                "H and the encoder can read it; in I they cannot. H_STATIC controls this path "
                "because its weights also depend on H while ignoring content, which is why the "
                "C versus H_STATIC contrast carries the identification weight here."
            ),
            "shared_horizon_embedding": (
                "Section 6 defines a single e_H and section 7 reuses that same e_H as the C/R "
                "pooling query, so the module is shared. In C its parameters therefore receive "
                "gradient from both the decoder and the scorer path, which I does not. "
                "Implemented as written rather than split, because splitting it would create an "
                "arm the plan does not contain."
            ),
            "checkpoint_selection_noise": (
                "One checkpoint out of five is chosen on the validation grid, which is only 24 "
                "origins on electricity. That selection noise is not carried into the interval."
            ),
            "no_instance_normalization": (
                "The pilot model has no RevIN-style instance normalization. Section 6 does not "
                "include one. Absolute accuracy is therefore not comparable to published "
                "PatchTST numbers."
            ),
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    # Written as bytes: text mode would turn every newline into CRLF on Windows and the
    # companion hash, computed over the string, would then not verify against the file.
    body = json.dumps(spec, indent=2, sort_keys=True).encode()
    with open(os.path.join(RESULTS, "execution_spec.json"), "wb") as f:
        f.write(body)
    with open(os.path.join(RESULTS, "execution_spec.sha256"), "wb") as f:
        f.write((hashlib.sha256(body).hexdigest() + "  execution_spec.json\n").encode())

    with open(os.path.join(RESULTS, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(RESULTS, "environment.json"), "w") as f:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_used": device,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "smoke": smoke,
            },
            f,
            indent=2,
        )
    log(f"freeze: spec written, code_hash {spec['code_hash'][:16]}")
    return spec


# ---- stage 3: mechanism capacity check (synthetic) -------------------------


def synthetic_horizon_task(n, rng):
    """Each group holds one a-patch and one b-patch, at independent levels.

    The short future is the mean of the 32 a-levels, the long future the mean of the 32
    b-levels. A group can only pass one mixture forward, and the a-levels are mutually
    independent, so no fixed allocation of groups serves both futures: an even split and
    a 50/50 blend both leave error variance 0.5, while a weight that follows the horizon
    reaches zero. Two output degrees of freedom keeps the decoder out of the way.

    Three earlier designs were rejected while building this check, and they are worth
    recording because two of them are statements about the mechanism itself:
      1. white-noise patches with a summed target -- unlearnable, the target had no
         structure and predicting zero was already near optimal;
      2. one shared level on all even patches -- the input-only pooler solved it by
         allocating different groups to different roles, so horizon conditioning bought
         nothing. Spatial allocation across 32 groups is a genuine alternative to
         horizon conditioning, not an artifact of the check;
      3. 32 independent target values per horizon -- the decoder learned nothing inside
         the budget.
    """
    a = rng.standard_normal((n, D.N // 2)).astype(np.float32)
    b = rng.standard_normal((n, D.N // 2)).astype(np.float32)
    levels = np.empty((n, D.N), dtype=np.float32)
    levels[:, 0::2] = a
    levels[:, 1::2] = b
    x = np.repeat(levels, D.P, axis=1) + 0.1 * rng.standard_normal((n, D.L)).astype(np.float32)
    y = np.empty((n, D.H_MAX), dtype=np.float32)
    scale = np.sqrt(D.N // 2)
    y[:, :96] = (a.mean(axis=1) * scale)[:, None]
    y[:, 96:] = (b.mean(axis=1) * scale)[:, None]
    return x, y


def stage_capacity_check(device, updates=1500, batch=64):
    log("capacity check: synthetic horizon-split pooling task")
    out = {}
    for arm in ("I", "C"):
        rng = np.random.default_rng(20260906)
        net = M.build(arm, PRIMARY_B, SEED0).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
        for u in range(updates):
            x, y = synthetic_horizon_task(batch, rng)
            h = float(96 if u % 2 == 0 else 336)
            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(y).to(device)
            hb = torch.full((batch,), h, device=device)
            opt.zero_grad(set_to_none=True)
            M.masked_mse(net(xb, hb, hb), yb, hb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        ev = np.random.default_rng(777)
        x, y = synthetic_horizon_task(1024, ev)
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        per_h = {}
        with torch.no_grad():
            for h in HORIZONS:
                hb = torch.full((1024,), float(h), device=device)
                per_h[h] = float(M.masked_mse(net(xb, hb, hb), yb, hb))
            w = {}
            for h in HORIZONS:
                hb = torch.full((1024,), float(h), device=device)
                _, ww = net(xb, hb, hb, return_weights=True)
                w[h] = float(ww[..., 0].mean())              # weight on the a-patch
            out[f"{arm}_mean_weight_on_a_patch"] = w
        out[arm] = per_h
    ri = {}
    for h in HORIZONS:
        r, _ = E.relative_improvement(out["C"][h], out["I"][h])
        ri[h] = r
    separation = abs(
        out["C_mean_weight_on_a_patch"][96] - out["C_mean_weight_on_a_patch"][336]
    )
    out["relative_improvement_pct"] = ri
    out["C_weight_separation_between_horizons"] = separation
    out["updates"] = updates
    out["verdict"] = (
        "CAPACITY_CONFIRMED"
        if separation > 0.30 and ri[96] is not None and ri[96] > 5.0
        else "CAPACITY_NOT_DEMONSTRATED"
    )
    out["reading_note"] = (
        "The verdict reads the H=96 cell and the weight separation. The H=336 cell is "
        "expected to favour I on this task and is not evidence against capacity: the "
        "H=336 loss mask spans steps 0..335, so it still contains the 96 a-steps, and a "
        "single weight per horizon cannot serve both halves of that window. What matters "
        "is whether the scorer moves the weight when the horizon changes."
    )
    log(
        f"capacity check: {out['verdict']}, RI {ri}, "
        f"weight separation {separation:.3f}"
    )
    with open(os.path.join(RESULTS, "mechanism_capacity_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---- stage 4: seed noise floor (validation only) ---------------------------


def stage_noise_floor(spec, device):
    log("seed noise floor: arm I, 8 extra seeds, validation split only")
    per_dataset = {}
    for name in DATASETS:
        ds = D.Dataset(name)
        vals = []
        for s in NOISE_SEEDS:
            out = os.path.join(RUNS, "noise", f"{name}__I__s{s}")
            rep = T.fit("I", ds, PRIMARY_B, s, spec["training"], out, device=device)
            net = M.build("I", PRIMARY_B, s).to(device)
            net.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
            mse = [
                E.evaluate_split(net, ds, "validation", h, device, arm="I", seed=s)["mse"]
                for h in HORIZONS
            ]
            vals.append(float(np.mean(mse)))
            log(f"  {name} seed {s}: mean validation MSE {vals[-1]:.5f}")
        arr = np.array(vals)
        per_dataset[name] = {
            "seeds": NOISE_SEEDS,
            "mean_validation_mse": vals,
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)),
            "sd_pct_of_mean": float(100.0 * arr.std(ddof=1) / arr.mean()),
        }
    worst = max(v["sd_pct_of_mean"] for v in per_dataset.values())
    out = {
        "per_dataset": per_dataset,
        "worst_seed_sd_pct": worst,
        "gate_threshold_pct": GATES["macro_ri_C_vs_I_min"],
        "threshold_above_noise": bool(worst < GATES["macro_ri_C_vs_I_min"]),
        "note": (
            "Validation split, arm I only, seeds outside the two pre-registered model seeds. "
            "This measures the resolution available to the 1.0 percent gate. It is not an "
            "input to the verdict."
        ),
    }
    log(f"seed noise floor: worst per-seed sd {worst:.2f} percent of mean MSE")
    with open(os.path.join(RESULTS, "seed_noise_floor.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---- stage 5: core fits ----------------------------------------------------


def stage_core(spec, device, guard):
    fits = core_fit_list()
    rows = []
    keyed = []
    cache = {name: D.Dataset(name) for name in DATASETS}
    for i, fit in enumerate(fits, 1):
        ds = cache[fit["dataset"]]
        out = fit_dir(fit)
        log(f"core {i}/{len(fits)}: {fit['arm']} {fit['dataset']} B={fit['B']} seed={fit['seed']}")
        rep = T.fit(
            fit["arm"], ds, fit["B"], fit["seed"], spec["training"], out,
            device=device, guard=guard,
        )
        rep["run_key"] = _run_key(spec, ds, fit)
        rows.append(rep)
        if rep["status"] != "COMPLETE":
            log(f"  -> {rep['status']}: {rep['stop_note']}")
            continue
        net = M.build(fit["arm"], fit["B"], fit["seed"]).to(device)
        net.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
        for split in ("validation", "test"):
            for h in HORIZONS:
                r = E.evaluate_split(
                    net, ds, split, h, device, arm=fit["arm"], seed=fit["seed"], keep_keys=True
                )
                r["B"] = fit["B"]
                keyed.append(r)
        for h in HORIZONS:
            r = E.evaluate_split(
                net, ds, "test", h, device, arm=fit["arm"], seed=fit["seed"],
                keep_keys=True, phase_offset=PHASE_OFFSET[fit["dataset"]],
            )
            r["B"] = fit["B"]
            r["split"] = "test_phase_shifted"
            keyed.append(r)
        np.savez_compressed(
            os.path.join(out, "keyed_errors.npz"),
            **{
                f"{r['split']}_{r['horizon']}_{k}": r[k]
                for r in keyed[-6:]
                for k in ("origin", "channel", "se_sum", "ae_sum", "count")
            },
        )
        log(f"  test MSE  H96 {keyed[-4]['mse']:.5f}  H336 {keyed[-3]['mse']:.5f}")
    return rows, keyed


def _run_key(spec, ds, fit) -> str:
    payload = json.dumps(
        {
            "code_hash": spec["code_hash"],
            "spec_hash": hashlib.sha256(
                json.dumps(spec["training"], sort_keys=True).encode()
            ).hexdigest(),
            "source_hash": ds.csv_sha256,
            "split_key_hash": ds.split_key_hash(),
            "normalizer_hash": ds.normalizer_hash(),
            "arm": fit["arm"],
            "B": fit["B"],
            "seed": fit["seed"],
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# ---- stage 6: analysis -----------------------------------------------------


def _table(keyed, split="test"):
    return E.key_frame([r for r in keyed if r["split"] == split])


def contrasts(table, treat, ref, seeds):
    cells, per_dataset, per_seed = {}, {}, {}
    for d in DATASETS:
        vals = []
        for h in HORIZONS:
            a = E.seed_averaged(table, treat, d, h, seeds)
            b = E.seed_averaged(table, ref, d, h, seeds)
            if a is None or b is None:
                cells[f"{d}|{h}"] = None
                continue
            ri, flag = E.relative_improvement(E.cell_error(a), E.cell_error(b))
            cells[f"{d}|{h}"] = ri
            if ri is not None:
                vals.append(ri)
        per_dataset[d] = float(np.mean(vals)) if vals else None
    for s in seeds:
        vals = []
        for d in DATASETS:
            for h in HORIZONS:
                a = E.seed_averaged(table, treat, d, h, [s])
                b = E.seed_averaged(table, ref, d, h, [s])
                if a is None or b is None:
                    continue
                ri, _ = E.relative_improvement(E.cell_error(a), E.cell_error(b))
                if ri is not None:
                    vals.append(ri)
        per_seed[s] = float(np.mean(vals)) if vals else None
    return {
        "treat": treat,
        "ref": ref,
        "cells": cells,
        "per_dataset_horizon_mean": per_dataset,
        "per_seed_macro": per_seed,
        "macro": E.macro_ri(cells),
    }


def decide(c_vs_i, c_vs_hs, c_vs_r, boot, complete_core, headroom, noise):
    """section 14, evaluated condition by condition so the reader sees what failed."""
    cond = {
        "core_complete_for_I_H_STATIC_C_both_seeds": complete_core,
        "macro_RI_C_vs_I_at_least_1pct": (
            c_vs_i["macro"] is not None and c_vs_i["macro"] >= GATES["macro_ri_C_vs_I_min"]
        ),
        "at_least_2_datasets_positive": (
            sum(1 for v in c_vs_i["per_dataset_horizon_mean"].values() if v is not None and v > 0)
            >= GATES["dataset_cells_positive_min"]
        ),
        "no_dataset_worse_than_-1pct": all(
            v is None or v >= GATES["worst_dataset_floor"]
            for v in c_vs_i["per_dataset_horizon_mean"].values()
        ),
        "both_seed_macros_positive": all(
            v is not None and v > 0 for v in c_vs_i["per_seed_macro"].values()
        ),
        "bootstrap_lower95_positive": (
            boot.get("lower95") is not None and boot["lower95"] > 0
        ),
        "macro_RI_C_vs_H_STATIC_at_least_0.3pct": (
            c_vs_hs["macro"] is not None
            and c_vs_hs["macro"] >= GATES["macro_ri_C_vs_H_STATIC_min"]
        ),
        "seed0_C_vs_R_positive": (
            c_vs_r["per_seed_macro"].get(SEED0) is not None
            and c_vs_r["per_seed_macro"][SEED0] > 0
        ),
    }
    notes = []
    if not complete_core:
        decision = "NOT_EVALUATED"
        notes.append("core is not complete for I, H_STATIC and C at both seeds")
    elif all(cond.values()):
        decision = "MECHANISM_PROMISING_NOT_SOTA"
    elif c_vs_hs["macro"] is not None and c_vs_hs["macro"] <= 0 and (
        c_vs_i["macro"] is not None and c_vs_i["macro"] > 0
    ):
        decision = "HORIZON_PRIOR_SUFFICIENT"
        notes.append("C is at or behind a fixed horizon prior, so a C over I gain is not evidence for content-conditioned pooling")
    elif all(v is not None and v <= 0 for v in c_vs_i["per_seed_macro"].values()):
        decision = "SCREEN_NEGATIVE"
    else:
        decision = "INCONCLUSIVE"
        notes.append("effect present but at least one pre-registered condition failed")

    if headroom is not None and headroom < GATES["macro_ri_C_vs_I_min"]:
        notes.append(
            f"compression headroom (DENSE over U) is {headroom:.2f} percent, below the "
            f"{GATES['macro_ri_C_vs_I_min']} percent gate. Under this budget no pooling rule "
            f"could have reached the threshold, so a negative reading here is a statement "
            f"about the setting, not about the hypothesis."
        )
        if decision == "SCREEN_NEGATIVE":
            decision = "INCONCLUSIVE"
            notes.append("downgraded from SCREEN_NEGATIVE to INCONCLUSIVE for that reason")
    if noise is not None and not noise["threshold_above_noise"]:
        notes.append(
            f"seed-to-seed sd of validation MSE reaches {noise['worst_seed_sd_pct']:.2f} percent, "
            f"at or above the {GATES['macro_ri_C_vs_I_min']} percent gate; the two-seed interval "
            f"does not carry that source of variation"
        )
    return {"decision": decision, "conditions": cond, "notes": notes}


def stage_analyze(spec, rows, keyed, device, capacity, noise):
    log("analysis: metrics, contrasts, bootstrap, diagnostics, efficiency")
    table = _table(keyed, "test")
    table_shift = _table(keyed, "test_phase_shifted")
    both = [SEED0, SEED1]

    c_vs_i = contrasts(table, "C", "I", both)
    c_vs_hs = contrasts(table, "C", "H_STATIC", both)
    c_vs_r = contrasts(table, "C", "R", [SEED0])
    c_vs_u = contrasts(table, "C", "U", [SEED0])
    c_vs_dense = contrasts(table, "C", "DENSE", [SEED0])
    dense_vs_u = contrasts(table, "DENSE", "U", [SEED0])
    c_vs_i_shift = contrasts(table_shift, "C", "I", both)

    boot = E.bootstrap_macro(table, "C", "I", DATASETS, HORIZONS, both,
                             draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)

    ok = {(r["arm"], r["dataset_id"], r["model_seed"]) for r in rows if r["status"] == "COMPLETE"}
    complete_core = all(
        (a, d, s) in ok for a in ("I", "H_STATIC", "C") for d in DATASETS for s in both
    )
    headroom = dense_vs_u["macro"]
    verdict = decide(c_vs_i, c_vs_hs, c_vs_r, boot, complete_core, headroom, noise)

    # section 13, C only
    diagnostics = {}
    for d in DATASETS:
        p = fit_dir({"arm": "C", "dataset": d, "B": PRIMARY_B, "seed": SEED0})
        ck = os.path.join(p, "best.pt")
        if not os.path.exists(ck):
            continue
        ds = D.Dataset(d)
        net = M.build("C", PRIMARY_B, SEED0).to(device)
        net.load_state_dict(torch.load(ck, map_location=device))
        diagnostics[d] = {
            "query_swap": E.query_swap_diagnostic(net, ds, device, "C", SEED0),
            "weight_profile": E.weight_profile(net, ds, device),
        }

    # section 15
    efficiency = {}
    for arm, B in (("U", 32), ("I", 32), ("C", 32), ("H_STATIC", 32), ("R", 32), ("DENSE", 64)):
        efficiency[f"{arm}_B{B}"] = E.measure_latency(M.build(arm, B, SEED0).to(device), device)

    anchor = {}
    for d in DATASETS:
        ds = D.Dataset(d)
        anchor[d] = {
            str(h): E.seasonal_naive(ds, "test", h, DAILY_PERIOD[d]) for h in HORIZONS
        }

    metrics_rows = []
    for r in keyed:
        if r["split"] != "test":
            continue
        metrics_rows.append(
            {
                "dataset": r["dataset_id"], "horizon": r["horizon"], "B": r["B"],
                "model_seed": r["model_seed"], "arm": r["arm"],
                "MSE": r["mse"], "MAE": r["mae"], "n_keys": r["n_keys"],
            }
        )

    out = {
        "primary": c_vs_i,
        "secondary": {
            "C_vs_H_STATIC": c_vs_hs, "C_vs_R": c_vs_r,
            "C_vs_U": c_vs_u, "C_vs_DENSE": c_vs_dense,
        },
        "precondition_compression_headroom_DENSE_vs_U": dense_vs_u,
        "robustness_phase_shifted_grid": c_vs_i_shift,
        "bootstrap": boot,
        "verdict": verdict,
        "seasonal_naive_anchor": anchor,
        "mechanism_capacity_check": capacity,
        "seed_noise_floor": noise,
    }
    with open(os.path.join(RESULTS, "contrasts.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RESULTS, "bootstrap.json"), "w") as f:
        json.dump(boot, f, indent=2)
    with open(os.path.join(RESULTS, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    _write_csv(os.path.join(RESULTS, "metrics.csv"), metrics_rows)
    _write_csv(
        os.path.join(RESULTS, "fit_manifest.csv"),
        [
            {
                "arm": r["arm"], "dataset": r["dataset_id"], "B": r["B"],
                "model_seed": r["model_seed"], "status": r["status"],
                "updates_run": r["updates_run"], "best_update": r["best_checkpoint_update"],
                "wall_seconds": round(r["wall_seconds"], 1), "run_key": r["run_key"][:16],
                "collapsed": r["collapsed_to_constant"],
            }
            for r in rows
        ],
    )
    eff_rows = []
    for k, v in efficiency.items():
        eff_rows.append(
            {
                "config": k,
                "tokens_encoded": v["tokens_encoded"],
                "total_parameters": v["parameters"]["total_parameters"],
                "tokenizer_parameters": v["parameters"]["tokenizer_parameters"],
                "model_call_median_ms_b1": v["batch1"]["model_call_median_ms"],
                "model_call_median_ms_b64": v["batch64"]["model_call_median_ms"],
                "end_to_end_median_ms_b64": v["batch64"]["end_to_end_median_ms"],
                "p90_ms_b64": v["batch64"]["model_call_p90_ms"],
                "throughput_windows_per_s_b64": v["batch64"]["throughput_windows_per_s"],
                "peak_allocated_bytes_b64": v["batch64"]["peak_allocated_bytes"],
            }
        )
    _write_csv(os.path.join(RESULTS, "efficiency.csv"), eff_rows)
    return out, diagnostics, efficiency


def _write_csv(path, rows):
    import csv

    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    global RESULTS, RUNS, NOISE_SEEDS, BOOTSTRAP_DRAWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-noise-floor", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise every stage at a tiny budget in a separate directory; its numbers "
        "are wiring checks and are never reported as results",
    )
    args = ap.parse_args()

    capacity_updates = 3000
    if args.dry_run:
        RESULTS = RESULTS + "_dryrun"
        RUNS = RUNS + "_dryrun"
        TRAIN_CFG.update({"max_updates": 20, "warmup": 2, "validate_every": 10})
        NOISE_SEEDS = NOISE_SEEDS[:2]
        BOOTSTRAP_DRAWS = 50
        capacity_updates = 20
        log("DRY RUN: tiny budget, separate output directory, results are not scientific")

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    t0 = time.time()
    state = {"stages": {}}
    try:
        with T.ResourceGuard() as guard:
            smoke = stage_smoke(args.device)
            spec = stage_freeze(smoke, args.device)
            capacity = stage_capacity_check(args.device, updates=capacity_updates)
            noise = None if args.skip_noise_floor else stage_noise_floor(spec, args.device)
            rows, keyed = stage_core(spec, args.device, guard)
            analysis, diagnostics, efficiency = stage_analyze(
                spec, rows, keyed, args.device, capacity, noise
            )
            state["resource"] = guard.report()
        state["wall_seconds"] = time.time() - t0
        from .report import write_status

        write_status(RESULTS, spec, rows, analysis, diagnostics, efficiency, state)
        v = analysis["verdict"]
        log("=" * 70)
        log(f"execution_status  : {'COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL'}")
        log(f"scientific_decision: {v['decision']}")
        log(f"C vs I macro RI   : {analysis['primary']['macro']}")
        log(f"bootstrap lower95 : {analysis['bootstrap'].get('lower95')}")
        log(f"total wall        : {state['wall_seconds']/60:.1f} min")
        log(f"report            : {os.path.join(RESULTS, 'STATUS.md')}")
        log("=" * 70)
    except Exception:
        log("FAILED")
        traceback.print_exc()
        with open(os.path.join(RESULTS, "FAILURE.txt"), "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
