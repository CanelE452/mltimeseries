"""Smoke run before the full pilot (instruction section 28).

One farm, small slices, arms H/M/S/D/P, one seed, three epochs. It checks that
the shapes, gradients, losses and invariances behave, and extrapolates a runtime
estimate. A smoke run never produces a scientific verdict.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.uncertain_covariate_path_pilot_v1.src.crps import crps_loss
from experiments.uncertain_covariate_path_pilot_v1.src.models import (
    ParticleForecaster,
    count_parameters,
)
from experiments.uncertain_covariate_path_pilot_v1.src.train import (
    Config,
    SEEDS,
    Split,
    asdict_shallow,
    load_all,
    train_arm,
)

from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir
SMOKE_ARMS = ("H", "M", "S", "D", "P")
MAX_EXAMPLES = {"train": 400, "val": 200, "test": 200}


def take(split: Split, keep: np.ndarray) -> Split:
    fields = asdict_shallow(split)
    idx = torch.as_tensor(np.flatnonzero(keep), device=split.base.device)
    for name, value in fields.items():
        fields[name] = value[keep] if isinstance(value, np.ndarray) else value[idx]
    return Split(**fields)


def subset(split: Split, farm: int, limit: int) -> Split:
    keep = (split.farm_idx.cpu().numpy() == farm)
    keep[np.flatnonzero(keep)[limit:]] = False
    return take(split, keep)


def main(tag: str = "full") -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, _, n_farms = load_all(device, SMOKE_ARMS)
    small = {s: subset(v, farm=0, limit=MAX_EXAMPLES[s]) for s, v in splits.items()}
    cfg = replace(Config(), max_epochs=3, patience=3, batch_size=64)

    report = {
        "arms": list(SMOKE_ARMS),
        "seed": SEEDS[0],
        "epochs": cfg.max_epochs,
        "n_examples": {s: len(v) for s, v in small.items()},
        "shapes": {
            "base": list(small["train"].base.shape),
            "weather": list(small["train"].weather.shape),
            "target": list(small["train"].target.shape),
        },
    }

    fits, timings = {}, {}
    for arm in SMOKE_ARMS:
        started = time.time()
        rec = train_arm(arm, SEEDS[0], small, n_farms, cfg, device, tag)
        timings[arm] = (time.time() - started) / cfg.max_epochs
        fits[arm] = {
            "n_parameters": rec["n_parameters"],
            "val_macro_scaled_crps": rec["best_val_macro_scaled_crps"],
            "seconds_per_epoch": round(timings[arm], 2),
            "loss_finite": bool(np.isfinite(rec["best_val_macro_scaled_crps"])),
        }
        print(f"{arm:2s} params {rec['n_parameters']:,} val {rec['best_val_macro_scaled_crps']:.4f} "
              f"{timings[arm]:.1f}s/epoch", flush=True)
    report["fits"] = fits

    d_params = fits["D"]["n_parameters"]
    p_params = fits["P"]["n_parameters"]
    report["d_p_parameter_equality"] = {
        "D": d_params,
        "P": p_params,
        "relative_difference_percent": 100.0 * abs(d_params - p_params) / d_params,
        "pass": d_params == p_params,
    }

    torch.manual_seed(0)
    model = ParticleForecaster(
        arm="P", n_base=small["train"].base.shape[-1],
        n_weather=small["train"].weather.shape[-1], n_farms=n_farms,
    ).to(device).eval()
    b = small["test"]
    sl = slice(0, min(32, len(b)))
    perm = torch.randperm(b.weather.shape[1], device=device)
    with torch.no_grad():
        a = model(b.base[sl], b.weather[sl], b.farm_idx[sl])
        c = model(b.base[sl], b.weather[sl][:, perm], b.farm_idx[sl])
    max_diff = float((a - c).abs().max())
    report["whole_path_permutation_invariance"] = {
        "max_abs_diff": max_diff, "tolerance": 1e-6, "pass": max_diff <= 1e-6,
    }

    grads = []
    model.train()
    loss = crps_loss(model(b.base[sl], b.weather[sl], b.farm_idx[sl]), b.target[sl])
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads.append(float(p.grad.abs().max()))
    report["gradients"] = {
        "loss": float(loss), "loss_finite": bool(np.isfinite(float(loss))),
        "max_abs_grad": max(grads), "all_finite": bool(np.isfinite(grads).all()),
        "pass": bool(np.isfinite(float(loss)) and np.isfinite(grads).all() and max(grads) > 0),
    }

    scale = len(splits["train"]) / max(len(small["train"]), 1)
    full_cfg = Config()
    report["runtime_estimate"] = {
        "seconds_per_epoch_scaled": {a: round(t * scale, 1) for a, t in timings.items()},
        "worst_case_gpu_hours_21_fits": round(
            sum(timings.values()) / len(timings) * scale * full_cfg.max_epochs * 21 / 3600, 2
        ),
        "note": "worst case assumes every fit runs the full 50 epochs without early stopping",
    }
    report["scope"] = "smoke run; no scientific verdict is taken from it"
    report["all_passed"] = bool(
        report["d_p_parameter_equality"]["pass"]
        and report["whole_path_permutation_invariance"]["pass"]
        and report["gradients"]["pass"]
        and all(f["loss_finite"] for f in fits.values())
    )

    (out_dir(tag) / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "fits"}, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
