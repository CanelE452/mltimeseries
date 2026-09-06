"""Score every fitted arm on validation and test, and write the metric tables.

Per-example, per-lead losses are stored so the bootstrap and the point estimate
read exactly the same numbers (test A17), and so seed aggregation can be checked
against the point estimate afterwards (test A18).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.uncertain_covariate_path_pilot_v1.src.crps import crps_efficient
from experiments.uncertain_covariate_path_pilot_v1.src.models import ParticleForecaster
from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir, runs_dir
from experiments.uncertain_covariate_path_pilot_v1.src.train import (
    ARMS,
    SEEDS,
    Config,
    Split,
    load_all,
    mc_particles,
)

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"
LEAD_GROUPS = {"6-24h": (6, 24), "30-48h": (30, 48), "54-72h": (54, 72)}
COVERAGE_LEVEL = 0.90


def predict(model, split: Split, arm: str, batch: int = 256) -> torch.Tensor:
    model.eval()
    pieces = []
    with torch.no_grad():
        for i in range(0, len(split), batch):
            sl = slice(i, min(i + batch, len(split)))
            if arm == "MC":
                pieces.append(mc_particles(model, split, sl))
            else:
                pieces.append(model(split.base[sl], split.weather[sl], split.farm_idx[sl]))
    return torch.cat(pieces)


def summarise(particles: torch.Tensor, split: Split, leads_h: np.ndarray) -> dict:
    """Metrics for one fit. Mean and median estimands are kept apart on purpose."""
    scaled_crps = crps_efficient(particles, split.target)  # (N, T)
    mw_scale = split.scale[:, None]
    mw_crps = scaled_crps * mw_scale

    mean_pred = particles.mean(dim=-1)
    median_pred = particles.median(dim=-1).values
    lo = torch.quantile(particles, (1 - COVERAGE_LEVEL) / 2, dim=-1)
    hi = torch.quantile(particles, 1 - (1 - COVERAGE_LEVEL) / 2, dim=-1)
    inside = ((split.target >= lo) & (split.target <= hi)).float()

    return {
        "scaled_crps": scaled_crps,
        "mw_crps": mw_crps,
        "mean_rmse_mw": float(
            torch.sqrt((((mean_pred - split.target) * mw_scale) ** 2).mean())
        ),
        "median_mae_mw": float(((median_pred - split.target) * mw_scale).abs().mean()),
        "coverage_90": float(inside.mean()),
        "width_90_mw": float(((hi - lo) * mw_scale).mean()),
        "leads_h": leads_h,
    }


def macro(per_example_lead: torch.Tensor, farm_idx: torch.Tensor, n_farms: int,
          lead_mask: np.ndarray | None = None) -> float:
    values = per_example_lead if lead_mask is None else per_example_lead[:, lead_mask]
    per_farm = [values[farm_idx == f].mean() for f in range(n_farms) if (farm_idx == f).any()]
    return float(torch.stack(per_farm).mean())


def main(arms=ARMS, seeds=SEEDS, splits_to_score=("val", "test"), tag: str = "full") -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, broken, n_farms = load_all(device, tuple(arms))
    farms = pd.read_csv(RESULTS / "farm_selection.csv")["farm_id"].tolist()
    leads_h = np.load(
        ROOT / "data_external/ucp_path_pilot_v1/chronos_cache/chronos_test.npz"
    )["leads_h"]
    cfg = Config()

    rows, by_farm, by_lead = [], [], []
    losses: dict[tuple[str, str], np.ndarray] = {}
    key_check = {}

    for arm in arms:
        use = broken if arm == "P_BROKEN" else splits
        for split_name in splits_to_score:
            split = use[split_name]
            if len(split) == 0:
                continue
            key_check.setdefault(split_name, list(split.example_id))
            if key_check[split_name] != list(split.example_id):
                raise RuntimeError(f"A03 violated: {arm} sees different keys on {split_name}")

            stacked = []
            for seed in seeds:
                model = ParticleForecaster(
                    arm=arm,
                    n_base=split.base.shape[-1],
                    n_weather=split.weather.shape[-1],
                    n_farms=n_farms,
                    hidden=cfg.hidden,
                    n_particles=cfg.n_particles,
                    dropout=cfg.dropout,
                ).to(device)
                model.load_state_dict(torch.load(runs_dir(tag) / f"{arm}_seed{seed}.pt"))
                stats = summarise(predict(model, split, arm), split, leads_h)
                stacked.append(stats["scaled_crps"])

                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "split": split_name,
                        "macro_scaled_crps": macro(stats["scaled_crps"], split.farm_idx, n_farms),
                        "macro_crps_mw": macro(stats["mw_crps"], split.farm_idx, n_farms),
                        "mean_rmse_mw": stats["mean_rmse_mw"],
                        "median_mae_mw": stats["median_mae_mw"],
                        "coverage_90": stats["coverage_90"],
                        "width_90_mw": stats["width_90_mw"],
                        "n_examples": len(split),
                    }
                )
                for f, farm in enumerate(farms):
                    sel = split.farm_idx == f
                    if sel.any():
                        by_farm.append(
                            {
                                "arm": arm, "seed": seed, "split": split_name, "farm": farm,
                                "scaled_crps": float(stats["scaled_crps"][sel].mean()),
                                "crps_mw": float(stats["mw_crps"][sel].mean()),
                                "n_examples": int(sel.sum()),
                            }
                        )
                for name, (lo_h, hi_h) in LEAD_GROUPS.items():
                    mask = (leads_h >= lo_h) & (leads_h <= hi_h)
                    by_lead.append(
                        {
                            "arm": arm, "seed": seed, "split": split_name, "lead_group": name,
                            "scaled_crps": macro(stats["scaled_crps"], split.farm_idx, n_farms, mask),
                        }
                    )

            # instruction section 24: the bootstrap consumes the 3-seed mean per example
            losses[(arm, split_name)] = torch.stack(stacked).mean(0).cpu().numpy()

    pd.DataFrame(rows).to_csv(out_dir(tag) / "metrics.csv", index=False)
    pd.DataFrame(by_farm).to_csv(out_dir(tag) / "metrics_by_farm.csv", index=False)
    pd.DataFrame(by_lead).to_csv(out_dir(tag) / "metrics_by_lead.csv", index=False)

    store = {f"{arm}__{split}": loss for (arm, split), loss in losses.items()}
    for split_name in splits_to_score:
        if len(splits[split_name]):
            store[f"meta__{split_name}__farm_idx"] = splits[split_name].farm_idx.cpu().numpy()
            store[f"meta__{split_name}__origin"] = np.array(
                [e.split("|")[1] for e in splits[split_name].example_id]
            )
            store[f"meta__{split_name}__example_id"] = splits[split_name].example_id
    np.savez_compressed(out_dir(tag) / "per_example_losses.npz", **store)

    (out_dir(tag) / "evaluation_keys.json").write_text(
        json.dumps({k: len(v) for k, v in key_check.items()}, indent=2)
    )
    print(pd.DataFrame(rows).groupby(["split", "arm"])["macro_scaled_crps"].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
