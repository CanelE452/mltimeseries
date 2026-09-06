"""Train every arm on the shared panel and checkpoint on validation CRPS.

All arms read the same examples in the same order, with the same optimiser,
batch size, budget and seed. The only thing that changes between D and P is
where the member pooling happens; between the other arms, only what the weather
branch is given.

Loss and checkpoint metric are empirical CRPS on the per-farm scaled target.
The 2021 test split is touched once, by evaluate.py, after every fit is done.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from experiments.uncertain_covariate_path_pilot_v1.src.crps import crps_efficient, crps_loss
from experiments.uncertain_covariate_path_pilot_v1.src.models import (
    ParticleForecaster,
    count_parameters,
)
from experiments.uncertain_covariate_path_pilot_v1.src.path_breaking import break_paths
from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir, runs_dir

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data_external/ucp_path_pilot_v1/processed"
CACHE = ROOT / "data_external/ucp_path_pilot_v1/chronos_cache"

ARMS = ("H", "M", "S", "D", "P", "P_BROKEN", "MC")
SEEDS = (2026090601, 2026090602, 2026090603)
MC_TRAIN_MEMBERS = 8  # fixed before any result was looked at (instruction section 18)


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dropout: float = 0.1
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    n_particles: int = 32
    hidden: int = 64


@dataclass
class Split:
    base: torch.Tensor  # (N, T, n_base)
    weather: torch.Tensor  # (N, K, T, D)
    target: torch.Tensor  # (N, T) scaled
    target_mw: torch.Tensor  # (N, T) MW
    farm_idx: torch.Tensor  # (N,)
    scale: torch.Tensor  # (N,) per-example target std in MW
    example_id: np.ndarray

    def __len__(self):
        return len(self.example_id)


def load_split(name: str, norm: dict, device: str) -> Split:
    panel = np.load(PROCESSED / f"panel_{name}.npz", allow_pickle=False)
    chronos = np.load(CACHE / f"chronos_{name}.npz", allow_pickle=False)
    if list(chronos["example_id"]) != list(panel["example_id"]):
        raise RuntimeError(f"{name}: Chronos cache and panel disagree on examples")

    farm_idx = panel["farm_idx"]
    mean = norm["target_mean"][farm_idx][:, None]
    std = norm["target_std"][farm_idx][:, None]

    quantiles = (chronos["quantiles"] - mean[:, :, None]) / std[:, :, None]
    leads_h = np.asarray(chronos["leads_h"], np.float32)
    lead_scalar = leads_h / leads_h.max()
    lead_feature = np.broadcast_to(lead_scalar[None, :, None], (len(farm_idx), len(leads_h), 1))
    base = np.concatenate([quantiles, lead_feature, panel["calendar"]], axis=-1).astype(np.float32)

    weather = (panel["weather"] - norm["weather_mean"]) / norm["weather_std"]
    target_scaled = (panel["target"] - mean) / std

    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return Split(
        base=t(base),
        weather=t(weather.astype(np.float32)),
        target=t(target_scaled.astype(np.float32)),
        target_mw=t(panel["target"]),
        farm_idx=t(farm_idx, torch.long),
        scale=t(std[:, 0]),
        example_id=panel["example_id"],
    )


def apply_path_breaking(split: Split) -> Split:
    broken = break_paths(split.weather.cpu().numpy(), split.example_id)
    return Split(**{**asdict_shallow(split), "weather": torch.as_tensor(
        broken, dtype=split.weather.dtype, device=split.weather.device)})


def asdict_shallow(split: Split) -> dict:
    return {f: getattr(split, f) for f in Split.__dataclass_fields__}


def per_example_crps(model, split: Split, arm: str, batch: int = 512) -> torch.Tensor:
    """(N, T) empirical CRPS on the scaled target, evaluated once per example."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(split), batch):
            sl = slice(i, min(i + batch, len(split)))
            if arm == "MC":
                particles = mc_particles(model, split, sl)
            else:
                particles = model(split.base[sl], split.weather[sl], split.farm_idx[sl])
            out.append(crps_efficient(particles, split.target[sl]))
    return torch.cat(out)


def mc_particles(model, split: Split, sl: slice, members: list[int] | None = None) -> torch.Tensor:
    """Union the particles produced under each member path (instruction section 18).

    This is a mixture of the member-conditional predictive distributions, which is
    what F_mix = (1/K) sum_k F_k means. Quantiles are never averaged.
    """
    weather = split.weather[sl]
    idx = range(weather.shape[1]) if members is None else members
    pieces = []
    for k in idx:
        one = weather[:, k : k + 1]
        pieces.append(model(split.base[sl], one, split.farm_idx[sl]))
    return torch.cat(pieces, dim=-1)


def macro_scaled_crps(per_example: torch.Tensor, farm_idx: torch.Tensor, n_farms: int) -> float:
    """Mean over leads and examples inside a farm, then equal weight across farms."""
    per_farm = []
    for f in range(n_farms):
        sel = farm_idx == f
        if sel.any():
            per_farm.append(per_example[sel].mean())
    return float(torch.stack(per_farm).mean())


def train_arm(arm: str, seed: int, splits: dict, n_farms: int, cfg: Config, device: str,
              tag: str = "full") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    train, val = splits["train"], splits["val"]

    model = ParticleForecaster(
        arm=arm,
        n_base=train.base.shape[-1],
        n_weather=train.weather.shape[-1],
        n_farms=n_farms,
        hidden=cfg.hidden,
        n_particles=cfg.n_particles,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ckpt = runs_dir(tag) / f"{arm}_seed{seed}.pt"
    best, best_epoch, epochs_since_best = float("inf"), -1, 0
    history, started = [], time.time()
    n = len(train)

    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            weather = train.weather[idx]
            if arm == "MC":
                # one member per example per epoch, cycling the fixed subset, so the
                # gradient budget matches the other arms exactly
                member = (epoch + torch.arange(len(idx), device=device)) % MC_TRAIN_MEMBERS
                weather = weather[torch.arange(len(idx), device=device), member].unsqueeze(1)
            particles = model(train.base[idx], weather, train.farm_idx[idx])
            loss = crps_loss(particles, train.target[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.detach()) * len(idx)

        val_crps = macro_scaled_crps(
            per_example_crps(model, val, arm).mean(dim=1), val.farm_idx, n_farms
        )
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_scaled_crps": val_crps})
        if val_crps < best - 1e-6:
            best, best_epoch, epochs_since_best = val_crps, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                break

    model.load_state_dict(torch.load(ckpt))
    return {
        "arm": arm,
        "seed": seed,
        "n_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_macro_scaled_crps": best,
        "train_seconds": round(time.time() - started, 1),
        "checkpoint": str(ckpt),
        "history": history,
    }


def load_all(device: str, arms: tuple[str, ...]) -> tuple[dict, dict, int]:
    norm = dict(np.load(PROCESSED / "normalisation.npz", allow_pickle=False))
    splits = {s: load_split(s, norm, device) for s in ("train", "val", "test")}
    n_farms = len(norm["target_mean"])
    broken = {s: apply_path_breaking(v) for s, v in splits.items()} if "P_BROKEN" in arms else {}
    return splits, broken, n_farms


def main(arms=ARMS, seeds=SEEDS, cfg: Config | None = None, tag: str = "full") -> int:
    cfg = cfg or Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, broken, n_farms = load_all(device, tuple(arms))

    records = []
    for arm in arms:
        use = broken if arm == "P_BROKEN" else splits
        for seed in seeds:
            rec = train_arm(arm, seed, use, n_farms, cfg, device, tag)
            records.append(rec)
            print(
                f"{arm:9s} seed {seed} | params {rec['n_parameters']:,} | "
                f"best epoch {rec['best_epoch']} | val {rec['best_val_macro_scaled_crps']:.5f} | "
                f"{rec['train_seconds']:.0f}s",
                flush=True,
            )

    manifest = {
        "config": asdict(cfg),
        "arms": list(arms),
        "seeds": list(seeds),
        "mc_train_members": MC_TRAIN_MEMBERS,
        "device": device,
        "fits": records,
        "total_train_seconds": round(sum(r["train_seconds"] for r in records), 1),
    }
    (out_dir(tag) / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total training {manifest['total_train_seconds'] / 3600:.2f} GPU-hours", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
