"""One training schedule, shared by all three arms.

R, M and O see exactly the same updates: the same channels, the same forecast
origins, the same report interval and the same operation, in the same order.  The
schedule is a pure function of (dataset, model_seed) and its SHA is recorded, so a
difference between arms can never be a difference in what they were shown.

Checkpoint selection uses validation loss over the *training* resolutions only
(r = 2, 4, 8).  Letting r = 3 or 6 into model selection would quietly turn the
unseen-resolution evaluation into a seen one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import model as M
from .data import BaseGrid, eligible_origins
from .operators import (
    FORECAST_BINS,
    HISTORY_BINS,
    base_bin_bounds,
    centre_basis,
    future_query_basis,
    observe,
    reconstruction_operator,
    token_features,
)

TRAIN_R = (2, 4, 8)
TRAIN_OPS = ("END_BIN", "INTERVAL_MEAN")
# The deterministic (r, op) cycle from the spec: every resolution and both
# operations appear equally often, in a fixed order.
CONDITION_CYCLE = tuple((r, op) for r in TRAIN_R for op in TRAIN_OPS)

BATCH_WINDOWS = 64
SCHEDULE_SEED = 2026090621
LAMBDA_CANDIDATES = (1e-4, 1e-2, 1.0, 100.0)


@dataclass
class Tier:
    name: str
    max_updates: int
    warmup: int
    validation_every: int


TIERS = {
    "FULL": Tier("FULL", 5000, 250, 500),
    "COMPACT": Tier("COMPACT", 3000, 150, 300),
    "SCREEN": Tier("SCREEN", 2000, 100, 250),
}


@dataclass
class Schedule:
    dataset: str
    model_seed: int
    n_updates: int
    conditions: np.ndarray      # (n_updates,) index into CONDITION_CYCLE
    channels: np.ndarray        # (n_updates, BATCH_WINDOWS)
    origins: np.ndarray         # (n_updates, BATCH_WINDOWS)
    sha: str = field(default="")

    def condition(self, u: int) -> tuple[int, str]:
        return CONDITION_CYCLE[u % len(CONDITION_CYCLE)]


def build_schedule(grid: BaseGrid, splits: dict, model_seed: int, n_updates: int) -> Schedule:
    per_channel = [
        eligible_origins(grid, splits, "train", j) for j in range(len(grid.channels))
    ]
    if any(o.size == 0 for o in per_channel):
        raise RuntimeError("a channel has no eligible training origin")

    # A stable digest of the dataset name -- Python's hash() is salted per process.
    tag = int.from_bytes(hashlib.sha256(grid.dataset.encode()).digest()[:4], "big")
    rng = np.random.default_rng([SCHEDULE_SEED, model_seed, tag])
    ch = rng.integers(0, len(grid.channels), size=(n_updates, BATCH_WINDOWS))
    org = np.empty_like(ch)
    for j, pool in enumerate(per_channel):
        m = ch == j
        org[m] = pool[rng.integers(0, pool.size, size=int(m.sum()))]

    cond = np.arange(n_updates) % len(CONDITION_CYCLE)
    sch = Schedule(grid.dataset, model_seed, n_updates, cond, ch, org)
    h = hashlib.sha256()
    for arr in (cond, ch, org):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(f"{grid.dataset}|{model_seed}|{n_updates}".encode())
    sch.sha = h.hexdigest()
    return sch


# ------------------------------------------------------------------ batch assembly


class ArmInputs:
    """Turns normalised base-grid windows into whatever one arm consumes.

    Every arm starts from the identical observed values -- the same operator applied
    to the same history.  R then reconstructs, M and O read the tokens directly.
    """

    def __init__(self, arm: str, device: torch.device, lam: float | None = None) -> None:
        self.arm = arm
        self.device = device
        self.lam = lam
        self._tok: dict[tuple[int, str], dict] = {}
        self._recon: dict[tuple[int, str], torch.Tensor] = {}
        self.query = torch.tensor(future_query_basis(), dtype=torch.float32, device=device)
        self.base_basis = torch.tensor(centre_basis(base_bin_bounds()), dtype=torch.float32,
                                       device=device)

    def _tokens(self, r: int, op: str) -> dict:
        key = (r, op)
        if key not in self._tok:
            f = token_features(r, op, self.arm)
            self._tok[key] = {
                "time_basis": torch.tensor(f["time_basis"], device=self.device),
                "width": torch.tensor(f["width"], device=self.device),
                "op_id": torch.tensor(f["op_id"], device=self.device),
            }
        return self._tok[key]

    def _recon_matrix(self, r: int, op: str) -> torch.Tensor:
        key = (r, op)
        if key not in self._recon:
            P = reconstruction_operator(r, op, self.lam)
            self._recon[key] = torch.tensor(P, dtype=torch.float32, device=self.device)
        return self._recon[key]

    def forward(self, net: torch.nn.Module, history: torch.Tensor, r: int, op: str) -> torch.Tensor:
        v = torch_observe(history, r, op)
        b = v.shape[0]
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        if self.arm == "R":
            x = v @ self._recon_matrix(r, op).T
            return net(x, self.base_basis.unsqueeze(0).expand(b, -1, -1), q)
        t = self._tokens(r, op)
        return net(
            v,
            t["time_basis"].unsqueeze(0).expand(b, -1, -1),
            t["width"].unsqueeze(0).expand(b, -1, -1),
            t["op_id"].unsqueeze(0).expand(b, -1),
            q,
        )


def torch_observe(history: torch.Tensor, r: int, op: str) -> torch.Tensor:
    """The observation operator on a torch batch.  Mirrors operators.observe."""
    n = HISTORY_BINS // r
    if op == "INTERVAL_MEAN":
        return history.reshape(history.shape[0], n, r).mean(dim=-1)
    if op == "END_BIN":
        return history[:, r - 1 :: r]
    raise ValueError(op)


class WindowStore:
    """Normalised base-grid values kept on the GPU so a batch is a gather, not a copy."""

    def __init__(self, grid: BaseGrid, mu: np.ndarray, sd: np.ndarray, device: torch.device) -> None:
        z = (grid.values - mu) / sd
        self.z = torch.tensor(np.nan_to_num(z, nan=0.0), dtype=torch.float32, device=device)
        self.device = device
        self.h_off = torch.arange(-HISTORY_BINS, 0, device=device)
        self.t_off = torch.arange(0, FORECAST_BINS, device=device)

    def batch(self, channels: np.ndarray, origins: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.as_tensor(channels, device=self.device, dtype=torch.long)
        o = torch.as_tensor(origins, device=self.device, dtype=torch.long)
        hi = o[:, None] + self.h_off[None, :]
        ti = o[:, None] + self.t_off[None, :]
        return self.z[hi, c[:, None]], self.z[ti, c[:, None]]


# ---------------------------------------------------------------------- validation


def validation_windows(grid: BaseGrid, splits: dict, stride: int = 72) -> tuple[np.ndarray, np.ndarray]:
    ch, org = [], []
    for j in range(len(grid.channels)):
        o = eligible_origins(grid, splits, "val", j, stride=stride)
        ch.append(np.full(o.size, j))
        org.append(o)
    return np.concatenate(ch), np.concatenate(org)


@torch.no_grad()
def validation_loss(net, inputs: ArmInputs, store: WindowStore, ch: np.ndarray, org: np.ndarray,
                    chunk: int = 512) -> dict:
    net.eval()
    tot = {"primary": 0.0, "mse10": 0.0, "mse60": 0.0}
    n = 0
    for r, op in CONDITION_CYCLE:                 # seen resolutions only
        for s in range(0, len(org), chunk):
            c, o = ch[s : s + chunk], org[s : s + chunk]
            hist, targ = store.batch(c, o)
            pred = inputs.forward(net, hist, r, op)
            loss, m10, m60 = M.primary_loss(pred, targ)
            w = len(o)
            tot["primary"] += float(loss) * w
            tot["mse10"] += float(m10) * w
            tot["mse60"] += float(m60) * w
            n += w
    net.train()
    return {k: v / n for k, v in tot.items()}


# ------------------------------------------------------------------------ the loop


def train_arm(arm: str, grid: BaseGrid, splits: dict, mu, sd, model_seed: int, tier: Tier,
              device: torch.device, lam: float | None = None, log=print,
              schedule: Schedule | None = None) -> dict:
    sch = schedule or build_schedule(grid, splits, model_seed, tier.max_updates)
    store = WindowStore(grid, mu, sd, device)
    val_ch, val_org = validation_windows(grid, splits)

    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    net = M.build(arm).to(device)
    inputs = ArmInputs(arm, device, lam=lam)

    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.01)

    def lr_at(u: int) -> float:
        if u < tier.warmup:
            return (u + 1) / tier.warmup
        p = (u - tier.warmup) / max(1, tier.max_updates - tier.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    curve, best = [], {"primary": float("inf"), "update": -1, "state": None}
    t0 = time.time()
    net.train()
    for u in range(tier.max_updates):
        r, op = sch.condition(u)
        hist, targ = store.batch(sch.channels[u], sch.origins[u])
        pred = inputs.forward(net, hist, r, op)
        loss, _, _ = M.primary_loss(pred, targ)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if (u + 1) % tier.validation_every == 0 or u + 1 == tier.max_updates:
            v = validation_loss(net, inputs, store, val_ch, val_org)
            curve.append({"update": u + 1, "train_loss": float(loss), **v})
            if v["primary"] < best["primary"]:
                best = {"primary": v["primary"], "update": u + 1,
                        "state": {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}}
            log(f"    [{arm} {grid.dataset} seed{model_seed}] u={u+1}/{tier.max_updates} "
                f"train={float(loss):.5f} val={v['primary']:.5f} "
                f"({time.time()-t0:.0f}s)")

    net.load_state_dict(best["state"])
    return {
        "arm": arm, "dataset": grid.dataset, "model_seed": model_seed,
        "tier": tier.name, "lambda": lam, "schedule_sha": sch.sha,
        "n_parameters": M.parameter_count(net),
        "best_update": best["update"], "best_val_primary": best["primary"],
        "final_val_primary": curve[-1]["primary"],
        "wall_seconds": time.time() - t0,
        "curve": curve, "net": net,
    }


def select_lambda(grid: BaseGrid, splits: dict, mu, sd, tier: Tier, device: torch.device,
                  model_seed: int, log=print) -> dict:
    """Pre-registered protocol: one reduced-budget R fit per candidate lambda at the
    first model seed, scored on validation only, one lambda frozen per dataset."""
    screen = Tier(f"{tier.name}-LAMBDA", max(200, tier.max_updates // 4),
                  max(50, tier.warmup // 4), max(100, tier.validation_every // 2))
    rows = []
    for lam in LAMBDA_CANDIDATES:
        log(f"  lambda screen {lam:g} ({screen.max_updates} updates)")
        out = train_arm("R", grid, splits, mu, sd, model_seed, screen, device, lam=lam, log=log)
        rows.append({"lambda": lam, "val_primary": out["best_val_primary"],
                     "best_update": out["best_update"], "wall_seconds": out["wall_seconds"]})
        del out
        torch.cuda.empty_cache()
    best = min(rows, key=lambda d: d["val_primary"])
    return {"dataset": grid.dataset, "screen_updates": screen.max_updates,
            "model_seed": model_seed, "candidates": rows, "selected_lambda": best["lambda"]}


def schedule_digest(sch: Schedule) -> dict:
    return {"dataset": sch.dataset, "model_seed": sch.model_seed,
            "n_updates": sch.n_updates, "sha256": sch.sha}


def dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
