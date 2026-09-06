"""Frozen-key evaluation, mechanism diagnostics and the unit-contract control.

Every arm is scored on one and the same set of logical keys

    (dataset, split, origin, channel, operation, report_interval_r, model_arm, model_seed)

built from the base grid alone, so the key set cannot depend on which model is
being scored.  Per-key sums of squared and absolute error are written to
runs/ and everything downstream -- point estimates, bootstrap, diagnostics --
reads that one frozen array.
"""

from __future__ import annotations

import numpy as np
import torch

from . import model as M
from .data import BaseGrid, eligible_origins
from .operators import (
    FORECAST_BINS,
    OPS,
    centre_basis,
    observation_support,
)
from .train import ArmInputs, WindowStore, torch_observe

EVAL_R = (2, 3, 4, 6, 8, 12)
SEEN_R = (2, 4, 8)
INTERP_R = (3, 6)
EXTRAP_R = (12,)
ROLE = {2: "SEEN", 4: "SEEN", 8: "SEEN", 3: "UNSEEN_INTERPOLATION",
        6: "UNSEEN_INTERPOLATION", 12: "UNSEEN_EXTRAPOLATION"}

DTYPE = np.dtype([
    ("split", "U5"), ("origin", "i4"), ("channel", "i2"), ("operation", "U13"),
    ("r", "i2"), ("role", "U21"), ("arm", "U2"), ("seed", "i8"),
    ("SE10", "f8"), ("count10", "i4"), ("SE60", "f8"), ("count60", "i4"),
    ("AE10", "f8"), ("AE60", "f8"),
])


def frozen_keys(grid: BaseGrid, splits: dict, stride: int = 72) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The evaluation origins, per split, identical for every arm and every r."""
    out = {}
    for split in ("val", "test"):
        ch, org = [], []
        for j in range(len(grid.channels)):
            o = eligible_origins(grid, splits, split, j, stride=stride)
            ch.append(np.full(o.size, j, dtype=np.int64))
            org.append(o)
        out[split] = (np.concatenate(ch), np.concatenate(org))
    return out


@torch.no_grad()
def _errors(pred: torch.Tensor, targ: torch.Tensor) -> dict[str, np.ndarray]:
    e = pred - targ
    p60, t60 = M.hourly_mean(pred), M.hourly_mean(targ)
    e60 = p60 - t60
    return {
        "SE10": (e ** 2).sum(-1).cpu().numpy(),
        "AE10": e.abs().sum(-1).cpu().numpy(),
        "SE60": (e60 ** 2).sum(-1).cpu().numpy(),
        "AE60": e60.abs().sum(-1).cpu().numpy(),
    }


@torch.no_grad()
def evaluate_model(net, arm: str, grid: BaseGrid, splits: dict, mu, sd, seed: int,
                   device: torch.device, lam: float | None, keys: dict,
                   chunk: int = 512) -> np.ndarray:
    net.eval()
    store = WindowStore(grid, mu, sd, device)
    inputs = ArmInputs(arm, device, lam=lam)
    rows = []
    for split, (ch, org) in keys.items():
        for op in OPS:
            for r in EVAL_R:
                for s in range(0, len(org), chunk):
                    c, o = ch[s : s + chunk], org[s : s + chunk]
                    hist, targ = store.batch(c, o)
                    err = _errors(inputs.forward(net, hist, r, op), targ)
                    n = len(o)
                    block = np.empty(n, dtype=DTYPE)
                    block["split"] = split
                    block["origin"] = o
                    block["channel"] = c
                    block["operation"] = op
                    block["r"] = r
                    block["role"] = ROLE[r]
                    block["arm"] = arm
                    block["seed"] = seed
                    block["count10"] = FORECAST_BINS
                    block["count60"] = FORECAST_BINS // 6
                    for k, v in err.items():
                        block[k] = v
                    rows.append(block)
    return np.concatenate(rows)


# ------------------------------------------------------------------- diagnostics


class _WrongSupport(ArmInputs):
    """O forced to describe every observation as if it had no extent: the centre
    basis in place of the integrated one.  Inference only, no retraining.

    For M this variant is the identity -- M already uses the centre basis -- so it
    is reported as not applicable rather than run.
    """

    def _tokens(self, r: int, op: str) -> dict:
        t = super()._tokens(r, op)
        if not t.get("corrupted"):
            t["time_basis"] = torch.tensor(
                centre_basis(observation_support(r, op)).astype(np.float32), device=self.device
            )
            t["corrupted"] = True
        return t


class _WidthMismatch(ArmInputs):
    """The width metadata replaced by the width of a different report interval,
    values / time basis / operation untouched.

    The spec asks for a within-batch permutation of the width metadata.  Under this
    batching every token in a batch shares one report interval and therefore one
    width, so a within-batch permutation is provably the identity; that fact is
    recorded in mechanism_diagnostics.json.  Mismatching the width against a
    different resolution is the same corruption with an effect: the arm is told the
    observations are wider or narrower than they are.
    """

    MISMATCH = {2: 12, 3: 8, 4: 6, 6: 4, 8: 3, 12: 2}

    def _tokens(self, r: int, op: str) -> dict:
        t = super()._tokens(r, op)
        if not t.get("corrupted"):
            wrong = observation_support(self.MISMATCH[r], op)
            t["width"] = torch.full_like(t["width"], float(wrong[0, 1] - wrong[0, 0]) / 288.0)
            t["corrupted"] = True
        return t


@torch.no_grad()
def diagnostic(net, arm: str, variant: str, grid: BaseGrid, splits: dict, mu, sd,
               device: torch.device, keys: dict, lam: float | None = None,
               chunk: int = 512) -> dict:
    """Mean primary loss per (operation, r) under a corrupted view of the metadata."""
    net.eval()
    store = WindowStore(grid, mu, sd, device)
    cls = {"BASE": ArmInputs, "WRONG_SUPPORT": _WrongSupport, "WIDTH_MISMATCH": _WidthMismatch}[variant]
    inputs = cls(arm, device, lam=lam)
    ch, org = keys["test"]
    out = {}
    for op in OPS:
        for r in EVAL_R:
            tot, n = 0.0, 0
            for s in range(0, len(org), chunk):
                hist, targ = store.batch(ch[s : s + chunk], org[s : s + chunk])
                loss, _, _ = M.primary_loss(inputs.forward(net, hist, r, op), targ)
                tot += float(loss) * len(org[s : s + chunk])
                n += len(org[s : s + chunk])
            out[f"{op}|r{r}"] = tot / n
    return out


def within_batch_width_permutation_is_identity() -> dict:
    """Evidence for the note above: every token of one batch carries one width."""
    rows = {}
    for r in EVAL_R:
        for op in OPS:
            sup = observation_support(r, op)
            w = np.unique(sup[:, 1] - sup[:, 0])
            rows[f"{op}|r{r}"] = {"distinct_widths_in_batch": int(w.size), "width_bins": w.tolist()}
    return {"claim": "a within-batch permutation of width metadata is the identity",
            "reason": "one batch holds one report interval, so all its tokens share one support width",
            "per_condition": rows}


# ------------------------------------------------------ SUM / MEAN unit contract


@torch.no_grad()
def sum_mean_invariance(net, arm: str, grid: BaseGrid, splits: dict, mu, sd,
                        device: torch.device, keys: dict, lam: float | None,
                        chunk: int = 256, max_windows: int = 2048) -> dict:
    """Section 14 / 37.  The discrete total over a report interval is formed by an
    independent sum over the base bins, then converted back with the operator's own
    report width.  It must reproduce the INTERVAL_MEAN observation exactly, and the
    arm must forecast identically from it.
    """
    net.eval()
    store = WindowStore(grid, mu, sd, device)
    inputs = ArmInputs(arm, device, lam=lam)
    ch, org = keys["test"]
    worst_repr = worst_pred = pred_scale = 0.0
    for r in EVAL_R:
        for s in range(0, min(len(org), max_windows), chunk):
            hist, _ = store.batch(ch[s : s + chunk], org[s : s + chunk])
            mean = torch_observe(hist, r, "INTERVAL_MEAN")
            # independent path: sum the base bins of each report interval
            total = hist.reshape(hist.shape[0], hist.shape[1] // r, r).sum(dim=-1)
            back = total / r
            worst_repr = max(worst_repr, float((mean - back).abs().max()))
            a = _forward_from_values(inputs, net, mean, r, "INTERVAL_MEAN")
            b = _forward_from_values(inputs, net, back, r, "INTERVAL_MEAN")
            worst_pred = max(worst_pred, float((a - b).abs().max()))
            pred_scale = max(pred_scale, float(a.abs().max()))
    return {"max_abs_representation_difference": worst_repr,
            "max_abs_prediction_difference": worst_pred,
            "prediction_scale": pred_scale,
            "relative_prediction_difference": worst_pred / max(pred_scale, 1e-12)}


def _forward_from_values(inputs: ArmInputs, net, values: torch.Tensor, r: int, op: str) -> torch.Tensor:
    b = values.shape[0]
    q = inputs.query.unsqueeze(0).expand(b, -1, -1)
    if inputs.arm == "R":
        x = values @ inputs._recon_matrix(r, op).T
        return net(x, inputs.base_basis.unsqueeze(0).expand(b, -1, -1), q)
    t = inputs._tokens(r, op)
    return net(values, t["time_basis"].unsqueeze(0).expand(b, -1, -1),
               t["width"].unsqueeze(0).expand(b, -1, -1),
               t["op_id"].unsqueeze(0).expand(b, -1), q)


# ------------------------------------------------------------- learning anchors


def seasonal_naive(grid: BaseGrid, splits: dict, mu, sd, keys: dict) -> dict:
    """Persistence and a one-day seasonal naive on the same frozen test keys, from
    the base grid only.  If every trained arm loses to these, nothing is decided."""
    ch, org = keys["test"]
    z = (grid.values - mu) / sd
    out = {}
    for name, lag in (("persistence", 0), ("seasonal_naive_1d", 144)):
        se10 = se60 = 0.0
        n = 0
        for c, o in zip(ch, org):
            targ = z[o : o + FORECAST_BINS, c]
            if lag == 0:
                pred = np.full(FORECAST_BINS, z[o - 1, c])
            else:
                if o - lag < 0 or not grid.valid[o - lag : o - lag + FORECAST_BINS, c].all():
                    continue
                pred = z[o - lag : o - lag + FORECAST_BINS, c]
            e = pred - targ
            se10 += float((e ** 2).mean())
            e60 = e.reshape(12, 6).mean(1)
            se60 += float((e60 ** 2).mean())
            n += 1
        out[name] = {"mse10": se10 / n, "mse60": se60 / n,
                     "primary": 0.5 * se10 / n + 0.5 * se60 / n, "n_keys": n}
    return out
