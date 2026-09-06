"""Keyed evaluation, metrics, paired block bootstrap, diagnostics and latency.

Contract source: 01_forecast_query_tokenization_CLI.txt sections 12, 13 and 15.

Every number downstream is derived from the same keyed error arrays produced here, so
the point estimate and the confidence interval can never be computed on different grids.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import data as D
from . import model as M

CHUNK = 4096


def _eval_pairs(dataset: D.Dataset, split: str, phase_offset: int = 0):
    """All (origin, channel) pairs of the frozen grid, in a fixed key order.

    phase_offset shifts every origin by a constant number of steps. The pre-registered
    grid is offset 0; a nonzero offset is only used for the robustness re-read, because
    stride 96 lands on exactly one time of day for ETTm2 (15 min) and electricity (1 h).
    """
    origins = dataset.eval_origins(split)
    if phase_offset:
        s, e = dataset.splits[split]
        origins = origins + phase_offset
        origins = origins[(origins >= s) & (origins <= e - D.H_MAX)]
    channels = np.arange(dataset.n_channels, dtype=np.int64)
    o = np.repeat(origins, channels.shape[0])
    c = np.tile(channels, origins.shape[0])
    return o, c


def seasonal_naive(dataset: D.Dataset, split: str, horizon: int, period: int) -> dict:
    """Anchor baseline: repeat the last full seasonal cycle of the input history.

    Computed in the same train-only standardized space and on the same frozen grid, so
    it can be read next to the arms without any rescaling.
    """
    o_all, c_all = _eval_pairs(dataset, split)
    n = o_all.shape[0]
    se = np.zeros(n, dtype=np.float64)
    ae = np.zeros(n, dtype=np.float64)
    lead = np.arange(horizon)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        o, c = o_all[s:e], c_all[s:e]
        x = dataset.inputs(o, c)
        y = dataset.targets(o, c)[:, :horizon]
        pred = x[:, D.L - period + (lead % period)]
        diff = pred - y
        se[s:e] = (diff**2).sum(axis=1)
        ae[s:e] = np.abs(diff).sum(axis=1)
    total = float(horizon) * n
    return {"mse": float(se.sum() / total), "mae": float(ae.sum() / total), "period": period}


@torch.no_grad()
def evaluate_split(
    net: M.PilotModel,
    dataset: D.Dataset,
    split: str,
    horizon: int,
    device: str,
    arm: str,
    seed: int,
    keep_keys: bool = False,
    keep_yhat: bool = False,
    query_horizon_override: int | None = None,
    phase_offset: int = 0,
):
    """Errors on one split at one horizon.

    query_horizon_override feeds a different horizon to the pooling query while the
    decoder still receives the true horizon (section 13 diagnostic A).
    """
    net.eval()
    o_all, c_all = _eval_pairs(dataset, split, phase_offset)
    n = o_all.shape[0]
    se = np.zeros(n, dtype=np.float64)
    ae = np.zeros(n, dtype=np.float64)
    yhat_store = [] if keep_yhat else None

    h_pool_all = None
    if arm == "R":
        h_pool_all = D.fake_horizon_eval(
            dataset.dataset_id, split, o_all, c_all, np.full(n, horizon, dtype=np.int64)
        )

    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        o, c = o_all[s:e], c_all[s:e]
        x = torch.from_numpy(dataset.inputs(o, c)).to(device)
        y = torch.from_numpy(dataset.targets(o, c)).to(device)[:, :horizon]
        ht = torch.full((x.shape[0],), float(horizon), device=device)
        if query_horizon_override is not None:
            hp = torch.full((x.shape[0],), float(query_horizon_override), device=device)
        elif arm == "R":
            hp = torch.from_numpy(h_pool_all[s:e]).to(device).float()
        else:
            hp = ht
        pred = net(x, ht, hp)[:, :horizon]
        diff = pred - y
        se[s:e] = (diff**2).sum(dim=1).double().cpu().numpy()
        ae[s:e] = diff.abs().sum(dim=1).double().cpu().numpy()
        if keep_yhat:
            yhat_store.append(pred.float().cpu().numpy())

    total = float(horizon) * n
    out = {
        "mse": float(se.sum() / total),
        "mae": float(ae.sum() / total),
        "n_keys": int(n),
        "lead_times": int(horizon),
    }
    if keep_keys:
        out.update(
            {
                "origin": o_all,
                "channel": c_all,
                "se_sum": se,
                "ae_sum": ae,
                "count": np.full(n, horizon, dtype=np.int64),
                "dataset_id": dataset.dataset_id,
                "split": split,
                "horizon": horizon,
                "arm": arm,
                "model_seed": seed,
            }
        )
    if keep_yhat:
        out["yhat"] = np.concatenate(yhat_store, axis=0)
    return out


# ---- aggregation (section 12) ----------------------------------------------


def key_frame(records: list) -> dict:
    """Stack keyed records into arrays indexed by (arm, dataset, horizon, seed)."""
    table = {}
    for r in records:
        k = (r["arm"], r["dataset_id"], r["horizon"], r["model_seed"])
        table[k] = {
            "origin": r["origin"],
            "channel": r["channel"],
            "se_sum": r["se_sum"],
            "ae_sum": r["ae_sum"],
            "count": r["count"],
        }
    return table


def seed_averaged(table: dict, arm: str, dataset_id: str, horizon: int, seeds: list):
    """Per-key error averaged over model seeds first, as section 12 requires."""
    parts = [table[(arm, dataset_id, horizon, s)] for s in seeds if (arm, dataset_id, horizon, s) in table]
    if not parts:
        return None
    ref = parts[0]
    for p in parts[1:]:
        if not (np.array_equal(p["origin"], ref["origin"]) and np.array_equal(p["channel"], ref["channel"])):
            raise AssertionError("key order differs between seeds; refusing to join by row order")
    se = np.mean([p["se_sum"] for p in parts], axis=0)
    ae = np.mean([p["ae_sum"] for p in parts], axis=0)
    return {
        "origin": ref["origin"],
        "channel": ref["channel"],
        "se_sum": se,
        "ae_sum": ae,
        "count": ref["count"],
        "n_seeds": len(parts),
    }


def cell_error(agg: dict) -> float:
    return float(agg["se_sum"].sum() / agg["count"].sum())


def relative_improvement(e_treat: float, e_ref: float):
    """RI = 100 * (1 - E_treat / E_ref); NULL when the denominator vanishes."""
    if e_ref == 0.0 or not np.isfinite(e_ref):
        return None, "ZERO_BASELINE_LOSS"
    return 100.0 * (1.0 - e_treat / e_ref), None


def macro_ri(cells: dict):
    """Equal weight over the six dataset x horizon cells, never pooled."""
    vals = [v for v in cells.values() if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


# ---- paired circular moving block bootstrap (section 12) -------------------

BLOCK_ORIGINS = 15          # ceil((1024 + 336) / 96)


def bootstrap_macro(
    table: dict,
    treat: str,
    ref: str,
    datasets: list,
    horizons: list,
    seeds: list,
    draws: int = 1000,
    seed: int = 2026090699,
):
    """Resample chronological origins in circular blocks, identically for both arms.

    All channels, horizons and model seeds belonging to a drawn origin travel together.
    Model seeds are NOT resampled, so the interval is a time-sample interval conditional
    on the two observed seeds.
    """
    rng = np.random.default_rng(seed)
    prepared = {}
    flags = []
    for d in datasets:
        origins = None
        per = {}
        for h in horizons:
            a_t = seed_averaged(table, treat, d, h, seeds)
            a_r = seed_averaged(table, ref, d, h, seeds)
            if a_t is None or a_r is None:
                per = None
                break
            uo = np.unique(a_t["origin"])
            if origins is None:
                origins = uo
            # group errors by origin so a drawn origin brings all its channels
            idx = {o: np.nonzero(a_t["origin"] == o)[0] for o in uo}
            per[h] = {
                "origins": uo,
                "t_se": np.array([a_t["se_sum"][idx[o]].sum() for o in uo]),
                "r_se": np.array([a_r["se_sum"][idx[o]].sum() for o in uo]),
                "cnt": np.array([a_t["count"][idx[o]].sum() for o in uo], dtype=np.float64),
            }
        if per is None:
            continue
        n_blocks = int(np.ceil(len(origins) / BLOCK_ORIGINS))
        if n_blocks < 5:
            flags.append(f"LOW_EFFECTIVE_TIME_BLOCKS:{d}:{n_blocks}")
        prepared[d] = {"per": per, "n_origins": len(origins), "n_blocks": n_blocks}

    if not prepared:
        return {"draws": 0, "flags": ["NO_PAIRED_CELLS"], "macro": None}

    macros = np.full(draws, np.nan)
    for b in range(draws):
        cells = {}
        for d, pack in prepared.items():
            n = pack["n_origins"]
            n_blocks = pack["n_blocks"]
            starts = rng.integers(0, n, size=n_blocks)
            sel = (starts[:, None] + np.arange(BLOCK_ORIGINS)[None, :]).ravel() % n
            sel = sel[:n]
            for h, arrs in pack["per"].items():
                t = arrs["t_se"][sel].sum() / arrs["cnt"][sel].sum()
                r = arrs["r_se"][sel].sum() / arrs["cnt"][sel].sum()
                ri, _ = relative_improvement(t, r)
                cells[(d, h)] = ri
        m = macro_ri(cells)
        if m is not None:
            macros[b] = m

    finite = macros[np.isfinite(macros)]
    if finite.size == 0:
        return {"draws": draws, "flags": flags + ["ALL_DRAWS_NULL"], "macro": None}
    return {
        "draws": draws,
        "bootstrap_seed": seed,
        "block_origins": BLOCK_ORIGINS,
        "effective_blocks": {d: p["n_blocks"] for d, p in prepared.items()},
        "macro_mean": float(finite.mean()),
        "lower95": float(np.percentile(finite, 2.5)),
        "upper95": float(np.percentile(finite, 97.5)),
        "flags": flags,
    }


def bootstrap_cell(
    t_se, r_se, cnt, origins, draws: int = 1000, seed: int = 2026090699
):
    """Same resampling as bootstrap_macro, applied to a single dataset x horizon cell.

    Takes per-origin error sums that were already grouped, so the interval and the point
    estimate necessarily come from the same arrays.
    """
    rng = np.random.default_rng(seed)
    n = origins.shape[0]
    n_blocks = int(np.ceil(n / BLOCK_ORIGINS))
    vals = np.empty(draws)
    for b in range(draws):
        starts = rng.integers(0, n, size=n_blocks)
        sel = (starts[:, None] + np.arange(BLOCK_ORIGINS)[None, :]).ravel() % n
        sel = sel[:n]
        t = t_se[sel].sum() / cnt[sel].sum()
        r = r_se[sel].sum() / cnt[sel].sum()
        vals[b] = 100.0 * (1.0 - t / r) if r else np.nan
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None
    return {
        "lower95": float(np.percentile(finite, 2.5)),
        "upper95": float(np.percentile(finite, 97.5)),
        "n_origins": int(n),
        "n_blocks": n_blocks,
    }


# ---- section 13 diagnostics ------------------------------------------------


@torch.no_grad()
def query_swap_diagnostic(net, dataset, device, arm, seed):
    """A: keep the decoder on the true horizon, feed the pooling query the other one."""
    out = {}
    for h in D.HORIZONS:
        other = 336 if h == 96 else 96
        base = evaluate_split(net, dataset, "test", h, device, arm=arm, seed=seed)
        swap = evaluate_split(
            net, dataset, "test", h, device, arm=arm, seed=seed, query_horizon_override=other
        )
        out[str(h)] = {
            "mse_true_query": base["mse"],
            "mse_swapped_query": swap["mse"],
            "delta_pct": 100.0 * (swap["mse"] / base["mse"] - 1.0) if base["mse"] else None,
            "swapped_to": other,
        }
    return out


@torch.no_grad()
def weight_profile(net, dataset, device, n_windows: int = 256):
    """B: pooling weights and token centers at H=96 vs H=336 on identical inputs."""
    if net.arm == "DENSE":
        return None
    origins = dataset.eval_origins("test")
    if origins.size == 0:
        return None
    o = np.repeat(origins[: max(1, n_windows // dataset.n_channels) + 1], dataset.n_channels)
    c = np.tile(np.arange(dataset.n_channels), o.shape[0] // dataset.n_channels)
    o, c = o[:n_windows], c[:n_windows]
    x = torch.from_numpy(dataset.inputs(o, c)).to(device)
    res = {}
    for h in D.HORIZONS:
        hh = torch.full((x.shape[0],), float(h), device=device)
        _, w = net(x, hh, hh, return_weights=True)
        res[h] = (w, net.weighted_centers(w))
    w96, c96 = res[96]
    w336, c336 = res[336]
    return {
        "n_windows": int(x.shape[0]),
        "mean_abs_weight_diff": float((w96 - w336).abs().mean()),
        "mean_abs_center_shift_samples": float((c96 - c336).abs().mean()),
        "max_abs_center_shift_samples": float((c96 - c336).abs().max()),
        "weights_h96_mean": w96.mean(dim=0).cpu().numpy().tolist(),
        "weights_h336_mean": w336.mean(dim=0).cpu().numpy().tolist(),
        "centers_h96_mean": c96.mean(dim=0).cpu().numpy().tolist(),
        "centers_h336_mean": c336.mean(dim=0).cpu().numpy().tolist(),
        "group_bounds": net.group_bounds(),
    }


# ---- section 15 efficiency -------------------------------------------------


@torch.no_grad()
def measure_latency(net, device: str, batch_sizes=(1, 64), warmup: int = 20, timed: int = 100):
    """Full pipeline latency: tokenizer + encoder + unmerge + decoder.

    Model-call time and end-to-end time including the host-to-device copy are reported
    separately, because only the second is what a user would actually wait for.
    """
    net.eval()
    out = {}
    for bs in batch_sizes:
        host = np.random.randn(bs, M.L).astype(np.float32)
        x = torch.from_numpy(host).to(device)
        h = torch.full((bs,), 336.0, device=device)
        for _ in range(warmup):
            net(x, h)
        if device == "cuda":
            torch.cuda.synchronize()

        model_ms, e2e_ms = [], []
        for _ in range(timed):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(x, h)
            if device == "cuda":
                torch.cuda.synchronize()
            model_ms.append((time.perf_counter() - t0) * 1e3)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            xx = torch.from_numpy(host).to(device)
            hh = torch.full((bs,), 336.0, device=device)
            net(xx, hh)
            if device == "cuda":
                torch.cuda.synchronize()
            e2e_ms.append((time.perf_counter() - t0) * 1e3)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            net(x, h)
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = None
        out[f"batch{bs}"] = {
            "model_call_median_ms": float(np.median(model_ms)),
            "model_call_p90_ms": float(np.percentile(model_ms, 90)),
            "end_to_end_median_ms": float(np.median(e2e_ms)),
            "end_to_end_p90_ms": float(np.percentile(e2e_ms, 90)),
            "throughput_windows_per_s": float(bs / (np.median(model_ms) / 1e3)),
            "peak_allocated_bytes": peak,
        }
    out["parameters"] = net.parameter_report()
    out["tokens_encoded"] = net.B
    out["tokens_input"] = M.N
    return out
