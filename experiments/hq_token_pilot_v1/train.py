"""Training loop, checkpoint selection and the resource supervisor.

Contract source: 01_forecast_query_tokenization_CLI.txt sections 9, 11 and 19.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
import torch

from . import data as D
from . import model as M
from .evaluate import evaluate_split


# ---- section 11: resource supervisor ---------------------------------------


class ResourceGuard:
    """Samples RSS / available RAM / GPU allocation every 5 s.

    Two consecutive breaches raise the stop flag; the training loop then checkpoints at
    the next batch boundary and exits. This bounds THIS job only. It does not touch any
    other process and is not a guarantee about overall machine safety.
    """

    def __init__(self, interval: float = 5.0):
        import psutil

        self.psutil = psutil
        self.proc = psutil.Process()
        total_ram = psutil.virtual_memory().total
        self.rss_upper = min(10 * 2**30, int(total_ram * 0.35))
        self.avail_floor = max(4 * 2**30, int(total_ram * 0.20))
        self.gpu_upper = (
            int(torch.cuda.get_device_properties(0).total_memory * 0.80)
            if torch.cuda.is_available()
            else None
        )
        self.interval = interval
        self.stop_requested = False
        self.reason = None
        self.consecutive = 0
        self.peak = {"rss": 0, "gpu": 0, "avail_min": total_ram}
        self.commit_available = False
        self._thread = None
        self._running = False

    def _sample(self):
        rss = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except Exception:
                pass
        avail = self.psutil.virtual_memory().available
        gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.peak["rss"] = max(self.peak["rss"], rss)
        self.peak["gpu"] = max(self.peak["gpu"], gpu)
        self.peak["avail_min"] = min(self.peak["avail_min"], avail)

        breach = None
        if rss > self.rss_upper:
            breach = f"process tree RSS {rss/2**30:.2f} GiB > {self.rss_upper/2**30:.2f} GiB"
        elif avail < self.avail_floor:
            breach = f"available RAM {avail/2**30:.2f} GiB < {self.avail_floor/2**30:.2f} GiB"
        elif self.gpu_upper is not None and gpu > self.gpu_upper:
            breach = f"GPU allocated {gpu/2**30:.2f} GiB > {self.gpu_upper/2**30:.2f} GiB"

        if breach:
            self.consecutive += 1
            if self.consecutive >= 2:
                self.stop_requested = True
                self.reason = breach
        else:
            self.consecutive = 0

    def _loop(self):
        while self._running:
            try:
                self._sample()
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
        return False

    def report(self) -> dict:
        return {
            "rss_upper_bytes": self.rss_upper,
            "available_floor_bytes": self.avail_floor,
            "gpu_upper_bytes": self.gpu_upper,
            "peak_process_tree_rss_bytes": self.peak["rss"],
            "peak_gpu_allocated_bytes": self.peak["gpu"],
            "min_available_ram_bytes": self.peak["avail_min"],
            "system_commit_available": self.commit_available,
            "stopped_for_resource": self.stop_requested,
            "stop_reason": self.reason,
        }


# ---- learning-rate schedule (section 9) ------------------------------------


def lr_at(update: int, base_lr: float, warmup: int, total: int) -> float:
    if update < warmup:
        return base_lr * (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---- one fit ---------------------------------------------------------------


def fit(
    arm: str,
    dataset: D.Dataset,
    B: int,
    seed: int,
    cfg: dict,
    out_dir: str,
    device: str = "cuda",
    guard: ResourceGuard | None = None,
) -> dict:
    """Train one arm on one dataset at one token budget and one seed."""
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    max_updates = cfg["max_updates"]
    warmup = cfg["warmup"]
    val_every = cfg["validate_every"]
    batch = cfg["batch"]

    sched = D.make_schedule(dataset, seed, max_updates, batch)
    fake_h = (
        D.fake_horizon_train(seed, dataset.dataset_id, max_updates, batch)
        if arm == "R"
        else None
    )

    net = M.build(arm, B, seed).to(device)
    torch.save(net.state_dict(), os.path.join(out_dir, "initial.pt"))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    train_curve, val_curve = [], []
    best = {"mean_val_mse": float("inf"), "update": -1}
    microbatch = batch
    oom_retries = 0
    status = "COMPLETE"
    stop_note = None

    for u in range(max_updates):
        if guard is not None and guard.stop_requested:
            status = "BLOCKED_RESOURCE"
            stop_note = guard.reason
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(u, cfg["learning_rate"], warmup, max_updates)

        ch = sched["channels"][u]
        og = sched["origins"][u]
        h_true = int(sched["horizons"][u])

        try:
            net.train()
            opt.zero_grad(set_to_none=True)
            n_micro = batch // microbatch
            total_loss = 0.0
            for k in range(n_micro):
                sl = slice(k * microbatch, (k + 1) * microbatch)
                x = torch.from_numpy(dataset.inputs(og[sl], ch[sl])).to(device)
                y = torch.from_numpy(dataset.targets(og[sl], ch[sl])).to(device)
                ht = torch.full((x.shape[0],), float(h_true), device=device)
                hp = (
                    torch.from_numpy(fake_h[u][sl]).to(device).float()
                    if arm == "R"
                    else ht
                )
                loss = M.masked_mse(net(x, ht, hp), y, ht) / n_micro
                loss.backward()
                total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
        except torch.cuda.OutOfMemoryError as e:
            # section 11: one retry at half microbatch with matching accumulation
            if oom_retries >= 1 or microbatch < 2:
                status = "BLOCKED_RESOURCE"
                stop_note = f"second OOM: {e}"
                break
            oom_retries += 1
            microbatch //= 2
            torch.cuda.empty_cache()
            opt.zero_grad(set_to_none=True)
            continue

        train_curve.append({"update": u, "horizon": h_true, "loss": total_loss})

        if (u + 1) % val_every == 0 or (u + 1) == max_updates:
            per_h = {}
            for h in D.HORIZONS:
                res = evaluate_split(net, dataset, "validation", h, device, arm=arm, seed=seed)
                per_h[h] = res["mse"]
            mean_val = float(np.mean(list(per_h.values())))
            val_curve.append({"update": u + 1, "mse_96": per_h[96], "mse_336": per_h[336], "mean": mean_val})
            if mean_val < best["mean_val_mse"]:
                best = {"mean_val_mse": mean_val, "update": u + 1}
                torch.save(net.state_dict(), os.path.join(out_dir, "best.pt"))

    torch.save(net.state_dict(), os.path.join(out_dir, "final.pt"))
    if best["update"] < 0:                       # stopped before any validation
        torch.save(net.state_dict(), os.path.join(out_dir, "best.pt"))
        best = {"mean_val_mse": float("nan"), "update": -1}

    wall = time.time() - t_start
    report = {
        "arm": arm,
        "dataset_id": dataset.dataset_id,
        "B": B,
        "model_seed": seed,
        "status": status,
        "stop_note": stop_note,
        "updates_run": len(train_curve),
        "updates_planned": max_updates,
        "microbatch": microbatch,
        "oom_retries": oom_retries,
        "best_checkpoint_update": best["update"],
        "best_mean_val_mse": best["mean_val_mse"],
        "wall_seconds": wall,
        "schedule_sha256": sched["schedule_sha256"],
        "n_eligible_train_origins": sched["n_eligible_origins"],
        "parameters": net.parameter_report(),
        "collapsed_to_constant": _constant_collapse(net, dataset, device, arm, seed),
    }
    with open(os.path.join(out_dir, "train_curve.json"), "w") as f:
        json.dump({"train": train_curve, "validation": val_curve}, f)
    with open(os.path.join(out_dir, "fit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _constant_collapse(net, dataset, device, arm, seed) -> bool:
    """section 9: flag a model whose forecast barely moves across different inputs."""
    net.eval()
    origins = dataset.eval_origins("validation")[:8]
    if origins.size == 0:
        return False
    ch = np.zeros(origins.shape[0], dtype=np.int64)
    with torch.no_grad():
        x = torch.from_numpy(dataset.inputs(origins, ch)).to(device)
        h = torch.full((x.shape[0],), 96.0, device=device)
        y = net(x, h)[:, :96]
    return bool(y.std(dim=0).mean().item() < 1e-4)
