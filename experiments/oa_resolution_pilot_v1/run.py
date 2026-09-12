"""OA-RESOLUTION-PILOT-v1 driver.

One process from the frozen spec to the verdict.  It does not stop at "training
finished": the contract tests, the runtime tier, the lambda screen, the twelve core
fits, the frozen-key evaluation, the bootstrap, the mechanism diagnostics, the unit
contract, the learning anchors and the verdict all run inside a single invocation,
in the order the spec fixes.

  python -m experiments.oa_resolution_pilot_v1.run
  python -m experiments.oa_resolution_pilot_v1.run --dry           # tiny end-to-end
  python -m experiments.oa_resolution_pilot_v1.run --verdict-only  # rescore only

The FlowState reference lives in references.py because the official package pins an
older Python and transformers than this project uses.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from . import evaluate as E
from . import report as RPT
from . import train as T

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "oa_resolution_pilot_v1"
RUNS = REPO / "runs" / "oa_resolution_pilot_v1"
ERRORS = RUNS / "errors"

MODEL_SEEDS = (2026090601, 2026090602)
ARMS = ("R", "M", "O")
WALL_CAP_SECONDS = 8 * 3600

PERIODS = {
    "jena": ("2023-01-01", "2025-01-01"),
    "uci": ("2007-01-01", "2009-01-01"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(name: str, obj) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def write_tables(cells) -> Path:
    """The three result tables, kept apart on purpose: one macro must never mix
    seen, unseen-interpolation and unseen-extrapolation resolutions."""
    sections = [
        ("primary_unseen_interpolation", RPT.markdown_table(cells, RPT.INTERP_R)),
        ("seen", RPT.markdown_table(cells, RPT.SEEN_R)),
        ("unseen_extrapolation", RPT.markdown_table(cells, RPT.EXTRAP_R)),
    ]
    body = "\n\n".join(f"## {name}\n\n{table}" for name, table in sections)
    p = RESULTS / "tables.md"
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------- resource guard


class ResourceGuard(threading.Thread):
    """Section 47.  Samples every 5 s and raises the stop flag after two
    consecutive breaches so a fit can be checkpointed rather than killed."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        import psutil

        self.psutil = psutil
        self.proc = psutil.Process()
        total_ram = psutil.virtual_memory().total
        self.rss_cap = min(10 * 2**30, total_ram * 0.35)
        self.avail_floor = max(4 * 2**30, total_ram * 0.20)
        self.gpu_cap = torch.cuda.get_device_properties(0).total_memory * 0.80 \
            if torch.cuda.is_available() else float("inf")
        self.breaches = 0
        self.stop_flag = False
        self.peak = {"rss": 0, "gpu": 0, "avail_min": total_ram}
        self._run = True

    def run(self) -> None:
        while self._run:
            try:
                rss = sum(p.memory_info().rss for p in [self.proc] + self.proc.children(True))
                avail = self.psutil.virtual_memory().available
                gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                self.peak["rss"] = max(self.peak["rss"], rss)
                self.peak["gpu"] = max(self.peak["gpu"], gpu)
                self.peak["avail_min"] = min(self.peak["avail_min"], avail)
                bad = rss > self.rss_cap or avail < self.avail_floor or gpu > self.gpu_cap
                self.breaches = self.breaches + 1 if bad else 0
                if self.breaches >= 2:
                    self.stop_flag = True
            except Exception:
                pass
            time.sleep(5)

    def close(self) -> dict:
        self._run = False
        return {
            "rss_cap_bytes": int(self.rss_cap), "available_floor_bytes": int(self.avail_floor),
            "gpu_allocated_cap_bytes": int(self.gpu_cap),
            "peak_process_tree_rss_bytes": int(self.peak["rss"]),
            "peak_gpu_allocated_bytes": int(self.peak["gpu"]),
            "min_available_ram_bytes": int(self.peak["avail_min"]),
            "blocked_resource": bool(self.stop_flag),
        }


# ------------------------------------------------------------------- provenance


def environment_json() -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": sh("git rev-parse HEAD"),
        "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
        "git_status_dirty": bool(sh("git status --porcelain")),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def data_manifest(grids: dict, splits: dict, scaling: dict) -> dict:
    src = json.loads((REPO / "data" / "oa_resolution_pilot_v1_source_hashes.json").read_text())
    out = {"source_archives": src["archives"], "source_files": src["files"], "datasets": {}}
    for ds, g in grids.items():
        sp = splits[ds]
        mu, sd = scaling[ds]
        out["datasets"][ds] = {
            "period_start": str(g.period_start), "period_end": str(g.period_end),
            "n_base_bins": g.n_bins, "base_interval_minutes": D.BASE_INTERVAL_MINUTES,
            "channels": list(g.channels),
            "coverage_overall": float(g.valid.mean()),
            "coverage_per_channel": {c: float(g.valid[:, j].mean()) for j, c in enumerate(g.channels)},
            "missing_base_bins_per_channel": {c: int((~g.valid[:, j]).sum())
                                              for j, c in enumerate(g.channels)},
            "splits": {k: [int(a), int(b)] for k, (a, b) in sp.items()},
            "train_scaling_mean": {c: float(mu[j]) for j, c in enumerate(g.channels)},
            "train_scaling_std": {c: float(sd[j]) for j, c in enumerate(g.channels)},
            "provenance": g.provenance,
            "grid_sha256": D.grid_fingerprint(g),
        }
    return out


def window_accounting(grids: dict, splits: dict) -> list[dict]:
    """Eligible / excluded windows per dataset, split, operation and r.  The counts
    are identical across operations and resolutions by construction; printing them
    per condition is the evidence for that."""
    rows = []
    for ds, g in grids.items():
        sp = splits[ds]
        for split in ("train", "val", "test"):
            stride = 1 if split == "train" else D.EVAL_ORIGIN_STRIDE
            lo, hi = sp[split]
            for j, ch in enumerate(g.channels):
                candidates = np.arange(lo, hi - D.FORECAST_BINS + 1)
                if stride > 1:
                    candidates = candidates[::stride]
                candidates = candidates[candidates - D.HISTORY_BINS >= 0]
                elig = D.eligible_origins(g, sp, split, j, stride=stride)
                for op in E.OPS:
                    for r in E.EVAL_R:
                        rows.append({
                            "dataset": ds, "split": split, "channel": ch, "operation": op, "r": r,
                            "candidate_windows": int(candidates.size),
                            "eligible_windows": int(elig.size),
                            "excluded_windows": int(candidates.size - elig.size),
                            "excluded_fraction": float(1 - elig.size / max(candidates.size, 1)),
                        })
    return rows


# -------------------------------------------------------------------- the tests


def run_contract_tests() -> dict:
    """T01-T10, A01-A04, F02-F04 and the analytical representation checks."""
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/oa_resolution_pilot_v1", "-q", "--tb=short"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {"exit_code": proc.returncode, "summary": tail,
            "seconds": round(time.time() - t0, 1),
            "status": "PASS" if proc.returncode == 0 else "INVALID_MEASUREMENT_PIPELINE",
            "stdout_tail": proc.stdout.strip()[-2000:]}


def analytical_representation_numbers() -> dict:
    from .operators import FOURIER_OMEGA, HISTORY_BINS, centre_basis, integrated_basis

    rng = np.random.default_rng(43)
    a = rng.uniform(0, HISTORY_BINS - 1, size=100)
    b = a + rng.uniform(1e-3, 24.0, size=100)
    bounds = np.stack([a, b], axis=1)
    closed = integrated_basis(bounds)
    errs = []
    for i, (lo, hi) in enumerate(bounds):
        t = (np.linspace(lo, hi, 200001) - HISTORY_BINS) / HISTORY_BINS
        ang = np.outer(t, FOURIER_OMEGA)
        num = np.concatenate([np.trapezoid(np.sin(ang), t, axis=0),
                              np.trapezoid(np.cos(ang), t, axis=0)]) / (t[-1] - t[0])
        errs.append(float(np.abs(num - closed[i]).max()))
    widths = [1.0, 1e-2, 1e-4, 1e-6]
    gaps = [float(np.abs(integrated_basis(np.array([[100.0, 100.0 + w]]))
                         - centre_basis(np.array([[100.0, 100.0 + w]]))).max()) for w in widths]
    return {
        "n_random_intervals": 100,
        "max_abs_error_vs_dense_quadrature": max(errs),
        "tolerance": 1e-5,
        "passes": max(errs) <= 1e-5,
        "shrinking_support_gap_to_centre_value": dict(zip(map(str, widths), gaps)),
        "converges_to_centre_value": all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)),
    }


# ------------------------------------------------------------------------- main


DRY = "--dry" in sys.argv


def main() -> int:
    global RESULTS, RUNS, ERRORS
    if DRY:
        RESULTS = REPO / "results" / "oa_resolution_pilot_v1_dryrun"
        RUNS = REPO / "runs" / "oa_resolution_pilot_v1_dryrun"
        ERRORS = RUNS / "errors"
        RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ERRORS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guard = ResourceGuard()
    guard.start()
    t_start = time.time()
    status = {"execution": "COMPLETE", "notes": []}

    log("01 environment")
    env = environment_json()
    write_json("environment.json", env)

    log("02 loading both datasets onto the shared 10-minute base grid")
    grids, splits, scaling, keys = {}, {}, {}, {}
    for ds, (a, b) in PERIODS.items():
        g = D.load_dataset(ds, a, b)
        sp = D.chronological_split(g.n_bins)
        mu, sd = D.train_scaling(g, sp)
        grids[ds], splits[ds], scaling[ds] = g, sp, (mu, sd)
        keys[ds] = E.frozen_keys(g, sp)
        log(f"    {ds}: {g.n_bins} bins, channels={g.channels}, "
            f"coverage={g.valid.mean():.5f}, test keys/channelset={len(keys[ds]['test'][0])}")

    write_json("data_manifest.json", data_manifest(grids, splits, scaling))
    pd.DataFrame(window_accounting(grids, splits)).to_csv(RESULTS / "window_accounting.csv", index=False)

    log("03 period selection audit")
    write_json("period_selection.json", {ds: D.select_period(ds) for ds in PERIODS})

    log("04 operator and representation contract tests")
    tests = run_contract_tests()
    write_json("operator_unit_tests.json", tests)
    analytical = analytical_representation_numbers()
    write_json("analytical_representation_tests.json", analytical)
    if tests["status"] != "PASS" or not analytical["passes"]:
        write_json("verdict.json", {"scientific_decision": "INVALID",
                                    "reason": "INVALID_MEASUREMENT_PIPELINE", "tests": tests})
        log("CONTRACT TESTS FAILED -- no fit is allowed to start")
        return 2
    log(f"    {tests['summary']}")

    log("05 100-update smoke and runtime tier freeze")
    smoke, projected = {}, 0.0
    smoke_tier = T.Tier("SMOKE", 100, 25, 100)
    for ds, g in grids.items():
        for arm in ARMS:
            out = T.train_arm(arm, g, splits[ds], *scaling[ds], MODEL_SEEDS[0], smoke_tier,
                              device, lam=1e-2 if arm == "R" else None, log=lambda s: None)
            smoke[f"{ds}/{arm}"] = {"seconds_per_100_updates": round(out["wall_seconds"], 2),
                                    "n_parameters": out["n_parameters"]}
            projected += out["wall_seconds"] * 50 * len(MODEL_SEEDS)
            del out
            torch.cuda.empty_cache()
    hours = projected / 3600
    tier = T.TIERS["FULL"] if hours <= 4 else (T.TIERS["COMPACT"] if hours <= 6 else T.TIERS["SCREEN"])
    if DRY:
        tier = T.Tier("DRY", 60, 10, 30)
    smoke["projected_full_tier_hours_for_12_fits"] = round(hours, 3)
    smoke["tier"] = tier.name
    write_json("runtime_tier.json", smoke)
    log(f"    projected {hours:.2f} h for 12 FULL fits -> tier {tier.name}")

    log("06 lambda selection for R (train/validation only)")
    lambdas = {}
    for ds, g in grids.items():
        sel = T.select_lambda(g, splits[ds], *scaling[ds], tier, device, MODEL_SEEDS[0],
                              log=lambda s: log("    " + s))  # noqa: B023
        lambdas[ds] = sel["selected_lambda"]
        write_json(f"lambda_selection_{ds}.json", sel)
        log(f"    {ds}: lambda = {sel['selected_lambda']:g}")

    log("07 freezing the execution spec")
    spec = {
        "experiment_id": "OA-RESOLUTION-PILOT-v1",
        "base_interval_minutes": 10,
        "history_bins": 288, "history_hours": 48,
        "forecast_bins": 72, "forecast_hours": 12,
        "train_report_intervals_r": list(T.TRAIN_R),
        "unseen_interpolation_r": list(E.INTERP_R),
        "unseen_extrapolation_r": list(E.EXTRAP_R),
        "operations": list(E.OPS),
        "periods": {ds: {"start": a, "end": b} for ds, (a, b) in PERIODS.items()},
        "period_selection_rule": "available from the official source, continuous two years, "
                                 "maximum coverage of the selected channels, ties to the earlier "
                                 "window; forecast accuracy plays no part",
        "channels": {ds: list(g.channels) for ds, g in grids.items()},
        "split_fractions": {"train": 0.70, "val": 0.10, "test": 0.20},
        "scaling": "per channel, mean and population std over the train base bins only",
        "arms": {"R": "operator-aware reconstruction + base-grid forecaster",
                 "M": "observed tokens with centre-time metadata",
                 "O": "observed tokens with the interval-integrated time basis"},
        "fourier_k": 16,
        "fourier_cycles_per_48h": "geometric, 1 to 128",
        "query_basis": "centre basis on the 72 target base bins, identical for M and O",
        "model": {"d_model": 128, "heads": 4, "ffn": 256, "dropout": 0.1,
                  "encoder_layers": 2, "decoder": "1 cross-attention block, pre-norm"},
        "loss": "0.5 * MSE_10min + 0.5 * MSE_60min, each averaged over its own lead axis first",
        "hourly_aggregation": "6-step block mean of the 72 ten-minute outputs, applied identically "
                              "to predictions and targets",
        "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 0.01,
                      "effective_batch_windows": T.BATCH_WINDOWS, "grad_clip": 1.0},
        "tier": asdict(tier),
        "checkpoint_selection": "minimum validation primary loss over the TRAINING resolutions only "
                                "(r = 2, 4, 8); unseen resolutions never enter model selection",
        "model_seeds": list(MODEL_SEEDS),
        "observation_schedule_seed": T.SCHEDULE_SEED,
        "bootstrap_seed": RPT.BOOTSTRAP_SEED,
        "bootstrap_draws": RPT.BOOTSTRAP_DRAWS,
        "bootstrap_block_days": RPT.BLOCK_DAYS,
        "evaluation_origin_stride_bins": D.EVAL_ORIGIN_STRIDE,
        "eligibility_rule": "every base bin of the 288-bin history and the 72-bin target must be a "
                            "real measurement; the rule ignores r and the operation so the window "
                            "set is identical across arms, operations and resolutions",
        "lambda_candidates": list(T.LAMBDA_CANDIDATES),
        "lambda_selected": lambdas,
        "go_no_go": RPT.GO,
        "core_fits": len(PERIODS) * len(ARMS) * len(MODEL_SEEDS),
    }
    spec_path = write_json("execution_spec.json", spec)
    sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    (RESULTS / "execution_spec.sha256").write_text(sha + "\n", encoding="utf-8")
    log(f"    spec sha256 {sha[:16]}...")

    log(f"08 {spec['core_fits']} core fits at tier {tier.name}")
    fits, all_rows, nets = [], [], {}
    for ds, g in grids.items():
        for seed in MODEL_SEEDS:
            sch = T.build_schedule(g, splits[ds], seed, tier.max_updates)
            for arm in ARMS:
                if guard.stop_flag or time.time() - t_start > WALL_CAP_SECONDS:
                    status["execution"] = "PARTIAL"
                    status["notes"].append("BLOCKED_RESOURCE" if guard.stop_flag else "WALL_CAP")
                    break
                out = T.train_arm(arm, g, splits[ds], *scaling[ds], seed, tier, device,
                                  lam=lambdas[ds] if arm == "R" else None, log=log, schedule=sch)
                curve = out.pop("curve")
                net = out.pop("net")
                fits.append({**out, "n_curve_points": len(curve),
                             "best_is_first": curve[0]["update"] == out["best_update"],
                             "best_is_last": curve[-1]["update"] == out["best_update"],
                             "last_interval_improving": bool(curve[-1]["primary"] < curve[-2]["primary"])
                             if len(curve) > 1 else None})
                pd.DataFrame(curve).assign(dataset=ds, arm=arm, seed=seed).to_csv(
                    RUNS / f"curve_{ds}_{arm}_{seed}.csv", index=False)
                rows = E.evaluate_model(net, arm, g, splits[ds], *scaling[ds], seed, device,
                                        lambdas[ds] if arm == "R" else None, keys[ds])
                np.save(ERRORS / f"errors_{ds}_{arm}_{seed}.npy", rows)
                all_rows.append(RPT.to_frame(rows, ds))
                nets[(ds, arm, seed)] = net
                log(f"    done {ds}/{arm}/seed{seed}: best_update={out['best_update']} "
                    f"val={out['best_val_primary']:.5f} ({out['wall_seconds']:.0f}s)")

    pd.DataFrame(fits).to_csv(RESULTS / "fit_manifest.csv", index=False)
    curves = pd.concat([pd.read_csv(p) for p in sorted(RUNS.glob("curve_*.csv"))], ignore_index=True)
    curves.to_csv(RESULTS / "training_curves_summary.csv", index=False)

    log("09 aggregating the frozen keys")
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(RUNS / "metrics_full.csv", index=False)
    sa = RPT.seed_averaged(df)
    cells = RPT.cell_table(sa, "test")
    cells.to_csv(RESULTS / "metrics.csv", index=False)

    contrasts = {}
    for role, rs in (("seen", RPT.SEEN_R), ("unseen_interpolation", RPT.INTERP_R),
                     ("unseen_extrapolation", RPT.EXTRAP_R)):
        contrasts[role] = {
            "O_vs_M": RPT.macro(cells, rs, "O", "M"),
            "O_vs_R": RPT.macro(cells, rs, "O", "R"),
            "M_vs_R": RPT.macro(cells, rs, "M", "R"),
            "per_dataset_O_vs_M": RPT.per_dataset_macro(cells, rs, "O", "M"),
        }
    per_seed = {}
    for seed in MODEL_SEEDS:
        c = RPT.cell_table(df[df["seed"] == seed], "test")
        per_seed[str(seed)] = RPT.macro(c, RPT.INTERP_R, "O", "M")["macro_relative_improvement_pct"]
    contrasts["per_seed_macro_O_vs_M_unseen_interpolation"] = per_seed
    write_json("primary_contrasts.json", contrasts)

    log("10 paired time-block bootstrap")
    boot = RPT.paired_block_bootstrap(sa, RPT.INTERP_R, "O", "M")
    boot_or = RPT.paired_block_bootstrap(sa, RPT.INTERP_R, "O", "R")
    boot_mr = RPT.paired_block_bootstrap(sa, RPT.INTERP_R, "M", "R")
    write_json("bootstrap.json", {"O_vs_M_unseen_interpolation": boot,
                                  "O_vs_R_unseen_interpolation": boot_or,
                                  "M_vs_R_unseen_interpolation": boot_mr})
    log(f"    O vs M interp: mean {boot['mean']:+.3f}%  95% CI "
        f"[{boot['lower95']:+.3f}, {boot['upper95']:+.3f}]")

    log("11 mechanism diagnostics and the unit contract")
    diag = {"within_batch_width_permutation": E.within_batch_width_permutation_is_identity(),
            "variants": {}, "note": "run on every completed model, inference only, no retraining"}
    unit = {}
    for (ds, arm, seed), net in nets.items():
        g, sp, (mu, sd) = grids[ds], splits[ds], scaling[ds]
        lam = lambdas[ds] if arm == "R" else None
        base = E.diagnostic(net, arm, "BASE", g, sp, mu, sd, device, keys[ds], lam=lam)
        entry = {"BASE": base,
                 "WIDTH_MISMATCH": E.diagnostic(net, arm, "WIDTH_MISMATCH", g, sp, mu, sd, device,
                                                keys[ds], lam=lam) if arm != "R" else "N/A (R has no "
                                                "per-token width metadata; the operator enters through "
                                                "the reconstruction matrix)"}
        entry["WRONG_SUPPORT"] = (
            E.diagnostic(net, arm, "WRONG_SUPPORT", g, sp, mu, sd, device, keys[ds], lam=lam)
            if arm == "O" else
            "N/A (identity for M, which already uses the centre basis; not defined for R)")
        diag["variants"][f"{ds}/{arm}/{seed}"] = entry
        unit[f"{ds}/{arm}/{seed}"] = E.sum_mean_invariance(net, arm, g, sp, mu, sd, device,
                                                           keys[ds], lam)
    write_json("mechanism_diagnostics.json", diag)
    write_json("unit_contract.json", unit)
    worst_unit = max(v["relative_prediction_difference"] for v in unit.values())
    if worst_unit > 1e-4:
        status["notes"].append(f"UNIT_REPRESENTATION_BUG (relative {worst_unit:.2e})")
    log(f"    worst relative SUM/MEAN prediction difference: {worst_unit:.3e}")

    log("12 learning sanity anchors")
    anchors = {ds: E.seasonal_naive(grids[ds], splits[ds], *scaling[ds], keys[ds])
               for ds in grids}
    write_json("learning_anchors.json", anchors)

    log("13 efficiency")
    eff = pd.DataFrame(fits)[["dataset", "arm", "model_seed", "n_parameters", "wall_seconds",
                              "best_update", "best_val_primary"]]
    eff.to_csv(RESULTS / "efficiency.csv", index=False)

    log("14 verdict")
    verdict = RPT.decide(contrasts, boot, per_seed,
                         contrasts["unseen_interpolation"]["per_dataset_O_vs_M"],
                         bootstrap_M_vs_R=boot_mr)
    verdict["execution_status"] = status["execution"]
    verdict["notes"] = status["notes"]
    verdict["resources"] = guard.close()
    verdict["wall_seconds_total"] = round(time.time() - t_start, 1)
    write_json("verdict.json", verdict)

    write_tables(cells)

    log("=" * 70)
    log(f"EXECUTION STATUS: {status['execution']}")
    log(f"SCIENTIFIC DECISION: {verdict['scientific_decision']}")
    log(f"O vs M unseen interpolation macro: "
        f"{verdict['primary_O_vs_M_unseen_interpolation_macro_pct']:+.3f}%")
    log(f"O vs R unseen interpolation macro: "
        f"{verdict['O_vs_R_unseen_interpolation_macro_pct']:+.3f}%")
    log("=" * 70)
    return 0


def verdict_only() -> int:
    """Recompute contrasts, bootstrap, tables and the verdict from the frozen
    per-key error arrays.  No model is retrained and no key set is rebuilt; this is
    the stage to rerun when the scoring code itself is corrected."""
    df = pd.read_csv(RUNS / "metrics_full.csv")
    sa = RPT.seed_averaged(df)
    cells = RPT.cell_table(sa, "test")
    cells.to_csv(RESULTS / "metrics.csv", index=False)

    contrasts = {}
    for role, rs in (("seen", RPT.SEEN_R), ("unseen_interpolation", RPT.INTERP_R),
                     ("unseen_extrapolation", RPT.EXTRAP_R)):
        contrasts[role] = {
            "O_vs_M": RPT.macro(cells, rs, "O", "M"),
            "O_vs_R": RPT.macro(cells, rs, "O", "R"),
            "M_vs_R": RPT.macro(cells, rs, "M", "R"),
            "per_dataset_O_vs_M": RPT.per_dataset_macro(cells, rs, "O", "M"),
        }
    per_seed = {}
    for seed in MODEL_SEEDS:
        c = RPT.cell_table(df[df["seed"] == seed], "test")
        per_seed[str(seed)] = RPT.macro(c, RPT.INTERP_R, "O", "M")["macro_relative_improvement_pct"]
    contrasts["per_seed_macro_O_vs_M_unseen_interpolation"] = per_seed
    write_json("primary_contrasts.json", contrasts)

    existing = json.loads((RESULTS / "bootstrap.json").read_text(encoding="utf-8"))
    boot = existing["O_vs_M_unseen_interpolation"]
    boot_mr = existing.get("M_vs_R_unseen_interpolation") or         existing["supplementary_post_hoc"]["intervals"]["M_vs_R_unseen_interpolation_all"]
    existing["M_vs_R_unseen_interpolation"] = boot_mr
    write_json("bootstrap.json", existing)

    prior = json.loads((RESULTS / "verdict.json").read_text(encoding="utf-8"))
    verdict = RPT.decide(contrasts, boot, per_seed,
                         contrasts["unseen_interpolation"]["per_dataset_O_vs_M"],
                         bootstrap_M_vs_R=boot_mr)
    for k in ("execution_status", "notes", "resources", "wall_seconds_total"):
        if k in prior:
            verdict[k] = prior[k]
    verdict["superseded_decision"] = {
        "token": prior["scientific_decision"],
        "why_changed": (
            "The first gate implementation read the section 35 clause 'M is significantly better "
            "than R' as a point-estimate macro threshold and dropped the significance requirement. "
            "The paired bootstrap puts the M vs R unseen-interpolation interval across zero and the "
            "sign flips between the two datasets, so that clause is not satisfied and the outcome "
            "falls through to INCONCLUSIVE. No threshold was lowered and no model was added; the "
            "correction makes the verdict weaker, not stronger. Only the scoring stage was rerun -- "
            "the fits, the frozen keys and the per-key errors are untouched."),
    }
    write_json("verdict.json", verdict)

    tables = {
        "primary_unseen_interpolation": RPT.markdown_table(cells, RPT.INTERP_R),
        "seen": RPT.markdown_table(cells, RPT.SEEN_R),
        "unseen_extrapolation": RPT.markdown_table(cells, RPT.EXTRAP_R),
    }
    write_tables(cells)
    log(f"SCIENTIFIC DECISION: {verdict['scientific_decision']} "
        f"(was {prior['scientific_decision']})")
    return 0


if __name__ == "__main__":
    sys.exit(verdict_only() if "--verdict-only" in sys.argv else main())
