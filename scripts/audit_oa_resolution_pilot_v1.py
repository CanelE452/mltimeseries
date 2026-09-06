"""OA-RESOLUTION-PILOT-v1-AUDIT-CLOSURE-v1.

Re-checks the finished pilot without refitting anything. It imports nothing from
`experiments.oa_resolution_pilot_v1.report` on purpose: every number it prints is
recomputed from the stored artifacts by code written a second time.

No model is built, no optimizer is created, no backward pass is run. The only
inputs are `results/oa_resolution_pilot_v1/` (read-only, hashed before and after)
and the local `runs/oa_resolution_pilot_v1/` error arrays, which are gitignored.

  python scripts/audit_oa_resolution_pilot_v1.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ORIG = REPO / "results" / "oa_resolution_pilot_v1"
RUNS = REPO / "runs" / "oa_resolution_pilot_v1"
OUT = ORIG / "audit_closure_v1"

DATASETS = ("jena", "uci")
ARMS = ("R", "M", "O")
OPS = ("END_BIN", "INTERVAL_MEAN")
SEEDS = (2026090601, 2026090602)
SEEN_R = (2, 4, 8)
INTERP_R = (3, 6)
EXTRAP_R = (12,)

BOOTSTRAP_SEED = 2026090699
BOOTSTRAP_DRAWS = 1000
BLOCK_DAYS = 7
BINS_PER_DAY = 144

IMMUTABLE = [
    "STATUS.md", "execution_spec.json", "execution_spec.sha256", "metrics.csv",
    "primary_contrasts.json", "bootstrap.json", "verdict.json", "fit_manifest.csv",
    "training_curves_summary.csv", "mechanism_diagnostics.json",
    "reconstruction_quality.json", "flowstate_reference.json", "unit_contract.json",
    "learning_anchors.json", "measurement_semantics.json", "data_manifest.json",
    "figures/captions.md", "figures/fig1_o_vs_m_unseen_interpolation.png",
    "figures/fig2_arms_by_resolution_role.png", "figures/fig3_training_curves.png",
    "figures/fig4_accuracy_vs_cost.png",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_originals() -> dict:
    return {rel: {"sha256": sha256(ORIG / rel), "bytes": (ORIG / rel).stat().st_size}
            for rel in IMMUTABLE if (ORIG / rel).exists()}


def write(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def sh(cmd: str) -> str | None:
    try:
        return subprocess.check_output(cmd, shell=True, cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ----------------------------------------------------------------- 5. arithmetic


def relative_improvement(a: float, b: float) -> float:
    """RI(A vs B) = 100 * (1 - Loss_A / Loss_B). Positive means A has lower error."""
    return 100.0 * (1.0 - a / b)


def macro_from_cells(cells: pd.DataFrame, rs, arm_a: str, arm_b: str) -> dict:
    """Equal weight over every (dataset, operation, r) cell in `rs`."""
    sub = cells[cells["r"].isin(rs)]
    piv = sub.pivot_table(index=["dataset", "operation", "r"], columns="arm", values="primary")
    ri = {idx: relative_improvement(row[arm_a], row[arm_b]) for idx, row in piv.iterrows()}
    return {"macro_pct": float(np.mean(list(ri.values()))), "n_cells": len(ri),
            "cells": [{"dataset": k[0], "operation": k[1], "r": int(k[2]), "ri_pct": v}
                      for k, v in ri.items()]}


def arithmetic_revalidation() -> dict:
    """Recompute every macro from metrics.csv and compare with primary_contrasts.json."""
    cells = pd.read_csv(ORIG / "metrics.csv")
    stored = json.loads((ORIG / "primary_contrasts.json").read_text(encoding="utf-8"))

    roles = {"seen": SEEN_R, "unseen_interpolation": INTERP_R, "unseen_extrapolation": EXTRAP_R}
    out, worst = {}, 0.0
    for role, rs in roles.items():
        out[role] = {}
        for a, b in (("O", "M"), ("O", "R"), ("M", "R")):
            mine = macro_from_cells(cells, rs, a, b)
            theirs = stored[role][f"{a}_vs_{b}"]["macro_relative_improvement_pct"]
            delta = abs(mine["macro_pct"] - theirs)
            worst = max(worst, delta)
            out[role][f"{a}_vs_{b}"] = {
                "recomputed_pct": mine["macro_pct"], "stored_pct": theirs,
                "abs_delta_pp": delta, "n_cells": mine["n_cells"], "cells": mine["cells"],
            }
        by_ds = {}
        for ds in DATASETS:
            sub = cells[cells["dataset"] == ds]
            by_ds[ds] = {f"{a}_vs_{b}": macro_from_cells(sub, rs, a, b)["macro_pct"]
                         for a, b in (("O", "M"), ("O", "R"), ("M", "R"))}
        out[role]["per_dataset"] = by_ds

    tol = 1e-10
    return {
        "basis": "results/oa_resolution_pilot_v1/metrics.csv, recomputed without importing report.py",
        "tolerance_pp": tol,
        "max_abs_delta_pp": worst,
        "status": "ARITHMETIC_REPRODUCTION_OK" if worst <= tol else "ARITHMETIC_REPRODUCTION_FAIL",
        "roles": out,
    }


# --------------------------------------------------------------- 7. raw errors


def raw_error_inventory(base_sha: str) -> dict:
    tracked = set((sh(f"git ls-tree -r --name-only {base_sha}") or "").splitlines())
    files = []
    for ds in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                rel = f"runs/oa_resolution_pilot_v1/errors/errors_{ds}_{arm}_{seed}.npy"
                p = REPO / rel
                files.append({
                    "path": rel, "local_exists": p.exists(),
                    "bytes": p.stat().st_size if p.exists() else None,
                    "sha256": sha256(p) if p.exists() else None,
                    "git_tracked": rel in tracked,
                })
    n_local = sum(f["local_exists"] for f in files)
    n_tracked = sum(f["git_tracked"] for f in files)
    if n_local == 12 and n_tracked == 0:
        status = "LOCAL_12_RAW_ERRORS_PRESENT_GIT_UNTRACKED"
    elif n_tracked == 12:
        status = "RAW_ERRORS_COMMITTED"
    elif n_local == 0:
        status = "RAW_ERRORS_MISSING"
    else:
        status = f"MIXED_local={n_local}_tracked={n_tracked}"
    return {
        "status": status, "n_local": n_local, "n_git_tracked": n_tracked,
        "gitignore_rule": sh("git check-ignore -v runs/oa_resolution_pilot_v1/errors/"
                             "errors_jena_M_2026090601.npy"),
        "correction": ("An earlier report said the 12 raw error arrays were present in the commit. "
                       "They are present locally under runs/ but runs/ is gitignored, so they were "
                       "never part of any commit."),
        "files": files,
    }


# -------------------------------------------------- 8/9. bootstrap reproduction


def load_seed_averaged() -> pd.DataFrame:
    """Per logical key, averaged over the two model seeds -- the unit the original
    bootstrap resamples. Rebuilt here from the raw per-key error arrays."""
    frames = []
    for ds in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                p = RUNS / "errors" / f"errors_{ds}_{arm}_{seed}.npy"
                rows = np.load(p)
                df = pd.DataFrame(rows)
                df["dataset"] = ds
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in ("split", "operation", "role", "arm"):
        if df[col].dtype != object:
            df[col] = df[col].astype(str)
    df["mse10"] = df["SE10"] / df["count10"]
    df["mse60"] = df["SE60"] / df["count60"]
    df["primary"] = 0.5 * df["mse10"] + 0.5 * df["mse60"]
    keys = ["dataset", "split", "origin", "channel", "operation", "r", "role", "arm"]
    return df.groupby(keys, as_index=False)[["primary", "mse10", "mse60"]].mean()


def block_id(origin: np.ndarray) -> np.ndarray:
    return origin // (BLOCK_DAYS * BINS_PER_DAY)


def sufficient_stats(sa: pd.DataFrame, rs) -> pd.DataFrame:
    """Per (dataset, block, operation, r, arm): the sum of the seed-averaged primary
    loss and the number of logical keys. A draw's cell mean is
    sum(primary_sum) / sum(n_keys) over the sampled blocks, which is exactly what the
    original per-draw groupby-mean computes."""
    d = sa[(sa["split"] == "test") & (sa["r"].isin(rs))].copy()
    d["block_id"] = block_id(d["origin"].to_numpy())
    g = d.groupby(["dataset", "block_id", "operation", "r", "arm"], as_index=False).agg(
        primary_sum=("primary", "sum"), n_keys=("primary", "size"))
    return g.sort_values(["dataset", "block_id", "operation", "r", "arm"]).reset_index(drop=True)


def bootstrap_from_stats(stats: pd.DataFrame, arm_a: str, arm_b: str,
                         draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired 7-day time-block (cluster) bootstrap, reproduced from block sums only.

    Not a moving-block bootstrap: the blocks are the fixed, non-overlapping
    `origin // (7 days)` partitions, and whole blocks are drawn with replacement.
    """
    per_ds = {}
    for ds in DATASETS:
        sub = stats[stats["dataset"] == ds]
        if not len(sub):
            continue
        blocks = np.sort(sub["block_id"].unique())
        lut = {}
        for b in blocks:
            bb = sub[sub["block_id"] == b]
            lut[b] = {(o, r, a): (s, n) for o, r, a, s, n in zip(
                bb["operation"], bb["r"], bb["arm"], bb["primary_sum"], bb["n_keys"])}
        per_ds[ds] = (blocks, lut)

    cell_keys = sorted({(ds, o, r) for ds in per_ds
                        for o in OPS for r in sorted(stats["r"].unique())})
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        acc: dict[tuple, list[float]] = {}
        for ds, (blocks, lut) in per_ds.items():
            pick = rng.choice(blocks, size=len(blocks), replace=True)
            for b in pick:
                for (o, r, a), (s, n) in lut[b].items():
                    k = (ds, o, r, a)
                    if k not in acc:
                        acc[k] = [0.0, 0]
                    acc[k][0] += s
                    acc[k][1] += n
        ris = []
        for ds, o, r in cell_keys:
            ka, kb = (ds, o, r, arm_a), (ds, o, r, arm_b)
            if ka in acc and kb in acc:
                la = acc[ka][0] / acc[ka][1]
                lb = acc[kb][0] / acc[kb][1]
                ris.append(relative_improvement(la, lb))
        if ris:
            out.append(float(np.mean(ris)))
    s = np.array(out)
    return {"arm_a": arm_a, "arm_b": arm_b, "draws": int(len(s)),
            "mean": float(s.mean()), "lower95": float(np.percentile(s, 2.5)),
            "upper95": float(np.percentile(s, 97.5)),
            "effective_blocks_per_dataset": {ds: int(len(b)) for ds, (b, _) in per_ds.items()}}


# -------------------------------------------------------- 16/17/19. r = 12 table


def unit_contract_recheck() -> dict:
    """Section 31.  The SUM = r x MEAN control, re-read rather than re-run.

    This is a unit check, not a performance arm: it asks whether routing the same
    observation through the discrete-total representation and back changes anything.
    """
    uc = json.loads((ORIG / "unit_contract.json").read_text(encoding="utf-8"))
    rows = [{"model": k, **v} for k, v in uc.items()]
    worst_repr = max(r["max_abs_representation_difference"] for r in rows)
    worst_pred = max(r["max_abs_prediction_difference"] for r in rows)
    worst_rel = max(r["relative_prediction_difference"] for r in rows)
    tol = 1e-4
    return {
        "role": "unit control only; SUM is never scored as a performance arm",
        "n_models_checked": len(rows),
        "max_abs_representation_difference": worst_repr,
        "max_abs_prediction_difference": worst_pred,
        "max_relative_prediction_difference": worst_rel,
        "tolerance_relative": tol,
        "status": "UNIT_CONTRACT_HOLDS" if worst_rel <= tol else "UNIT_REPRESENTATION_BUG",
        "reading": ("Forming the discrete total by an independent sum over base bins and dividing "
                    "by the report width reproduces the INTERVAL_MEAN observation exactly, and the "
                    "arms forecast identically from it.  Nothing here is a model result."),
        "per_model": rows,
    }


def paired_seed_effects(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Section 6.  O vs M per (dataset, operation, r, seed), plus the per-seed macros.

    Two seeds is not a sample to build an interval from, so no standard error or seed
    CI is computed here -- only the paired values themselves.
    """
    rows = []
    for seed in SEEDS:
        sub = per_seed[per_seed["seed"] == seed]
        piv = sub.pivot_table(index=["dataset", "operation", "r"], columns="arm", values="primary")
        for idx, row in piv.iterrows():
            rows.append({
                "dataset": idx[0], "operation": idx[1], "r": int(idx[2]), "seed": seed,
                "role": ("SEEN" if idx[2] in SEEN_R else
                         "UNSEEN_INTERPOLATION" if idx[2] in INTERP_R else
                         "UNSEEN_EXTRAPOLATION"),
                "M": row["M"], "O": row["O"], "R": row["R"],
                "O_vs_M_pct": relative_improvement(row["O"], row["M"]),
                "O_vs_R_pct": relative_improvement(row["O"], row["R"]),
                "M_vs_R_pct": relative_improvement(row["M"], row["R"]),
            })
    df = pd.DataFrame(rows)
    macros = []
    for seed in SEEDS:
        for role, rs in (("SEEN", SEEN_R), ("UNSEEN_INTERPOLATION", INTERP_R),
                         ("UNSEEN_EXTRAPOLATION", EXTRAP_R)):
            sub = df[(df["seed"] == seed) & (df["r"].isin(rs))]
            macros.append({"dataset": "MACRO", "operation": "MACRO", "r": -1, "seed": seed,
                           "role": role, "M": np.nan, "O": np.nan, "R": np.nan,
                           "O_vs_M_pct": sub["O_vs_M_pct"].mean(),
                           "O_vs_R_pct": sub["O_vs_R_pct"].mean(),
                           "M_vs_R_pct": sub["M_vs_R_pct"].mean()})
    return pd.concat([df, pd.DataFrame(macros)], ignore_index=True)


def per_seed_cells(sa_or_full: pd.DataFrame) -> pd.DataFrame:
    """Mean primary per (dataset, operation, r, seed, arm) on the test split.

    The join key is explicit and complete, so an R value can never be taken from a
    different operation than the M and O values beside it.
    """
    d = sa_or_full[sa_or_full["split"] == "test"]
    return d.groupby(["dataset", "operation", "r", "seed", "arm"],
                     as_index=False)["primary"].mean()


def select_cell(cells: pd.DataFrame, dataset: str, operation: str, r: int,
                arm: str, seed: int) -> float:
    """The only way this audit reads a number for a table. Raises if the key is not
    unique, so a silent cross-cell copy is impossible."""
    m = cells[(cells["dataset"] == dataset) & (cells["operation"] == operation)
              & (cells["r"] == r) & (cells["arm"] == arm) & (cells["seed"] == seed)]
    if len(m) != 1:
        raise KeyError(f"{dataset}/{operation}/r{r}/{arm}/seed{seed} matched {len(m)} rows")
    return float(m["primary"].iloc[0])


def r12_seed_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for op in OPS:
            for seed in SEEDS:
                v = {a: select_cell(per_seed, ds, op, 12, a, seed) for a in ARMS}
                rows.append({
                    "dataset": ds, "operation": op, "r": 12, "seed": seed,
                    "R": v["R"], "M": v["M"], "O": v["O"],
                    "O_vs_M_pct": relative_improvement(v["O"], v["M"]),
                    "O_vs_R_pct": relative_improvement(v["O"], v["R"]),
                    "M_vs_R_pct": relative_improvement(v["M"], v["R"]),
                })
    return pd.DataFrame(rows)


def status_md_r12_provenance(r12: pd.DataFrame) -> dict:
    """Where did the R column of the STATUS.md r = 12 seed table come from?

    The table is parsed out of STATUS.md rather than transcribed, and matched at the
    precision STATUS actually printed, so the answer does not depend on how the
    audit reproduces the last decimals.
    """
    text = (ORIG / "STATUS.md").read_text(encoding="utf-8")
    printed = {}
    for line in text.splitlines():
        if line.startswith("| seed ") and line.count("|") >= 6:
            parts = [c.strip() for c in line.strip("|").split("|")]
            seed = int(parts[0].split()[1])
            printed[seed] = {"M": float(parts[1]), "O": float(parts[2]), "R": float(parts[3])}
    if not printed:
        return {"status": "STATUS_TABLE_NOT_FOUND"}

    dp = 5  # STATUS printed five decimals
    def rowof(op, seed):
        m = r12[(r12.dataset == "jena") & (r12.operation == op) & (r12.seed == seed)]
        return {a: float(m[a].iloc[0]) for a in ARMS}

    checks = {}
    for seed, vals in printed.items():
        im, eb = rowof("INTERVAL_MEAN", seed), rowof("END_BIN", seed)
        checks[seed] = {
            "printed": vals,
            "correct_INTERVAL_MEAN": im,
            "END_BIN_same_r_and_seed": eb,
            "printed_R_matches_INTERVAL_MEAN": round(vals["R"], dp) == round(im["R"], dp),
            "printed_R_matches_END_BIN": round(vals["R"], dp) == round(eb["R"], dp),
            "printed_M_matches_INTERVAL_MEAN": round(vals["M"], dp) == round(im["M"], dp),
            "printed_O_matches_INTERVAL_MEAN": round(vals["O"], dp) == round(im["O"], dp),
        }
    all_r_from_end_bin = all(c["printed_R_matches_END_BIN"] and not c["printed_R_matches_INTERVAL_MEAN"]
                             for c in checks.values())
    mo_ok = all(c["printed_M_matches_INTERVAL_MEAN"] and c["printed_O_matches_INTERVAL_MEAN"]
                for c in checks.values())
    return {
        "parsed_from": "results/oa_resolution_pilot_v1/STATUS.md",
        "matched_at_decimals": dp,
        "per_seed": checks,
        "diagnosis": ("The R column was taken from the END_BIN row at the same r and seed, while M "
                      "and O came from INTERVAL_MEAN as intended."
                      if all_r_from_end_bin and mo_ok else "See per_seed."),
        "r_column_wrong": all_r_from_end_bin,
        "m_and_o_columns_correct": mo_ok,
    }


def r12_decomposition(cells: pd.DataFrame, r12: pd.DataFrame) -> dict:
    ex = macro_from_cells(cells, EXTRAP_R, "O", "M")
    dominant = max(ex["cells"], key=lambda c: c["ri_pct"])
    rest = [c for c in ex["cells"] if c is not dominant]
    # does metrics.csv agree with the per-seed mean for the disputed cell?
    seed_mean = {a: float(r12[(r12.dataset == "jena") & (r12.operation == "INTERVAL_MEAN")][a].mean())
                 for a in ARMS}
    stored = cells[(cells.dataset == "jena") & (cells.operation == "INTERVAL_MEAN") & (cells.r == 12)]
    stored_v = {a: float(stored[stored.arm == a]["primary"].iloc[0]) for a in ARMS}
    agree = {a: abs(seed_mean[a] - stored_v[a]) for a in ARMS}
    return {
        "all_four_cells": ex["cells"],
        "macro_all_pct": ex["macro_pct"],
        "dominant_cell": dominant,
        "macro_excluding_dominant_pct": float(np.mean([c["ri_pct"] for c in rest])),
        "metrics_csv_vs_per_seed_mean": {
            "per_seed_mean": seed_mean, "stored_in_metrics_csv": stored_v,
            "max_abs_delta": max(agree.values()),
            "aggregate_is_consistent": max(agree.values()) < 1e-9,
        },
        "interpretation": (
            "The extrapolation macro is carried by one cell. Removing it leaves a macro of the "
            "opposite sign and near zero. This is an exploratory unseen-extrapolation observation, "
            "not evidence that the representation works at wide support in general."),
    }


# ------------------------------------------- 11/12/13. FlowState native semantics


def can_form_hourly_mean_from_native(operation: str, r: int) -> bool:
    """Whether averaging native coarse predictions yields the hourly mean.

    INTERVAL_MEAN: each prediction is the mean over its own r base bins, so
    averaging consecutive ones tiles the hour whenever 6 % r == 0.
    END_BIN: each prediction is a single base bin at the end of its interval. The
    other r-1 bins are never represented, so no average of END_BIN values is the
    hourly mean, at any r.
    """
    if operation == "INTERVAL_MEAN":
        return 6 % r == 0
    return False


def flowstate_semantic_audit() -> dict:
    fs = json.loads((ORIG / "flowstate_reference.json").read_text(encoding="utf-8"))
    truth = np.arange(1, 7, dtype=float)
    counterexamples = []
    for r in (2, 3, 6):
        end_bin = truth[r - 1 :: r]
        blocks = truth.reshape(6 // r, r) if 6 % r == 0 else None
        counterexamples.append({
            "r": r, "base_bins": truth.tolist(), "true_hourly_mean": float(truth.mean()),
            "perfect_END_BIN_predictions": end_bin.tolist(),
            "END_BIN_average": float(end_bin.mean()),
            "END_BIN_bias": float(end_bin.mean() - truth.mean()),
            "perfect_INTERVAL_MEAN_predictions": blocks.mean(axis=1).tolist(),
            "INTERVAL_MEAN_average": float(blocks.mean(axis=1).mean()),
        })

    per_ds = {}
    for ds, d in fs["datasets"].items():
        rows = d["native_rate"]["cells"]
        per_ds[ds] = {
            "INTERVAL_MEAN": {"status": "SEMANTICALLY_VALID_WHERE_THE_GRID_TILES_THE_HOUR",
                              "cells": [c for c in rows if c["operation"] == "INTERVAL_MEAN"]},
            "END_BIN": {"status": "INVALID_FOR_CORE_HOURLY_MEAN_METRIC",
                        "reason": ("Averaging END_BIN forecasts cannot equal the hourly mean at any "
                                   "r: only the last base bin of each report interval is predicted, "
                                   "so r-1 of every r bins never enter the average. Even a perfect "
                                   "END_BIN forecaster is biased against this metric."),
                        "cells": [{**c, "note": "HISTORICAL_NUMBER_NOT_COMPARABLE"}
                                  for c in rows if c["operation"] == "END_BIN"]},
        }

    return {
        "FLOWSTATE_RESAMPLED": {
            "status": "VALID_REFERENCE_WITH_LIMITS",
            "information_condition": ("MODEL_WEIGHTS_ZERO_SHOT + TARGET_VALIDATION_TUNED_PREPROCESSOR"
                                      " -- the FlowState weights never saw these series, but the "
                                      "reconstruction lambda feeding them was chosen on target "
                                      "validation data, so the pipeline is not target-data-free."),
            "reconstruction_lambda": {ds: d["resampled"]["reconstruction_lambda"]
                                      for ds, d in fs["datasets"].items()},
        },
        "FLOWSTATE_NATIVE_RATE": per_ds,
        "counterexample": counterexamples,
        "contract_function": {
            "name": "can_form_hourly_mean_from_native(operation, r)",
            "INTERVAL_MEAN": {str(r): can_form_hourly_mean_from_native("INTERVAL_MEAN", r)
                              for r in (2, 3, 4, 6, 8, 12)},
            "END_BIN": {str(r): can_form_hourly_mean_from_native("END_BIN", r)
                        for r in (2, 3, 4, 6, 8, 12)},
        },
    }


# ----------------------------------------------- 21/22/23. interpretation audits


def mechanism_interpretation_audit(cells: pd.DataFrame) -> dict:
    diag = json.loads((ORIG / "mechanism_diagnostics.json").read_text(encoding="utf-8"))
    rows = []
    for key, v in diag["variants"].items():
        ds, arm, seed = key.split("/")
        if arm != "O" or not isinstance(v.get("WRONG_SUPPORT"), dict):
            continue
        for cond, base in v["BASE"].items():
            op, r = cond.split("|r")
            rows.append({"dataset": ds, "operation": op, "r": int(r), "seed": int(seed),
                         "base": base, "wrong_support": v["WRONG_SUPPORT"][cond],
                         "sensitivity_pct": 100.0 * (v["WRONG_SUPPORT"][cond] / base - 1.0)})
    sens = pd.DataFrame(rows).groupby(["dataset", "operation", "r"], as_index=False)[
        "sensitivity_pct"].mean()

    perf = cells.pivot_table(index=["dataset", "operation", "r"], columns="arm", values="primary")
    joined = []
    for _, s in sens.iterrows():
        idx = (s["dataset"], s["operation"], int(s["r"]))
        if idx in perf.index:
            joined.append({
                "dataset": idx[0], "operation": idx[1], "r": idx[2],
                "wrong_support_sensitivity_pct": float(s["sensitivity_pct"]),
                "O_vs_M_pct": relative_improvement(perf.loc[idx, "O"], perf.loc[idx, "M"]),
            })
    counter = [j for j in joined if j["wrong_support_sensitivity_pct"] > 1.0
               and j["O_vs_M_pct"] <= 0.0]
    return {
        "what_this_diagnostic_shows": (
            "A trained O model's predictions change when its inputs are re-encoded with the centre "
            "basis at inference. That is sensitivity to a train/inference representation mismatch."),
        "what_it_does_not_show": (
            "It does not show that the integrated basis is superior to the centre basis, because "
            "the comparison is between a model and a corrupted version of itself, not between two "
            "models each trained on its own representation."),
        "per_condition": joined,
        "direct_counterexamples": counter,
        "counterexample_reading": (
            "Where sensitivity is large yet O vs M is not positive, representation dependence and "
            "predictive superiority come apart in the same cell."),
    }


def width_interpretation_audit() -> dict:
    diag = json.loads((ORIG / "mechanism_diagnostics.json").read_text(encoding="utf-8"))
    rows = []
    for key, v in diag["variants"].items():
        ds, arm, seed = key.split("/")
        if not isinstance(v.get("WIDTH_MISMATCH"), dict):
            continue
        for cond, base in v["BASE"].items():
            op, r = cond.split("|r")
            rows.append({"dataset": ds, "arm": arm, "operation": op, "r": int(r),
                         "sensitivity_pct": 100.0 * (v["WIDTH_MISMATCH"][cond] / base - 1.0)})
    df = pd.DataFrame(rows)
    informative = df[df["operation"] == "INTERVAL_MEAN"]
    return {
        "allowed_statement": ("The explicit scalar width channel showed little sensitivity under the "
                              "WIDTH_MISMATCH diagnostic."),
        "withdrawn_statement": "The model does not use interval width.",
        "why": ("Resolution is available to the arms through the token count, the token spacing, the "
                "time basis and the operation embedding, so a flat width channel does not establish "
                "that interval width is unused."),
        "end_bin_rows_are_uninformative": ("For END_BIN the support is one base bin at every r, so the "
                                           "injected width equals the true width and the variant is "
                                           "the identity by construction."),
        "max_abs_sensitivity_pct_interval_mean": float(informative["sensitivity_pct"].abs().max()),
        "per_arm_max_abs_pct": informative.groupby("arm")["sensitivity_pct"].apply(
            lambda s: float(s.abs().max())).to_dict(),
    }


def reconstruction_interpretation_audit() -> dict:
    rq = json.loads((ORIG / "reconstruction_quality.json").read_text(encoding="utf-8"))
    scope = {}
    for ds, d in rq["datasets"].items():
        sel = [r for r in d["rows"] if r["lambda"] != 1e-4]
        scope[ds] = {
            "channel_tested": d["channel"], "n_windows": d["n_windows"],
            "n_channels_in_dataset_total": 3 if ds == "jena" else 4,
            "true_base_grid_std": d["true_base_grid_std"],
            "reconstruction_rmse_at_selected_lambda": [
                min(r["reconstruction_rmse_vs_true_base_grid"] for r in sel),
                max(r["reconstruction_rmse_vs_true_base_grid"] for r in sel)],
        }
    return {
        "diagnostic_scope": scope,
        "scope_limits": ("One channel per dataset and 200 test windows, at the selected lambda only. "
                         "It is not a dataset-wide smoothness measurement."),
        "allowed_statement": ("The dataset-level performance reversal is consistent with the "
                              "reconstruction-quality diagnostic: the tested Jena channel was "
                              "reconstructed far more accurately than the tested UCI channel."),
        "withdrawn_statement": "Reconstruction quality caused the forecasting reversal.",
        "why": ("No intervention was run on reconstruction quality, so no causal claim is supported. "
                "The R arm also differs from M and O in its whole tokenisation path, not only in how "
                "well it reconstructs."),
        "M_vs_R_confound": ("M and R differ in more than metadata: R reconstructs to 288 base bins and "
                            "forecasts from that grid, while M reads the coarse tokens directly. The "
                            "metadata contribution and the tokenisation-path contribution are not "
                            "separated by this pilot."),
    }


# ----------------------------------------------------- 28/30. selection and loss


def checkpoint_selection_audit() -> pd.DataFrame:
    fm = pd.read_csv(ORIG / "fit_manifest.csv")
    keep = ["dataset", "arm", "model_seed", "best_update", "best_val_primary",
            "final_val_primary", "best_is_first", "best_is_last", "last_interval_improving"]
    fm = fm[keep].copy()
    fm["final_over_best"] = fm["final_val_primary"] / fm["best_val_primary"]
    pairs = []
    for ds in DATASETS:
        for seed in SEEDS:
            sub = fm[(fm.dataset == ds) & (fm.model_seed == seed)].set_index("arm")
            pairs.append({"dataset": ds, "model_seed": seed,
                          "M_best_update": int(sub.loc["M", "best_update"]),
                          "O_best_update": int(sub.loc["O", "best_update"]),
                          "M_O_same_best_update": bool(
                              sub.loc["M", "best_update"] == sub.loc["O", "best_update"])})
    fm.attrs["pairs"] = pairs
    return fm


def loss_component_audit(sa: pd.DataFrame) -> dict:
    d = sa[sa["split"] == "test"]
    cells = d.groupby(["dataset", "operation", "r", "arm"], as_index=False)[
        ["primary", "mse10", "mse60"]].mean()
    out = {}
    for role, rs in (("seen", SEEN_R), ("unseen_interpolation", INTERP_R),
                     ("unseen_extrapolation", EXTRAP_R)):
        sub = cells[cells["r"].isin(rs)]
        entry = {}
        for metric in ("primary", "mse10", "mse60"):
            piv = sub.pivot_table(index=["dataset", "operation", "r"], columns="arm", values=metric)
            entry[metric] = {f"{a}_vs_{b}_macro_pct": float(np.mean(
                [relative_improvement(row[a], row[b]) for _, row in piv.iterrows()]))
                for a, b in (("O", "M"), ("O", "R"), ("M", "R"))}
        out[role] = entry
    return {
        "basis": "runs/oa_resolution_pilot_v1/errors/*.npy, seed-averaged per logical key",
        "purpose": ("Whether the near-zero primary contrast hides a 10-minute gain cancelling a "
                    "60-minute loss, or whether both components are near zero. Not a new gate."),
        "roles": out,
    }


# ------------------------------------------------------------------------ main


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base_sha = sh("git rev-parse origin/oa-resolution-pilot-v1")

    before = hash_originals()
    write("original_artifact_manifest.json",
          {"audited_base_sha": base_sha, "n_files": len(before), "files": before})

    write("git_context.json", {
        "audited_base_sha": base_sha,
        "audit_branch": sh("git rev-parse --abbrev-ref HEAD"),
        "audit_head": sh("git rev-parse HEAD"),
        "origin_main": sh("git rev-parse origin/main"),
        "remote": sh("git remote get-url origin"),
        "working_tree_clean_of_tracked_changes": sh("git status --porcelain --untracked-files=no") == "",
        "runs_dir_gitignored": True,
        "local_raw_errors_present": len(list((RUNS / "errors").glob("*.npy"))),
    })

    print("Q1 arithmetic revalidation", flush=True)
    arith = arithmetic_revalidation()
    write("arithmetic_revalidation.json", arith)
    print(f"   {arith['status']}  max delta {arith['max_abs_delta_pp']:.3e} pp", flush=True)

    print("raw error inventory", flush=True)
    inv = raw_error_inventory(base_sha)
    write("raw_error_inventory.json", inv)
    print(f"   {inv['status']}", flush=True)

    if inv["n_local"] != 12:
        write("bootstrap_revalidation.json", {"status": "RAW_ERROR_ARTIFACT_NOT_AVAILABLE"})
        print("   raw errors missing; bootstrap not reproduced", flush=True)
        return 1

    print("rebuilding seed-averaged keys from raw errors", flush=True)
    sa = load_seed_averaged()

    stats = sufficient_stats(sa, INTERP_R + EXTRAP_R)
    stats.to_csv(OUT / "bootstrap_block_sufficient_stats.csv", index=False)

    print("bootstrap revalidation", flush=True)
    stored = json.loads((ORIG / "bootstrap.json").read_text(encoding="utf-8"))
    boot = {"naming_correction": {
        "original_docstring": "Moving-block bootstrap over evaluation origins",
        "actual_implementation": ("fixed non-overlapping 7-day blocks from origin // (7 * 144), "
                                  "whole blocks drawn with replacement, paired across arms"),
        "correct_name": "paired 7-day time-block (cluster) bootstrap"}}
    interp_stats = stats[stats["r"].isin(INTERP_R)]
    worst = 0.0
    for a, b in (("O", "M"), ("O", "R"), ("M", "R")):
        mine = bootstrap_from_stats(interp_stats, a, b)
        key = f"{a}_vs_{b}_unseen_interpolation"
        theirs = stored.get(key)
        entry = {"recomputed": mine}
        if theirs:
            deltas = {k: abs(mine[k] - theirs[k]) for k in ("mean", "lower95", "upper95")}
            worst = max(worst, max(deltas.values()))
            entry["stored"] = {k: theirs[k] for k in ("mean", "lower95", "upper95")}
            entry["abs_delta_pp"] = deltas
        boot[key] = entry
    boot["O_vs_M_unseen_extrapolation_exploratory"] = {
        "recomputed": bootstrap_from_stats(stats[stats["r"].isin(EXTRAP_R)], "O", "M")}
    boot["max_abs_delta_pp"] = worst
    boot["tolerance_pp"] = 1e-6
    boot["status"] = "BOOTSTRAP_REPRODUCED" if worst <= 1e-6 else "BOOTSTRAP_DIFFERS"
    write("bootstrap_revalidation.json", boot)
    print(f"   {boot['status']}  max delta {worst:.3e} pp", flush=True)

    print("per-seed effects and r=12 table", flush=True)
    full = pd.concat([
        pd.DataFrame(np.load(RUNS / "errors" / f"errors_{ds}_{arm}_{seed}.npy")).assign(dataset=ds)
        for ds in DATASETS for arm in ARMS for seed in SEEDS], ignore_index=True)
    for col in ("split", "operation", "role", "arm"):
        full[col] = full[col].astype(str)
    full["primary"] = 0.5 * (full["SE10"] / full["count10"]) + 0.5 * (full["SE60"] / full["count60"])
    per_seed = per_seed_cells(full)

    paired_seed_effects(per_seed).to_csv(OUT / "paired_seed_effects.csv", index=False)

    r12 = r12_seed_table(per_seed)
    r12.to_csv(OUT / "r12_seed_table_corrected.csv", index=False)

    cells = pd.read_csv(ORIG / "metrics.csv")
    decomp = r12_decomposition(cells, r12)
    prov = status_md_r12_provenance(r12)
    # The narrative table was wrong, but is the aggregate wrong too?  metrics.csv is
    # rebuilt from the per-seed cells here; if it agrees, the bug never reached the
    # scientific result.
    prov["scope"] = ("REPORTING_CELL_MAPPING_BUG_ONLY"
                     if decomp["metrics_csv_vs_per_seed_mean"]["aggregate_is_consistent"]
                     else "SCIENTIFIC_RESULT_AFFECTED")
    decomp["status_md_r_column_provenance"] = prov
    write("r12_effect_decomposition.json", decomp)

    print("loss components", flush=True)
    write("loss_component_audit.json", loss_component_audit(sa))

    print("unit contract recheck", flush=True)
    unit = unit_contract_recheck()
    write("unit_contract_recheck.json", unit)
    print(f"   {unit['status']}", flush=True)

    print("checkpoint selection", flush=True)
    ck = checkpoint_selection_audit()
    ck.to_csv(OUT / "checkpoint_selection_audit.csv", index=False)
    write("checkpoint_pairs.json", {"m_o_pairs": ck.attrs["pairs"]})

    print("FlowState semantics", flush=True)
    write("flowstate_reference_semantic_audit.json", flowstate_semantic_audit())

    print("interpretation audits", flush=True)
    write("mechanism_interpretation_audit.json", mechanism_interpretation_audit(cells))
    write("width_interpretation_audit.json", width_interpretation_audit())
    write("reconstruction_interpretation_audit.json", reconstruction_interpretation_audit())

    fm = pd.read_csv(ORIG / "fit_manifest.csv")
    sched = fm.groupby(["dataset", "model_seed"])["schedule_sha"].nunique()
    schedule_ok = bool((sched == 1).all())

    after = hash_originals()
    changed = [k for k in before if before[k]["sha256"] != after.get(k, {}).get("sha256")]
    write("original_artifact_immutability.json", {
        "files_checked": len(before), "changed": changed,
        "status": "ORIGINAL_ARTIFACTS_UNCHANGED" if not changed
                  else "ORIGINAL_ARTIFACT_MUTATION_HARD_STOP"})

    verdict = {
        "audit_name": "OA-RESOLUTION-PILOT-v1-AUDIT-CLOSURE-v1",
        "audited_base_sha": base_sha,
        "scientific_decision_original": json.loads(
            (ORIG / "verdict.json").read_text(encoding="utf-8"))["scientific_decision"],
        "audit_recommendation_current_implementation":
            "STOP_SCALING_CURRENT_INTERVAL_INTEGRATED_FOURIER_O",
        "broader_topic_status": "OPEN_NOT_DIRECTLY_TESTED",
        "core_model_fits_in_audit": 0,
        "primary_result_reproduced": arith["status"] == "ARITHMETIC_REPRODUCTION_OK",
        "bootstrap_reproduced_from_local_raw_errors": boot["status"] == "BOOTSTRAP_REPRODUCED",
        "flowstate_native_end_bin_status": "INVALID_FOR_HOURLY_MEAN_COMPARISON",
        "r12_reporting_bug_scope": decomp["status_md_r_column_provenance"]["scope"],
        "raw_error_git_status": inv["status"],
        "train_schedule_fairness": "OK" if schedule_ok else "TRAIN_SCHEDULE_FAIRNESS_FAIL",
        "original_artifacts_unchanged": not changed,
        "bootstrap_name_correction": "paired 7-day time-block (cluster) bootstrap",
        "unit_contract_status": unit["status"],
    }
    write("audit_verdict.json", verdict)

    print("=" * 68, flush=True)
    for k in ("scientific_decision_original", "audit_recommendation_current_implementation",
              "broader_topic_status", "primary_result_reproduced",
              "bootstrap_reproduced_from_local_raw_errors", "r12_reporting_bug_scope",
              "raw_error_git_status", "original_artifacts_unchanged"):
        print(f"  {k}: {verdict[k]}", flush=True)
    print("=" * 68, flush=True)
    return 0


def from_stats_only() -> int:
    """Section 33.  Regenerate every interval from the committed CSV alone, with no
    raw error arrays present.  This is what makes the audit reproducible from the
    repository."""
    stats = pd.read_csv(OUT / "bootstrap_block_sufficient_stats.csv")
    out = {"source": "audit_closure_v1/bootstrap_block_sufficient_stats.csv",
           "raw_errors_needed": False, "contrasts": {}}
    for a, b in (("O", "M"), ("O", "R"), ("M", "R")):
        out["contrasts"][f"{a}_vs_{b}_unseen_interpolation"] = bootstrap_from_stats(
            stats[stats["r"].isin(INTERP_R)], a, b)
    out["contrasts"]["O_vs_M_unseen_extrapolation_exploratory"] = bootstrap_from_stats(
        stats[stats["r"].isin(EXTRAP_R)], "O", "M")
    write("bootstrap_from_committed_stats.json", out)
    for k, v in out["contrasts"].items():
        print(f"  {k}: {v['mean']:+.4f} [{v['lower95']:+.4f}, {v['upper95']:+.4f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(from_stats_only() if "--from-stats-only" in sys.argv else main())
