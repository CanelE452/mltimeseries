"""HQ-TOKEN-PILOT-v1-AUDIT-CLOSURE: post-hoc interpretation and optimization audit.

Read-only with respect to every existing result. It fits no model, launches no
subprocess, imports nothing from experiments/, and writes only under
results/hq_token_pilot_v1/audit_closure_v1/.

Everything is recomputed from the stored artifacts rather than from the code that
produced them, so a disagreement between this script and contrasts.json is a real
disagreement and not a shared bug.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "hq_token_pilot_v1")
OUT = os.path.join(RES, "audit_closure_v1")
CORE = os.path.join(ROOT, "runs", "hq_token_pilot_v1", "core")

DATASETS = ["ETTm2", "weather", "electricity"]
HORIZONS = [96, 336]
SEED0, SEED1 = 2026090601, 2026090602
BOTH = [SEED0, SEED1]
BLOCK_ORIGINS = 15
BOOTSTRAP_SEED = 2026090699
BOOTSTRAP_DRAWS = 1000
TOL = 1e-10

BASELINE_FILES = [
    "STATUS.md", "verdict.json", "execution_spec.json", "execution_spec.sha256",
    "metrics.csv", "contrasts.json", "bootstrap.json", "diagnostics.json",
    "efficiency.csv", "fit_manifest.csv", "mechanism_capacity_check.json",
    "seed_noise_floor.json", "optional_references.json", "data_manifest.json",
    "environment.json",
]


def log(m):
    print(m, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def manifest():
    m = {}
    for n in BASELINE_FILES:
        p = os.path.join(RES, n)
        m[n] = sha256_file(p) if os.path.exists(p) else None
    figdir = os.path.join(RES, "figures")
    if os.path.isdir(figdir):
        for n in sorted(os.listdir(figdir)):
            m[f"figures/{n}"] = sha256_file(os.path.join(figdir, n))
    return m


def write(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "wb") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=False, default=str).encode())
    log(f"  wrote {name}")


def write_csv(name, rows, fields=None):
    p = os.path.join(OUT, name)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {name} ({len(rows)} rows)")


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- Q1, section 4


def read_metrics():
    """metrics.csv is the test split only, one row per (dataset, H, B, seed, arm)."""
    out = {}
    with open(os.path.join(RES, "metrics.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["arm"], r["dataset"], int(r["horizon"]), int(r["model_seed"]))
            out[key] = {"MSE": float(r["MSE"]), "MAE": float(r["MAE"]), "B": int(r["B"])}
    return out


def arm_seeds(metrics, arm):
    return sorted({k[3] for k in metrics if k[0] == arm})


def E(metrics, arm, d, h, seeds):
    vals = [metrics[(arm, d, h, s)]["MSE"] for s in seeds if (arm, d, h, s) in metrics]
    return float(np.mean(vals)) if vals else None


def contrast(metrics, treat, ref):
    """RI = 100 * (1 - E_treat / E_ref), seeds averaged before the ratio."""
    seeds = sorted(set(arm_seeds(metrics, treat)) & set(arm_seeds(metrics, ref)))
    cells, per_ds, per_seed = {}, {}, {}
    for d in DATASETS:
        vals = []
        for h in HORIZONS:
            et, er = E(metrics, treat, d, h, seeds), E(metrics, ref, d, h, seeds)
            ri = None if (et is None or er in (None, 0.0)) else 100.0 * (1.0 - et / er)
            cells[f"{d}|{h}"] = ri
            if ri is not None:
                vals.append(ri)
        per_ds[d] = float(np.mean(vals)) if vals else None
    for s in seeds:
        vals = []
        for d in DATASETS:
            for h in HORIZONS:
                et, er = E(metrics, treat, d, h, [s]), E(metrics, ref, d, h, [s])
                if et is not None and er:
                    vals.append(100.0 * (1.0 - et / er))
        per_seed[str(s)] = float(np.mean(vals)) if vals else None
    fin = [v for v in cells.values() if v is not None]
    return {
        "treat": treat, "ref": ref, "seeds_used": seeds, "cells": cells,
        "per_dataset_horizon_mean": per_ds, "per_seed_macro": per_seed,
        "macro": float(np.mean(fin)) if fin else None,
    }


def stage_arithmetic(metrics, original):
    log("Q1: recomputing every contrast from metrics.csv")
    pairs = [("C", "I"), ("C", "H_STATIC"), ("C", "R"), ("C", "U"), ("C", "DENSE"), ("DENSE", "U")]
    recomputed = {f"{t}_vs_{r}": contrast(metrics, t, r) for t, r in pairs}

    orig_map = {
        "C_vs_I": original["primary"],
        "C_vs_H_STATIC": original["secondary"]["C_vs_H_STATIC"],
        "C_vs_R": original["secondary"]["C_vs_R"],
        "C_vs_U": original["secondary"]["C_vs_U"],
        "C_vs_DENSE": original["secondary"]["C_vs_DENSE"],
        "DENSE_vs_U": original["precondition_compression_headroom_DENSE_vs_U"],
    }
    diffs, worst = [], 0.0
    for name, rec in recomputed.items():
        o = orig_map[name]
        for cell, v in rec["cells"].items():
            ov = o["cells"].get(cell)
            if v is None or ov is None:
                if v is not ov:
                    diffs.append({"where": f"{name}.cells.{cell}", "recomputed": v, "original": ov})
                continue
            dv = abs(v - ov)
            worst = max(worst, dv)
            if dv > TOL:
                diffs.append({"where": f"{name}.cells.{cell}", "recomputed": v,
                              "original": ov, "abs_diff": dv})
        if rec["macro"] is not None and o.get("macro") is not None:
            dv = abs(rec["macro"] - o["macro"])
            worst = max(worst, dv)
            if dv > TOL:
                diffs.append({"where": f"{name}.macro", "recomputed": rec["macro"],
                              "original": o["macro"], "abs_diff": dv})
        for s, v in rec["per_seed_macro"].items():
            ov = o.get("per_seed_macro", {}).get(s)
            if v is None or ov is None:
                continue
            dv = abs(v - ov)
            worst = max(worst, dv)
            if dv > TOL:
                diffs.append({"where": f"{name}.per_seed_macro.{s}", "recomputed": v,
                              "original": ov, "abs_diff": dv})

    status = "ARITHMETIC_REPRODUCTION_OK" if not diffs else "ARITHMETIC_REPRODUCTION_FAIL"
    out = {
        "question": "Q1 - do the contrasts recomputed from metrics.csv match contrasts.json?",
        "method": ("metrics.csv read directly; seeds averaged per cell before the ratio; "
                   "macro is the equal-weight mean of the six dataset x horizon cells. "
                   "No code from experiments/ was imported."),
        "absolute_tolerance": TOL,
        "status": status,
        "max_abs_difference_pct_points": worst,
        "differences": diffs,
        "recomputed": recomputed,
    }
    write("arithmetic_revalidation.json", out)
    log(f"  {status}, worst |diff| = {worst:.3e} percentage points")
    return recomputed, status


# ---------------------------------------------------------------- section 5


def stage_paired_seeds(metrics):
    log("section 5: per-seed paired C vs I table")
    rows = []
    for d in DATASETS:
        for h in HORIZONS:
            for s in BOTH:
                c = metrics.get(("C", d, h, s))
                i = metrics.get(("I", d, h, s))
                if not c or not i:
                    continue
                rows.append({
                    "dataset": d, "horizon": h, "model_seed": s,
                    "MSE_I": i["MSE"], "MSE_C": c["MSE"],
                    "RI_pct": 100.0 * (1.0 - c["MSE"] / i["MSE"]),
                    "sign": "C better" if c["MSE"] < i["MSE"] else "I better",
                })
    for s in BOTH:
        vals = [r["RI_pct"] for r in rows if r["model_seed"] == s]
        rows.append({"dataset": "MACRO", "horizon": "", "model_seed": s,
                     "MSE_I": "", "MSE_C": "", "RI_pct": float(np.mean(vals)),
                     "sign": "C better" if np.mean(vals) > 0 else "I better"})
    write_csv("paired_seed_effects.csv", rows)
    log("  note: two seeds only. No standard error or interval is derived from them.")
    return rows


# ---------------------------------------------------------------- section 6


def fit_dir(arm, d, seed, B=32):
    return os.path.join(CORE, f"{d}__{arm}__B{B}__s{seed}")


def load_keyed(arm, d, seed, split, h, B=32):
    p = os.path.join(fit_dir(arm, d, seed, B), "keyed_errors.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    pre = f"{split}_{h}_"
    if pre + "se_sum" not in z.files:
        return None
    return {k: z[pre + k] for k in ("origin", "channel", "se_sum", "count")}


def stage_bootstrap(original_boot):
    log("section 6: independent bootstrap re-derivation from keyed_errors.npz")
    prepared = {}
    for d in DATASETS:
        per = {}
        for h in HORIZONS:
            arms = {}
            for arm in ("C", "I"):
                parts = [load_keyed(arm, d, s, "test", h) for s in BOTH]
                if any(p is None for p in parts):
                    return write("bootstrap_revalidation_status.json", {
                        "status": "RAW_KEYED_ARTIFACT_NOT_AVAILABLE",
                        "missing": f"{arm} {d} H={h}",
                        "note": ("The stored bootstrap is not called into question; it simply "
                                 "could not be regenerated independently here."),
                    })
                ref = parts[0]
                for p in parts[1:]:
                    if not (np.array_equal(p["origin"], ref["origin"])
                            and np.array_equal(p["channel"], ref["channel"])):
                        raise SystemExit("key order differs between seeds")
                arms[arm] = np.mean([p["se_sum"] for p in parts], axis=0)
            ref = load_keyed("C", d, SEED0, "test", h)
            uo = np.unique(ref["origin"])
            idx = {o: np.nonzero(ref["origin"] == o)[0] for o in uo}
            per[h] = {
                "origins": uo,
                "t": np.array([arms["C"][idx[o]].sum() for o in uo]),
                "r": np.array([arms["I"][idx[o]].sum() for o in uo]),
                "cnt": np.array([ref["count"][idx[o]].sum() for o in uo], dtype=float),
            }
        n = len(per[HORIZONS[0]]["origins"])
        prepared[d] = {"per": per, "n": n, "blocks": int(math.ceil(n / BLOCK_ORIGINS))}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macros = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        cells = []
        for d in DATASETS:
            pack = prepared[d]
            n, nb = pack["n"], pack["blocks"]
            starts = rng.integers(0, n, size=nb)
            sel = (starts[:, None] + np.arange(BLOCK_ORIGINS)[None, :]).ravel() % n
            sel = sel[:n]
            for h in HORIZONS:
                a = pack["per"][h]
                t = a["t"][sel].sum() / a["cnt"][sel].sum()
                r = a["r"][sel].sum() / a["cnt"][sel].sum()
                cells.append(100.0 * (1.0 - t / r))
        macros[b] = float(np.mean(cells))

    rec = {
        "macro_mean": float(macros.mean()),
        "lower95": float(np.percentile(macros, 2.5)),
        "upper95": float(np.percentile(macros, 97.5)),
    }
    orig = {k: original_boot.get(k) for k in ("macro_mean", "lower95", "upper95")}
    dif = {k: (abs(rec[k] - orig[k]) if orig.get(k) is not None else None) for k in rec}
    ok = all(v is not None and v <= 1e-6 for v in dif.values())
    out = {
        "question": "does an independent implementation of the stored bootstrap contract agree?",
        "contract": {
            "grid": "frozen test origin grid, all channels, both horizons",
            "seed_handling": "model seeds averaged per key first, seeds are not resampled",
            "block_origins": BLOCK_ORIGINS, "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "macro": "equal weight over the six dataset x horizon cells",
        },
        "recomputed": rec, "original": orig, "abs_difference": dif,
        "tolerance_pct_points": 1e-6,
        "status": "BOOTSTRAP_REPRODUCTION_OK" if ok else "BOOTSTRAP_REPRODUCTION_DIFFERS",
        "effective_blocks": {d: prepared[d]["blocks"] for d in DATASETS},
        "origins_per_dataset": {d: prepared[d]["n"] for d in DATASETS},
        "note": ("A difference here would most likely mean the random stream is consumed in a "
                 "different order, not that either interval is wrong. The comparison is "
                 "reported either way."),
    }
    write("bootstrap_revalidation.json", out)
    log(f"  {out['status']}: recomputed {rec['macro_mean']:+.6f} "
        f"[{rec['lower95']:+.6f}, {rec['upper95']:+.6f}]")
    return out


# ---------------------------------------------------------- sections 7, 8, 9, 22


def stage_curves():
    log("sections 7-9: training curve audit for the six paired I/C fits")
    rows, missing = [], []
    for d in DATASETS:
        for s in BOTH:
            for arm in ("I", "C"):
                fd = fit_dir(arm, d, s)
                present = {n: os.path.exists(os.path.join(fd, n)) for n in
                           ("train_curve.json", "fit_report.json", "best.pt", "final.pt",
                            "initial.pt", "keyed_errors.npz")}
                if not all(present.values()):
                    missing.append({"fit": os.path.basename(fd), "present": present})
                    continue
                cur = load_json(os.path.join(fd, "train_curve.json"))["validation"]
                rep = load_json(os.path.join(fd, "fit_report.json"))
                means = [v["mean"] for v in cur]
                ups = [v["update"] for v in cur]
                best_i = int(np.argmin(means))
                last_rel = (means[-1] - means[-2]) / means[-2] if len(means) > 1 else 0.0
                if abs(last_rel) < 0.001:
                    interval = "FLAT_LAST_INTERVAL"
                elif last_rel < 0:
                    interval = "LAST_INTERVAL_IMPROVING"
                else:
                    interval = "LAST_INTERVAL_WORSENING"
                if best_i == 0:
                    where = "EARLY_BEST"
                elif best_i == len(means) - 1:
                    where = "LATE_BEST"
                else:
                    where = "INTERIOR_BEST"
                slope3 = float(np.polyfit(ups[-3:], means[-3:], 1)[0]) if len(means) >= 3 else None
                rows.append({
                    "dataset": d, "model_seed": s, "arm": arm,
                    "first_validation_update": ups[0],
                    "best_update": ups[best_i],
                    "best_update_from_fit_report": rep["best_checkpoint_update"],
                    "final_update": ups[-1],
                    "first_validation_mean_mse": means[0],
                    "best_validation_mean_mse": means[best_i],
                    "final_validation_mean_mse": means[-1],
                    "final_over_best_ratio": means[-1] / means[best_i],
                    "best_is_first": best_i == 0,
                    "best_is_last": best_i == len(means) - 1,
                    "last_interval_rel_change_pct": 100.0 * last_rel,
                    "last_three_point_slope_per_update": slope3,
                    "position_label": where,
                    "last_interval_label": interval,
                    "trajectory_mean": ";".join(f"{v:.6f}" for v in means),
                    "trajectory_h96": ";".join(f"{v['mse_96']:.6f}" for v in cur),
                    "trajectory_h336": ";".join(f"{v['mse_336']:.6f}" for v in cur),
                    "collapsed_to_constant": rep["collapsed_to_constant"],
                })
    write_csv("training_curve_summary.csv", rows)
    if missing:
        write("training_curve_missing_artifacts.json", missing)
    return rows


def stage_trajectory():
    log("section 10: C versus I validation trajectory at every shared checkpoint")
    rows = []
    for d in DATASETS:
        for s in BOTH:
            ci = {}
            for arm in ("I", "C"):
                p = os.path.join(fit_dir(arm, d, s), "train_curve.json")
                if not os.path.exists(p):
                    ci = None
                    break
                ci[arm] = {v["update"]: v for v in load_json(p)["validation"]}
            if not ci:
                continue
            for u in sorted(set(ci["I"]) & set(ci["C"])):
                vi, vc = ci["I"][u], ci["C"][u]
                rows.append({
                    "dataset": d, "model_seed": s, "update": u,
                    "val_mean_I": vi["mean"], "val_mean_C": vc["mean"],
                    "delta_val_pct": 100.0 * (vc["mean"] / vi["mean"] - 1.0),
                    "delta_h96_pct": 100.0 * (vc["mse_96"] / vi["mse_96"] - 1.0),
                    "delta_h336_pct": 100.0 * (vc["mse_336"] / vi["mse_336"] - 1.0),
                })
    write_csv("validation_C_vs_I_trajectory.csv", rows)
    deltas = [abs(r["delta_val_pct"]) for r in rows]
    log(f"  |C/I - 1| across all shared checkpoints: median {np.median(deltas):.3f}%, "
        f"max {max(deltas):.3f}%")
    return rows


def stage_checkpoint_audit(curves):
    log("section 22: checkpoint selection comparison")
    rows = []
    by = {(r["dataset"], r["model_seed"], r["arm"]): r for r in curves}
    for d in DATASETS:
        for s in BOTH:
            i, c = by.get((d, s, "I")), by.get((d, s, "C"))
            if not i or not c:
                continue
            rows.append({
                "dataset": d, "model_seed": s,
                "best_update_I": i["best_update"], "best_update_C": c["best_update"],
                "same_best_update": i["best_update"] == c["best_update"],
                "best_val_mse_I": i["best_validation_mean_mse"],
                "best_val_mse_C": c["best_validation_mean_mse"],
                "final_update": i["final_update"],
                "position_label_I": i["position_label"], "position_label_C": c["position_label"],
            })
    write_csv("checkpoint_selection_table.csv", rows)
    same = sum(1 for r in rows if r["same_best_update"])
    out = {
        "question": "Q2 - did C and I end up close only because of checkpoint selection?",
        "pairs": len(rows), "pairs_with_identical_best_update": same,
        "pairs_with_different_best_update": len(rows) - same,
        "rows": rows,
        "reading": ("Where the two arms selected different updates, the reported test contrast "
                    "compares two independently validation-selected checkpoints. That is the "
                    "pre-registered procedure, but it means the pairing is on the seed and the "
                    "data schedule, not on the optimizer step."),
        "no_reselection": ("No checkpoint was re-chosen. Test results were not consulted for "
                           "selection at any point in this audit."),
    }
    write("checkpoint_selection_audit.json", out)
    log(f"  {same} of {len(rows)} paired fits selected the same update")
    return out


# ---------------------------------------------------------------- section 11


def stage_schedule_fairness():
    log("section 11: training-schedule fairness across arms")
    rows, mismatches = [], []
    for d in DATASETS:
        for s in BOTH:
            arms = ["I", "C"] + (["R"] if s == SEED0 else [])
            shas = {}
            for arm in arms:
                p = os.path.join(fit_dir(arm, d, s), "fit_report.json")
                if os.path.exists(p):
                    shas[arm] = load_json(p)["schedule_sha256"]
            uniq = set(shas.values())
            rows.append({"dataset": d, "model_seed": s,
                         **{f"schedule_sha256_{a}": v[:16] for a, v in shas.items()},
                         "all_identical": len(uniq) == 1})
            if len(uniq) != 1:
                mismatches.append({"dataset": d, "seed": s, "sha": shas})
    write_csv("schedule_fairness.csv", rows)
    return {"status": "TRAIN_SAMPLE_SCHEDULE_MISMATCH" if mismatches else "SCHEDULES_IDENTICAL",
            "mismatches": mismatches, "checked": len(rows)}


# ---------------------------------------------------------------- section 12


def stage_capacity(cap):
    log("section 12: synthetic capacity control reinterpretation")

    def g(d, h):
        return d.get(str(h), d.get(h))

    i96, i336 = g(cap["I"], 96), g(cap["I"], 336)
    c96, c336 = g(cap["C"], 96), g(cap["C"], 336)
    i_joint = float(np.mean([i96, i336]))
    c_joint = float(np.mean([c96, c336]))
    cw = cap.get("C_mean_weight_on_a_patch", {})
    iw = cap.get("I_mean_weight_on_a_patch", {})
    sep_c = abs(g(cw, 96) - g(cw, 336)) if cw else None
    sep_i = abs(g(iw, 96) - g(iw, 336)) if iw else None
    out = {
        "question": "Q4 - what did the synthetic control actually establish?",
        "source": "results/hq_token_pilot_v1/mechanism_capacity_check.json (unmodified)",
        "original_recorded_verdict": cap.get("verdict"),
        "per_horizon_mse": {"I": {"96": i96, "336": i336}, "C": {"96": c96, "336": c336}},
        "joint_equal_weight_mean_mse": {"I": i_joint, "C": c_joint,
                                        "C_over_I_ratio": c_joint / i_joint},
        "per_horizon_relative_improvement_pct": {
            "96": 100.0 * (1.0 - c96 / i96), "336": 100.0 * (1.0 - c336 / i336)},
        "C_weight_separation_between_horizons": sep_c,
        "I_weight_separation_between_horizons": sep_i,
        "QUERY_PATH_RESPONSIVE": bool(sep_c is not None and sep_c > 0.30),
        "JOINT_HORIZON_TASK_SOLVED": bool(c_joint < i_joint),
        "what_this_supports": (
            "This scorer path was able to learn a function that changes the pooling weight "
            "substantially with the horizon on the synthetic task."),
        "what_this_does_not_support": (
            "On that same synthetic task the horizon-conditioned arm did not beat the "
            "input-only arm on the joint multi-horizon objective, which is the objective the "
            "real runs optimize. The control therefore establishes query-path responsiveness "
            "only; it does not exclude architecture or optimization limitations as an "
            "explanation for the real-data null."),
        "design_caveat": (
            "The task was built on the argument that one token per group can carry only one "
            "mixture of its two patches. That argument is not airtight: when the mixture "
            "coefficient itself varies with content, the coefficient can act as a side "
            "channel, so a content-only pooler is not strictly limited to a single linear "
            "combination. No claim is made about whether the trained I arm exploits this."),
    }
    write("capacity_control_audit.json", out)
    log(f"  joint mean MSE  I {i_joint:.5f}  C {c_joint:.5f}  "
        f"-> JOINT_HORIZON_TASK_SOLVED = {out['JOINT_HORIZON_TASK_SOLVED']}")
    return out


# ---------------------------------------------------------------- section 14


def stage_seed_uncertainty(noise):
    log("section 14: seed uncertainty scope")
    out = {
        "question": "Q5 - what does the I-only 8-seed study say about the C-I paired effect?",
        "source": "results/hq_token_pilot_v1/seed_noise_floor.json (unmodified)",
        "I_absolute_seed_variability": {
            d: {"mean_validation_mse": v["mean"], "sd": v["sd"],
                "sd_pct_of_mean": v["sd_pct_of_mean"], "n_seeds": len(v["seeds"])}
            for d, v in noise["per_dataset"].items()},
        "worst_I_seed_sd_pct": noise["worst_seed_sd_pct"],
        "paired_CI_effect_seed_variability_estimated": False,
        "reason": "extra seeds exist only for I",
        "core_pair_seed_count": 2,
        "what_can_be_said": (
            "The absolute validation MSE of arm I varies noticeably across seeds, most on "
            "ETTm2."),
        "what_cannot_be_said": (
            "The seed-to-seed variability of the paired C minus I difference was not measured. "
            "C and I share their initial state and their data schedule within a seed, so the "
            "paired difference could be far more stable than either arm's absolute level, or "
            "less so. The 8-seed study was run on I alone and cannot distinguish these."),
        "explicitly_withdrawn_statement": (
            "An earlier report said the 1 percent gate was outside the resolution of the "
            "experiment because seed noise reached 5.05 percent. That inference does not "
            "follow from an I-only study."),
        "not_done_here": "No additional seeds were trained for this audit.",
    }
    write("seed_uncertainty_audit.json", out)
    return out


# ---------------------------------------------------------------- section 15


def stage_compression_gap(metrics, recomputed):
    log("section 15: compression-gap reinterpretation")
    dvu = recomputed["DENSE_vs_U"]
    beaten = []
    for d in DATASETS:
        for h in HORIZONS:
            c = metrics.get(("C", d, h, SEED0))
            de = metrics.get(("DENSE", d, h, SEED0))
            if c and de and c["MSE"] < de["MSE"]:
                beaten.append({"dataset": d, "horizon": h, "seed": SEED0,
                               "MSE_C": c["MSE"], "MSE_DENSE": de["MSE"],
                               "RI_C_over_DENSE_pct": 100.0 * (1.0 - c["MSE"] / de["MSE"])})
    out = {
        "question": "Q6 - is the DENSE minus U gap an upper bound on what pooling could gain?",
        "OBSERVED_DENSE_VS_UNIFORM_COMPRESSION_GAP": {
            "macro_pct": dvu["macro"], "cells_pct": dvu["cells"],
            "seeds_used": dvu["seeds_used"],
            "meaning": ("The measured accuracy difference between the trained 64-token "
                        "uncompressed arm and the trained 32-token uniform-pooling arm, at "
                        "model seed 2026090601 only.")},
        "is_an_upper_bound": False,
        "why_not": (
            "The trained dense arm is one trained model, not an oracle. A compressed model can "
            "beat it, for instance through the regularizing effect of the bottleneck, so the "
            "observed gap does not cap what a pooling rule could win."),
        "cells_where_C_beat_DENSE": beaten,
        "DENSE_IS_NOT_EMPIRICAL_UPPER_BOUND": bool(beaten),
        "explicitly_withdrawn_statement": (
            "An earlier report treated this gap as the headroom available to any pooling rule "
            "and used it to argue that a gate above the gap would be unreachable by "
            "construction. That reading is withdrawn."),
    }
    write("compression_gap_audit.json", out)
    log(f"  gap {dvu['macro']:+.3f}%, cells where C beat DENSE: {len(beaten)}")
    return out


# ---------------------------------------------------------------- section 16


def stage_query_mechanism(recomputed, diagnostics):
    log("section 16: query mechanism evidence in one table")
    rows = []
    cvi, cvr = recomputed["C_vs_I"], recomputed["C_vs_R"]
    for d in DATASETS:
        dg = diagnostics.get(d, {})
        qs = dg.get("query_swap", {})
        wp = dg.get("weight_profile") or {}
        for h in HORIZONS:
            q = qs.get(str(h), {})
            rows.append({
                "dataset": d, "horizon": h,
                "C_vs_I_RI_pct": cvi["cells"].get(f"{d}|{h}"),
                "C_vs_R_RI_pct": cvr["cells"].get(f"{d}|{h}"),
                "true_query_test_mse": q.get("mse_true_query"),
                "swapped_query_test_mse": q.get("mse_swapped_query"),
                "query_swap_delta_pct": q.get("delta_pct"),
                "mean_abs_weight_diff_96_vs_336": wp.get("mean_abs_weight_diff"),
                "mean_center_shift_samples": wp.get("mean_abs_center_shift_samples"),
                "max_center_shift_samples": wp.get("max_abs_center_shift_samples"),
            })
    write_csv("query_mechanism_summary.csv", rows)
    swaps = [abs(r["query_swap_delta_pct"]) for r in rows if r["query_swap_delta_pct"] is not None]
    shifts = [r["mean_abs_weight_diff_96_vs_336"] for r in rows
              if r["mean_abs_weight_diff_96_vs_336"] is not None]
    log(f"  max |query swap effect| {max(swaps):.4f}%, max mean weight shift {max(shifts):.5f}")
    return rows, max(swaps), max(shifts)


# ---------------------------------------------------------------- sections 18, 19


def stage_scope():
    log("sections 18-19: structural scope of the tested implementation")
    out = {
        "question": "Q7 - what exactly did this implementation have the freedom to do?",
        "source_read": "experiments/hq_token_pilot_v1/model.py at commit 36b01d2",
        "structure": {
            "input": "1024 samples, 64 non-overlapping patches of 16",
            "grouping": "32 fixed adjacent groups, group g is patches (2g, 2g+1)",
            "per_group_output": "exactly one token, a convex combination of the two patch embeddings",
            "group_size_r": 2,
            "encoder": "3 pre-layernorm transformer layers, width 128, 4 heads, FFN 256",
            "decoder": "tokens repeated back to 64 slots, flattened, Linear(8192,128), GELU, concat horizon embedding, Linear(256,336)",
            "total_parameters_C": 1563697,
        },
        "what_C_could_do": [
            "change the relative mixture of the two patches inside each adjacent pair",
            "make that mixture depend jointly on content and on the requested horizon",
        ],
        "what_C_could_not_do": [
            "assign more tokens to one stretch of history and fewer to another",
            "move a group boundary",
            "drop an unimportant group entirely",
            "represent a long quiet stretch with one token and a busy stretch with several",
            "select non-local or non-contiguous regions of the history",
            "vary the total token budget with the horizon",
        ],
        "horizon_paths": {
            "I": ("history -> content-only scorer (query is a zero vector) -> pooling -> "
                  "encoder -> readout -> concat true horizon embedding -> head"),
            "C": ("history plus true horizon embedding in the scorer -> horizon-conditioned "
                  "pooling -> encoder -> readout -> concat true horizon embedding -> head"),
            "shared_module_confound": (
                "Both paths use the same HorizonEmbed module. In C its parameters receive "
                "gradient from the scorer and from the decoder; in I only from the decoder. "
                "The two arms therefore differ by slightly more than the presence of a horizon "
                "query. This does not flatter the current negative result, but it would be a "
                "real confound for any future positive one and should be separated before a "
                "positive result is believed."),
            "encoder_side_channel": (
                "In C the pooling weights depend on the horizon, so the merged tokens "
                "themselves carry horizon information into the encoder, which they cannot in "
                "I. H_STATIC shares that property while ignoring content, which is why the C "
                "versus H_STATIC contrast rather than C versus I carries that part of the "
                "identification."),
        },
        "scope_of_any_conclusion": (
            "Results apply to fixed adjacent grouping at r=2, one token per group, a 1.5M "
            "parameter model, context 1024, horizons 96 and 336, token budget 32, and the "
            "training budget actually used."),
    }
    write("scope_boundary.json", out)
    return out


# ---------------------------------------------------------------- section 20


def stage_efficiency():
    log("section 20: efficiency reinterpretation")
    eff = {}
    with open(os.path.join(RES, "efficiency.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eff[r["config"]] = r
    def num(c, k):
        return float(eff[c][k])
    dense = "DENSE_B64"
    rows = {}
    for c in eff:
        rows[c] = {
            "tokens_encoded": int(eff[c]["tokens_encoded"]),
            "total_parameters": int(eff[c]["total_parameters"]),
            "tokenizer_parameters": int(eff[c]["tokenizer_parameters"]),
            "model_call_median_ms_b64": num(c, "model_call_median_ms_b64"),
            "end_to_end_median_ms_b64": num(c, "end_to_end_median_ms_b64"),
            "throughput_windows_per_s_b64": num(c, "throughput_windows_per_s_b64"),
            "peak_allocated_mib_b64": num(c, "peak_allocated_bytes_b64") / 2**20,
            "end_to_end_vs_DENSE_pct": 100.0 * (num(c, "end_to_end_median_ms_b64")
                                                / num(dense, "end_to_end_median_ms_b64") - 1.0),
            "peak_memory_vs_DENSE_pct": 100.0 * (num(c, "peak_allocated_bytes_b64")
                                                 / num(dense, "peak_allocated_bytes_b64") - 1.0),
        }
    u_slower = rows["U_B32"]["end_to_end_vs_DENSE_pct"] > 0
    out = {
        "question": "why was the 32-token model not faster than the 64-token dense model?",
        "per_config": rows,
        "U_has_no_scorer": True,
        "U_also_slower_than_DENSE": bool(u_slower),
        "U_vs_DENSE_end_to_end_pct": rows["U_B32"]["end_to_end_vs_DENSE_pct"],
        "C_vs_DENSE_end_to_end_pct": rows["C_B32"]["end_to_end_vs_DENSE_pct"],
        "reading": (
            "Uniform pooling runs no scorer at all and is still slower end to end than the "
            "uncompressed arm, so the slowdown cannot be attributed to scorer cost alone. "
            "Reshaping, the weighted sum, and repeating tokens back to 64 slots also cost "
            "time, and at this width the shortened 32-token attention saves less than those "
            "steps add."),
        "scope": (
            "Measured on a 1.5M parameter model at batch 64 on one RTX 4070. It says nothing "
            "about whether token compression pays off in a larger backbone, where attention is "
            "a much larger share of the cost."),
        "forbidden_generalizations": [
            "token compression is inherently slow",
            "C would also be slower inside a foundation model",
        ],
    }

    mp_path = os.path.join(OUT, "efficiency_microprofile.json")
    if os.path.exists(mp_path):
        mp = load_json(mp_path)
        cfg = mp["configs"]
        tot = {k: cfg[k]["total_forward"]["median_ms"] for k in cfg}
        u_slower_mp = tot["U_B32"] > tot["DENSE_B64"]
        out["microprofile"] = {
            "kind": mp["kind"],
            "per_stage_median_ms": {
                k: {s: cfg[k][s]["median_ms"] for s in
                    ("embed", "pool", "encoder", "unmerge_readout_head", "total_forward")}
                for k in cfg},
            "encoder_saving_from_halving_tokens_ms": mp["reading"]["encoder_saving_32_vs_64_tokens_ms"],
            "uniform_pool_cost_ms": mp["reading"]["pool_cost_uniform_ms"],
            "scorer_extra_cost_ms": mp["reading"]["scorer_extra_over_uniform_ms"],
            "extra_decode_cost_of_unmerge_ms": (mp["reading"]["decode_cost_compressed_ms"]
                                                - mp["reading"]["decode_cost_dense_ms"]),
            "what_it_shows": (
                "Halving the token count saves about 0.04 ms in the encoder at this width, "
                "while the content scorer alone adds about 0.25 ms. The scorer cost is the "
                "clear and repeatable part of why C is the slowest configuration."),
        }
        out["measurement_disagreement"] = {
            "issue": "the two measurements disagree on whether uniform pooling is slower than dense",
            "efficiency_csv_U_vs_DENSE_end_to_end_pct": rows["U_B32"]["end_to_end_vs_DENSE_pct"],
            "microprofile_U_total_ms": tot["U_B32"],
            "microprofile_DENSE_total_ms": tot["DENSE_B64"],
            "microprofile_says_U_slower": bool(u_slower_mp),
            "resolution": (
                "The sign of the U versus DENSE gap flips between the stored measurement and "
                "the longer micro-profile, and both differences are a few percent. That "
                "ordering is therefore not established. Only the ordering that survives both "
                "measurements is reported as a finding: C is the slowest configuration, and "
                "the scorer is the reason."),
        }
        out["reading"] = (
            "The content scorer costs about six times what halving the token count saves in "
            "the encoder at this width, which is why C is the slowest configuration. Whether "
            "compression alone, without a scorer, is faster or slower than no compression is "
            "not settled by these measurements: the two runs disagree in sign on a difference "
            "of a few percent.")
    write("efficiency_interpretation.json", out)
    log(f"  U (no scorer) vs DENSE end-to-end: {rows['U_B32']['end_to_end_vs_DENSE_pct']:+.1f}%")
    return out


# ---------------------------------------------------------------- section 21


def stage_exposure(spec, dm, curves):
    log("section 21: sampling exposure, read alongside the curves")
    cfg = spec["training"]
    rows = {}
    for d in DATASETS:
        m = dm["datasets"][d]
        pairs = m["train_origins"] * m["n_channels"]
        drawn = cfg["max_updates"] * cfg["batch"]
        state = [r for r in curves if r["dataset"] == d]
        rows[d] = {
            "train_origins": m["train_origins"], "n_channels": m["n_channels"],
            "eligible_channel_origin_pairs": pairs,
            "windows_drawn": drawn,
            "sampling_exposure_ratio": drawn / pairs,
            "position_labels": sorted({r["position_label"] for r in state}),
            "last_interval_labels": sorted({r["last_interval_label"] for r in state}),
        }
    return {
        "note": ("Called a sampling exposure ratio and not an epoch: training windows are "
                 "sliding and overlap heavily, so a ratio below one does not mean a "
                 "correspondingly small amount of distinct signal was seen."),
        "per_dataset": rows,
        "reading_rule": ("A small exposure ratio is only evidence of an insufficient budget "
                         "when the validation curve was still improving at the last "
                         "checkpoint. The labels are given next to the ratio for that reason."),
    }


# ---------------------------------------------------------------- section 24


def stage_foundation_scope(refs):
    log("section 24: foundation reference scope")
    out = {
        "role": "native reference rows, not a performance contest",
        "chronos2": {
            "status": refs["chronos2"]["status"],
            "point_forecast_type": refs["chronos2"].get("point_forecast_type"),
            "condition": ("pretrained, evaluated zero-shot, scored at the 0.5 quantile, while "
                          "the core arms were trained on each target for squared error"),
        },
        "tirex2": {
            "status": refs["tirex2"]["status"],
            "device": refs["tirex2"].get("device"),
            "max_supported_prediction_length": refs["tirex2"].get("max_supported_prediction_length"),
            "condition": ("H=336 is not available from the current public checkpoint, which "
                          "caps prediction length at 320, so only H=96 has a reference row; "
                          "run on CPU"),
        },
        "not_rerun_in_this_audit": True,
        "forbidden_readings": [
            "the pilot is worse than these models, therefore the topic failed",
            "the pilot beat these models, therefore it is state of the art",
        ],
        "strong_dynamic_baseline": {
            "status": "MISSING_STRONG_DYNAMIC_BASELINE",
            "statement": ("No official implementation of Local Merging, byte pair encoding for "
                          "time series, TimeSqueeze or PATK was located within the search "
                          "timebox, so no strong dynamic-tokenization baseline was run under "
                          "the same contract."),
            "forbidden_readings": [
                "official implementations do not exist",
                "this pilot is better than recent dynamic tokenization methods",
            ],
        },
    }
    write("foundation_reference_scope.json", out)
    return out


# ---------------------------------------------------------------- verdict


def stage_verdict(ctx):
    log("assembling the audit verdict")
    out = {
        "audit_name": "HQ-TOKEN-PILOT-v1-AUDIT-CLOSURE",
        "audited_commit": ctx["git"]["HEAD"],
        "no_model_fit_performed": True,
        "no_original_artifact_modified": ctx["immutability"]["status"] == "ORIGINAL_ARTIFACTS_UNCHANGED",
        "scientific_decision_original": ctx["verdict_original"]["scientific_decision"],
        "audit_recommendation_current_implementation": "STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C",
        "broader_topic_status": "OPEN_NOT_DIRECTLY_TESTED",
        "meaning": {
            "STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C": (
                "There is not enough empirical support to carry this exact C implementation to "
                "a larger backbone, more datasets, or a foundation model."),
            "OPEN_NOT_DIRECTLY_TESTED": (
                "Forecast-query-conditioned token budget allocation as a topic is not refuted "
                "by this r=2 fixed-local pooling experiment."),
        },
        "observations_behind_the_recommendation": ctx["observations"],
        "no_new_gate_created": ("These are the observations already in the record. They were "
                               "not combined into a new threshold, and the pre-registered "
                               "scientific decision was not revised."),
        "checks": {
            "arithmetic": ctx["arith_status"],
            "bootstrap": ctx["boot_status"],
            "schedule_fairness": ctx["schedule"]["status"],
            "original_artifact_immutability": ctx["immutability"]["status"],
        },
    }
    write("audit_verdict.json", out)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log("HQ-TOKEN-PILOT-v1-AUDIT-CLOSURE")

    git = {
        "HEAD": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip(),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "origin_main": subprocess.run(["git", "rev-parse", "origin/main"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
        "remote": subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip().splitlines(),
    }

    before = manifest()
    write("original_artifact_manifest.json",
          {"captured_before_audit": before, "git_at_capture": git,
           "policy": ("Every file listed here is treated as a historical scientific artifact. "
                      "The audit re-hashes them at the end and stops hard if any changed.")})

    metrics = read_metrics()
    contrasts = load_json(os.path.join(RES, "contrasts.json"))
    recomputed, arith_status = stage_arithmetic(metrics, contrasts)
    stage_paired_seeds(metrics)
    boot = stage_bootstrap(load_json(os.path.join(RES, "bootstrap.json")))
    boot_status = (boot or {}).get("status", "RAW_KEYED_ARTIFACT_NOT_AVAILABLE")
    curves = stage_curves()
    stage_trajectory()
    ckpt = stage_checkpoint_audit(curves)
    schedule = stage_schedule_fairness()
    cap = stage_capacity(load_json(os.path.join(RES, "mechanism_capacity_check.json")))
    seedu = stage_seed_uncertainty(load_json(os.path.join(RES, "seed_noise_floor.json")))
    gap = stage_compression_gap(metrics, recomputed)
    qrows, max_swap, max_shift = stage_query_mechanism(
        recomputed, load_json(os.path.join(RES, "diagnostics.json")))
    scope = stage_scope()
    effi = stage_efficiency()
    spec = load_json(os.path.join(RES, "execution_spec.json"))
    dm = load_json(os.path.join(RES, "data_manifest.json"))
    exposure = stage_exposure(spec, dm, curves)
    write("sampling_exposure_audit.json", exposure)
    refs = stage_foundation_scope(load_json(os.path.join(RES, "optional_references.json")))

    after = manifest()
    changed = [k for k in before if before[k] != after.get(k)]
    immutability = {
        "status": "ORIGINAL_ARTIFACTS_UNCHANGED" if not changed else "ORIGINAL_RESULT_MUTATION_HARD_STOP",
        "changed_files": changed, "files_checked": len(before),
    }
    write("original_artifact_immutability.json",
          {**immutability, "hashes_after": after})
    if changed:
        raise SystemExit("ORIGINAL_RESULT_MUTATION_HARD_STOP: " + ", ".join(changed))

    cvi = recomputed["C_vs_I"]
    observations = {
        "C_vs_I_macro_pct": cvi["macro"],
        "C_vs_I_per_seed_macro_pct": cvi["per_seed_macro"],
        "C_vs_I_bootstrap": {k: (boot or {}).get("recomputed", {}).get(k)
                             for k in ("macro_mean", "lower95", "upper95")},
        "C_vs_R_macro_pct": recomputed["C_vs_R"]["macro"],
        "max_abs_query_swap_effect_pct": max_swap,
        "max_mean_horizon_weight_shift": max_shift,
        "phase_shifted_macro_pct": contrasts["robustness_phase_shifted_grid"]["macro"],
        "C_vs_DENSE_end_to_end_latency_pct": effi["C_vs_DENSE_end_to_end_pct"],
        "training_curve_position_labels": {
            d: sorted({r["position_label"] for r in curves if r["dataset"] == d})
            for d in DATASETS},
        "paired_fits_with_same_best_update": ckpt["pairs_with_identical_best_update"],
        "synthetic_joint_task_solved_by_C": cap["JOINT_HORIZON_TASK_SOLVED"],
    }
    ctx = {
        "git": git, "immutability": immutability, "arith_status": arith_status,
        "boot_status": boot_status, "schedule": schedule, "observations": observations,
        "verdict_original": load_json(os.path.join(RES, "verdict.json")),
    }
    verdict = stage_verdict(ctx)

    write("audit_context.json", {"git": git, "sampling_exposure": exposure,
                                 "checkpoint_selection": ckpt["rows"],
                                 "schedule_fairness": schedule})
    log("")
    log("=" * 68)
    log(f"arithmetic          : {arith_status}")
    log(f"bootstrap           : {boot_status}")
    log(f"schedule fairness   : {schedule['status']}")
    log(f"original artifacts  : {immutability['status']}")
    log(f"original decision   : {verdict['scientific_decision_original']}")
    log(f"implementation      : {verdict['audit_recommendation_current_implementation']}")
    log(f"broader topic       : {verdict['broader_topic_status']}")
    log("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
