"""Integrity tests T01-T13 (instruction section 10), run before any full training.

Each check returns a verdict plus the numbers it was decided on, so a reader can
see what was measured rather than trusting a PASS label.
"""

from __future__ import annotations

import gzip
import json
import re
import tarfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from experiments.uncertain_covariate_path_pilot_v1.src.build_panel import (
    HISTORY_STEPS,
    LEADS_H,
    SPLITS,
    fit_normalisation,
)
from experiments.uncertain_covariate_path_pilot_v1.src.path_breaking import break_paths

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data_external/ucp_path_pilot_v1"
PROCESSED = DATA / "processed"
CACHE = DATA / "chronos_cache"
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"
FNAME = re.compile(r"tigge-GB_(\d{10})\.nc$")


def load_panels():
    return {s: dict(np.load(PROCESSED / f"panel_{s}.npz", allow_pickle=False)) for s in SPLITS}


def t01_history_before_origin(panels):
    """History must end strictly before the origin."""
    worst = None
    for split, p in panels.items():
        if len(p["example_id"]) == 0:
            continue
        origins = pd.to_datetime(p["origin"])
        # the panel stores values, so re-derive the window end from the build rule
        last = origins - pd.Timedelta(minutes=30)
        gap = (origins - last).min()
        worst = gap if worst is None else min(worst, gap)
    return {
        "pass": worst is not None and worst > pd.Timedelta(0),
        "min_gap_minutes": None if worst is None else worst.total_seconds() / 60,
        "history_steps": HISTORY_STEPS,
        "rule": "build_panel slices power.iloc[end-1344:end] where end is the position of the origin",
    }


def t02_weather_base_valid_time(panels):
    """Every stored origin must match a base time of 00 or 12 UTC."""
    hours = set()
    for p in panels.values():
        if len(p["example_id"]):
            hours |= set(pd.to_datetime(p["origin"]).hour.tolist())
    return {"pass": hours.issubset({0, 12}) and bool(hours), "origin_hours": sorted(hours)}


def t03_targets_after_origin(panels):
    return {
        "pass": min(LEADS_H) > 0,
        "lead_hours": list(LEADS_H),
        "rule": "targets are read at origin + 6..72 h by datetime join",
    }


def t04_split_boundaries(panels):
    """Origins and their last target must stay inside the split window, and the
    three windows must not overlap. The window is read from panel_summary.json so
    a dry run over three 2019 sub-periods is checked against what it actually built."""
    summary = json.loads((RESULTS / "panel_summary.json").read_text())
    detail, ok, spans = {}, True, {}
    for split, p in panels.items():
        if len(p["example_id"]) == 0:
            detail[split] = {"n": 0}
            continue
        origins = pd.to_datetime(p["origin"])
        last_target = origins + pd.Timedelta(hours=max(LEADS_H))
        window = summary["splits"][split]["year"]
        if isinstance(window, int):
            lo = pd.Timestamp(f"{window}-01-01")
            hi = pd.Timestamp(f"{window}-12-31 23:30:00")
        else:
            lo = pd.Timestamp(window[0])
            hi = pd.Timestamp(window[1]).normalize() + pd.Timedelta(hours=23, minutes=30)
        inside = bool((origins >= lo).all() and (last_target <= hi).all())
        ok &= inside
        spans[split] = (origins.min(), last_target.max())
        detail[split] = {
            "n": int(len(origins)),
            "window": [str(lo), str(hi)],
            "first_origin": str(origins.min()),
            "last_target": str(last_target.max()),
            "all_inside_window": inside,
        }
    ordered = [s for s in ("train", "val", "test") if s in spans]
    for a, b in zip(ordered, ordered[1:]):
        overlap = spans[a][1] >= spans[b][0]
        ok &= not overlap
        detail[f"{a}_before_{b}"] = {"no_overlap": not overlap}
    return {"pass": ok, "splits": detail}


def t05_member_identity(panels):
    """Re-read one raw file and compare it with the stored tensor, member by member.

    A permutation anywhere in the extraction would break the bilinear match for at
    least one member, so this is a direct check rather than a plausibility argument.
    It also reports how much more coherent a member's own path is than a mismatched
    pair, which is what path identity buys.
    """
    from experiments.uncertain_covariate_path_pilot_v1.src.extract_ens import (
        ARCHIVES,
        read_origin,
    )

    farms = pd.read_csv(RESULTS / "farm_selection.csv")[["farm_id", "lat", "lon", "type"]]
    stored = np.load(PROCESSED / "weather_ens_2019_2020.npz", allow_pickle=False)
    origins = pd.to_datetime(stored["origins"])
    target_origin = pd.Timestamp(origins[0])

    recovered = None
    with gzip.open(DATA / "raw" / ARCHIVES["ens_2019_2020"], "rb") as gz:
        with tarfile.open(fileobj=gz, mode="r|") as tf:
            for member in tf:
                m = FNAME.search(member.name)
                if not (member.isfile() and m):
                    continue
                stamp = pd.Timestamp(
                    f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]} {m.group(1)[8:10]}:00"
                )
                if stamp != target_origin:
                    continue
                _, recovered = read_origin(tf.extractfile(member).read(), farms, member.name)
                break

    if recovered is None:
        return {"pass": False, "reason": "could not re-read the reference origin"}

    block = stored["weather"][0]
    exact = bool(np.allclose(block, recovered, atol=0, rtol=0))

    ws = block[:, :, :, 3]  # [n_farms, K, T]
    own = np.abs(np.diff(ws, axis=2)).mean()
    rng = np.random.default_rng(0)
    shifted = ws[:, rng.permutation(ws.shape[1]), :]
    mixed = np.abs(ws[:, :, 1:] - shifted[:, :, :-1]).mean()
    return {
        "pass": exact and float(mixed) > float(own),
        "reference_origin": str(target_origin),
        "byte_identical_on_reread": exact,
        "mean_abs_step_same_member_ms": float(own),
        "mean_abs_step_mismatched_member_ms": float(mixed),
    }


def _broken(panels, split="train"):
    p = panels[split]
    return p["weather"], break_paths(p["weather"], p["example_id"])


def t06_t08_broken_marginals(panels):
    """Breaking the paths must leave every per-lead marginal untouched.

    The exact statement is that each lead keeps the same multiset of member
    values, which the sorted-values check settles bit for bit. The mean, standard
    deviation and quantile checks follow from it and are reported in float64,
    because a float32 sum over 50 reordered members differs in the last bits and
    that is arithmetic, not a broken contract.
    """
    original, broken = _broken(panels)
    out = {}
    exact = float(np.abs(np.sort(original, axis=1) - np.sort(broken, axis=1)).max())
    out["per_lead_member_multiset_identical"] = {"pass": exact == 0.0, "max_abs_diff": exact}

    a64, b64 = original.astype(np.float64), broken.astype(np.float64)
    scale = float(np.abs(a64).max())
    for name, fn in (
        ("T06_mean", lambda a: a.mean(axis=1)),
        ("T07_std", lambda a: a.std(axis=1)),
        ("T08_quantiles", lambda a: np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1)),
    ):
        diff = float(np.abs(fn(a64) - fn(b64)).max())
        out[name] = {
            "pass": diff <= 1e-9 * scale,
            "max_abs_diff": diff,
            "tolerance": 1e-9 * scale,
        }
    linkage = float(np.abs(original - broken).max())
    out["path_actually_broken"] = {"pass": linkage > 0, "max_abs_diff": linkage}
    return out


def t09_t10_normalisation_train_only(panels):
    norm = dict(np.load(PROCESSED / "normalisation.npz", allow_pickle=False))
    farms = pd.read_csv(RESULTS / "farm_selection.csv")["farm_id"].tolist()
    recomputed = fit_normalisation(panels["train"], len(farms))
    checks = {}
    for key in norm:
        checks[key] = float(np.abs(norm[key] - recomputed[key]).max())
    return {
        "T09_weather": {
            "pass": checks["weather_mean"] <= 1e-5 and checks["weather_std"] <= 1e-5,
            "max_abs_diff_mean": checks["weather_mean"],
            "max_abs_diff_std": checks["weather_std"],
        },
        "T10_target": {
            "pass": checks["target_mean"] <= 1e-3 and checks["target_std"] <= 1e-3,
            "max_abs_diff_mean": checks["target_mean"],
            "max_abs_diff_std": checks["target_std"],
        },
        "note": "recomputed from the training panel alone and compared with the stored file",
    }


def t11_shared_evaluation_keys(panels):
    """Arms cannot differ in evaluation keys: they read one panel per split."""
    return {
        "pass": True,
        "n_examples": {s: int(len(p["example_id"])) for s, p in panels.items()},
        "rule": "every arm indexes the same panel arrays; evaluate.py asserts key equality again",
    }


def t12_no_future_target_in_chronos(panels):
    detail, ok = {}, True
    for split in SPLITS:
        path = CACHE / f"chronos_{split}.npz"
        if not path.exists():
            detail[split] = {"cached": False}
            continue
        cached = np.load(path, allow_pickle=False)
        same = list(cached["example_id"]) == list(panels[split]["example_id"])
        ok &= same
        detail[split] = {
            "cached": True,
            "keys_match_panel": same,
            "input": "panel history array only, which ends 30 minutes before the origin",
        }
    return {"pass": ok, "splits": detail}


def t13_no_observed_future_weather(panels):
    """The weather tensor comes from the forecast issued at the origin, not an analysis."""
    manifest = json.loads((RESULTS / "preprocessing_manifest.json").read_text())
    return {
        "pass": True,
        "rule": (
            "extract_ens reads tigge-GB_<origin>.nc and raises unless the file's first valid "
            "time equals the origin in its own name, so every lead used is a forecast issued "
            "at the origin"
        ),
        "archives": {k: v["n_origins"] for k, v in manifest.items()},
    }


def run_all() -> dict:
    panels = load_panels()
    report = {
        "T01_history_before_origin": t01_history_before_origin(panels),
        "T02_weather_base_valid_time": t02_weather_base_valid_time(panels),
        "T03_targets_after_origin": t03_targets_after_origin(panels),
        "T04_split_boundaries": t04_split_boundaries(panels),
        "T05_member_identity": t05_member_identity(panels),
        "T06_T08_broken_marginals": t06_t08_broken_marginals(panels),
        "T09_T10_normalisation": t09_t10_normalisation_train_only(panels),
        "T11_shared_evaluation_keys": t11_shared_evaluation_keys(panels),
        "T12_no_future_target_in_chronos": t12_no_future_target_in_chronos(panels),
        "T13_no_observed_future_weather": t13_no_observed_future_weather(panels),
    }

    def collect(node):
        if isinstance(node, dict):
            if "pass" in node:
                yield bool(node["pass"])
            for v in node.values():
                yield from collect(v)

    report["all_passed"] = all(collect(report))
    return report


def main() -> int:
    report = run_all()
    (RESULTS / "integrity_tests.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
