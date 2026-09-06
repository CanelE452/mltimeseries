"""Analysis-mode figures for OA-RESOLUTION-PILOT-v1.

Reads only the frozen result files in results/oa_resolution_pilot_v1 and writes
four PNGs plus captions.md into results/oa_resolution_pilot_v1/figures.

Style and colour come from the viz-expert assets; nothing is hard-coded here.

The question the pilot asked: does O, which puts the observation support into the
positional representation as an integral, reach a lower error than M, which gets
the same metadata as a centre time, at time resolutions that were never trained?
The answer was no, and these figures are meant to show that plainly.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ASSETS = Path.home() / ".claude/agents/viz-expert/assets"
plt.style.use(ASSETS / "analysis.mplstyle")
sys.path.insert(0, str(ASSETS))
from palette import color_for, SEMANTIC, LINESTYLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "oa_resolution_pilot_v1"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SRC = "results/oa_resolution_pilot_v1/"
STAMP = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

DATASETS = ["jena", "uci"]
OPS = ["END_BIN", "INTERVAL_MEAN"]
ROLES = ["SEEN", "UNSEEN_INTERPOLATION", "UNSEEN_EXTRAPOLATION"]
ROLE_R = {"SEEN": [2, 4, 8], "UNSEEN_INTERPOLATION": [3, 6],
          "UNSEEN_EXTRAPOLATION": [12]}
ROLE_SHORT = {"SEEN": "seen", "UNSEEN_INTERPOLATION": "unseen interpolation",
              "UNSEEN_EXTRAPOLATION": "unseen extrapolation"}
SEEDS = ["2026090601", "2026090602"]
GATE = 1.0

# Arm letters are the strings that really appear in metrics.csv, so they stay as
# the plotted label and the gloss is carried in the legend.
ARM_KEY = {"O": "ours", "M": "baseline", "R": "resample-baseline"}
ARM_COLOR = {a: color_for(k) for a, k in ARM_KEY.items()}
FS_COLOR = color_for("foundation-model")
DS_COLOR = {"jena": color_for("jena"), "uci": color_for("uci-household-power")}
assert len(set(list(ARM_COLOR.values()) + [FS_COLOR])) == 4, "arm colour collision"
assert len(set(DS_COLOR.values())) == 2, "dataset colour collision"

ARM_GLOSS = {
    "R": "R - operator-aware reconstruction, then a base-grid forecaster",
    "M": "M - observed tokens, interval centre time as metadata",
    "O": "O - observed tokens, interval-integrated time basis",
}
OP_STYLE = {"END_BIN": ("-", "o"), "INTERVAL_MEAN": ("--", "s")}
PRIMARY = "primary loss = 0.5*MSE(10 min) + 0.5*MSE(60 min), normalised units"


def load():
    with open(RES / "metrics.csv") as f:
        met = list(csv.DictReader(f))
    for r in met:
        r["r"] = int(r["r"])
        r["primary"] = float(r["primary"])
    with open(RES / "efficiency.csv") as f:
        eff = list(csv.DictReader(f))
    for r in eff:
        r["n_parameters"] = int(r["n_parameters"])
        r["wall_seconds"] = float(r["wall_seconds"])
        r["best_update"] = int(r["best_update"])
        r["best_val_primary"] = float(r["best_val_primary"])
    with open(RES / "training_curves_summary.csv") as f:
        cur = list(csv.DictReader(f))
    for r in cur:
        r["update"] = int(r["update"])
        r["primary"] = float(r["primary"])
    boot = json.loads((RES / "bootstrap.json").read_text())
    con = json.loads((RES / "primary_contrasts.json").read_text())
    ver = json.loads((RES / "verdict.json").read_text())
    fs = json.loads((RES / "flowstate_reference.json").read_text())
    anc = json.loads((RES / "learning_anchors.json").read_text())
    return met, eff, cur, boot, con, ver, fs, anc


def cell(met, ds, op, r, arm):
    for row in met:
        if (row["dataset"] == ds and row["operation"] == op
                and row["r"] == r and row["arm"] == arm):
            return row["primary"]
    raise KeyError((ds, op, r, arm))


def footer(fig):
    fig.text(0.004, 0.004, "src: " + SRC, ha="left", fontsize=7, color="gray")
    fig.text(0.996, 0.004, STAMP, ha="right", fontsize=7, color="gray")


def mono_box(ax, lines, x=0.0, y=1.0, fs=7.6):
    ax.axis("off")
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=fs,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def free_corner(ax, pts, w=0.46, h=0.38):
    """Corner of ax holding the fewest data points, so a box never covers data."""
    inv = ax.transAxes.inverted()
    fr = [inv.transform(ax.transData.transform(p)) for p in pts]
    boxes = {"sw": (0.0, w, 0.0, h), "se": (1 - w, 1.0, 0.0, h),
             "nw": (0.0, w, 1 - h, 1.0), "ne": (1 - w, 1.0, 1 - h, 1.0)}
    spec = {"sw": (0.015, 0.035, "left", "bottom"),
            "se": (0.985, 0.035, "right", "bottom"),
            "nw": (0.015, 0.965, "left", "top"),
            "ne": (0.985, 0.965, "right", "top")}
    best, bestn = "sw", None
    for name, (x0, x1, y0, y1) in boxes.items():
        n = sum(1 for x, y in fr if x0 <= x <= x1 and y0 <= y <= y1)
        if bestn is None or n < bestn:
            best, bestn = name, n
    return spec[best]


def corner_box(ax, lines, pts, fs=7.4):
    x, y, ha, va = free_corner(ax, pts)
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, va=va, ha=ha,
            family="monospace", fontsize=fs, zorder=6,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


# ---------------------------------------------------------------- figure 1 ---
def fig1(con, boot, ver):
    cells = con["unseen_interpolation"]["O_vs_M"]["cells"]
    macro = con["unseen_interpolation"]["O_vs_M"]["macro_relative_improvement_pct"]
    per_seed = con["per_seed_macro_O_vs_M_unseen_interpolation"]
    post = boot["supplementary_post_hoc"]["intervals"]
    scopes = [("all", "all"), ("jena", "jena"), ("uci", "uci")]

    order = [c for ds in DATASETS for op in OPS for c in cells
             if c["dataset"] == ds and c["operation"] == op]
    n = len(order)
    ys = [n - 1 - i for i in range(n)]

    fig = plt.figure(figsize=(16.4, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.18],
                          height_ratios=[n, 4.0])
    ax_c = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_c)
    ax_t = fig.add_subplot(gs[:, 1])

    for ax in (ax_c, ax_b):
        ax.axvline(0.0, color=SEMANTIC["zero"], ls=LINESTYLE["zero"], lw=1.0, zorder=1)
        ax.axvline(GATE, color=SEMANTIC["threshold"], ls=LINESTYLE["threshold"],
                   lw=1.4, zorder=1)

    for c, y in zip(order, ys):
        ax_c.plot(c["relative_improvement_pct"], y, marker="o", ms=7,
                  color=DS_COLOR[c["dataset"]], ls="", zorder=3)
    ax_c.set_yticks(ys)
    ax_c.set_yticklabels(["{:<5s} {:<13s} r={:<2d}".format(
        c["dataset"], c["operation"].lower(), c["r"]) for c in order],
        fontfamily="monospace", fontsize=9)
    ax_c.set_ylim(-0.7, n - 0.3)
    ax_c.set_title("per-cell point estimate\n"
                   "8 unseen-interpolation cells: 2 datasets x 2 operations x r in 3, 6",
                   fontsize=10.5)
    ax_c.axhline(3.5, color="0.85", lw=0.8)
    ax_c.text(GATE - 0.04, 0.2, "pre-registered condition +1.0%",
              color=SEMANTIC["threshold"], fontsize=8.5, ha="right",
              va="bottom", rotation=90)
    plt.setp(ax_c.get_xticklabels(), visible=False)

    yb = [2, 1, 0]
    for (key, lab), y in zip(scopes, yb):
        b = post["O_vs_M_unseen_interpolation_" + key]
        col = "0.35" if key == "all" else DS_COLOR[key]
        ax_b.errorbar(b["mean"], y,
                      xerr=[[b["mean"] - b["lower95"]], [b["upper95"] - b["mean"]]],
                      fmt="D", ms=7, color=col, elinewidth=1.8, capsize=4, zorder=3)
        ax_b.text(b["upper95"] + 0.05, y, "[{:+.3f}, {:+.3f}]  contains 0".format(
            b["lower95"], b["upper95"]), va="center", fontsize=8.2, color="0.3",
            family="monospace")
    for v in per_seed.values():
        ax_b.plot(v, 2, marker="|", ms=13, color="0.45", mew=1.5, zorder=4)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(["all (8 cells)", "jena only", "uci only"],
                         fontfamily="monospace", fontsize=9)
    ax_b.set_ylim(-0.6, 2.6)
    ax_b.set_xlim(-1.15, 1.95)
    ax_b.set_xlabel("relative reduction of primary loss, O vs M (%)  =  "
                    "100 * (M - O) / M          "
                    "negative = M lower error   |   positive = O lower error")
    ax_b.set_title("paired moving-block bootstrap, 95% interval "
                   "(1000 draws, 7-day blocks)\n"
                   "vertical ticks = macro of each model seed", fontsize=10.5)

    lines = [
        "O vs M at the resolutions never seen in training (r = 3, 6)",
        "relative reduction of primary loss = 100 * (M - O) / M",
        "positive = O lower error,  negative = M lower error",
        "",
        "{:<5s} {:<13s} {:>3s} {:>9s} {:>9s} {:>8s}".format(
            "data", "operation", "r", "M", "O", "value %"),
        "-" * 52,
    ]
    for c in order:
        lines.append("{:<5s} {:<13s} {:>3d} {:>9.5f} {:>9.5f} {:>+8.2f}".format(
            c["dataset"], c["operation"].lower(), c["r"], c["M"], c["O"],
            c["relative_improvement_pct"]))
    lines += [
        "-" * 52,
        "{:<33s} {:>+8.3f}".format("macro over the 8 cells", macro),
        "",
        "bootstrap: 1000 draws, 7-day blocks, 14 origins per block,",
        "at least {} effective blocks per dataset".format(
            post["O_vs_M_unseen_interpolation_all"][
                "effective_blocks_min_over_datasets"]),
        "{:<8s} {:>8s}   {:<22s}".format("scope", "mean %", "95% interval"),
        "-" * 52,
    ]
    for key, lab in scopes:
        b = post["O_vs_M_unseen_interpolation_" + key]
        lines.append("{:<8s} {:>+8.3f}   [{:>+7.3f}, {:>+7.3f}]".format(
            lab, b["mean"], b["lower95"], b["upper95"]))
    lines += [
        "",
        "per-seed macro   {:>+8.3f}  (seed {})".format(per_seed[SEEDS[0]], SEEDS[0]),
        "                 {:>+8.3f}  (seed {})".format(per_seed[SEEDS[1]], SEEDS[1]),
        "the two seeds disagree in sign, and their spread ({:.2f} pts)".format(
            abs(per_seed[SEEDS[0]] - per_seed[SEEDS[1]])),
        "is {:.0f}x the macro estimate ({:.3f} pts).".format(
            abs(per_seed[SEEDS[0]] - per_seed[SEEDS[1]]) / abs(macro), abs(macro)),
        "",
        "pre-registered conditions on this contrast",
        "-" * 52,
        "{:<40s} {:>11s}".format("macro >= +1.0%", "NOT MET"),
        "{:<40s} {:>11s}".format("bootstrap lower95 > 0", "NOT MET"),
        "{:<40s} {:>11s}".format("both datasets positive", "NOT MET"),
        "{:<40s} {:>11s}".format("both seeds positive", "NOT MET"),
        "{:<40s} {:>11s}".format("recorded decision", ver["scientific_decision"]),
        "",
        "the same contrast at r = 12 is a DIFFERENT question",
        "(unseen extrapolation, never mixed into the axis at left)",
        "-" * 52,
    ]
    for key, lab in scopes:
        b = post["O_vs_M_unseen_extrapolation_" + key]
        lines.append("{:<8s} {:>+8.3f}   [{:>+7.3f}, {:>+7.3f}]".format(
            lab, b["mean"], b["lower95"], b["upper95"]))
    lines += [
        "that r = 12 gap rests on one cell, jena interval_mean",
        "(+7.44%); see fig2.",
    ]
    mono_box(ax_t, lines)

    handles = [Line2D([], [], marker="o", ls="", color=DS_COLOR[d], ms=7,
                      label=d + " cell") for d in DATASETS]
    handles += [
        Line2D([], [], marker="D", ls="", color="0.35", ms=7,
               label="bootstrap mean, 95% interval"),
        Line2D([], [], marker="|", ls="", color="0.45", ms=11, mew=1.5,
               label="per-seed macro (2 seeds)"),
        Line2D([], [], color=SEMANTIC["zero"], ls=LINESTYLE["zero"], lw=1.0,
               label="no difference"),
        Line2D([], [], color=SEMANTIC["threshold"], ls=LINESTYLE["threshold"],
               lw=1.4, label="pre-registered condition +1.0%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.018), frameon=True)
    fig.text(0.5, 0.988,
             "OA-RESOLUTION-PILOT-v1  -  O vs M at unseen interpolation resolutions   "
             "macro {:+.3f}%, bootstrap [{:+.3f}, {:+.3f}], the interval contains 0".format(
                 macro, post["O_vs_M_unseen_interpolation_all"]["lower95"],
                 post["O_vs_M_unseen_interpolation_all"]["upper95"]),
             ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.960, PRIMARY + "   |   the two model seeds are aggregated "
             "inside each cell value",
             ha="center", va="top", fontsize=9, color="0.3")
    fig.get_layout_engine().set(rect=(0, 0.062, 1, 0.888))
    footer(fig)
    p = OUT / "fig1_o_vs_m_unseen_interpolation.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2 ---
def fig2(met, anc):
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.4), sharey="row",
                             gridspec_kw={"width_ratios": [3.0, 2.0, 1.25]})
    for i, ds in enumerate(DATASETS):
        for j, role in enumerate(ROLES):
            ax = axes[i, j]
            rs = ROLE_R[role]
            for arm in ["R", "M", "O"]:
                for op in OPS:
                    ls, mk = OP_STYLE[op]
                    ys = [cell(met, ds, op, r, arm) for r in rs]
                    ax.plot(rs, ys, ls=ls, marker=mk, ms=6, lw=1.4,
                            color=ARM_COLOR[arm])
            pad = 0.6 if len(rs) > 1 else 1.0
            ax.set_xlim(min(rs) - pad, max(rs) + pad)
            ax.set_xticks(rs)
            ax.set_xticklabels([str(r) for r in rs])
            if i == 1:
                ax.set_xlabel("resolution factor r")
            if j == 0:
                ax.set_ylabel(ds + "\nprimary loss")
            title = "{} - {}, r = {}".format(
                ds, ROLE_SHORT[role], ", ".join(str(r) for r in rs))
            title += " (trained)" if role == "SEEN" else " (never trained)"
            ax.set_title(title, fontsize=10)

    handles = [Line2D([], [], color=ARM_COLOR[a], lw=2.0, label=ARM_GLOSS[a])
               for a in ["R", "M", "O"]]
    handles += [Line2D([], [], color="0.35", ls=OP_STYLE[op][0],
                       marker=OP_STYLE[op][1], ms=6, label="operation " + op.lower())
                for op in OPS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.016), frameon=True)
    fig.text(0.5, 0.988,
             "OA-RESOLUTION-PILOT-v1  -  primary loss by arm, split by resolution role   "
             "the three roles are never put on one axis",
             ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.962,
             PRIMARY + "   |   bin width = 10 min x r   |   the y scale differs per "
             "dataset and is not shared between rows   |   R is lower on jena, "
             "M and O are lower on uci\n"
             "sanity floor, jena / uci:  persistence {:.3f} / {:.3f},  "
             "seasonal naive at one day {:.3f} / {:.3f}  -  every arm in every panel "
             "is far below both".format(
                 anc["jena"]["persistence"]["primary"],
                 anc["uci"]["persistence"]["primary"],
                 anc["jena"]["seasonal_naive_1d"]["primary"],
                 anc["uci"]["seasonal_naive_1d"]["primary"]),
             ha="center", va="top", fontsize=9, color="0.3")
    fig.get_layout_engine().set(rect=(0, 0.088, 1, 0.848))
    footer(fig)
    p = OUT / "fig2_arms_by_resolution_role.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 3 ---
def fig3(cur, eff):
    zoom_from = 2000
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.6))
    seed_ls = {SEEDS[0]: "-", SEEDS[1]: "--"}
    for j, ds in enumerate(DATASETS):
        for row, lo in enumerate([0, zoom_from]):
            ax = axes[row, j]
            for arm in ["R", "M", "O"]:
                for sd in SEEDS:
                    pts = sorted((r["update"], r["primary"]) for r in cur
                                 if r["dataset"] == ds and r["arm"] == arm
                                 and r["seed"] == sd)
                    xs = [u for u, _ in pts if u >= lo]
                    yy = [v for u, v in pts if u >= lo]
                    ax.plot(xs, yy, ls=seed_ls[sd], lw=1.3, color=ARM_COLOR[arm],
                            marker=".", ms=4)
                    e = next(r for r in eff if r["dataset"] == ds
                             and r["arm"] == arm and r["model_seed"] == sd)
                    if e["best_update"] >= lo:
                        ax.plot(e["best_update"], e["best_val_primary"], marker="D",
                                ms=8, mfc="white", mec=ARM_COLOR[arm], mew=1.6,
                                ls="", zorder=5)
            ax.set_xticks(range(500, 5001, 500))
            ax.set_xlim(lo - 150 if lo else 300, 5250)
            if row == 1:
                ax.set_xlabel("update")
            ax.set_ylabel("validation primary loss")
            if row == 0:
                ax.set_title("{} - all updates".format(ds), fontsize=10)
            else:
                vals = [r["primary"] for r in cur if r["dataset"] == ds
                        and r["update"] >= zoom_from]
                span = max(vals) - min(vals)
                ax.set_ylim(min(vals) - 0.30 * span, max(vals) + 0.06 * span)
                ax.set_title("{} - from update {} on, y rescaled".format(
                    ds, zoom_from), fontsize=10)
    for j, ds in enumerate(DATASETS):
        lines = ["{:<3s} {:<6s} {:>7s} {:>10s}".format(
            "arm", "seed", "best", "val loss"), "-" * 29]
        for arm in ["R", "M", "O"]:
            for sd in SEEDS:
                e = next(r for r in eff if r["dataset"] == ds and r["arm"] == arm
                         and r["model_seed"] == sd)
                lines.append("{:<3s} {:<6s} {:>7d} {:>10.5f}".format(
                    arm, ".." + sd[-2:], e["best_update"], e["best_val_primary"]))
        pts = [(r["update"], r["primary"]) for r in cur
               if r["dataset"] == ds and r["update"] >= zoom_from]
        corner_box(axes[1, j], lines, pts)

    handles = [Line2D([], [], color=ARM_COLOR[a], lw=2.0, label=ARM_GLOSS[a])
               for a in ["R", "M", "O"]]
    handles += [Line2D([], [], color="0.35", ls=seed_ls[s], label="model seed " + s)
                for s in SEEDS]
    handles += [Line2D([], [], marker="D", ls="", mfc="white", mec="0.35", mew=1.6,
                       ms=8, label="selected checkpoint (lowest validation loss)")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.016), frameon=True)
    fig.text(0.5, 0.988,
             "OA-RESOLUTION-PILOT-v1  -  validation primary loss during training",
             ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.960,
             "validation uses the trained resolutions r = 2, 4, 8 only, so the unseen "
             "resolutions never enter checkpoint selection   |   " + PRIMARY,
             ha="center", va="top", fontsize=9, color="0.3")
    fig.get_layout_engine().set(rect=(0, 0.092, 1, 0.858))
    footer(fig)
    p = OUT / "fig3_training_curves.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 4 ---
def interp_stats(met, ds, arm):
    vals = [cell(met, ds, op, r, arm)
            for op in OPS for r in ROLE_R["UNSEEN_INTERPOLATION"]]
    return float(np.mean(vals)), min(vals), max(vals)


def fs_interp_stats(fs, ds):
    vals = [c["primary"] for c in fs["datasets"][ds]["resampled"]["cells"]
            if c["role"] == "UNSEEN_INTERPOLATION"]
    return float(np.mean(vals)), min(vals), max(vals)


def fig4(met, eff, fs):
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.8), sharey="row")
    fs_wall = fs["wall_seconds"]
    fs_par = fs["n_parameters"]
    # M and O land on the same point, which is the result itself, so the markers
    # are never moved apart - only the labels are dodged.
    DODGE = {"R": (0, 15, "center", "bottom"),
             "M": (-15, 9, "right", "bottom"),
             "O": (15, -13, "left", "top")}
    for i, ds in enumerate(DATASETS):
        # fix the row scale before anything is placed, so that the dodged O label
        # has room under the lowest marker and the table lands on a free corner
        ends = [v for arm in ["R", "M", "O"] for v in interp_stats(met, ds, arm)[1:]]
        ends += list(fs_interp_stats(fs, ds)[1:])
        span = max(ends) - min(ends)
        axes[i, 0].set_ylim(min(ends) - 0.15 * span, max(ends) + 0.07 * span)
        for j, xkey in enumerate(["wall_seconds", "n_parameters"]):
            ax = axes[i, j]
            pts = []
            for arm in ["R", "M", "O"]:
                rows = [r for r in eff if r["dataset"] == ds and r["arm"] == arm]
                xs = [r[xkey] for r in rows]
                x = float(np.mean(xs))
                m, lo, hi = interp_stats(met, ds, arm)
                xerr = ([[x - min(xs)], [max(xs) - x]]
                        if xkey == "wall_seconds" else None)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], xerr=xerr,
                            fmt="o", ms=9, color=ARM_COLOR[arm], elinewidth=1.4,
                            capsize=4, zorder=3)
                pts += [(x, lo), (x, hi)]
                if xkey == "wall_seconds":
                    dx, dy, ha, va = DODGE[arm]
                    ax.annotate("{} {:.4f}".format(arm, m), xy=(x, m),
                                xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=8.5, color=ARM_COLOR[arm],
                                family="monospace", zorder=6)
            m, lo, hi = fs_interp_stats(fs, ds)
            x = fs_wall if xkey == "wall_seconds" else fs_par
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="D", ms=10,
                        color=FS_COLOR, mfc="white", mew=1.8, elinewidth=1.4,
                        capsize=4, zorder=3)
            ax.annotate("FlowState {:.4f}\nzero-shot".format(m), xy=(x, m),
                        xytext=(-14, 4), textcoords="offset points", ha="right",
                        va="bottom", fontsize=8.5, color=FS_COLOR,
                        family="monospace", zorder=6)
            pts += [(x, lo), (x, hi)]
            if xkey == "wall_seconds":
                ax.set_xlim(0, 108)
                if i == 1:
                    ax.set_xlabel("wall seconds spent on this task")
            else:
                ax.set_xscale("log")
                ax.set_xlim(2.0e5, 6.0e7)
                if i == 1:
                    ax.set_xlabel("trainable parameters (log scale)")
                ax.annotate(
                    "R, M and O sit on one another here:\n"
                    "{:,} parameters (R) and {:,} (M and O)".format(
                        [r for r in eff if r["arm"] == "R"][0]["n_parameters"],
                        [r for r in eff if r["arm"] == "M"][0]["n_parameters"]),
                    xy=(4.1e5, 0.5), xycoords=("data", "axes fraction"),
                    xytext=(24, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=8.2, color="0.35", family="monospace",
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
            if j == 0:
                ax.set_ylabel(ds + "\nprimary loss, unseen interpolation")
            ax.set_title(ds, fontsize=10)
            if j == 0:
                lines = ["{:<10s} {:>7s} {:>8s} {:>9s}".format(
                    "arm", "wall s", "params", "loss"), "-" * 37]
                for arm in ["R", "M", "O"]:
                    rows = [r for r in eff if r["dataset"] == ds and r["arm"] == arm]
                    mm, _, _ = interp_stats(met, ds, arm)
                    lines.append("{:<10s} {:>7.1f} {:>7.2f}M {:>9.4f}".format(
                        arm, float(np.mean([r["wall_seconds"] for r in rows])),
                        rows[0]["n_parameters"] / 1e6, mm))
                mm, _, _ = fs_interp_stats(fs, ds)
                lines.append("{:<10s} {:>7.1f} {:>7.2f}M {:>9.4f}".format(
                    "FlowState", fs_wall, fs_par / 1e6, mm))
                lines.append("FlowState wall time is evaluation only")
                corner_box(ax, lines, pts, fs=7.3)

    handles = [Line2D([], [], marker="o", ls="", color=ARM_COLOR[a], ms=9,
                      label=ARM_GLOSS[a] + ", trained on this dataset")
               for a in ["R", "M", "O"]]
    handles += [Line2D([], [], marker="D", ls="", color=FS_COLOR, mfc="white",
                       mew=1.8, ms=10,
                       label="FlowState r1.1, zero-shot pretrained, never trained on "
                             "these series - different information condition")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.014), frameon=True)
    fig.text(0.5, 0.990,
             "OA-RESOLUTION-PILOT-v1  -  unseen-interpolation loss against cost",
             ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.962,
             "y = mean over the 4 unseen-interpolation cells (r = 3, 6 x 2 operations), "
             "bar = min to max over those cells   |   R / M / O: training wall time, "
             "5000 updates, per seed\n"
             "FlowState is not a matched training budget: it saw no training data from "
             "these series, its wall time is evaluation only and its pre-training compute "
             "is not counted on this axis.\nIt is read on its resampled variant so that "
             "the same primary loss is defined; its native-rate variant reports only the "
             "60-minute term and only at r = 2, 3, 6.",
             ha="center", va="top", fontsize=9, color="0.3")
    fig.get_layout_engine().set(rect=(0, 0.086, 1, 0.820))
    footer(fig)
    p = OUT / "fig4_accuracy_vs_cost.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def captions(paths, con, boot, ver, anc):
    post = boot["supplementary_post_hoc"]["intervals"]
    b = post["O_vs_M_unseen_interpolation_all"]
    macro = con["unseen_interpolation"]["O_vs_M"]["macro_relative_improvement_pct"]
    per_ds = con["unseen_interpolation"]["per_dataset_O_vs_M"]
    per_seed = con["per_seed_macro_O_vs_M_unseen_interpolation"]
    txt = [
        "# OA-RESOLUTION-PILOT-v1 - figure captions",
        "",
        "Generated {} by `experiments/oa_resolution_pilot_v1/figures.py`.".format(STAMP),
        "These are analysis-mode figures: they are meant to be read together with the",
        "numbers in STATUS.md, not as standalone paper figures.",
        "",
        "## fig1_o_vs_m_unseen_interpolation.png",
        "",
        "Relative reduction of the primary loss of the interval-integrated arm O over the",
        "centre-time metadata arm M at the two resolutions that were never trained",
        "(r = 3, 6), for each of the eight dataset x operation x r cells, next to the",
        "paired moving-block bootstrap 95% intervals over all cells and per dataset; the",
        "macro is {:+.3f}% with an interval of [{:+.3f}, {:+.3f}] that contains zero, and".format(
            macro, b["lower95"], b["upper95"]),
        "the per-dataset intervals contain zero as well, so the pre-registered +1.0%",
        "condition is not approached from either side.",
        "Limitation: the intervals resample time origins only and not model seeds, and the",
        "two seeds put the macro at {:+.3f}% and {:+.3f}%, a spread wider than the estimate.".format(
            per_seed[SEEDS[0]], per_seed[SEEDS[1]]),
        "",
        "## fig2_arms_by_resolution_role.png",
        "",
        "Primary loss of the three arms against the resolution factor r, with one panel per",
        "resolution role so that trained resolutions, unseen interpolation and unseen",
        "extrapolation are never read off a single axis, and one row per dataset because",
        "the loss scales differ by a factor of three; the ordering of the arms flips",
        "between the datasets, R being lowest on jena and M and O being lowest on uci",
        "(O vs M at unseen interpolation is {:+.3f}% on jena and {:+.3f}% on uci).".format(
            per_ds["jena"], per_ds["uci"]),
        "Limitation: each point is a single number aggregated over two model seeds and",
        "carries no interval here; the intervals are in fig1 and in bootstrap.json.",
        "",
        "## fig3_training_curves.png",
        "",
        "Validation primary loss against update for the three arms and both seeds, with the",
        "selected checkpoint marked, shown at full range and again from update 2000 on",
        "where the arms separate; validation uses only the trained resolutions r = 2, 4, 8,",
        "so nothing about the unseen resolutions enters checkpoint selection, and the same",
        "dataset-dependent ordering as in fig2 is already visible during training.",
        "Limitation: validation loss is not the quantity the pilot asks about, and the gap",
        "between the two seeds is of the same size as the gap between M and O.",
        "",
        "## fig4_accuracy_vs_cost.png",
        "",
        "Unseen-interpolation primary loss against wall time and against parameter count for",
        "the three trained arms and for FlowState r1.1 read zero-shot, the bars giving the",
        "spread over the four unseen-interpolation cells; R costs about 2.5 times the wall",
        "time of M and O for its reconstruction step, while M and O have the same parameter",
        "count, the same wall time to within the seed-to-seed spread, and land on the same",
        "loss.",
        "Limitation: FlowState is not a matched-budget competitor. It never saw these",
        "series, its pre-training compute is not on the wall-time axis, and it is read on",
        "the resampled variant because its native-rate variant defines only the 60-minute",
        "term and only at r = 2, 3, 6.",
        "",
        "## sanity floors",
        "",
        "Every arm in every figure sits far below the naive anchors: persistence is",
        "{:.3f} (jena) and {:.3f} (uci), seasonal naive at one day is {:.3f} and {:.3f}.".format(
            anc["jena"]["persistence"]["primary"], anc["uci"]["persistence"]["primary"],
            anc["jena"]["seasonal_naive_1d"]["primary"],
            anc["uci"]["seasonal_naive_1d"]["primary"]),
        "",
        "Recorded decision for the pilot: {}.".format(ver["scientific_decision"]),
        "",
        "Files:",
    ]
    txt += ["- `{}`".format(p.relative_to(ROOT).as_posix()) for p in paths]
    (OUT / "captions.md").write_text("\n".join(txt) + "\n", encoding="utf-8")
    return OUT / "captions.md"


def main():
    met, eff, cur, boot, con, ver, fs, anc = load()
    paths = [fig1(con, boot, ver), fig2(met, anc), fig3(cur, eff), fig4(met, eff, fs)]
    paths.append(captions(paths, con, boot, ver, anc))
    for p in paths:
        print("wrote", p.relative_to(ROOT).as_posix(),
              "({:.0f} kB)".format(p.stat().st_size / 1024))


if __name__ == "__main__":
    main()
