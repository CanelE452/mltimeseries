"""Analysis-mode figures for HQ-TOKEN-PILOT-v1.

Reads only the frozen result files in results/hq_token_pilot_v1 and writes three
PNGs plus captions.md into results/hq_token_pilot_v1/figures.

Style and colour come from the viz-expert assets; nothing is hard-coded here.
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
from palette import color_for, colors_for, SEMANTIC, LINESTYLE, ALPHA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "hq_token_pilot_v1"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["ETTm2", "weather", "electricity"]
HORIZONS = [96, 336]
SEED0, SEED1 = "2026090601", "2026090602"
GATE = 1.0
FLOOR = -1.0

DS_COLOR = dict(zip(DATASETS, colors_for(DATASETS)))

# Arm codes are the strings that really appear in metrics.csv / efficiency.csv,
# so they stay as the plotted label and the gloss is carried in the legend.
ARMS = ["U", "H_STATIC", "I", "C", "R", "DENSE"]
ARM_KEY = {
    "C": "ours",                       # treatment: horizon-conditioned query
    "I": "baseline",                   # primary reference: input-only query
    "U": "uniform-pooling",
    "H_STATIC": "horizon-static-weights",
    "R": "fake-horizon-control",
    "DENSE": "no-compression",
}
_ORDER = ["C", "I", "U", "H_STATIC", "R", "DENSE"]
ARM_COLOR = dict(zip(_ORDER, colors_for([ARM_KEY[a] for a in _ORDER])))
ARM_GLOSS = {
    "U": "U - uniform pooling, no scorer (32 tokens)",
    "H_STATIC": "H_STATIC - learned per-horizon weights, no content (32 tokens)",
    "I": "I - content scorer, input-only query (32 tokens)",
    "C": "C - content scorer, horizon-conditioned query (32 tokens)",
    "R": "R - same as C fed a fake horizon (32 tokens)",
    "DENSE": "DENSE - no compression (64 tokens)",
}
ARM_CFG = {"U": "U_B32", "I": "I_B32", "C": "C_B32", "H_STATIC": "H_STATIC_B32",
           "R": "R_B32", "DENSE": "DENSE_B64"}
STAMP = dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def load():
    boot = json.loads((RES / "bootstrap.json").read_text())
    con = json.loads((RES / "contrasts.json").read_text())
    diag = json.loads((RES / "diagnostics.json").read_text())
    cap = json.loads((RES / "mechanism_capacity_check.json").read_text())
    with open(RES / "metrics.csv") as f:
        met = list(csv.DictReader(f))
    with open(RES / "efficiency.csv") as f:
        eff = {r["config"]: r for r in csv.DictReader(f)}
    return boot, con, diag, cap, met, eff


SRC = "results/hq_token_pilot_v1/"


def footer(fig):
    fig.text(0.004, 0.005, "src: " + SRC, ha="left", fontsize=7, color="gray")
    fig.text(0.996, 0.005, STAMP, ha="right", fontsize=7, color="gray")


# ---------------------------------------------------------------- figure 1 ---
def fig1(boot, con):
    cells = boot["per_cell_C_vs_I"]
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            c = cells[ds + "|" + str(h)]
            rows.append(dict(ds=ds, h=h, ri=c["point_ri_pct"], lo=c["lower95"],
                             hi=c["upper95"], n=c["n_origins"], nb=c["n_blocks"]))
    n = len(rows)
    ys = [n - 1 - i for i in range(n)]
    y_macro, y_shift = -1.6, -2.7

    macro = con["primary"]["macro"]
    m_lo, m_hi = boot["lower95"], boot["upper95"]
    seed_macros = con["primary"]["per_seed_macro"]
    shift_macro = con["robustness_phase_shifted_grid"]["macro"]

    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.25, 1.5])
    ax_f = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[0, 1], sharey=ax_f)
    ax_t = fig.add_subplot(gs[0, 2])

    for ax in (ax_f, ax_z):
        ax.axvline(0.0, color=SEMANTIC["zero"], ls=LINESTYLE["zero"], lw=0.9, zorder=1)
        for r, y in zip(rows, ys):
            low_blocks = r["nb"] < 8
            ax.errorbar(r["ri"], y, xerr=[[r["ri"] - r["lo"]], [r["hi"] - r["ri"]]],
                        fmt="o", ms=6.5, color=DS_COLOR[r["ds"]],
                        markerfacecolor="white" if low_blocks else DS_COLOR[r["ds"]],
                        markeredgewidth=1.4, elinewidth=1.5, capsize=3.5, zorder=3)
        ax.errorbar(macro, y_macro, xerr=[[macro - m_lo], [m_hi - macro]], fmt="D",
                    ms=7, color="black", elinewidth=1.8, capsize=4, zorder=4)
        for s in seed_macros.values():
            ax.plot(s, y_macro, marker="|", ms=11, color="0.45", mew=1.4, zorder=5)
        ax.plot(shift_macro, y_shift, marker="s", ms=6.5, color="0.45",
                markerfacecolor="white", mew=1.3, zorder=4)
        ax.axhline(y_macro + 0.75, color="0.8", lw=0.8)
        ax.set_axisbelow(True)

    ax_f.axvspan(FLOOR, GATE, color="0.5", alpha=0.07, zorder=0)
    ax_f.axvline(GATE, color=SEMANTIC["threshold"], ls=LINESTYLE["threshold"], lw=1.5)
    ax_f.axvline(FLOOR, color=SEMANTIC["baseline"], ls=LINESTYLE["baseline"], lw=1.2)
    ax_f.set_xlim(-1.45, 1.45)
    ax_f.set_xlabel("relative improvement of C over I, MSE (%)   positive = C better")
    ax_f.set_title("full scale - the pre-registered gate is on this axis")
    ax_f.text(GATE - 0.05, 2.5, "gate +1.0%", color=SEMANTIC["threshold"],
              fontsize=8.5, ha="right", va="center", rotation=90)
    ax_f.text(FLOOR + 0.05, 2.5, "floor -1.0%", color="0.45",
              fontsize=8.5, ha="left", va="center", rotation=90)

    ax_z.set_xlim(-0.62, 0.28)
    zoom_ratio = (1.45 + 1.45) / (0.28 + 0.62)
    ax_z.set_xlabel("same quantity, zoomed x-scale (%)")
    ax_z.set_title("zoom - x-scale {:.1f}x narrower".format(zoom_ratio))
    ax_z.annotate("gate +1.0% is off-scale to the right", xy=(0.98, 0.03),
                  xycoords="axes fraction", ha="right", va="bottom",
                  fontsize=8.5, color=SEMANTIC["threshold"])
    plt.setp(ax_z.get_yticklabels(), visible=False)

    labels = ["{:<11s} H={:>3d}".format(r["ds"], r["h"]) + ("  *" if r["nb"] < 8 else "")
              for r in rows]
    ax_f.set_yticks(ys + [y_macro, y_shift])
    ax_f.set_yticklabels(labels + ["macro (6 cells)", "macro, shifted grid"],
                         fontfamily="monospace", fontsize=9)
    ax_f.set_ylim(y_shift - 0.8, n - 0.3)

    lines = [
        "C vs I  -  relative improvement in test MSE, positive = C better",
        "",
        "{:<18}{:>8}   {:<20}{:>8}{:>8}".format("cell", "RI %", "95% interval",
                                                "origins", "blocks"),
        "-" * 64,
    ]
    for r in rows:
        lines.append("{:<18}{:>+8.3f}   [{:>+6.3f}, {:>+6.3f}]{:>8d}{:>8d}".format(
            r["ds"] + "  H=" + str(r["h"]), r["ri"], r["lo"], r["hi"], r["n"], r["nb"])
            + ("  *" if r["nb"] < 8 else ""))
    lines += [
        "-" * 64,
        "{:<18}{:>+8.3f}   [{:>+6.3f}, {:>+6.3f}]".format(
            "macro (6 cells)", macro, m_lo, m_hi),
        "{:<18}{:>+8.3f}   1000 draws, block = {} origins".format(
            "bootstrap mean", boot["macro_mean"], boot["block_origins"]),
        "",
        "per-seed macro     {:>+8.3f} (seed {})".format(seed_macros[SEED0], SEED0),
        "                   {:>+8.3f} (seed {})".format(seed_macros[SEED1], SEED1),
        "phase-shifted grid {:>+8.3f} (robustness, no interval)".format(shift_macro),
        "",
        "* electricity has only 4 effective time blocks. Its interval is",
        "  narrow because the resample has few independent time units,",
        "  not because the estimate is better resolved.",
        "",
        "pre-registered conditions",
        "-" * 64,
        "{:<44}{:>10}".format("macro RI >= +1.0%", "NOT MET"),
        "{:<44}{:>10}".format("bootstrap lower95 > 0", "NOT MET"),
        "{:<44}{:>10}".format("at least 2 datasets positive", "NOT MET"),
        "{:<44}{:>10}".format("both per-seed macros positive", "NOT MET"),
        "{:<44}{:>10}".format("no dataset worse than -1.0%", "met"),
        "",
        "gate / |observed macro| = {:.0f}x".format(GATE / abs(macro)),
        "seed-to-seed sd of validation MSE reaches 5.05% (ETTm2),",
        "which the two-seed interval above does not carry.",
    ]
    ax_t.axis("off")
    ax_t.text(0.0, 1.0, "\n".join(lines), transform=ax_t.transAxes, va="top",
              ha="left", family="monospace", fontsize=7.6,
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    handles = [
        Line2D([], [], marker="o", ls="", color="0.35", ms=6.5,
               label="cell estimate, 8 effective time blocks"),
        Line2D([], [], marker="o", ls="", color="0.35", ms=6.5, mfc="white", mew=1.4,
               label="cell estimate, 4 effective time blocks (electricity)"),
        Line2D([], [], marker="D", ls="", color="black", ms=7,
               label="macro over the 6 cells, bootstrap 95% interval"),
        Line2D([], [], marker="|", ls="", color="0.45", ms=11, mew=1.4,
               label="macro within one model seed (2 seeds)"),
        Line2D([], [], marker="s", ls="", color="0.45", ms=6.5, mfc="white", mew=1.3,
               label="macro on the phase-shifted grid (no interval)"),
        Line2D([], [], color=SEMANTIC["threshold"], ls=LINESTYLE["threshold"], lw=1.5,
               label="pre-registered gate +1.0%"),
        Line2D([], [], color=SEMANTIC["baseline"], ls=LINESTYLE["baseline"], lw=1.2,
               label="worst-dataset floor -1.0%"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle(
        "HQ-TOKEN-PILOT-v1  primary contrast  C vs I  -  horizon-conditioned pooling "
        "does not beat input-only pooling at 32 tokens\n"
        "macro {:+.3f}%  95% CI [{:+.3f}, {:+.3f}]  against a pre-registered gate of "
        "+1.0%   ->  decision INCONCLUSIVE".format(macro, m_lo, m_hi),
        fontsize=12)
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.955))
    footer(fig)
    p = OUT / "fig1_primary_contrast.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2 ---
def fig2(con, met, eff):
    mse = {}
    for r in met:
        mse[(r["dataset"], int(r["horizon"]), r["arm"], r["model_seed"])] = float(r["MSE"])
    lat = {a: float(eff[ARM_CFG[a]]["end_to_end_median_ms_b64"]) for a in ARMS}

    def cell_ri(key, ds, h):
        node = con["primary"] if key == "primary" else con["secondary"][key]
        return node["cells"][ds + "|" + str(h)]

    # label offsets tuned once to the fixed x positions of the six configs
    off = {"DENSE": (0, 10), "U": (0, 10), "H_STATIC": (0, -16), "I": (0, 10),
           "R": (0, -16), "C": (0, 10)}

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), sharex=True)
    for i, h in enumerate(HORIZONS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            y0 = []
            for a in ARMS:
                v = mse[(ds, h, a, SEED0)]
                y0.append(v)
                ax.plot(lat[a], v, marker="s" if a == "DENSE" else "o", ms=8,
                        color=ARM_COLOR[a], ls="", zorder=4)
                k1 = (ds, h, a, SEED1)
                if k1 in mse:
                    ax.plot([lat[a], lat[a]], [v, mse[k1]], color=ARM_COLOR[a],
                            lw=1.0, alpha=0.55, zorder=2)
                    ax.plot(lat[a], mse[k1], marker="o", ms=7, ls="",
                            color=ARM_COLOR[a], markerfacecolor="white", mew=1.3,
                            zorder=3)
                ax.annotate(a, (lat[a], v), textcoords="offset points",
                            xytext=off[a], ha="center", fontsize=8,
                            color=ARM_COLOR[a])
            u = mse[(ds, h, "U", SEED0)]
            ax.axhline(u, color=SEMANTIC["baseline"], ls=LINESTYLE["baseline"], lw=1.0,
                       zorder=1)
            ax.annotate("U level", (0.012, u), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color="0.45", va="bottom")

            allv = y0 + [mse[(ds, h, a, SEED1)] for a in ARMS if (ds, h, a, SEED1) in mse]
            lo, hi = min(allv), max(allv)
            pad = (hi - lo) * 0.34
            ax.set_ylim(lo - pad, hi + pad)

            txt = ["C vs U {:>+7.2f}%    C vs H_STATIC {:>+7.2f}%".format(
                       cell_ri("C_vs_U", ds, h), cell_ri("C_vs_H_STATIC", ds, h)),
                   "C vs I {:>+7.2f}%    C vs DENSE    {:>+7.2f}%".format(
                       cell_ri("primary", ds, h), cell_ri("C_vs_DENSE", ds, h))]
            ax.set_title("{}   H={}".format(ds, h), fontsize=11, pad=30)
            ax.text(0.5, 1.012, "\n".join(txt), transform=ax.transAxes, ha="center",
                    va="bottom", family="monospace", fontsize=7.8, color="0.25")
            if i == 1:
                ax.set_xlabel("end-to-end median latency, batch 64 (ms)")
            if j == 0:
                ax.set_ylabel("test MSE")
    axes[0, 0].set_xlim(1.85, 2.92)

    handles = [Line2D([], [], marker="s" if a == "DENSE" else "o", ls="", ms=8,
                      color=ARM_COLOR[a], label=ARM_GLOSS[a]) for a in ARMS]
    handles += [
        Line2D([], [], marker="o", ls="", ms=8, color="0.35",
               label="filled = model seed " + SEED0 + " (only seed with all six arms)"),
        Line2D([], [], marker="o", ls="", ms=7, color="0.35", mfc="white", mew=1.3,
               label="open = model seed " + SEED1 + " (I, H_STATIC, C only)"),
        Line2D([], [], color=SEMANTIC["baseline"], ls=LINESTYLE["baseline"], lw=1.0,
               label="U level (uniform pooling at the same 32-token budget)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5)
    dense_gain = con["precondition_compression_headroom_DENSE_vs_U"]["macro"]
    header = (
        "HQ-TOKEN-PILOT-v1  accuracy vs latency at the token budgets actually run  "
        "(32 tokens: U, H_STATIC, I, C, R   |   64 tokens: DENSE)\n"
        "Learned pooling buys real accuracy over uniform pooling (C vs U macro "
        "{:+.2f}%); which query the scorer gets barely matters (C vs I macro "
        "{:+.2f}%).\n"
        "Accuracy the 64 to 32 compression gave up: DENSE vs U {:+.2f}%.  Panel "
        "numbers are the pre-registered per-cell contrasts; vs U and vs DENSE exist "
        "for seed {} only.\n"
        "Latency note: at 1.5M parameters the pooling scorer costs more than the "
        "shorter sequence saves, so compressing to 32 tokens does not make this model "
        "faster end to end.".format(
            con["secondary"]["C_vs_U"]["macro"], con["primary"]["macro"], dense_gain,
            SEED0))
    fig.text(0.5, 0.996, header, ha="center", va="top", fontsize=10.5)
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.855))
    footer(fig)
    p = OUT / "fig2_accuracy_vs_latency.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 3 ---
def fig3(diag, cap):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), sharey=True,
        gridspec_kw=dict(width_ratios=[2.5, 1.0]))

    g = np.arange(1, 33)
    max_gap = {}
    for ds in DATASETS:
        wp = diag[ds]["weight_profile"]
        w96 = np.array(wp["weights_h96_mean"])[:, 0]
        w336 = np.array(wp["weights_h336_mean"])[:, 0]
        max_gap[ds] = float(np.abs(w96 - w336).max())
        axL.plot(g, w96, color=DS_COLOR[ds], ls="-", marker="o", ms=4, lw=1.2)
        axL.plot(g, w336, color=DS_COLOR[ds], ls="--", marker="x", ms=5, lw=1.2,
                 alpha=0.9)
    axL.axhline(0.5, color=SEMANTIC["mean"], ls=LINESTYLE["mean"], lw=1.0, zorder=1)
    axL.set_xlim(0.3, 32.7)
    axL.set_ylim(0.0, 1.05)
    axL.set_yticks(np.arange(0, 1.01, 0.2))
    axL.set_xlabel("pooling group index (32 groups, r = 2 patches per group)")
    axL.set_ylabel("mean pooling weight on the first patch of the group")
    axL.set_title("real data, arm C - the same 256 test windows scored at H=96 and "
                  "H=336 (seed " + SEED0 + ")")

    stats = ["weight change when the horizon changes", "-" * 40,
             "{:<14}{:>10}{:>10}".format("", "mean", "max")]
    for ds in DATASETS:
        stats.append("{:<14}{:>10.5f}{:>10.5f}".format(
            ds, diag[ds]["weight_profile"]["mean_abs_weight_diff"], max_gap[ds]))
    stats += ["-" * 40,
              "{:<14}{:>10.5f}   (right panel)".format(
                  "synthetic C", cap["C_weight_separation_between_horizons"])]
    axL.text(0.02, 0.98, "\n".join(stats), transform=axL.transAxes, va="top", ha="left",
             family="monospace", fontsize=8.2,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    hL = [Line2D([], [], color=DS_COLOR[d], lw=1.6, label=d) for d in DATASETS]
    hL += [Line2D([], [], color="0.35", ls="-", marker="o", ms=4, label="H = 96"),
           Line2D([], [], color="0.35", ls="--", marker="x", ms=5, label="H = 336"),
           Line2D([], [], color=SEMANTIC["mean"], ls=LINESTYLE["mean"], lw=1.0,
                  label="uniform pooling, w = 1/r = 0.5")]
    axL.legend(handles=hL, loc="upper right", ncol=2, fontsize=8.5)

    syn = {"C": cap["C_mean_weight_on_a_patch"], "I": cap["I_mean_weight_on_a_patch"]}
    xpos = {"C": 0.0, "I": 1.0}
    for arm, x in xpos.items():
        w96, w336 = syn[arm]["96"], syn[arm]["336"]
        axR.plot([x, x], [w96, w336], color=ARM_COLOR[arm], lw=2.0, zorder=2)
        axR.plot(x, w96, marker="o", ms=10, color=ARM_COLOR[arm], ls="", zorder=3)
        axR.plot(x, w336, marker="x", ms=11, mew=2.5, color=ARM_COLOR[arm], ls="",
                 zorder=3)
        sep = abs(w96 - w336)
        axR.annotate("", xy=(x + 0.16, w96), xytext=(x + 0.16, w336),
                     arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.1))
        axR.text(x + 0.20, (w96 + w336) / 2, "separation\n{:.3f}".format(sep),
                 fontsize=8.5, va="center", ha="left", color="0.25")
    axR.axhline(0.5, color=SEMANTIC["mean"], ls=LINESTYLE["mean"], lw=1.0, zorder=1)
    axR.set_xlim(-0.45, 1.80)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["C\nhorizon-conditioned", "I\ninput-only"], fontsize=9)
    axR.set_title("synthetic positive control")
    axR.set_xlabel("arm (same y-axis and ticks as the left panel)")

    fig.suptitle(
        "HQ-TOKEN-PILOT-v1  pooling weights under a horizon change  -  the same "
        "architecture separates the horizons by {:.3f} when it has to, and by at most "
        "{:.4f} on real data\n"
        "Both panels share the same y-axis and ticks. On real data the H=96 and H=336 "
        "curves lie on top of each other; C behaves like a fixed, input-only "
        "pooler.".format(cap["C_weight_separation_between_horizons"],
                         max(max_gap.values())),
        fontsize=11.5)
    fig.get_layout_engine().set(rect=(0, 0.022, 1, 0.973))
    footer(fig)
    p = OUT / "fig3_pooling_weights.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- captions ---
def captions(boot, con, diag, cap):
    macro = con["primary"]["macro"]
    txt = """# HQ-TOKEN-PILOT-v1 - figure captions

Generated {stamp} by `experiments/hq_token_pilot_v1/figures.py`.
These are analysis-mode figures: they are meant to be read together with the
numbers in STATUS.md, not as standalone paper figures.

## fig1_primary_contrast.png

Relative improvement in test MSE of the horizon-conditioned pooler (C) over the
input-only pooler (I) for each dataset x horizon cell, with paired moving-block
bootstrap 95% intervals, shown at full scale next to the pre-registered +1.0% gate
and again on a 3.2x narrower x-scale; the macro over the six cells is {macro:+.3f}%
[{lo:+.3f}, {hi:+.3f}], about {ratio:.0f} times smaller than the gate it had to
clear.
Limitation: the intervals resample time origins only and not model seeds, so they
do not carry the seed-to-seed spread of up to 5.05% measured on ETTm2, and the two
electricity cells rest on 4 effective time blocks, which makes their intervals
narrow for a reason that has nothing to do with precision.

## fig2_accuracy_vs_latency.png

Test MSE against measured end-to-end latency for every configuration that was
actually trained - the five 32-token arms (U, H_STATIC, I, C, R) and the 64-token
DENSE arm - one panel per dataset x horizon cell, showing that learned pooling is
worth about {cu:+.2f}% over uniform pooling at the same token budget while the
choice of query is worth {macro:+.2f}%, and that the 64 to 32 compression itself
gave up {du:+.2f}% of accuracy.
Limitation: filled markers are the single model seed {seed0}, the only seed that
has all six arms, the open markers show the second seed for the three core arms and
the gap between seeds is often wider than the gap between arms; latency was measured
on this 1.5M-parameter pilot, where the pooling scorer costs more than the shorter
sequence saves, so the x-axis must not be read as the cost profile of a large model.

## fig3_pooling_weights.png

Mean pooling weight on the first patch of each of the 32 groups at H=96 against
H=336 for the same inputs: on the three real datasets the two horizon curves are
indistinguishable (mean absolute weight change {d0:.5f}, {d1:.5f}, {d2:.5f}, and no
single group moves more than {mx:.5f}), while the same architecture trained on a
synthetic task where the horizon must matter separates the two horizons by {sep:.3f}
on an identical y-axis.
Limitation: the right panel is a capacity check on a synthetic task and is not a
benchmark result, and the left panel is a mean over 256 test windows of the first
weight in a group of two, so it shows that the horizon-conditioned query is inert on
average, not that no individual window ever moved.
""".format(
        stamp=STAMP,
        macro=macro,
        lo=boot["lower95"],
        hi=boot["upper95"],
        ratio=GATE / abs(macro),
        cu=con["secondary"]["C_vs_U"]["macro"],
        du=con["precondition_compression_headroom_DENSE_vs_U"]["macro"],
        seed0=SEED0,
        d0=diag["ETTm2"]["weight_profile"]["mean_abs_weight_diff"],
        d1=diag["weather"]["weight_profile"]["mean_abs_weight_diff"],
        d2=diag["electricity"]["weight_profile"]["mean_abs_weight_diff"],
        mx=max(float(np.abs(np.array(diag[d]["weight_profile"]["weights_h96_mean"])[:, 0]
                            - np.array(diag[d]["weight_profile"]["weights_h336_mean"])
                            [:, 0]).max()) for d in DATASETS),
        sep=cap["C_weight_separation_between_horizons"],
    )
    p = OUT / "captions.md"
    p.write_text(txt, encoding="utf-8")
    return p


def main():
    boot, con, diag, cap, met, eff = load()
    paths = [fig1(boot, con), fig2(con, met, eff), fig3(diag, cap),
             captions(boot, con, diag, cap)]
    for p in paths:
        ok = "OK " if p.exists() else "MISSING "
        size = p.stat().st_size if p.exists() else 0
        print("{}{}  {} bytes".format(ok, p, size))


if __name__ == "__main__":
    main()
