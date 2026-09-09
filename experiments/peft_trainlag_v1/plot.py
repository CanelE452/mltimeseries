"""Standalone figures for numerically verified train-only lag comparisons."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("F0", "ATTN", "ALIGN_F0", "ALIGN_ATTN", "RAW", "ORACLE")
COLORS = dict(zip(METHODS, ("#64748b", "#2563eb", "#db2777", "#7c3aed", "#15803d", "#111827")))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/peft_trainlag_v1"
    study = root / "runs/peft_trainlag_v1"
    assert read(output / "verification.json")["passed"]
    with (output / "selected_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18 and {row["method"] for row in rows} == set(METHODS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for index, method in enumerate(METHODS):
        values = np.array([float(row["score"]) for row in rows if row["method"] == method])
        assert len(values) == 3
        y = len(METHODS) - index - 1
        ax.scatter(values, y + np.array([-.1, 0, .1]), color=COLORS[method], s=15, alpha=.55)
        ax.errorbar(values.mean(), y, xerr=values.std(ddof=1), fmt="o", color=COLORS[method],
                    capsize=4, markersize=7, linewidth=1.6)
        ax.annotate(f"{values.mean():.4f}", (values.mean(), y), xytext=(8, 10),
                    textcoords="offset points", fontsize=9, color=COLORS[method])
    ax.set_yticks(range(6), list(reversed(METHODS)))
    ax.set_xlabel("Raw mean 2-pinball (lower is better)")
    ax.set_ylim(-.5, 5.6)
    ax.margins(x=.12)
    ax.grid(axis="x", color="#e2e8f0")
    fig.suptitle("Train-estimated lags: native TSFM adaptation and a linear control", fontsize=15, y=.98)
    fig.text(.5, .02, "Means over 3 training corpora; whiskers: ±1 repetition SD, not confidence intervals.\n"
             "Fresh 512 shared evaluation episodes from the same Q00 family. ORACLE has privileged generator information.\n"
             "ALIGN and RAW share train-estimated lags; ALIGN_F0 includes supervised preprocessing, not zero-shot inference.",
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .15, 1, .94))
    save(fig, output, "score_comparison")

    effects = read(output / "effects.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].axvspan(-1, 1, color="#dcfce7", alpha=.65)
    axes[0].axvline(0, color="#64748b", linewidth=1)
    for y, key, method in ((1, "alignment_over_attention", "ALIGN_F0"),
                            (0, "attention_after_alignment", "ALIGN_ATTN")):
        effect = effects["primary"]["effects"][key]
        value = 100 * effect["value"]
        low, high = 100 * np.asarray(effect["ci"])
        axes[0].errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", color=COLORS[method], capsize=5)
        axes[0].annotate(f"{value:+.2f}% [{low:+.2f}, {high:+.2f}]", (value, y),
                         xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_yticks([0, 1], ["Attention after alignment", "Alignment vs raw attention"])
    axes[0].set_title("Primary family · paired 97.5% CIs", fontsize=11)
    axes[0].set_xlabel("Improvement / F0 (%) · positive = better")
    axes[0].set_ylim(-.5, 1.6)
    axes[0].margins(x=.2)
    raw = effects["raw_oracle_diagnostic"]
    value, (low, high) = 100 * raw["value"], 100 * np.asarray(raw["ci"])
    axes[1].errorbar(value, 0, xerr=[[value - low], [high - value]], fmt="o", color=COLORS["RAW"], capsize=5)
    axes[1].annotate(f"{value:+.3f}%\n[{low:+.3f}, {high:+.3f}]", (value, 0),
                     xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9)
    axes[1].axvline(1, color="#15803d", linestyle="--", linewidth=1)
    axes[1].axvline(0, color="#64748b", linewidth=1)
    axes[1].set_xlim(min(-.25, low - .25), max(1.25, high + .25))
    axes[1].set_yticks([])
    axes[1].set_ylim(-.5, .6)
    axes[1].set_title("Separate RAW diagnostic · paired 95% CI", fontsize=11)
    axes[1].set_xlabel("(RAW − ORACLE) / F0 (%)")
    for ax in axes:
        ax.grid(axis="x", color="#e2e8f0")
    fig.suptitle("Paired effects and proximity to the privileged oracle", fontsize=15, y=.98)
    fig.text(.5, .02, "4,000 paired episode bootstraps, conditional on 3 fitted repetitions. Green band/line: 1% operational margin.\n"
             "The RAW diagnostic is a separate exploratory family. There is no joint family-wise claim across both panels.",
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .15, 1, .94))
    save(fig, output, "paired_effects")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True)
    selections = [read(study / "data" / f"Q00_c{c}_lags.json") for c in range(3)]
    colors = ("#2563eb", "#d97706", "#15803d")
    for ax, channel in zip(axes, ("Y", "U", "V")):
        for c, selection in enumerate(selections):
            lags = np.array(selection["candidate_lags"])
            values = abs(np.array(selection["pearson_correlations"][channel]))
            chosen = selection["selected_lags"][channel]
            ax.plot(lags, values, color=colors[c], linewidth=1, alpha=.8, label=f"corpus {c}")
            ax.scatter(chosen, values[lags == chosen][0], color=colors[c], s=23)
        peaks = ", ".join(str(item["selected_lags"][channel]) for item in selections)
        ax.set_title(f"{channel} · selected lags {peaks}", fontsize=11)
        ax.set_xlabel("Candidate lag (16 to 128)")
        ax.grid(color="#e2e8f0", linewidth=.6)
    axes[0].set_ylabel("Absolute train Pearson correlation")
    axes[-1].legend(frameon=False)
    fig.suptitle("Lag selection uses only the 64 × 16 future-Y training labels", fontsize=15, y=.98)
    fig.text(.5, .02, "113 prespecified candidates per named channel; no validation, evaluation or oracle values enter selection.\n"
             "Each panel has its own y-axis scale. Peaks are observed training statistics; selection success is not a data QC gate.",
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .14, 1, .94))
    save(fig, output, "lag_search")

    with (output / "all_results.csv").open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    raw_f0 = np.mean([float(row["val_score"]) for row in rows if row["method"] == "F0"])
    aligned_f0 = np.mean([float(row["val_score"]) for row in rows if row["method"] == "ALIGN_F0"])
    for ax, lr in zip(axes, (3e-5, 1e-4)):
        for method in ("ATTN", "ALIGN_ATTN"):
            chosen = [row for row in all_rows if row["method"] == method and float(row["lr"]) == lr]
            assert len(chosen) == 3
            histories = [read(root / row["path"] / "result.json")["validation_history"] for row in chosen]
            steps = [item["step"] for item in histories[0]]
            assert all([item["step"] for item in history] == steps for history in histories)
            values = np.array([[item["val_score"] / raw_f0 for item in history] for history in histories])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(steps, mean, label=method, color=COLORS[method], linewidth=1.8)
            ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[method], alpha=.12)
        ax.axhline(1, label="F0", color=COLORS["F0"], linestyle="--", linewidth=1)
        ax.axhline(aligned_f0 / raw_f0, label="ALIGN_F0", color=COLORS["ALIGN_F0"], linestyle="--", linewidth=1)
        ax.set_title(f"LR {lr:.0e}")
        ax.set_xlabel("Optimizer update")
        ax.grid(color="#e2e8f0", linewidth=.6)
    axes[0].set_ylabel("Validation score / raw F0 (lower is better)")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Training trajectories with and without estimated-lag alignment", fontsize=15, y=.98)
    fig.text(.5, .02, "Mean ±1 repetition SD across 3 fits; descriptive curves, not confidence intervals.\n"
             "Each fit selects its best validation checkpoint including step 0; corpus0 validation selects the LR per procedure.",
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .14, 1, .94))
    save(fig, output, "validation_trajectories")
    print(json.dumps({"figures": [str(output / f"{name}.{suffix}") for name in
                                  ("score_comparison", "paired_effects", "lag_search", "validation_trajectories")
                                  for suffix in ("png", "pdf")]}))


if __name__ == "__main__":
    main()
