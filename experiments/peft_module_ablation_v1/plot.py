"""Standalone figures for the verified Q00 module ablations."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("F0", "OUT_ONLY", "ATTN_ONLY", "BOTH")
COLORS = {"F0": "#64748b", "OUT_ONLY": "#e8790d", "ATTN_ONLY": "#0891b2", "BOTH": "#2563eb"}
LABELS = ("F0 · 0", "OUT_ONLY · 27,264", "ATTN_ONLY · 1,179,648", "BOTH · 1,206,912")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def chosen(records, method, rates):
    lr = .001 if method == "F0" else (3e-5 if method == "BOTH" else rates[method])
    rows = sorted((row for row in records if row["method"] == method and float(row["lr"]) == lr),
                  key=lambda row: int(row["corpus"]))
    assert len(rows) == 3
    return rows


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/peft_module_ablation_v1"
    assert read(output / "verification.json")["passed"]
    with (output / "selected_results.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    effects = read(output / "effects.json")
    same_rates = effects["primary"]["lr_by_method"] == effects["secondary"]["lr_by_method"]
    families = ("primary",) if same_rates else ("primary", "secondary")
    agreement = " Both procedures select LR 3e-5." if same_rates else ""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})

    fig, grid = plt.subplots(1, len(families), figsize=(10 if same_rates else 12, 5.4), sharex=True, sharey=True, squeeze=False)
    axes = grid[0]
    for ax, family, title in zip(axes, families,
                                 ("Primary: fixed LR 3e-5", "Secondary: corpus0 validation LR")):
        rates = effects[family]["lr_by_method"]
        for index, method in enumerate(METHODS):
            values = np.array([float(row["score"]) for row in chosen(records, method, rates)])
            y = len(METHODS) - index - 1
            ax.scatter(values, y + np.array([-.11, 0, .11]), color=COLORS[method], s=15, alpha=.55)
            ax.errorbar(values.mean(), y, xerr=values.std(ddof=1), fmt="o", color=COLORS[method],
                        capsize=4, markersize=7, linewidth=1.5)
            ax.annotate(f"{values.mean():.4f}", (values.mean(), y), xytext=(7, 13),
                        textcoords="offset points", fontsize=9, color=COLORS[method])
        ax.set_title(title, pad=20)
        ax.set_xlabel("Raw mean 2-pinball (lower is better)")
        ax.grid(axis="x", color="#e2e8f0")
        ax.set_ylim(-.5, 3.6)
    axes[0].set_yticks(range(4), list(reversed(LABELS)))
    fig.suptitle("Q00 native LoRA: output projection vs attention updates", fontsize=15, y=.98)
    fig.text(.5, .02, "Large dots: mean of 3 corpus/optimizer repetitions; whiskers: ±1 repetition SD, not CIs.\n"
             "Small dots: individual repetitions. Labels include trainable parameters; capacities are unequal.\n"
             "512 shared evaluation episodes, previously examined Q00. F0/BOTH are reused references." + agreement,
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .16, 1, .94))
    save(fig, output, "score_comparison")

    fig, grid = plt.subplots(1, len(families), figsize=(10 if same_rates else 12, 4.8), sharex=True, sharey=True, squeeze=False)
    axes = grid[0]
    for ax, family, title in zip(axes, families, ("Primary: fixed LR", "Secondary: selected LR")):
        ax.axvspan(-1, 1, color="#dcfce7", alpha=.65)
        ax.axvline(0, color="#64748b", linewidth=1)
        for y, key, method in ((1, "attention_deletion_penalty", "OUT_ONLY"),
                                (0, "output_deletion_penalty", "ATTN_ONLY")):
            effect = effects[family][key]
            value = 100 * effect["value"]
            low, high = 100 * np.array(effect["ci"])
            ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", color=COLORS[method],
                        capsize=5, markersize=7, linewidth=2)
            ax.annotate(f"{value:+.2f}% [{low:+.2f}, {high:+.2f}]", (value, y), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title, pad=15)
        ax.set_xlabel("Deletion penalty / F0 (%) · positive = worse")
        ax.set_ylim(-.5, 1.6)
        ax.grid(axis="x", color="#e2e8f0")
        ax.margins(x=.2)
    axes[0].set_yticks([0, 1], ["Remove output LoRA\n(ATTN_ONLY − BOTH)",
                               "Remove attention LoRA\n(OUT_ONLY − BOTH)"])
    fig.suptitle("Retraining ablation effects and paired 97.5% intervals", fontsize=15, y=.98)
    fig.text(.5, .02, "Each panel: two contrasts, 4,000 paired episode bootstraps. Green band: ±1% practical margin.\n"
             "Intervals are conditional on these 3 fitted repetitions; secondary comparisons are exploratory.\n"
             "Removing modules before retraining does not identify the mechanism of an already trained BOTH model." + agreement,
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .19, 1, .94))
    save(fig, output, "deletion_effects")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    f0 = float(next(row["val_score"] for row in records if row["method"] == "F0"))
    for ax, lr in zip(axes, (3e-5, 1e-4)):
        for method in METHODS[1:]:
            if method == "BOTH" and lr != 3e-5:
                continue
            rows = chosen(records, method, {"OUT_ONLY": lr, "ATTN_ONLY": lr})
            histories = [read(root / row["path"] / "result.json")["validation_history"] for row in rows]
            steps = [item["step"] for item in histories[0]]
            assert all([item["step"] for item in history] == steps for history in histories)
            values = np.array([[item["val_score"] / f0 for item in history] for history in histories])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(steps, mean, color=COLORS[method], label=method, linewidth=1.8)
            ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[method], alpha=.1)
        ax.axhline(1, color=COLORS["F0"], linestyle="--", linewidth=1)
        ax.set_title(f"LR {lr:.0e}")
        ax.set_xlabel("Optimizer updates")
        ax.grid(color="#e2e8f0", linewidth=.6)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation score / F0 (lower is better)")
    fig.suptitle("Validation trajectories explain checkpoint selection", fontsize=15, y=.98)
    fig.text(.5, .02, "Mean ±1 repetition SD over 3 fits. Each method selects its best validation checkpoint, including step 0.\n"
             "BOTH reference is shown only at its reused LR 3e-5. Curves are descriptive, not hypothesis tests.",
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .14, 1, .94))
    save(fig, output, "validation_trajectories")
    print(json.dumps({"figures": [str(output / f"{name}.{suffix}")
                                  for name in ("score_comparison", "deletion_effects", "validation_trajectories")
                                  for suffix in ("png", "pdf")]}))


if __name__ == "__main__":
    main()
