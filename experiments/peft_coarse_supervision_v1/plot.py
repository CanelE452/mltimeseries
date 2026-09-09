"""Render the already sealed study17 metrics; never select or fit a model."""

from pathlib import Path
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fetch import ROOT, STUDY, read, sha, write_once


def run():
    path = ROOT / "results" / STUDY / "metrics.json"
    result = read(path)
    if not result.get("completed"):
        raise AssertionError("Complete evaluation before plotting")
    output = path.parent / "figures"
    output.mkdir(exist_ok=True)
    names = ("PROFILE", "F0", "COARSE_LIFT", "FROZEN_HEAD", "ATTN_LORA")
    labels = ("Profile", "Frozen FM", "Level ridge", "Frozen head", "Attention LoRA")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(12.5, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.4, 1), hspace=.45, wspace=.28)
    for column, (site, title) in enumerate((("0", "Eagle"), ("1", "Lamb"))):
        ax = figure.add_subplot(grid[0, column])
        arms = result["sites"][site]["arms"]
        levels = np.array([arms[name]["level_mse"] for name in names])
        patterns = np.array([arms[name]["pattern_mse"] for name in names])
        indices = np.arange(len(names))
        ax.bar(indices, levels, color="#3377A3", label="Monthly level error")
        ax.bar(indices, patterns, bottom=levels, color="#D28B32", label="Within-month pattern error")
        ax.set_xticks(indices, labels, rotation=18, ha="right")
        ax.set_ylabel("MSE / training annual mean squared")
        ax.set_title(f"{title}: 8 targets, 3 forecast months", loc="left", weight="bold")
        ax.set_ylim(0, max(levels + patterns) * 1.2)
        for x, total in enumerate(levels + patterns):
            ax.text(x, total, f"{total:.4f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        if column == 0:
            ax.legend(frameon=False, fontsize=9)
    ax = figure.add_subplot(grid[1, :])
    entries = [result["effects"]["sites"][key] for key in ("0", "1")] + [result["effects"]["pooled"]]
    for y, (label, entry) in enumerate(zip(("Eagle", "Lamb", "Pooled"), entries)):
        point = entry["point"] * 100
        lower, upper = np.array(entry["ci95"]) * 100
        ax.plot([lower, upper], [y, y], color="#3377A3", linewidth=2)
        ax.plot(point, y, "o", color="#223A52")
        ax.annotate(f"{point:+.2f}% [{lower:+.2f}, {upper:+.2f}]", (upper, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.axvline(1, color="#777777", linewidth=1, linestyle=":")
    ax.set_yticks(range(3), ("Eagle", "Lamb", "Pooled"))
    ax.invert_yaxis()
    ax.margins(x=.35, y=.3)
    ax.set_xlabel("(Selected simple baseline MSE - LoRA MSE) / frozen FM MSE, %; positive favors LoRA")
    ax.set_title("Target-cluster 95% intervals (4,000 resamples)", loc="left", weight="bold")
    figure.suptitle("Monthly supervision: does internal adaptation add useful fine-scale forecasts?", fontsize=15, x=.07, ha="left")
    policy = result["effects"]["simple_policy"]
    figure.text(.07, .925, f"Validation-selected simple baseline: {policy}. Evaluation: April-June 2017. Lower MSE is better.", fontsize=10)
    figure.text(.07, .025, "Exploratory standard-LoRA screen, not a new-method claim. Intervals exclude temporal and training-seed uncertainty.\n"
                f"Decision: {result['decision']}", fontsize=9)
    figure.subplots_adjust(top=.87, bottom=.14, left=.08, right=.94)
    files = []
    for suffix in ("png", "pdf"):
        destination = output / f"coarse_supervision.{suffix}"
        if destination.exists():
            raise FileExistsError(destination)
        figure.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
        files.append(destination)
    plt.close(figure)
    record = {"completed": True, "metrics_sha256": sha(path), "source_sha256": sha(__file__),
              "figure_hashes": {str(p): sha(p) for p in files}}
    write_once(output / "manifest.json", record)
    return record


if __name__ == "__main__":
    print(run())
