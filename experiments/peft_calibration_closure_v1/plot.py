"""Plot the completed, already observed development calibration diagnostic."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / "results/peft_calibration_closure_v1"
    verified = json.loads((folder / "verification.json").read_text())
    if not verified["passed"]:
        raise RuntimeError("Analysis must pass before plotting")
    rows = json.loads((folder / "summary.json").read_text())["summary"]
    methods = ["F0", "ATTN", "ALIGN_F0", "ALIGN_ATTN"]
    lookup = {(row["method"], row["procedure"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = {"SORT": "#64748b", "QCAL": "#16784b"}
    for ax, metric, title, scale in zip(axes, ("score", "coverage80"),
                                       ("Quantile score (lower is better)", "80% interval coverage (%)"), (1, 100)):
        for i, method in enumerate(methods):
            values = [lookup[(method, p)][metric] * scale for p in ("SORT", "QCAL")]
            ax.plot(values, [i, i], color="#b6c2d0", linewidth=2, zorder=1)
            for value, procedure, dy in zip(values, ("SORT", "QCAL"), (-.14, .15)):
                ax.scatter(value, i, color=colors[procedure], s=44, label=procedure if i == 0 else None, zorder=2)
                ax.annotate(f"{value:.4f}" if scale == 1 else f"{value:.2f}", (value, i + dy),
                            ha="center", va="center", color=colors[procedure], fontsize=9)
        ax.set_yticks(np.arange(4), methods)
        ax.set_ylim(3.5, -.6)
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=.14)
    axes[1].axvspan(78, 82, color="#16784b", alpha=.06)
    axes[1].axvline(80, linestyle="--", color="#16784b", alpha=.6)
    axes[1].legend(loc="lower left", ncol=2)
    fig.suptitle("A simple quantile correction closes the observed coverage gap", fontsize=15)
    fig.text(.5, .055, "Means across 3 fitted corpora; previously observed Gaussian development evaluation.\n"
             "21 offsets fitted on reused validation labels; no new GPU training or conformal coverage guarantee.",
             ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=(0, .15, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"calibration_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
