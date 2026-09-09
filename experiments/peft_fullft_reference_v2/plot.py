"""English study20 score and measured-cost figures, exported as PNG and PDF."""

import argparse
import json
import os
from pathlib import Path
import sys

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "2"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.peft_fullft_reference_v2.contract import ROOT, STUDY, digest, save_json


def render(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    if summary.get("completed") is not True:
        raise ValueError("Plot only completed frozen analysis")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / (name + suffix) for name in ("scores", "compute_memory") for suffix in (".png", ".pdf")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve existing study20 figures")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    methods = ("F0", "SIMPLE", "HEAD_ONLY", "LORA", "FULL_FT")
    labels = ("Frozen", "Simple", "Head only", "LoRA", "Full FT")
    colors = ("#6B7280", "#BE8543", "#218C83", "#2878B5", "#975CB4")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    for ax, dataset in zip(axes, ("bike", "household")):
        data = summary["datasets"][dataset]["arms"]
        means = [data[method]["SORT"]["score_mean"] for method in methods]
        ax.bar(np.arange(5), means, width=.62, color=colors, alpha=.77, zorder=2)
        for index, method in enumerate(methods):
            values = data[method]["SORT"]["seed_values"]
            spread = np.linspace(-.14, .14, len(values)) if len(values) > 1 else np.array([0.])
            ax.scatter(index + spread, values, s=20, color="#1F2937", edgecolors="white", linewidths=.45, zorder=4)
            ax.scatter(index, data[method]["QCAL"]["score_mean"], marker="D", s=28,
                       facecolor="white", edgecolor="#111827", linewidth=1., zorder=5)
        ax.axhline(means[0], color=colors[0], linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
        ax.set_xticks(np.arange(5), labels)
        ax.set_title("Bike sharing" if dataset == "bike" else "Household power", loc="left", fontweight="bold")
        ax.set_ylabel("Normalized mean 2-pinball loss (lower is better)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="", color="#1F2937", label="Individual optimizer seed"),
               Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="#111827", label="Common QCAL mean (secondary)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Study20: forecast quality after frozen validation selection", fontsize=12, fontweight="bold")
    for suffix in (".png", ".pdf"):
        fig.savefig(output / ("scores" + suffix), dpi=300)
    plt.close(fig)

    costs = summary["costs"]["by_arm"]
    train_methods = methods[2:]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    x = np.arange(3)
    rates = [costs[method]["optimizer_seconds_per_step_mean"] for method in train_methods]
    axes[0].bar(x, rates, color=colors[2:], width=.58)
    axes[0].set_ylabel("Mean optimizer step time (seconds)")
    axes[0].set_title("Compute across all candidate fits", loc="left", fontweight="bold")
    memory = [costs[method]["peak_cuda_allocated_bytes"] / 1024 ** 3 for method in train_methods]
    reserve = [costs[method]["peak_cuda_reserved_bytes"] / 1024 ** 3 for method in train_methods]
    axes[1].bar(x - .16, memory, width=.30, color=colors[2:], label="Peak allocated")
    axes[1].bar(x + .16, reserve, width=.30, color=colors[2:], alpha=.35, hatch="//", label="Peak reserved")
    axes[1].set_ylabel("Maximum CUDA memory over candidate fits (GiB)")
    axes[1].set_title("Allocator peaks; not system RSS", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels[2:])
        ax.grid(axis="y", color="#E5E7EB", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    fig.suptitle("54 candidate fits included; head training uses direct forward passes", fontsize=11, fontweight="bold")
    for suffix in (".png", ".pdf"):
        fig.savefig(output / ("compute_memory" + suffix), dpi=300)
    plt.close(fig)
    return {str(path): digest(path) for path in paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary_path = args.root / "results" / STUDY / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    directory = summary_path.parent / "figures"
    hashes = render(summary, directory)
    save_json(directory / "manifest.json", {"completed": True, "summary_sha256": digest(summary_path), "outputs": hashes})
    print(json.dumps({"completed": True, "files": list(hashes)}))
