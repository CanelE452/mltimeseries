"""Static result figures; run only after numerical verification passes."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/peft_shift_mechanism_v1"
    if not json.loads((output / "verification.json").read_text())["passed"]:
        raise RuntimeError("Verify numeric results before plotting")
    with (output / "selected_results.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    conditions = ("Q00", "Q10", "Q01", "Q11")
    methods = ("F0", "H_LIN", "H_MLP", "OFF_LORA", "JOINT", "LP", "TIME", "GROUP", "RAW", "F0_RAW", "ORACLE")
    alignment_path = output / "alignment_diagnostic/result.json"
    aligned = {}
    if alignment_path.exists():
        diagnostic = json.loads(alignment_path.read_text())
        if not diagnostic["completed"]:
            raise RuntimeError("Alignment result is incomplete")
        aligned = {row["condition"]: row["eval_score"] for row in diagnostic["conditions"]}
        assert set(aligned) == set(conditions)
        methods = methods[:8] + ("ALIGNED*",) + methods[8:]
    titles = ("Q00 · reference", "Q10 · self lag 32 → 64", "Q01 · driver angle 15° → 45°", "Q11 · both changes")
    colors = {"F0": "#64748b", "H_LIN": "#8b5cf6", "H_MLP": "#a78bfa", "OFF_LORA": "#2563eb",
              "JOINT": "#dc2626", "LP": "#e8790d", "TIME": "#0891b2", "GROUP": "#a16207",
              "RAW": "#15803d", "F0_RAW": "#14b8a6", "ORACLE": "#111827", "ALIGNED*": "#db2777"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharex=True, sharey=True)
    for ax, condition, title in zip(axes, conditions, titles):
        for index, method in enumerate(methods):
            y = len(methods) - index - 1
            if method == "ALIGNED*":
                ax.plot(aligned[condition], y, marker="D", color=colors[method], markersize=6)
                continue
            values = [float(r["score"]) for r in records if r["condition"] == condition and r["method"] == method]
            assert len(values) == 3
            ax.errorbar(np.mean(values), y, xerr=np.std(values, ddof=1), fmt="o",
                        color=colors[method], capsize=3, markersize=6, linewidth=1.6)
        ax.set_title(title, pad=12, fontsize=11)
        ax.set_xlabel("Raw mean 2-pinball ↓")
        ax.grid(axis="x", color="#e2e8f0", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(methods)), list(reversed(methods)))
    fig.suptitle("Chronos-2 adaptation on controlled lagged-covariate processes", fontsize=16, y=.98)
    footer = "Circles: mean of 3 corpus/optimizer repetitions; whiskers: ±1 repetition SD, not confidence intervals.\n"
    footer += "512 shared evaluation episodes. RAW uses a known lag dictionary and named drivers."
    if aligned:
        footer += "\n*ALIGNED: exploratory frozen-input diagnostic, one validation-selected lag; no training repetitions or extra future observations."
    fig.text(.5, .02, footer,
             ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .14 if aligned else .105, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"score_comparison.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
    for ax, condition, title in zip(axes, conditions, titles):
        f0 = float(next(r["val_score"] for r in records if r["condition"] == condition and r["method"] == "F0"))
        for method in ("H_LIN", "OFF_LORA", "JOINT", "LP", "TIME", "GROUP"):
            histories = []
            for record in records:
                if record["condition"] == condition and record["method"] == method:
                    result = json.loads((root / record["path"] / "result.json").read_text())
                    histories.append(result["validation_history"])
            steps = [record["step"] for record in histories[0]]
            assert all([record["step"] for record in history] == steps for history in histories)
            values = np.array([[record["val_score"] / f0 for record in history] for history in histories])
            ax.plot(steps, values.mean(axis=0), color=colors[method], label=method, linewidth=1.8)
        ax.axhline(1, color="#64748b", linestyle="--", linewidth=1)
        ax.axvline(80, color="#cbd5e1", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Optimizer update")
        ax.grid(color="#e2e8f0", linewidth=.6)
    axes[0].set_ylabel("Validation score / F0 (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(.5, .075), frameon=False)
    fig.suptitle("Training trajectories explain the selected checkpoint", fontsize=16, y=.98)
    fig.text(.5, .02, "Means across the selected-LR repetitions. F0 line = 1. LP activates LoRA after update 80.\n"
             "The best of checkpoints 0, 40, 80, 120, 160, 200 is selected on validation; curves are descriptive.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .16, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"validation_trajectories.{suffix}", dpi=200)
    plt.close(fig)
    print(json.dumps({"figures": [str(output / name) for name in
                                   ("score_comparison.png", "score_comparison.pdf", "validation_trajectories.png", "validation_trajectories.pdf")]}))


if __name__ == "__main__":
    main()
