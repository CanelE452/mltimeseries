"""Plot only verified objective-alignment summaries, never prediction archives."""

import argparse
import csv
from pathlib import Path

from . import analyse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("F0", *analyse.ARMS)
COLORS = {"F0": "#555555", "NATIVE": "#0072B2", "NORM_ALIGNED": "#D55E00", "RAW_ALIGNED": "#009E73"}
LABELS = {"bike": "Bike sharing", "household": "Household power"}
FOOTER = ("Development diagnostic: two previously evaluated sources, one optimizer seed (12000), LR 3e-5, 200 updates.\n"
          "Initial gradient norms are matched; AdamW updates and subsequent clipping are not matched. This is not a novel PEFT method.\n"
          "Primary: SORT NORM_ALIGNED minus RAW_ALIGNED, divided by SORT F0. Date-block CIs exclude training uncertainty.")


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if analyse.sha(name) != expected:
            raise AssertionError(f"Verified plot input changed: {name}")


def load_verified(root):
    folder = Path(root) / "results" / analyse.STUDY
    verification = analyse.read_json(folder / "verification.json")
    if (verification.get("passed") is not True or verification.get("completed") is not True
            or verification.get("selected_procedure_rows") != 16 or verification.get("new_fits") != 6
            or verification.get("new_forecasts") != 6 or verification.get("reused_forecasts") != 2
            or verification.get("s0_count") != 6 or verification.get("native_replay_verified") is not True):
        raise AssertionError("The full objective/replay/forecast audit must pass before plotting")
    hashes = {str(folder/name): value for name, value in verification["output_hashes"].items()}
    hashes[str(folder/"verification.json")] = analyse.sha(folder/"verification.json")
    verify_hashes(hashes)
    with (folder/"selected_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(source, arm, procedure) for source in analyse.DATASETS for arm in ARMS for procedure in analyse.PROCEDURES}
    if len(rows) != 16 or {(r["source"], r["arm"], r["procedure"]) for r in rows} != expected:
        raise AssertionError("The procedure score table is partial or duplicated")
    effects = analyse.read_json(folder/"effects.json")
    diagnostics = analyse.read_json(folder/"diagnostics.json")
    if (effects.get("completed") is not True or effects["bootstrap_seed"] != 2026090815
            or effects["primary_confidence_each"] != .975 or effects["primary_family_size"] != 2):
        raise AssertionError("Objective comparison inferential contract changed")
    for source in analyse.DATASETS:
        scores = {r["arm"]: float(r["score"]) for r in rows if r["source"] == source and r["procedure"] == "SORT"}
        point = (scores["NORM_ALIGNED"]-scores["RAW_ALIGNED"])/scores["F0"]
        reported = effects["sources"][source]["SORT"]["NORM_ALIGNED_vs_RAW_ALIGNED"]["daily"]["7"]["value"]
        if not np.isclose(point, reported, rtol=1e-10, atol=1e-12):
            raise AssertionError("Plot effect differs from its own procedure scores")
    return rows, effects, diagnostics, hashes


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})


def save(fig, output, name):
    fig.text(.06, .025, FOOTER, fontsize=7.8, va="bottom", linespacing=1.5, color="#444444")
    for extension in ("png", "pdf"):
        fig.savefig(Path(output)/f"{name}.{extension}", dpi=200)
    plt.close(fig)


def plot_effects(rows, effects, output):
    fig = plt.figure(figsize=(12, 9.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.], left=.10, right=.97, bottom=.20, top=.89, hspace=.55, wspace=.25)
    fig.suptitle("Does training in the evaluated space improve forecasting?", fontsize=15, y=.96, fontweight="bold")
    for index, source in enumerate(analyse.DATASETS):
        ax = fig.add_subplot(grid[0, index])
        lookup = {(r["arm"], r["procedure"]): float(r["score"]) for r in rows if r["source"] == source}
        for shift, procedure in ((-.18, "SORT"), (.18, "QCAL")):
            ax.bar(np.arange(4)+shift, [lookup[(arm, procedure)] for arm in ARMS], width=.34,
                   color=[COLORS[arm] for arm in ARMS], alpha=1 if procedure == "SORT" else .45,
                   edgecolor="#333333", linewidth=.3, label=procedure)
        ax.set_xticks(np.arange(4), [arm.replace("_", "\n") for arm in ARMS])
        ax.set_ylabel("Train-scale-normalized quantile score (lower is better)")
        ax.set_title(LABELS[source], loc="left")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.15)
    ax = fig.add_subplot(grid[1, :])
    for index, source in enumerate(analyse.DATASETS):
        record = effects["sources"][source]["SORT"]["NORM_ALIGNED_vs_RAW_ALIGNED"]
        point = record["daily"]["7"]["value"]*100
        lo, hi = np.asarray(record["daily"]["7"]["ci"])*100
        ax.plot([lo, hi], [1-index]*2, color="#009E73", linewidth=2)
        ax.scatter([point], [1-index], color="#009E73", marker="D", s=45)
        ax.annotate(f"{point:+.2f}%  [{lo:+.2f}, {hi:+.2f}]", (hi, 1-index), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0, color="#888888", linewidth=1)
    ax.axvline(1, color="#D55E00", linestyle="--", linewidth=1)
    ax.margins(x=.30)
    ax.set_yticks([1, 0], [LABELS[s] for s in analyse.DATASETS])
    ax.set_ylim(-.5, 1.5)
    ax.set_title("Primary: seven-day paired blocks, 97.5% CI each; dashed line is the +1% entry threshold", loc="left", fontsize=10)
    ax.set_xlabel("100 × (NORM_ALIGNED − RAW_ALIGNED) / F0; positive favors raw-space training")
    ax.grid(axis="x", alpha=.15)
    save(fig, output, "01_scores_and_primary_effect")


def plot_trajectories(rows, diagnostics, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.30, top=.82, wspace=.24)
    fig.suptitle("Validation trajectories and the fixed future evaluation", y=.96, fontsize=15, fontweight="bold")
    fig.text(.5, .90, "Lines: raw SORT validation at six fixed checkpoints · diamonds: selected checkpoint's future SORT score", ha="center", fontsize=9)
    for ax, source in zip(axes, analyse.DATASETS):
        for arm in analyse.ARMS:
            record = diagnostics["sources"][source][arm]
            history = record["validation_history"]
            ax.plot([x["step"] for x in history], [x["val_score"] for x in history], marker="o", markersize=3, color=COLORS[arm], label=arm)
            best = min(history, key=lambda x: x["val_score"])
            ax.scatter([best["step"]], [best["val_score"]], facecolors="none", edgecolors=COLORS[arm], s=90)
            score = float(next(r["score"] for r in rows if (r["source"], r["arm"], r["procedure"]) == (source, arm, "SORT")))
            ax.scatter([240], [score], marker="D", color=COLORS[arm], s=35)
        f0 = float(next(r["score"] for r in rows if (r["source"], r["arm"], r["procedure"]) == (source, "F0", "SORT")))
        ax.scatter([240], [f0], marker="X", s=40, color=COLORS["F0"], label="F0 future")
        ax.axvline(220, color="#aaaaaa", linestyle=":")
        ax.set_xticks([0, 40, 80, 120, 160, 200, 240], ["0", "40", "80", "120", "160", "200", "Future"])
        ax.set_title(LABELS[source], loc="left")
        ax.set_xlabel("Optimizer updates (future point is not a training checkpoint)")
        ax.set_ylabel("Raw SORT score; V and future have different time distributions")
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .185), ncol=4, frameon=False, fontsize=8)
    save(fig, output, "02_validation_and_future")


def plot_gradients(diagnostics, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.20, top=.87, hspace=.55, wspace=.25)
    fig.suptitle("What changed after matching initial gradient norms?", fontsize=15, y=.96, fontweight="bold")
    fig.text(.5, .91, "Top: initial gradient direction cosines · bottom: scaled, pre-clipping training gradient norms", ha="center", fontsize=9)
    for index, source in enumerate(analyse.DATASETS):
        calibration = diagnostics["sources"][source]["NATIVE"]["gradient_calibration"]
        matrix = np.eye(3)
        for i, left in enumerate(analyse.ARMS):
            for j in range(i+1, 3):
                matrix[i, j] = matrix[j, i] = calibration["cosines"][f"{left}__{analyse.ARMS[j]}"]
        ax = axes[0, index]
        ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", color="white" if abs(matrix[i,j]) > .65 else "black", fontsize=8)
        ax.set_xticks(range(3), [arm.replace("_", "\n") for arm in analyse.ARMS], fontsize=8)
        ax.set_yticks(range(3), [arm.replace("_", "\n") for arm in analyse.ARMS], fontsize=8)
        ax.set_title(LABELS[source], loc="left")
        ax = axes[1, index]
        for arm in analyse.ARMS:
            record = diagnostics["sources"][source][arm]
            norms = record["gradient_norms"]
            scalar = record["gradient_calibration"]["multipliers"][arm]
            ax.plot(np.arange(1, len(norms)+1), norms, linewidth=.8, color=COLORS[arm],
                    label=f"{arm}: multiplier {scalar:.3g}, clipped {len(record['clipping_steps'])}/{len(norms)}")
        ax.axhline(1, color="#777777", linestyle="--", linewidth=.8)
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("L2 norm before clip(1); logarithmic scale")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=.15)
    save(fig, output, "03_gradient_directions_and_clipping")


def run(root, output=None):
    root = Path(root).resolve()
    rows, effects, diagnostics, hashes = load_verified(root)
    folder = root/"results"/analyse.STUDY
    output = Path(output).resolve() if output else folder/"figures"
    if output == folder or not output.is_relative_to(folder):
        raise ValueError("Save figures inside a separate objective result subdirectory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve previous complete and partial figure outputs")
    hashes[str(Path(__file__).resolve())] = analyse.sha(__file__)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    plot_effects(rows, effects, output)
    plot_trajectories(rows, diagnostics, output)
    plot_gradients(diagnostics, output)
    verify_hashes(hashes)
    files = {path.name: analyse.sha(path) for path in sorted(output.iterdir()) if path.suffix in (".png", ".pdf")}
    if len(files) != 6:
        raise AssertionError("All three figures must export PNG and PDF")
    manifest = {"completed": True, "input_hashes": hashes, "figure_hashes": files,
                "scope": "Verified development diagnostic; no model or future prediction archive is opened"}
    analyse.write_json(output/"plot_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output")
    args = parser.parse_args()
    print(run(args.root, args.output)["completed"])


if __name__ == "__main__":
    main()
