"""Plot verified candidate scores and selection losses without reading predictions."""

import argparse
import csv
import json
from pathlib import Path

from . import analyse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CELLS = ("s12_bike", "s12_household", "s13_bike", "s13_household")
COLORS = {"F0": "#222222", "OFF_LORA": "#0072B2", "H_FULL": "#D55E00", "H_MLP": "#009E73"}
MARKERS = {"F0": "X", "OFF_LORA": "o", "H_FULL": "s", "H_MLP": "^"}
FOOTER = (
    "Exploratory reuse of two sources in two time blocks; one stored optimizer seed (12000), not four independent sources.\n"
    "Selections use original V only. C13 / E83 origins use H48 at a daily stride. QCAL uses C only and gives no coverage guarantee.\n"
    "CIs condition on saved checkpoints and fixed selectors. Hindsight oracle regret is descriptive and cannot be used for deployment."
)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_hashes(values):
    for path, expected in values.items():
        if analyse.sha(path) != expected:
            raise AssertionError(f"Verified plot input changed: {path}")


def load_verified(root):
    folder = Path(root) / "results" / analyse.STUDY
    path = folder / "verification.json"
    verification = analyse.read_json(path)
    if (verification.get("passed") is not True or verification.get("completed") is not True
            or verification.get("candidate_count") != 40 or verification.get("candidate_procedure_rows") != 80
            or verification.get("selected_procedure_rows") != 56 or verification.get("new_forecasts") != 28
            or verification.get("reused_forecasts") != 12 or verification.get("new_training_updates") != 0):
        raise AssertionError("All fixed candidates, forecasts and selection audits must complete before plotting")
    hashes = {str(folder / name): digest for name, digest in verification["output_hashes"].items()}
    hashes[str(path)] = analyse.sha(path)
    verify_hashes(hashes)
    candidates, selected = read_csv(folder / "candidate_results.csv"), read_csv(folder / "selected_results.csv")
    effects = analyse.read_json(folder / "effects.json")
    expected = {(cell, selector, procedure) for cell in CELLS for selector in analyse.SELECTORS for procedure in ("SORT", "QCAL")}
    if (len(candidates) != 80 or len({(r["candidate_id"], r["procedure"]) for r in candidates}) != 80
            or len(selected) != 56 or {(r["cell"], r["selector"], r["procedure"]) for r in selected} != expected):
        raise AssertionError("Candidate or selector plot grid is partial or duplicated")
    if (effects.get("completed") is not True or effects["primary_confidence_each"] != .9875
            or effects["primary_family_size"] != 4 or effects["bootstrap_seed"] != 2026090814
            or set(effects["cells"]) != set(CELLS)):
        raise AssertionError("Selection-loss inferential contract changed")
    for cell in CELLS:
        if any(sum(row["cell"] == cell and row["procedure"] == procedure for row in candidates) != 10 for procedure in ("SORT", "QCAL")):
            raise AssertionError("Every cell must show all ten candidates under both procedures")
        scores = {row["selector"]: float(row["score"]) for row in selected if row["cell"] == cell and row["procedure"] == "SORT"}
        point = (scores["LORA_V"]-scores["FIXED_LOW"])/scores["F0"]
        reported = effects["cells"][cell]["SORT"]["LORA_V_vs_FIXED_LOW"]["daily"]["7"]["value"]
        if not np.isclose(point, reported, rtol=1e-10, atol=1e-12):
            raise AssertionError("Primary plot effect disagrees with its own selected candidate scores")
    return candidates, selected, effects, hashes


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.axisbelow": True})


def footer(fig):
    fig.text(.05, .035, FOOTER, fontsize=8, color="#444444", va="bottom", linespacing=1.4)


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(Path(output) / f"{name}.{extension}", dpi=300)
    plt.close(fig)


def cell_label(cell):
    number, dataset = cell.split("_", 1)
    return f"Block {number.removeprefix('s')} · {'Bike sharing' if dataset == 'bike' else 'Household power'}"


def plot_candidates(candidates, selected, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.4), squeeze=False)
    fig.subplots_adjust(left=.08, right=.97, bottom=.235, top=.88, hspace=.4, wspace=.27)
    fig.suptitle("Validation and future rankings of all saved candidates", fontsize=15, y=.97, fontweight="bold")
    fig.text(.5, .93, "SORT scores divided by each phase's own F0 score · all 10 candidates in every cell · lower is better",
             ha="center", fontsize=10)
    for ax, cell in zip(axes.flat, CELLS):
        rows = [row for row in candidates if row["cell"] == cell and row["procedure"] == "SORT"]
        f0 = next(row for row in rows if row["method"] == "F0")
        f0v, f0e = float(f0["val_score"]), float(f0["score"])
        policies = {r["selector"]: r["candidate_id"] for r in selected if r["cell"] == cell and r["procedure"] == "SORT"}
        groups = {}
        for row in rows:
            groups.setdefault(row["method"], []).append(float(row["lr"]))
        for row in rows:
            method = row["method"]
            rank = sorted(groups[method]).index(float(row["lr"]))
            x, y = float(row["val_score"]) / f0v, float(row["score"]) / f0e
            ax.scatter(x, y, marker=MARKERS[method], s=35 + 18 * rank,
                       color=COLORS[method], alpha=.8, edgecolor="white", linewidth=.6, zorder=3)
            names = [label for rule, label in (("LORA_V", "V"), ("FIXED_LOW", "LOW")) if policies[rule] == row["candidate_id"]]
            if names:
                ax.annotate(" / ".join(names), (x, y), xytext=(7, 7 if names[0] == "V" else -12),
                            textcoords="offset points", fontsize=8, color="#333333")
        ax.axhline(1, color="#999999", linewidth=.9, linestyle="--")
        ax.axvline(1, color="#999999", linewidth=.9, linestyle="--")
        ax.set_xlabel("Validation score / validation F0")
        ax.set_ylabel("Future score / future F0")
        ax.set_title(cell_label(cell), loc="left")
        ax.grid(alpha=.15)
        ax.margins(.13)
    handles = [Line2D([], [], marker=MARKERS[m], color=COLORS[m], linestyle="none", label=m) for m in COLORS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .166), ncol=4, frameon=False)
    fig.text(.5, .14, "V: full-validation LoRA choice · LOW: fixed lowest LoRA LR · marker size increases with LR rank within family; identical points may overlap",
             ha="center", fontsize=8)
    footer(fig)
    save(fig, output, "01_all_candidates_validation_and_future")


def plot_primary(effects, output):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=.235, right=.97, bottom=.245, top=.82)
    fig.suptitle("Does validation choose a worse LoRA learning rate?", fontsize=15, y=.96, fontweight="bold")
    fig.text(.5, .91, "SORT: 100 × (score_LORA_V − score_FIXED_LOW) / score_F0 · positive values mean selection harm",
             ha="center", fontsize=10)
    bounds = [-1., 0., 1.]
    for index, cell in enumerate(CELLS):
        y = 3-index
        result = effects["cells"][cell]["SORT"]
        primary = result["LORA_V_vs_FIXED_LOW"]["daily"]["7"]
        if primary["confidence"] != .9875:
            raise AssertionError("All four primary intervals must retain 98.75% confidence")
        point, lower, upper = 100 * np.asarray([primary["value"], *primary["ci"]])
        ax.hlines(y, lower, upper, color="#0072B2", linewidth=2.2)
        ax.vlines([lower, upper], y-.055, y+.055, color="#0072B2", linewidth=1.5)
        ax.scatter(point, y, color="#0072B2", marker="D", s=42, zorder=3)
        recent = 100 * result["LORA_RECENT7_vs_FIXED_LOW"]["daily"]["7"]["value"]
        ax.scatter(recent, y+.17, color="#D55E00", marker="s", s=31, zorder=3)
        for parity, offset, marker in (("even", -.15, "o"), ("odd", -.27, "^")):
            value = 100 * result["LORA_V_vs_FIXED_LOW"]["nonoverlapping"][parity]["7"]["value"]
            ax.scatter(value, y+offset, facecolors="white", edgecolors="#0072B2", marker=marker, s=30)
            bounds.append(value)
        bounds.extend((point, lower, upper, recent))
    ax.axvline(0, color="#999999", linewidth=1)
    for threshold in (-1, 1):
        ax.axvline(threshold, color="#009E73", linestyle="--", linewidth=1.2)
    span = max(bounds)-min(bounds)
    ax.set_xlim(min(bounds)-.1*span, max(bounds)+.1*span)
    ax.set_ylim(-.6, 3.55)
    ax.set_yticks([3, 2, 1, 0], [cell_label(cell) for cell in CELLS])
    ax.set_xlabel("Selection loss (% of the matching F0 score)")
    ax.grid(axis="x", alpha=.15)
    handles = [Line2D([], [], color="#0072B2", marker="D", label="LORA_V mean + 98.75% CI"),
               Line2D([], [], color="#D55E00", marker="s", linestyle="none", label="Recent7 point (descriptive)"),
               Line2D([], [], color="#0072B2", marker="o", markerfacecolor="white", linestyle="none", label="Even origins point"),
               Line2D([], [], color="#0072B2", marker="^", markerfacecolor="white", linestyle="none", label="Odd origins point")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .115), ncol=2, frameon=False, fontsize=8)
    footer(fig)
    save(fig, output, "02_selection_loss_and_nonoverlap")


def plot_oracle_regret(selected, output):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.2), squeeze=False)
    fig.subplots_adjust(left=.22, right=.9, bottom=.24, top=.86, hspace=.42)
    fig.suptitle("How much does each fixed rule miss the hindsight best?", fontsize=15, y=.97, fontweight="bold")
    fig.text(.5, .925, "100 × (rule score − future minimum of all 10 candidates) / F0 · descriptive point regret; the oracle is unavailable at deployment",
             ha="center", fontsize=9)
    matrices = []
    for procedure in ("SORT", "QCAL"):
        lookup = {(r["cell"], r["selector"]): float(r["all_oracle_regret_over_f0"])*100 for r in selected if r["procedure"] == procedure}
        matrices.append(np.asarray([[lookup[(cell, selector)] for selector in analyse.SELECTORS] for cell in CELLS]))
    vmax = max(1., *(float(matrix.max()) for matrix in matrices))
    for row, (procedure, matrix) in enumerate(zip(("SORT", "QCAL"), matrices)):
        ax = axes[row, 0]
        shown = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        for i in range(4):
            for j in range(7):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i,j] > .6*vmax else "#222222", fontsize=8)
        ax.set_xticks(np.arange(7), [s.replace("_", "\n", 1) for s in analyse.SELECTORS])
        ax.set_yticks(np.arange(4), [cell_label(cell) for cell in CELLS])
        ax.set_title(f"{procedure} · {'primary objective' if procedure == 'SORT' else 'calibration interaction diagnostic'}", loc="left")
    cax = fig.add_axes([.92, .26, .017, .59])
    fig.colorbar(shown, cax=cax, label="Hindsight regret (% of F0)")
    fig.text(.5, .16, "Head-six and LoRA-three family minima are reported separately in oracle_regret.json. No oracle-selected confidence interval is computed.", ha="center", fontsize=8)
    footer(fig)
    save(fig, output, "03_hindsight_oracle_point_regret")


def run(root, output=None):
    root = Path(root).resolve()
    candidates, selected, effects, hashes = load_verified(root)
    folder = root / "results" / analyse.STUDY
    output = Path(output).resolve() if output else folder / "figures"
    if output == folder or not output.is_relative_to(folder):
        raise ValueError("Save figures only in a separate subdirectory of the verified diagnostic result")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve earlier complete or partial figures")
    hashes[str(Path(__file__).resolve())] = analyse.sha(__file__)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    plot_candidates(candidates, selected, output)
    plot_primary(effects, output)
    plot_oracle_regret(selected, output)
    verify_hashes(hashes)
    figures = {p.name: analyse.sha(p) for p in sorted(output.iterdir()) if p.suffix in (".png", ".pdf")}
    if len(figures) != 6:
        raise AssertionError("Three figures must each export PNG and PDF")
    manifest = {"completed": True, "input_hashes": hashes, "figure_hashes": figures,
                "prediction_archives_opened": False, "candidate_rows": 80, "selector_rows": 56,
                "primary_interval": "Four 98.75% paired temporal intervals, conditional on one stored optimizer seed",
                "oracle_regret": "Descriptive hindsight point estimate only", "dpi": 300,
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__}
    analyse.write_json(output / "plot_manifest.json", manifest)
    print(json.dumps({"completed": True, "output": str(output), "files": list(figures)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
