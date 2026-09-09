"""Render audited aggregate results without opening prediction or target archives."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STUDY = "peft_external_gap_v1"
DATASETS = ("bike", "household")
SOURCE_NAMES = {"bike": "Bike sharing · 5 inputs / 2 targets",
                "household": "Household power · 4 inputs / 2 targets"}
ROLES = ("F0", "H", "OFF_LORA", "RAW")
METHODS = ("F0", "H_MLP", "H_FULL", "OFF_LORA")
COLORS = {"SORT": "#0072B2", "QCAL": "#D55E00"}
FOOTER = (
    "Chronological seasonal/distribution-change screen; no isolated causal shift mechanism. "
    "L = 336 h, H = 48 h; daily origins overlap by 24 h.\n"
    "V_select: 13 origins; C_cal: 13 origins; evaluation: 84 days / 83 origins per source. "
    "QCAL uses C_cal only and gives no coverage guarantee.\n"
    "Temporal CIs condition on the fixed selections and three fitted optimizer seeds; "
    "they exclude selection/training uncertainty and do not represent independent source replications."
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise AssertionError(f"Plot input changed after verification: {path}")


def load_verified(root):
    folder = root / "results" / STUDY
    verification_path = folder / "verification.json"
    verification = read_json(verification_path)
    if (verification.get("passed") is not True or verification.get("completed") is not True
            or verification.get("fit_trial_count") != 28 or verification.get("forecast_count") != 14
            or verification.get("selected_procedure_rows") != 32):
        raise AssertionError("Plotting requires the complete passed production analysis")
    required = {"selected_results.csv", "all_results.csv", "effects.json", "diagnostics.json"}
    if not required.issubset(verification["output_hashes"]):
        raise AssertionError("Verification does not cover the required plot inputs")
    hashes = {str(folder / name): value for name, value in verification["output_hashes"].items()}
    hashes[str(verification_path)] = sha(verification_path)
    verify_hashes(hashes)
    rows = read_csv(folder / "selected_results.csv")
    trials = [row for row in read_csv(folder / "all_results.csv") if row["stage"] == "GPU_FIT_SORT_SELECTION"]
    effects, diagnostics = read_json(folder / "effects.json"), read_json(folder / "diagnostics.json")
    expected = {(dataset, role, seed, procedure) for dataset in DATASETS for role in ROLES
                for seed in ((12000,) if role == "F0" else ((-1,) if role == "RAW" else (12000, 12001, 12002)))
                for procedure in COLORS}
    if len(rows) != 32 or {(r["source"], r["role"], int(r["seed"]), r["procedure"]) for r in rows} != expected:
        raise AssertionError("Selected result grid must contain all 32 procedure/seed rows")
    if len(trials) != 28 or len({row["path"] for row in trials}) != 28:
        raise AssertionError("Development figure must retain all 28 GPU fits")
    if sum(row["selected_for_forecast"] == "True" for row in trials) != 14:
        raise AssertionError("Exactly fourteen fits must be retained for holdout forecasting")
    if (not effects.get("completed") or set(effects["sources"]) != set(DATASETS)
            or effects["primary_confidence_each"] != .975 or effects["primary_block_days"] != 7):
        raise AssertionError("Primary inferential contract differs from the declared screen")
    for row in rows:
        if not np.isfinite([float(row["score"]), float(row["coverage80"])]).all():
            raise AssertionError("Cannot render a nonfinite selected summary")
    for trial in trials:
        history = diagnostics[trial["path"]]["validation_history"]
        expected_steps = [0] if trial["method"] == "F0" else list(range(0, 201, 40))
        if [item["step"] for item in history] != expected_steps:
            raise AssertionError("A development trajectory is incomplete")
        if any(not np.isfinite(item["val_score"]) or item["val_score"] <= 0 for item in history):
            raise AssertionError("Development log-score plot requires positive finite scores")
    return rows, trials, effects, diagnostics, hashes


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "axes.axisbelow": True})


def footer(fig, extra=""):
    fig.text(.04, .032, FOOTER, fontsize=8, color="#444444", ha="left", va="bottom", linespacing=1.45)
    if extra:
        fig.text(.04, .114, extra, fontsize=8, color="#333333", ha="left", va="bottom")


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300)
    plt.close(fig)


def selected_values(rows, dataset, role, procedure, metric):
    selected = sorted((r for r in rows if (r["source"], r["role"], r["procedure"]) == (dataset, role, procedure)),
                      key=lambda r: int(r["seed"]))
    return np.asarray([float(row[metric]) for row in selected])


def plot_scores(rows, effects, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), squeeze=False)
    fig.subplots_adjust(left=.075, right=.97, bottom=.235, top=.87, hspace=.37, wspace=.25)
    fig.suptitle("Selected procedures on the future window", fontsize=15, y=.965, fontweight="bold")
    fig.text(.5, .925, "Dots: fixed optimizer seeds · diamonds: mean · bars: ±1 sample SD across seeds (not a confidence interval)",
             ha="center", fontsize=9)
    for column, dataset in enumerate(DATASETS):
        selected_head = effects["sources"][dataset]["selected_head_family"]
        labels = ["F0", f"H\n({selected_head})", "OFF_LORA", "RAW"]
        for row_index, metric in enumerate(("score", "coverage80")):
            ax = axes[row_index, column]
            for index, role in enumerate(ROLES):
                for procedure, offset, marker in (("SORT", -.16, "o"), ("QCAL", .16, "s")):
                    values = selected_values(rows, dataset, role, procedure, metric)
                    values = values * (100 if metric == "coverage80" else 1)
                    x = index + offset
                    mean = float(values.mean())
                    if len(values) > 1:
                        ax.errorbar(x, mean, yerr=float(values.std(ddof=1)), color=COLORS[procedure],
                                    fmt="none", capsize=4, elinewidth=1.3, zorder=2)
                    jitter = np.linspace(-.07, .07, len(values)) if len(values) > 1 else np.asarray([0.])
                    ax.scatter(x + jitter, values, s=28, marker=marker, facecolors="white",
                               edgecolors=COLORS[procedure], linewidths=1.3, zorder=3)
                    ax.scatter(x, mean, s=25, marker="D", color=COLORS[procedure], zorder=4)
            ax.set_xticks(np.arange(4), labels)
            ax.set_xlim(-.5, 3.5)
            ax.grid(axis="y", alpha=.2)
            ax.set_title(f"{'ABCD'[row_index * 2 + column]}  {SOURCE_NAMES[dataset]}", loc="left", pad=10)
            if metric == "score":
                ax.set_ylabel("Scaled 2-pinball score ↓")
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(80, color="#555555", linestyle="--", linewidth=1)
                ax.text(3.43, 82, "Nominal 80%", ha="right", color="#555555", fontsize=8)
                ax.set_ylim(0, 105)
                ax.set_ylabel("80% interval coverage (%)")
    handles = [Line2D([], [], marker=marker, markerfacecolor="white", markeredgecolor=COLORS[p],
                      color=COLORS[p], linestyle="none", label=p)
               for p, marker in (("SORT", "o"), ("QCAL", "s"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .153), ncol=2, frameon=False)
    footer(fig, "H and OFF_LORA: n = 3 fixed seeds; F0 and RAW: one deterministic procedure each, no seed SD. Equal weights for the two targets.")
    save(fig, output, "01_scores_and_coverage")


def plot_effects(rows, effects, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), squeeze=False)
    fig.subplots_adjust(left=.135, right=.97, bottom=.285, top=.77, wspace=.38)
    fig.suptitle("Standard LoRA versus the selected output head", fontsize=15, y=.955, fontweight="bold")
    fig.text(.5, .905, "QCAL: 100 × (score_H − score_OFF_LORA) / score_F0   ·   positive values favor OFF_LORA", ha="center", fontsize=10)
    for column, dataset in enumerate(DATASETS):
        ax, source = axes[0, column], effects["sources"][dataset]
        h = selected_values(rows, dataset, "H", "QCAL", "score")
        lora = selected_values(rows, dataset, "OFF_LORA", "QCAL", "score")
        f0 = selected_values(rows, dataset, "F0", "QCAL", "score")[0]
        if not np.isclose(((h - lora) / f0).mean(), source["primary"]["value"], rtol=1e-10, atol=1e-12):
            raise AssertionError("Effect point estimate disagrees with verified selected scores")
        bounds = [-1., 1., 0.]
        for block, y in ((3, 2), (7, 1), (14, 0)):
            value = source["block_sensitivity"][str(block)]
            if value["confidence"] != .975 or value["block_days"] != block:
                raise AssertionError("QCAL block sensitivity must retain 97.5% intervals")
            point, lower, upper = 100 * np.asarray([value["value"], *value["ci"]])
            color, width = ("#0072B2", 2.4) if block == 7 else ("#777777", 1.5)
            ax.hlines(y, lower, upper, color=color, linewidth=width, zorder=2)
            ax.vlines([lower, upper], y - .06, y + .06, color=color, linewidth=width)
            ax.scatter(point, y, marker="D" if block == 7 else "o", s=38, color=color, zorder=4)
            if block == 7:
                seed_values = 100 * np.asarray(value["seed_values"])
                ax.scatter(seed_values, np.full(3, y + .17), marker="o", s=24, facecolor="white", edgecolor=color, zorder=3)
                bounds.extend(seed_values.tolist())
            bounds.extend((lower, upper, point))
        ax.axvline(0, color="#999999", linewidth=1)
        ax.axvline(1, color="#009E73", linestyle="--", linewidth=1.5)
        ax.axvline(-1, color="#AAAAAA", linestyle=":", linewidth=.9)
        span = max(bounds) - min(bounds)
        ax.set_xlim(min(bounds) - .12 * span, max(bounds) + .12 * span)
        ax.set_ylim(-.45, 2.6)
        ax.set_yticks([2, 1, 0], ["3-day sensitivity", "7-day primary", "14-day sensitivity"])
        ax.get_yticklabels()[1].set_fontweight("bold")
        ax.set_xlabel("Paired improvement (% of F0 score)")
        ax.set_title(f"{'AB'[column]}  {SOURCE_NAMES[dataset]}", loc="left", pad=38)
        decision = source["primary"]["decision"].replace("conditional_practical_", "").replace("_", " ")
        veto = "yes" if source["raw_veto"]["applies"] else "no"
        ax.text(0, 1.025, f"Primary: {decision}  |  RAW point-score veto: {veto}", transform=ax.transAxes, fontsize=8.5)
        ax.grid(axis="x", alpha=.15)
    handles = [Line2D([], [], color="#0072B2", marker="D", label="Mean + temporal 97.5% CI"),
               Line2D([], [], color="#0072B2", marker="o", markerfacecolor="white", linestyle="none", label="Three fixed-seed effects"),
               Line2D([], [], color="#009E73", linestyle="--", label="1% of F0 entry threshold")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .185), ncol=3, frameon=False)
    footer(fig, "97.5% CI per source; 7-day blocks define the two-source primary family. 3/14-day intervals are descriptive sensitivity, not extra entry tests.")
    save(fig, output, "02_primary_effect_and_block_sensitivity")


def plot_trajectories(trials, diagnostics, output):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.8), squeeze=False)
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.865, hspace=.47, wspace=.28)
    fig.suptitle("Every development fit and its selected checkpoint", fontsize=16, y=.965, fontweight="bold")
    fig.text(.5, .925, "28 GPU fits shown: 26 adaptations + 2 F0 jobs · 14 retained fits · solid/dashed/dotted = seeds 12000/12001/12002", ha="center", fontsize=10)
    palette = ("#0072B2", "#D55E00", "#009E73")
    styles = {12000: "-", 12001: "--", 12002: ":"}
    for row_index, dataset in enumerate(DATASETS):
        for column, method in enumerate(METHODS):
            ax = axes[row_index, column]
            ax.set_yscale("log")
            selected = sorted((r for r in trials if (r["source"], r["method"]) == (dataset, method)),
                              key=lambda r: (float(r["lr_or_lambda"]), int(r["seed"])))
            rates = sorted({float(r["lr_or_lambda"]) for r in selected})
            colors = {rate: palette[i] for i, rate in enumerate(rates)}
            kept, panel_scores = 0, []
            for trial in selected:
                detail = diagnostics[trial["path"]]
                retained = trial["selected_for_forecast"] == "True"
                if retained != detail["selected_for_forecast"]:
                    raise AssertionError("Diagnostics and all-results selection flags disagree")
                kept += int(retained)
                history = detail["validation_history"]
                x, y = [p["step"] for p in history], [p["val_score"] for p in history]
                panel_scores.extend(y)
                seed, rate = int(trial["seed"]), float(trial["lr_or_lambda"])
                color = colors[rate]
                label = ("F0 · zero updates" if method == "F0" else f"LR {rate:.0e} · s{seed - 12000}")
                label += " *" if retained else ""
                ax.plot(x, y, color=color, linestyle=styles[seed], linewidth=2.2 if retained else 1.1,
                        alpha=1 if retained else .55, marker="." if method == "F0" else None, label=label)
                if method == "F0":
                    ax.axhline(y[0], color=color, alpha=.3, linestyle=":", linewidth=1)
                if retained:
                    best = next(p for p in history if p["step"] == detail["best_step"])
                    ax.scatter(best["step"], best["val_score"], marker="*", s=95, color=color,
                               edgecolor="white", linewidth=.6, zorder=5, clip_on=True)
            minimum, maximum = min(panel_scores), max(panel_scores)
            if np.isclose(minimum, maximum, rtol=1e-12, atol=0):
                # A constant log-series can acquire a one-ULP data range from mixed artists.
                ax.set_ylim(minimum / 1.05, maximum * 1.05)
            ax.set_xlim(-6, 206)
            ax.set_xticks([0, 40, 80, 120, 160, 200])
            ax.grid(axis="y", which="both", alpha=.15)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel("Validation score (log scale)")
            title = f"{'ABCDEFGH'[row_index * 4 + column]}  {dataset.capitalize()} · {method}\n{len(selected)} fits / {kept} retained"
            ax.set_title(title, loc="left", fontsize=10, pad=10)
            ax.legend(loc="best", fontsize=7, frameon=False, handlelength=2.1)
    footer(fig, "* / star: retained fit / selected checkpoint; faint lines: unused candidates. Validation uses SORT only. H searched 6 candidates; OFF_LORA searched 3.")
    save(fig, output, "03_all_development_trajectories")


def run(root, output=None):
    root = Path(root).resolve()
    rows, trials, effects, diagnostics, hashes = load_verified(root)
    folder = root / "results" / STUDY
    output = Path(output).resolve() if output else folder / "figures"
    if output == folder or not output.is_relative_to(folder):
        raise ValueError("Figure output must be a separate subdirectory of the verified result folder")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing complete or partial figures; choose a fresh figure directory")
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    plot_scores(rows, effects, output)
    plot_effects(rows, effects, output)
    plot_trajectories(trials, diagnostics, output)
    verify_hashes(hashes)
    figures = {p.name: sha(p) for p in sorted(output.iterdir()) if p.suffix in (".png", ".pdf")}
    if len(figures) != 6:
        raise AssertionError("All three figures require both PNG and PDF exports")
    manifest = {"completed": True, "verified_analysis_required": True, "prediction_archives_opened": False,
                "input_hashes": hashes, "figure_hashes": figures, "selected_rows": 32, "fit_trajectories": 28,
                "seed_error_bars": "sample SD across three fixed optimizer seeds; absent for single F0/RAW",
                "effect_intervals": "verified paired temporal 97.5% CI; 7-day primary, 3/14-day descriptive sensitivity",
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "dpi": 300}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
