"""Plot the verified replication and separate block-12/block-13 estimates."""

import argparse
import json
from pathlib import Path

from . import analyse


STUDY = analyse.STUDY
_core = analyse.load_reference("_shared_plot", "plot.py", analyse.REFERENCE_PLOT_SHA256)
_core.STUDY = STUDY
_core.FOOTER = (
    "Temporal replication on the next 190-day block of the same two sources; no isolated causal mechanism or new-source replication.\n"
    "L = 336 h, H = 48 h; daily origins overlap by 24 h. V_select / C_cal: 13 origins each; evaluation: 83 origins. QCAL has no coverage guarantee.\n"
    "Each temporal CI conditions on that block's frozen selections and three fitted seeds; it excludes selection/training uncertainty. Blocks 12/13 are not pooled."
)


def load_comparison(root, hashes):
    path = Path(root) / "results" / STUDY / "temporal_comparison.json"
    if str(path) not in hashes:
        raise AssertionError("The separate temporal comparison is not covered by verification")
    comparison = _core.read_json(path)
    if (comparison.get("completed") is not True or set(comparison["blocks"]) != {"12", "13"}
            or any(comparison.get(key) is not False for key in ("pooled_estimate_computed",
                    "pooled_confidence_interval_computed", "cross_block_difference_test_computed"))):
        raise AssertionError("Plot only the completed, separate temporal estimates")
    _core.verify_hashes(comparison["input_hashes"])
    hashes.update(comparison["input_hashes"])
    return comparison


def plot_temporal_comparison(comparison, output):
    plt, np = _core.plt, _core.np
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), squeeze=False)
    fig.subplots_adjust(left=.12, right=.97, bottom=.31, top=.78, wspace=.36)
    fig.suptitle("Two time blocks, two separate estimates", fontsize=15, y=.955, fontweight="bold")
    fig.text(.5, .901, "QCAL: 100 × (score_H − score_OFF_LORA) / score_F0 · positive values favor OFF_LORA",
             ha="center", fontsize=10)
    for column, source in enumerate(_core.DATASETS):
        ax, bounds, labels = axes[0, column], [-1., 0., 1.], []
        dates = comparison["calendar_audits"][source]
        for number, y, color, date_key in ((12, 1, "#777777", "previous_boundaries"),
                                           (13, 0, "#0072B2", "replication_boundaries")):
            result = comparison["blocks"][str(number)]["sources"][source]
            value = result["primary"]
            if value["confidence"] != .975 or value["block_days"] != 7:
                raise AssertionError("The temporal comparison must use each block's 97.5% primary interval")
            point, lower, upper = 100 * np.asarray([value["value"], *value["ci"]])
            ax.hlines(y, lower, upper, color=color, linewidth=2.2)
            ax.vlines([lower, upper], y - .06, y + .06, color=color, linewidth=1.7)
            ax.scatter(point, y, marker="D", s=42, color=color, zorder=3)
            seed_values = 100 * np.asarray(value["seed_values"])
            ax.scatter(seed_values, np.full(3, y + .17), s=25, facecolors="white", edgecolors=color)
            bounds.extend([point, lower, upper, *seed_values])
            start, end = dates[date_key]["eval"]
            labels.append(f"Block {number}\n{start[:10]} to\n{end[:10]} (exclusive)")
            ax.annotate(f"{point:+.2f}% [{lower:+.2f}, {upper:+.2f}]", (point, y),
                        xytext=(0, -22), textcoords="offset points", ha="center", fontsize=8, color=color)
        span = max(bounds) - min(bounds)
        ax.set_xlim(min(bounds) - .19 * span, max(bounds) + .19 * span)
        ax.set_ylim(-.5, 1.55)
        ax.set_yticks([1, 0], labels)
        ax.axvline(0, color="#999999", linewidth=1)
        ax.axvline(1, color="#009E73", linewidth=1.5, linestyle="--")
        ax.axvline(-1, color="#AAAAAA", linewidth=.9, linestyle=":")
        ax.grid(axis="x", alpha=.15)
        ax.set_title(f"{'AB'[column]}  {_core.SOURCE_NAMES[source]}", loc="left", pad=18)
        ax.set_xlabel("Paired improvement (% of that block's F0 score)")
    fig.text(.5, .23, "Diamonds / bars: mean / 97.5% temporal CI from 7-day blocks · open dots: fixed-seed effects · green line: 1% threshold",
             ha="center", fontsize=9)
    _core.footer(fig, "No pooled estimate, combined confidence interval, or cross-block difference test. A later result does not revise the earlier block's decision.")
    _core.save(fig, output, "04_separate_temporal_replication")


def run(root, output=None):
    root = Path(root).resolve()
    rows, trials, effects, diagnostics, hashes = _core.load_verified(root)
    verification = _core.read_json(root / "results" / STUDY / "verification.json")
    if (verification.get("wrapped_completed") is not True or verification.get("bootstrap_seed") != analyse.BOOTSTRAP_SEED
            or verification.get("comparison_without_pooling") is not True):
        raise AssertionError("Temporal wrapper and bootstrap verification must complete before plotting")
    comparison = load_comparison(root, hashes)
    folder = root / "results" / STUDY
    output = Path(output).resolve() if output else folder / "figures"
    if output == folder or not output.is_relative_to(folder):
        raise ValueError("Figure output must be a separate subdirectory of this verified result folder")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve previous complete or partial figures; choose a fresh directory")
    for path in (Path(__file__).resolve(), Path(_core.__file__).resolve(), Path(analyse.__file__).resolve()):
        hashes[str(path)] = _core.sha(path)
    output.mkdir(parents=True, exist_ok=True)
    _core.configure()
    _core.plot_scores(rows, effects, output)
    _core.plot_effects(rows, effects, output)
    _core.plot_trajectories(trials, diagnostics, output)
    plot_temporal_comparison(comparison, output)
    _core.verify_hashes(hashes)
    figures = {p.name: _core.sha(p) for p in sorted(output.iterdir()) if p.suffix in (".png", ".pdf")}
    if len(figures) != 8:
        raise AssertionError("All four figures require PNG and PDF exports")
    manifest = {"completed": True, "verified_analysis_required": True, "prediction_archives_opened": False,
                "input_hashes": hashes, "figure_hashes": figures, "selected_rows": 32, "fit_trajectories": 28,
                "temporal_comparison": "Two separate estimates on the same sources; no pooled estimate or new-source inference",
                "seed_error_bars": "Sample SD across three fixed optimizer seeds; absent for single F0/RAW",
                "effect_intervals": "Paired temporal 97.5% CI; 7-day primary, 3/14-day descriptive sensitivity",
                "matplotlib_version": _core.matplotlib.__version__, "numpy_version": _core.np.__version__, "dpi": 300}
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
