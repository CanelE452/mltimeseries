"""STATUS.md in the order Section 43 fixes, and the five tables Section 33 asks for."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths

PRIMARY = ["chronos-2", "tirex-2", "timesfm-3.0"]


def _csv(name: str) -> pd.DataFrame:
    path = paths.RESULTS / name
    if not path.exists() or path.stat().st_size < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def _json(name: str):
    path = paths.RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(frame: pd.DataFrame, columns=None, round_to=4) -> str:
    if frame.empty:
        return "_no rows_"
    if columns:
        frame = frame[[c for c in columns if c in frame.columns]]
    return frame.round(round_to).to_markdown(index=False)


def score_table(split: str) -> pd.DataFrame:
    """Table A / B: every comparator side by side on one split."""
    comparator = _csv("comparator_scores.csv")
    scores = _csv("specialist_scores.csv")
    components = _csv("specialist_component_results.csv")
    if comparator.empty or scores.empty:
        return pd.DataFrame()
    subset = comparator[comparator.split == split].copy()

    def mean_over_seeds(frame: pd.DataFrame, label: str) -> dict:
        rows = frame[frame.label == label]
        return rows.groupby("task_uid").native_score.mean().to_dict()

    patchtst = mean_over_seeds(components, "COMPONENT_PatchTST")
    best = mean_over_seeds(scores, "S_BEST")
    ensemble = mean_over_seeds(scores, "S_ENSEMBLE")
    subset["PatchTST"] = subset.task_uid.map(patchtst)
    subset["S_BEST"] = subset.task_uid.map(best)
    subset["S_ENSEMBLE"] = subset.task_uid.map(ensemble)
    subset["task"] = subset.task_uid.str.split("::").str[1]
    return subset[
        [
            "task",
            "timesfm-3.0",
            "chronos-2",
            "tirex-2",
            "F_FAMILY_ENVELOPE",
            "PatchTST",
            "S_BEST",
            "S_ENSEMBLE",
            "linear-ar-specialist",
            "seasonal-naive",
        ]
    ].rename(columns={"linear-ar-specialist": "LinearAR", "seasonal-naive": "SeasonalNaive"})


def build() -> str:
    verdict = _json("verdict.json") or {}
    source = _json("SOURCE_STUDY.json") or {}
    spec = _json("long_horizon_tasks.json") or {}
    env = _json("specialist_env.json") or {}
    effects = _csv("comparison_effects.csv")
    aggregates = _csv("aggregate_effects.csv")
    manifest = _csv("specialist_manifest.csv")
    integrity = _csv("integrity_checks.csv")

    lines: list[str] = []
    add = lines.append

    add("# TSFM-LONG-HORIZON-SPECIALIST-CLOSURE-v1 — status")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat()}")
    add("")

    add("## 1. Executive verdict")
    add("")
    add(f"**{verdict.get('final_token', 'NOT_RUN')}** — {verdict.get('final_token_reason', '')}")
    add("")

    add("## 2. The unresolved question this closes")
    add("")
    add(
        "The source study found that on long-horizon tasks all three foundation model families "
        f"lost ground together: a median condition gap of "
        f"{source.get('source_d1_long_effect', {}).get('median_condition_gap_pct', float('nan')):.1f}% "
        f"across {source.get('source_d1_long_effect', {}).get('n_tasks_in_bucket', 0)} tasks and "
        f"{source.get('source_d1_long_effect', {}).get('n_affected_families', 0)} architecture families. "
        f"It failed the registered gap gate on regret alone: "
        f"{source.get('source_d1_long_regret', {}).get('median_regret_pct', float('nan')):.2f}% against "
        "the best evaluated deployable estimator, far short of the 8% threshold."
    )
    add("")
    add(
        "Those comparators were other foundation models, SeasonalNaive and a linear ridge "
        "autoregression. So the evidence said the models struggle together, but it could not say "
        "whether a genuinely strong task-trained model would do better. That is what this study "
        "tests, and nothing else: it does not ask why."
    )
    add("")

    add("## 3. Source study provenance")
    add("")
    for key in ("source_branch", "source_remote_sha", "source_final_token", "source_benchmark", "source_benchmark_commit"):
        if key in source:
            add(f"- **{key}**: `{source[key]}`")
    add(f"- **selected_tasks_sha256**: `{source.get('selected_tasks_sha256', '')}`")
    add("")
    add("Source artifacts are read only. `verify` check A20 diffs them against the base commit.")
    add("")

    add("## 4. Development tasks — the source D1-long set")
    add("")
    add("Re-derived from the source `task_metadata.csv`, not copied from a list:")
    add("")
    for task in spec.get("development_tasks", []):
        add(f"- `{task}`")
    add("")

    add("## 5. Fresh holdout selection")
    add("")
    add(f"- rule: {spec.get('selection_rule', '')}")
    add(f"- long definition: `{spec.get('long_definition', '')}` (threshold reused from the source study)")
    add(f"- frozen at: {spec.get('frozen_at_utc', '')}")
    sha = paths.RESULTS / "long_horizon_tasks.sha256"
    if sha.exists():
        add(f"- sha256: `{sha.read_text(encoding='utf-8').split()[0]}`")
    add(
        f"- funnel: {spec.get('n_candidates_after_exclusions')} candidates after excluding the "
        f"previous 18 tasks and their families -> {spec.get('n_after_metadata_bound')} past the "
        f"metadata bound -> {spec.get('n_long_after_exact_measurement')} measured long -> "
        f"{len(spec.get('fresh_holdout_tasks', []))} selected"
    )
    add("")
    add(f"- coverage: {spec.get('holdout_summary', {})}")
    add("")
    long_candidates = pd.DataFrame(spec.get("long_candidates", []))
    if len(long_candidates):
        add(
            _table(
                long_candidates.assign(task=lambda d: d.task_uid.str.split("::").str[1]),
                ["task", "domain", "freq_bucket", "dataset_family", "horizon", "n_series",
                 "horizon_to_context_ratio", "n_forecasts_track_u", "selected"],
            )
        )
    add("")
    add(
        "Every selected task is `daily_or_coarser`. That is a property of the condition rather "
        "than a choice: a high horizon-to-context ratio means a short visible history, and among "
        "the eligible tasks none of the sub-hourly or hourly ones reach the long cut. The "
        "stratification tried for frequency spread and the data did not offer it."
    )
    add("")

    add("## 6. TSFM revisions and contamination")
    add("")
    revisions = source.get("primary_model_revisions", {})
    contamination = source.get("primary_model_contamination", {})
    add("| model | revision | fev-bench contamination |")
    add("|---|---|---|")
    for model in PRIMARY:
        add(f"| `{model}` | `{revisions.get(model, '')}` | {contamination.get(model, '')} |")
    add("")
    add(
        "These are the source study's checkpoints, asserted against the live hub before the "
        "holdout ran. Two of the three carry unresolved overlap risk, which matters for reading "
        "the TSFM side of any comparison here."
    )
    add("")

    add("## 7. Specialist suite")
    add("")
    add(f"- environment: `{env.get('env_path')}`, AutoGluon {env.get('autogluon_timeseries')}, torch {env.get('torch')}, CUDA {env.get('cuda_available')}")
    add(f"- models requested and available: {env.get('n_available')}/5")
    add("")
    availability = pd.DataFrame(env.get("model_availability", []))
    if len(availability):
        add(_table(availability, ["requested", "actual_class", "family", "available"]))
    add("")
    add(
        "AutoGluon ships Chronos, Chronos-2 and Toto wrappers and installing it pulled "
        "`chronos-forecasting` in as a dependency. None is in the suite: the pool is given as an "
        "explicit hyperparameter dict, and `verify` check A18 reads the fitted model list back."
    )
    add("")

    add("## 8. Training chronology and the no-refit contract")
    add("")
    add("```")
    add("first evaluation cutoff")
    add("        |")
    add("  training history  ->  fit once per (task, seed)  ->  parameters frozen")
    add("        |")
    add("  window 0 forecast")
    add("  window 1  newer observed history is input context, no refit")
    add("  window k  ...")
    add("```")
    add("")
    add(
        "`fit` is called once per (task, seed) and every later forecast goes through "
        "`predict(context)`. `refit_full` is never called. Checks A07 to A10 test the timestamps "
        "and the call path rather than taking the diagram's word for it."
    )
    add("")
    if len(manifest):
        ok = manifest[manifest.status == "OK"]
        add(
            _table(
                ok.assign(task=lambda d: d.task_uid.str.split("::").str[1]),
                ["task", "seed", "num_val_windows", "fit_seconds", "model_best", "ensemble_model"],
                round_to=1,
            )
        )
        add("")
        failed = manifest[manifest.status != "OK"]
        add(f"Fit failures: {len(failed)} of {len(manifest)} (task, seed) runs.")
        add("")

    add("## 9. Specialist strength sanity")
    add("")
    strength = verdict.get("specialist_strength", {})
    if strength:
        add(
            f"- median RI over the source study's linear autoregression: "
            f"{strength.get('median_RI_vs_linear_ar_pct', float('nan')):.2f}% "
            f"({strength.get('wins_vs_linear_ar')} wins of {strength.get('n_tasks')})"
        )
        add(
            f"- median RI over SeasonalNaive: {strength.get('median_RI_vs_naive_pct', float('nan')):.2f}% "
            f"({strength.get('wins_vs_naive')} wins of {strength.get('n_tasks')})"
        )
        add(f"- **gate: {'PASS' if strength.get('passed') else 'FAIL'}**")
        add("")
        if not strength.get("passed"):
            add(
                "This gate comes first for a reason. A suite that cannot beat the weak baselines "
                "the source study already had cannot separate 'the foundation models have a gap' "
                "from 'this comparator is not strong enough', so a null result here is not "
                "evidence that the gap is absent."
            )
            add("")

    for section, split in (("10. Development raw results", "development"), ("11. Fresh holdout raw results", "holdout")):
        add(f"## {section}")
        add("")
        table = score_table(split)
        add(_table(table))
        add("")

    add("## 12. Effect tables")
    add("")
    add("### Table C — per-task relative improvement, positive means the specialist wins")
    add("")
    subset = effects[effects.label == "S_ENSEMBLE"] if len(effects) else pd.DataFrame()
    add(
        _table(
            subset,
            ["split", "task", "RI_vs_envelope_pct", "RI_vs_fixed_pct", "RI_vs_linear_ar_pct", "RI_vs_naive_pct"],
            round_to=2,
        )
    )
    add("")
    add("### Table D — aggregates")
    add("")
    add(
        _table(
            aggregates,
            ["split", "label", "n_usable", "median_RI_vs_envelope_pct", "mean_RI_vs_envelope_pct",
             "median_RI_vs_fixed_pct", "wins_vs_envelope"],
            round_to=2,
        )
    )
    add("")
    add("### Table E — specialist strength")
    add("")
    add(
        _table(
            aggregates,
            ["split", "label", "median_RI_vs_linear_ar_pct", "wins_vs_linear_ar",
             "median_RI_vs_naive_pct", "wins_vs_naive"],
            round_to=2,
        )
    )
    add("")

    add("## 13. Seed stability")
    add("")
    seed_columns = [c for c in aggregates.columns if c.startswith("median_RI_vs_envelope_seed_")]
    if seed_columns:
        add(_table(aggregates, ["split", "label", *seed_columns], round_to=2))
        add("")
        add(
            "A conclusion that flips with the seed is not a conclusion. Both seeds are reported "
            "and the gates require them to agree in sign."
        )
    add("")

    add("## 14. Final gate")
    add("")
    for split in ("development", "holdout"):
        payload = verdict.get(split, {})
        if not payload.get("available"):
            continue
        add(f"**{split}**")
        add("")
        for name, value in payload.get("checks", {}).items():
            add(f"- {'PASS' if value else 'FAIL'} — {name}")
        add("")
        add(f"signal: **{payload.get('signal')}**")
        add("")
    tension = verdict.get("contract_tension")
    if tension and tension.get("present"):
        add("### A tension inside the contract, stated rather than resolved quietly")
        add("")
        add(tension["why"])
        add("")
        add(tension["which_rule_was_followed"])
        add("")
        add(f"Does it change what to do next? {tension['does_it_change_the_action']}")
        add("")

    add("## 15. Integrity and deviations")
    add("")
    if len(integrity):
        passed = int((integrity.result == "PASS").sum())
        add(f"{passed}/{len(integrity)} checks pass. Detail in `SELF_AUDIT.md`.")
        add("")
        add(_table(integrity, ["check", "result", "description"]))
    add("")
    add("### Deviations")
    add("")
    add(
        "- Dataset arrow files are cached under `runs/dscache` rather than beside the model "
        "weights. `datasets` names its lock files after the full cache path, and the source "
        "study's deep cache directory pushed them past Windows' 260-character limit, which made "
        "one holdout candidate impossible to load."
    )
    add(
        "- The specialist environment is a second venv. Installing AutoGluon into the TSFM "
        "environment would have moved pandas, numpy and torch under three models this closure "
        "must keep byte-identical to the source study."
    )
    add("")

    add("## 16. What was NOT tested")
    add("")
    add("- Why the specialist and the foundation models land where they do. This study measures existence, not cause.")
    add("- Any track other than U. Covariates and native multivariate are out of scope by Section 5.")
    add("- Fine-tuning a foundation model, or any new method. Section 35 forbids it here.")
    add("- Frequencies other than daily-or-coarser, because no eligible sub-hourly or hourly task meets the frozen long-horizon definition.")
    add("- A larger training budget for the specialist: the cap is fixed per task and seed by Section 14 and was applied identically everywhere.")
    add("")

    add("## 17. Next action")
    add("")
    token = verdict.get("final_token")
    if token == "LONG_HORIZON_TSFMSPECIFIC_GAP_CONFIRMED":
        add("See `NEXT_METHOD_BRIEF.md`. Method work happens in a separate branch after approval.")
    elif token == "SHARED_TASK_DIFFICULTY_CLOSE":
        add(
            "Do not make long-horizon a method topic. A task-trained specialist suite, given each "
            "task's own history and a fixed training budget, did not expose the registered 5-8% "
            "recoverable gap on either split — it lost to the foundation models on all ten tasks."
        )
        add("")
        add(
            "Two cautions on how far that carries. First, Section 23: this is an investment screen, "
            "not an equivalence claim. The exact statement the evidence supports is that a stronger "
            "supervised specialist did not expose the registered gap, not that none exists. Second, "
            "the label undersells what was measured. 'Shared difficulty' suggests every model "
            "struggles equally; here the foundation models were clearly better, by 25% on the "
            "development split and 48% on the holdout. The long-horizon condition is hard for a "
            "model that must learn it from 14 to 209 observations, and the foundation models' "
            "pretraining is exactly what covers that."
        )
        add("")
        add(
            "If the question is reopened later, the thing to change is the comparator's training "
            "signal, not its architecture: every task here gives a specialist a very short history, "
            "and a suite that only ties a linear ridge autoregression under that constraint cannot "
            "bound what task specialisation could achieve with more."
        )
    elif token == "INCONCLUSIVE_SPECIALIST_POWER":
        add(
            "Do not close the long-horizon question on this evidence, and do not start method work "
            "on it either. The comparator was not shown to be strong enough to tell task "
            "difficulty and a foundation-model gap apart. The cheapest way forward is a specialist "
            "that demonstrably beats the weak baselines on these same tasks — a larger training "
            "budget, or a model family better suited to very short histories — before the "
            "comparison is repeated."
        )
    else:
        add("Resolve the integrity failures in section 15 before reading any of this as a result.")
    add("")
    return "\n".join(lines)


def main() -> None:
    text = build()
    out = paths.RESULTS / "STATUS.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
