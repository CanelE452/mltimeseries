"""NEXT_METHOD_BRIEF.md, written only when the verdict is GAP_CANDIDATE_READY.

Section 34 asks for sixteen specific items and, in the same breath, forbids
treating any of the proposed methods as the answer. The generator therefore
assembles the measured parts from the artifacts and leaves the design candidates
as explicitly competing options, each with its own novelty risk.
"""

from __future__ import annotations

import json

import pandas as pd

from . import paths

# Design directions per descriptor, written before results. Each entry offers at
# least two conventional candidates plus one deliberately unusual one, so the
# brief cannot collapse into a single foregone conclusion.
DESIGN_SPACE = {
    "D1": {
        "component": "the horizon-side decoder and the way context is compressed before it",
        "candidates": [
            "A horizon-aware readout that conditions the decoder on the ratio of requested horizon to available context, rather than on the horizon alone.",
            "Multi-resolution context summarisation: keep a fine view of the recent past and a coarse view of the distant past, so long horizons are not forced through the same receptive field as short ones.",
        ],
        "creative": "Train the model to emit its own effective context length as an auxiliary output and use it to decide, per series, how much history to attend to.",
    },
    "D2": {
        "component": "patch or tokenisation stride and its relation to the sampling period",
        "candidates": [
            "Period-aligned patching, choosing the stride from the series' detected period instead of a fixed constant.",
            "A frequency embedding supplied as an explicit input so one backbone can specialise its normalisation per regime.",
        ],
        "creative": "Resample every series to a canonical rate at inference time and map the forecast back, so the backbone never sees an unfamiliar frequency.",
    },
    "D3": {
        "component": "cross-variate attention and how it is normalised as the variate count grows",
        "candidates": [
            "Sparse or top-k cross-variate attention, so a large panel does not dilute each series' own history.",
            "A gate that learns per series how much cross-variate information to admit, defaulting to channel independence.",
        ],
        "creative": "Predict the dependence structure first and condition the forecaster on it, rather than letting attention discover it implicitly at inference time.",
    },
    "D4": {
        "component": "the covariate encoder and where covariate information enters the backbone",
        "candidates": [
            "Encode known-future covariates as a separate stream fused at the horizon side, so covariate use cannot corrupt the target's own dynamics.",
            "Learn a per-task relevance weight for each covariate from the visible history and shrink irrelevant ones towards zero.",
        ],
        "creative": "Treat the covariate as a second forecasting problem and condition on its predicted uncertainty, instead of assuming the supplied future path is exact.",
    },
    "D5": {
        "component": "past-covariate conditioning",
        "candidates": [
            "A past-covariate encoder trained with dropout over covariates so the model degrades gracefully when they are absent.",
            "Residual conditioning: forecast the target alone, then correct with a covariate-driven term.",
        ],
        "creative": "Learn which past covariates lead the target and shift them by the learned lag before conditioning.",
    },
    "D6": {
        "component": "the output distribution head",
        "candidates": [
            "A zero-inflated head that places explicit mass at zero alongside the continuous quantiles.",
            "A two-stage formulation: predict occurrence, then magnitude conditional on occurrence.",
        ],
        "creative": "Let the head choose its own support at inference from the observed history, rather than fixing a continuous family in advance.",
    },
    "D7": {
        "component": "how gaps in the input are represented",
        "candidates": [
            "An explicit observed/missing mask channel instead of imputing before the model sees the series.",
            "Time-aware positional encoding so an irregular gap is represented as elapsed time rather than as a step.",
        ],
        "creative": "Give the model the imputation uncertainty as an input so it can widen its own intervals where the history was reconstructed.",
    },
    "D8": {
        "component": "in-context normalisation",
        "candidates": [
            "Recency-weighted normalisation statistics, so a level shift inside the context does not bias the scale.",
            "A learned test-time correction fitted on the most recent observed window before each origin.",
        ],
        "creative": "Predict the shift itself as an auxiliary target and let the forecast inherit its uncertainty.",
    },
    "D9": {
        "component": "how the seasonal component is represented",
        "candidates": [
            "Explicit decomposition into seasonal and remainder, forecast separately and recombined.",
            "A learned seasonality strength gate that falls back to a robust local model when the seasonal signal is weak.",
        ],
        "creative": "Let the model select its own period from a learned basis rather than receiving one from the task metadata.",
    },
}


def build() -> str | None:
    verdict = json.loads((paths.RESULTS / "verdict.json").read_text(encoding="utf-8"))
    if verdict.get("final_token") != "GAP_CANDIDATE_READY" or not verdict.get("top_candidate"):
        return None

    top = verdict["top_candidate"]
    candidate_id = top["id"]
    spec = json.loads((paths.RESULTS / "candidate_probe_spec.json").read_text(encoding="utf-8"))
    candidate = next(c for c in spec["candidates"] if c["candidate_id"] == candidate_id)
    headroom = pd.read_csv(paths.RESULTS / "headroom_table.csv")
    head = headroom[headroom.candidate_id == candidate_id].iloc[0]
    confirmation = pd.read_csv(paths.RESULTS / "confirmation_results.csv")
    confirm = confirmation[confirmation.candidate_id == candidate_id].iloc[0]
    ranking = pd.read_csv(paths.RESULTS / "candidate_ranking.csv")
    rank_row = ranking[ranking.candidate_id == candidate_id].iloc[0]
    literature = json.loads((paths.RESULTS / "literature_audit.json").read_text(encoding="utf-8"))
    entry = literature.get(candidate_id, {})
    results = pd.read_csv(paths.RESULTS / "benchmark_results.csv")
    design = DESIGN_SPACE[candidate["descriptor"]]

    affected = results[
        (results.task_uid.isin(candidate["affected_tasks"]))
        & (results.track == "U")
        & (results.status == "OK")
    ]

    lines = [
        f"# Next method brief — {candidate_id}",
        "",
        "This brief records what the evidence supports. It does not pick a method: the",
        "candidates below are competing options, and the numeric go/no-go at the end is what",
        "decides between them.",
        "",
        "## 1. Problem in one sentence",
        "",
        f"{candidate['observed_condition']}: {len(candidate['affected_families'])} architecture "
        f"families lose ground together on this condition, and a diagnostic oracle shows "
        f"{head.H_oracle_pct:.1f}% of that loss is recoverable while the strongest simple "
        f"deployable fix recovers only {100 * head.R_simple:.0f}% of it.",
        "",
        "## 2. Why it matters — benchmark evidence",
        "",
        f"- benchmark: fev-bench, evaluated by the `fev` library, task definition pinned to `{spec.get('track', 'U')}` track",
        f"- discovery condition gap: {candidate['severity']['median_condition_gap_pct']:.1f}%",
        f"- median regret against the best deployable estimator: {candidate['severity']['median_regret_pct']:.1f}%",
        f"- confirmation gap on held-out tasks: {confirm.median_gap_pct:.1f}%",
        "",
        "## 3. Which models failed",
        "",
        f"{candidate['affected_models']} spanning {candidate['affected_families']}.",
        "",
        "Per-model medians inside the condition (relative to SeasonalNaive):",
        "",
        json.dumps(candidate["severity"]["per_model_bucket_median_relative"], indent=2),
        "",
        "## 4. Under which task conditions",
        "",
        f"Descriptor {candidate['descriptor']}, bucket `{candidate['bucket']}`, "
        f"{candidate['n_affected_tasks']} discovery tasks:",
        "",
        *[f"- `{task}`" for task in candidate["affected_tasks"]],
        "",
        f"Reproduced on {confirm.n_confirmation_tasks} held-out tasks.",
        "",
        "## 5. Raw effect size",
        "",
        affected.pivot_table(
            index="task_uid", columns="model", values="relative_to_naive"
        ).round(3).to_markdown(),
        "",
        "## 6. Oracle headroom",
        "",
        f"- baseline (strongest single foundation model): {head.baseline_loss:.4f}",
        f"- diagnostic oracle (per-origin minimum across models, reads evaluation labels): {head.oracle_loss:.4f}",
        f"- **H_oracle = {head.H_oracle_pct:.1f}%**",
        "",
        "The oracle is not achievable. It bounds what perfect per-origin model choice would buy,",
        "and nothing more.",
        "",
        "## 7. Simple baseline residual",
        "",
        f"- best simple deployable fix: `{head.best_simple_fix}` at {head.simple_loss:.4f}",
        f"- it recovers R_simple = {head.R_simple:.2f} of the oracle headroom",
        f"- **H_residual = {head.H_residual_pct:.1f}%** remains",
        f"- confirmation residual: {confirm.H_residual_pct:.1f}%",
        "",
        "## 8. Closest 2025–2026 work",
        "",
        f"Novelty status: **{entry.get('novelty_status', 'UNCLEAR')}**",
        "",
        *[
            f"- {paper['title']} ({paper.get('year')}/{paper.get('venue', 'arXiv preprint')}) — {paper.get('relevance', '')}"
            for paper in entry.get("papers", [])
        ],
        "",
        f"Closest method: {entry.get('closest_method', 'not identified')}",
        f"Difference: {entry.get('difference_substantial_or_cosmetic', 'not assessed')}",
        "",
        "## 9. Which model component to change",
        "",
        design["component"],
        "",
        "## 10. Method candidates",
        "",
        *[f"{i}. {candidate_text}" for i, candidate_text in enumerate(design["candidates"], start=1)],
        "",
        "## 11. Creative candidate",
        "",
        design["creative"],
        "",
        "## 12. Largest novelty risk per candidate",
        "",
        *[
            f"- candidate {i}: the intervention may reduce to something the "
            f"{entry.get('novelty_status', 'UNCLEAR').lower().replace('_', ' ')} literature already "
            "does under another name; the first experiment must include the closest published "
            "method as an arm, not just the untouched backbone."
            for i in range(1, len(design["candidates"]) + 2)
        ],
        "",
        "## 13. Next minimum experiment",
        "",
        f"Take the {candidate['n_affected_tasks']} affected discovery tasks and the "
        f"{confirm.n_confirmation_tasks} confirmation tasks. Arms: the untouched strongest "
        "foundation model; the best simple fix from this study "
        f"(`{head.best_simple_fix}`); the closest published method; and one method candidate. "
        "Same windows, same metric, same information condition. Report the per-origin losses so "
        "the effect can be recomputed.",
        "",
        "## 14. Numeric go / no-go",
        "",
        "Continue past the first experiment only if the method candidate, on the affected tasks:",
        "",
        f"- beats the untouched foundation model by at least {max(3.0, head.H_residual_pct / 2):.1f}% relative on the native metric,",
        f"- beats the best simple fix (`{head.best_simple_fix}`) by at least 2% relative,",
        "- reproduces both margins on the confirmation tasks with the same sign,",
        "- and does not lose more than 1% relative on the tasks outside the condition.",
        "",
        "Otherwise stop: the condition is characterised, not solvable by this intervention.",
        "",
        "## 15. Expected table",
        "",
        "| arm | affected tasks (SQL / naive) | outside tasks | confirmation |",
        "|---|---|---|---|",
        f"| strongest foundation model | {head.baseline_loss:.4f} | — | — |",
        f"| best simple fix (`{head.best_simple_fix}`) | {head.simple_loss:.4f} | — | — |",
        "| closest published method | to measure | to measure | to measure |",
        "| method candidate | to measure | to measure | to measure |",
        f"| diagnostic oracle (not achievable) | {head.oracle_loss:.4f} | — | — |",
        "",
        "## 16. Compute estimate",
        "",
        f"Inference over {candidate['n_affected_tasks'] + int(confirm.n_confirmation_tasks)} tasks "
        "for four arms is the same order as this study's discovery sweep, which measured "
        f"{affected.inference_seconds.sum() / 3600:.2f} GPU-hours for its own arms on the affected "
        "tasks. An adapter-style intervention trained on the visible history adds a small "
        "per-task fit; a change that requires pretraining from scratch does not fit this budget "
        "and would drop the candidate's priority under Section 33's feasibility clause.",
        "",
        f"Candidate score: {rank_row.total_score}/33.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    text = build()
    if text is None:
        print("verdict is not GAP_CANDIDATE_READY; no brief written (Section 34)")
        return
    out = paths.RESULTS / "NEXT_METHOD_BRIEF.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
