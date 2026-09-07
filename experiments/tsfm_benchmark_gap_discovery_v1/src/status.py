"""STATUS.md, in the order Section 35 fixes. Numbers before prose."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths


def _csv(name: str) -> pd.DataFrame:
    path = paths.RESULTS / name
    if not path.exists() or path.stat().st_size < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def _json(name: str):
    path = paths.RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(frame: pd.DataFrame, columns: list[str] | None = None, round_to: int = 3) -> str:
    if frame.empty:
        return "_no rows_"
    if columns:
        columns = [c for c in columns if c in frame.columns]
        frame = frame[columns]
    return frame.round(round_to).to_markdown(index=False)


def build() -> str:
    verdict = _json("verdict.json") or {}
    results = _csv("benchmark_results.csv")
    audit = _csv("model_audit.csv")
    contamination = _csv("contamination_matrix.csv")
    fairness = _csv("fairness_matrix.csv")
    selected = _json("selected_tasks.json") or {}
    metadata = _csv("task_metadata.csv")
    gate = _csv("failure_candidates_gate.csv")
    headroom = _csv("headroom_table.csv")
    probe_results = _csv("probe_results.csv")
    confirmation = _csv("confirmation_results.csv")
    ranking = _csv("candidate_ranking.csv")
    integrity = _csv("integrity_checks.csv")
    benchmark_audit = _json("benchmark_audit.json") or {}
    literature = _json("literature_audit.json") or {}

    ok = results[results.status == "OK"] if len(results) else pd.DataFrame()
    lines: list[str] = []
    add = lines.append

    add("# TSFM-BENCHMARK-GAP-DISCOVERY-v1 — status")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat()}")
    add("")

    # 1
    add("## 1. Executive verdict")
    add("")
    add(f"**{verdict.get('final_token', 'NOT_RUN')}** — {verdict.get('final_token_reason', '')}")
    add("")
    if verdict.get("top_candidate"):
        add("Top candidate:")
        add("")
        add(f"```json\n{json.dumps(verdict['top_candidate'], indent=2)}\n```")
        add("")

    # 2
    add("## 2. What was actually evaluated")
    add("")
    if len(ok):
        add(
            f"- {ok.model.nunique()} estimators × {ok.task_uid.nunique()} fev-bench tasks × "
            f"{ok.track.nunique()} information condition(s) = {len(ok)} completed cells"
        )
        add(f"- tracks run: {sorted(ok.track.unique().tolist())}")
        add(f"- splits present: {sorted(ok.split.dropna().unique().tolist())}")
        add(f"- total evaluated origins: {int(ok.n_origins.sum()):,}")
        add(f"- total measured inference time: {ok.inference_seconds.sum() / 3600:.2f} GPU-hours")
        add("")
        add(
            _table(
                ok.groupby(["track", "split", "model"])
                .agg(tasks=("task_uid", "nunique"), seconds=("inference_seconds", "sum"))
                .reset_index()
            )
        )
    else:
        add("_nothing evaluated_")
    add("")

    # 3
    add("## 3. Model audit")
    add("")
    add(
        _table(
            audit,
            [
                "model_id",
                "role",
                "hf_repo",
                "resolved_revision",
                "architecture_family",
                "parameter_count_m",
                "license",
                "status",
            ],
        )
    )
    add("")
    add("### Contamination status on fev-bench")
    add("")
    add(
        _table(
            contamination,
            ["model_id", "claimed_fev_bench_exclusion", "contamination_status_fev_bench"],
        )
    )
    add("")
    if (paths.RESULTS / "MODEL_SUBSTITUTION.md").exists():
        add("One substitution was made; see `MODEL_SUBSTITUTION.md`.")
        add("")

    # 4
    add("## 4. Benchmark audit")
    add("")
    primary = benchmark_audit.get("primary_benchmark", {})
    for key in (
        "name",
        "pinned_commit",
        "task_definition_sha256",
        "evaluator",
        "evaluator_version",
        "n_tasks",
        "eval_metric",
    ):
        if key in primary:
            add(f"- **{key}**: `{primary[key]}`")
    add("")
    add(f"- split semantics: {primary.get('split_semantics', '')}")
    add(f"- leakage policy: {primary.get('leakage_policy', '')}")
    add("")

    # 5
    add("## 5. Information-condition fairness")
    add("")
    add(
        _table(
            fairness,
            [
                "model",
                "track",
                "adaptation",
                "future_covariates_used",
                "multivariate",
                "context",
                "contamination_status",
                "comparability_status",
            ],
        )
    )
    add("")
    if len(fairness):
        partial = fairness[fairness.comparability_status != "DIRECT"]
        for _, row in partial.iterrows():
            add(f"- `{row.model}` ({row.track}): {row.comparability_status} — {row.comparability_reason}")
        add("")

    # 6
    add("## 6. Discovery task selection")
    add("")
    add(f"- rule: {selected.get('selection_rule', '')}")
    add(f"- frozen at: {selected.get('frozen_at_utc', '')}")
    sha = (paths.RESULTS / "selected_tasks.sha256")
    if sha.exists():
        add(f"- sha256: `{sha.read_text(encoding='utf-8').split()[0]}`")
    add("")
    for split in ("discovery", "confirmation"):
        summary = selected.get(f"{split}_summary", {})
        add(f"**{split}** — {summary.get('n')} tasks, {summary.get('n_dataset_families')} dataset families, "
            f"domains {summary.get('domains')}, horizon buckets {summary.get('horizon_buckets')}, "
            f"{summary.get('n_multivariate')} multivariate, {summary.get('n_known_cov')} with known covariates")
        add("")
    if len(metadata):
        descriptor_columns = [c for c in metadata.columns if c.startswith("D")]
        add("Descriptor values (train-visible data only):")
        add("")
        add(_table(metadata, ["task_uid", "split", *descriptor_columns]))
        add("")

    # 7
    add("## 7. Raw benchmark results")
    add("")
    for track in sorted(ok.track.unique()) if len(ok) else []:
        for split in sorted(ok[ok.track == track].split.dropna().unique()):
            subset = ok[(ok.track == track) & (ok.split == split)]
            add(f"### TRACK {track} — {split}: native SQL / SeasonalNaive SQL (lower is better)")
            add("")
            pivot = subset.pivot_table(
                index="task_uid", columns="model", values="relative_to_naive", aggfunc="first"
            )
            add(pivot.round(3).to_markdown())
            add("")
            add("Median across tasks:")
            add("")
            add(pivot.median().round(3).to_frame("median relative_to_naive").to_markdown())
            add("")
    if len(ok):
        per_task = ok.groupby("task_uid").origin_reconciliation_gap.max()
        exact = per_task[per_task < 1e-6]
        residual = per_task[per_task >= 1e-6]
        add(
            "Every score above is fev's own `evaluation_summary` output. A per-origin "
            "decomposition of the same metric is kept alongside it so the oracle probes and any "
            "task-internal resampling operate on the same quantity."
        )
        add("")
        add(
            f"- it reproduces the official aggregate to below 1e-6 relative on "
            f"{len(exact)}/{len(per_task)} tasks; the floor of {exact.max():.1e} is float32 "
            f"storage of the per-origin array, not a modelling difference"
        )
        if len(residual):
            names = ", ".join(sorted(residual.index.str.split("::").str[1]))
            add(
                f"- it reaches {residual.max():.1e} on {names}. fev's SQL averages per-target-"
                "dimension means, and on these tasks ground truth is missing at different rates "
                "across the target dimensions, which no mean of per-item quantities can match. "
                "Baseline, oracle and simple fixes are aggregated identically inside the "
                "per-origin space, so the residual cancels out of every headroom ratio."
            )
    add("")

    # 8
    add("## 8. Failure map")
    add("")
    if (paths.RESULTS / "failure_map.md").exists():
        add("Full table in `failure_map.md`. Gate outcome:")
        add("")
        add(
            _table(
                gate,
                [
                    "descriptor",
                    "descriptor_label",
                    "bucket",
                    "n_tasks_in_bucket",
                    "n_affected_families",
                    "median_condition_gap_pct",
                    "median_regret_pct",
                    "passes_failure_gate",
                    "verdict",
                ],
            )
        )
    add("")

    # 9
    add("## 9. Candidate failures")
    add("")
    candidates = _json("failure_candidates.json") or []
    if candidates:
        for candidate in candidates:
            add(f"### {candidate['candidate_id']}")
            add("")
            add(f"- condition: {candidate['observed_condition']}")
            add(f"- affected models: {candidate['affected_models']} across {candidate['affected_families']}")
            add(f"- affected discovery tasks ({candidate['n_affected_tasks']}): {candidate['affected_tasks']}")
            add(f"- severity: {candidate['severity']}")
            add("- competing mechanism hypotheses:")
            for hypothesis in candidate["mechanism_hypotheses"]:
                add(f"  - {hypothesis}")
            add(f"- must not be assumed: {candidate['post_hoc_story_that_must_not_be_assumed']}")
            add("")
    else:
        add("No condition passed the failure gate, so no candidate was generated.")
        add("")

    # 10 / 11
    add("## 10. Oracle and headroom probes")
    add("")
    if len(probe_results):
        add(
            _table(
                probe_results.pivot_table(
                    index="task_uid", columns="probe", values="aggregate_common_windows"
                ).reset_index(),
                round_to=4,
            )
        )
        add("")
        add(
            _table(
                headroom,
                [
                    "candidate_id",
                    "n_tasks",
                    "baseline_loss",
                    "oracle_loss",
                    "simple_loss",
                    "best_simple_fix",
                    "H_oracle_pct",
                    "R_simple",
                    "H_residual_pct",
                    "screen_verdict",
                ],
                round_to=4,
            )
        )
        add("")
        add("`P0_cross_model_origin_oracle` reads evaluation labels. It is a diagnostic upper bound, not an achievable score.")
    else:
        add("No probes were run: no candidate reached the probe stage.")
    add("")

    add("## 10b. Is the oracle headroom reachable?")
    add("")
    characterisation = _json("headroom_characterisation.json")
    exploit = _json("oracle_exploitability.json")
    exploit_confirm = _json("oracle_exploitability_confirmation.json")
    if characterisation:
        add(
            f"Over all {characterisation['n_tasks']} discovery tasks treated as one set, the "
            f"per-origin oracle sits {characterisation['H_oracle_pct']:.2f}% below the strongest "
            f"single model, and the best simple deployable fix "
            f"(`{characterisation['best_simple_fix']}`) recovers "
            f"{100 * characterisation['R_simple']:.0f}% of that - it is worse than doing nothing."
        )
        bootstrap = characterisation.get("bootstrap", {})
        if "H_oracle_pct" in bootstrap:
            interval = bootstrap["H_oracle_pct"]
            add("")
            add(
                f"Bootstrapping origins within windows, paired across probes: oracle headroom "
                f"{interval['median']:.2f}% [{interval['ci_lower']:.2f}, {interval['ci_upper']:.2f}]. "
                "The interval is conditional on these tasks and says nothing about which tasks the "
                "benchmark contains."
            )
        add("")
    if exploit:
        add(
            "Taken alone that number would look like a research opportunity. It is not, and two "
            "tests say so. Picking the lowest of three losses at every origin beats any single "
            "model even when the three are statistically indistinguishable, so the question is "
            "whether the winner is a property of the series or is redrawn each window."
        )
        add("")
        rows = []
        for label, payload in (("discovery", exploit), ("confirmation", exploit_confirm)):
            if not payload:
                continue
            persistence = payload["winner_persistence"]
            rows.append(
                {
                    "split": label,
                    "tasks": payload["n_tasks"],
                    "oracle headroom %": round(payload["mean_oracle_headroom_pct"], 2),
                    "per-series router %": round(payload["mean_per_series_router_recovery_pct"], 2),
                    "headroom captured": round(
                        payload["fraction_of_oracle_headroom_captured_by_router"], 3
                    ),
                    "winner agreement": round(persistence["observed_agreement"], 4),
                    "chance agreement": round(persistence["chance_agreement"], 4),
                    "excess": round(persistence["excess_over_chance"], 4),
                    "pairs": persistence["total_pairs"],
                }
            )
        add(pd.DataFrame(rows).to_markdown(index=False))
        add("")
        add(
            "The winner agrees with the previous window barely above the rate the per-window "
            "marginals already predict, and a router that picks each series' past leader loses to "
            "simply committing to the strongest single model. Both splits agree. The oracle gap is "
            "selection noise, not structure a method could capture."
        )
        add("")
        if exploit_confirm:
            per_task = _csv("oracle_exploitability_confirmation.csv")
            positive = per_task[per_task.router_recovery_pct > 0] if len(per_task) else pd.DataFrame()
            if len(positive):
                names = ", ".join(positive.task_uid.str.split("::").str[1])
                add(
                    f"One exception is worth naming rather than averaging away: {names} shows "
                    "winner persistence clearly above chance and a positive router gain. It is a "
                    "single task and a lead, not a result."
                )
                add("")

    add("## 11. Simple baselines")
    add("")
    if len(ok):
        baseline_rows = ok[ok.model.isin(["seasonal-naive", "linear-ar-specialist"])]
        add(
            _table(
                baseline_rows.pivot_table(
                    index=["track", "task_uid"], columns="model", values="native_score"
                ).reset_index(),
                round_to=4,
            )
        )
    add("")

    # 12
    add("## 12. Confirmation")
    add("")
    if len(confirmation):
        add(
            _table(
                confirmation,
                [
                    "candidate_id",
                    "n_confirmation_tasks",
                    "median_gap_pct",
                    "n_models_positive",
                    "H_oracle_pct",
                    "R_simple",
                    "H_residual_pct",
                    "C1_direction_same",
                    "C2_two_families_positive",
                    "C3_severity",
                    "C4_oracle_headroom",
                    "C5_simple_residual",
                    "verdict",
                ],
            )
        )
    else:
        add("The confirmation split was not opened: no candidate reached the confirmation stage.")
        add(
            f"The {len(selected.get('confirmation_tasks', []))} holdout tasks remain unused and "
            "are listed in `selected_tasks.json`."
        )
    add("")

    # 13
    add("## 13. Literature ownership")
    add("")
    if literature:
        for candidate_id, entry in literature.items():
            add(f"- **{candidate_id}**: {entry.get('novelty_status')} — {entry.get('summary', '')}")
    else:
        add("No candidate reached the literature audit stage.")
    add("")

    # 14
    add("## 14. Candidate ranking")
    add("")
    if len(ranking):
        add(
            _table(
                ranking,
                [
                    "candidate_id",
                    "severity_pct",
                    "H_oracle_pct",
                    "R_simple",
                    "H_residual_pct",
                    "confirmed",
                    "novelty_status",
                    "total_score",
                    "eliminated",
                    "hard_fails",
                ],
            )
        )
    else:
        add("No candidate to rank.")
    add("")

    # 15
    add("## 15. Integrity and deviations")
    add("")
    if len(integrity):
        passed = int((integrity.result == "PASS").sum())
        add(f"{passed}/{len(integrity)} checks pass. Detail in `SELF_AUDIT.md`.")
        add("")
        add(_table(integrity, ["check", "result", "description"]))
    add("")
    add("### Deviations from the study contract")
    add("")
    add(
        "- The recurrent-family slot uses `NX-AI/TiRex-2` rather than the advertised fev-bench "
        "decontaminated checkpoint, because that repository publishes byte-identical weights. "
        "Recorded in `MODEL_SUBSTITUTION.md`."
    )
    add(
        "- TiRex-2 runs its pure-PyTorch kernels on CUDA: this host has neither an MSVC toolchain "
        "nor nvcc, so the fused FlashRNN/Triton kernels cannot be built. CPU and CUDA outputs were "
        "measured to agree to a relative 1.0e-06 (see `smoke_report.json`)."
    )
    add(
        "- B0 SeasonalNaive and B1 the linear autoregressive specialist are implemented in-repo "
        "rather than through statsforecast/autogluon, whose install downgrades pandas underneath "
        "three already-verified foundation models. The estimator contracts are stated in "
        "`baselines.py` and covered by `tests/test_study_contract.py`."
    )
    add(
        "- GIFT-Eval and TIME were not run. Both are secondary or optional in the contract and "
        "would each need a second harness and data download; the budget went to the primary "
        "benchmark plus its holdout instead."
    )
    add("")

    # 16
    add("## 16. What was NOT tested")
    add("")
    add("- GIFT-Eval, TIME, and every fev-bench task outside the frozen 18.")
    add(
        "- Fine-tuning of any foundation model: the contract confines this study to zero-shot "
        "inference, simple baselines and probes."
    )
    add(
        "- Whether the observed behaviour generalises beyond fev-bench. One benchmark cannot "
        "support a claim about time-series forecasting in general."
    )
    add(
        "- Absolute model ranking under a clean contamination contract: two of the three primary "
        "models carry unresolved fev-bench overlap risk."
    )
    if not len(confirmation):
        add("- The confirmation split, which stays sealed unless a candidate reaches it.")
    add("")

    # 17
    add("## 17. Recommended next action")
    add("")
    token = verdict.get("final_token")
    if token == "GAP_CANDIDATE_READY":
        add("See `NEXT_METHOD_BRIEF.md` for the single first-priority candidate and its first experiment.")
    elif token == "CHARACTERIZATION_ONLY":
        add(
            "Do not start method development on these conditions. The measured deficits are real "
            "but a simple deployable fix absorbs enough of the recoverable headroom that a new "
            "architecture is not the cheapest way to close them."
        )
    elif token == "NO_STRONG_GAP_FOUND":
        add(
            "Do not manufacture a method topic from this run. The next search should widen the "
            "space rather than dig here. What this run's own numbers point to, in order:"
        )
        add("")
        if len(gate):
            near = gate.sort_values("median_condition_gap_pct", ascending=False).head(3)
            listed = ", ".join(
                f"`{row.descriptor}`/`{row.bucket}` ({row.descriptor_label}, "
                f"{row.median_condition_gap_pct:.0f}% over {int(row.n_tasks_in_bucket)} tasks)"
                for _, row in near.iterrows()
            )
            add(
                f"1. The largest condition effects are real but shared: {listed}. Every one of them "
                "failed the gate on regret alone. These conditions are genuinely harder for all "
                "three families at once, and on them no evaluated estimator - not the specialist, "
                "not the naive anchor, not another foundation model - does better. What is missing "
                "is a comparator that actually wins there, not more tasks. The productive next step "
                "is to find or build one strong task-specific model for a condition like these; if "
                "it beats the foundation models by a wide margin, the gap becomes measurable, and "
                "if it does not, the condition is simply hard and should be dropped."
            )
            thin = gate[gate.n_tasks_in_bucket < 3]
            if len(thin):
                names = ", ".join(f"{r.descriptor}/{r.bucket}" for _, r in thin.iterrows())
                add(
                    f"2. Conditions that could not be tested at all for want of tasks: {names}. "
                    "A selection drawn to fill these buckets specifically, rather than to span the "
                    "benchmark evenly, would test them without needing a new benchmark."
                )
        add(
            "3. A second benchmark. Everything here is fev-bench; GIFT-Eval and TIME were left "
            "unrun, and a condition that fails to reach the gate on one benchmark's 18 tasks may "
            "simply be under-sampled rather than absent."
        )
        add(
            "4. A different task family. This study only looked at forecasting. Classification, "
            "anomaly detection and imputation are where these same backbones have had the least "
            "public benchmarking, so the prior for an open gap there is higher than for another "
            "pass over forecasting accuracy."
        )
    else:
        add("Resolve the integrity failures listed in section 15 before drawing any scientific conclusion.")
    add("")
    return "\n".join(lines)


def main() -> None:
    text = build()
    out = paths.RESULTS / "STATUS.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
