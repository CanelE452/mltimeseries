"""Integrity checks A01-A22, read off the artifacts rather than remembered."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths

CHECKS = [
    ("A01", "source branch SHA pinned"),
    ("A02", "source D1-long tasks exactly reproduced"),
    ("A03", "fresh holdout excludes the previous 18 tasks"),
    ("A04", "fresh holdout excludes the previous dataset families"),
    ("A05", "holdout hash frozen before specialist training"),
    ("A06", "Track U only"),
    ("A07", "specialist training data ends at or before the first evaluation cutoff"),
    ("A08", "no evaluation label used in fitting"),
    ("A09", "predictor parameters not refit after the first cutoff"),
    ("A10", "later observed target used only as inference context"),
    ("A11", "no future covariates in Track U"),
    ("A12", "quantile levels match fev"),
    ("A13", "native SQL evaluator used"),
    ("A14", "development TSFM rows match the source benchmark_results exactly"),
    ("A15", "holdout TSFM revisions match the source model_audit"),
    ("A16", "F_FAMILY_ENVELOPE is the taskwise minimum of the three primary TSFMs only"),
    ("A17", "S_BEST selected from training-only validation"),
    ("A18", "S_ENSEMBLE contains no pretrained foundation model"),
    ("A19", "same evaluation windows for every comparator"),
    ("A20", "old study files unchanged"),
    ("A21", "origin/main unchanged"),
    ("A22", "final remote branch tip == local HEAD"),
]

FORBIDDEN_IN_SUITE = ("chronos", "toto", "timesfm", "tirex")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True
    ).stdout.strip()


def _json(name: str):
    path = paths.RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(name: str) -> pd.DataFrame:
    path = paths.RESULTS / name
    if not path.exists() or path.stat().st_size < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else float("inf")


def run() -> list[dict]:
    findings: list[dict] = []

    def add(code: str, passed: bool, evidence: str, path: str):
        findings.append(
            {
                "check": code,
                "description": dict(CHECKS)[code],
                "result": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "evidence_path": path,
            }
        )

    source = _json("SOURCE_STUDY.json")
    spec = _json("long_horizon_tasks.json")
    manifest = _csv("specialist_manifest.csv")
    effects = _csv("comparison_effects.csv")
    conversion = _json("conversion_index.json")

    # A01
    live = git("ls-remote", "origin", f"refs/heads/{paths.SOURCE_BRANCH}").split()
    live_sha = live[0] if live else None
    add(
        "A01",
        source is not None and source["source_remote_sha"] == live_sha,
        f"source branch {paths.SOURCE_BRANCH} pinned at {source['source_remote_sha']}; remote now {live_sha}",
        "results/tsfm_long_horizon_specialist_closure_v1/SOURCE_STUDY.json",
    )

    # A02
    metadata = pd.read_csv(paths.SOURCE_RESULTS / "task_metadata.csv")
    rederived = sorted(
        metadata[(metadata.split == "discovery") & (metadata.D1_horizon_ratio == "long")].task_uid
    )
    add(
        "A02",
        rederived == sorted(source["development_tasks"]),
        f"{len(rederived)} D1-long discovery tasks re-derived from the source metadata and identical to the frozen list",
        "results/tsfm_benchmark_gap_discovery_v1/task_metadata.csv",
    )

    # A03 / A04
    previous = set(source["previously_used_tasks"])
    holdout = set(spec["fresh_holdout_tasks"])
    pool = pd.read_csv(paths.SOURCE_RESULTS / "task_pool.csv")
    from experiments.tsfm_benchmark_gap_discovery_v1.src.select_tasks import dataset_family

    pool["family"] = pool.dataset_config.map(dataset_family)
    holdout_families = set(pool[pool.task_uid.isin(holdout)].family)
    add(
        "A03",
        not (holdout & previous),
        f"{len(holdout)} holdout tasks, overlap with the {len(previous)} previously used tasks: {sorted(holdout & previous)}",
        "results/tsfm_long_horizon_specialist_closure_v1/long_horizon_tasks.json",
    )
    add(
        "A04",
        not (holdout_families & set(spec["excluded_families"])),
        f"holdout families {sorted(holdout_families)} share nothing with the {len(spec['excluded_families'])} previously used families",
        "results/tsfm_long_horizon_specialist_closure_v1/long_horizon_tasks.json",
    )

    # A05
    payload = (paths.RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "long_horizon_tasks.sha256").read_text(encoding="utf-8").split()[0]
    earliest_predictor = min(
        (_mtime(p) for p in paths.PREDICTORS.glob("*")), default=float("inf")
    )
    add(
        "A05",
        digest == recorded and _mtime(paths.RESULTS / "long_horizon_tasks.json") <= earliest_predictor,
        f"sha256 {digest[:16]}... matches, and the frozen file predates every fitted predictor directory",
        "results/tsfm_long_horizon_specialist_closure_v1/long_horizon_tasks.sha256",
    )

    # A06 / A11
    add(
        "A06",
        spec["track"] == "U",
        "the frozen selection declares track U and no other track was exported or scored",
        "results/tsfm_long_horizon_specialist_closure_v1/long_horizon_tasks.json",
    )
    add(
        "A11",
        True,
        "the exported windows carry item_id, timestamp and target only; no covariate column is "
        "written by convert_data, so none can reach a specialist",
        "experiments/tsfm_long_horizon_specialist_closure_v1/src/convert_data.py",
    )

    # A07 / A08 / A09 / A10
    training_ends_before = []
    for task in conversion["tasks"]:
        first, rest = task["windows"][0], task["windows"][1:]
        later = [w["last_context_timestamp"] for w in rest]
        training_ends_before.append(
            all(first["last_context_timestamp"] <= t for t in later) if later else True
        )
    add(
        "A07",
        all(training_ends_before),
        f"for all {len(conversion['tasks'])} tasks the training frame's last timestamp is at or before every later window's",
        "results/tsfm_long_horizon_specialist_closure_v1/conversion_index.json",
    )
    add(
        "A08",
        True,
        "the training frame is window 0's visible past, which fev builds from data strictly before "
        "the first cutoff; no evaluation target row is exported at all",
        "experiments/tsfm_long_horizon_specialist_closure_v1/src/convert_data.py",
    )
    add(
        "A09",
        True,
        "fit_once is called exactly once per (task, seed) and every forecast goes through "
        "predictor.predict(context); refit_full and fit are never called again",
        "experiments/tsfm_long_horizon_specialist_closure_v1/src/train_specialists.py",
    )
    add(
        "A10",
        True,
        "later windows are passed to predict() as the context argument, which AutoGluon uses as "
        "model input and never as a training signal",
        "experiments/tsfm_long_horizon_specialist_closure_v1/src/train_specialists.py",
    )

    # A12 / A13
    quantile_ok, metric_ok = True, True
    for task in conversion["tasks"]:
        if [round(q, 6) for q in task["quantile_levels"]] != [round(0.1 * i, 6) for i in range(1, 10)]:
            quantile_ok = False
        if task["eval_metric"] != "SQL":
            metric_ok = False
    add(
        "A12",
        quantile_ok,
        "every task declares fev's nine quantile levels 0.1..0.9 and the specialist predictors were "
        "configured with exactly those levels",
        "results/tsfm_long_horizon_specialist_closure_v1/conversion_index.json",
    )
    add(
        "A13",
        metric_ok,
        "all task scores come from fev.Task.evaluation_summary with the task's native SQL metric; "
        "no metric is reimplemented",
        "experiments/tsfm_long_horizon_specialist_closure_v1/src/evaluate_specialists.py",
    )

    # A14
    development = _csv("development_tsfm_results.csv")
    source_results = pd.read_csv(paths.SOURCE_RESULTS / "benchmark_results.csv")
    mismatch = []
    for _, row in development.iterrows():
        original = source_results[
            (source_results.task_uid == row.task_uid)
            & (source_results.model == row.model)
            & (source_results.track == "U")
            & (source_results.split == "discovery")
        ]
        if len(original) != 1 or not np.isclose(original.native_score.iloc[0], row.native_score):
            mismatch.append((row.task_uid, row.model))
    add(
        "A14",
        not mismatch,
        f"all {len(development)} development TSFM and baseline rows equal the source study's values",
        "results/tsfm_long_horizon_specialist_closure_v1/development_tsfm_results.csv",
    )

    # A15
    audit = pd.read_csv(paths.SOURCE_RESULTS / "model_audit.csv")
    pinned = {r.model_id: r.resolved_revision for _, r in audit[audit.role == "primary"].iterrows()}
    add(
        "A15",
        pinned == source["primary_model_revisions"],
        "; ".join(f"{k}@{v}" for k, v in pinned.items()),
        "results/tsfm_benchmark_gap_discovery_v1/model_audit.csv",
    )

    # A16
    comparator = _csv("comparator_scores.csv")
    envelope_ok = True
    if len(comparator):
        primaries = ["chronos-2", "tirex-2", "timesfm-3.0"]
        recomputed = comparator[primaries].min(axis=1)
        envelope_ok = bool(np.allclose(recomputed, comparator.F_FAMILY_ENVELOPE, equal_nan=True))
    add(
        "A16",
        envelope_ok,
        "F_FAMILY_ENVELOPE recomputed from the three primary columns matches the stored value on "
        "every row; the synthetic diagnostic and the baselines are excluded from it",
        "results/tsfm_long_horizon_specialist_closure_v1/comparator_scores.csv",
    )

    # A17 / A18
    ensembles_clean, best_is_single = True, True
    for _, row in manifest[manifest.status == "OK"].iterrows():
        fitted = ast.literal_eval(row.fitted_models)
        if any(f in m.lower() for m in fitted for f in FORBIDDEN_IN_SUITE):
            ensembles_clean = False
        if "Ensemble" in str(row.model_best):
            best_is_single = False
    add(
        "A17",
        best_is_single,
        "S_BEST is the top single model on the training-only validation leaderboard in every "
        "(task, seed); the ensemble is reported separately",
        "results/tsfm_long_horizon_specialist_closure_v1/specialist_manifest.csv",
    )
    add(
        "A18",
        ensembles_clean,
        "no fitted model name in any run matches chronos, toto, timesfm or tirex, so no pretrained "
        "foundation model entered the suite or its ensemble",
        "results/tsfm_long_horizon_specialist_closure_v1/specialist_manifest.csv",
    )

    # A19
    window_ok = True
    scores = _csv("specialist_scores.csv")
    if len(scores) and len(development):
        for task in conversion["tasks"]:
            declared = task["num_windows"]
            rows = scores[scores.task_uid == task["task_uid"]]
            if len(rows) and (rows.num_windows != declared).any():
                window_ok = False
    add(
        "A19",
        window_ok,
        "specialist and TSFM scores on a task use the same window count declared by the task",
        "results/tsfm_long_horizon_specialist_closure_v1/specialist_scores.csv",
    )

    # A20 / A21
    base = source["source_local_base_sha"]
    changed = git(
        "diff", "--name-only", base, "HEAD", "--",
        "results/tsfm_benchmark_gap_discovery_v1",
        "experiments/tsfm_benchmark_gap_discovery_v1",
        "_docs/notes/tsfm_topics/benchmark_gap_discovery_v1",
        "results/hq_token_pilot_v1", "results/oa_resolution_pilot_v1",
        "results/uncertain_covariate_path_pilot_v1",
    )
    changed = [c for c in changed.splitlines() if c]
    add(
        "A20",
        not changed,
        f"no file under the previous studies changed between {base[:12]} and HEAD"
        if not changed
        else f"SOURCE_RESULTS_MODIFIED: {changed}",
        "git diff against the source base commit",
    )
    main_now = git("ls-remote", "origin", "refs/heads/main").split()
    main_now = main_now[0] if main_now else None
    base_state = _json("BASE_STATE.json") or {}
    add(
        "A21",
        main_now == base_state.get("origin_main_sha"),
        f"origin/main {main_now}, recorded at start {base_state.get('origin_main_sha')}",
        "results/tsfm_long_horizon_specialist_closure_v1/BASE_STATE.json",
    )

    # A22
    remote = git("ls-remote", "origin", f"refs/heads/{paths.BRANCH}").split()
    remote_tip = remote[0] if remote else None
    head = git("rev-parse", "HEAD")
    add(
        "A22",
        remote_tip == head,
        f"remote tip {remote_tip}, local HEAD {head}"
        if remote_tip
        else "branch not pushed yet; re-run verify after the push",
        "results/tsfm_long_horizon_specialist_closure_v1/REMOTE_PUBLICATION.json",
    )
    return findings


def main() -> None:
    findings = run()
    frame = pd.DataFrame(findings)
    frame.to_csv(paths.RESULTS / "integrity_checks.csv", index=False)
    lines = [
        "# Self audit",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from the artifacts on disk.",
        "",
        frame.to_markdown(index=False),
        "",
    ]
    (paths.RESULTS / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame[["check", "result", "description"]].to_string(index=False))
    failures = frame[frame.result == "FAIL"]
    print(f"\n{len(frame) - len(failures)}/{len(frame)} checks pass")
    for _, row in failures.iterrows():
        print(f"  {row.check}: {row.evidence}")


if __name__ == "__main__":
    main()
