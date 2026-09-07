"""Choose fresh long-horizon holdout tasks, then freeze and hash them.

Everything here runs before any specialist is trained. The long-horizon
definition is not recomputed: the D1 cut points frozen by the source study are
reused verbatim, so this study cannot move the boundary to suit itself.

Two-step candidate filter, both parts train-visible only:

1. A metadata bound. The visible context at window 0 is `median series length +
   cutoff[0]`, and dropping series that are too short at the cutoff can only
   raise the surviving median. So the metadata estimate of horizon / context is
   an upper bound on the true ratio, and anything it puts at or below the
   threshold cannot be long. On the source study's own 18 tasks the bound is
   exact on 17 and errs upward on the one boundary case, as the mechanism
   predicts.
2. The exact ratio, for whatever survives step 1, computed with the source
   study's own `_visible_history` so the number means the same thing it did
   there.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import yaml

from experiments.tsfm_benchmark_gap_discovery_v1.src.descriptors import _visible_history
from experiments.tsfm_benchmark_gap_discovery_v1.src.select_tasks import dataset_family

from . import paths

TARGET_HOLDOUT = 6
MIN_HOLDOUT = 4


def _source() -> dict:
    return json.loads((paths.RESULTS / "SOURCE_STUDY.json").read_text(encoding="utf-8"))


def _tasks_yaml() -> list[dict]:
    return yaml.safe_load(
        (paths.SOURCE_DATA / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]


def candidate_pool(source: dict) -> tuple[pd.DataFrame, set[str]]:
    """Everything not used before, and not from a family used before."""
    pool = pd.read_csv(paths.SOURCE_RESULTS / "task_pool.csv")
    pool["dataset_family"] = pool.dataset_config.map(dataset_family)
    previous = set(source["previously_used_tasks"])
    previous_families = set(pool[pool.task_uid.isin(previous)].dataset_family)
    candidates = pool[
        (~pool.task_uid.isin(previous)) & (~pool.dataset_family.isin(previous_families))
    ].copy()
    return candidates, previous_families


def metadata_upper_bound(candidates: pd.DataFrame, raw: list[dict]) -> pd.DataFrame:
    """Upper bound on horizon / visible-context, or infinity when it needs data."""
    import fev

    bounds, needs_data = [], []
    for _, row in candidates.iterrows():
        task = fev.Task(**raw[int(row.yaml_index)])
        try:
            first_cutoff = task.cutoffs[0]
        except ValueError:
            # A timestamp cutoff cannot be resolved without the data; keep the task
            # rather than risk dropping a long one on a technicality.
            bounds.append(float("inf"))
            needs_data.append(True)
            continue
        context = row.median_length + first_cutoff
        bounds.append(row.horizon / context if context > 0 else float("inf"))
        needs_data.append(False)
    return candidates.assign(metadata_ratio_upper_bound=bounds, cutoff_needs_data=needs_data)


def exact_ratios(shortlist: pd.DataFrame, raw: list[dict]) -> pd.DataFrame:
    """The frozen D1 quantity, computed exactly, from window 0's visible past."""
    import fev
    import numpy as np

    ratios, contexts = [], []
    for _, row in shortlist.iterrows():
        task = fev.Task(**raw[int(row.yaml_index)])
        task.load_full_dataset(num_proc=1)
        stats = _visible_history(task)
        context = float(np.median(stats["length"]))
        contexts.append(context)
        ratios.append(float(row.horizon / context) if context > 0 else float("inf"))
    return shortlist.assign(visible_context_median=contexts, horizon_to_context_ratio=ratios)


def deterministic_pick(long_tasks: pd.DataFrame, limit: int) -> tuple[list[str], list[dict]]:
    """Round-robin over (domain, frequency bucket), lexical inside, one per family."""
    strata: dict[tuple[str, str], list[str]] = {}
    for _, row in long_tasks.sort_values("task_uid").iterrows():
        strata.setdefault((row.domain, row.freq_bucket), []).append(row.task_uid)
    keys = sorted(strata)
    families = long_tasks.set_index("task_uid").dataset_family

    picked: list[str] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    depth = 0
    while len(picked) < limit and any(len(v) > depth for v in strata.values()):
        for key in keys:
            if len(picked) >= limit:
                break
            queue = strata[key]
            if len(queue) <= depth:
                continue
            task = queue[depth]
            family = families[task]
            if family in seen:
                skipped.append({"task_uid": task, "reason": "family already drawn", "family": family})
                continue
            picked.append(task)
            seen.add(family)
        depth += 1
    return picked, skipped


def build() -> dict:
    source = _source()
    raw = _tasks_yaml()
    thresholds = source["d1_horizon_ratio_thresholds"]
    long_cut = float(thresholds["high"])

    candidates, previous_families = candidate_pool(source)
    bounded = metadata_upper_bound(candidates, raw)
    shortlist = bounded[bounded.metadata_ratio_upper_bound > long_cut].copy()
    measured = exact_ratios(shortlist, raw)
    long_tasks = measured[measured.horizon_to_context_ratio > long_cut].copy()

    picked, skipped = deterministic_pick(long_tasks, TARGET_HOLDOUT)
    indexed = long_tasks.set_index("task_uid")
    coverage = {
        "n": len(picked),
        "domains": indexed.loc[picked].domain.value_counts().to_dict() if picked else {},
        "freq_buckets": indexed.loc[picked].freq_bucket.value_counts().to_dict() if picked else {},
        "dataset_families": sorted(indexed.loc[picked].dataset_family.unique()) if picked else [],
        "total_track_u_forecasts": int(indexed.loc[picked].n_forecasts_track_u.sum()) if picked else 0,
    }

    status = "OK"
    if len(picked) < MIN_HOLDOUT:
        status = "HOLDOUT_COVERAGE_INSUFFICIENT"
    elif len(picked) < TARGET_HOLDOUT:
        status = "REDUCED_HOLDOUT"

    return {
        "experiment": "TSFM-LONG-HORIZON-SPECIALIST-CLOSURE-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_remote_sha": source["source_remote_sha"],
        "track": "U",
        "d1_thresholds": thresholds,
        "d1_threshold_reused_from": source["d1_threshold_source"],
        "long_definition": f"horizon_to_context_ratio > {long_cut}",
        "selection_rule": (
            "Exclude the source study's 18 tasks and every task sharing a dataset family with "
            "them. Bound horizon/visible-context from metadata, keep whatever the bound cannot "
            "rule out, measure the exact ratio on window 0's visible past, keep the ones above the "
            "frozen long cut. Then round-robin over (domain, frequency bucket) strata with tasks "
            "in lexical order inside a stratum, taking at most one task per dataset family, and "
            f"stop at {TARGET_HOLDOUT}."
        ),
        "development_tasks": source["development_tasks"],
        "fresh_holdout_tasks": picked,
        "excluded_previous_tasks": source["previously_used_tasks"],
        "excluded_families": sorted(previous_families),
        "n_candidates_after_exclusions": int(len(candidates)),
        "n_after_metadata_bound": int(len(shortlist)),
        "n_long_after_exact_measurement": int(len(long_tasks)),
        "long_candidates": [
            {
                "task_uid": row.task_uid,
                "domain": row.domain,
                "freq_bucket": row.freq_bucket,
                "dataset_family": row.dataset_family,
                "horizon": int(row.horizon),
                "num_windows": int(row.num_windows),
                "n_series": int(row.n_series),
                "n_targets": int(row.n_targets),
                "visible_context_median": float(row.visible_context_median),
                "horizon_to_context_ratio": round(float(row.horizon_to_context_ratio), 6),
                "n_forecasts_track_u": int(row.n_forecasts_track_u),
                "selected": row.task_uid in picked,
            }
            for _, row in long_tasks.sort_values("task_uid").iterrows()
        ],
        "skipped_for_family_duplication": skipped,
        "holdout_summary": coverage,
        "status": status,
    }


def main() -> None:
    spec = build()
    payload = json.dumps(spec, indent=2, sort_keys=True)
    out = paths.RESULTS / "long_horizon_tasks.json"
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (paths.RESULTS / "long_horizon_tasks.sha256").write_text(
        f"{digest}  long_horizon_tasks.json\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    print(f"sha256 {digest}")
    print(
        f"candidates {spec['n_candidates_after_exclusions']} -> metadata bound "
        f"{spec['n_after_metadata_bound']} -> measured long {spec['n_long_after_exact_measurement']}"
    )
    print(f"status: {spec['status']}")
    print("\nfresh holdout:")
    for task in spec["fresh_holdout_tasks"]:
        entry = next(c for c in spec["long_candidates"] if c["task_uid"] == task)
        print(
            f"  {task.split('::')[1]:28s} {entry['domain']:11s} {entry['freq_bucket']:17s} "
            f"h={entry['horizon']:<4d} ratio={entry['horizon_to_context_ratio']:.3f} "
            f"forecasts={entry['n_forecasts_track_u']}"
        )
    print(f"\ncoverage: {spec['holdout_summary']}")
    if spec["status"] == "HOLDOUT_COVERAGE_INSUFFICIENT":
        raise SystemExit("HOLDOUT_COVERAGE_INSUFFICIENT -> INCONCLUSIVE_SPECIALIST_POWER")


if __name__ == "__main__":
    main()
