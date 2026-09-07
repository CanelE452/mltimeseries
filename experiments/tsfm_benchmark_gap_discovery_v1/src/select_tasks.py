"""Freeze the discovery / confirmation split before any model is run.

The rule below only reads benchmark metadata (Section 7). It is deterministic:
no random seed, no scores, no model output. The resulting file is hashed and the
hash is what later stages check against.

Revision note: the first draft of this rule stratified on (frequency, horizon)
only. Drawn lexically that put four ETT tasks and two SZ-Taxi tasks into a
twelve-task discovery set covering four of the benchmark's seven domains, so the
same underlying dataset would have counted as several independent observations.
The rule was rewritten to stratify on (domain, frequency) and to allow at most
one task per dataset family. No model had been run and no score existed when the
rule changed; the earliest prediction artifact postdates the freeze recorded here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd

from . import paths

DISCOVERY_TARGET = 12
CONFIRMATION_TARGET = 6

FREQ_TOKEN = re.compile(r"_(5T|10T|15T|30T|1T|1H|1D|1W|1M|1Q|1Y|T|H|D|W|M|Q|Y)$")


def dataset_family(dataset_config: str) -> str:
    """Group configs that come from the same upstream dataset.

    ETT_15T / ETT_1H / ETT_1D / ETT_1W are one family, as are boomlet_1062 and
    boomlet_1855. Tasks from one family are not independent evidence about a
    model's behaviour, so discovery may take at most one of them.
    """
    stripped = FREQ_TOKEN.sub("", dataset_config)
    head = stripped.split("_")[0]
    return head if head else stripped


def horizon_buckets(pool: pd.DataFrame) -> tuple[dict[str, float], pd.Series]:
    """D1 buckets: tertiles of horizon / median series length over the full pool."""
    low, high = pool.horizon_to_median_length.quantile([1 / 3, 2 / 3]).tolist()
    thresholds = {"short_max": float(low), "medium_max": float(high)}

    def bucket(value: float) -> str:
        if value <= low:
            return "short"
        if value <= high:
            return "medium"
        return "long"

    return thresholds, pool.horizon_to_median_length.map(bucket)


def requirements() -> dict[str, tuple[Callable[[pd.Series], bool], int, int]]:
    """condition name -> (predicate, minimum in discovery, minimum in confirmation)."""
    return {
        "multivariate": (lambda r: bool(r.is_multivariate), 3, 2),
        "known_future_covariates": (lambda r: bool(r.has_known_cov), 3, 2),
        "horizon_short": (lambda r: r.horizon_bucket == "short", 2, 1),
        "horizon_medium": (lambda r: r.horizon_bucket == "medium", 2, 1),
        "horizon_long": (lambda r: r.horizon_bucket == "long", 2, 1),
    }


def deterministic_order(pool: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Round-robin over (domain, frequency bucket) strata, lexical within a stratum.

    One pass over the strata therefore touches every domain and every frequency
    regime before any of them is drawn from twice. A task is deferred if its
    dataset family has already been drawn, so the head of the order carries as
    many distinct upstream datasets as there are picks.
    """
    strata: dict[tuple[str, str], list[str]] = {}
    for _, row in pool.sort_values("task_uid").iterrows():
        strata.setdefault((row.domain, row.freq_bucket), []).append(row.task_uid)
    stratum_keys = sorted(strata)
    families = pool.set_index("task_uid").dataset_config.map(dataset_family)

    order: list[str] = []
    deferred: list[dict] = []
    seen_families: set[str] = set()
    remaining = {key: list(tasks) for key, tasks in strata.items()}
    # First pass: at most one task per dataset family, round-robin over strata.
    while any(remaining.values()):
        progressed = False
        for key in stratum_keys:
            queue = remaining[key]
            pick_index = next(
                (i for i, task in enumerate(queue) if families[task] not in seen_families), None
            )
            if pick_index is None:
                continue
            task = queue.pop(pick_index)
            order.append(task)
            seen_families.add(families[task])
            progressed = True
        if not progressed:
            break
    # Second pass: everything still unplaced, in stratum then lexical order.
    for key in stratum_keys:
        for task in remaining[key]:
            deferred.append({"task_uid": task, "family": families[task]})
            order.append(task)
    return order, deferred


def _repair(
    chosen: list[str],
    order: list[str],
    pool: pd.DataFrame,
    slot: int,
    reserved: set[str],
    log: list[dict],
    label: str,
) -> list[str]:
    """Swap in tasks supplying a missing required condition, deterministically.

    The task swapped out is the latest-drawn member of the most represented
    stratum that is not itself the last supplier of another requirement.
    """
    indexed = pool.set_index("task_uid")
    reqs = requirements()
    for name, (predicate, *minimums) in reqs.items():
        minimum = minimums[slot]
        while sum(predicate(indexed.loc[t]) for t in chosen) < minimum:
            donor = next(
                (
                    t
                    for t in order
                    if t not in chosen and t not in reserved and predicate(indexed.loc[t])
                ),
                None,
            )
            if donor is None:
                raise SystemExit(f"HARD STOP: cannot satisfy {label} requirement {name}")
            counts = pd.Series([indexed.loc[t, "stratum"] for t in chosen]).value_counts()
            droppable = []
            for candidate in chosen:
                if predicate(indexed.loc[candidate]):
                    continue
                blocks_other = any(
                    other_pred(indexed.loc[candidate])
                    and sum(other_pred(indexed.loc[u]) for u in chosen) <= other_min[slot]
                    for other_name, (other_pred, *other_min) in reqs.items()
                    if other_name != name
                )
                if not blocks_other:
                    droppable.append(candidate)
            droppable.sort(key=lambda t: (-counts[indexed.loc[t, "stratum"]], -order.index(t)))
            if not droppable:
                raise SystemExit(f"HARD STOP: no droppable task for {label}/{name}")
            dropped = droppable[0]
            chosen = [donor if t == dropped else t for t in chosen]
            log.append(
                {"set": label, "requirement": name, "swapped_in": donor, "swapped_out": dropped}
            )
    return chosen


def build() -> dict:
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv")
    thresholds, buckets = horizon_buckets(pool)
    pool = pool.assign(horizon_bucket=buckets)
    pool["stratum"] = pool.domain + "|" + pool.freq_bucket
    pool["dataset_family"] = pool.dataset_config.map(dataset_family)

    order, deferred = deterministic_order(pool)
    discovery = order[:DISCOVERY_TARGET]
    confirmation = order[DISCOVERY_TARGET : DISCOVERY_TARGET + CONFIRMATION_TARGET]

    repairs: list[dict] = []
    discovery = _repair(discovery, order, pool, 0, set(confirmation), repairs, "discovery")
    confirmation = _repair(confirmation, order, pool, 1, set(discovery), repairs, "confirmation")
    if set(discovery) & set(confirmation):
        raise SystemExit("HARD STOP: discovery and confirmation overlap")

    indexed = pool.set_index("task_uid")

    def summarise(tasks: list[str]) -> dict:
        sub = indexed.loc[tasks]
        return {
            "n": len(tasks),
            "domains": sub.domain.value_counts().to_dict(),
            "freq_buckets": sub.freq_bucket.value_counts().to_dict(),
            "horizon_buckets": sub.horizon_bucket.value_counts().to_dict(),
            "dataset_families": sorted(sub.dataset_family.unique().tolist()),
            "n_dataset_families": int(sub.dataset_family.nunique()),
            "n_multivariate": int(sub.is_multivariate.sum()),
            "n_known_cov": int(sub.has_known_cov.sum()),
            "n_past_cov": int(sub.has_past_cov.sum()),
            "total_track_u_forecasts": int(sub.n_forecasts_track_u.sum()),
        }

    return {
        "experiment": "TSFM-BENCHMARK-GAP-DISCOVERY-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "fev-bench",
        "tasks_yaml_sha256": "c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9",
        "selection_rule": (
            "Round-robin over (domain, frequency bucket) strata with tasks sorted lexically by "
            "task_uid inside a stratum, deferring any task whose dataset family has already been "
            "drawn; the first 12 tasks are discovery, the next 6 confirmation. Documented repair "
            "swaps then guarantee the registered coverage minimums."
        ),
        "d1_horizon_ratio_thresholds": thresholds,
        "required_coverage": {
            name: {"discovery_min": mins[0], "confirmation_min": mins[1]}
            for name, (_, *mins) in requirements().items()
        },
        "repairs_applied": repairs,
        "deferred_same_family_tasks": deferred[:20],
        "draw_order": order,
        "discovery_tasks": discovery,
        "confirmation_tasks": confirmation,
        "discovery_summary": summarise(discovery),
        "confirmation_summary": summarise(confirmation),
    }


def main() -> None:
    spec = build()
    out = paths.RESULTS / "selected_tasks.json"
    payload = json.dumps(spec, indent=2, sort_keys=True)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (paths.RESULTS / "selected_tasks.sha256").write_text(
        f"{digest}  selected_tasks.json\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    print(f"sha256 {digest}")
    for label in ("discovery", "confirmation"):
        print(f"\n{label.upper()}")
        for task in spec[f"{label}_tasks"]:
            print("  ", task)
        print(json.dumps(spec[f"{label}_summary"], indent=2))
    if spec["repairs_applied"]:
        print("\nREPAIRS")
        for repair in spec["repairs_applied"]:
            print("  ", repair)


if __name__ == "__main__":
    main()
