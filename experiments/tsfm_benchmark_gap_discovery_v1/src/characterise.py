"""Oracle and simple-fix characterisation of the whole discovery set.

No condition passed the failure gate, so no candidate exists and the probe bank
was never triggered. Section 32 permits stopping there. But Section 40 still asks
the final report to answer, with numbers, how much headroom a diagnostic oracle
shows and how much of it a simple deployable fix already takes. Those numbers do
not depend on a candidate: the per-origin losses are already on disk, so the same
probe bank runs over the discovery set as a whole.

What comes out is a characterisation, not a candidate. It describes the models as
a group on the twelve discovery tasks and is not evidence for any condition.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths, probes, run_probes, uncertainty
from .models import PRIMARY_MODELS

TRACK = "U"


def main() -> None:
    selected = json.loads((paths.RESULTS / "selected_tasks.json").read_text(encoding="utf-8"))
    tasks = selected["discovery_tasks"]

    probe_results, raw = run_probes.run(TRACK, tasks)
    probe_results = probe_results.assign(scope="discovery_set_characterisation")
    probe_results.to_csv(paths.RESULTS / "probe_results.csv", index=False)

    pseudo = {
        "candidate_id": "CHARACTERISATION_discovery_set",
        "affected_tasks": tasks,
        "affected_families": sorted(
            pd.read_csv(paths.RESULTS / "model_audit.csv")
            .query("role == 'primary'")
            .architecture_family.unique()
            .tolist()
        ),
    }
    table = run_probes.headroom(probe_results, [pseudo], TRACK)
    if not table.empty:
        table = run_probes.screen(table)
    if table.empty:
        print("no probe covered every discovery task; nothing to characterise")
        return
    row = table.iloc[0]

    interval = {}
    if row.best_simple_fix in next(iter(raw.values())):
        interval = uncertainty.headroom_interval(raw, row.best_simple_fix)

    per_probe = probe_results.pivot_table(
        index="task_uid", columns="probe", values="aggregate_common_windows"
    )
    payload = {
        "scope": "all 12 TRACK U discovery tasks, treated as one set",
        "not_a_candidate": (
            "No descriptor bucket passed the failure gate. These numbers characterise the "
            "model set as a whole and are not evidence that any condition is a gap."
        ),
        "baseline_model": json.loads(
            probe_results[probe_results.probe == "BASELINE_strongest_foundation_model"].detail.iloc[0]
        ),
        "n_tasks": int(row.n_tasks),
        "baseline_loss": float(row.baseline_loss),
        "oracle_loss": float(row.oracle_loss),
        "oracle_is_deployable": False,
        "simple_loss": float(row.simple_loss),
        "best_simple_fix": row.best_simple_fix,
        "per_simple_fix_loss": json.loads(row.per_simple_fix_loss),
        "H_oracle_pct": float(row.H_oracle_pct),
        "R_simple": float(row.R_simple),
        "H_residual_pct": float(row.H_residual_pct),
        "bootstrap": interval,
        "screen_if_it_had_been_a_candidate": {
            "passes_oracle_headroom": bool(row.passes_oracle_headroom),
            "passes_simple_recovery": bool(row.passes_simple_recovery),
            "passes_residual": bool(row.passes_residual),
            "simple_baseline_solves": bool(row.simple_baseline_solves),
            "verdict": row.screen_verdict,
        },
    }
    (paths.RESULTS / "headroom_characterisation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    per_probe.to_csv(paths.RESULTS / "headroom_characterisation_per_task.csv")

    print(json.dumps({k: v for k, v in payload.items() if k != "baseline_model"}, indent=2))
    print("\nper-task probe aggregates:")
    print(per_probe.round(4).to_string())


if __name__ == "__main__":
    main()
