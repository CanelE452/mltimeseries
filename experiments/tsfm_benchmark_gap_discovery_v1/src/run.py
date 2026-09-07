"""Single entry point for the study stages.

    python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run <stage>

Stages run in the order the contract requires and each one refuses to start if
the freeze it depends on is missing. `all-discovery` chains everything that is
allowed before the confirmation split is opened, so a long run ends with a
verdict rather than with a finished job someone has to notice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = {
    "base-state": "base_state",
    "task-pool": "task_pool",
    "select-tasks": "select_tasks",
    "audit-models": "audit_models",
    "audit-benchmarks": "audit_benchmarks",
    "descriptors": "descriptors",
    "smoke": "smoke",
    "evaluate": "evaluate",
    "fairness": "fairness",
    "failure-map": "failure_map",
    "freeze-candidates": "make_candidates",
    "probes": "run_probes",
    "confirmation": "confirm_candidates",
    "literature": "literature_audit",
    "rank": "rank_candidates",
    "verify": "verify",
    "report": "report",
    "status": "status",
    "brief": "next_method_brief",
}

CHAINS = {
    "setup": ["base-state", "task-pool", "select-tasks", "audit-models", "audit-benchmarks"],
    "analyse-discovery": [
        "descriptors",
        "evaluate",
        "fairness",
        "failure-map",
        "freeze-candidates",
        "probes",
    ],
    "finish": [
        "evaluate",
        "fairness",
        "literature",
        "rank",
        "verify",
        "report",
        "brief",
        "status",
    ],
}


def _run(module: str, extra: list[str]) -> int:
    command = [sys.executable, "-u", "-m", f"experiments.tsfm_benchmark_gap_discovery_v1.src.{module}", *extra]
    print(f"\n=== {module} ===", flush=True)
    return subprocess.run(command).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[*STAGES, *CHAINS, "models"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.stage == "models":
        raise SystemExit(_run("run_models", args.rest))
    if args.stage in CHAINS:
        for stage in CHAINS[args.stage]:
            code = _run(STAGES[stage], [])
            if code != 0:
                print(f"\nstage {stage} exited {code}; chain stopped", flush=True)
                raise SystemExit(code)
        raise SystemExit(0)
    raise SystemExit(_run(STAGES[args.stage], args.rest))


if __name__ == "__main__":
    main()
