"""Single entry point for the closure stages (Section 38).

    python -m experiments.tsfm_long_horizon_specialist_closure_v1.src.run <stage>

Two interpreters are involved. Stages that fit specialists must run under
`.venv-tsfm-specialist`; everything that touches fev or a foundation model runs
under `.venv-tsfm`. Each stage declares which it needs and refuses to run under
the wrong one rather than failing halfway with an import error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

TSFM_ENV = "tsfm"
SPECIALIST_ENV = "specialist"

STAGES: dict[str, tuple[str, str]] = {
    "audit-source": ("source_audit", TSFM_ENV),
    "select-holdout": ("select_holdout", TSFM_ENV),
    "convert": ("convert_data", TSFM_ENV),
    "audit-specialists": ("specialist_env", SPECIALIST_ENV),
    "train-development": ("train_specialists", SPECIALIST_ENV),
    "train-holdout": ("train_specialists", SPECIALIST_ENV),
    "run-holdout-tsfms": ("run_holdout_tsfms", TSFM_ENV),
    "evaluate": ("evaluate_specialists", TSFM_ENV),
    "compare": ("compare", TSFM_ENV),
    "fairness": ("fairness", TSFM_ENV),
    "verdict": ("verdict", TSFM_ENV),
    "report": ("report", TSFM_ENV),
    "verify": ("verify", TSFM_ENV),
    "publication-scan": ("publication", TSFM_ENV),
    "publication-verify": ("publication", TSFM_ENV),
}

DEFAULT_ARGS = {
    "train-development": ["--split", "development", "--tag", "_development"],
    "train-holdout": ["--split", "holdout", "--tag", "_holdout"],
    "publication-scan": ["scan"],
    "publication-verify": ["verify"],
}


def _expected_env() -> str:
    return SPECIALIST_ENV if "specialist" in sys.executable.lower() else TSFM_ENV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    module, needed = STAGES[args.stage]
    running = _expected_env()
    if needed != running:
        interpreter = (
            ".venv-tsfm-specialist/Scripts/python.exe"
            if needed == SPECIALIST_ENV
            else ".venv-tsfm/Scripts/python.exe"
        )
        raise SystemExit(
            f"stage '{args.stage}' needs the {needed} environment but this is {running}. "
            f"Run it with {interpreter}."
        )

    extra = DEFAULT_ARGS.get(args.stage, []) + list(args.rest)
    command = [
        sys.executable,
        "-u",
        "-m",
        f"experiments.tsfm_long_horizon_specialist_closure_v1.src.{module}",
        *extra,
    ]
    print(f"=== {module} {' '.join(extra)} ===", flush=True)
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
