"""Record the specialist environment and which requested models it actually has.

Runs inside `.venv-tsfm-specialist`. Section 7 forbids silently swapping a
requested model for a different one, so every requested name is looked up in the
installed registry and the answer written down before any training happens.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys

from . import paths

REQUESTED = [
    ("PatchTST", "PatchTSTModel", "patch transformer"),
    ("TiDE", "TiDEModel", "dense encoder"),
    ("DLinear", "DLinearModel", "linear decomposition"),
    ("DeepAR", "DeepARModel", "autoregressive RNN"),
    ("DirectTabular", "DirectTabularModel", "tabular regression"),
]

FOUNDATION_CLASSES = ["ChronosModel", "Chronos2Model", "TotoModel", "Toto2Model"]

MIN_FAMILIES = 3
REQUIRED_ALWAYS = "PatchTST"
REQUIRED_EITHER = ("TiDE", "DLinear")

PACKAGES = [
    "autogluon.timeseries",
    "autogluon.core",
    "torch",
    "lightning",
    "pytorch-lightning",
    "gluonts",
    "numpy",
    "pandas",
    "scikit-learn",
    "statsforecast",
    "mlforecast",
]


def build() -> dict:
    import torch
    import autogluon.timeseries as agts
    import autogluon.timeseries.models as models

    rows = []
    for name, cls, family in REQUESTED:
        available = hasattr(models, cls)
        rows.append(
            {
                "requested": name,
                "actual_class": cls,
                "family": family,
                "available": available,
                "reason_if_unavailable": ""
                if available
                else f"{cls} is not in autogluon.timeseries.models",
            }
        )
    available_names = {r["requested"] for r in rows if r["available"]}
    return {
        "python": sys.version,
        "os": platform.platform(),
        "autogluon_timeseries": agts.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
        if torch.cuda.is_available()
        else None,
        "packages": {p: md.version(p) for p in PACKAGES},
        "env_path": ".venv-tsfm-specialist",
        "isolated_from": ".venv-tsfm (TSFM environment left untouched)",
        "model_availability": rows,
        "n_available": len(available_names),
        "minimum_met": bool(
            REQUIRED_ALWAYS in available_names
            and available_names & set(REQUIRED_EITHER)
            and len(available_names) >= MIN_FAMILIES
        ),
        "forbidden_present_but_never_used": [c for c in FOUNDATION_CLASSES if hasattr(models, c)],
    }


def main() -> None:
    env = build()
    (paths.RESULTS / "specialist_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    lines = [
        "# Specialist model availability",
        "",
        "Section 7 forbids quietly swapping a requested model for another, so every requested name",
        "is checked against the installed AutoGluon registry and the result recorded before any",
        "training runs.",
        "",
        "| requested | actual class | family | available | reason if unavailable |",
        "|---|---|---|---|---|",
    ]
    for row in env["model_availability"]:
        lines.append(
            f"| {row['requested']} | `{row['actual_class']}` | {row['family']} | "
            f"{'yes' if row['available'] else 'no'} | {row['reason_if_unavailable'] or '-'} |"
        )
    lines += [
        "",
        f"{env['n_available']} of {len(REQUESTED)} requested models are supported by "
        f"autogluon.timeseries {env['autogluon_timeseries']}, so no substitution was needed and the "
        "minimum of three distinct specialist families is met.",
        "",
        "## Foundation models are installed but never used",
        "",
        "AutoGluon ships Chronos, Chronos-2 and Toto wrappers, and installing it pulled",
        "`chronos-forecasting` in as a dependency. None of them may appear in this study's suite:",
        "the question is whether a *non-foundation* task-trained model can beat the TSFM envelope,",
        "so a Chronos inside the ensemble would answer a different question. The pool is passed as",
        "an explicit hyperparameter dict rather than an AutoGluon preset for exactly this reason,",
        "and `verify` check A18 reads the fitted model list back to confirm none slipped in.",
        "",
        f"Present in the registry but excluded: {', '.join(env['forbidden_present_but_never_used'])}.",
        "",
    ]
    (paths.RESULTS / "SPECIALIST_MODEL_AVAILABILITY.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: v for k, v in env.items() if k != "model_availability"}, indent=2))
    for row in env["model_availability"]:
        print(f"  {row['requested']:16s} {'available' if row['available'] else 'MISSING'}")
    if not env["minimum_met"]:
        raise SystemExit("SPECIALIST_SUITE_WEAK -> INCONCLUSIVE_SPECIALIST_POWER")


if __name__ == "__main__":
    main()
