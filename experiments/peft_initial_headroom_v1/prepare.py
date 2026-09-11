"""Prepare Study34 development-only initial-headroom data archives."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from experiments.peft_decision_transfer_v1 import prepare as base
from experiments.peft_external_gap_v1.data import sha256_file
from experiments.peft_fullft_reference_v3 import data as archive_data

ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_initial_headroom_v1"
VERSION = "peft_initial_headroom_v1.prepare.20260911"

EPISODE = "diag"
EPISODE_STARTS = {
    EPISODE: {"jena": datetime(2019, 5, 4), "bmra": datetime(2018, 1, 4)}
}
JENA_FILES = {
    EPISODE: (
        ROOT / "data/jena_mpi_roof/mpi_roof_2019a.csv",
        ROOT / "data/jena_mpi_roof/mpi_roof_2019b.csv",
    )
}

KNOWN_EXPOSURES = {
    "jena": [
        {
            "study": "peft_decision_transfer_v1",
            "path": "runs/peft_decision_transfer_v1/prepared/summary.json",
            "start": "2021-05-04T00:00:00",
            "end_exclusive": "2023-01-01T00:00:00",
            "label_use": "Study33 development/test target labels",
        },
        {
            "study": "peft_contribution_freeze_v1",
            "path": "runs/peft_contribution_freeze_v1/prepared/summary.json",
            "start": "2023-05-04T00:00:00",
            "end_exclusive": "2024-01-01T00:00:00",
            "label_use": "Study31 development diagnostics",
        },
        {
            "study": "peft_adaptation_scope_v1",
            "path": "runs/peft_adaptation_scope_v1/prepared/manifest.json",
            "start": "2024-09-07T00:00:00",
            "end_exclusive": "2025-01-01T00:00:00",
            "label_use": "Jena 2024 validation/evaluation target labels exposed in S1",
        },
    ],
    "bmra": [
        {
            "study": "ucp_path_pilot_v1",
            "path": "data_external/ucp_path_pilot_v1/processed/panel_test.npz",
            "start": "2019-01-01T00:00:00",
            "end_exclusive": "2022-01-01T00:00:00",
            "label_use": "raw source was processed into 2019 train, 2020 val, 2021 test UCP panels",
        },
        {
            "study": "peft_decision_transfer_v1",
            "path": "runs/peft_decision_transfer_v1/prepared/summary.json",
            "start": "2022-01-04T00:00:00",
            "end_exclusive": "2023-09-03T00:00:00",
            "label_use": "Study33 development/test target labels",
        },
    ],
}


def _install_contract_globals() -> None:
    base.STUDY = STUDY
    base.VERSION = VERSION
    base.EPISODE_STARTS = EPISODE_STARTS
    base.JENA_FILES = JENA_FILES
    base.KNOWN_EXPOSURES = KNOWN_EXPOSURES
    archive_data.STUDY = STUDY
    archive_data.VERSION = VERSION
    archive_data.PANEL_STARTS = dict(EPISODE_STARTS[EPISODE])


def _write_json_x(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha256_file(path)


def prepare(output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    audit_path = output_dir / "data_audit.json"
    if summary_path.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing summary/audit in {output_dir}")

    _install_contract_globals()
    resource = base.require_cpu_resources()
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    files: dict[str, dict] = {}
    all_qc_passed = True

    loaders = {"jena": base.load_jena, "bmra": base.load_bmra}
    for panel, loader in loaders.items():
        spec, timestamps, values, loader_metadata = loader(EPISODE)
        if panel == "jena":
            spec = replace(spec, source_notes=(
                "Four-channel order follows Study31/33 Jena parser.",
                "First two columns are scored targets; next two are context-only covariates masked from target loss/scoring.",
                "The 2019 diagnostic block is nonoverlapping with checked local PEFT target-label windows; source/pretraining exposure is not ruled out.",
            ))
        else:
            spec = replace(spec, source_notes=(
                "First two fixed E_ generator columns are scored targets; next two are context-only covariates.",
                "The 2018 diagnostic block is earlier than checked UCP 2019/2020/2021 and Study33 2022/2023 target-label windows; raw/source exposure is not ruled out.",
            ))
        source = base.source_manifest(panel, EPISODE, spec, loader_metadata)
        summary = archive_data.write_panel_archives(panel, timestamps, values, spec.channels, spec.target_indices, output_dir, source)
        spec_summary = base.dataset_manifest(summary)
        data[panel] = spec_summary
        files[f"{panel}_fit"] = {"path": summary["fit_path"], "sha256": summary["fit_sha256"]}
        files[f"{panel}_holdout"] = {"path": summary["holdout_path"], "sha256": summary["holdout_sha256"]}
        min_finite = base.min_target_finite(summary)
        all_qc_passed = all_qc_passed and bool(min_finite >= base.MIN_TARGET_FINITE_FRACTION)
        audits[panel] = {
            "source_qc": loader_metadata,
            "exposure": base.exposure_audit(panel, summary["contract"]),
            "chronology": base.chronology_audit(summary["contract"]),
            "min_target_window_finite_fraction": min_finite,
            "archive_checks": {
                "fit": base.independent_archive_check(ROOT / summary["fit_path"], panel, "fit"),
                "holdout": base.independent_archive_check(ROOT / summary["holdout_path"], panel, "holdout"),
            },
        }

    data_contract = {
        "fit_archives": "train+validation only; physically truncated at validation end with no cal/D origin arrays",
        "holdout_archives": "full 242-day block with calibration/D origins; read only after V selections and correction choices are sealed",
        "context_target_alignment": "origin is first target timestamp; context [origin-336h, origin), target [origin, origin+48h)",
        "origin_stride_hours": base.ORIGIN_STRIDE_HOURS,
        "target_window_overlap_hours_within_split": base.HORIZON - base.ORIGIN_STRIDE_HOURS,
        "between_split_target_overlap": "none across train/V and V/cal due 48h embargo; cal last target end equals D first origin under half-open windows",
        "missing_policy": "target_values preserve raw hourly NaNs; context_values use precontext-only median fallback then causal forward fill; no future backfill and no target imputation",
        "fit_statistics": "mean/std/median fitted from observed train interval rows only",
        "score_columns": "only target_indices [0,1] are scored",
        "diagnostic_subsample_note": "archives keep all 80 D origins; run.py forecasts eval_origins[::4] for the bounded initial screen",
    }
    summary_payload = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": STUDY,
        "episode": EPISODE,
        "all_qc_passed": bool(all_qc_passed),
        "data": data,
        "files": files,
        "data_contract": data_contract,
        "exposure_summary": {
            "claim_limit": "new target-label periods under checked local PEFT histories only; not new raw source families; FM pretraining overlap unknown",
            "selection_limit": "BMRA channel order reuses Study33 and has a pre-run availability screen for 2018; no forecast performance used",
            "checked_previous_sources": KNOWN_EXPOSURES,
        },
        "resource_snapshot": resource,
    }
    summary_sha = _write_json_x(summary_path, summary_payload)
    audit_payload = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": STUDY,
        "summary_path": base.rel(summary_path),
        "summary_sha256": summary_sha,
        "audits": audits,
        "all_qc_passed": bool(all_qc_passed),
        "source_hashes": {
            base.rel(path): sha256_file(path)
            for path in [Path(__file__), Path(__file__).with_name("PURPOSE_DATA.md"), ROOT / "experiments/peft_decision_transfer_v1/panel.py"]
            if path.exists()
        },
    }
    audit_sha = _write_json_x(audit_path, audit_payload)
    summary_payload["summary_path"] = base.rel(summary_path)
    summary_payload["summary_sha256"] = summary_sha
    summary_payload["data_audit_path"] = base.rel(audit_path)
    summary_payload["data_audit_sha256"] = audit_sha
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs") / STUDY / "prepared")
    args = parser.parse_args(argv)
    try:
        summary = prepare(args.output)
    except BaseException as exc:
        base.record_failure(args.output, exc)
        raise
    print(json.dumps({
        "completed": True,
        "summary_path": summary["summary_path"],
        "summary_sha256": summary["summary_sha256"],
        "data_audit_path": summary["data_audit_path"],
        "data_audit_sha256": summary["data_audit_sha256"],
        "all_qc_passed": summary["all_qc_passed"],
        "data": {
            panel: {
                "fit_path": spec["fit_path"],
                "fit_sha256": spec["fit_sha256"],
                "holdout_path": spec["holdout_path"],
                "holdout_sha256": spec["holdout_sha256"],
                "origin_counts": spec["origin_counts"],
            }
            for panel, spec in summary["data"].items()
        },
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
