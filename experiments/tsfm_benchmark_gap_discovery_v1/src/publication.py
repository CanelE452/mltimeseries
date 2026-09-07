"""Pre-push safety scan and the remote publication manifest (Section 38A).

Two modes:

    python -m ...publication scan     before committing: refuse on anything that
                                      must not leave this machine
    python -m ...publication verify   after pushing: read the remote back and
                                      record what is actually there
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
from datetime import datetime, timezone

from . import paths

BRANCH = "tsfm-benchmark-gap-discovery-v1"

FORBIDDEN_PREFIXES = ("data_external/", "runs/", "checkpoints/")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key")
SECRET_NAME_PATTERN = re.compile(
    r"(^|/)(\.env|.*credential.*|.*secret.*|.*password.*|id_rsa.*|.*\.pem|.*\.key)$", re.I
)
SECRET_CONTENT_PATTERN = re.compile(
    r"(aws_secret_access_key|BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,})"
)

# The single deliberate exception to the data_external rule: the fev-bench task
# definition is the frozen evaluation contract this study is judged against, not raw
# data. It is 48 KB of YAML, fetched from a pinned upstream commit, and its sha256 is
# re-checked on every run. Tracking it means an auditor can read the contract without
# trusting a download.
TRACKED_EXCEPTIONS = {
    "data_external/tsfm_benchmark_gap_discovery_v1/fev_bench_tasks.yaml",
}

AUDIT_CRITICAL = [
    "results/tsfm_benchmark_gap_discovery_v1/BASE_STATE.json",
    "results/tsfm_benchmark_gap_discovery_v1/model_audit.csv",
    "results/tsfm_benchmark_gap_discovery_v1/benchmark_audit.json",
    "results/tsfm_benchmark_gap_discovery_v1/contamination_matrix.csv",
    "results/tsfm_benchmark_gap_discovery_v1/fairness_matrix.csv",
    "results/tsfm_benchmark_gap_discovery_v1/task_pool.csv",
    "results/tsfm_benchmark_gap_discovery_v1/selected_tasks.json",
    "results/tsfm_benchmark_gap_discovery_v1/selected_tasks.sha256",
    "results/tsfm_benchmark_gap_discovery_v1/task_metadata.csv",
    "results/tsfm_benchmark_gap_discovery_v1/smoke_report.json",
    "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    "results/tsfm_benchmark_gap_discovery_v1/baseline_results.csv",
    "results/tsfm_benchmark_gap_discovery_v1/failure_map.csv",
    "results/tsfm_benchmark_gap_discovery_v1/failure_map.md",
    "results/tsfm_benchmark_gap_discovery_v1/verdict.json",
    "results/tsfm_benchmark_gap_discovery_v1/STATUS.md",
    "results/tsfm_benchmark_gap_discovery_v1/SELF_AUDIT.md",
    "results/tsfm_benchmark_gap_discovery_v1/integrity_checks.csv",
    "results/tsfm_benchmark_gap_discovery_v1/MODEL_SUBSTITUTION.md",
    "results/tsfm_benchmark_gap_discovery_v1/RESULT_EXCLUSION.md",
]

CONDITIONAL_CRITICAL = [
    "results/tsfm_benchmark_gap_discovery_v1/failure_candidates.json",
    "results/tsfm_benchmark_gap_discovery_v1/candidate_probe_spec.json",
    "results/tsfm_benchmark_gap_discovery_v1/candidate_probe_spec.sha256",
    "results/tsfm_benchmark_gap_discovery_v1/probe_results.csv",
    "results/tsfm_benchmark_gap_discovery_v1/headroom_table.csv",
    "results/tsfm_benchmark_gap_discovery_v1/confirmation_results.csv",
    "results/tsfm_benchmark_gap_discovery_v1/literature_audit.md",
    "results/tsfm_benchmark_gap_discovery_v1/candidate_ranking.csv",
    "results/tsfm_benchmark_gap_discovery_v1/NEXT_METHOD_BRIEF.md",
]


def stage_origin_losses() -> dict:
    """Copy the per-origin loss archives into results/ so they can be committed.

    Section 38A.2 allows a small audit-critical binary to be tracked when it holds
    only prediction losses, identifiers and timestamps, carries no raw target data
    and stays within a sensible size. These arrays are exactly that, and without
    them an outside reader cannot recompute the oracle, the validation-selected
    model or the bootstrap interval. The raw quantile forecasts, which are 169 MB,
    stay untracked and are regenerated from the committed code plus the pinned
    checkpoints.
    """
    import hashlib
    import shutil

    destination = paths.RESULTS / "origin_losses"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    total = 0
    for source in sorted(paths.PRED_CACHE.glob("*__origins.npz")):
        target = destination / source.name
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest[source.name] = {"sha256": digest, "bytes": target.stat().st_size}
        total += target.stat().st_size
    payload = {
        "description": (
            "Per-origin scaled quantile losses for every (model, track, task) cell. Arrays: "
            "origin_loss (float32) and origin_key ('w{window}::{series id}'). No target values, "
            "no covariates, no raw benchmark data."
        ),
        "n_files": len(manifest),
        "total_bytes": total,
        "files": manifest,
    }
    (destination / "MANIFEST.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True
    ).stdout.strip()


def scan() -> dict:
    tracked = [line for line in git("ls-files").splitlines() if line]
    findings: dict = {"blockers": [], "warnings": []}

    for path in tracked:
        if path in TRACKED_EXCEPTIONS:
            continue
        if path.startswith(FORBIDDEN_PREFIXES) or path.endswith(FORBIDDEN_SUFFIXES):
            findings["blockers"].append(f"forbidden tracked path: {path}")
        if SECRET_NAME_PATTERN.search(path):
            findings["blockers"].append(f"credential-shaped filename tracked: {path}")

    large = []
    for path in tracked:
        full = paths.REPO / path
        if full.exists():
            size = full.stat().st_size
            if size > 10 * 1024**2:
                large.append((path, size))
    for path, size in large:
        findings["blockers"].append(f"tracked file over 10 MB: {path} ({size / 1024**2:.1f} MB)")

    # This file holds the detection patterns as string literals, so scanning it
    # matches them by construction. It is the scanner; excluding it is not a blind
    # spot, and every other tracked text file is still read.
    self_path = str(pathlib.Path(__file__).resolve().relative_to(paths.REPO)).replace("\\", "/")

    scanned = 0
    for path in tracked:
        if path == self_path:
            continue
        full = paths.REPO / path
        if not full.exists() or full.stat().st_size > 2 * 1024**2:
            continue
        if full.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npz", ".parquet"}:
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        scanned += 1
        if SECRET_CONTENT_PATTERN.search(text):
            # Report the file, never the matched value.
            findings["blockers"].append(f"credential-shaped content in {path}")

    base = json.loads((paths.RESULTS / "BASE_STATE.json").read_text(encoding="utf-8"))
    if git("rev-parse", "origin/main") != base["origin_main_sha"]:
        findings["blockers"].append("origin/main moved since the study started")

    missing = [p for p in AUDIT_CRITICAL if not (paths.REPO / p).exists()]
    if missing:
        findings["blockers"].append(f"audit-critical artifacts missing: {missing}")
    untracked_critical = []
    for path in AUDIT_CRITICAL + CONDITIONAL_CRITICAL:
        if (paths.REPO / path).exists() and path not in tracked:
            untracked_critical.append(path)
    if untracked_critical:
        findings["warnings"].append(
            f"audit-critical artifacts present but not yet staged: {untracked_critical}"
        )

    findings["n_tracked_files"] = len(tracked)
    findings["n_text_files_scanned"] = scanned
    findings["content_scan_skipped"] = [self_path]
    findings["largest_tracked_files"] = sorted(
        (
            (p, (paths.REPO / p).stat().st_size)
            for p in tracked
            if (paths.REPO / p).exists()
        ),
        key=lambda item: -item[1],
    )[:10]
    findings["large_file_scan_passed"] = not large
    findings["secret_scan_passed"] = not [
        b for b in findings["blockers"] if "credential" in b
    ]
    findings["passed"] = not findings["blockers"]
    return findings


def verify_remote() -> dict:
    base = json.loads((paths.RESULTS / "BASE_STATE.json").read_text(encoding="utf-8"))
    remote_branch = git("ls-remote", "origin", f"refs/heads/{BRANCH}")
    remote_main = git("ls-remote", "origin", "refs/heads/main")
    remote_tip = remote_branch.split()[0] if remote_branch else None
    main_now = remote_main.split()[0] if remote_main else None
    local_head = git("rev-parse", "HEAD")
    findings = scan()
    return {
        "repo": "CanelE452/mltimeseries",
        "branch": BRANCH,
        "local_head": local_head,
        "remote_tip": remote_tip,
        "remote_matches_local": remote_tip == local_head,
        "origin_main_at_start": base["origin_main_sha"],
        "origin_main_after_push": main_now,
        "main_unchanged": main_now == base["origin_main_sha"],
        "pushed_at": datetime.now(timezone.utc).isoformat(),
        "push_status": "VERIFIED" if remote_tip == local_head else "PENDING_REMOTE_VERIFY",
        "audit_critical_files_tracked": [
            p for p in AUDIT_CRITICAL + CONDITIONAL_CRITICAL if p in git("ls-files").splitlines()
        ],
        "deliberately_tracked_exceptions": sorted(TRACKED_EXCEPTIONS),
        "intentionally_untracked_artifacts": [
            "runs/tsfm_benchmark_gap_discovery_v1/** - model weights, HF caches and raw quantile "
            "forecasts; regenerable from the committed code plus the pinned checkpoints",
            "data_external/tsfm_benchmark_gap_discovery_v1/** - the fev-bench task definition and "
            "the paper HTML it was parsed from; both are fetched by task_pool.py from the pinned "
            "commit and the sha256 is checked on every run",
        ],
        "large_file_scan_passed": findings["large_file_scan_passed"],
        "secret_scan_passed": findings["secret_scan_passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["scan", "verify"])
    args = parser.parse_args()
    if args.mode == "scan":
        staged = stage_origin_losses()
        print(f"staged {staged['n_files']} origin-loss archives ({staged['total_bytes'] / 1024:.0f} KB)")
        findings = scan()
        print(json.dumps(findings, indent=2))
        if not findings["passed"]:
            raise SystemExit("PUSH_BLOCKED: " + "; ".join(findings["blockers"]))
        print("\nscan passed")
    else:
        manifest = verify_remote()
        (paths.RESULTS / "REMOTE_PUBLICATION.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
