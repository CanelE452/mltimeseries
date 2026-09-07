"""Pre-push safety scan and the remote publication manifest (Sections 41-42)."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
from datetime import datetime, timezone

from . import paths

FORBIDDEN_PREFIXES = ("runs/", "checkpoints/")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key")
SECRET_NAME_PATTERN = re.compile(
    r"(^|/)(\.env|.*credential.*|.*secret.*|.*password.*|id_rsa.*|.*\.pem|.*\.key)$", re.I
)
SECRET_CONTENT_PATTERN = re.compile(
    r"(aws_secret_access_key|BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,})"
)

# The source study tracks the fev-bench task definition for the same reason: it is
# the frozen evaluation contract, not raw data.
TRACKED_EXCEPTIONS = {
    "data_external/tsfm_benchmark_gap_discovery_v1/fev_bench_tasks.yaml",
}

AUDIT_CRITICAL = [
    "BASE_STATE.json",
    "SOURCE_STUDY.json",
    "long_horizon_tasks.json",
    "long_horizon_tasks.sha256",
    "SPECIALIST_MODEL_AVAILABILITY.md",
    "specialist_env_plan.json",
    "specialist_env.json",
    "specialist_manifest.csv",
    "development_tsfm_results.csv",
    "holdout_tsfm_results.csv",
    "specialist_scores.csv",
    "specialist_component_results.csv",
    "comparator_scores.csv",
    "comparison_effects.csv",
    "aggregate_effects.csv",
    "fairness_matrix.csv",
    "integrity_checks.csv",
    "verdict.json",
    "STATUS.md",
    "SELF_AUDIT.md",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True
    ).stdout.strip()


def stage_origin_losses() -> dict:
    """Copy the compact per-origin loss archives into results/ so they can be committed.

    Section 40 asks for the small audit-critical arrays and forbids raw targets.
    These hold scaled losses, a window index and an item id, and nothing else. The
    AutoGluon predictor directories and the raw quantile forecasts stay in runs/.
    """
    destination = paths.RESULTS / "origin_losses"
    destination.mkdir(parents=True, exist_ok=True)
    manifest, total = {}, 0
    for source in sorted(paths.RUNS.glob("origins__*.npz")):
        target = destination / source.name
        shutil.copy2(source, target)
        manifest[source.name] = {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
        total += target.stat().st_size
    payload = {
        "description": (
            "Per-origin scaled quantile losses for every comparator on every task. Arrays: "
            "origin_loss (float32), origin_window (int16), origin_key. No target values, no "
            "covariates, no raw benchmark data."
        ),
        "n_files": len(manifest),
        "total_bytes": total,
        "files": manifest,
    }
    (destination / "MANIFEST.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def scan() -> dict:
    tracked = [line for line in git("ls-files").splitlines() if line]
    findings: dict = {"blockers": [], "warnings": []}
    # Any file that defines these detection patterns matches them by construction:
    # this scanner and the source study's, which lives in the same repository. Both
    # are skipped by name, not by a wildcard, so a new file cannot inherit the pass.
    skip = {
        str(pathlib.Path(__file__).resolve().relative_to(paths.REPO)).replace("\\", "/"),
        "experiments/tsfm_benchmark_gap_discovery_v1/src/publication.py",
    }

    for path in tracked:
        if path in TRACKED_EXCEPTIONS:
            continue
        if path.startswith(FORBIDDEN_PREFIXES) or path.endswith(FORBIDDEN_SUFFIXES):
            findings["blockers"].append(f"forbidden tracked path: {path}")
        if path.startswith("data_external/"):
            findings["blockers"].append(f"raw data tracked: {path}")
        if SECRET_NAME_PATTERN.search(path):
            findings["blockers"].append(f"credential-shaped filename tracked: {path}")

    large = []
    for path in tracked:
        full = paths.REPO / path
        if full.exists() and full.stat().st_size > 10 * 1024**2:
            large.append((path, full.stat().st_size))
    for path, size in large:
        findings["blockers"].append(f"tracked file over 10 MB: {path} ({size / 1024**2:.1f} MB)")

    scanned = 0
    for path in tracked:
        if path in skip:
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
            findings["blockers"].append(f"credential-shaped content in {path}")

    base = json.loads((paths.RESULTS / "BASE_STATE.json").read_text(encoding="utf-8"))
    main_now = git("ls-remote", "origin", "refs/heads/main").split()
    if (main_now[0] if main_now else None) != base["origin_main_sha"]:
        findings["blockers"].append("ORIGIN_MAIN_MOVED since the closure started")

    source_changed = git(
        "diff", "--name-only", base["base_sha"], "HEAD", "--",
        "results/tsfm_benchmark_gap_discovery_v1",
        "experiments/tsfm_benchmark_gap_discovery_v1",
        "_docs/notes/tsfm_topics/benchmark_gap_discovery_v1",
    )
    if source_changed.strip():
        findings["blockers"].append(f"SOURCE_RESULTS_MODIFIED: {source_changed.splitlines()}")

    payload = (paths.RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "long_horizon_tasks.sha256").read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        findings["blockers"].append("HOLDOUT_SELECTION_NOT_FROZEN: long_horizon_tasks hash changed")

    missing = [f for f in AUDIT_CRITICAL if not (paths.RESULTS / f).exists()]
    if missing:
        findings["blockers"].append(f"audit-critical results missing: {missing}")

    findings.update(
        {
            "n_tracked_files": len(tracked),
            "n_text_files_scanned": scanned,
            "content_scan_skipped": sorted(skip),
            "large_file_scan_passed": not large,
            "secret_scan_passed": not [b for b in findings["blockers"] if "credential" in b],
            "holdout_hash_frozen": digest == recorded,
            "passed": not findings["blockers"],
        }
    )
    return findings


def verify_remote() -> dict:
    base = json.loads((paths.RESULTS / "BASE_STATE.json").read_text(encoding="utf-8"))
    remote = git("ls-remote", "origin", f"refs/heads/{paths.BRANCH}").split()
    main_now = git("ls-remote", "origin", "refs/heads/main").split()
    findings = scan()
    head = git("rev-parse", "HEAD")
    return {
        "repo": "CanelE452/mltimeseries",
        "branch": paths.BRANCH,
        "local_head": head,
        "remote_tip": remote[0] if remote else None,
        "remote_matches_local": (remote[0] if remote else None) == head,
        "origin_main_at_start": base["origin_main_sha"],
        "origin_main_after_push": main_now[0] if main_now else None,
        "main_unchanged": (main_now[0] if main_now else None) == base["origin_main_sha"],
        "source_branch_sha": base["source_branch_sha"],
        "pushed_at": datetime.now(timezone.utc).isoformat(),
        "push_status": "VERIFIED" if (remote and remote[0] == head) else "PENDING_REMOTE_VERIFY",
        "audit_files_tracked": [
            f"results/tsfm_long_horizon_specialist_closure_v1/{f}"
            for f in AUDIT_CRITICAL
            if f"results/tsfm_long_horizon_specialist_closure_v1/{f}" in git("ls-files").splitlines()
        ],
        "intentionally_untracked": [
            "runs/tsfm_long_horizon_specialist_closure_v1/predictors/** - AutoGluon predictor "
            "directories with fitted model weights; regenerable from the committed code and the "
            "frozen task list",
            "runs/tsfm_long_horizon_specialist_closure_v1/forecasts/** - raw quantile forecasts",
            "data_external/tsfm_long_horizon_specialist_closure_v1/** - exported window arrays, "
            "which contain benchmark target values and are not redistributable",
        ],
        "secret_scan_passed": findings["secret_scan_passed"],
        "large_file_scan_passed": findings["large_file_scan_passed"],
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
