"""Summarize executed guards and verify the completed study17 evidence."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from .fetch import ROOT, STUDY, read, sha, write_once
from . import prepare
from .run_study import verify_result, verify_selection, guard_ok


def resources():
    run = ROOT / "runs" / STUDY
    paths = [run / name / "status.json" for name in ("data_entry_guard", "data_build_guard", "plot_guard")]
    paths += [run / "guards" / name / "status.json" for name in ("cache", "ridge", "smoke", "train", "forecast", "analysis")]
    statuses = {str(path.relative_to(ROOT)): guard_ok(path) for path in paths}
    samples = []
    for path in paths:
        with (path.parent / "resource_log.jsonl").open(encoding="utf-8") as stream:
            samples += [row for line in stream if (row := __import__("json").loads(line)).get("available_ram_gib") is not None]
    gpu = [device for row in samples for device in (row.get("gpus") or [])]
    summary = {"completed": True, "guard_count": len(statuses), "guard_seconds": sum(s["elapsed_seconds"] for s in statuses.values()),
               "guards": statuses, "resource_sample_count": len(samples),
               "minimum_available_ram_gib": min(s["available_ram_gib"] for s in samples),
               "minimum_available_commit_gib": min(s["available_commit_gib"] for s in samples if s["available_commit_gib"] is not None),
               "maximum_child_rss_gib": max(s["child_tree_rss_gib"] for s in samples if s["child_tree_rss_gib"] is not None),
               "maximum_git_processes": max(s["git_process_count"] for s in samples),
               "maximum_observed_gpu_mib": max(s["memory_used_mib"] for s in gpu),
               "maximum_observed_gpu_temperature_c": max(s["temperature_c"] for s in gpu),
               "scope": "Periodic guard observations, not exact system-wide peaks; short CPU stages may only have startup samples"}
    write_once(ROOT / "results" / STUDY / "resources.json", summary)
    return summary


def run():
    contract = prepare.validate()
    verify_selection(contract)
    for stage in ("cache", "ridge", "smoke", "train", "forecast"):
        verify_result(contract, stage)
    result_dir = ROOT / "results" / STUDY
    summary = resources()
    metrics = read(result_dir / "metrics.json")
    if not metrics["completed"] or metrics["novel_method_demonstrated"] is not False:
        raise AssertionError("Wrong claim for a standard-LoRA screen")
    for filename in ("ridge_independent_audit.json", "ridge_contrast_independent_audit.json", "independent_evaluation_audit.json"):
        audit = read(result_dir / filename)
        if not (audit.get("passed") or audit.get("verdict") == "PASS"):
            raise AssertionError(f"Independent audit failed: {filename}")
        for key in ("input_hashes_before_after", "audited_artifact_sha256"):
            if key in audit:
                prepare.verify(ROOT, audit[key])
    prepared = read(ROOT / "runs" / STUDY / "data/prepared_data_audit.json")
    if prepared["verdict"] != "PASS":
        raise AssertionError("Prepared data audit failed")
    prepare.verify(ROOT, prepared["protected_file_sha256"])
    figure = read(result_dir / "figures/manifest.json")
    prepare.verify(ROOT, figure["figure_hashes"])
    if figure["metrics_sha256"] != sha(result_dir / "metrics.json"):
        raise AssertionError("Figure uses different metrics")
    report = ROOT / "_docs/notes/tsfm_topics/17_coarse_supervision_results_20260908.md"
    text = report.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", text)
    local = [link for link in links if not link.startswith(("http", "#"))]
    if any(not (report.parent / link).resolve().exists() for link in local):
        raise AssertionError("Report contains a missing local artifact")
    for path in (ROOT / "runs/peft_adaptation_scope_v1/.guard.lock", ROOT / "runs" / STUDY / ".runner.lock"):
        if path.exists():
            raise AssertionError(f"A runner/guard lock remains: {path}")
    artifacts = {}
    for folder in (ROOT / "runs" / STUDY, result_dir):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "final_audit.json":
                artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    for filename in ("plot.py", "finalize.py", "README.md"):
        path = ROOT / "experiments" / STUDY / filename
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    final = {"passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "contract_sha256": sha(ROOT / "runs" / STUDY / "study_contract.json"),
             "protected_count": len(contract["protected_hashes"]), "core_source_count": len(contract["source_hashes"]),
             "runtime_count": len(contract["runtime_hashes"]), "report_path": str(report), "report_sha256": sha(report),
             "local_report_links": len(local), "artifact_hashes": artifacts,
             "guard_count": summary["guard_count"], "guard_seconds": summary["guard_seconds"],
             "decision": metrics["decision"], "novel_method_demonstrated": False,
             "windows_events": read(result_dir / "windows_events.json")["count"]}
    write_once(result_dir / "final_audit.json", final)
    return {key: value for key, value in final.items() if key != "artifact_hashes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources-only", action="store_true")
    print(resources() if parser.parse_args().resources_only else run())
