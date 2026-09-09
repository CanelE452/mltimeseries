"""Verify the completed CPU entry artifacts without changing frozen results."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .run import ROOT, OUT, RUN, sha, verify_contract, write_json


def main():
    contract = verify_contract()
    independent = json.loads((OUT / "independent_audit.json").read_text(encoding="utf-8"))
    assert independent["status"] == "PASS" and not independent["failures"]
    assert sha(Path(__file__).with_name("independent_audit.py")) == independent["source_sha256"]
    for name, expected in independent["input_hashes"].items():
        assert sha(ROOT / name) == expected, name
    post = json.loads((RUN / "posthoc_contract.json").read_text(encoding="utf-8"))
    assert sha(Path(__file__).with_name("diagnose.py")) == post["source_sha256"]
    assert sha(ROOT / "_docs/notes/tsfm_topics/19c_revision_posthoc_diagnostic_20260908.md") == post["plan_sha256"]
    assert sha(OUT / "cpu_results.json") == post["original_results_sha256"]
    guard_paths = sorted(RUN.rglob("status.json"))
    assert len(guard_paths) == 7
    expected_failures = {"fetch_guard/status.json", "fetch_guard/diagnostic_small/status.json"}
    statuses, samples = [], []
    for path in guard_paths:
        status = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(RUN).as_posix()
        assert not status["reasons"], relative
        if relative in expected_failures:
            assert status["state"] == "child_failed" and status["returncode"] == 1
            assert "TimeoutError" in (path.parent / "stderr.log").read_text(encoding="utf-8")
        else:
            assert status["completed"] and status["returncode"] == 0, relative
        statuses.append({"path": str(path.relative_to(ROOT)), "elapsed_seconds": status["elapsed_seconds"],
                         "state": status["state"], "returncode": status["returncode"]})
        for line in (path.parent / "resource_log.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "available_ram_gib" in row:
                samples.append(row)
    assert not (ROOT / "runs/peft_adaptation_scope_v1/.guard.lock").exists()
    windows = json.loads((OUT / "windows_events.json").read_text(encoding="utf-8-sig"))
    assert not windows["query_errors"]
    report = ROOT / "_docs/notes/tsfm_topics/19_peft_revision_entry_results_20260908.md"
    local_links = []
    for name in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
        if name.startswith("http"):
            continue
        target = (report.parent / name).resolve()
        assert target.is_file(), target
        local_links.append(str(target))
    artifacts = {}
    for directory in (RUN, OUT, ROOT / "data_external/alfred_revision_entry_v1"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                artifacts[str(path.relative_to(ROOT))] = sha(path)
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        artifacts[str(path.relative_to(ROOT))] = sha(path)
    result = {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_count": len(contract["protected_hashes"]), "frozen_core_sources": len(contract["source_hashes"]),
        "frozen_input_files": len(contract["input_hashes"]), "guards": statuses,
        "guard_wall_seconds_sum": sum(row["elapsed_seconds"] for row in statuses),
        "expected_transport_failures": 2, "resource_violation_stops": 0, "gpu_training_runs": 0,
        "resource_logged_samples": len(samples),
        "minimum_logged_available_ram_gib": min(row["available_ram_gib"] for row in samples),
        "minimum_logged_available_commit_gib": min(row["available_commit_gib"] for row in samples),
        "resource_scope": "Logged samples only, not guaranteed child/browser peaks. Browser UI downloads were outside child guard.",
        "windows_event_count": windows["count"], "windows_query_errors": [],
        "report_sha256": sha(report), "report_local_links": len(local_links),
        "independent_audit_sha256": sha(OUT / "independent_audit.json"), "artifact_hashes": artifacts,
        "original_screen": "PRIORITIZE_FM_DIAGNOSTIC", "research_decision": "DEFER_NEW_PEFT_GPU_ON_CURRENT_TWO_SERIES",
    }
    write_json(OUT / "final_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_hashes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
