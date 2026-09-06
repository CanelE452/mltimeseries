"""Drive UCP-PATH-PILOT-v1 from the raw archives to the verdict, in one process.

Every step is judged by the artefact it is supposed to produce, never by an exit
code or by a process still being alive: both have lied before. The run does not
stop at "training finished" -- it carries on through evaluation, the bootstrap,
latency and the pre-registered screens, and its final message is the verdict and
the numbers behind it.

Two interpreters are used on purpose. Extraction and scoring run in the base
environment, which has netCDF4; training and Chronos run in the `mlts`
environment, which has torch and chronos. Neither is disturbed by the other.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = "experiments.uncertain_covariate_path_pilot_v1.src"
DATA = ROOT / "data_external/ucp_path_pilot_v1"
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"

BASE_PY = sys.executable if "envs" not in sys.executable else r"C:\Users\User\anaconda3\python.exe"
MLTS_PY = r"C:\Users\User\anaconda3\envs\mlts\python.exe"

ARCHIVES = {
    "ecmwf_ens_2019-2020.tar.gz": (36034983151, "5bc1122fb5a64bdd9ac980b9f1db2e64"),
    "ecmwf_ens_2021.tar.gz": (18003586760, "24caeb6b55d5f2038b913855f5085b0b"),
}
STALL_MINUTES = 45
POLL_SECONDS = 120


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def start_downloader() -> subprocess.Popen | None:
    """Resume any incomplete archive. Nothing happens if both are already whole."""
    def incomplete(name: str) -> bool:
        path = DATA / "raw" / name
        return not path.exists() or path.stat().st_size < ARCHIVES[name][0]

    todo = [
        key
        for key, name in (("ens_2019_2020", "ecmwf_ens_2019-2020.tar.gz"),
                          ("ens_2021", "ecmwf_ens_2021.tar.gz"))
        if incomplete(name)
    ]
    if not todo:
        log("archives already complete")
        return None
    log(f"starting download for {todo}")
    logfile = open(DATA / "download_ens.log", "a", buffering=1)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [BASE_PY, "-u", "-m", f"{SRC}.data_sources", *todo],
        cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT,
    )


def wait_for_archives() -> None:
    """STEP 0. Completion is decided by bytes on disk plus the recorded md5."""
    sizes, last_change = {}, time.time()
    while True:
        remaining = []
        for name, (size, _) in ARCHIVES.items():
            path = DATA / "raw" / name
            have = path.stat().st_size if path.exists() else 0
            if have != sizes.get(name):
                sizes[name] = have
                last_change = time.time()
            if have < size:
                remaining.append(f"{name} {have / 1e9:.2f}/{size / 1e9:.2f} GB")
        if not remaining:
            break
        if (time.time() - last_change) / 60 > STALL_MINUTES:
            raise RuntimeError(f"DOWNLOAD_STALLED for {STALL_MINUTES} min: {remaining}")
        log("waiting on " + " | ".join(remaining))
        time.sleep(POLL_SECONDS)

    for name, (_, expected) in ARCHIVES.items():
        got = md5_of(DATA / "raw" / name)
        if got != expected:
            raise RuntimeError(f"CHECKSUM_MISMATCH {name}: {got} != {expected}")
        log(f"{name} md5 verified")


def step(name: str, python: str, args: list[str], artefacts: list[Path], rerun: bool = True) -> None:
    """Run one stage and confirm it by the files it should have produced."""
    if not rerun and all(a.exists() for a in artefacts):
        log(f"{name}: artefacts already present, skipping")
        return
    log(f"{name}: starting")
    started = time.time()
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    result = subprocess.run([python, "-u", *args], cwd=ROOT, env=env)
    missing = [str(a) for a in artefacts if not a.exists()]
    if missing:
        raise RuntimeError(f"{name}: missing artefacts {missing} (return code {result.returncode})")
    if result.returncode != 0:
        raise RuntimeError(f"{name}: return code {result.returncode}")
    log(f"{name}: done in {(time.time() - started) / 60:.1f} min")


def main() -> int:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    downloader = start_downloader()
    try:
        wait_for_archives()
    finally:
        if downloader and downloader.poll() is None:
            downloader.terminate()

    step("extract_ens", BASE_PY, ["-m", f"{SRC}.extract_ens"],
         [DATA / "processed/weather_ens_2019_2020.npz", DATA / "processed/weather_ens_2021.npz"])
    step("build_panel", BASE_PY, ["-m", f"{SRC}.build_panel"],
         [DATA / "processed/panel_test.npz", RESULTS / "panel_summary.json"])
    step("cache_chronos", MLTS_PY, ["-m", f"{SRC}.cache_chronos"],
         [DATA / "chronos_cache/chronos_test.npz", RESULTS / "chronos_manifest.json"])

    step("integrity_tests", BASE_PY, ["-m", f"{SRC}.data_contract"],
         [RESULTS / "integrity_tests.json"])
    report = json.loads((RESULTS / "integrity_tests.json").read_text())
    if not report["all_passed"]:
        (RESULTS / "STATUS.md").write_text(
            "# UCP-PATH-PILOT-v1\n\nNOT_EVALUATED_DATA_OR_INTEGRITY: an integrity test failed. "
            "See integrity_tests.json.\n"
        )
        raise RuntimeError("NOT_EVALUATED_DATA_OR_INTEGRITY: integrity tests failed, stopping")
    log("integrity tests T01-T13 all pass")

    step("smoke", MLTS_PY, ["-m", f"{SRC}.smoke"], [RESULTS / "smoke_report.json"])
    smoke = json.loads((RESULTS / "smoke_report.json").read_text())
    if not smoke["all_passed"]:
        raise RuntimeError(f"smoke checks failed: {smoke}")
    log(f"smoke passed; worst-case estimate "
        f"{smoke['runtime_estimate']['worst_case_gpu_hours_21_fits']} GPU-hours for 21 fits")

    step("train", MLTS_PY, ["-m", f"{SRC}.train"], [RESULTS / "model_manifest.json"])
    step("evaluate", MLTS_PY, ["-m", f"{SRC}.evaluate"],
         [RESULTS / "metrics.csv", RESULTS / "per_example_losses.npz"])
    step("latency", MLTS_PY, ["-m", f"{SRC}.latency"], [RESULTS / "latency.csv"])
    step("bootstrap", BASE_PY, ["-m", f"{SRC}.bootstrap"],
         [RESULTS / "bootstrap_primary.json", RESULTS / "seed_effects.csv"])
    step("verdict", BASE_PY, ["-m", f"{SRC}.verdict"], [RESULTS / "verdict.json"])
    step("report", MLTS_PY, ["-m", f"{SRC}.report"],
         [RESULTS / "STATUS.md", RESULTS / "SELF_AUDIT.md"])

    verdict = json.loads((RESULTS / "verdict.json").read_text())
    elapsed = (time.time() - started) / 3600
    log("=" * 72)
    log(f"FINAL TOKEN: {verdict['final_token']}")
    for name, check in verdict["primary_contrasts"].items():
        log(f"  {name}: RI {check['ri_percent']:+.2f}% "
            f"[{check['ci_lower']:+.2f}, {check['ci_upper']:+.2f}] "
            f"seeds positive {check['seeds_positive']['n_positive']}/{check['seeds_positive']['n_seeds']}")
    log(f"  screens: {verdict['screens']}")
    audit = (RESULTS / "SELF_AUDIT.md").read_text()
    marker = "All checks passed: **"
    passed = audit.split(marker)[1][:6] if marker in audit else "unknown"
    log(f"  self audit all passed: {passed.strip('*')}")
    log(f"wall clock {elapsed:.2f} h")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
