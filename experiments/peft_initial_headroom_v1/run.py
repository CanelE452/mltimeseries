"""Run the Study34 development-only initial-headroom diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded
from experiments.peft_optimization_control_v1.run import sha, save

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_initial_headroom_v1"
CODE = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(plan: dict) -> None:
    for path, expected in {**plan["source_hashes"], **plan["input_hashes"]}.items():
        assert sha(ROOT / path) == expected, path


def prepare_plan() -> None:
    assert not (RUN / "plan.json").exists()
    data = read_json(RUN / "prepared/summary.json")
    previous = read_json(ROOT / "runs/peft_decision_transfer_v1/plan.json")
    sources = dict(previous["source_hashes"])
    for path in [
        *(p for p in CODE.glob("*.py") if not p.name.startswith('independent_')),
        *(p for p in CODE.glob("*.md")),
        ROOT / "experiments/peft_decision_transfer_v1/panel.py",
        ROOT / "experiments/peft_capacity_probe_v1/model.py",
    ]:
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    inputs = {"runs/peft_initial_headroom_v1/prepared/summary.json": sha(RUN / "prepared/summary.json")}
    for spec in data["data"].values():
        for role in ("fit", "holdout"):
            inputs[spec[f"{role}_path"]] = spec[f"{role}_sha256"]

    head_lrs = [1e-5, 3e-5, 1e-4, 3e-4]
    joint_recipes = [
        {"head_lr": 1e-5, "lora_lr": 1e-5},
        {"head_lr": 3e-5, "lora_lr": 3e-5},
        {"head_lr": 1e-4, "lora_lr": 3e-5},
        {"head_lr": 1e-4, "lora_lr": 1e-4},
    ]
    jobs = []
    for seed in (29000, 29001):
        for dataset in sorted(data["data"]):
            for condition in ("FULL90", "SPREAD30"):
                rows = list(range(90)) if condition == "FULL90" else list(range(0, 90, 3))
                for family in ("HEAD", "WIDE", "JOINT"):
                    for recipe in range(4):
                        if family == "JOINT":
                            rates = joint_recipes[recipe]
                        else:
                            rates = {"head_lr": head_lrs[recipe], "lora_lr": 0.0}
                        job = {
                            "dataset": dataset,
                            "condition": condition,
                            "seed": seed,
                            "family": family,
                            "recipe": recipe,
                            "arm": family,
                            "head": "mlp",
                            "rank": 8,
                            "blocks": list(range(12)) if family == "JOINT" else [],
                            "training_rows": rows,
                            **rates,
                        }
                        key = f"{dataset}/{condition}/{family}/r{recipe}/s{seed}"
                        jobs.append({"key": key, "job": job})
    plan = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": "peft_initial_headroom_v1",
        "run_dir": RUN.relative_to(ROOT).as_posix(),
        "source_hashes": sources,
        "input_hashes": inputs,
        "data": data["data"],
        "checkpoint": previous["checkpoint"],
        "jobs": jobs,
        "schedules": {
            "FULL90": [0, 1, 2, 4, 8, 15, 30, 60, 120, 180],
            "SPREAD30": [0, 1, 2, 4, 5, 10, 20, 40, 60],
        },
        "eval_origin_subsample_stride": 4,
        "gate_band_pct_F0": 0.25,
        "D_is_development_diagnostic": True,
        "no_final_test_labels_prepared": True,
    }
    verify(plan)
    save(RUN / "plan.json", plan)
    print(json.dumps({"prepared": True, "jobs": len(jobs)}, indent=2), flush=True)


def execute(plan: dict, index: int, stage: str, smoke: bool = False) -> None:
    entry = plan["jobs"][index]
    key = f"smoke/{entry['job']['family']}" if smoke else f"{stage}/{entry['key']}"
    parent = RUN / key
    if not parent.exists():
        admission(RUN, key)
        command = [
            sys.executable,
            "-m",
            "experiments.peft_initial_headroom_v1.fit",
            "--plan",
            str(RUN / "plan.json"),
            "--index",
            str(index),
            "--output",
            str(parent / "output"),
            "--stage",
            stage,
        ]
        if smoke:
            command.append("--smoke")
        print(json.dumps({"starting": key}, ensure_ascii=False), flush=True)
        status = run_guarded(command, parent / "guard", ROOT, 900, require_gpu=True)
        assert status["completed"] and not status["reasons"], f"Failure preserved: {key}"
    status = read_json(parent / "guard/status.json")
    result = read_json(parent / "output/result.json")
    assert status["completed"] and status["returncode"] == 0 and not status["reasons"], key
    assert result["completed"] and result["plan_sha256"] == sha(RUN / "plan.json") and result["job"] == entry["job"], key


def seal_selection(plan: dict) -> None:
    if (RUN / "selection_sealed.json").exists():
        seal = read_json(RUN / "selection_sealed.json")
        assert seal["plan_sha256"] == sha(RUN / "plan.json")
        return
    from experiments.peft_initial_headroom_v1.analyse import summarize_sealed_fit, write_json

    payload = summarize_sealed_fit(plan)
    payload["sealed_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload["plan_sha256"] = sha(RUN / "plan.json")
    write_json(RUN / "selection_sealed.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        from experiments.peft_initial_headroom_v1.prepare import prepare

        if not (RUN / "prepared/summary.json").exists():
            prepare(RUN / "prepared")
        prepare_plan()
        return 0

    assert not (RUN / "completed.json").exists()
    plan = read_json(RUN / "plan.json")
    verify(plan)
    started = time.perf_counter()

    smoke_indices = []
    first_cell = plan["jobs"][0]["job"]
    for family in ("HEAD", "WIDE", "JOINT"):
        smoke_indices.append(next(
            i for i, entry in enumerate(plan["jobs"])
            if entry["job"]["dataset"] == first_cell["dataset"]
            and entry["job"]["condition"] == first_cell["condition"]
            and entry["job"]["seed"] == first_cell["seed"]
            and entry["job"]["family"] == family
            and entry["job"]["recipe"] == 0
        ))
    for index in smoke_indices:
        execute(plan, index, "fit", smoke=True)
    if args.smoke_only:
        return 0

    for index in range(len(plan["jobs"])):
        execute(plan, index, "fit")
        save(RUN/'progress.json', {'stage': 'fit', 'done': index+1, 'total': len(plan['jobs']), 'seconds': time.perf_counter()-started})
    seal_selection(plan)
    if args.fit_only:
        return 0

    seal = read_json(RUN / "selection_sealed.json")
    for count, index in enumerate(seal["forecast_indices"], 1):
        execute(plan, index, "forecast")
        save(RUN/'progress.json', {'stage': 'forecast_D', 'done': count, 'total': len(seal['forecast_indices']), 'seconds': time.perf_counter()-started})
    verify(plan)
    save(RUN / "completed.json", {
        "completed": True,
        "finished_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seconds": time.perf_counter() - started,
        "plan_sha256": sha(RUN / "plan.json"),
        "fit_jobs": len(plan["jobs"]),
        "forecast_jobs": len(seal["forecast_indices"]),
        "meaning": "GPU execution complete; analysis has a separate summary.json receipt",
    })
    from experiments.peft_initial_headroom_v1.analyse import analyse
    analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
