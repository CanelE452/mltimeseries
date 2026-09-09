"""Sequential, resumable S0/S1 execution for the registered PEFT screen."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import psutil

from .guard import run_guarded


PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "runs/peft_adaptation_scope_v1"
CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
GRIDS = {
    "AFF": [1e-3, 3e-3, 1e-2],
    "H_LIN": [1e-4, 3e-4, 1e-3],
    "H_MLP": [1e-4, 3e-4, 1e-3],
    "H_FULL": [3e-5, 1e-4, 3e-4],
    "OFF_LORA": [1e-5, 3e-5, 1e-4],
    "FULL": [1e-6, 3e-6, 1e-5],
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_trial(panel, method, lr, seed, args, smoke=False):
    name = f"{panel}_{method}_lr{lr:g}_seed{seed}"
    if smoke:
        name += f"_ampnocache_micro{args.micro_groups}"
    destination = RUN / ("preflight" if smoke else "trials") / name
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("completed") is True:
            print(f"RESUME {name}", flush=True)
            return existing
        raise RuntimeError(f"Incomplete trial requires inspection before retry: {destination}")
    command = [
        sys.executable, "-u", "-m", "experiments.peft_adaptation_scope_v1.train",
        "--data", str(RUN / "prepared" / f"{panel}.npz"),
        "--output", str(destination), "--method", method,
        "--lr", str(lr), "--seed", str(seed),
        "--steps", str(5 if smoke else (0 if method == "F0" else args.steps)),
        "--eval-every", str(5 if smoke else 50),
        "--effective-groups", "8", "--micro-groups", str(args.micro_groups),
        "--checkpoint", str(CHECKPOINT), "--device", "cuda",
        "--cache", str(RUN / ("smoke_cache" if smoke else "cache") / panel),
    ]
    if smoke:
        command.append("--smoke")
    save_json(RUN / "progress.json", {"status": "running", "trial": name, "stage": "smoke" if smoke else "s1", "started": time.time(), "pid": os.getpid()})
    print(f"START {name}", flush=True)
    outcome = run_guarded(command, destination / "guard", PROJECT, timeout_seconds=args.trial_timeout)
    if not metrics_path.exists():
        raise RuntimeError(f"No successful metrics for {name}: {outcome}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("completed") is not True or outcome.get("completed") is not True or outcome.get("returncode") != 0:
        raise RuntimeError(f"Failed or unsafe trial {name}: {outcome}")
    print(f"DONE {name} val={metrics['val_score']:.6f} step={metrics['selected_step']}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "s1"], required=True)
    parser.add_argument("--panels", nargs="+", default=["ettm2", "jena"])
    parser.add_argument("--micro-groups", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trial-timeout", type=int, default=7200)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONIOENCODING": "utf-8"})
    lock = RUN / "runner.lock"
    if lock.exists():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        try:
            owner = psutil.Process(previous["pid"])
            if abs(owner.create_time() - previous["created"]) < 0.01:
                raise RuntimeError(f"Runner is already active: {previous['pid']}")
        except psutil.NoSuchProcess:
            pass
        lock.unlink()
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, handle)
    try:
        if args.stage == "smoke":
            results = []
            for panel in args.panels:
                for method in ["OFF_LORA", "FULL"]:
                    results.append(run_trial(panel, method, GRIDS[method][1], 0, args, smoke=True))
            save_json(RUN / "preflight_summary.json", {"completed": True, "micro_groups": args.micro_groups, "autocast_weight_cache": False, "trials": results})
        else:
            preflight = RUN / "preflight_summary.json"
            if not preflight.exists() or not json.loads(preflight.read_text())["completed"]:
                raise RuntimeError("Successful S0 is required before S1")
            if json.loads(preflight.read_text())["micro_groups"] != args.micro_groups:
                raise RuntimeError("S1 microbatch must match the validated S0 configuration")
            contract = {
                "stage": "S1 development screen", "panels": args.panels, "grids": GRIDS,
                "seeds": [0, 1, 2], "steps": args.steps, "eval_every": 50,
                "effective_groups": 8, "micro_groups": args.micro_groups,
                "gradient_clip_norm": 1.0, "autocast": "bfloat16", "weight_dtype": "float32",
                "inverse_normalization_dtype": "float32", "dropout": 0.0,
                "autocast_weight_cache": False,
                "checkpoint": str(CHECKPOINT), "primary_metric": "fit_std_scaled_2_pinball_target_macro",
                "selection": "validation-only LR selection at seed 0; seeds 1 and 2 reuse that LR",
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "peft", "transformers", "chronos-forecasting", "numpy"]},
                "source_sha256": {p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest() for p in ["modeling.py", "train.py", "data.py"]},
            }
            contract_path = RUN / "contract.json"
            if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise RuntimeError("Contract changed; preserve existing results and inspect before resuming")
            save_json(contract_path, contract)
            selected = {}
            for panel in args.panels:
                run_trial(panel, "F0", 0.0, 0, args)
                for method, grid in GRIDS.items():
                    candidates = [run_trial(panel, method, lr, 0, args) for lr in grid]
                    best = min(zip(grid, candidates), key=lambda pair: pair[1]["val_score"])
                    selected[f"{panel}/{method}"] = {"lr": best[0], "val_score": best[1]["val_score"], "selected_step": best[1]["selected_step"]}
                    save_json(RUN / "selection.json", selected)
            for panel in args.panels:
                for method in GRIDS:
                    for seed in [1, 2]:
                        run_trial(panel, method, selected[f"{panel}/{method}"]["lr"], seed, args)
        save_json(RUN / "progress.json", {"status": "completed", "stage": args.stage, "finished": time.time(), "pid": os.getpid()})
    except BaseException as exc:
        save_json(RUN / "progress.json", {"status": "stopped", "stage": args.stage, "error": str(exc), "time": time.time(), "pid": os.getpid()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
