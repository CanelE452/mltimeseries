"""CPU-only independent audit for the future-utility continuation diagnostic."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_future_utility_v1"
RES = ROOT / "results/peft_future_utility_v1"
OUT = RUN / "independent_audit.json"
BASE_ARRAYS = ("prediction", "target", "quantiles", "scale")
TOL = 1e-10


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def relpath(p):
    return ROOT / str(p).replace("\\", "/")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pinball_loss(pred, target, scale, quantiles):
    pred = np.sort(np.asarray(pred, dtype=np.float64), axis=2)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(target)
    counts = valid.sum(axis=(0, 2))
    if not np.all(counts > 0):
        raise AssertionError("empty target count in split")
    err = target[:, :, None, :] - pred
    q = np.asarray(quantiles, dtype=np.float64)[None, None, :, None]
    loss = np.where(valid[:, :, None, :], 2.0 * np.maximum(q * err, (q - 1.0) * err), 0.0)
    return float((loss.sum(axis=(0, 3)) / counts[:, None] / np.asarray(scale, dtype=np.float64)[:, None]).mean())


def npz_scores(path, key="prediction"):
    with np.load(path, allow_pickle=False) as z:
        return {
            "S": pinball_loss(z[key][0:14], z["target"][0:14], z["scale"], z["quantiles"]),
            "D": pinball_loss(z[key][16:30], z["target"][16:30], z["scale"], z["quantiles"]),
        }


def same_npz(a, b, keys=BASE_ARRAYS):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        for k in keys:
            if k not in x.files or k not in y.files:
                raise AssertionError(f"missing {k}: {a} vs {b}")
            if x[k].shape != y[k].shape or x[k].dtype != y[k].dtype:
                raise AssertionError(f"shape/dtype mismatch {k}: {a} vs {b}")
            if not np.array_equal(x[k], y[k], equal_nan=True):
                raise AssertionError(f"array mismatch {k}: {a} vs {b}")


def sample_hash(seed, rows, cap, start=0):
    samples = np.random.default_rng(seed).integers(rows, size=(cap, 8))
    return hashlib.sha256(samples[start:].tobytes()).hexdigest()


def earliest_by_s(history):
    best = history[0]
    for h in history[1:]:
        if h["S"] < best["S"]:
            best = h
    return best


def ranks(v):
    a = np.asarray(v)
    r = np.empty(len(a), dtype=float)
    for x in np.unique(a):
        ids = np.flatnonzero(a == x)
        r[ids] = np.sum(a < x) + 0.5 * (len(ids) - 1)
    return r


def correlation(x, y):
    x, y = ranks(x), ranks(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def close(a, b, where="root"):
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            raise AssertionError(f"{where}: {a!r} != {b!r}")
    elif a is None or b is None:
        if a is not b:
            raise AssertionError(f"{where}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not np.isfinite(float(a)) or not np.isfinite(float(b)) or abs(float(a) - float(b)) > TOL:
            raise AssertionError(f"{where}: {a!r} != {b!r}")
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise AssertionError(f"{where}: key mismatch {set(a) ^ set(b)}")
        for k in a:
            close(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"{where}: len {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{where}[{i}]")
    elif a != b:
        raise AssertionError(f"{where}: {a!r} != {b!r}")


def check_plan(plan):
    checked = 0
    for group in ("source_hashes", "input_hashes"):
        for p, expected in plan[group].items():
            actual = sha(relpath(p))
            if actual != expected:
                raise AssertionError(f"{group} hash mismatch: {p}")
            checked += 1
    forbidden = [p for p in plan["input_hashes"] if "holdout" in p.lower() or "/eval/" in p.replace("\\", "/").lower()]
    if forbidden:
        raise AssertionError(f"E/holdout-like input paths present: {forbidden[:3]}")
    return checked


def read_completed(plan_sha):
    c = read_json(RUN / "completed.json")
    if not c.get("completed") or c.get("plan_sha256") != plan_sha:
        raise AssertionError("completed.json does not match plan")
    return c


def check_guards():
    paths = sorted(RUN.rglob("guard/status.json"), key=lambda p: p.as_posix())
    bad = []
    for p in paths:
        s = read_json(p)
        if not (s.get("completed") and s.get("returncode") == 0 and not s.get("reasons")):
            bad.append(p.relative_to(ROOT).as_posix())
    if len(paths) != 13 or bad:
        raise AssertionError(f"guard failure count={len(paths)} bad={bad[:3]}")
    return paths, sum(float(read_json(p).get("elapsed_seconds", 0.0)) for p in paths)


def verify_result(entry, result, out, plan, score_errors):
    job, plan_sha = entry["job"], sha(RUN / "plan.json")
    if not (result.get("completed") and result.get("plan_sha256") == plan_sha and result.get("job") == job):
        raise AssertionError(f"bad result header: {entry['key']}")
    if result.get("holdout_opened") is not False or result.get("S_rows") != plan["selection_rows"] or result.get("D_rows") != plan["diagnostic_rows"]:
        raise AssertionError(f"bad leakage/split fields: {entry['key']}")
    cap, expected_forks = plan["schedules"][job["condition"]][-1], [plan["schedules"][job["condition"]][-1] // 3, 2 * plan["schedules"][job["condition"]][-1] // 3]
    if result["cap"] != cap or [f["fork"] for f in result["forks"]] != expected_forks:
        raise AssertionError(f"bad fork schedule: {entry['key']}")
    if result["sample_index_sha256"] != sample_hash(job["seed"], len(job["training_rows"]), cap):
        raise AssertionError(f"sample hash mismatch: {entry['key']}")
    old = ROOT / entry["reference_fit"] / "output"
    for p in sorted((out / "joint").glob("point_*.npz")):
        same_npz(p, old / p.name)
    init_scores = npz_scores(out / "joint/point_0.npz")
    close(init_scores["S"], result["initial"]["S"], "initial.S")
    close(init_scores["D"], result["initial"]["D"], "initial.D")


def row_for_fork(entry, result, out, fork_rec, score_errors):
    job, fork, branches = entry["job"], fork_rec["fork"], fork_rec["branches"]
    if set(branches) != {"JOINT", "HEAD_ONLY", "MASKED_UPDATE"}:
        raise AssertionError(f"bad branch set: {entry['key']} fork{fork}")
    joint = branches["JOINT"]
    prefix = [h for h in joint["history"] if h["step"] <= fork]
    current, previous = prefix[-1], prefix[-2]
    if fork_rec["prefix_model_hash"] != current["model_hash"] or fork_rec["suffix_sample_sha256"] != sample_hash(job["seed"], len(job["training_rows"]), result["cap"], fork):
        raise AssertionError(f"fork metadata mismatch: {entry['key']} fork{fork}")
    s0, d0 = result["initial"]["S"], result["initial"]["D"]
    row = {k: job[k] for k in ("dataset", "condition", "seed")}
    row.update({
        "fork": fork, "cap": result["cap"],
        "C_S_pct_F0": 100 * (current["S_off"] - current["S"]) / s0,
        "C_slope_per_update": (100 * (current["S_off"] - current["S"]) / s0 - 100 * (previous["S_off"] - previous["S"]) / s0) / (fork - previous["step"]),
        "S_improvement_per_update": 100 * (previous["S"] - current["S"]) / (s0 * (fork - previous["step"])),
        "D_F0": d0, "D_STOP": current["D"],
    })
    for mode, branch in branches.items():
        if branch["selected"] != earliest_by_s(branch["history"]) or branch["final"] != branch["history"][-1]:
            raise AssertionError(f"S-only earliest/final mismatch: {entry['key']} {mode} fork{fork}")
        if branch["history"][:len(prefix)] != prefix:
            raise AssertionError(f"branch prefix history mismatch: {entry['key']} {mode} fork{fork}")
        for h in branch["history"]:
            folder = out / "joint" if mode == "JOINT" or h["step"] <= fork else out / f"fork_{fork}" / mode
            scores = npz_scores(folder / f"point_{h['step']}.npz")
            score_errors.extend([abs(scores["S"] - h["S"]), abs(scores["D"] - h["D"])])
            if "S_off" in h:
                if h["step"] == 0:
                    score_errors.append(abs(h["S"] - h["S_off"]))
                else:
                    score_errors.append(abs(npz_scores(folder / f"point_{h['step']}.npz", "off_prediction")["S"] - h["S_off"]))
        row[f"{mode}_D_final"] = branch["final"]["D"]
        row[f"{mode}_D_selected"] = branch["selected"]["D"]
        row[f"{mode}_selected_step"] = branch["selected"]["step"]
        row[f"{mode}_gain_vs_STOP"] = 100 * (current["D"] - branch["final"]["D"]) / d0
        row[f"{mode}_selection_benefit"] = 100 * (branch["final"]["D"] - branch["selected"]["D"]) / d0
        if mode != "JOINT":
            if not (branch["restore_verified"] and branch["adapter_unchanged"] and branch["head_changed"]):
                raise AssertionError(f"branch state flags failed: {entry['key']} {mode} fork{fork}")
            same_npz(out / f"fork_{fork}" / mode / f"point_{fork}.npz", out / "joint" / f"point_{fork}.npz")
            row[f"{mode}_clipped_steps"] = branch["clipped_steps"]
    row["U_final_pct_F0"] = 100 * (row["HEAD_ONLY_D_final"] - row["JOINT_D_final"]) / d0
    row["U_selected_pct_F0"] = 100 * (row["HEAD_ONLY_D_selected"] - row["JOINT_D_selected"]) / d0
    row["U_masked_pct_F0"] = 100 * (row["MASKED_UPDATE_D_final"] - row["JOINT_D_final"]) / d0
    row["clipping_control_difference_pct_F0"] = 100 * (row["HEAD_ONLY_D_final"] - row["MASKED_UPDATE_D_final"]) / d0
    return row


def metrics_for(rows, band):
    metrics = {}
    for name in ("all", "bdg2", "jena"):
        group = [r for r in rows if name == "all" or r["dataset"] == name]
        meaningful = [r for r in group if abs(r["U_final_pct_F0"]) > band]
        rules = {}
        if meaningful:
            fns = {
                "always_continue": lambda r: True,
                "always_freeze": lambda r: False,
                "positive_C": lambda r: r["C_S_pct_F0"] > 0,
                "increasing_C": lambda r: r["C_slope_per_update"] > 0,
                "improving_S": lambda r: r["S_improvement_per_update"] > 0,
            }
            rules = {k: sum(fn(r) == (r["U_final_pct_F0"] > band) for r in meaningful) / len(meaningful) for k, fn in fns.items()}
        metrics[name] = {
            "forks": len(group),
            "joint_better_than_head": sum(r["U_final_pct_F0"] > band for r in group),
            "head_better_than_joint": sum(r["U_final_pct_F0"] < -band for r in group),
            "negligible": len(group) - len(meaningful),
            "joint_better_than_head_and_STOP": sum(r["U_final_pct_F0"] > band and r["JOINT_gain_vs_STOP"] > band for r in group),
            "head_better_than_joint_and_STOP": sum(r["U_final_pct_F0"] < -band and r["HEAD_ONLY_gain_vs_STOP"] > band for r in group),
            "both_continuations_worse_than_STOP": sum(r["JOINT_gain_vs_STOP"] < -band and r["HEAD_ONLY_gain_vs_STOP"] < -band for r in group),
            "selection_reverses_meaningful_direction": sum(r["U_final_pct_F0"] * r["U_selected_pct_F0"] < 0 and min(abs(r["U_final_pct_F0"]), abs(r["U_selected_pct_F0"])) > band for r in group),
            "mean_U_final_pct_F0": float(np.mean([r["U_final_pct_F0"] for r in group])),
            "mean_U_selected_pct_F0": float(np.mean([r["U_selected_pct_F0"] for r in group])),
            "max_abs_clipping_control_difference_pct_F0": max(abs(r["clipping_control_difference_pct_F0"]) for r in group),
            "spearman_C_vs_U": correlation([r["C_S_pct_F0"] for r in group], [r["U_final_pct_F0"] for r in group]),
            "spearman_C_slope_vs_U": correlation([r["C_slope_per_update"] for r in group], [r["U_final_pct_F0"] for r in group]),
            "meaningful_forks_for_rule_accuracy": len(meaningful),
            "descriptive_rule_accuracy": rules,
        }
    return metrics


def sign_pairs(rows, band):
    pairs = []
    for ds in ("bdg2", "jena"):
        for condition in ("FULL90", "SPREAD30", "RECENT30"):
            for fraction in (1 / 3, 2 / 3):
                pair = [r for r in rows if r["dataset"] == ds and r["condition"] == condition and abs(r["fork"] / r["cap"] - fraction) < 1e-9]
                if len(pair) != 2:
                    raise AssertionError(f"seed pair coverage mismatch {ds} {condition} {fraction}")
                classes = [1 if r["U_final_pct_F0"] > band else -1 if r["U_final_pct_F0"] < -band else 0 for r in pair]
                pairs.append({"dataset": ds, "condition": condition, "fork_fraction": fraction, "classes": classes, "agrees": classes[0] == classes[1]})
    return pairs


def compare_csv(rows):
    actual = list(csv.DictReader((RES / "metrics.csv").open(encoding="utf-8", newline="")))
    key = lambda r: (r["dataset"], r["condition"], int(r["seed"]), int(r["fork"]))
    got = {key(r): r for r in actual}
    exp = {key(r): r for r in rows}
    if set(got) != set(exp):
        raise AssertionError("metrics.csv row-key coverage mismatch")
    for k, row in exp.items():
        raw = got[k]
        for field, val in row.items():
            if isinstance(val, int):
                close(int(raw[field]), val, f"metrics.csv.{k}.{field}")
            elif isinstance(val, float):
                close(float(raw[field]), val, f"metrics.csv.{k}.{field}")
            else:
                close(raw[field], val, f"metrics.csv.{k}.{field}")


def verify_smoke(plan, plan_sha, score_errors):
    r = read_json(RUN / "smoke/output/result.json")
    if not (r.get("completed") and r.get("smoke") and r.get("plan_sha256") == plan_sha and r.get("holdout_opened") is False):
        raise AssertionError("bad smoke result")
    if r["sample_index_sha256"] != sample_hash(r["job"]["seed"], len(r["job"]["training_rows"]), r["cap"]):
        raise AssertionError("smoke sample hash mismatch")
    out = RUN / "smoke/output"
    for f in r["forks"]:
        if set(f["branches"]) != {"JOINT", "HEAD_ONLY", "MASKED_UPDATE", "JOINT_REPLAY"}:
            raise AssertionError("bad smoke branch set")
        for mode, b in f["branches"].items():
            if b["selected"] != earliest_by_s(b["history"]) or b["final"] != b["history"][-1]:
                raise AssertionError(f"smoke selection/final mismatch {mode}")
            for h in b["history"]:
                folder = out / "joint" if mode == "JOINT" or h["step"] <= f["fork"] else out / f"fork_{f['fork']}" / mode
                s = npz_scores(folder / f"point_{h['step']}.npz")
                score_errors.extend([abs(s["S"] - h["S"]), abs(s["D"] - h["D"])])
            if mode != "JOINT":
                same_npz(out / f"fork_{f['fork']}" / mode / f"point_{f['fork']}.npz", out / "joint" / f"point_{f['fork']}.npz")
            if mode == "JOINT_REPLAY":
                for h in b["history"]:
                    if h["step"] > f["fork"]:
                        same_npz(out / f"fork_{f['fork']}" / mode / f"point_{h['step']}.npz", out / "joint" / f"point_{h['step']}.npz")


def audit():
    plan_sha = sha(RUN / "plan.json")
    plan = read_json(RUN / "plan.json")
    if read_json(RES / "summary.json").get("completed") is not True:
        raise AssertionError("analysis summary is not completed")
    completed = read_completed(plan_sha)
    checked_hashes = check_plan(plan)
    guard_paths, guard_seconds = check_guards()
    rows, histories, score_errors, seen = [], [], [], set()
    verify_smoke(plan, plan_sha, score_errors)
    for entry in plan["jobs"]:
        out = RUN / entry["key"] / "output"
        result = read_json(out / "result.json")
        if result.get("prefix_exact_study31") is not True:
            raise AssertionError(f"prefix flag failed: {entry['key']}")
        verify_result(entry, result, out, plan, score_errors)
        histories.append(result)
        for fork_rec in result["forks"]:
            row = row_for_fork(entry, result, out, fork_rec, score_errors)
            seen.add((row["dataset"], row["condition"], row["seed"], row["fork"]))
            rows.append(row)
    if len(rows) != plan["expected_forks"] or len(seen) != plan["expected_forks"]:
        raise AssertionError("24-row fork coverage failed")
    max_err = max(score_errors) if score_errors else 0.0
    if max_err >= TOL:
        raise AssertionError(f"raw score mismatch {max_err}")
    band = plan["negligible_band_pct_F0"]
    metrics, pairs = metrics_for(rows, band), sign_pairs(rows, band)
    summary = read_json(RES / "summary.json")
    close(summary["metrics"], metrics, "summary.metrics")
    close(summary["seed_sign_pairs"], pairs, "summary.seed_sign_pairs")
    close(summary["raw_score_max_abs_error"], max_err, "summary.raw_score_max_abs_error")
    close(summary["guard_jobs"], len(guard_paths), "summary.guard_jobs")
    close(summary["diagnostic_guard_seconds"], guard_seconds, "summary.diagnostic_guard_seconds")
    close(summary["evaluation_E_used"], False, "summary.evaluation_E_used")
    close(read_json(RES / "histories.json"), histories, "histories.json")
    compare_csv(rows)
    payload = {
        "completed": True,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "self_sha256": sha(Path(__file__)),
        "plan_sha256": plan_sha,
        "completed_sha256": sha(RUN / "completed.json"),
        "checked_plan_hashes": checked_hashes,
        "guard_status_count": len(guard_paths),
        "guard_elapsed_seconds_sum": guard_seconds,
        "main_results": len(histories),
        "fork_rows": len(rows),
        "unique_fork_keys": len(seen),
        "raw_npz_score_checks": len(score_errors),
        "raw_score_max_abs_error": max_err,
        "metrics_csv_verified": True,
        "summary_verified": True,
        "histories_verified": True,
        "prefix_npz_exact_verified": True,
        "branch_start_npz_exact_verified": True,
        "S_only_earliest_selection_verified": True,
        "holdout_opened_all_false": True,
        "E_paths_opened_by_audit": False,
        "recomputed_summary": {"metrics": metrics, "seed_sign_pairs": pairs},
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload["result_payload_sha256"] = hashlib.sha256(stable).hexdigest()
    return payload


def write_json_exclusive(path, payload):
    text = json.dumps(payload, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def write_failure(exc):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "completed": False,
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "self_sha256": sha(Path(__file__)),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=20),
    }
    for i in range(100):
        suffix = "" if i == 0 else f"_{i:02d}"
        path = RUN / f"independent_audit_failure_{stamp}{suffix}.json"
        try:
            write_json_exclusive(path, payload)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not preserve failure log")


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing audit output: {OUT}")
    try:
        payload = audit()
        write_json_exclusive(OUT, payload)
        print(json.dumps({"completed": True, "output": str(OUT), "result_payload_sha256": payload["result_payload_sha256"]}, ensure_ascii=False))
    except Exception as exc:
        path = write_failure(exc)
        print(json.dumps({"completed": False, "failure_log": str(path), "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
