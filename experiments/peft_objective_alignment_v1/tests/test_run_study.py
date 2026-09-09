import copy
from pathlib import Path

import pytest

from experiments.peft_objective_alignment_v1 import prepare, run_study as runner
from experiments.peft_adaptation_scope_v1 import guard


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    contract = {"settings": dict(prepare.SETTINGS), "plan_sha256": "plan", "source_hashes": {"train": "source"}}
    for field, phase in (("smoke_jobs", "smoke"), ("fit_jobs", "trials")):
        contract[field] = [{"dataset": d, "arm": a, "output": f"runs/{prepare.STUDY}/{phase}/{d}/{a}"}
                           for d in prepare.DATASETS for a in prepare.ARMS]
    prepare.write(root / "runs" / prepare.STUDY / "study_contract.json", contract)
    monkeypatch.setattr(prepare, "validate_contract", lambda *_: copy.deepcopy(contract))
    calls = []

    def guarded(command, output, cwd, timeout_seconds):
        output = Path(output)
        output.mkdir(parents=True)
        path = output.parent
        stage = "fit" if command[2].endswith(".train") else "forecast"
        smoke = "--smoke" in command
        dataset, arm = [command[command.index(flag) + 1] for flag in ("--dataset", "--arm")]
        if stage == "forecast":
            assert len(prepare.read(root / "runs" / prepare.STUDY / "selection.json")["selected"]) == 6
        calls.append((stage, smoke, dataset, arm))
        (path / "predictions.npz").write_bytes(b"mock predictions for lifecycle only")
        status = {"completed": True, "returncode": 0, "reasons": [], "elapsed_seconds": 1.}
        prepare.write(output / "status.json", status)
        meta = {"completed": True, "study": prepare.STUDY, "stage": stage, "smoke": smoke,
                "dataset": dataset, "arm": arm, "method": "OFF_LORA", "seed": 12000,
                "lr": 1e-5 if smoke else 3e-5, "best_step": 0,
                "study_contract_sha256": prepare.sha(root / "runs" / prepare.STUDY / "study_contract.json"),
                "source_hashes": contract["source_hashes"], "plan_sha256": contract["plan_sha256"],
                "predictions_sha256": prepare.sha(path / "predictions.npz")}
        if stage == "fit":
            (path / "best_trainable.pt").write_bytes(b"mock checkpoint")
            meta.update({"steps_completed": 5 if smoke else 200, "holdout_file_opened": False,
                         "checkpoint_sha256": prepare.sha(path / "best_trainable.pt"),
                         "audits": {k: True for k in ("zero_update_identity", "checkpoint_reload_verified", "frozen_parameters_verified", "finite_nonzero_gradient_verified")},
                         "native_replay": {"passed": True}})
        else:
            selection_path = root / "runs" / prepare.STUDY / "selection.json"
            fit = next(e for e in prepare.read(selection_path)["selected"] if (e["dataset"], e["arm"]) == (dataset, arm))
            meta.update({"selection_sha256": prepare.sha(selection_path), "fit_result_sha256": fit["result_sha256"],
                         "fit_checkpoint_sha256": fit["checkpoint_sha256"], "optimizer_steps": 0,
                         "model_unchanged": True, "global_selection_verified_before_holdout_load": True,
                         "checkpoint_reload_verified": True})
        prepare.write(path / "result.json", meta)
        return status

    monkeypatch.setattr(runner, "run_guarded", guarded)
    return root, contract, calls


def test_six_smokes_then_six_fits_then_six_forecasts_restart_reuses(project):
    root, contract, calls = project
    runner.run(root, smoke=True)
    runner.run(root)
    assert len(calls) == 18
    assert [x[:2] for x in calls] == [("fit", True)] * 6 + [("fit", False)] * 6 + [("forecast", False)] * 6
    runner.run(root)
    assert len(calls) == 18
    records = [prepare.read(p) for p in (root / "runs" / prepare.STUDY / "invocations").glob("*.json")]
    reuse = [r for r in records if len(r["reused_completed_jobs"]) == 12]
    assert len(reuse) == 1 and reuse[0]["attempts"] == []
    assert not (root / "runs" / prepare.STUDY / ".runner.lock").exists()


def test_production_without_s0_launches_nothing(project):
    root, _, calls = project
    with pytest.raises(FileNotFoundError):
        runner.run(root)
    assert calls == []
    record = prepare.read(next((root / "runs" / prepare.STUDY / "invocations").glob("*.json")))
    assert not record["completed"] and record["attempts"] == []


def test_partial_work_is_not_overwritten(project):
    root, _, calls = project
    path = root / "runs" / prepare.STUDY / "smoke/bike/NATIVE"
    path.mkdir(parents=True)
    (path / "failure.json").write_text("preserve", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial"):
        runner.run(root, smoke=True)
    assert calls == [] and (path / "failure.json").read_text(encoding="utf-8") == "preserve"


def test_stop_is_logged_and_retry_requires_preservation(project, monkeypatch):
    root, _, calls = project
    def stopped(command, output, cwd, timeout_seconds):
        status = {"completed": False, "returncode": None, "reasons": ["Git process limit"], "elapsed_seconds": 6.125}
        prepare.write(Path(output) / "status.json", status)
        return status
    monkeypatch.setattr(runner, "run_guarded", stopped)
    with pytest.raises(RuntimeError, match="stopped"):
        runner.run(root, smoke=True)
    records = [prepare.read(p) for p in (root / "runs" / prepare.STUDY / "invocations").glob("*.json")]
    attempt = records[0]["attempts"][0]
    assert attempt["guard_status"]["elapsed_seconds"] == 6.125 and not attempt["completed"]
    with pytest.raises(RuntimeError, match="partial"):
        runner.run(root, smoke=True)


def test_tampered_s0_identity_cannot_start_production(project):
    root, _, calls = project
    runner.run(root, smoke=True)
    path = root / "runs" / prepare.STUDY / "smoke_completed.json"
    done = prepare.read(path)
    done["trials"][0]["arm"] = "RAW_ALIGNED"
    prepare.write(path, done)
    with pytest.raises(AssertionError):
        runner.run(root)
    assert len(calls) == 6


def test_existing_completion_records_are_immutable(tmp_path):
    path = tmp_path / "done.json"
    prepare.write(path, {"completed": True, "fits": 5, "invocation_wall_seconds": 1})
    with pytest.raises(AssertionError):
        runner._complete_once(path, {"completed": True, "fits": 6, "invocation_wall_seconds": 2}, ("invocation_wall_seconds",))


def test_production_import_uses_actual_shared_guard():
    # The function patched in the fixture must originate from the established safety module.
    assert runner.run_guarded is guard.run_guarded
