import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.peft_coarse_supervision_v1 import cache, forecast, run_study
from experiments.peft_coarse_supervision_v1.model import PointModel
from experiments.peft_coarse_supervision_v1.tests.test_model import DummyBase
from experiments.peft_adaptation_scope_v1.train import save_trainable


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("status", [
    {"completed": False, "returncode": 0, "reasons": []},
    {"completed": True, "returncode": 1, "reasons": []},
    {"completed": True, "returncode": 0, "reasons": ["timeout"]},
])
def test_guard_success_cannot_hide_failure(tmp_path, status):
    path = tmp_path / "status.json"
    write(path, status)
    with pytest.raises(AssertionError, match="Guard"):
        run_study.guard_ok(path)
    status = {"completed": True, "returncode": 0, "reasons": []}
    write(path, status)
    assert run_study.guard_ok(path) == status


def selection_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run_study, "ROOT", tmp_path)
    root = tmp_path / "runs" / run_study.STUDY
    write(root / "study_contract.json", {})
    checkpoint = root / "train/best_trainable.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fixed selected checkpoint")
    contract = {"paths": {"selection": str(root / "selection.json")}}
    selection = {"completed": True, "all_choices_frozen_before_evaluation": True,
                 "contract_sha256": run_study.sha(root / "study_contract.json"),
                 "input_hashes": {str(checkpoint): run_study.sha(checkpoint)},
                 "simple_policy": "FROZEN_HEAD", "best_step": 40, "best_trainable_path": str(checkpoint)}
    results = {"ridge": {"simple_policy": "FROZEN_HEAD"},
               "train": {"best_step": 40, "best_trainable_path": str(checkpoint)}, "smoke": {}}
    monkeypatch.setattr(run_study, "verify_result", lambda contract, stage: results[stage])
    return contract, selection, checkpoint


@pytest.mark.parametrize("field,value", [
    ("completed", False), ("all_choices_frozen_before_evaluation", False),
    ("contract_sha256", "foreign"), ("best_step", 80),
    ("simple_policy", "PROFILE"), ("best_trainable_path", "another-checkpoint.pt"),
])
def test_incomplete_or_mismatched_selection_is_rejected(tmp_path, monkeypatch, field, value):
    contract, selection, _ = selection_fixture(tmp_path, monkeypatch)
    write(contract["paths"]["selection"], selection)
    assert run_study.verify_selection(contract) == selection
    selection[field] = value
    write(contract["paths"]["selection"], selection)
    with pytest.raises(AssertionError):
        run_study.verify_selection(contract)


def test_selected_checkpoint_mutation_is_rejected(tmp_path, monkeypatch):
    contract, selection, checkpoint = selection_fixture(tmp_path, monkeypatch)
    write(contract["paths"]["selection"], selection)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Frozen artifact"):
        run_study.verify_selection(contract)


def test_guard_directory_is_separate_from_cache_fresh_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_study, "ROOT", tmp_path)
    contract = {"fixture": True}
    monkeypatch.setattr(run_study.prepare, "validate", lambda: contract)
    monkeypatch.setattr(run_study, "verify_result", lambda contract, stage: {"completed": True})
    called = []

    def fake_guard(command, guard, cwd, timeout_seconds, require_gpu):
        output = Path(command[command.index("--output") + 1])
        guard.mkdir(parents=True)
        assert not guard.is_relative_to(output) and not output.is_relative_to(guard)
        cache.fresh_output(output)
        called.append((timeout_seconds, require_gpu))
        return {"completed": True, "elapsed_seconds": 0.}

    monkeypatch.setattr(run_study, "run_guarded", fake_guard)
    assert run_study.run_stage(contract, "cache")["completed"]
    assert called == [(600, True)]
    with pytest.raises(FileExistsError, match="partial"):
        run_study.run_stage(contract, "cache")
    assert called == [(600, True)]


def test_runner_persists_interruption_without_acquiring_shared_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(run_study, "ROOT", tmp_path)
    monkeypatch.setattr(run_study.prepare, "validate", lambda: {})

    class DummyLock:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def stopped(contract, stage):
        raise RuntimeError("fixture safety stop")

    monkeypatch.setattr(run_study, "StudyLock", DummyLock)
    monkeypatch.setattr(run_study, "run_stage", stopped)
    with pytest.raises(RuntimeError, match="fixture safety stop"):
        run_study.run("cache")
    records = list((tmp_path / "runs" / run_study.STUDY / "invocations").glob("*.json"))
    assert len(records) == 1
    record = cache.read_json(records[0])
    assert record["completed"] is False and record["stages_completed"] == []
    assert "fixture safety stop" in record["error"]


@pytest.mark.parametrize("add_future_total", [False, True])
def test_forecast_promised_input_schema_and_no_truth_numeric_load(tmp_path, monkeypatch, add_future_total):
    n = 48
    inputs, head, coarse, checkpoint = [tmp_path / name for name in
        ("evaluation_inputs.npz", "head.npz", "coarse.npz", "best_trainable.pt")]
    arrays = {"context": np.ones((n, 512), np.float32), "horizon": np.full(n, 16),
              "scale": np.ones(n), "site": np.tile(np.repeat([0, 1], 8), 3),
              "target_id": np.tile(np.arange(16).astype(str), 3),
              "month": np.repeat(["2017-04", "2017-05", "2017-06"], 16),
              "profile": np.ones((n, 744), np.float32)}
    if add_future_total:
        arrays["total"] = np.ones(n)
    np.savez(inputs, **arrays)
    weight, bias = np.zeros((16, 768), np.float32), np.zeros(16, np.float32)
    np.savez(head, weight=weight, bias=bias)
    np.savez(coarse, weight=np.ones((2, 7), np.float64))
    model = PointModel(DummyBase(), "ATTN_LORA_FIXED_HEAD", seed=forecast.SEED).load_point_head(weight, bias)
    save_trainable(model, checkpoint)
    contract_path, selection_path = tmp_path / "contract.json", tmp_path / "selection.json"
    write(contract_path, {})
    selection = {"best_trainable_path": str(checkpoint)}
    write(selection_path, selection)
    truth = tmp_path / "evaluation_truth.npz"
    contract = {"arms": ["PROFILE", "F0", "COARSE_LIFT", "FROZEN_HEAD", "ATTN_LORA"],
        "data": {"evaluation_inputs": {"path": str(inputs), "sha256": cache.file_hash(inputs)},
                 "evaluation_truth": {"path": str(truth)}},
        "paths": {"selection": str(selection_path), "forecast_result": str(tmp_path / "forecast/result.json"),
                  "ridge_head": str(head), "ridge_coarse": str(coarse)}}
    events = []

    def selected(contract):
        events.append("selection verified")
        return selection

    original_load = np.load

    def checked_load(path, *args, **kwargs):
        assert Path(path).resolve() != truth.resolve(), "Forecast must not load evaluation truth"
        if Path(path).resolve() == inputs.resolve():
            assert events and events[0] == "selection verified"
            events.append("evaluation inputs loaded")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(forecast, "validate_stage", lambda args: (contract_path, contract))
    monkeypatch.setattr(run_study, "verify_selection", selected)
    monkeypatch.setattr(forecast, "native_args", lambda contract: SimpleNamespace(device="cpu", min_free_ram_gib=0))
    monkeypatch.setattr(forecast, "load_base", lambda args: DummyBase())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(np, "load", checked_load)
    args = SimpleNamespace(contract=str(contract_path), output=str(tmp_path / "forecast"))
    if add_future_total:
        with pytest.raises(AssertionError, match="unexpected information"):
            forecast.run(args)
        assert cache.read_json(tmp_path / "forecast/result.json")["completed"] is False
    else:
        result = forecast.run(args)
        assert result["completed"] and result["evaluation_truth_opened"] is False
        assert result["weights_unchanged"] is True
        with original_load(tmp_path / "forecast/predictions.npz", allow_pickle=False) as archive:
            for arm in contract["arms"]:
                assert archive[arm].shape == (48, 744)
                assert np.isfinite(archive[arm][:, :16]).all()
            np.testing.assert_array_equal(archive["FROZEN_HEAD"][:, :16], archive["ATTN_LORA"][:, :16])
