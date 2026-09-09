import json
from types import SimpleNamespace

import pytest

from experiments.peft_objective_alignment_v1 import forecast, train


@pytest.fixture
def selected_six(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(forecast, "ROOT", tmp_path)
    study = tmp_path / "runs" / train.STUDY
    study.mkdir(parents=True)
    contract_path = study / "study_contract.json"
    contract_path.write_text("{}")
    contract_hash = train.file_hash(contract_path)
    contract = {"source_hashes": {}, "plan_sha256": "plan", "fit_jobs": [], "datasets": {}}
    entries = []
    for dataset in ("bike", "household"):
        contract["datasets"][dataset] = {"native_reference_fit": "unused-reference"}
        for arm in train.ARMS:
            path = study / "trials" / dataset / arm
            (path / "guard").mkdir(parents=True)
            relative = str(path.relative_to(tmp_path))
            contract["fit_jobs"].append({"dataset": dataset, "arm": arm, "output": relative})
            entry = {"dataset": dataset, "arm": arm, "method": "OFF_LORA", "seed": 12000,
                     "lr": 3e-5, "smoke": False, "best_step": 40, "path": relative}
            meta = {**entry, "completed": True, "stage": "fit", "study": train.STUDY,
                    "steps_completed": 200, "holdout_file_opened": False, "source_hashes": {},
                    "plan_sha256": "plan", "study_contract_sha256": contract_hash,
                    "native_replay": {"passed": True}, "protected_hashes": {}, "cache_array_hashes": {}}
            (path / "result.json").write_text(json.dumps(meta))
            (path / "best_trainable.pt").write_bytes(b"checkpoint")
            (path / "predictions.npz").write_bytes(b"validation archive")
            (path / "guard/status.json").write_text(json.dumps({"completed": True, "returncode": 0, "reasons": []}))
            for filename, key in (("result.json", "result_sha256"), ("best_trainable.pt", "checkpoint_sha256"),
                                  ("predictions.npz", "predictions_sha256")):
                entry[key] = train.file_hash(path / filename)
            entries.append(entry)
    selection = {"completed": True, "global_choices_frozen": True, "fit_trial_count": 6,
                 "study_contract_sha256": contract_hash, "selected": entries}
    selection_path = study / "selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(forecast.prepare, "validate_contract", lambda root, path: contract)
    monkeypatch.setattr(forecast, "verify_native_replay", lambda *args: {"passed": True})
    return SimpleNamespace(root=tmp_path, study=study, contract=contract, contract_path=contract_path,
                           selection=selection, selection_path=selection_path, entries=entries)


def authorize(fixture):
    entry = fixture.entries[0]
    return forecast.validate_selection(fixture.contract_path, fixture.selection_path,
                                       fixture.root / entry["path"], entry["dataset"], entry["arm"])


def test_all_six_are_authorized_and_hashes_bound(selected_six):
    contract, entry, fit, protected = authorize(selected_six)
    assert entry["arm"] == "NATIVE" and fit["steps_completed"] == 200
    assert str(selected_six.selection_path) in protected
    assert len(protected) == 26


@pytest.mark.parametrize("mutation", ["missing-fit", "changed-checkpoint", "failed-guard", "replay-failed", "source-changed"])
def test_invalid_sibling_fit_blocks_all_forecasts_before_holdout(selected_six, monkeypatch, mutation):
    fixture = selected_six
    entry = fixture.entries[-1]
    path = fixture.root / entry["path"]
    if mutation == "missing-fit":
        fixture.selection["selected"].pop()
        fixture.selection_path.write_text(json.dumps(fixture.selection))
    elif mutation == "changed-checkpoint":
        (path / "best_trainable.pt").write_bytes(b"changed")
    elif mutation == "failed-guard":
        (path / "guard/status.json").write_text(json.dumps({"completed": True, "returncode": 1, "reasons": []}))
    elif mutation == "replay-failed":
        monkeypatch.setattr(forecast, "verify_native_replay", lambda *args: {"passed": False})
    else:
        fixture.contract["source_hashes"] = {"changed.py": {"path": "changed.py", "sha256": "x"}}
    opened = []
    monkeypatch.setattr(forecast.shared, "Panel", lambda *args: opened.append(args))
    first = fixture.entries[0]
    args = SimpleNamespace(contract=str(fixture.contract_path), selection=str(fixture.selection_path),
                           fit_trial=str(fixture.root / first["path"]), dataset=first["dataset"], arm=first["arm"],
                           output=str(fixture.study / "forecasts/bike/NATIVE"), smoke=False)
    with pytest.raises(AssertionError):
        forecast.run(args)
    assert opened == []


def test_forecast_cli_has_no_smoke_or_training_budget_escape():
    args = forecast.parser().parse_args(["--contract", "c", "--dataset", "bike", "--arm", "RAW_ALIGNED",
                                         "--output", "o", "--selection", "s", "--fit-trial", "f"])
    assert args.smoke is False
    with pytest.raises(SystemExit):
        forecast.parser().parse_args(["--smoke"])
