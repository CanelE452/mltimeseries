import copy
from pathlib import Path

import pytest

from experiments.peft_objective_alignment_v1 import prepare


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for name in prepare.SOURCE_FILES:
        path = root / "experiments" / prepare.STUDY / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source\n", encoding="utf-8")
    path = root / prepare.PLAN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Fixed plan\n", encoding="utf-8")
    parent = root / "parent.txt"
    parent.write_text("Protected parent\n", encoding="utf-8")
    protected = {"parent.txt": prepare.sha(parent)}
    monkeypatch.setattr(prepare, "_collect_parent", lambda _: dict(protected))
    monkeypatch.setattr(prepare, "_datasets", lambda *_: ({"bike": {}, "household": {}}, str(root / "checkpoint")))
    return root


def test_freeze_replays_and_rejects_changed_settings(project):
    contract = prepare.run(project)
    assert prepare.validate_contract(project) == contract
    path = project / "runs" / prepare.STUDY / "study_contract.json"
    altered = copy.deepcopy(contract)
    altered["settings"]["lr"] = 1e-4
    prepare.write(path, altered)
    with pytest.raises(AssertionError, match="contract changed"):
        prepare.validate_contract(project)
    with pytest.raises(FileExistsError):
        prepare.run(project)


def test_missing_core_source_cannot_be_frozen(project):
    path = project / "experiments" / prepare.STUDY / "forecast.py"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        prepare.run(project)
    assert not (project / "runs" / prepare.STUDY / "study_contract.json").exists()


def test_parent_change_rejected_and_no_new_contract(project):
    (project / "parent.txt").write_text("Changed parent\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Protected artifact changed"):
        prepare.run(project)


def test_omitted_parent_or_changed_plan_rejected(project):
    contract = prepare.run(project)
    path = project / "runs" / prepare.STUDY / "study_contract.json"
    altered = copy.deepcopy(contract)
    altered["protected_hashes"] = {}
    prepare.write(path, altered)
    with pytest.raises(AssertionError):
        prepare.validate_contract(project)
    prepare.write(path, contract)
    (project / prepare.PLAN).write_text("New plan\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        prepare.validate_contract(project)


def test_merge_rejects_conflicting_aliases(project):
    result = {"parent.txt": "old"}
    with pytest.raises(AssertionError, match="Conflicting"):
        prepare.merge(project, result, {str(project / "parent.txt"): "different"})
