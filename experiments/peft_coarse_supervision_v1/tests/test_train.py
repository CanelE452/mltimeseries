from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.peft_coarse_supervision_v1 import cache
from experiments.peft_coarse_supervision_v1.tests.test_model import DummyBase
from experiments.peft_coarse_supervision_v1.model import PointModel


def test_cache_uses_actual_horizons_and_keeps_padding_zero():
    class Panel:
        arrays = {"horizon": np.array([17, 32]), "site": np.array(["Eagle", "Lamb"]),
                  "target_id": np.array(["a", "b"]), "month": np.array(["2016-03", "2016-04"])}

        def __len__(self):
            return 2

        def context(self, index, device):
            return torch.full((1, 512), index + 1., device=device)

    panel, model = Panel(), PointModel(DummyBase(), "HEAD")
    arrays, dtypes = cache.encode_arrays(model, {"train": panel}, SimpleNamespace(device="cpu", min_free_ram_gib=0))
    assert arrays["train_hidden"].shape == (2, 47, 768)
    assert arrays["train_base_point"].shape == (2, 752)
    for index, horizon in enumerate(panel.arrays["horizon"]):
        assert np.count_nonzero(arrays["train_base_point"][index, horizon:]) == 0
        assert np.count_nonzero(arrays["train_hidden"][index, (horizon + 15) // 16:]) == 0
        expected = model(panel.context(index, "cpu"), torch.tensor([0]), int(horizon)).detach().numpy()[0]
        np.testing.assert_array_equal(arrays["train_base_point"][index, :horizon], expected)
    assert dtypes == ["torch.float32"]
    np.testing.assert_array_equal(arrays["train_target_id"], panel.arrays["target_id"])


def test_reject_partial_output_and_preserve_it(tmp_path):
    path = tmp_path / "partial"
    path.mkdir()
    (path / "progress.jsonl").write_text("interrupted", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cache.fresh_output(path)
    assert (path / "progress.jsonl").read_text(encoding="utf-8") == "interrupted"


def test_validation_gate_rejects_eight_valid_months_per_site():
    class Panel:
        arrays = {"site": np.tile(np.repeat([0, 1], 8), 3),
                  "target_id": np.tile(np.arange(16).astype(str), 3),
                  "month": np.repeat(["2017-01", "2017-02", "2017-03"], 16),
                  "label_valid": np.ones(48, dtype=bool)}

        def __len__(self):
            return 48

    panel = Panel()
    assert cache.validation_eligibility(panel) == {"0": 24, "1": 24}
    panel.arrays["label_valid"][16:] = False
    with pytest.raises(AssertionError, match="sixteen"):
        cache.validation_eligibility(panel)


def test_macro_score_preserves_equal_sites_and_targets_with_missing_months():
    from experiments.peft_coarse_supervision_v1.train import macro_score

    site = np.array(["Eagle"] * 4 + ["Lamb"] * 2)
    target = np.array(["a", "a", "b", "b", "c", "c"])
    losses = np.array([1., 3., 10., 999., 4., 8.])
    result = macro_score(losses, np.array([1, 1, 1, 0, 1, 1], dtype=bool), site, target)
    assert result["score"] == 6.
    assert result["site_scores"] == {"Eagle": 6., "Lamb": 6.}
    assert result["target_scores"]["Eagle/a"] == 2.


def test_permutation_sampler_repeats_full_cycles_before_reusing_rows():
    from experiments.peft_coarse_supervision_v1.train import sampler

    values = sampler(7, steps=5, accumulation=4, seed=8)
    assert values.shape == (5, 4)
    np.testing.assert_array_equal(np.sort(values.ravel()[:7]), np.arange(7))
    np.testing.assert_array_equal(np.sort(values.ravel()[7:14]), np.arange(7))
    np.testing.assert_array_equal(values, sampler(7, 5, 4, 8))


def test_cached_head_matches_model_float32_readout():
    from experiments.peft_coarse_supervision_v1.train import cached_point

    model = PointModel(DummyBase(), "HEAD")
    weight = torch.linspace(-.003, .007, 16 * 768).reshape(16, 768)
    bias = torch.linspace(-.1, .1, 16)
    model.load_point_head(weight, bias)
    context, groups = torch.full((1, 512), .314159), torch.tensor([0])
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        hidden, base, _, scale = model.encode(context, groups, 17)
        expected = model(context, groups, 17)
        actual = cached_point(hidden, base, scale, weight, bias, 17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gradient_audit_accepts_inactive_initial_a_but_requires_some_b():
    from experiments.peft_coarse_supervision_v1.train import gradient_summary

    a = torch.nn.Parameter(torch.ones(2, 2))
    b = torch.nn.Parameter(torch.zeros(2, 2))
    a.grad, b.grad = torch.zeros_like(a), torch.ones_like(b)
    summary = gradient_summary([("base.x.lora_A.default.weight", a), ("base.x.lora_B.default.weight", b)])
    assert summary["nonzero_B_count"] == 1 and summary["nonzero_A_count"] == 0
    b.grad.zero_()
    with pytest.raises(AssertionError, match="LoRA B"):
        gradient_summary([("base.x.lora_B.default.weight", b)])


@pytest.mark.parametrize("corrupt_reload", [False, True])
def test_smoke_lifecycle_preserves_fitted_head_and_fails_closed(tmp_path, monkeypatch, corrupt_reload):
    from experiments.peft_coarse_supervision_v1 import train

    n = 176
    data_path, head_path = tmp_path / "train.npz", tmp_path / "head_weights.npz"
    np.savez(data_path, context=np.ones((n, 512), np.float32), horizon=np.full(n, 744),
             scale=np.ones(n), total=np.full(n, 1600.), label_valid=np.ones(n, bool),
             site=np.zeros(n, int), target_id=np.full(n, "target"), month=np.full(n, "2016-03"),
             profile=np.ones((n, 744), np.float32))
    weight, bias = np.full((16, 768), .0002, np.float32), np.zeros(16, np.float32)
    np.savez(head_path, weight=weight, bias=bias)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    cache_path, cache_result_path, ridge_path = [tmp_path / name for name in ("cache.npz", "cache_result.json", "ridge_result.json")]
    cache_path.write_bytes(b"fixture-only-cache")
    cache_result_path.write_text("{}", encoding="utf-8")
    cache.atomic_json(ridge_path, {"completed": True, "head_weights_sha256": cache.file_hash(head_path),
                                 "contract_sha256": cache.file_hash(contract_path),
                                 "cache_sha256": cache.file_hash(cache_path)})
    contract = {"checkpoint": "dummy", "source_hashes": {}, "data": {
        "train": {"path": str(data_path), "sha256": cache.file_hash(data_path)}},
        "paths": {"train_result": str(tmp_path / "run/train/result.json"), "ridge_head": str(head_path),
                  "ridge_result": str(ridge_path), "cache": str(cache_path), "cache_result": str(cache_result_path)}}
    reference = PointModel(DummyBase(), "HEAD")
    with torch.no_grad():
        h, p, _, scale = reference.encode(torch.ones(1, 512), torch.tensor([0]), 744)
    arrays = {"train_hidden": h.numpy(), "train_base_point": p.numpy(), "train_native_scale": scale.numpy().ravel()}
    monkeypatch.setattr(train, "validate_stage", lambda args: (contract_path, contract))
    monkeypatch.setattr(train, "load_verified_cache", lambda *args: (arrays, {"cache_sha256": cache.file_hash(cache_path)}))
    monkeypatch.setattr(train, "native_args", lambda contract: SimpleNamespace(device="cpu", min_free_ram_gib=0))
    monkeypatch.setattr(train, "load_base", lambda args: DummyBase())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    if corrupt_reload:
        original = train.load_trainable

        def corrupted(model, path, device):
            original(model, path, device)
            with torch.no_grad():
                model.base.encoder.block[0].layer[0].self_attention.v.lora_B.default.weight.add_(.2)

        monkeypatch.setattr(train, "load_trainable", corrupted)
    output = tmp_path / "run/smoke/fixture"
    args = SimpleNamespace(contract=str(contract_path), output=str(output), smoke=True)
    if corrupt_reload:
        with pytest.raises(AssertionError, match="Reloaded checkpoint"):
            train.run(args)
        result = cache.read_json(output / "result.json")
        assert result["completed"] is False
        assert (output / "best_trainable.pt").exists()
    else:
        result = train.run(args)
        assert result["completed"] and result["steps_completed"] == 3
        assert [item["step"] for item in result["history"]] == [0, 3]
        assert result["audits"]["checkpoint_reload_verified"] and result["audits"]["fixed_head_unchanged"]
        assert result["head_weights_sha256"] == cache.file_hash(head_path)
        assert result["evaluation_data_opened"] is False
        assert result["first_gradient"]["nonzero_B_count"] > 0
        assert result["first_gradient"]["nonzero_A_count"] == 0
