from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from experiments.peft_objective_alignment_v1 import losses


QUANTILES = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)


def _rows(values):
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1, values.shape[-1])


def test_aligned_microbatch_gradients_equal_full_batch_gradient():
    n, c, q, h = 8, 3, 3, 4
    target_indices = [0, 2]
    rng = np.random.default_rng(123)
    raw_target = rng.normal(size=(n, c, h))
    raw_target[1, 0, 2] = np.nan
    raw_target[5, 2, 0] = np.nan
    train_counts = losses.training_denominators(raw_target, np.arange(n), target_indices, h)

    base = torch.linspace(-1.3, 1.7, n * c * q * h, dtype=torch.float64).reshape(n * c, q, h)
    target = _rows(raw_target)
    loc = torch.zeros(n * c, 1, dtype=torch.float64)
    scale = torch.ones(n * c, 1, dtype=torch.float64)

    full_norm = base.clone().detach().requires_grad_(True)
    full_loss = losses.compute_loss(
        "RAW_ALIGNED", full_norm, target, loc, scale, QUANTILES, False,
        target_indices=target_indices, channel_count=c,
        train_valid_counts=train_counts, train_origin_count=n,
        train_target_scale=torch.ones(len(target_indices), dtype=torch.float64),
    )
    full_loss.backward()
    full_grad = full_norm.grad.detach().clone()

    micro_norm = base.clone().detach().requires_grad_(True)
    for start in (0, 4):
        row = slice(start * c, (start + 4) * c)
        micro = losses.compute_loss(
            "RAW_ALIGNED", micro_norm[row], target[row], loc[row], scale[row], QUANTILES, False,
            target_indices=target_indices, channel_count=c,
            train_valid_counts=train_counts, train_origin_count=n,
            train_target_scale=torch.ones(len(target_indices), dtype=torch.float64),
        )
        (micro / 2).backward()
    torch.testing.assert_close(micro_norm.grad, full_grad, rtol=0, atol=1e-12)


def test_missing_targets_and_non_targets_have_zero_gradient():
    n, c, q, h = 4, 4, 3, 3
    target_indices = [1, 3]
    raw_target = np.zeros((n, c, h), dtype=np.float64)
    raw_target[:, 2, :] = 999.0
    raw_target[0, 1, 1] = np.nan
    raw_target[2, 3, 2] = np.nan
    train_counts = losses.training_denominators(raw_target, np.arange(n), target_indices, h)
    norm = torch.randn(n * c, q, h, dtype=torch.float64, requires_grad=True)
    target = _rows(raw_target)
    loc = torch.zeros(n * c, 1, dtype=torch.float64)
    scale = torch.ones(n * c, 1, dtype=torch.float64)

    loss = losses.compute_loss(
        "NORM_ALIGNED", norm, target, loc, scale, QUANTILES, False,
        target_indices=target_indices, channel_count=c,
        train_valid_counts=train_counts, train_origin_count=n,
        train_target_scale=torch.ones(len(target_indices), dtype=torch.float64),
    )
    loss.backward()
    grad = norm.grad.reshape(n, c, q, h)
    assert torch.count_nonzero(grad[:, [0, 2]]) == 0
    assert torch.count_nonzero(grad[0, 1, :, 1]) == 0
    assert torch.count_nonzero(grad[2, 3, :, 2]) == 0
    assert torch.count_nonzero(grad[:, target_indices]) > 0


def test_raw_aligned_sorts_after_inverse_transform_and_matches_target_macro_qmean():
    n, c, q, h = 2, 2, 3, 2
    target_indices = [0, 1]
    raw_target = np.array([[[2.0, 4.0], [1.0, np.nan]], [[3.0, 5.0], [2.0, 6.0]]])
    train_counts = losses.training_denominators(raw_target, np.arange(n), target_indices, h)
    norm_values = torch.tensor([
        [[3.0, 0.0], [1.0, 2.0], [2.0, 1.0]],
        [[0.5, 2.5], [2.5, 0.5], [1.5, 1.5]],
        [[6.0, 6.0], [4.0, 7.0], [5.0, 5.0]],
        [[1.0, 8.0], [3.0, 4.0], [2.0, 6.0]],
    ], dtype=torch.float64, requires_grad=True)
    target = _rows(raw_target)
    loc = torch.zeros(n * c, 1, dtype=torch.float64)
    scale = torch.ones(n * c, 1, dtype=torch.float64)
    target_scale = torch.tensor([2.0, 4.0], dtype=torch.float64)

    actual = losses.compute_loss(
        "RAW_ALIGNED", norm_values, target, loc, scale, QUANTILES, False,
        target_indices=target_indices, channel_count=c,
        train_valid_counts=train_counts, train_origin_count=n,
        train_target_scale=target_scale,
    )

    pred = norm_values.detach().numpy().reshape(n, c, q, h)
    expected_targets = []
    for j, channel in enumerate(target_indices):
        vals = []
        sorted_pred = np.sort(pred[:, channel], axis=1)
        for i in range(n):
            for hh in range(h):
                y = raw_target[i, channel, hh]
                if np.isfinite(y):
                    err = y - sorted_pred[i, :, hh]
                    pin = 2 * np.maximum(QUANTILES.numpy() * err, (QUANTILES.numpy() - 1) * err)
                    vals.append(pin.mean() / target_scale[j].item())
        expected_targets.append(np.mean(vals))
    assert actual.item() == pytest.approx(float(np.mean(expected_targets)), rel=0, abs=1e-12)


def test_raw_aligned_gradient_contains_inverse_transform_jacobian():
    norm = torch.tensor([[[0.2]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.0]], dtype=torch.float64)
    loc = torch.tensor([[0.3]], dtype=torch.float64)
    scale = torch.tensor([[2.0]], dtype=torch.float64)
    q = torch.tensor([0.5], dtype=torch.float64)
    loss = losses.compute_loss(
        "RAW_ALIGNED", norm, target, loc, scale, q, True,
        target_indices=[0], channel_count=1,
        train_valid_counts=np.array([1]), train_origin_count=1,
        train_target_scale=torch.tensor([4.0], dtype=torch.float64),
    )
    loss.backward()
    expected = -2.0 * math.cosh(0.2) / 4.0
    assert norm.grad.item() == pytest.approx(expected, rel=0, abs=1e-12)


def test_norm_and_raw_gradients_are_not_generally_fixed_scalar_multiples():
    target_indices = [0]
    q = torch.tensor([0.5], dtype=torch.float64)
    target = torch.tensor([[5.0], [5.0]], dtype=torch.float64)
    loc = torch.zeros(2, 1, dtype=torch.float64)
    scale = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    norm_base = torch.tensor([[[0.1]], [[0.1]]], dtype=torch.float64)

    norm_param = norm_base.clone().detach().requires_grad_(True)
    norm_loss = losses.compute_loss(
        "NORM_ALIGNED", norm_param, target, loc, scale, q, True,
        target_indices=target_indices, channel_count=1,
        train_valid_counts=np.array([2]), train_origin_count=2,
        train_target_scale=torch.ones(1, dtype=torch.float64),
    )
    norm_loss.backward()

    raw_param = norm_base.clone().detach().requires_grad_(True)
    raw_loss = losses.compute_loss(
        "RAW_ALIGNED", raw_param, target, loc, scale, q, True,
        target_indices=target_indices, channel_count=1,
        train_valid_counts=np.array([2]), train_origin_count=2,
        train_target_scale=torch.ones(1, dtype=torch.float64),
    )
    raw_loss.backward()

    ratio = raw_param.grad.flatten() / norm_param.grad.flatten()
    assert ratio[0].item() != pytest.approx(ratio[1].item(), rel=0, abs=1e-10)


def test_native_arm_reuses_shared_native_pinball_without_target_macro_realignment():
    norm = torch.zeros(4, 2, 3, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(4, 3, dtype=torch.float64)
    loc = torch.zeros(4, 1, dtype=torch.float64)
    scale = torch.ones(4, 1, dtype=torch.float64)
    q = torch.tensor([0.25, 0.75], dtype=torch.float64)
    actual = losses.compute_loss(
        "NATIVE", norm, target, loc, scale, q, False,
        target_indices=[0], channel_count=1,
        train_valid_counts=np.array([12]), train_origin_count=4,
        train_target_scale=torch.ones(1, dtype=torch.float64),
    )
    expected = losses.native.native_pinball(norm, target, loc, scale, q, False)
    assert actual is expected or actual.item() == pytest.approx(expected.item(), rel=0, abs=0)


def test_training_denominators_match_study12_fit_archives_when_present():
    root = Path(__file__).resolve().parents[3]
    prepared = root / "runs" / "peft_external_gap_v1" / "prepared"
    if not prepared.exists():
        pytest.skip("study12 prepared archives are not present")
    expected = {
        "bike": ([2870, 2870], [616, 616]),
        "household": ([3024, 3024], [624, 624]),
    }
    for dataset, (train_expected, val_expected) in expected.items():
        with np.load(prepared / f"{dataset}_fit.npz", allow_pickle=False) as archive:
            target_values = archive["target_values"]
            target_indices = archive["target_indices"].astype(int)
            train = archive["train_origins"].astype(int)
            val = archive["val_origins"].astype(int)
            assert losses.training_denominators(target_values, train, target_indices, 48).tolist() == train_expected
            assert losses.training_denominators(target_values, val, target_indices, 48).tolist() == val_expected
            assert np.linspace(0, len(train) - 1, 8).astype(int).tolist() == [0, 8, 17, 26, 35, 44, 53, 62]


def test_sort_raw_alias_is_rejected_to_keep_arm_contract_single_named():
    norm = torch.zeros(1, 1, 1, dtype=torch.float64)
    target = torch.zeros(1, 1, dtype=torch.float64)
    loc = torch.zeros(1, 1, dtype=torch.float64)
    scale = torch.ones(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown objective arm"):
        losses.compute_loss(
            "SORT_RAW", norm, target, loc, scale, torch.tensor([0.5], dtype=torch.float64), False,
            target_indices=[0], channel_count=1,
            train_valid_counts=np.array([1]), train_origin_count=1,
            train_target_scale=torch.ones(1, dtype=torch.float64),
        )
