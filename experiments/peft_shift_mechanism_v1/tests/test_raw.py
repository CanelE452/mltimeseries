import numpy as np

from experiments.peft_shift_mechanism_v1 import data, raw


def test_raw_candidate_recovers_known_linear_generator():
    archive = data.build_archive("Q00", 0)
    quantiles = archive["quantiles"].astype(np.float64)
    features = {split: archive[f"raw_features_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    target = {split: archive[f"target_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    bases = {split: np.zeros_like(target[split]) for split in target}

    predictions, offsets, model = raw.fit_candidate(features, target, bases, alpha=1e-3, quantiles=quantiles)
    median = int(np.argmin(abs(quantiles - 0.5)))
    oracle = archive["oracle_mean_eval"].astype(np.float64)
    learned_median = predictions["eval"][:, median, :]

    mse = float(np.mean((learned_median - oracle) ** 2))
    assert mse < 5e-3
    assert offsets.shape == (21,)
    assert model[3].shape == (9,)


def test_fit_candidate_predictions_do_not_depend_on_val_eval_targets():
    archive = data.build_archive("Q01", 1)
    quantiles = archive["quantiles"].astype(np.float64)
    features = {split: archive[f"raw_features_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    target = {split: archive[f"target_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    bases = {split: np.zeros_like(target[split]) for split in target}

    original, _, _ = raw.fit_candidate(features, target, bases, alpha=0.1, quantiles=quantiles)
    perturbed_target = {split: value.copy() for split, value in target.items()}
    perturbed_target["val"][:] = 12345.0
    perturbed_target["eval"][:] = -54321.0
    perturbed, _, _ = raw.fit_candidate(features, perturbed_target, bases, alpha=0.1, quantiles=quantiles)

    np.testing.assert_array_equal(original["val"], perturbed["val"])
    np.testing.assert_array_equal(original["eval"], perturbed["eval"])


def test_episode_oof_path_shapes_and_nontrivial_offsets():
    archive = data.build_archive("Q11", 2)
    quantiles = archive["quantiles"].astype(np.float64)
    features = {split: archive[f"raw_features_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    target = {split: archive[f"target_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    bases = {split: np.zeros_like(target[split]) for split in target}

    predictions, offsets, _ = raw.fit_candidate(features, target, bases, alpha=10.0, quantiles=quantiles)

    assert predictions["val"].shape == (128, 21, 16)
    assert predictions["eval"].shape == (512, 21, 16)
    assert offsets.shape == (21,)
    assert np.all(np.diff(offsets) >= 0.0)
    assert abs(float(offsets[int(np.argmin(abs(quantiles - 0.5)))])) < 0.1
