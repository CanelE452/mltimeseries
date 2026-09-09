import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


class BaselineTests(unittest.TestCase):
    def module(self):
        name = "experiments.peft_fullft_reference_v2.baselines"
        self.assertIsNotNone(importlib.util.find_spec(name), "baseline implementation is missing")
        return importlib.import_module(name)

    def test_masked_dual_ridge_matches_independent_primal_with_intercept(self):
        b = self.module()
        x = np.array([[1., 2.], [2., 4.], [4., 1.], [5., 3.], [7., 2.]])
        y = np.array([[1., 2.], [2., np.nan], [4., 6.], [8., np.nan], [9., 12.]])
        model = b.fit_ridge(x, y, 1.0)
        z = (x - x.mean(0)) / x.std(0)
        expected = np.empty_like(y)
        for j in range(2):
            ok = np.isfinite(y[:, j])
            design = np.column_stack([np.ones(ok.sum()), z[ok]])
            penalty = np.diag([0., 1., 1.])
            coef = np.linalg.solve(design.T @ design + penalty, design.T @ y[ok, j])
            expected[:, j] = np.column_stack([np.ones(len(x)), z]) @ coef
        np.testing.assert_allclose(b.predict_ridge(model, x), expected, rtol=1e-11, atol=1e-11)
        self.assertEqual(model["observed_counts"].tolist(), [5, 3])

    def test_ridge_is_translation_equivariant_and_handles_constant_features(self):
        b = self.module()
        x = np.column_stack([np.arange(6.), np.ones(6)])
        y = np.arange(6.)[:, None] ** 2
        first = b.predict_ridge(b.fit_ridge(x, y, .01), x)
        shifted = b.predict_ridge(b.fit_ridge(x + 100, y + 17, .01), x + 100)
        np.testing.assert_allclose(shifted, first + 17, atol=1e-10)

    def test_oof_folds_purge_unarrived_48_hour_targets(self):
        b = self.module()
        origins = 336 + np.arange(89) * 24
        folds = b.oof_folds(origins, 48)
        self.assertEqual([len(f[0]) for f in folds], [29, 59])
        self.assertEqual([len(f[1]) for f in folds], [30, 29])
        for past, future in folds:
            self.assertLessEqual(int((origins[past] + 48).max()), int(origins[future[0]]))
        self.assertEqual(np.concatenate([f[1] for f in folds]).tolist(), list(range(30, 89)))

    def test_oof_predictions_and_transforms_ignore_fold_future(self):
        b = self.module()
        rng = np.random.default_rng(31)
        x = rng.normal(size=(89, 3))
        y = (2 * x[:, :1] + rng.normal(size=(89, 1)))[:, :, None]
        origins = 336 + np.arange(89) * 24
        first, indices, _ = b.ridge_oof_predictions(x, y, origins, 48, 1.0)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[60:] += 1000
        changed_y[30:] += 1000
        second, _, _ = b.ridge_oof_predictions(changed_x, changed_y, origins, 48, 1.0)
        # First fold predictions may use their input x, but never their target y.
        np.testing.assert_array_equal(first[indices < 60], second[indices < 60])

    def test_residual_quantiles_pool_only_observed_cells_per_target(self):
        b = self.module()
        residual = np.array([[[0., 2.], [10., np.nan]], [[4., 6.], [20., 30.]]])
        offsets, counts = b.residual_quantiles(residual, np.array([0., .5, 1.]))
        np.testing.assert_array_equal(offsets, [[0., 3., 6.], [10., 20., 30.]])
        self.assertEqual(counts.tolist(), [4, 3])

    def test_score_weights_targets_equally_despite_missingness(self):
        b = self.module()
        target = np.array([[[2., 2.], [8., np.nan]]])
        prediction = np.zeros((1, 2, 3, 2))
        result = b.score_predictions(prediction, target, np.array([1., 2.]), np.array([.1, .5, .9]))
        self.assertAlmostEqual(result["score"], 3.)
        self.assertEqual(result["target_scores"], [2., 4.])
        self.assertEqual(result["target_median_mse"], [4., 64.])

    def test_weekly_point_prediction_uses_only_corresponding_past_hours(self):
        b = self.module()
        values = np.arange(500.)[:, None]
        predicted = b.weekly_predictions(values, np.array([336]), np.array([0]), 48)
        np.testing.assert_array_equal(predicted[0, 0], np.arange(168., 216.))
        values[336:] = -999
        np.testing.assert_array_equal(b.weekly_predictions(values, np.array([336]), np.array([0]), 48), predicted)

    def test_four_candidate_fit_uses_no_validation_labels_in_model_parameters(self):
        b = self.module()
        self.assertTrue(hasattr(b, "fit_candidates"), "candidate fitting is missing")
        rng = np.random.default_rng(48)
        values = rng.normal(size=(2660, 2))
        panel = {"context_values": values, "target_values": values.copy(),
                 "target_indices": np.array([0, 1]), "fit_std": np.ones(2),
                 "quantiles": np.array([.01, .05] + list(np.arange(.1, 1., .05)) + [.99]),
                 "context": 336, "horizon": 48,
                 "origins": {"train": 336 + np.arange(89) * 24, "val": np.array([2520, 2544])}}
        models, records, predictions = b.fit_candidates(panel)
        self.assertEqual(len(records), 4)
        self.assertEqual(predictions.shape, (4, 2, 2, 21, 48))
        panel["target_values"][2520:] += 1000
        other_models, other_records, other_predictions = b.fit_candidates(panel)
        for left, right in zip(models, other_models):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
        np.testing.assert_array_equal(predictions, other_predictions)
        self.assertNotEqual(records[0]["validation"]["score"], other_records[0]["validation"]["score"])

    def test_fit_never_needs_holdout_and_forecast_requires_frozen_hashes(self):
        b = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "synthetic"
            run.mkdir(parents=True)
            fit_path, holdout_path, output = root / "fit.npz", root / "holdout.npz", root / "output"
            t = np.arange(3000.)
            values = np.column_stack([np.sin(t / 24) + 2, np.cos(t / 24) + 4])
            quantiles = np.array([.01, .05] + list(np.arange(.1, 1., .05)) + [.99])
            common = dict(target_indices=np.array([0, 1]), fit_std=np.array([1., 2.]),
                          quantiles=quantiles, context=np.array(336), horizon=np.array(48),
                          channels=np.array(["first", "second"]), manifest_json=np.array('{"dataset":"bike"}'))
            np.savez_compressed(fit_path, context_values=values[:2640], target_values=values[:2640],
                                timestamps=np.arange(2640), train_origins=336 + np.arange(90) * 24,
                                val_origins=np.array([2568, 2592]), **common)
            hidden = root / "hidden_holdout.npz"
            np.savez_compressed(hidden, context_values=values, target_values=values,
                                timestamps=np.arange(3000), cal_origins=np.array([2700, 2724]),
                                eval_origins=np.array([2800, 2824]), **common)
            contract = {"datasets": {"bike": {"fit_data_path": str(fit_path), "holdout_data_path": str(holdout_path),
                        "holdout_data_sha256": hashlib.sha256(hidden.read_bytes()).hexdigest()}}}
            contract_path = run / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            module = types.ModuleType("experiments.peft_fullft_reference_v2.contract")
            module.ROOT, module.STUDY = root, "synthetic"
            module.digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            module.read_contract = lambda path, verify: json.loads(Path(path).read_text(encoding="utf-8"))
            def save_json(path, value):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
            module.save_json = save_json
            with patch.dict("sys.modules", {module.__name__: module}):
                result = b.fit_stage(contract_path, output)
                self.assertEqual(result["bike"]["candidate_count"], 4)
                self.assertFalse(holdout_path.exists())
                with self.assertRaisesRegex(ValueError, "Global choices"):
                    b.forecast_stage(contract_path, output)
                hidden.replace(holdout_path)
                directory = output / "bike"
                frozen = {"completed": True, "global_choices_frozen": True,
                          "contract_sha256": module.digest(contract_path), "baselines": {"bike": {
                              "model_sha256": module.digest(directory / "model.npz"),
                              "selection_sha256": module.digest(directory / "selection.json")}}}
                frozen_path = run / "selection.json"
                save_json(frozen_path, frozen)
                changed = dict(frozen, global_choices_frozen=False)
                save_json(frozen_path, changed)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    b.forecast_stage(contract_path, output)
                save_json(frozen_path, frozen)
                original_model = (directory / "model.npz").read_bytes()
                with (directory / "model.npz").open("ab") as stream:
                    stream.write(b"hash tamper")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    b.forecast_stage(contract_path, output)
                frozen["baselines"]["bike"]["model_sha256"] = module.digest(directory / "model.npz")
                save_json(frozen_path, frozen)
                with self.assertRaisesRegex(ValueError, "Local model"):
                    b.forecast_stage(contract_path, output)
                (directory / "model.npz").write_bytes(original_model)
                frozen["baselines"]["bike"]["model_sha256"] = module.digest(directory / "model.npz")
                save_json(frozen_path, frozen)
                original_holdout = holdout_path.read_bytes()
                with holdout_path.open("ab") as stream:
                    stream.write(b"holdout tamper")
                with self.assertRaisesRegex(ValueError, "Holdout hash"):
                    b.forecast_stage(contract_path, output)
                holdout_path.write_bytes(original_holdout)
                forecast = b.forecast_stage(contract_path, output)
                self.assertEqual(forecast["bike"]["predictions"]["E"]["origins"], 2)
                with np.load(directory / "model.npz", allow_pickle=False) as model, np.load(directory / "E_predictions.npz", allow_pickle=False) as pred:
                    origins = np.array([2800, 2824])
                    if str(model["method"]) == "SEASONAL_WEEK":
                        points = np.stack([values[o - 168:o - 168 + 48].T for o in origins])
                    else:
                        x = np.stack([values[o - 336:o].T.reshape(-1) for o in origins])
                        points = (((x - model["feature_mean"]) / model["feature_scale"]) @ model["coefficients"] + model["intercept"]).reshape(2, 2, 48)
                    expected = points[:, :, None, :] + model["residual_quantiles"][None, :, :, None]
                    np.testing.assert_allclose(pred["prediction"], expected, atol=1e-12)
                    np.testing.assert_array_equal(pred["target"], np.stack([values[o:o + 48].T for o in origins]))


if __name__ == "__main__":
    unittest.main()
