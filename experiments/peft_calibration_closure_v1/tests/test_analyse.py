import csv
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.peft_calibration_closure_v1 import analyse as module


class CalibrationTests(unittest.TestCase):
    def test_quantile_axis_sort_and_calibration_do_not_mutate_inputs(self):
        prediction = np.asarray([[[9., 3.], [1., 2.], [5., 1.]]])
        original = prediction.copy()
        sorted_prediction = module.sort_quantiles(prediction)
        np.testing.assert_array_equal(sorted_prediction, [[[1., 1.], [5., 2.], [9., 3.]]])
        offsets = np.asarray([8., 0., -8.])
        corrected = module.apply_offsets(prediction, offsets)
        self.assertTrue(np.all(np.diff(corrected, axis=1) >= 0))
        np.testing.assert_array_equal(prediction, original)
        np.testing.assert_array_equal(offsets, [8., 0., -8.])

    def test_offsets_are_empirical_quantiles_of_each_sorted_validation_residual(self):
        prediction = np.asarray([[[7., 3.], [-1., 0.], [2., 1.]], [[8., 6.], [0., 2.], [4., 4.]]])
        target = np.asarray([[2., 3.], [5., -2.]])
        quantiles = np.asarray([.1, .5, .9])
        actual = module.fit_offsets(prediction, target, quantiles)
        expected = []
        sorted_prediction = np.sort(prediction, axis=1)
        for index, q in enumerate(quantiles):
            values = sorted((target - sorted_prediction[:, index]).ravel())
            position = q * (len(values) - 1)
            left, right = math.floor(position), math.ceil(position)
            expected.append(values[left] + (values[right] - values[left]) * (position - left))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        self.assertEqual(actual.shape, (3,))

    def test_evaluation_target_cannot_enter_fit_api(self):
        prediction = np.zeros((4, 3, 2))
        target = np.arange(8).reshape(4, 2)
        q = np.asarray([.1, .5, .9])
        original = module.fit_offsets(prediction, target, q)
        eval_target = np.ones((5, 2))
        module.evaluate(np.zeros((5, 3, 2)), eval_target, q)
        eval_target[:] = 100000
        module.evaluate(np.zeros((5, 3, 2)), eval_target, q)
        np.testing.assert_array_equal(original, module.fit_offsets(prediction, target, q))
        self.assertEqual(list(inspect.signature(module.fit_offsets).parameters), ["val_prediction", "val_target", "quantiles"])
        with self.assertRaises(TypeError):
            module.fit_offsets(prediction, target, q, eval_target=eval_target)

    def test_pinball_score_matches_independent_scalar_definition(self):
        prediction = np.asarray([[[-1., 1.], [0., 2.], [2., 4.]], [[0., -3.], [1., 0.], [5., 1.]]])
        target = np.asarray([[.5, 1.], [3., -1.]])
        quantiles = np.asarray([.1, .5, .9])
        metrics, losses, coverage = module.evaluate(prediction, target, quantiles)
        reference = []
        for n in range(2):
            terms = []
            for qi, q in enumerate(quantiles):
                for h in range(2):
                    y, forecast = target[n, h], prediction[n, qi, h]
                    terms.append(2 * (q * (y - forecast) if y >= forecast else (1 - q) * (forecast - y)))
            reference.append(sum(terms) / len(terms))
        np.testing.assert_allclose(losses, reference, rtol=0, atol=1e-14)
        self.assertAlmostEqual(metrics["score"], sum(reference) / 2)
        self.assertAlmostEqual(metrics["coverage80"], coverage.mean())

    def test_episode_reordering_is_rejected_even_if_aggregate_score_could_match(self):
        ids = np.asarray(["a", "b", "c"])
        targets = np.asarray([[1., 2.], [3., 4.], [5., 6.]])
        module.verify_episode_pairing(ids, targets, ids.copy(), targets.copy())
        with self.assertRaises(AssertionError):
            module.verify_episode_pairing(ids, targets, ids[::-1], targets[::-1])
        with self.assertRaises(AssertionError):
            module.verify_episode_pairing(ids, targets, ids, targets + 1)

    def test_paired_bootstrap_recomputes_F0_denominator_and_averages_fitted_corpora(self):
        sort_loss = np.zeros((3, 2))
        qcal_loss = np.tile([1., 2.], (3, 1))
        f0_loss = np.asarray([[1., 4.], [2., 8.], [4., 16.]])
        sort_coverage = np.full((3, 2), .5)
        qcal_coverage = np.tile([.6, .8], (3, 1))
        weights = np.eye(2)
        actual = module.paired_effects(sort_loss, qcal_loss, f0_loss, sort_coverage, qcal_coverage, weights)
        draws = np.asarray([1. / (7./3), 2. / (28./3)])
        np.testing.assert_allclose(actual["score_change_over_F0"]["ci95"], np.quantile(draws, [.025, .975]))
        self.assertAlmostEqual(actual["score_change_over_F0"]["value"], 1.5 / (35./6))
        np.testing.assert_allclose(actual["coverage_change"]["ci95"], [.105, .295])
        self.assertAlmostEqual(actual["coverage_change"]["value"], .2)

    def test_closure_requires_both_all_corpus_score_and_mean_coverage(self):
        success = module.closure_decision([1., 1., 1.], [1., .999, 1.001], [1., 1., 1.], [.76, .8, .84])
        self.assertTrue(success["supports_simple_calibration_closure"])
        score_failure = module.closure_decision([1., 1., 1.], [1., 1., 1.02], [1., 1., 1.], [.8, .8, .8])
        self.assertFalse(score_failure["supports_simple_calibration_closure"])
        coverage_failure = module.closure_decision([1., 1., 1.], [.9, .9, .9], [1., 1., 1.], [.7, .7, .7])
        self.assertFalse(coverage_failure["supports_simple_calibration_closure"])

    def test_hash_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("before", encoding="utf-8")
            contract = {str(path): module.file_hash(path)}
            module.verify_hashes(contract)
            path.write_text("after", encoding="utf-8")
            with self.assertRaises(AssertionError):
                module.verify_hashes(contract)

    def test_complete_synthetic_fixture_emits_all_24_rows_and_verifies_completion_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_results = root / "results" / module.PARENT
            source_results.mkdir(parents=True)
            with (source_results / "selected_results.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ("method", "corpus", "score", "coverage80", "width80", "median_mse")
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for method in ("RAW", "ORACLE"):
                    for c in range(3):
                        writer.writerow(dict(zip(fields, (method, c, 1., .8, 2., .5))))
            quantiles = np.asarray([.01, .05, *np.arange(.1, 1., .05), .99])
            self.assertEqual(len(quantiles), 21)
            entries = []
            for corpus in range(3):
                for method in module.METHODS:
                    relative = f"runs/{module.PARENT}/trials/c{corpus}/{method}"
                    path = root / relative
                    path.mkdir(parents=True)
                    arrays = {"quantiles": quantiles}
                    scores = {}
                    for split, count in (("val", 128), ("eval", 512)):
                        target = np.tile(np.linspace(-.8, .8, 16), (count, 1))
                        prediction = np.broadcast_to(np.linspace(-1., 1., 21)[None, :, None], (count, 21, 16)).copy()
                        prediction += corpus * .01 if method != "F0" else 0
                        arrays.update({f"{split}_predictions": prediction, f"{split}_target": target,
                                       f"{split}_episode_ids": np.asarray([f"{split}_{i}" for i in range(count)])})
                        scores[split] = module.evaluate(prediction, target, quantiles)[0]["score"]
                    np.savez_compressed(path / "predictions.npz", **arrays)
                    module.write_json(path / "result.json", {"best_step": 0, "eval_score": scores["eval"]})
                    entries.append({"method": method, "corpus": corpus, "lr": .001, "path": relative,
                                    "val_score": scores["val"]})
            protected = source_results / "selected_results.csv"
            contract = {"protected_hashes": {str(protected): module.file_hash(protected)}}
            with patch.object(module, "source_contract", return_value=(entries, contract)), \
                 patch.object(module, "bootstrap_weights", return_value=module.bootstrap_weights(replicates=12)):
                result = module.run(root)
            self.assertTrue(result["completed"])
            self.assertEqual(result["row_count"], 24)
            output = root / "results" / module.STUDY
            with (output / "all_results.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            self.assertEqual({(row["method"], row["procedure"]) for row in rows},
                             {(m, p) for m in module.METHODS for p in module.PROCEDURES})
            for name, expected in result["outputs"].items():
                self.assertEqual(module.file_hash(output / name), expected)
            with self.assertRaises(FileExistsError):
                module.run(root)


if __name__ == "__main__":
    unittest.main()
