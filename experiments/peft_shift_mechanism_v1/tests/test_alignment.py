from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.peft_shift_mechanism_v1.diagnose_alignment import (
    CONDITIONS, LAGS, aligned_inputs, paired_comparisons, require_completed_study,
    select_lags, source_indices,
)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.context = np.arange(2 * 3 * 256, dtype=np.float32).reshape(2, 3, 256)

    def test_every_supplied_source_index_is_originally_observed(self):
        for lag in LAGS:
            past, future = source_indices(lag)
            supplied = np.r_[past[past >= 0], future]
            self.assertGreaterEqual(supplied.min(), 0)
            self.assertLess(supplied.max(), 256)
            np.testing.assert_array_equal(future, 256 + np.arange(16) - lag)

    def test_alignment_is_contiguous_across_the_forecast_boundary(self):
        for lag in LAGS:
            shifted, future, _ = aligned_inputs(self.context, lag)
            joined = np.concatenate((shifted[:, 1:, lag:], future[:, 1:]), axis=-1)
            np.testing.assert_array_equal(joined, self.context[:, 1:, :256 + 16 - lag])

    def test_nan_padding_and_future_masks_are_channel_specific(self):
        for lag in LAGS:
            shifted, future, mask = aligned_inputs(self.context, lag)
            self.assertTrue(np.isnan(shifted[:, 1:, :lag]).all())
            self.assertTrue(np.isfinite(shifted[:, 1:, lag:]).all())
            np.testing.assert_array_equal(shifted[:, 0], self.context[:, 0])
            self.assertTrue((future[:, 0] == 0).all())
            self.assertTrue((mask[:, 0] == 0).all())
            self.assertTrue((mask[:, 1:] == 1).all())

    def test_original_context_is_not_modified_or_aliased(self):
        before = self.context.copy()
        shifted, future, mask = aligned_inputs(self.context, 48)
        np.testing.assert_array_equal(self.context, before)
        shifted[:, 0] = -100
        future[:] = -200
        mask[:] = 0
        np.testing.assert_array_equal(self.context, before)

    def test_selection_uses_all_validation_candidates_and_fixed_tie_break(self):
        scores = {condition: {32: .3, 48: .2, 64: .2} for condition in CONDITIONS}
        self.assertEqual(select_lags(scores), {condition: 48 for condition in CONDITIONS})
        scores["Q00"].pop(64)
        with self.assertRaises(ValueError):
            select_lags(scores)

    def test_paired_constant_improvement_has_the_expected_score_and_interval(self):
        baseline = np.ones((4, 512))
        result = paired_comparisons(baseline * .8, baseline)
        for condition in CONDITIONS:
            self.assertAlmostEqual(result[condition]["aligned_minus_f0"], -.2)
            np.testing.assert_allclose(result[condition]["aligned_minus_f0_ci"], [-.2, -.2], atol=1e-12)
            np.testing.assert_allclose(result[condition]["relative_improvement_ci"], [.2, .2], atol=1e-12)

    def test_missing_study_completion_blocks_before_any_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "must complete"):
                require_completed_study(Path(directory), Path(directory))


if __name__ == "__main__":
    unittest.main()
