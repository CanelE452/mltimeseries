import unittest
import tempfile
from pathlib import Path
import numpy as np
from experiments.peft_initial_headroom_v1.analyse import correction_candidates, correction_parameters, score_arrays, pinball_components, apply_correction


class ContractTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(34001)
        median = rng.normal(size=(8, 2, 48))
        quantiles = np.linspace(.01, .99, 21)
        pred = median[:, :, None, :]+np.linspace(-2, 2, 21)[None, None, :, None]
        return {'prediction': pred, 'target': 2*median+3, 'quantiles': quantiles, 'scale': np.array([1., 2.])}

    def test_loss_matches_scalar_reference_and_missing_mask(self):
        z = self.data(); z['target'][0, 0, :12] = np.nan
        score = score_arrays(z['prediction'], z['target'], z['quantiles'], z['scale'])
        numerator, count = pinball_components(z['prediction'], z['target'], z['quantiles'])
        scores = []
        for channel in range(2):
            values = []
            for i in range(8):
                for h in range(48):
                    y = z['target'][i, channel, h]
                    if np.isfinite(y):
                        for qi, q in enumerate(z['quantiles']):
                            error = y-z['prediction'][i, channel, qi, h]
                            values.append(2*(q*error if error >= 0 else (q-1)*error)/z['scale'][channel])
            scores.append(np.mean(values))
        self.assertAlmostEqual(score, np.mean(scores), places=12)
        self.assertEqual(count[0, 0], 36)
        self.assertEqual(numerator.shape, (8, 2, 21))

    def test_correction_fits_train_relation_and_preserves_quantiles(self):
        z = self.data(); candidates = correction_candidates()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'train.npz'
            np.savez(path, **z)
            params = correction_parameters(path)
        self.assertEqual(len(candidates), 7)
        np.testing.assert_allclose(params['slopes'], [2, 2], atol=1e-5)
        np.testing.assert_allclose(params['intercepts'], [3, 3], atol=1e-5)
        for candidate in candidates:
            self.assertTrue(np.all(np.diff(apply_correction(z['prediction'], params, candidate), axis=2) >= 0))
        np.testing.assert_array_equal(apply_correction(z['prediction'], params, candidates[0]), z['prediction'])


if __name__ == '__main__':
    unittest.main()
