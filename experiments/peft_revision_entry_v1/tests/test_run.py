from datetime import date, timedelta
import unittest

import numpy as np

from experiments.peft_revision_entry_v1.run import CausalPanel, month, start, end, bootstrap


class FakeSeries:
    events = tuple(range(month(1989, 12), month(2024, 12) + 1))
    vintage_dates = tuple(start(e + 1) + timedelta(days=14) for e in events)

    def __init__(self, perturb_future=False):
        self.perturb_future = perturb_future

    def snapshot(self, event, asof):
        return 100.0 if asof >= start(event + 1) + timedelta(days=14) else float("nan")

    def growth(self, event, asof):
        if not np.isfinite(self.snapshot(event, asof)):
            return float("nan")
        age = min(6, asof.year * 12 + asof.month - 1 - event)
        value = .1 * (event % 11) + .01 * age
        if self.perturb_future and asof >= date(2020, 1, 1):
            value += 1000
        return value

    def first_growth(self, event):
        released = start(event + 1) + timedelta(days=14)
        return released, self.growth(event, released)


class CausalContractTests(unittest.TestCase):
    def test_future_vintages_cannot_change_current_training_or_prediction_input(self):
        origin = month(2020, 1)
        original = CausalPanel(FakeSeries()).origin(origin)
        altered = CausalPanel(FakeSeries(True)).origin(origin)
        np.testing.assert_array_equal(original["x"], altered["x"])
        self.assertEqual(original["first_bias"], altered["first_bias"])
        for arm in original["data"]:
            for a, b in zip(original["data"][arm], altered["data"][arm]):
                np.testing.assert_array_equal(a, b, err_msg=arm)

    def test_month_origin_is_two_events_after_latest_release(self):
        panel = CausalPanel(FakeSeries())
        event = month(2020, 2)
        record = panel.origin(event)
        self.assertEqual(record["cutoff"], "2020-01-31")
        self.assertEqual(record["latest"], month(2019, 12))
        self.assertEqual(event - record["latest"], 2)
        self.assertEqual(record["x"][1], panel.growth(month(2019, 12), date(2020, 1, 31)))
        _, y, ids = record["data"]["REVISED_FIXED_X"]
        for label, historical_event in zip(y, ids):
            self.assertEqual(label, panel.growth(int(historical_event), min(date(2020, 1, 31), end(int(historical_event) + 6))))

    def test_block_bootstrap_preserves_constant_paired_difference(self):
        np.testing.assert_allclose(bootstrap(np.full((2, 60), .25)), [.25, .25])


if __name__ == "__main__":
    unittest.main()
