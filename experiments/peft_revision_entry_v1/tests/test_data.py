from datetime import date
import math
from pathlib import Path
import tempfile
import unittest

from experiments.peft_revision_entry_v1.data import Record, VintageSeries, parse_csv


class VintageDataTests(unittest.TestCase):
    def test_inclusive_end_gap_and_future_are_not_backfilled(self):
        s = VintageSeries("T", {1: [Record(date(2000,1,1),date(2000,1,3),100),
                                     Record(date(2000,1,5),date.max,120)]}, [])
        self.assertEqual(s.snapshot(1,date(2000,1,3)),100)
        self.assertTrue(math.isnan(s.snapshot(1,date(2000,1,4))))
        self.assertTrue(math.isnan(s.snapshot(1,date(1999,12,31))))
        self.assertEqual(s.snapshot(1,date(2000,1,5)),120)
        self.assertTrue(math.isnan(s.snapshot(1,date(2025,7,1))))

    def test_growth_uses_same_vintage_previous_level_and_first_joint_arrival(self):
        s = VintageSeries("T", {1: [Record(date(2000,1,1),date(2000,1,9),100),
                                     Record(date(2000,1,10),date.max,200)],
                                 2: [Record(date(2000,1,10),date.max,220)]}, [])
        d, value = s.first_growth(2)
        self.assertEqual(d,date(2000,1,10))
        self.assertAlmostEqual(value,100*math.log(1.1))
        self.assertTrue(math.isnan(s.growth(2,date(2000,1,9))))

    def test_interval_overlap_even_at_one_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            VintageSeries("T", {1: [Record(date(2000,1,1),date(2000,1,3),100),
                                      Record(date(2000,1,3),date.max,110)]}, [])

    def test_invalid_level_and_reversed_interval_rejected(self):
        for value in (0,-1,float("nan"),float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VintageSeries("T", {1:[Record(date(2000,1,1),date.max,value)]}, [])
        with self.assertRaisesRegex(ValueError,"Reversed"):
            VintageSeries("T", {1:[Record(date(2000,1,3),date(2000,1,1),100)]}, [])

    def test_actual_csv_schema_open_end_and_month_ordinal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"data.csv"
            path.write_text("period_start_date,PAYEMS,realtime_start_date,realtime_end_date\n2000-01-01,100,2000-02-04,\n",encoding="utf-8")
            series = parse_csv(path,"PAYEMS",[date(2000,2,4)])
            self.assertEqual(series.events,(24000,))
            self.assertEqual(series.records(24000)[0].end,date.max)
            self.assertEqual(series.snapshot(24000,date(2000,2,4)),100)
            self.assertIsNone(series.first_growth(24000)[0])
            path.write_text("date,value,realtime_start,realtime_end\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"schema"):
                parse_csv(path,"PAYEMS",[])


if __name__ == "__main__":
    unittest.main()
