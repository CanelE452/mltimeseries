import unittest

from experiments.peft_fullft_reference_v2.run_study import select_trials


class SelectionTests(unittest.TestCase):
    def test_shared_rate_uses_all_seed_mean_instead_of_best_seed(self):
        entries = [dict(dataset="d",arm="a",lr=lr,seed=s,val_score=v)
                   for lr, values in ((.1,[1,1,10]),(.2,[3,3,3])) for s,v in enumerate(values)]
        selected, choices = select_trials(entries,["d"],{"a":[.1,.2]},[0,1,2])
        self.assertEqual({e["lr"] for e in selected},{.2})
        self.assertEqual(len(selected),3)
        self.assertEqual(choices[0]["selected_lr"],.2)

    def test_missing_or_duplicated_seed_fails(self):
        entry = dict(dataset="d",arm="a",lr=.1,seed=0,val_score=1)
        for entries in ([entry],[entry,entry]):
            with self.assertRaises(AssertionError):
                select_trials(entries,["d"],{"a":[.1]},[0,1])

    def test_tie_uses_registered_rate_order(self):
        entries = [dict(dataset="d",arm="a",lr=lr,seed=0,val_score=1) for lr in (.1,.2)]
        selected,_ = select_trials(entries,["d"],{"a":[.1,.2]},[0])
        self.assertEqual(selected[0]["lr"],.1)
