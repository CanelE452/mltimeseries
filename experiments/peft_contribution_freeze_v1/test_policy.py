import unittest
from experiments.peft_contribution_freeze_v1.policy import contribution_plateau, early_stop


def history(values, steps=(0,5,10,20)):
    return [dict(step=s,score=1.,contribution_halves=v) for s,v in zip(steps,values)]


class PolicyTests(unittest.TestCase):
    def test_warmup_and_minimum(self):
        h=history([[0,0],[1,1],[1,1],[1,1]])
        self.assertFalse(contribution_plateau(h[:3],10)[0])
        self.assertFalse(contribution_plateau(h,21)[0])
        self.assertTrue(contribution_plateau(h,20)[0])

    def test_requires_both_halves(self):
        self.assertFalse(contribution_plateau(history([[0,0],[1,1],[2,2],[2.1,5]]),20)[0])

    def test_interval_normalization_and_boundary(self):
        self.assertTrue(contribution_plateau(history([[0,0],[1,1],[2,2],[2.5,2.5]]),20)[0])
        self.assertFalse(contribution_plateau(history([[0,0],[1,1],[2,2],[2.51,2.51]]),20)[0])

    def test_decline_and_recovery(self):
        self.assertTrue(contribution_plateau(history([[0,0],[-1,-1],[-2,-2],[-3,-3]]),20)[0])
        self.assertFalse(contribution_plateau(history([[0,0],[-1,-1],[-2,-2],[0,0]]),20)[0])

    def test_early_stop_strict_and_reset(self):
        self.assertTrue(early_stop([{'score':v} for v in [1,.9,.9,.9]]))
        self.assertFalse(early_stop([{'score':v} for v in [1,.9,.9,.8]]))
        self.assertTrue(early_stop([{'score':v} for v in [1,1.1,1.2]]))


if __name__=='__main__': unittest.main()
