"""Check that decisions cannot benefit from observations after their cutoff."""
import copy
import unittest
from experiments.peft_decision_transfer_v1.policy import paths,selected,probe_choice


class PolicyChronologyTests(unittest.TestCase):
    def record(self):
        def points(mode,values):
            return [{'step':s,'V':v,'checkpoint':f'{mode}_{s}'} for s,v in zip((0,5,10,20,25,40,60),values)]
        joint=points('JOINT',(1.,.98,.97,.96,.95,.94,.93))
        head=joint[:4]+points('HEAD1',(1.,.98,.97,.96,.90,.92,.94))[4:]
        return {'fork':20,'probe_steps':5,'job':{'episode':'test','dataset':'example','condition':'SPREAD30','seed':1},
                'current_C_pct_F0':1.,'histories':{'JOINT':joint,'HEAD0':head,'HEAD1':head,'HEAD2':head,'MASKED1':head}}

    def test_action_ignores_post_probe_and_test_outcomes(self):
        first=self.record();changed=copy.deepcopy(first)
        for mode,hist in changed['histories'].items():
            for p in hist:
                if p['step']>25:p['V']=1e-5 if mode=='JOINT' else 100.
        changed['evaluation_E']={'perfect_future_action':'JOINT'}
        self.assertEqual(paths(first)['PROBE']['action'],'HEAD')
        self.assertEqual(paths(changed)['PROBE']['action'],'HEAD')
        self.assertNotEqual(paths(first)['JOINT']['selected']['V'],paths(changed)['JOINT']['selected']['V'])

    def test_early_stop_cannot_see_later_recovery(self):
        h=[{'step':s,'V':v} for s,v in enumerate((1.,.8,.9,.95,.01))]
        self.assertEqual(selected(h,2),(h[1],3))
        self.assertEqual(selected(h[:4],2),selected(h,2))

    def test_near_tie_prefers_lower_work_without_looking_at_missing_action(self):
        self.assertEqual(probe_choice({'STOP':1.,'HEAD':.999,'JOINT':.998},1.,.25),'STOP')
        self.assertEqual(probe_choice({'STOP':1.,'HEAD':.9},1.,.25),'HEAD')


if __name__=='__main__':unittest.main()
