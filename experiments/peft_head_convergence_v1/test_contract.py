import unittest
import numpy as np
from experiments.peft_head_convergence_v1.analyse import choose, gate, score_arrays, pinball_components


class ContractTests(unittest.TestCase):
    def test_selection_ties_prefer_early_step_before_recipe(self):
        candidates=[{'V':1.,'step':180,'recipe':0},{'V':1.,'step':2,'recipe':3},{'V':1.,'step':2,'recipe':1}]
        self.assertEqual(choose(candidates),candidates[2])

    def test_gate_needs_both_seeds_and_strong_F0(self):
        rows=[]
        for dataset,values in [('a',[.99,1.001]),('b',[.995,.994])]:
            for seed,loss in enumerate(values):
                for family,score in [('F0',1.),('HEAD',1.02),('JOINT',loss)]:
                    rows.append({'cell':f'{dataset}/{seed}','dataset':dataset,'seed':seed,'budget':'L720','family':family,'D_score':score,'F0_score':1.})
        result=gate(rows,'JOINT',('F0','HEAD'))
        self.assertFalse(result['source_passes']['a']); self.assertTrue(result['source_passes']['b'])
        self.assertTrue(result['passes'])

    def test_scalar_missing_target_score(self):
        q=np.array([.25,.75]); p=np.array([[[[0.,2.],[1.,3.]],[[1.,2.],[2.,3.]]]])
        y=np.array([[[.5,np.nan],[1.5,4.]]]); scale=np.array([1.,2.])
        terms=[]
        for c in range(2):
            losses=[]
            for h in range(2):
                if np.isfinite(y[0,c,h]):
                    for k,quantile in enumerate(q):
                        e=y[0,c,h]-p[0,c,k,h]; losses.append(2*max(quantile*e,(quantile-1)*e)/scale[c])
            terms.append(sum(losses)/len(losses))
        expected=sum(terms)/2
        self.assertAlmostEqual(score_arrays(p,y,q,scale),expected)
        num,count=pinball_components(p,y,q)
        self.assertAlmostEqual(float((num.sum(0)/count.sum(0)[:,None]/scale[:,None]).mean()),expected)


if __name__=='__main__': unittest.main()
