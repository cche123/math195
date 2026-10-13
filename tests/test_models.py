import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_models import fit_ridge, splits, nested

class ModelTests(unittest.TestCase):
    def test_mean_loss_normal_equations_and_constant_feature(self):
        x=np.array([[1.,4.,3.],[2.,1.,3.],[4.,3.,3.],[7.,5.,3.]])
        y=np.array([.2,.7,.4,.8]);penalty=.3
        fit=fit_ridge(x,y,penalty)
        z=(x-fit.mean)/fit.scale
        expected=np.linalg.solve(z.T@z+len(y)*penalty*np.eye(3),z.T@(y-y.mean()))
        np.testing.assert_allclose(fit.coef,expected,atol=1e-12)
        self.assertEqual(fit.coef[-1],0.)
        self.assertAlmostEqual(fit.intercept,y.mean())
        np.testing.assert_allclose(fit.predict(x),fit.intercept+fit.contributions(x).sum(1))
    def test_unseen_data_does_not_change_scaling(self):
        x=np.array([[0.,1.],[1.,3.],[2.,4.]])
        fit=fit_ridge(x,np.array([.2,.3,.5]),1.)
        before=fit.mean.copy()
        fit.predict(np.array([[1000.,999.]]))
        np.testing.assert_array_equal(before,fit.mean)
        np.testing.assert_allclose(fit.mean,x.mean(0))
    def test_groups_and_complete_out_of_fold_prediction(self):
        rng=np.random.default_rng(195)
        groups=np.repeat(np.arange(15),2)
        x=rng.normal(size=(30,3));y=.5+.1*x[:,0]+rng.normal(0,.02,30)
        for train,test in splits(groups,5,195):
            self.assertFalse(set(groups[train]) & set(groups[test]))
        a=nested(x,y,groups);b=nested(x,y,groups)
        np.testing.assert_allclose(a['pred'],b['pred'])
        self.assertTrue(np.isfinite(a['pred']).all())
        self.assertEqual(set(a['folds']),{1,2,3,4,5})

if __name__=='__main__':unittest.main()
