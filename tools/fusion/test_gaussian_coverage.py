"""Physical invariants for an explicit covariance-only side coverage repair."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from gaussian_coverage import apply_coverage_repair
from pipeline import transform_gaussians


def covariance(data):
    q=data[:,3:7].astype(float)
    r=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix()
    axes=r*np.exp(data[:,None,7:10].astype(float))
    return np.einsum('nik,njk->nij',axes,axes)


class CoverageInvariants(unittest.TestCase):
    def setUp(self):
        self.data=np.zeros((3,14),np.float32)
        self.data[:,:3]=[[.1,.2,.3],[.3,-.2,.7],[-.4,.6,.5]]
        quat=Rotation.from_euler('xyz',[[.2,.7,.4],[.3,.8,.1],[.9,.2,.4]]).as_quat()
        self.data[:,3:7]=np.c_[quat[:,3],quat[:,:3]]
        self.data[:,7:10]=np.log([[.0001,.0005,.001],[.001,.0002,.00005],[.0003,.0001,.002]])
        self.data[:,10:]=np.arange(12).reshape(3,4)/10
        self.ids=np.array([3,7,11],np.uint32)
        self.patch={'indices':self.ids[[0,2]],'factors':np.array([[.0003,-.0008,.0002],[.001,.0001,-.0002]]),
                    'original_parameters':self.data[[0,2],3:10].copy()}

    def test_positive_covariance_addition_without_moving_centers(self):
        out,count=apply_coverage_repair(self.data,self.ids,self.patch,.6)
        self.assertEqual(count,2)
        np.testing.assert_array_equal(out[:,:3],self.data[:,:3])
        np.testing.assert_array_equal(out[:,10:],self.data[:,10:])
        np.testing.assert_array_equal(out[1],self.data[1])
        expected=covariance(self.data)
        expected[[0,2]]+=.6*np.einsum('ni,nj->nij',self.patch['factors'],self.patch['factors'])
        np.testing.assert_allclose(covariance(out),expected,atol=3e-12,rtol=3e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance(out))>0))

    def test_repair_moves_rigidly_with_the_source(self):
        rotation=Rotation.from_euler('xyz',[.2,.4,.7]).as_matrix();scale=.928
        repaired,_=apply_coverage_repair(self.data,self.ids,self.patch)
        final=transform_gaussians(repaired,rotation,np.array([1.,2.,3.]),scale)
        expected=scale**2*np.einsum('ij,njk,lk->nil',rotation,covariance(repaired),rotation)
        np.testing.assert_allclose(covariance(final),expected,atol=3e-12,rtol=3e-6)
        original_moved=transform_gaussians(self.data,rotation,np.array([1.,2.,3.]),scale)
        np.testing.assert_array_equal(final[:,:3],original_moved[:,:3])

    def test_zero_strength_is_exactly_reversible(self):
        out,count=apply_coverage_repair(self.data,self.ids,self.patch,0)
        self.assertEqual(count,0);np.testing.assert_array_equal(out,self.data)

    def test_reject_changed_source_or_invalid_strength(self):
        changed=self.data.copy();changed[0,7]+=.1
        with self.assertRaises(ValueError):apply_coverage_repair(changed,self.ids,self.patch)
        for value in [-.01,1.01,float('nan')]:
            with self.assertRaises(ValueError):apply_coverage_repair(self.data,self.ids,self.patch,value)


if __name__=='__main__':unittest.main()
