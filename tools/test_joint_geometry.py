#!/usr/bin/env python3
"""Independent synthetic checks for the joint geometry measurements."""
import unittest
import numpy as np
from analyze_joint_geometry import fit_section, hardware_cap_pair, unit, C0

class GeometryChecks(unittest.TestCase):
    def test_ellipse_center_ignores_front_sampling_density(self):
        rng=np.random.default_rng(419)
        theta=np.r_[rng.uniform(-np.pi,np.pi,3000),rng.uniform(0,np.pi,24000)]
        center=np.array([.011,-.007]);uv=np.c_[.046*np.cos(theta),.057*np.sin(theta)]+center+rng.normal(0,.00035,(len(theta),2))
        points=np.zeros((len(theta),14));points[:,:2]=uv;points[:,2]=rng.uniform(-.006,.006,len(theta))
        fit=fit_section(points,np.zeros(3),np.array([0.,0.,1.]),0,radius=.1)
        self.assertIsNotNone(fit)
        # basis([0,0,1]) maps u=-X and v=-Y; compare back in world XYZ.
        self.assertLess(np.linalg.norm(fit['center'][:2]-center),.001)
        self.assertGreaterEqual(fit['angularBins'],46)

    def test_partial_shell_does_not_establish_a_center(self):
        theta=np.linspace(-.8,.8,1000);points=np.zeros((1000,14));points[:,:2]=np.c_[.05*np.cos(theta),.05*np.sin(theta)]
        self.assertIsNone(fit_section(points,np.zeros(3),np.array([0.,0.,1.]),0,radius=.1))

    def test_opposing_metal_caps_establish_axis_and_midpoint(self):
        rng=np.random.default_rng(11);centers=np.array([[-.047,-.004,.011],[.047,.007,-.011]])
        points=[]
        for center in centers:
            q=np.zeros((300,14));q[:,:3]=center+rng.normal(0,.0022,(300,3));q[:,11:14]=(.67-.5)/C0;points.append(q)
        skin=np.zeros((500,14));skin[:,:3]=rng.uniform(-.06,.06,(500,3));skin[:,11:14]=(np.array([.64,.43,.27])-.5)/C0
        fit=hardware_cap_pair(np.concatenate([*points,skin]),np.zeros(3))
        self.assertIsNotNone(fit);self.assertEqual(fit['candidatePairCount'],1)
        self.assertLess(np.linalg.norm(fit['center']-centers.mean(0)),.0007)
        expected=unit(centers[1]-centers[0]);self.assertGreater(fit['axis']@expected,.999)
        self.assertIsNone(hardware_cap_pair(np.concatenate([points[0],skin]),np.zeros(3)))

if __name__=='__main__':unittest.main()
