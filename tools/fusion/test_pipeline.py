"""Numerical invariants for Gaussian transforms, registration, and fusion.

Run without loading either large source scan:
  .venv-fusion/bin/python -B -m unittest discover -s tools/fusion -p test_pipeline.py
"""
import unittest
import json
from pathlib import Path
import tempfile
import contextlib
import io
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

import pipeline
from cleanup_masks import load_cleanup_masks, keep_source_rows, validate_restorations, restore_surface_weights
from appearance_colors import load_color_overrides, apply_color_values


def gaussians(points, alpha=.7):
    result = np.zeros((len(points), len(pipeline.FIELDS)), dtype=np.float64)
    result[:, :3] = points
    result[:, 3] = 1  # Stored quaternion order is wxyz.
    result[:, 7:10] = np.log([.012, .025, .006])
    result[:, 10] = np.log(alpha / (1-alpha))
    result[:, 11:14] = [.15, -.3, .7]
    return result


def covariance(record):
    q = record[3:7]
    rotation = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
    return rotation @ np.diag(np.exp(2 * record[7:10])) @ rotation.T


class CleanupTests(unittest.TestCase):
    def test_restoration_uses_source_identity_and_preserves_other_weights(self):
        indices=np.array([7,2,9,3],np.uint32);weights=np.array([.002,.3,.8,.04])
        selections=[np.array([3,7],np.uint32)];rules=[{'part':'hand','minimumWeight':.6}]
        parts=[('hand','Hand','a','b',.1,.1),('arm','Arm','b','c',.1,.1)]
        labels=np.array([0,1,0,0]);validate_restorations(labels,indices,selections,rules,parts,.035)
        revised,selected=restore_surface_weights(weights,indices,selections,rules)
        assert_allclose(revised,[.6,.3,.8,.6]);assert_allclose(weights,[.002,.3,.8,.04])
        assert_array_equal(selected,[True,False,False,True])
        cleaned=keep_source_rows(indices,np.array([2,3,7],np.uint32))|selected
        assert_array_equal(cleaned,[True,False,True,True])
        with self.assertRaises(ValueError):validate_restorations(np.array([1,1,0,0]),indices,selections,rules,parts,.035)
        with self.assertRaises(ValueError):validate_restorations(labels,indices,selections*2,rules*2,parts,.035)
        with self.assertRaises(ValueError):validate_restorations(labels,indices,selections,[{'part':'hand','minimumWeight':1.1}],parts,.035)

    def test_color_repair_is_sparse_reversible_and_checks_original_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.array([[.1, .2, .3], [.2, .3, .4]], dtype=np.float32)
            repaired = original + np.float32(.05)
            np.savez(root / 'colors.npz', indices=np.array([2, 7], np.uint32), f_dc=repaired, original_f_dc=original)
            sources = {'front': {'file': 'front.ply', 'sha256': 'front-hash', 'count': 10},
                       'back': {'file': 'back.ply', 'sha256': 'back-hash', 'count': 12}}
            doc = {'version': 1, 'sourceCapture': 'back', 'sourceFile': 'back.ply', 'sourceSha256': 'back-hash',
                   'sourceHashes': {'front.ply': 'front-hash', 'back.ply': 'back-hash'},
                   'overrideFile': 'colors.npz', 'uniqueSourceIndices': 2}
            (root / 'colors.json').write_text(json.dumps(doc))
            patches, _ = load_color_overrides(['colors.json'], sources, root)
            colors = np.array([original[1], [0, 0, 0], original[0]], dtype=np.float32)
            indices = np.array([7, 5, 2], np.uint32)
            result, count = apply_color_values(colors, indices, patches['back'])
            self.assertEqual(count, 2)
            assert_array_equal(result, [repaired[1], colors[1], repaired[0]])
            assert_array_equal(apply_color_values(colors, indices, patches['back'], 0)[0], colors)
            assert_array_equal(apply_color_values(colors, indices, patches['front'])[0], colors)
            incorrect = colors.copy(); incorrect[0, 0] += .01
            with self.assertRaises(ValueError):
                apply_color_values(incorrect, indices, patches['back'])
            with self.assertRaises(ValueError):
                load_color_overrides(['colors.json', 'colors.json'], sources, root)
            sources['front']['sha256'] = 'changed-front'
            with self.assertRaises(ValueError):
                load_color_overrides(['colors.json'], sources, root)

    def test_source_masks_follow_original_rows_and_reject_stale_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'mask.npy', np.array([2, 7], dtype=np.uint32))
            sources = {'front': {'file': 'front.ply', 'sha256': 'front-hash', 'count': 10},
                       'back': {'file': 'back.ply', 'sha256': 'back-hash', 'count': 12}}
            document = {'version': 1, 'sourceCapture': 'back', 'sourceFile': 'back.ply',
                        'sourceSha256': 'back-hash', 'sourceHashes': {'front.ply': 'front-hash', 'back.ply': 'back-hash'},
                        'maskFile': 'mask.npy', 'uniqueSourceIndices': 2}
            (root / 'mask.json').write_text(json.dumps(document))
            masks, _ = load_cleanup_masks(['mask.json'], sources, root)
            assert_array_equal(keep_source_rows(np.array([7, 1, 9, 2]), masks['back']), [False, True, True, False])
            assert_array_equal(keep_source_rows(np.array([7, 2]), masks['front']), [True, True])
            sources['front']['sha256'] = 'different-capture'
            with self.assertRaises(ValueError):
                load_cleanup_masks(['mask.json'], sources, root)
            sources['front']['sha256'] = 'front-hash'
            np.save(root / 'mask.npy', np.array([7, 2], dtype=np.uint32))
            with self.assertRaises(ValueError):
                load_cleanup_masks(['mask.json'], sources, root)

    def test_hair_shape_covariance_follows_local_deformation_and_protects_face(self):
        from hair_shape import deform_head_hair, hair_displacement
        points = np.array([[0, .09, -.22], [.045, .07, -.30], [0, -.1, -.2], [-.08, -.035, -.09]])
        data = gaussians(points)
        xyzw = Rotation.from_euler('xyz', [[20, 35, 11], [11, -24, 60], [0, 0, 0], [10, 20, 30]], degrees=True).as_quat()
        data[:, 3:7] = np.c_[xyzw[:, 3], xyzw[:, :3]]
        settings = {'amount': .014}
        corrected, audit = deform_head_hair(data, settings)
        assert_allclose(corrected[:, :3], points + hair_displacement(points, settings), atol=1e-12)
        assert_array_equal(corrected[2:], data[2:])
        assert_array_equal(corrected[:, 10:], data[:, 10:])
        for index in [0, 1]:
            jacobian = np.eye(3)
            for axis in range(3):
                step = np.zeros(3); step[axis] = 2e-7
                jacobian[:, axis] += (hair_displacement(points[index:index+1] + step, settings)[0] -
                                      hair_displacement(points[index:index+1] - step, settings)[0]) / (4e-7)
            assert_allclose(covariance(corrected[index]), jacobian @ covariance(data[index]) @ jacobian.T,
                            rtol=2e-7, atol=1e-11)
        self.assertGreater(audit['minJacobianDeterminant'], 0)
        unchanged, _ = deform_head_hair(data, {'amount': 0})
        assert_array_equal(unchanged, data)


class TransformTests(unittest.TestCase):
    def test_crossing_hardware_assignment_uses_source_rows_and_rejects_spill(self):
        parts=[('upper',),('lower',),('hand',)]
        labels=np.array([1,2,0,1],np.uint8);indices=np.array([91,7,80,5],np.uint32)
        selections=[np.array([80,91],np.uint32)]
        rules=[{'part':'upper','allowedParts':['upper','lower']}]
        result=pipeline.assign_source_parts(labels,indices,selections,rules,parts)
        assert_array_equal(result,[0,2,0,1]);assert_array_equal(labels,[1,2,0,1])
        with self.assertRaisesRegex(ValueError,'allowed body parts'):
            pipeline.assign_source_parts(labels,indices,[np.array([7],np.uint32)],rules,parts)
        with self.assertRaisesRegex(ValueError,'overlap'):
            pipeline.assign_source_parts(labels,indices,selections*2,rules*2,parts)

    def test_joint_boundary_splits_only_neighboring_parts_inside_collar(self):
        points=np.array([[0,0,-.01],[0,0,.01],[0,0,.02],[.2,0,-.02],[0,0,.2]])
        labels=np.array([1,0,2,1,0],np.uint8)
        landmarks={'collar':[0,0,-.03]}
        rule={'proximalPart':'upper','distalPart':'lower','pivot':[0,0,0],'axis':[0,0,1],
              'maximumAbsStation':.11,'maximumRadius':.11,'landmark':'collar'}
        result,updated=pipeline.apply_joint_boundaries(points,labels,landmarks,[rule],[('upper',),('lower',),('other',)])
        assert_array_equal(result,[0,1,2,1,0])
        assert_array_equal(labels,[1,0,2,1,0])
        self.assertEqual(updated['collar'],[0,0,0])
        self.assertEqual(landmarks['collar'],[0,0,-.03])
        with self.assertRaises(ValueError):
            pipeline.apply_joint_boundaries(points,labels,landmarks,[dict(rule,axis=[0,0,0])],[('upper',),('lower',)])

    def test_axial_collar_keeps_axis_fixed_and_rejects_an_independent_bend(self):
        axis=np.array([-.62,.18,.76]);axis/=np.linalg.norm(axis)
        pivot=np.array([-.35,-.04,.28]);parent_R=Rotation.from_euler('xyz',[10,-5,177],degrees=True).as_matrix();parent_t=np.array([.02,-.08,.37])
        constraint={'parentPart':'parent','axisFrontRaw':axis.tolist(),'pivotFrontRaw':pivot.tolist(),'twistDegrees':25.}
        R,t=pipeline.axial_child_transform(parent_R,parent_t,constraint)
        # All points on the parent-aligned collar axis stay there under any twist.
        target=pivot+np.array([-.2,0,.3])[:,None]*axis
        source=(target-parent_t)@parent_R
        assert_allclose(pipeline.apply_transform(source,R,t),target,atol=1e-12)
        hashes={'front':'f','back':'b'}
        def entry(r,translation):return {'sourceToFrontRaw':{'rotation':r.tolist(),'translation':translation.tolist(),'scale':1}}
        doc={'version':1,'sourceHashes':hashes,'parts':{'parent':entry(parent_R,parent_t),'child':{**entry(R,t),'mechanicalConstraint':constraint}}}
        parts=[('parent',),('child',)]
        pipeline.validate_feature_transforms(doc,hashes,parts,1)
        doc['parts']['child']['sourceToFrontRaw']['rotation']=(Rotation.from_rotvec([.02,0,0]).as_matrix()@R).tolist()
        with self.assertRaisesRegex(ValueError,'axial collar constraint'):
            pipeline.validate_feature_transforms(doc,hashes,parts,1)

    def test_transform_means_and_full_anisotropic_covariances(self):
        data = gaussians([[.7, -.1, 2], [-.6, .8, 1.1], [0, 0, 0]])
        original_rotations = Rotation.from_euler('xyz', [[20, 35, -17], [80, -15, 110], [-8, 65, 20]], degrees=True)
        xyzw = original_rotations.as_quat()
        data[:, 3:7] = np.c_[xyzw[:, 3], xyzw[:, :3]]
        untouched = data.copy()
        rotation = Rotation.from_euler('xyz', [37, -61, 28], degrees=True).as_matrix()
        translation = np.array([-.7, 2.1, .4])
        scale = 1.7
        transformed = pipeline.transform_gaussians(data, rotation, translation, scale)
        assert_allclose(transformed[:, :3], scale * data[:, :3] @ rotation.T + translation, atol=1e-12)
        assert_allclose(np.linalg.norm(transformed[:, 3:7], axis=1), 1, atol=1e-12)
        for before, after in zip(data, transformed):
            assert_allclose(covariance(after), scale**2 * rotation @ covariance(before) @ rotation.T, atol=1e-12)
        assert_array_equal(transformed[:, 10:], data[:, 10:])
        assert_array_equal(data, untouched)

    def test_zero_and_unnormalized_quaternions_are_normalized(self):
        data = gaussians([[0, 0, 0], [1, 2, 3]])
        data[0, 3:7] = 0
        data[1, 3:7] = [3, 0, 0, 0]
        rotation = Rotation.from_euler('z', 70, degrees=True).as_matrix()
        transformed = pipeline.transform_gaussians(data, rotation, np.zeros(3))
        for record in transformed:
            assert_allclose(np.linalg.norm(record[3:7]), 1, atol=1e-12)
            quat = np.r_[record[4:7], record[3]]
            assert_allclose(Rotation.from_quat(quat).as_matrix(), rotation, atol=1e-12)

    def test_fit_rigid_recovers_weighted_pose_and_never_reflects(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [-.2, .5, 2]])
        expected_rotation = Rotation.from_euler('xyz', [25, 48, -71], degrees=True).as_matrix()
        expected_translation = np.array([-.3, .5, 1.8])
        target = source @ expected_rotation.T + expected_translation
        target[-1] += 1000  # A zero-weight outlier must not move the solution.
        rotation, translation = pipeline.fit_rigid(source, target, [1, 3, 2, 4, 0])
        assert_allclose(rotation, expected_rotation, atol=1e-12)
        assert_allclose(translation, expected_translation, atol=1e-12)
        mirrored = source @ np.diag([-1., 1., 1.])
        rotation, translation = pipeline.fit_rigid(source, mirrored)
        self.assertAlmostEqual(np.linalg.det(rotation), 1., places=12)
        assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.norm(source @ rotation.T + translation - mirrored), .1)

    def test_align_vectors_handles_parallel_and_antiparallel_axes(self):
        for first, second in [([1., 0, 0], [-1., 0, 0]),
                              ([.3, -.7, .4], [-.3, .7, -.4]),
                              ([0., 2, 0], [0., 5, 0]),
                              ([1., 2, 3], [-2., 1, 1])]:
            first, second = np.array(first), np.array(second)
            rotation = pipeline.align_vectors(first, second)
            assert_allclose(rotation @ (first / np.linalg.norm(first)), second / np.linalg.norm(second), atol=1e-12)
            assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1., places=12)


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.landmarks = {'a': [0, 0, 0], 'b': [0, 0, 1]}
        self.part = ('test', 'Test limb', 'a', 'b', .15, .15)
        self.settings = {'feather': .015, 'donorColumnRadius': .025}

    def test_opposing_shells_remain_separate_and_keep_their_coverage(self):
        front = gaussians([[0, -.12, .2], [.05, -.12, .7]])
        back = gaussians([[0, .12, .2], [.05, .12, .7]])
        fw, bw = pipeline.coverage_weights(front, back, self.landmarks, self.part, self.settings)
        self.assertTrue(np.all(fw > .999))
        self.assertTrue(np.all(bw > .999))
        output_front, fkeep = pipeline.attenuate(front, fw, .005)
        output_back, bkeep = pipeline.attenuate(back, bw, .005)
        self.assertTrue(fkeep.all() and bkeep.all())
        assert_array_equal(output_front[:, :3], front[:, :3])
        assert_array_equal(output_back[:, :3], back[:, :3])
        assert_allclose(output_back[:, 1] - output_front[:, 1], .24, atol=1e-12)

    def test_missing_donor_column_preserves_unique_surface(self):
        # Two front points are on the normally discarded half. Only one has a
        # matching preferred back column, so only that one can be attenuated.
        front = gaussians([[0, .1, .2], [.3, .1, .2]])
        back = gaussians([[0, .1, .2]])
        fw, bw = pipeline.coverage_weights(front, back, self.landmarks, self.part, self.settings)
        self.assertLess(fw[0], .002)
        self.assertEqual(fw[1], 1.)
        self.assertEqual(bw[0], 1.)
        unique, keep = pipeline.attenuate(front, fw, .01)
        assert_array_equal(keep, [False, True])
        assert_array_equal(unique[:, :3], front[1:, :3])

    def test_empty_source_keeps_other_source_and_supported_seam_feathers(self):
        front = gaussians([[0, 0, .5], [0, -.12, .5]])
        empty = gaussians(np.empty((0, 3)))
        fw, bw = pipeline.coverage_weights(front, empty, self.landmarks, self.part, self.settings)
        assert_array_equal(fw, [1, 1])
        self.assertEqual(len(bw), 0)
        back = gaussians([[0, 0, .5], [0, .12, .5]])
        fw, bw = pipeline.coverage_weights(front, back, self.landmarks, self.part, self.settings)
        self.assertEqual(fw[0], .5)
        self.assertEqual(bw[0], .5)

    def test_part_quality_bias_prefers_crisp_source_but_preserves_uncovered_back(self):
        front=gaussians([[0,.02,.5]])
        back=gaussians([[0,.02,.5],[.3,.1,.5]])
        settings={**self.settings,'partOverrides':{'test':{'depthBias':.06}}}
        fw,bw=pipeline.coverage_weights(front,back,self.landmarks,self.part,settings)
        self.assertGreater(fw[0],.9)
        self.assertLess(bw[0],.1)
        self.assertEqual(bw[1],1.)

    def test_inspected_ghost_source_can_disable_unique_coverage_fallback_locally(self):
        front=gaussians([[0,-.1,.5]])
        back=gaussians([[.3,-.1,.5],[0,.1,.5]])
        settings={**self.settings,'partOverrides':{'test':{'preserveUniqueBack':False}}}
        fw,bw=pipeline.coverage_weights(front,back,self.landmarks,self.part,settings)
        self.assertGreater(fw[0],.99)
        self.assertLess(bw[0],.002)
        self.assertGreater(bw[1],.99)

    def test_attenuation_uses_optical_density_not_linear_alpha(self):
        alpha = np.array([.2, .75, .95, .6])
        weight = np.array([.5, .25, 1., .001])
        data = gaussians([[0, 0, 0]] * 4)
        data[:, 10] = np.log(alpha / (1-alpha))
        out, keep = pipeline.attenuate(data, weight, .01)
        assert_array_equal(keep, [True, True, True, False])
        actual_alpha = 1 / (1 + np.exp(-out[:, 10]))
        expected_alpha = 1 - (1-alpha[:3])**weight[:3]
        assert_allclose(actual_alpha, expected_alpha, atol=1e-12)
        self.assertGreater(abs(actual_alpha[0] - alpha[0] * weight[0]), .005)
        half, _ = pipeline.attenuate(gaussians([[0, 0, 0]], alpha=.75), np.array([.5]), .01)
        half_alpha = 1 / (1 + np.exp(-half[0, 10]))
        self.assertAlmostEqual(1-(1-half_alpha)**2, .75, places=12)


class SegmentationTests(unittest.TestCase):
    def test_cervical_planes_keep_whole_sleeve_and_respect_radial_support(self):
        parts=[('head','Head','a','b',.1,.1),('neck','Neck','b','c',.1,.1),('torso','Torso','c','d',.2,.2)]
        spec={'referenceTransform':{'rotation':np.eye(3).tolist(),'translation':[0,0,0],'scale':1},
              'skullPlane':{'point':[0,0,0],'normal':[0,0,1]},
              'cuffPlane':{'point':[0,0,.1],'normal':[0,0,1]},
              'cuffRadialCenter':[0,0],'cuffRadialRadii':[.1,.1]}
        points=np.array([[0,0,-.05],[0,0,.05],[0,0,.15],[.3,0,.05]])
        labels=pipeline.cervical_labels(points,np.array([1,0,1,2]),spec,parts)
        assert_array_equal(labels,[0,1,2,2])

    def test_rigid_hardware_override_keeps_whole_pad_without_grabbing_neighbors(self):
        landmarks={'a':[0,0,0],'b':[0,0,1],'c':[.2,0,0],'d':[.2,0,1]}
        parts=[('torso','Torso','a','b',.2,.2),('arm','Arm','c','d',.1,.1)]
        points=np.array([[.07,0,.5],[.13,0,.5],[.07,.09,.5],[.07,0,.8]])
        rules=[{'part':'arm','center':[.1,0,.5],'axis':[0,1,0],'radius':.05,'halfLength':.02}]
        labels,_=pipeline.segment_points(points,landmarks,parts,rules)
        assert_array_equal(labels,[1,1,0,0])
        ellipse=[{'part':'arm','type':'orientedEllipseSlab','center':[.1,0,.5],
                  'axes':[[1,0,0],[0,0,1],[0,1,0]],'radii':[.05,.04],'halfDepth':.02}]
        ellipse_labels,_=pipeline.segment_points(points,landmarks,parts,ellipse)
        assert_array_equal(ellipse_labels,labels)

    def test_skull_and_slanted_neck_use_separate_anatomical_anchors(self):
        landmarks = {'neck': [0, .06, .14], 'skull_base': [0, 0, 0],
                     'crown': [.015, -.02, -.25]}
        parts = [('head', 'Skull / face', 'skull_base', 'crown', .13, .145),
                 ('neck', 'Neck', 'neck', 'skull_base', .075, .08)]
        skull = (np.array(landmarks['skull_base']) + landmarks['crown']) / 2
        neck = (np.array(landmarks['neck']) + landmarks['skull_base']) / 2
        labels, distance = pipeline.segment_points(np.array([skull, neck]), landmarks, parts)
        assert_array_equal(labels, [0, 1])
        assert_allclose(distance, 0, atol=1e-12)

    def test_filtering_preserves_original_vertex_indices_through_exclusions(self):
        points = [[x, -.04, z] for x in np.linspace(-.008, .008, 5)
                  for z in np.linspace(.1, .116, 5)]
        data = gaussians(points)
        data[0, 0] = np.nan
        data[1, 10] = -20
        landmarks = {'a': [0, 0, 0], 'b': [0, 0, 1]}
        parts = [('test', 'Test', 'a', 'b', .1, .1)]
        settings = {'minOpacity': .025, 'minScale': .000015, 'maxScale': .04,
                    'anatomyRadiusMultiplier': 1.8, 'densityVoxel': .0025,
                    'isolatedSpacing': .026, 'contactDepth': .005, 'contactSpacing': .012}
        excluded = [{'min': [.006, -.1, 0], 'max': [.01, .1, 1]}]
        cleaned, labels, stats, source_indices = pipeline.clean_scan(data, landmarks, settings, parts, excluded)
        assert_array_equal(source_indices, np.arange(2, 20))
        assert_array_equal(cleaned, data[source_indices])
        assert_array_equal(labels, np.zeros(18, dtype=np.uint8))
        self.assertEqual(stats['retained'], 18)
        data[4,7:10]=np.log(.030)
        cleaned,labels,stats,source_indices=pipeline.clean_scan(data,landmarks,settings,parts,excluded,part_filters={'test':{'maxScale':.028}})
        self.assertNotIn(4,source_indices)
        self.assertEqual(stats['partQualityRejected'],1)
        assert_array_equal(cleaned,data[source_indices])

    def test_native_segmentation_distinguishes_depth_not_only_projection(self):
        landmarks = {'a': [0, 0, 0], 'b': [0, 0, 1],
                     'c': [0, .3, 0], 'd': [0, .3, 1]}
        parts = [('front', 'Front depth', 'a', 'b', .1, .1),
                 ('back', 'Back depth', 'c', 'd', .1, .1)]
        points = np.array([[0, 0, .5], [0, .3, .5], [0, .28, .5], [.04, 0, .5]])
        labels, distance = pipeline.segment_points(points, landmarks, parts)
        assert_array_equal(labels, [0, 1, 1, 0])
        assert_allclose(distance, [0, 0, .2, .4], atol=1e-12)

    def test_distal_hand_and_foot_beat_proximal_capsules(self):
        landmarks = {'elbow': [0, 0, 0], 'wrist': [0, 0, .4], 'tip': [0, 0, .6],
                     'knee': [.5, 0, 0], 'ankle': [.5, 0, .4], 'toe': [.5, 0, .6]}
        parts = [('forearm', 'Forearm', 'elbow', 'wrist', .07, .06),
                 ('hand', 'Hand', 'wrist', 'tip', .06, .04),
                 ('shin', 'Shin', 'knee', 'ankle', .09, .08),
                 ('foot', 'Foot', 'ankle', 'toe', .07, .05)]
        points = np.array([[0, 0, .2], [0, 0, .52], [.5, 0, .2], [.5, 0, .52], [0, 0, .62]])
        labels, distance = pipeline.segment_points(points, landmarks, parts)
        assert_array_equal(labels, [0, 1, 2, 3, 1])
        self.assertLess(distance[-1], 1.)


class RigidPartConstraintTests(unittest.TestCase):
    parts = [('right_hand','Hand','wrist','tip',.1,.1),
             ('right_forearm','Forearm','elbow','wrist',.1,.1),
             ('left_hand','Left hand','wrist','tip',.1,.1)]

    def test_rigid_child_rejects_free_rotation_translation_twist_and_warps(self):
        base={'rigidParts':{'right_hand':'right_forearm'}}
        self.assertEqual(pipeline.validate_rigid_parts(base,self.parts),base['rigidParts'])
        for key,value in [('rotationDegrees',[0,.01,0]),('translation',[0,0,1e-9]),
                          ('twistDegrees',.1),('twistDegrees',float('nan'))]:
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'disallows independent'):
                pipeline.validate_rigid_parts({**base,'adjustments':{'right_hand':{key:value}}},self.parts)
        with self.assertRaisesRegex(ValueError,'rightWristRegistration'):
            pipeline.validate_rigid_parts({**base,'rightWristRegistration':{'enabled':True}},self.parts)
        left={'rigidParts':{'left_hand':'right_forearm'},'leftHandDigitAlignment':{'amount':.0033}}
        with self.assertRaisesRegex(ValueError,'leftHandDigitAlignment'):
            pipeline.validate_rigid_parts(left,self.parts)
        left['leftHandDigitAlignment']['amount']=0
        pipeline.validate_rigid_parts(left,self.parts)
        # An unrelated child's deformation remains available for old reproductions.
        pipeline.validate_rigid_parts({**base,'leftHandDigitAlignment':{'amount':.0033}},self.parts)

    def test_rigid_mapping_rejects_cycles_and_unknown_parents(self):
        for mapping in [[],{'right_hand':'missing'},{'right_hand':'right_hand'},
                        {'right_hand':'right_forearm','right_forearm':'right_hand'}]:
            with self.subTest(mapping=mapping),self.assertRaises(ValueError):
                pipeline.validate_rigid_parts({'rigidParts':mapping},self.parts)
        order=pipeline.alignment_order(self.parts,{'right_hand':'right_forearm'}, {},True)
        self.assertEqual(order,[1,0,2])

    def test_pipeline_uses_changed_effective_parent_even_when_icp_is_skipped(self):
        from verify_output import verify_rigid_part_constraints
        # Parent deliberately follows child in label order. A contradictory saved
        # child fit must never introduce a second pose, in either alignment mode.
        parts=self.parts[:2];landmarks={'elbow':[-.3,0,0],'wrist':[0,0,0],'tip':[.15,0,.05]}
        data=gaussians([[.02,0,0],[.05,0,0],[-.20,0,0],[-.25,0,0]])
        labels=np.array([0,0,1,1],np.uint8);source_indices=np.arange(4,dtype=np.uint32)
        initial_R=Rotation.from_euler('xyz',[3,6,-8],degrees=True).as_matrix();initial_t=np.array([.03,-.01,.02])
        feature_R=Rotation.from_euler('xyz',[11,-4,29],degrees=True).as_matrix();feature_t=np.array([-.02,.04,.01])
        child_R=Rotation.from_euler('xyz',[-40,31,12],degrees=True).as_matrix()
        for skip_icp,parent_shift in [(False,[.04,.02,-.01]),(False,[-.06,.03,.02]),(True,[.04,.02,-.01])]:
            with self.subTest(skip_icp=skip_icp,parent_shift=parent_shift),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                for source in ['front','back']:(root/(source+'.json')).write_text(json.dumps({'landmarks':landmarks}))
                def entry(R,t):return {'sourceToFrontRaw':{'scale':.928,'rotation':R.tolist(),'translation':t.tolist()}}
                features={'version':1,'sourceHashes':{'front':'f','back':'b'},'parts':{
                    'right_forearm':entry(feature_R,feature_t),
                    'right_hand':{**entry(child_R,np.array([.3,-.4,.2])),'carriedByPart':'right_forearm'}}}
                (root/'features.json').write_text(json.dumps(features))
                config={'front':'front.ply','back':'back.ply','frontLandmarks':'front.json','backLandmarks':'back.json',
                        'featureTransforms':'features.json','filter':{},'fusion':{'minWeight':.001},'alignment':{},
                        'output':'output','review':'review','previewMaxPerSource':20,
                        'rigidParts':{'right_hand':'right_forearm'},
                        'adjustments':{'right_forearm':{'rotationDegrees':[7,-3,5],'translation':parent_shift},
                                       'right_hand':{'rotationDegrees':[0,0,0],'translation':[0,0,0],'twistDegrees':0}}}
                config_path=root/'config.json';config_path.write_text(json.dumps(config))
                def read(path):
                    name=path.stem
                    return data.copy(),{'file':path.name,'sha256':'f' if name=='front' else 'b','count':len(data)}
                def clean(scan,*args):return scan.copy(),labels.copy(),{'retained':len(scan)},source_indices.copy()
                args=['pipeline.py','--config',str(config_path)]+(['--skip-icp'] if skip_icp else [])
                with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'BODY_PARTS',parts),\
                     patch.object(pipeline,'read_scan',side_effect=read),patch.object(pipeline,'clean_scan',side_effect=clean),\
                     patch.object(pipeline,'initialize_transforms',return_value=(.928,[(child_R,np.zeros(3)),(initial_R,initial_t)],{'scale':.928})),\
                     patch.object(pipeline,'coverage_weights',side_effect=lambda f,b,*args:(np.ones(len(f)),np.ones(len(b)))),\
                     patch.object(pipeline,'refine_overlap',side_effect=AssertionError('Rigid child must not run ICP')),\
                     patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):
                    report=pipeline.main()
                child,parent=report['parts'];self.assertEqual([child['id'],parent['id']],['right_hand','right_forearm'])
                self.assertEqual(child['rigidWithPart'],'right_forearm')
                self.assertEqual(child['sourceToFrontRaw'],parent['sourceToFrontRaw'])
                self.assertNotIn('carriedByPart',child)
                self.assertEqual(verify_rigid_part_constraints(report),config['rigidParts'])
                base_R,base_t=(initial_R,initial_t) if skip_icp else (feature_R,feature_t)
                Q=Rotation.from_euler('xyz',[7,-3,5],degrees=True).as_matrix();pivot=np.array([-.15,0,0])
                expected_R=Q@base_R;expected_t=Q@(base_t-pivot)+pivot+parent_shift
                assert_allclose(child['sourceToFrontRaw']['rotation'],expected_R,atol=1e-14)
                assert_allclose(child['sourceToFrontRaw']['translation'],expected_t,atol=1e-14)
                columns,_,_=pipeline.read_ply(root/'output/mannequin_fused.ply')
                exported=np.column_stack([columns[field] for field in pipeline.FIELDS])
                saved_labels=np.load(root/'output/part-labels.npy');saved_capture=np.load(root/'output/capture-labels.npy')
                expected=pipeline.transform_gaussians(data[:2],expected_R,expected_t,.928)
                expected=pipeline.transform_gaussians(expected,np.diag([1.,-1.,-1.]),np.zeros(3))
                assert_allclose(exported[(saved_labels==0)&(saved_capture==1),:10],expected[:,:10],atol=3e-7)

    def test_verifier_rejects_independent_child_pose_and_hidden_warp(self):
        from copy import deepcopy
        from verify_output import verify_rigid_part_constraints
        pose={'scale':.928,'rotation':np.eye(3).tolist(),'translation':[.1,.2,.3]}
        report={'parameters':{'rigidParts':{'right_hand':'right_forearm'}},'parts':[
            {'id':'right_forearm','sourceToFrontRaw':pose},
            {'id':'right_hand','rigidWithPart':'right_forearm','sourceToFrontRaw':deepcopy(pose)}]}
        verify_rigid_part_constraints(report)
        for mutation in ['pose','warp','manual','marker']:
            broken=deepcopy(report)
            if mutation=='pose':broken['parts'][1]['sourceToFrontRaw']['translation'][0]+=.001
            elif mutation=='warp':broken['parts'][1]['wristRegistration']={'affected':1}
            elif mutation=='manual':broken['parameters']['adjustments']={'right_hand':{'twistDegrees':.1}}
            else:broken['parts'][1].pop('rigidWithPart')
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):verify_rigid_part_constraints(broken)


class WristPadTransitionTests(unittest.TestCase):
    def parameters(self):
        return {'referenceForearm':{'rotation':np.eye(3).tolist(),'translation':[0,0,0],'scale':1},
                'referenceHand':{'rotation':Rotation.from_euler('y',20,degrees=True).as_matrix().tolist(),
                                 'translation':[.005,0,-.003],'scale':1},
                'origin':[0,0,0],'longitudinalAxis':[0,0,1], 'longitudinalFade':[.065,.125]}

    def test_pad_is_exactly_rigid_and_distal_endpoint_matches_prior_pose(self):
        from right_wrist_transition import preserve_pad_align_fingers
        parameters=self.parameters();before=gaussians([[.004,0,.03],[.003,.01,.16]])
        after,_=preserve_pad_align_fingers(before,parameters)
        assert_array_equal(before[0],after[0])
        expected=pipeline.transform_gaussians(before,np.array(parameters['referenceHand']['rotation']),np.array(parameters['referenceHand']['translation']))
        assert_allclose(after[1,:3],expected[1,:3],atol=1e-15)
        assert_allclose(covariance(after[1]),covariance(expected[1]),atol=1e-15)
        assert_array_equal(after[:,10:],before[:,10:])

    def test_analytic_covariance_matches_numerical_map_in_transition(self):
        from right_wrist_transition import preserve_pad_align_fingers, displacement_and_jacobian
        parameters=self.parameters();point=np.array([.012,-.008,.094]);before=gaussians([point])
        after,_=preserve_pad_align_fingers(before,parameters)
        numerical=np.eye(3);epsilon=1e-7
        for axis in range(3):
            step=np.eye(3)[axis]*epsilon
            plus=displacement_and_jacobian([point+step],parameters)[0][0]
            minus=displacement_and_jacobian([point-step],parameters)[0][0]
            numerical[:,axis]+=(plus-minus)/(2*epsilon)
        assert_allclose(covariance(after[0]),numerical@covariance(before[0])@numerical.T,atol=1e-12)

    def test_new_parent_pose_carries_pad_and_fingers_together(self):
        from right_wrist_transition import preserve_pad_align_fingers
        parameters=self.parameters();before=gaussians([[0,0,.025],[0,0,.16]])
        corrected,_=preserve_pad_align_fingers(before,parameters)
        rotation=Rotation.from_euler('xyz',[12,-23,8],degrees=True).as_matrix();translation=np.array([.3,-.2,.1])
        after=pipeline.transform_gaussians(corrected,rotation,translation)
        assert_allclose(after[:,:3],pipeline.rotate(corrected[:,:3],rotation)+translation,atol=1e-15)
        for i in range(len(after)):
            assert_allclose(covariance(after[i]),rotation@covariance(corrected[i])@rotation.T,atol=1e-15)


class FingerAlignmentTests(unittest.TestCase):
    def test_wrist_thumb_and_distal_thickness_are_preserved(self):
        from hand_digits import align_left_digits, settings
        c=settings({});u=c['longitudinalAxis'];v=c['lateralAxis'];origin=c['origin']
        points=np.array([origin, origin+.15*u+.07*v,
                         origin+.16*u+[0,-.008,0], origin+.16*u+[0,.008,0]])
        before=gaussians(points);after,report=align_left_digits(before,{})
        assert_array_equal(after[:2],before[:2])
        assert_allclose(after[2:,:3]-before[2:,:3],np.tile(.0033*u,(2,1)),atol=1e-15)
        assert_allclose(after[3,:3]-after[2,:3],before[3,:3]-before[2,:3],atol=1e-15)
        assert_array_equal(after[:,3:],before[:,3:])
        self.assertEqual(report['maximumDepthDisplacement'],0)

    def test_taper_covariance_matches_numerical_displacement_derivative(self):
        from hand_digits import align_left_digits, displacement_and_jacobian, settings
        c=settings({});u=c['longitudinalAxis'];v=c['lateralAxis']
        p=c['origin']+.104*u+.048*v;before=gaussians([p])
        before[0,3:7]=np.r_[.7,.2,-.1,.3]/np.linalg.norm([.7,.2,-.1,.3])
        after,report=align_left_digits(before,{})
        numerical=np.eye(3);epsilon=1e-7
        for axis in range(3):
            step=np.eye(3)[axis]*epsilon
            plus=displacement_and_jacobian([p+step],{})[0][0]
            minus=displacement_and_jacobian([p-step],{})[0][0]
            numerical[:,axis]+=(plus-minus)/(2*epsilon)
        assert_allclose(covariance(after[0]),numerical@covariance(before[0])@numerical.T,atol=1e-12)
        assert_array_equal(after[:,10:],before[:,10:])
        self.assertGreater(report['minJacobianDeterminant'],0)
        self.assertLess(report['maxJacobianStretch'],1.2)

    def test_zero_amount_is_exact_identity_and_invalid_amount_rejected(self):
        from hand_digits import align_left_digits
        before=gaussians([[.81,.08,.68]])
        after,_=align_left_digits(before,{'amount':0})
        assert_array_equal(after,before)
        with self.assertRaises(ValueError):align_left_digits(before,{'amount':.04})


class FeatureTransformTests(unittest.TestCase):
    def test_source_guard_and_rigid_rotation_validation(self):
        from copy import deepcopy
        hashes={'front':'first','back':'second'}
        parts=[('head','Head','a','b',.1,.1)]
        entry={'sourceToFrontRaw':{'rotation':np.eye(3).tolist(),'translation':[1,2,3],'scale':.9}}
        document={'version':1,'sourceHashes':hashes,'parts':{'head':entry}}
        result=pipeline.validate_feature_transforms(document,hashes,parts,.9)
        assert_allclose(result['head'][1],[1,2,3])
        with self.assertRaises(ValueError):
            pipeline.validate_feature_transforms(document,{'front':'changed','back':'second'},parts,.9)
        wrong=deepcopy(document);wrong['parts']['head']['sourceToFrontRaw']['rotation'][0][0]=-1
        with self.assertRaises(ValueError):pipeline.validate_feature_transforms(wrong,hashes,parts,.9)
        with self.assertRaises(ValueError):pipeline.validate_feature_transforms(document,hashes,parts,1.)


if __name__ == '__main__':
    unittest.main()
