import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import egg_workflow as ew


class WorkflowChecks(unittest.TestCase):
    def test_mask_metrics_identity_and_disjoint(self):
        a=np.zeros((8,8),np.uint8); a[2:5,2:5]=255
        self.assertEqual(ew.mask_metrics(a,a),(1.0,1.0))
        b=np.zeros_like(a); b[5:7,5:7]=255
        self.assertEqual(ew.mask_metrics(a,b),(0.0,0.0))

    def test_metric_shape_mismatch_fails(self):
        with self.assertRaises(ValueError): ew.mask_metrics(np.zeros((2,2)),np.zeros((3,3)))

    def test_binary_mask(self):
        mask=np.array([[0,1,127,255]],np.uint8)
        np.testing.assert_array_equal(ew.binary_mask(mask),[[0,255,255,255]])
        with self.assertRaises(ValueError): ew.binary_mask(np.zeros((2,2,3),np.uint8))

    def test_manifest_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.txt"; p.write_text("missing.jpg\n")
            with self.assertRaises(FileNotFoundError): ew.read_manifest(p)

    def test_ellipse_proposal_geometry(self):
        image=np.zeros((220,220,3),np.uint8)
        cv2.ellipse(image,(110,110),(45,70),25,0,360,(230,230,230),-1)
        cfg=ew.Config(mser_min_area=100,mser_max_area_ratio=.8,min_area_ratio=.004,
                      max_area_ratio=.8,score_threshold=0.0)
        candidates,edges=ew.proposal_candidates(image,cfg)
        self.assertIsInstance(candidates,list)
        self.assertEqual(edges.shape,image.shape[:2])
        for c in candidates:
            self.assertEqual(c.mask.shape,image.shape[:2])
            self.assertTrue(set(np.unique(c.mask)).issubset({0,255}))

    def test_summary_empty(self):
        self.assertEqual(ew.summarize([]),{"n":0})


if __name__=="__main__": unittest.main()
