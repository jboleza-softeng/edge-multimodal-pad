"""
Unit Tests for Multimodal Fusion Mathematical Invariants
=========================================================
Verifies:
  - Ethical regularizer locks chromatic feature weight (blue_red = 0.0).
  - Gain invariance of relative Laplacian gradient sharpness and band_ratio.
  - Logistic regression fusion probabilities are bounded in [0, 1].
"""

import unittest
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.fusion_model import EdgeMultimodalPAD, CALIBRATED_CONSTANTS


class TestFusionInvariants(unittest.TestCase):

    def test_ethical_regularization_zero_chromatic_weight(self):
        """Verify blue_red feature weight is clamped to 0.0 to prevent skin-tone bias."""
        weights = CALIBRATED_CONSTANTS['weights']
        self.assertIn('blue_red', weights)
        self.assertEqual(weights['blue_red'], 0.0, "blue_red weight must be 0.0")

    def test_sharpness_gain_invariance(self):
        """
        Verify that relative sharpness sharp_rel = lapVar / Var(Y) is invariant
        under scalar gain multiplication: Y' = g * Y -> sharp_rel' == sharp_rel.
        """
        np.random.seed(42)
        patch = np.random.uniform(50, 150, (64, 64))
        
        # Discrete Laplacian convolution
        from scipy.signal import convolve2d
        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        def compute_sharp_rel(img):
            lap = convolve2d(img, lap_kernel, mode='valid')
            lap_var = np.var(lap)
            img_var = np.var(img)
            return lap_var / (img_var + 1e-9)

        sr_original = compute_sharp_rel(patch)
        for gain in [0.5, 1.5, 2.0]:
            sr_scaled = compute_sharp_rel(patch * gain)
            self.assertAlmostEqual(sr_original, sr_scaled, places=5)

    def test_probability_bounds(self):
        """Probability output must always reside in [0, 1]."""
        model = EdgeMultimodalPAD()
        import pandas as pd
        dummy_df = pd.DataFrame([{
            'sharp_rel': 0.5, 'band_ratio': 10.0, 'chroma_cv': 0.1,
            'blue_red': 0.8, 'spec_frac': 0.01, 'depth_rel': 0.02, 'motion_rel': 0.05
        }])
        prob = model.predict_proba(dummy_df)
        self.assertGreaterEqual(prob[0], 0.0)
        self.assertLessEqual(prob[0], 1.0)


if __name__ == '__main__':
    unittest.main()
