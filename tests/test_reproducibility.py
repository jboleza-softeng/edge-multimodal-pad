"""
Regression and Reproducibility Verification
===========================================
Strict verification test asserting:
  1. The held-out test split (S11–S15, N = 75) achieves 0 errors:
     - APCER == 0.00%
     - BPCER == 0.00%
     - ACER == 0.00%
  2. The empirical score separation margin Delta_gap is positive and >= +0.48.
  3. Calibrated threshold matches tau = 0.4894.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.metrics import evaluate_threshold
from src.fusion_model import CALIBRATED_CONSTANTS


class TestReproducibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, 'data', 'benchmark_trials_225n.csv')
        cls.assertTrue(os.path.exists(cls.data_path), f"Missing data file: {cls.data_path}")
        cls.df = pd.read_csv(cls.data_path)

        if 'corpus_split' in cls.df.columns:
            cls.test_df = cls.df[cls.df['corpus_split'] == 'test'].copy()
        else:
            cls.test_df = cls.df[cls.df['subject_id'].isin(['S11', 'S12', 'S13', 'S14', 'S15'])].copy()

    def test_held_out_cohort_size(self):
        """Verify held-out split comprises exactly 75 trials across 5 subjects."""
        self.assertEqual(len(self.test_df), 75)
        self.assertEqual(self.test_df['subject_id'].nunique(), 5)
        self.assertEqual(sum(self.test_df['ground_truth_is_live'] == 1), 25)
        self.assertEqual(sum(self.test_df['ground_truth_is_live'] == 0), 50)

    def test_zero_classification_errors_held_out(self):
        """Verify APCER = 0.00%, BPCER = 0.00%, ACER = 0.00% on unseen test subjects."""
        tau = CALIBRATED_CONSTANTS['tau']
        y_true = self.test_df['ground_truth_is_live'].values.astype(int)
        scores = self.test_df['fused_score'].values.astype(float)
        categories = self.test_df['sample_category'].values

        res = evaluate_threshold(y_true, scores, threshold=tau, categories=categories)
        self.assertEqual(res['fn_errors'], 0, "Bona fide false rejection must be 0")
        self.assertEqual(res['fp_errors'], 0, "Attack false acceptance must be 0")
        self.assertEqual(res['bpcer'], 0.0)
        self.assertEqual(res['apcer'], 0.0)
        self.assertEqual(res['acer'], 0.0)

    def test_score_separation_margin(self):
        """Verify empirical score separation margin Delta_gap >= +0.48."""
        y_true = self.test_df['ground_truth_is_live'].values.astype(int)
        scores = self.test_df['fused_score'].values.astype(float)
        live_min = float(np.min(scores[y_true == 1]))
        attack_max = float(np.max(scores[y_true == 0]))

        delta_gap = live_min - attack_max
        self.assertAlmostEqual(live_min, 0.8836, places=3)
        self.assertAlmostEqual(attack_max, 0.3943, places=3)
        self.assertAlmostEqual(delta_gap, 0.4893, places=3)
        self.assertGreater(delta_gap, 0.48)


if __name__ == '__main__':
    unittest.main()
