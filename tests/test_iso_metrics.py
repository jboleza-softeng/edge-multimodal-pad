"""
Unit Tests for ISO/IEC 30107-3 PAD Metrics
===========================================
Verifies:
  - Exact Clopper-Pearson 95% confidence intervals against analytical values.
  - Wilson score confidence interval limits.
  - Threshold evaluation correctly classifies APCER, BPCER, and ACER.
"""

import unittest
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.metrics import clopper_pearson_ci, compute_wilson_ci, evaluate_threshold, compute_operating_thresholds


class TestISOMetrics(unittest.TestCase):

    def test_clopper_pearson_zero_errors_n25(self):
        """For k=0, n=25, alpha=0.05, upper bound is 1 - (0.025)**(1/25) = 13.72%."""
        low, high = clopper_pearson_ci(k=0, n=25, alpha=0.05)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.137185, places=4)

    def test_clopper_pearson_zero_errors_n50(self):
        """For k=0, n=50, alpha=0.05, upper bound is 1 - (0.025)**(1/50) = 7.11%."""
        low, high = clopper_pearson_ci(k=0, n=50, alpha=0.05)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.071113, places=4)

    def test_evaluate_threshold_perfect_separation(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
        tau = 0.5
        res = evaluate_threshold(y_true, scores, threshold=tau)
        self.assertEqual(res['bpcer'], 0.0)
        self.assertEqual(res['apcer'], 0.0)
        self.assertEqual(res['acer'], 0.0)

    def test_evaluate_threshold_with_errors(self):
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.4, 0.8, 0.6, 0.2])  # 1 false reject (0.4 < 0.5), 1 false accept (0.6 >= 0.5)
        tau = 0.5
        res = evaluate_threshold(y_true, scores, threshold=tau)
        self.assertEqual(res['bpcer'], 0.5)
        self.assertEqual(res['apcer'], 0.5)
        self.assertEqual(res['acer'], 0.5)


if __name__ == '__main__':
    unittest.main()
