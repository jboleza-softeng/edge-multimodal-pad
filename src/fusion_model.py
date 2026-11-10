"""
Lightweight Multimodal Presentation Attack Detection (PAD) Fusion Model
========================================================================
Implements the 5-stream gain-invariant feature extraction and logistic regression
fusion model proposed in the manuscript.

Features evaluated:
  1. band_ratio  : Spatial frequency energy persistence (dual-scale Laplacian ratio)
  2. sharp_rel   : Gain-invariant relative Laplacian texture gradient sharpness
  3. spec_frac   : Multi-channel indicator specular highlight saturation fraction
  4. depth_rel   : Normalized perspective depth disparity
  5. motion_rel  : Temporal optical micro-motion gradient
  6. blue_red    : Chromatic color-temperature balance (regularized to 0.0 weight)
  7. chroma_cv   : Chrominance coefficient of variation

Classification rule:
  z = (x - mu^(device)) / sigma^(device)
  score = 1 / (1 + exp(- (w^T z + b)))
  verdict = 'LIVE' if score >= tau else 'ATTACK'
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression


DEFAULT_FEATURES = [
    "sharp_rel",
    "band_ratio",
    "chroma_cv",
    "blue_red",
    "spec_frac",
    "depth_rel",
    "motion_rel",
]

# Paper calibrated constants
CALIBRATED_CONSTANTS = {
    "version": "calib-150n-5s-iso",
    "tau": 0.4894,
    "weights": {
        "sharp_rel": 0.757,
        "band_ratio": 1.333,
        "chroma_cv": 0.222,
        "blue_red": 0.000,    # Explicit ethical regularization to prevent demographic skin-tone bias
        "spec_frac": 1.120,
        "depth_rel": 0.412,
        "motion_rel": -0.537,
    },
    "intercept": -0.1245
}


class EdgeMultimodalPAD:
    """
    Inference and evaluation class for Edge Multimodal PAD.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        enforce_zero_chromatic_bias: bool = True,
        tau: float = 0.4894
    ):
        self.features = features or list(DEFAULT_FEATURES)
        self.enforce_zero_chromatic_bias = enforce_zero_chromatic_bias
        self.tau = tau
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.device_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.global_mu: Optional[np.ndarray] = None
        self.global_sigma: Optional[np.ndarray] = None

    def fit_calibration(
        self,
        df_calib: pd.DataFrame,
        device_col: str = 'device_label',
        label_col: str = 'ground_truth_is_live',
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Fit standardization parameters per device and logistic weights from calibration split.
        Enforces zero chromatic weight if enforce_zero_chromatic_bias is True.
        """
        # 1. Compute per-device statistics (mu, sigma)
        devices = df_calib[device_col].dropna().unique()
        for dev in devices:
            dev_df = df_calib[df_calib[device_col] == dev]
            X_dev = dev_df[self.features].values.astype(float)
            mu = np.nanmean(X_dev, axis=0)
            sigma = np.nanstd(X_dev, axis=0)
            sigma[sigma < 1e-6] = 1.0
            self.device_stats[str(dev)] = {'mu': mu, 'sigma': sigma}

        # Global fallback
        X_all = df_calib[self.features].values.astype(float)
        self.global_mu = np.nanmean(X_all, axis=0)
        self.global_sigma = np.nanstd(X_all, axis=0)
        self.global_sigma[self.global_sigma < 1e-6] = 1.0

        # Standardize calibration data
        Z_list = []
        for _, row in df_calib.iterrows():
            dev = str(row.get(device_col, ''))
            stats_d = self.device_stats.get(dev, {'mu': self.global_mu, 'sigma': self.global_sigma})
            x = row[self.features].values.astype(float)
            z = (x - stats_d['mu']) / stats_d['sigma']
            Z_list.append(z)
        Z_calib = np.array(Z_list)
        y_calib = df_calib[label_col].values.astype(int)

        # Fit logistic regression
        clf = LogisticRegression(penalty='l2', C=1.0, random_state=seed, max_iter=1000)
        clf.fit(Z_calib, y_calib)

        w = clf.coef_[0].copy()
        if self.enforce_zero_chromatic_bias:
            for feat_name in ['blue_red', 'blue_ratio']:
                if feat_name in self.features:
                    idx = self.features.index(feat_name)
                    w[idx] = 0.0

        self.weights = w
        self.intercept = float(clf.intercept_[0])

        return {
            'features': self.features,
            'weights': {f: float(w[i]) for i, f in enumerate(self.features)},
            'intercept': self.intercept,
            'devices': list(self.device_stats.keys())
        }

    def predict_proba(
        self,
        df: pd.DataFrame,
        device_col: str = 'device_label'
    ) -> np.ndarray:
        """
        Compute predicted P(Live) probability scores for each trial.
        """
        if self.weights is None:
            # Fall back to calibrated constants if fit_calibration was not called
            self.weights = np.array([CALIBRATED_CONSTANTS['weights'].get(f, 0.0) for f in self.features])
            self.intercept = CALIBRATED_CONSTANTS['intercept']

        scores = []
        for _, row in df.iterrows():
            dev = str(row.get(device_col, ''))
            stats_d = self.device_stats.get(dev)
            if stats_d is None:
                stats_d = {'mu': self.global_mu if self.global_mu is not None else np.zeros(len(self.features)),
                           'sigma': self.global_sigma if self.global_sigma is not None else np.ones(len(self.features))}

            x = row[self.features].values.astype(float)
            z = (x - stats_d['mu']) / stats_d['sigma']
            logit = float(np.dot(self.weights, z) + self.intercept)
            prob = 1.0 / (1.0 + np.exp(-logit))
            scores.append(prob)

        return np.array(scores)

    def evaluate_separation_margin(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        tau: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate score separation margins: Delta_gap = min(Live) - max(Attack).
        """
        tau_val = tau if tau is not None else self.tau
        live_scores = scores[y_true == 1]
        attack_scores = scores[y_true == 0]

        min_live = float(np.min(live_scores)) if len(live_scores) > 0 else 0.0
        max_attack = float(np.max(attack_scores)) if len(attack_scores) > 0 else 0.0
        mean_live = float(np.mean(live_scores)) if len(live_scores) > 0 else 0.0
        mean_attack = float(np.mean(attack_scores)) if len(attack_scores) > 0 else 0.0

        delta_gap = min_live - max_attack
        cushion_live = min_live - tau_val
        buffer_attack = tau_val - max_attack

        return {
            'tau': tau_val,
            'min_live': min_live,
            'max_attack': max_attack,
            'mean_live': mean_live,
            'mean_attack': mean_attack,
            'delta_gap': delta_gap,
            'cushion_live': cushion_live,
            'buffer_attack': buffer_attack
        }
