"""
ISO/IEC 30107-3 Standard Biometric PAD Evaluation Metrics
=========================================================
Adheres to ISO/IEC 30107-3: Information technology — Biometric presentation attack detection
— Part 3: Testing and reporting.

Metrics computed:
  - APCER (Attack Presentation Classification Error Rate / False Acceptance)
  - BPCER (Bona Fide Presentation Classification Error Rate / False Rejection)
  - ACER (Average Classification Error Rate = (APCER + BPCER) / 2)
  - EER (Equal Error Rate) and EER Threshold
  - AUC (Area Under the ROC Curve)
  - Exact Two-Sided Clopper-Pearson Binomial 95% Confidence Intervals
  - Wilson Score 95% Confidence Intervals
  - Operational Operating Points (APCER @ BPCER = 1%, APCER @ BPCER = 5%)
"""

import math
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute exact two-sided Clopper-Pearson binomial confidence interval.
    For k = 0, lower bound is exactly 0.0 and upper bound is 1 - (alpha/2)**(1/n).
    """
    if n <= 0:
        return (0.0, 1.0)
    
    # Lower bound
    if k == 0:
        low = 0.0
    else:
        low = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
        
    # Upper bound
    if k == n:
        high = 1.0
    else:
        high = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
        
    return (max(0.0, low), min(1.0, high))


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """Computes Wilson Score interval for a proportion k/n."""
    if n <= 0:
        return {'lower': 0.0, 'upper': 0.0, 'point': 0.0}
    p = k / n
    z = 1.95996  # 95% two-sided
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p + (z2 / (2.0 * n))) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    return {
        'point': round(p * 100.0, 2),
        'lower': max(0.0, round((center - half) * 100.0, 2)),
        'upper': min(100.0, round((center + half) * 100.0, 2))
    }


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    categories: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates APCER, BPCER, ACER, and per-attack breakdown at a fixed decision threshold tau.
    Convention under ISO/IEC 30107-3:
      y_true = 1 for bona_fide, 0 for attack
      Decision: score >= threshold -> classified as bona fide (live)
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    is_bona = (y_true == 1)
    is_attack = (y_true == 0)

    n_bona = int(np.sum(is_bona))
    n_attack = int(np.sum(is_attack))

    # Bona Fide Rejected (False Reject)
    fn = int(np.sum(is_bona & (scores < threshold)))
    bpcer = (fn / n_bona) if n_bona > 0 else 0.0

    # Presentation Attack Accepted (False Accept)
    fp = int(np.sum(is_attack & (scores >= threshold)))
    apcer = (fp / n_attack) if n_attack > 0 else 0.0

    acer = (apcer + bpcer) / 2.0

    bpcer_ci = clopper_pearson_ci(fn, n_bona)
    apcer_ci = clopper_pearson_ci(fp, n_attack)

    results = {
        'threshold': float(threshold),
        'n_bona': n_bona,
        'n_attack': n_attack,
        'fn_errors': fn,
        'fp_errors': fp,
        'bpcer': float(bpcer),
        'apcer': float(apcer),
        'acer': float(acer),
        'bpcer_cp95': bpcer_ci,
        'apcer_cp95': apcer_ci,
        'bpcer_pct': round(bpcer * 100.0, 4),
        'apcer_pct': round(apcer * 100.0, 4),
        'acer_pct': round(acer * 100.0, 4),
    }

    # Per-attack category breakdown
    if categories is not None:
        categories = np.asarray(categories)
        attack_types = [c for c in np.unique(categories[is_attack])]
        per_attack = {}
        for atype in attack_types:
            mask = (categories == atype) & is_attack
            n_sub = int(np.sum(mask))
            fp_sub = int(np.sum(mask & (scores >= threshold)))
            rate = (fp_sub / n_sub) if n_sub > 0 else 0.0
            per_attack[str(atype)] = {
                'n': n_sub,
                'fp': fp_sub,
                'apcer': float(rate),
                'apcer_pct': round(rate * 100.0, 2),
                'cp95': clopper_pearson_ci(fp_sub, n_sub)
            }
        results['per_attack'] = per_attack
        # Worst-case APCER across all PAI species
        if len(per_attack) > 0:
            results['worst_apcer'] = max(v['apcer'] for v in per_attack.values())
        else:
            results['worst_apcer'] = float(apcer)

    return results


def compute_operating_thresholds(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Find threshold at Equal Error Rate (EER) and threshold minimizing ACER.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    apcer = fpr
    bpcer = 1.0 - tpr
    acer = (apcer + bpcer) / 2.0

    eer_idx = int(np.argmin(np.abs(apcer - bpcer)))
    acer_idx = int(np.argmin(acer))

    return {
        'tau_eer': float(thresholds[eer_idx]),
        'tau_min_acer': float(thresholds[acer_idx]),
        'eer_val': float((apcer[eer_idx] + bpcer[eer_idx]) / 2.0),
        'min_acer_val': float(acer[acer_idx])
    }


def compute_apcer_at_target_bpcer(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_bpcer: float = 0.01
) -> Tuple[float, float]:
    """
    Compute APCER at an operational operating threshold where BPCER <= target_bpcer.
    """
    live_scores = scores[y_true == 1]
    attack_scores = scores[y_true == 0]

    if len(live_scores) == 0 or len(attack_scores) == 0:
        return np.nan, np.nan

    tau = float(np.percentile(live_scores, target_bpcer * 100.0))
    apcer = float(np.mean(attack_scores >= tau))
    return apcer, tau


def compute_iso_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    categories: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Full ISO/IEC 30107-3 benchmark evaluation suite with ROC, EER, and bootstrap intervals.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    auc_val = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan
    th_info = compute_operating_thresholds(y_true, scores)
    tau_eer = th_info['tau_eer']
    tau_min_acer = th_info['tau_min_acer']

    eval_eer = evaluate_threshold(y_true, scores, tau_eer, categories)
    eval_min_acer = evaluate_threshold(y_true, scores, tau_min_acer, categories)

    apcer_at_1pct_bpcer, tau_1pct = compute_apcer_at_target_bpcer(y_true, scores, 0.01)
    apcer_at_5pct_bpcer, tau_5pct = compute_apcer_at_target_bpcer(y_true, scores, 0.05)

    # Bootstrap 95% CI for AUC
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], scores[idx]))

    auc_ci = (
        float(np.percentile(boot_aucs, 2.5)),
        float(np.percentile(boot_aucs, 97.5))
    ) if len(boot_aucs) > 0 else (np.nan, np.nan)

    return {
        'auc': auc_val,
        'auc_ci95': auc_ci,
        'eer': th_info['eer_val'],
        'tau_eer': tau_eer,
        'tau_min_acer': tau_min_acer,
        'at_eer': eval_eer,
        'at_min_acer': eval_min_acer,
        'operational': {
            'apcer_at_1pct_bpcer': apcer_at_1pct_bpcer,
            'tau_1pct_bpcer': tau_1pct,
            'apcer_at_5pct_bpcer': apcer_at_5pct_bpcer,
            'tau_5pct_bpcer': tau_5pct
        }
    }
