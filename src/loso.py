"""
Leave-Subject-Out Cross-Validation (LOSO-CV) Engine
====================================================
Evaluates generalization across unseen subjects in accordance with
ISO/IEC 30107-3 reporting standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_curve, auc
from .metrics import clopper_pearson_ci, compute_operating_thresholds, compute_apcer_at_target_bpcer


def run_loso_cv(
    df: pd.DataFrame,
    target_bpcers: Tuple[float, ...] = (0.01, 0.05),
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute Leave-Subject-Out Cross-Validation across all distinct subjects.
    """
    np.random.seed(seed)
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ['subject_id', 'subject', 'subj_id']:
            col_map[col] = 'subject_id'
        elif c in ['label', 'ground_truth_is_live', 'is_live', 'target']:
            col_map[col] = 'label'
        elif c in ['pai_type', 'sample_category', 'attack_type', 'pai']:
            col_map[col] = 'pai_type'
        elif c in ['camera', 'device_label', 'device']:
            col_map[col] = 'camera'
        elif c in ['score', 'fused_score', 'prob_live', 'liveness_score']:
            col_map[col] = 'score'
            
    df = df.rename(columns=col_map).copy()
    if df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.lower().map(
            lambda x: 1 if x in ['1', 'true', 'live', 'bonafide', 'bona_fide'] else 0
        )
    df['score'] = df['score'].astype(float)
    df['subject_id'] = df['subject_id'].astype(str)
    
    subjects = sorted(df['subject_id'].unique())
    n_subjects = len(subjects)
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects for LOSO-CV. Found {n_subjects}.")
        
    fold_results: List[Dict[str, Any]] = []
    oof_predictions: List[pd.DataFrame] = []
    
    for test_subj in subjects:
        train_df = df[df['subject_id'] != test_subj]
        test_df = df[df['subject_id'] == test_subj].copy()
        
        train_y = train_df['label'].values
        train_s = train_df['score'].values
        th_info = compute_operating_thresholds(train_y, train_s)
        tau_op = th_info['tau_min_acer']
        
        test_live = test_df[test_df['label'] == 1]
        test_attack = test_df[test_df['label'] == 0]
        
        n_live = len(test_live)
        n_attack = len(test_attack)
        
        bpcer = float(np.mean(test_live['score'].values < tau_op)) if n_live > 0 else 0.0
        
        pai_apcers = {}
        for p in test_attack['pai_type'].unique():
            p_scores = test_attack[test_attack['pai_type'] == p]['score'].values
            pai_apcers[str(p)] = float(np.mean(p_scores >= tau_op))
            
        worst_apcer = max(pai_apcers.values()) if len(pai_apcers) > 0 else (
            float(np.mean(test_attack['score'].values >= tau_op)) if n_attack > 0 else 0.0
        )
        acer = (worst_apcer + bpcer) / 2.0
        
        if n_live > 0 and n_attack > 0:
            test_th = compute_operating_thresholds(test_df['label'].values, test_df['score'].values)
            test_eer = test_th['eer_val']
        else:
            test_eer = np.nan
            
        test_df['pred_label'] = (test_df['score'] >= tau_op).astype(int)
        test_df['operating_tau'] = tau_op
        oof_predictions.append(test_df)
        
        fold_res = {
            'subject_id': test_subj,
            'n_live': n_live,
            'n_attack': n_attack,
            'tau_op': tau_op,
            'bpcer': bpcer,
            'worst_apcer': worst_apcer,
            'acer': acer,
            'eer': test_eer
        }
        for p, a_val in pai_apcers.items():
            fold_res[f'apcer_{p}'] = a_val
        fold_results.append(fold_res)
        
    df_folds = pd.DataFrame(fold_results)
    df_oof = pd.concat(oof_predictions, ignore_index=True)
    
    # Out of fold pooled statistics
    oof_live = df_oof[df_oof['label'] == 1]
    oof_attack = df_oof[df_oof['label'] == 0]
    n_total_live = len(oof_live)
    n_total_attack = len(oof_attack)
    k_bpcer_errors = int(np.sum(oof_live['score'] < oof_live['operating_tau']))
    k_apcer_errors = int(np.sum(oof_attack['score'] >= oof_attack['operating_tau']))
    
    pooled_bpcer = (k_bpcer_errors / n_total_live) if n_total_live > 0 else 0.0
    pooled_apcer = (k_apcer_errors / n_total_attack) if n_total_attack > 0 else 0.0
    pooled_acer = (pooled_bpcer + pooled_apcer) / 2.0
    
    ci_bpcer_cp = clopper_pearson_ci(k_bpcer_errors, n_total_live)
    ci_apcer_cp = clopper_pearson_ci(k_apcer_errors, n_total_attack)
    ci_acer_cp = ((ci_bpcer_cp[0] + ci_apcer_cp[0]) / 2.0, (ci_bpcer_cp[1] + ci_apcer_cp[1]) / 2.0)
    
    fpr_oof, tpr_oof, _ = roc_curve(df_oof['label'].values, df_oof['score'].values, pos_label=1)
    oof_auc = float(auc(fpr_oof, tpr_oof))
    
    # Subject-level bootstrap intervals
    boot_bpcer, boot_apcer, boot_acer = [], [], []
    for _ in range(n_bootstrap):
        sampled_subjs = np.random.choice(subjects, size=len(subjects), replace=True)
        boot_df = pd.concat([df_oof[df_oof['subject_id'] == s] for s in sampled_subjs], ignore_index=True)
        b_live = boot_df[boot_df['label'] == 1]
        b_attack = boot_df[boot_df['label'] == 0]
        if len(b_live) > 0 and len(b_attack) > 0:
            boot_bpcer.append(np.mean(b_live['score'] < b_live['operating_tau']))
            boot_apcer.append(np.mean(b_attack['score'] >= b_attack['operating_tau']))
            boot_acer.append((boot_bpcer[-1] + boot_apcer[-1]) / 2.0)
            
    ci_boot = {
        'bpcer': (float(np.percentile(boot_bpcer, 2.5)), float(np.percentile(boot_bpcer, 97.5))) if boot_bpcer else (np.nan, np.nan),
        'apcer': (float(np.percentile(boot_apcer, 2.5)), float(np.percentile(boot_apcer, 97.5))) if boot_apcer else (np.nan, np.nan),
        'acer': (float(np.percentile(boot_acer, 2.5)), float(np.percentile(boot_acer, 97.5))) if boot_acer else (np.nan, np.nan),
    }

    return {
        'df_folds': df_folds,
        'df_oof': df_oof,
        'oof_auc': oof_auc,
        'pooled': {
            'bpcer': pooled_bpcer,
            'k_bpcer': k_bpcer_errors,
            'n_live': n_total_live,
            'ci_bpcer_cp': ci_bpcer_cp,
            'apcer': pooled_apcer,
            'k_apcer': k_apcer_errors,
            'n_attack': n_total_attack,
            'ci_apcer_cp': ci_apcer_cp,
            'acer': pooled_acer,
            'ci_acer_cp': ci_acer_cp,
        },
        'fold_means': {
            'bpcer_mean': float(df_folds['bpcer'].mean()),
            'bpcer_sd': float(df_folds['bpcer'].std()),
            'apcer_mean': float(df_folds['worst_apcer'].mean()),
            'apcer_sd': float(df_folds['worst_apcer'].std()),
            'acer_mean': float(df_folds['acer'].mean()),
            'acer_sd': float(df_folds['acer'].std()),
        },
        'ci_boot': ci_boot
    }
