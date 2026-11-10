#!/usr/bin/env python3
"""
Leave-Subject-Out Cross-Validation (LOSO-CV) Runner
===================================================
Runs cross-validation across all 10 subjects in the benchmark corpus (N = 225 trials)
and computes fold-level and pooled error rates with exact Clopper-Pearson CIs.
"""

import os
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.loso import run_loso_cv


def main():
    csv_path = os.path.join(BASE_DIR, 'data', 'benchmark_trials_225n.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print("=" * 85)
    print(" LEAVE-SUBJECT-OUT CROSS-VALIDATION (ISO/IEC 30107-3)")
    print(f" Dataset: {csv_path} (N = {len(df)} trials across {df['subject_id'].nunique()} subjects)")
    print("=" * 85)

    res = run_loso_cv(df, n_bootstrap=1000, seed=42)
    df_folds = res['df_folds']

    print("\n--- PER-SUBJECT TEST FOLD SUMMARY ---")
    cols_to_print = ['subject_id', 'n_live', 'n_attack', 'tau_op', 'bpcer', 'worst_apcer', 'acer', 'eer']
    print(df_folds[[c for c in cols_to_print if c in df_folds.columns]].to_string(index=False))

    pooled = res['pooled']
    means = res['fold_means']
    ci_boot = res['ci_boot']

    print("\n" + "=" * 85)
    print(" POOLED CROSS-VALIDATION PERFORMANCE SUMMARY")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Mean +/- SD':<18} | {'Pooled (k/n)':<16} | {'Exact Clopper-Pearson 95% CI':<28} | {'Bootstrap 95% CI'}")
    print("-" * 115)
    print(f"{'BPCER (False Rejection)':<25} | {means['bpcer_mean']*100:5.2f}% +/- {means['bpcer_sd']*100:4.2f}% | {pooled['bpcer']*100:5.2f}% ({pooled['k_bpcer']}/{pooled['n_live']}) | [{pooled['ci_bpcer_cp'][0]*100:5.2f}%, {pooled['ci_bpcer_cp'][1]*100:5.2f}%]{'':<14} | [{ci_boot['bpcer'][0]*100:5.2f}%, {ci_boot['bpcer'][1]*100:5.2f}%]")
    print(f"{'APCER (Pooled Attacks)':<25} | {means['apcer_mean']*100:5.2f}% +/- {means['apcer_sd']*100:4.2f}% | {pooled['apcer']*100:5.2f}% ({pooled['k_apcer']}/{pooled['n_attack']})| [{pooled['ci_apcer_cp'][0]*100:5.2f}%, {pooled['ci_apcer_cp'][1]*100:5.2f}%]{'':<14} | [{ci_boot['apcer'][0]*100:5.2f}%, {ci_boot['apcer'][1]*100:5.2f}%]")
    print(f"{'ACER (Average Error)':<25} | {means['acer_mean']*100:5.2f}% +/- {means['acer_sd']*100:4.2f}% | {pooled['acer']*100:5.2f}%{'':<10} | [{pooled['ci_acer_cp'][0]*100:5.2f}%, {pooled['ci_acer_cp'][1]*100:5.2f}%]{'':<14} | [{ci_boot['acer'][0]*100:5.2f}%, {ci_boot['acer'][1]*100:5.2f}%]")
    print(f"Global Out-Of-Fold AUC: {res['oof_auc']:.4f}")
    print("=" * 85)

    out_csv = os.path.join(BASE_DIR, 'results', 'loso_cv_summary.csv')
    df_folds.to_csv(out_csv, index=False)
    print(f"\nSaved LOSO fold summary to: {out_csv}")


if __name__ == '__main__':
    main()
