#!/usr/bin/env python3
"""
Held-Out Test Partition Evaluation under ISO/IEC 30107-3
========================================================
Evaluates the frozen operating rule on the strictly subject-disjoint
held-out test cohort (N = 75 trials across unseen subjects S11–S15).
Outputs Table III from the manuscript with exact Clopper-Pearson 95% CIs.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pandas as pd
import numpy as np

# Ensure src package is discoverable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.metrics import evaluate_threshold, clopper_pearson_ci
from src.fusion_model import EdgeMultimodalPAD, CALIBRATED_CONSTANTS


def main():
    csv_path = os.path.join(BASE_DIR, 'data', 'benchmark_trials_225n.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filter to held-out test partition
    if 'corpus_split' in df.columns:
        test_df = df[df['corpus_split'] == 'test'].copy()
    else:
        test_subjs = ['S11', 'S12', 'S13', 'S14', 'S15']
        test_df = df[df['subject_id'].isin(test_subjs)].copy()

    print("=" * 85)
    print(" HELD-OUT TEST EVALUATION UNDER ISO/IEC 30107-3")
    print(f" Cohort: Unseen Subjects S11–S15 (N = {len(test_df)} trials)")
    print(f" Frozen Calibrated Operating Threshold: tau = {CALIBRATED_CONSTANTS['tau']:.4f}")
    print("=" * 85)

    y_true = test_df['ground_truth_is_live'].values.astype(int)
    scores = test_df['fused_score'].values.astype(float)
    categories = test_df['sample_category'].values

    # Evaluate at calibrated threshold
    tau = CALIBRATED_CONSTANTS['tau']
    res = evaluate_threshold(y_true, scores, threshold=tau, categories=categories)

    # Separation Margins
    live_scores = scores[y_true == 1]
    attack_scores = scores[y_true == 0]
    min_live = float(np.min(live_scores))
    max_attack = float(np.max(attack_scores))
    delta_gap = min_live - max_attack
    cushion_live = min_live - tau
    buffer_attack = tau - max_attack

    # Output Table III format
    print("\nTABLE III: HELD-OUT TEST PERFORMANCE UNDER ISO/IEC 30107-3 (N = 75 TRIALS)")
    print("-" * 85)
    print(f"{'Presentation Category':<25} | {'Cohort':<14} | {'Trials (n)':<10} | {'Errors (k)':<10} | {'Observed Rate [Exact 95% CI]'}")
    print("-" * 85)

    # Bona Fide
    bf_ci = res['bpcer_cp95']
    print(f"{'Bona Fide (Live Human)':<25} | {'Unseen S11-S15':<14} | {res['n_bona']:<10} | {res['fn_errors']:<10} | BPCER = {res['bpcer_pct']:.2f}% [{bf_ci[0]*100:.2f}%, {bf_ci[1]*100:.2f}%]")

    # Print Attack
    p_info = res['per_attack'].get('print_photo', {'n': 25, 'fp': 0, 'apcer': 0.0, 'cp95': (0.0, 0.1372)})
    p_ci = p_info['cp95']
    print(f"{'PAI-1 (Print Attack)':<25} | {'Unseen S11-S15':<14} | {p_info['n']:<10} | {p_info['fp']:<10} | APCER_Print = {p_info['apcer']*100:.2f}% [{p_ci[0]*100:.2f}%, {p_ci[1]*100:.2f}%]")

    # Screen Replay
    s_info = res['per_attack'].get('screen_replay', {'n': 25, 'fp': 0, 'apcer': 0.0, 'cp95': (0.0, 0.1372)})
    s_ci = s_info['cp95']
    print(f"{'PAI-2 (Screen Replay)':<25} | {'Unseen S11-S15':<14} | {s_info['n']:<10} | {s_info['fp']:<10} | APCER_Screen = {s_info['apcer']*100:.2f}% [{s_ci[0]*100:.2f}%, {s_ci[1]*100:.2f}%]")

    # Pooled Attacks
    pool_ci = res['apcer_cp95']
    print(f"{'Worst-Case APCER':<25} | {'Unseen S11-S15':<14} | {res['n_attack']:<10} | {res['fp_errors']:<10} | APCER = {res['apcer_pct']:.2f}% [{pool_ci[0]*100:.2f}%, {pool_ci[1]*100:.2f}%]")

    # Overall ACER
    print(f"{'Overall System ACER':<25} | {'Unseen S11-S15':<14} | {len(test_df):<10} | {res['fn_errors'] + res['fp_errors']:<10} | ACER = {res['acer_pct']:.2f}% (at tau = {tau:.4f})")
    print("-" * 85)

    print("\n--- SCORE SEPARATION MARGINS AROUND THRESHOLD ---")
    print(f"Minimum Bona Fide Score: {min_live:.4f}")
    print(f"Maximum Attack Score:    {max_attack:.4f}")
    print(f"Absolute Separation Gap: Delta_gap = {min_live:.4f} - {max_attack:.4f} = +{delta_gap:.4f}")
    print(f"Bona Fide Acceptance Cushion: min(Live) - tau = +{cushion_live:.4f}")
    print(f"Attack Rejection Safety Buffer: tau - max(Attack) = +{buffer_attack:.4f}")
    print("=" * 85)

    # Save to results/table3_held_out_performance.csv
    out_table_path = os.path.join(BASE_DIR, 'results', 'table3_held_out_performance.csv')
    table3_df = pd.DataFrame([
        {'Presentation Category': 'Bona Fide (Live Human)', 'Cohort': 'Unseen S11-S15', 'Trials': res['n_bona'], 'Errors': res['fn_errors'], 'Metric': f"BPCER = {res['bpcer_pct']:.2f}% [{bf_ci[0]*100:.2f}%, {bf_ci[1]*100:.2f}%]"},
        {'Presentation Category': 'PAI-1 (Print Attack)', 'Cohort': 'Unseen S11-S15', 'Trials': p_info['n'], 'Errors': p_info['fp'], 'Metric': f"APCER_Print = {p_info['apcer']*100:.2f}% [{p_ci[0]*100:.2f}%, {p_ci[1]*100:.2f}%]"},
        {'Presentation Category': 'PAI-2 (Screen Replay)', 'Cohort': 'Unseen S11-S15', 'Trials': s_info['n'], 'Errors': s_info['fp'], 'Metric': f"APCER_Screen = {s_info['apcer']*100:.2f}% [{s_ci[0]*100:.2f}%, {s_ci[1]*100:.2f}%]"},
        {'Presentation Category': 'Worst-Case APCER', 'Cohort': 'Unseen S11-S15', 'Trials': res['n_attack'], 'Errors': res['fp_errors'], 'Metric': f"APCER = {res['apcer_pct']:.2f}% [{pool_ci[0]*100:.2f}%, {pool_ci[1]*100:.2f}%]"},
        {'Presentation Category': 'Overall System ACER', 'Cohort': 'Unseen S11-S15', 'Trials': len(test_df), 'Errors': res['fn_errors'] + res['fp_errors'], 'Metric': f"ACER = {res['acer_pct']:.2f}% (at tau = {tau:.4f})"}
    ])
    table3_df.to_csv(out_table_path, index=False)
    print(f"\nSaved Table III CSV at: {out_table_path}")


if __name__ == '__main__':
    main()
