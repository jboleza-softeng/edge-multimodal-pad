#!/usr/bin/env python3
"""
Feature Separability and Modality Ablation Runner
=================================================
Reproduces Table IV from the manuscript:
  - Single-feature univariate AUC
  - Calibrated logistic fusion weights (z)
  - Leave-one-feature-out ablation impact (ΔAUC)
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

from src.ablation import evaluate_feature_ablation


def main():
    # Use calibration split or full dataset
    csv_path = os.path.join(BASE_DIR, 'data', 'calib_final.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'data', 'benchmark_trials_225n.csv')

    df = pd.read_csv(csv_path)
    if 'corpus_split' in df.columns:
        calib_df = df[df['corpus_split'] == 'calibration'].copy()
        if len(calib_df) > 0:
            df = calib_df

    print("=" * 85)
    print(" FEATURE SEPARABILITY AND ABLATION MATRIX (TABLE IV)")
    print(f" Corpus: {csv_path} (N = {len(df)} trials)")
    print("=" * 85)

    df_abl = evaluate_feature_ablation(df)

    display_cols = ['Feature Stream', 'Modality Focus', 'Univariate AUC', 'Fusion Weight (z)', 'Ablation Impact (ΔAUC)']
    print(df_abl[[c for c in display_cols if c in df_abl.columns]].to_string(index=False))
    print("=" * 85)

    out_csv = os.path.join(BASE_DIR, 'results', 'table4_ablation_matrix.csv')
    df_abl.to_csv(out_csv, index=False)
    print(f"\nSaved Table IV CSV at: {out_csv}")


if __name__ == '__main__':
    main()
