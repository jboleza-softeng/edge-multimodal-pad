"""
Publication-Quality Figure Generator (IEEE / Nature Standards)
==============================================================
Generates publication figures matching the manuscript:
  - Figure 2: ROC and DET Curves (2-Panel)
  - Figure 3: Score Distribution Histograms (Bona Fide, Print PAI-1, Screen PAI-2)
  - Figure 5: Feature Stream Distributions Across Modalities
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import Optional


# Publication typography & styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'mathtext.fontset': 'cm',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#D9D9D9',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

PALETTE = {
    'live': '#1F77B4',       # Muted Blue (Bona Fide)
    'print': '#D62728',      # Brick Red (Print Attack)
    'screen': '#FF7F0E',     # Safety Orange (Screen Replay)
    'pooled_attack': '#8C1515',
    'threshold': '#2CA02C',  # Forest Green
    'midpoint': '#9467BD'    # Purple
}


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if c_low in ['subject_id', 'subject', 'subj_id']:
            col_map[col] = 'subject_id'
        elif c_low in ['label', 'ground_truth_is_live', 'is_live', 'target']:
            col_map[col] = 'label'
        elif c_low in ['pai_type', 'sample_category', 'attack_type', 'pai']:
            col_map[col] = 'pai_type'
        elif c_low in ['camera', 'device_label', 'device']:
            col_map[col] = 'camera'
        elif c_low in ['score', 'fused_score', 'prob_live', 'liveness_score']:
            col_map[col] = 'score'
    df = df.rename(columns=col_map)
    if 'label' in df.columns and df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.lower().map(
            lambda x: 1 if x in ['1', 'true', 'live', 'bonafide', 'bona_fide'] else 0
        )
    return df


def load_and_standardize_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return standardize_dataframe(df)


def generate_figure_2_roc_det(df: pd.DataFrame, tau_val: float = 0.4894, out_dir: str = "figures") -> str:
    """Figure 2: Two-panel ROC and DET curves."""
    df = standardize_dataframe(df)
    os.makedirs(out_dir, exist_ok=True)
    y_true = df['label'].values.astype(int)
    scores = df['score'].values.astype(float)
    is_live = (y_true == 1)
    
    fpr_pooled, tpr_pooled, _ = roc_curve(y_true, scores, pos_label=1)
    auc_pooled = auc(fpr_pooled, tpr_pooled)
    
    # Subspecies curves
    sub_data = {}
    for p_name, p_label in [('print_photo', 'Print Attack (PAI-1)'), ('screen_replay', 'Screen Replay (PAI-2)')]:
        p_mask = (df['pai_type'] == p_name) | is_live
        if p_mask.sum() > 0 and len(np.unique(y_true[p_mask])) > 1:
            fpr_p, tpr_p, _ = roc_curve(y_true[p_mask], scores[p_mask], pos_label=1)
            auc_p = auc(fpr_p, tpr_p)
            sub_data[p_name] = {'fpr': fpr_p, 'tpr': tpr_p, 'auc': auc_p, 'label': p_label}
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    
    # (a) ROC
    ax1.plot(fpr_pooled, tpr_pooled, color=PALETTE['pooled_attack'], lw=2.2,
             label=f'Pooled Attacks (AUC = {auc_pooled:.4f})')
    if 'print_photo' in sub_data:
        ax1.plot(sub_data['print_photo']['fpr'], sub_data['print_photo']['tpr'],
                 color=PALETTE['print'], lw=1.6, ls='--',
                 label=f"Print Attacks (AUC = {sub_data['print_photo']['auc']:.4f})")
    if 'screen_replay' in sub_data:
        ax1.plot(sub_data['screen_replay']['fpr'], sub_data['screen_replay']['tpr'],
                 color=PALETTE['screen'], lw=1.6, ls='-.',
                 label=f"Screen Replays (AUC = {sub_data['screen_replay']['auc']:.4f})")
    ax1.plot([0, 1], [0, 1], color='#888888', lw=1.0, ls=':', label='Random Guess (AUC = 0.5000)')
    ax1.set_xlabel('False Acceptance Rate (APCER)')
    ax1.set_ylabel('True Acceptance Rate (1 - BPCER)')
    ax1.set_title('(a) Receiver Operating Characteristic (ROC)')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.92)
    ax1.grid(True)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # (b) DET Curve
    eps = 1e-4
    fpr_c = np.clip(fpr_pooled, eps, 1.0 - eps)
    fnr_c = np.clip(1.0 - tpr_pooled, eps, 1.0 - eps)
    norm_ppf = stats.norm.ppf
    dev_x = norm_ppf(fpr_c)
    dev_y = norm_ppf(fnr_c)
    ax2.plot(dev_x, dev_y, color=PALETTE['pooled_attack'], lw=2.2, label='Pooled DET Trajectory')
    
    tick_locs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]
    tick_labels = ['0.1%', '1%', '5%', '10%', '20%', '40%', '60%']
    norm_ticks = [norm_ppf(t) for t in tick_locs]
    ax2.set_xticks(norm_ticks)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(norm_ticks)
    ax2.set_yticklabels(tick_labels)
    ax2.set_xlabel('False Acceptance Rate (APCER)')
    ax2.set_ylabel('False Rejection Rate (BPCER)')
    ax2.set_title('(b) Detection Error Tradeoff (DET)')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.92)
    ax2.grid(True)
    ax2.set_xlim([norm_ppf(0.001), norm_ppf(0.7)])
    ax2.set_ylim([norm_ppf(0.001), norm_ppf(0.7)])
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "figure2_roc_det.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def generate_figure_3_score_histograms(df: pd.DataFrame, tau_val: float = 0.4894, out_dir: str = "figures") -> str:
    """Figure 3: Density histograms by class and PAI species."""
    df = standardize_dataframe(df)
    os.makedirs(out_dir, exist_ok=True)
    live_scores = df[df['label'] == 1]['score'].values
    print_scores = df[(df['label'] == 0) & (df['pai_type'] == 'print_photo')]['score'].values
    screen_scores = df[(df['label'] == 0) & (df['pai_type'] == 'screen_replay')]['score'].values
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0.0, 1.0, 35)
    
    ax.hist(live_scores, bins=bins, color=PALETTE['live'], alpha=0.60, edgecolor=PALETTE['live'],
            density=True, label=f'Bona Fide Live (n = {len(live_scores)})', lw=1.2)
    if len(print_scores) > 0:
        ax.hist(print_scores, bins=bins, color=PALETTE['print'], alpha=0.55, edgecolor=PALETTE['print'],
                density=True, label=f'Print Attacks (n = {len(print_scores)})', lw=1.2)
    if len(screen_scores) > 0:
        ax.hist(screen_scores, bins=bins, color=PALETTE['screen'], alpha=0.55, edgecolor=PALETTE['screen'],
                density=True, label=f'Screen Replays (n = {len(screen_scores)})', lw=1.2)
                
    ax.axvline(tau_val, color=PALETTE['threshold'], lw=2.2, ls='-',
               label=f'Calibrated Operating Threshold (τ = {tau_val:.4f})')
    
    ax.set_xlabel('Multimodal Liveness Probability Score P(Bona Fide | z)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Figure 3: Score Distribution Density Across Bona Fide and PAI Modalities')
    ax.legend(loc='upper center', frameon=True, framealpha=0.92, bbox_to_anchor=(0.5, 0.98))
    ax.grid(True)
    ax.set_xlim([-0.02, 1.02])
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "figure3_score_histograms.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def generate_figure_5_per_feature_distributions(df: pd.DataFrame, out_dir: str = "figures") -> str:
    """Figure 5: Box plots across 6 feature streams."""
    df = standardize_dataframe(df)
    os.makedirs(out_dir, exist_ok=True)
    features = ['band_ratio', 'sharp_rel', 'spec_frac', 'depth_rel', 'motion_rel', 'blue_red']
    feature_labels = {
        'band_ratio': 'Spatial Frequency (band_ratio)',
        'sharp_rel': 'Laplacian Gradient (sharp_rel)',
        'spec_frac': 'Specular Highlights (spec_frac)',
        'depth_rel': 'Depth Disparity (depth_rel)',
        'motion_rel': 'Temporal Micro-Motion (motion_rel)',
        'blue_red': 'Chromatic Ratio (blue_red) [Weight=0]'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    axes = axes.flatten()
    cats = ['bona_fide', 'print_photo', 'screen_replay']
    cat_names = ['Bona Fide', 'Print PAI', 'Screen PAI']
    cat_colors = [PALETTE['live'], PALETTE['print'], PALETTE['screen']]
    
    for f_idx, f_col in enumerate(features):
        ax = axes[f_idx]
        if f_col not in df.columns:
            ax.set_visible(False)
            continue
            
        data_to_plot = []
        valid_indices = []
        for c_idx, c_val in enumerate(cats):
            subset = df[df['pai_type'] == c_val][f_col].dropna().values
            if len(subset) > 0:
                data_to_plot.append(subset)
                valid_indices.append(c_idx)
                
        if len(data_to_plot) == 0:
            ax.set_visible(False)
            continue
            
        box = ax.boxplot(data_to_plot, patch_artist=True, widths=0.55,
                         medianprops=dict(color='black', lw=1.2),
                         whiskerprops=dict(color='#555555', lw=0.9),
                         capprops=dict(color='#555555', lw=0.9),
                         flierprops=dict(marker='o', markersize=3, alpha=0.4))
                         
        for p_idx, patch in enumerate(box['boxes']):
            orig_cat_idx = valid_indices[p_idx]
            patch.set_facecolor(cat_colors[orig_cat_idx])
            patch.set_alpha(0.65)
            
        ax.set_xticks(np.arange(len(valid_indices)) + 1)
        ax.set_xticklabels([cat_names[idx] for idx in valid_indices], fontsize=8.5)
        ax.set_title(feature_labels.get(f_col, f_col), fontsize=9.5, pad=6)
        ax.grid(True, axis='y')
        
    plt.suptitle('Figure 5: Feature Stream Distributions Across Bona Fide and PAI Modalities', fontsize=12, y=0.99)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "figure5_feature_distributions.png")
    plt.savefig(out_path)
    plt.close()
    return out_path
