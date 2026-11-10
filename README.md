# Edge Multimodal Presentation Attack Detection (PAD) Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Standard: ISO/IEC 30107-3](https://img.shields.io/badge/Standard-ISO%2FIEC%2030107--3-orange.svg)](https://www.iso.org/standard/67381.html)
[![Reproducibility: Verified](https://img.shields.io/badge/Reproducibility-Verified%20100%25-brightgreen.svg)](reproduce_all.py)

Official open-source benchmark repository and reproducibility suite for the research manuscript:

> **"Edge-Based Multimodal Presentation Attack Detection: Feature Fusion, Calibration Protocol, and Hardware-Conditioned Evaluation Under ISO/IEC 30107-3"**  
> **Author:** Jeric G. Boleza¹,*  
> ¹ *College of Mathematics and Computing Science, University of Abra, Main Campus, Lagangilang, Abra, Philippines*  
> *ORCID:* [0009-0001-6308-6484](https://orcid.org/0009-0001-6308-6484) | **Corresponding Author:* `jboleza@asist.edu.ph`

---

## 📌 Abstract Summary

Facial recognition systems deployed on unattended edge computing kiosks face operational security vulnerabilities from Presentation Attack Instruments (PAIs), primarily printed photographs and dynamic video replay screens. While deep neural networks (CNNs/ViTs) achieve high accuracy on standardized benchmarks, their computational footprint (>200 ms per frame), domain shift susceptibility, and demographic skin-tone confounding hinder deployment on constrained edge terminals.

This repository provides the reference implementation, empirical dataset ($N = 225$ trials across 10 human subjects), calibration protocol, and automated replication suite for our lightweight, deterministic edge PAD architecture. The framework concurrently extracts five low-complexity spatial, spectral, and kinematic feature descriptors:
1. **Spatial Frequency Energy Persistence** (`band_ratio`): Dual-scale Laplacian variance ratio.
2. **Laplacian Gradient Sharpness** (`sharp_rel`): Gain-invariant relative edge gradient steepness.
3. **Specular Highlight Dynamics** (`spec_frac`): Multi-channel indicator clipping saturation fraction.
4. **Perspective Depth Disparity** (`depth_rel`): Anthropometric landmark planarity constraint.
5. **Optical Micro-Motion** (`motion_rel`): Inter-frame temporal biological kinematic gradient.

To eliminate demographic and ambient illuminant bias, chromatic ratio features (`blue_red`) are regularized to zero weight ($w = 0.000$). Decision boundaries are governed by device-conditioned statistical standardization $(\mu^{(d)}, \sigma^{(d)})$ and logistic regression fusion evaluated under **ISO/IEC 30107-3** standards.

---

## 🚀 Quickstart: 1-Click Verification (Under 15 Seconds)

Reviewers can clone and verify the entire experimental suite with a single command:

```bash
# 1. Clone repository
git clone https://github.com/your-username/edge-pad-benchmark.git
cd edge-pad-benchmark

# 2. Install minimal dependencies
pip install -r requirements.txt

# 3. Replicate all tables, figures, and unit tests
python reproduce_all.py
```

### Expected Output Summary
```text
==========================================================================================
 REPRODUCIBILITY SUMMARY
==========================================================================================
Total Execution Time: ~15 seconds
Test Suite Status:    ALL TESTS PASSED (10/10)
Held-Out ACER:        0.00% (0 misclassifications on unseen S11-S15)
Separation Margin:    Delta_gap = +0.4893
Generated Figures:    figures/ (Figure 2, Figure 3, Figure 5 at 300 DPI)
Exported Results:     results/ (Table III, Table IV, LOSO-CV Summary)
==========================================================================================
```

---

## 📊 Empirical Findings & Manuscript Replication

### Table III: Held-Out Test Evaluation Under ISO/IEC 30107-3 ($N = 75$ Trials)
Evaluated on unseen subjects **S11–S15** under the frozen calibrated operating threshold ($\tau = 0.4894$):

| Presentation Category | Evaluated Cohort | Trials ($n$) | Errors ($k$) | Observed Error Rate | Exact Clopper-Pearson 95% CI |
|---|---|:---:|:---:|:---:|:---:|
| **Bona Fide (Live Human)** | Unseen S11–S15 | 25 | 0 | **BPCER = 0.00%** | $[0.00\%,\, 13.72\%]$ |
| **PAI-1 (Print Attack)** | Unseen S11–S15 | 25 | 0 | **APCER = 0.00%** | $[0.00\%,\, 13.72\%]$ |
| **PAI-2 (Screen Replay)** | Unseen S11–S15 | 25 | 0 | **APCER = 0.00%** | $[0.00\%,\, 13.72\%]$ |
| **Worst-Case APCER** | Unseen S11–S15 | 50 | 0 | **APCER = 0.00%** | $[0.00\%,\, 7.11\%]$ |
| **Overall System ACER** | Unseen S11–S15 | 75 | 0 | **ACER = 0.00%** | $[0.00\%,\, 4.81\%]$ |

> **Methodological Note on 95% CIs:** Under ISO/IEC 30107-3 methodology, an empirical error rate of $0.00\%$ on small cohorts ($n = 25$) must not be interpreted as absolute operational invulnerability. The exact Clopper-Pearson 95% confidence interval upper bound of $13.72\%$ reflects statistical power boundaries and highlights the necessity of multi-site evaluation cohorts.

### Score Separation Margin Dynamics
The calibrated decision threshold ($\tau = 0.4894$) bisects the empirical score distributions with complete separation:
* **Minimum Bona Fide Score:** $\min_{i \in \text{Live}} P(\text{Live} \mid z_i) = 0.8836$
* **Maximum Attack Score:** $\max_{j \in \text{Attack}} P(\text{Live} \mid z_j) = 0.3943$
* **Absolute Separation Gap:** $\Delta_{\text{gap}} = 0.8836 - 0.3943 = \mathbf{+0.4893}$
* **Bona Fide Cushion:** $\min(\text{Live}) - \tau = \mathbf{+0.3942}$
* **Attack Safety Buffer:** $\tau - \max(\text{Attack}) = \mathbf{+0.0951}$

---

### Table IV: Feature Separability and Modality Ablation Matrix

| Feature Stream | Modality Focus | Univariate AUC | Fusion Weight ($z$) | Ablation Impact ($\Delta\text{AUC}$) | Primary Role |
|---|---|:---:|:---:|:---:|---|
| `band_ratio` | Frequency Energy | **0.957** | $+1.728$ | $-0.0104$ | Screen grid / Moiré pattern detection |
| `sharp_rel` | Laplacian Texture | **0.867** | $+0.752$ | $-0.0010$ | Print ink dispersion & gradient blur |
| `spec_frac` | Surface Glare | **0.814** | $+1.637$ | **$-0.0128$** | Non-Lambertian screen glass / gloss reflections |
| `depth_rel` | 3D Perspective | **0.669** | $+0.808$ | $-0.0044$ | Planar rigidity / anthropometric disparity |
| `motion_rel` | Micro-Kinematics | **0.659** | $-0.470$ | $-0.0002$ | Involuntary biological micro-saccades |
| `blue_red` | Chromatic Balance | **0.532** | $0.000$ (Locked) | **N/A** | **Ethical anti-bias regularization** |

---

## 🛠️ Individual Step-by-Step Execution Guides

### 1. Evaluate Held-Out Test Split (Table III)
```bash
python run_evaluation.py
```
*Evaluates the held-out test cohort (S11–S15, $N=75$) and exports results to `results/table3_held_out_performance.csv`.*

### 2. Feature Separability & Ablation Analysis (Table IV)
```bash
python run_ablation.py
```
*Computes single-feature AUCs, full logistic fusion weights, and leave-one-feature-out degradation ($\Delta\text{AUC}$).*

### 3. Leave-Subject-Out Cross-Validation (LOSO-CV)
```bash
python run_loso_cv.py
```
*Performs exhaustive cross-validation across all 10 subjects ($N=225$) with fold-level metrics and 1,000 bootstrap iterations.*

### 4. Device Calibration & Confound Check
```bash
python run_calibration.py --csv data/calib_final.csv --out results/calibration_report.md
```
*Audits calibration criteria, device threshold stability, and tests against potential camera shortcuts (e.g. `crop_min_luma`).*

### 5. Run Verification & Invariant Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
*Executes 10 automated unit and regression tests verifying ISO mathematical formulas, gain invariance, and reproducibility.*

---

## 📁 Repository Directory Structure

```text
edge-pad-benchmark/
├── README.md                      # Reviewer documentation & study reproduction guide
├── LICENSE                        # MIT Open Source License
├── requirements.txt               # Pinned Python package dependencies
├── reproduce_all.py               # Master 1-click execution script reproducing all paper results
├── run_evaluation.py              # Standalone ISO/IEC 30107-3 test evaluation (Table III)
├── run_ablation.py                # Standalone feature separability and ablation runner (Table IV)
├── run_loso_cv.py                 # Standalone Leave-Subject-Out Cross-Validation runner
├── run_calibration.py             # Standalone calibration protocol & confound audit runner
├── data/
│   ├── README.md                  # Comprehensive dataset documentation, schema & ethics card
│   ├── benchmark_trials_225n.csv  # Full empirical benchmark dataset (10 subjects, N = 225 trials)
│   ├── benchmark_trials_225n.json # Full benchmark in structured JSON format
│   ├── calib_final.csv            # Dedicated 150-trial calibration partition (S01, S03, S07-S09)
│   ├── biometric_benchmark_trials.csv # Historical pilot baseline dataset (175 trials)
│   └── splits.json                # Seeded burst/subject-disjoint split mappings
├── src/                           # Modular core package
│   ├── __init__.py                # Package metadata
│   ├── metrics.py                 # ISO/IEC 30107-3 metric calculations & exact Clopper-Pearson CIs
│   ├── fusion_model.py            # Multimodal feature fusion & decision threshold engine
│   ├── ablation.py                # Feature ablation matrix & univariate AUC calculation
│   ├── loso.py                    # Leave-Subject-Out cross-validation engine
│   └── figures.py                 # IEEE Transactions publication figure rendering
├── figures/                       # Publication-quality figures (300 DPI)
│   ├── figure2_roc_det.png        # Figure 2: ROC and DET Curves (2-Panel)
│   ├── figure3_score_histograms.png # Figure 3: Score Distribution Histograms
│   └── figure5_feature_distributions.png # Figure 5: Feature Stream Box Plots
├── results/                       # Exported benchmark evaluations & logs
│   ├── table3_held_out_performance.csv # Table III numerical CSV
│   ├── table4_ablation_matrix.csv      # Table IV numerical CSV
│   ├── loso_cv_summary.csv             # LOSO-CV per-fold summary
│   └── calibration_report.md           # Sensor calibration diagnostic audit
└── tests/                         # Automated test suite
    ├── __init__.py
    ├── test_iso_metrics.py        # Validates Clopper-Pearson CI and ISO error rate math
    ├── test_fusion_invariants.py  # Validates gain invariance and ethical zero-weight clamp
    └── test_reproducibility.py    # Regression test asserting 0 errors on held-out test split
```

---

## ⚡ Edge Hardware Computational Profiling

The feature extraction algorithms and logistic fusion engine were profiled across two edge computing architectures without discrete GPU or NPU acceleration:

| Platform Architecture | Microprocessor | Architecture | Mean Per-Frame Latency | Throughput (FPS) |
|---|---|---|:---:|:---:|
| **Platform A (Edge Kiosk)** | Intel Core i3-1115G4 @ 3.00 GHz | x86-64 | **8.8 ms** | **113.6 fps** |
| **Platform B (Embedded IoT)** | Broadcom BCM2711 Cortex-A72 @ 1.50 GHz | ARMv8 (64-bit) | **21.8 ms** | **45.8 fps** |

Both platforms operate well within real-time video streaming constraints (30 fps requires $\le 33.3\text{ ms}$ per frame).

---

## 📄 Citation

If you find this benchmark, protocol, or reference implementation useful in your research, please cite:

```bibtex
@article{boleza2026edgepad,
  title   = {Edge-Based Multimodal Presentation Attack Detection: Feature Fusion, Calibration Protocol, and Hardware-Conditioned Evaluation Under {ISO/IEC 30107-3}},
  author  = {Boleza, Jeric G.},
  journal = {Preprint submitted for peer review},
  year    = {2026},
  url     = {https://github.com/your-username/edge-pad-benchmark}
}
```

---

## ⚖️ License & Ethical Data Governance

* **Code License:** Released under the [MIT License](LICENSE).
* **Biometric Privacy Notice:** In compliance with ethical research standards and personal data protection regulations, raw high-resolution facial videos are preserved in secure institutional storage. Extracted, de-identified mathematical feature matrices and trial scores are distributed in `data/`, enabling 100% computational replication while protecting human subject privacy.
