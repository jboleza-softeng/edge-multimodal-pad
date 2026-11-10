# Benchmark Dataset Documentation & Data Card

This directory contains the empirical presentation attack detection trial datasets supporting the manuscript:
> **"Edge-Based Multimodal Presentation Attack Detection: Feature Fusion, Calibration Protocol, and Hardware-Conditioned Evaluation Under ISO/IEC 30107-3"**  
> *Jeric G. Boleza, College of Mathematics and Computing Science, University of Abra, Philippines.*

---

## 1. Dataset Files Overview

| File Name | Trials ($N$) | Cohort | Description |
|---|---|---|---|
| `benchmark_trials_225n.csv` | 225 | S01, S03, S07–S09, S11–S15 (10 subjects) | Complete benchmark dataset combining both the calibration cohort ($N=150$) and the blind held-out evaluation cohort ($N=75$). |
| `benchmark_trials_225n.json` | 225 | 10 subjects | Structured JSON format of the 225-trial benchmark. |
| `calib_final.csv` | 150 | S01, S03, S07, S08, S09 (5 subjects) | Dedicated calibration partition used to estimate device-conditioned $(\mu, \sigma)$ parameters and logistic weights ($w, b$). |
| `biometric_benchmark_trials.csv` | 175 | Pilot cohort | Historical baseline corpus containing multi-stream scores across depth, rPPG, and texture. |
| `splits.json` | 175 | Pilot cohort | Split mapping dictionary defining `train`, `dev`, and `test` assignments for baseline benchmarking. |

---

## 2. Experimental Partitioning Protocol

To enforce strict compliance with ISO/IEC 30107-3 and prevent data leakage, the $N = 225$ trials are divided into two mutually exclusive, strictly subject-disjoint partitions:

### A. Calibration Partition ($N = 150$ Trials, Subjects S01, S03, S07, S08, S09)
* **Bona Fide Live Trials:** 50 presentations (10 per subject)
* **Presentation Attacks (PAIs):** 100 presentations (50 Print Photo PAI-1, 50 Screen Video Replay PAI-2)
* **Role:** Estimating sensor-conditioned z-score parameters $(\mu^{(d)}, \sigma^{(d)})$, fitting the logistic regression fusion weights $w$, and fixing the operating threshold ($\tau = 0.4894$).

### B. Held-Out Test Partition ($N = 75$ Trials, Subjects S11, S12, S13, S14, S15)
* **Bona Fide Live Trials:** 25 presentations (5 per subject)
* **PAI-1 (Print Attack):** 25 presentations (5 per subject)
* **PAI-2 (Screen Replay):** 25 presentations (5 per subject)
* **Role:** Unbiased, zero-leakage evaluation of the frozen decision rule. Completely unseen subjects during training.

---

## 3. Presentation Attack Instruments (PAI) Taxonomy

In accordance with ISO/IEC 30107-3 attack potential criteria, two principal 2D PAI species were fabricated:

| PAI ID | Attack Category | Fabrication Medium | Optical / Display Profile |
|---|---|---|---|
| **PAI-1** | Printed Photograph | Canon PIXMA Pro-200 / HP Color LaserJet | 600+ DPI matte paper & 260 gsm glossy photographic substrate |
| **PAI-2** | Dynamic Screen Video Replay | Samsung Galaxy S23 (Dynamic AMOLED 2X, 120 Hz, 425 ppi) & iPad Pro (Liquid Retina IPS, 264 ppi) | 1080p / 4K native dynamic facial video recordings played at natural distances |

---

## 4. Hardware Acquisition Optical Sensors

Biometric presentations were acquired across two distinct camera sensor architectures:
1. **Dedicated External Optical Sensor (USB-Cam):**
   * Hardware ID: `4c4a:4a55`
   * Characteristics: Wide-angle glass optics, linear Image Signal Processor (ISP), minimal non-linear edge filtering.
2. **Commodity Integrated HD Webcam (HD-UVC):**
   * Hardware ID: `322e:2103`
   * Characteristics: Small-aperture sensor, aggressive 3D temporal noise reduction (3D-NR), dynamic auto-gain control (AGC).

---

## 5. Data Schema & Feature Column Dictionary

The dataset CSV contains 35 validated columns:

| Column Name | Data Type | Description |
|---|---|---|
| `trial_id` | Integer | Unique primary key identifier for the capture trial |
| `session_id` | String | Capture session batch identifier (e.g., `S01-20261109-A`) |
| `subject_id` | String | De-identified participant code (`S01` through `S15`) |
| `sample_category` | String | Presentation category: `bona_fide`, `print_photo`, or `screen_replay` |
| `ground_truth_is_live` | Integer | Binary ground truth label (`1` = Bona Fide Live, `0` = Attack) |
| `device_label` | String | Hardware identifier of the optical sensor used |
| `execution_ms` | Float | Edge microprocessor feature extraction latency in milliseconds |
| `sharp_rel` | Float | **Stream 2:** Gain-invariant relative Laplacian texture gradient sharpness |
| `band_ratio` | Float | **Stream 1:** Dual-scale spatial frequency energy persistence ratio |
| `chroma_cv` | Float | Spatial chrominance coefficient of variation across facial crop |
| `blue_red` | Float | Chromatic ratio (explicitly clamped to weight 0.0 for ethical bias prevention) |
| `spec_frac` | Float | **Stream 3:** Multi-channel indicator specular highlight saturation fraction |
| `depth_rel` | Float | **Stream 4:** Perspective-induced 3D anthropometric depth disparity |
| `motion_rel` | Float | **Stream 5:** Optical micro-motion temporal gradient |
| `crop_min_luma` | Integer | Darkest 8-bit luma pixel in the 64x64 ROI (checked for sensor AGC confound) |
| `illum_condition` | String | Controlled ambient illumination setting (e.g., `L4 - daylight, blinds open`) |
| `fused_score` | Float | Calibrated multimodal liveness probability $P(\text{Bona Fide} \mid z) \in [0, 1]$ |
| `verdict` | String | Operational system decision (`LIVE` or `ATTACK`) |
| `pad_constants_version` | String | Provenance hash stamp of the frozen calibration constants |
| `corpus_split` | String | Designated protocol split: `calibration` or `test` |
| `created_at` | Timestamp | ISO 8601 acquisition timestamp |

---

## 6. Ethical Compliance, Privacy, and Data Governance

* **Institutional Review:** The empirical data collection protocol complied with institutional research governance standards.
* **Informed Consent:** All human participants provided explicit written informed consent for facial biometric recording and presentation attack simulation.
* **Biometric Privacy Safeguards:** To ensure compliance with international privacy regulations (including GDPR and national Data Privacy Acts), raw 1080p biometric video facial captures are retained in secure air-gapped institutional storage. The public benchmark repository distributes extracted, de-identified mathematical feature descriptors, sensor logs, and scalar liveness scores, enabling full mathematical and algorithmic reproducibility without releasing identifiable biometric facial portraits.
