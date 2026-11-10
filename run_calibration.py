#!/usr/bin/env python
"""Calibrate the PAD decision rule and judge whether the benchmark is fit to take test data.

Treatment of data
-----------------
Calibration rows are *for fitting*. Test rows are *for measuring*, and are never
touched here. Any row marked corpus_split='test' in the input is dropped with a
warning: fitting on it would destroy the only honest number the study can produce.

What it produces
----------------
1. The constants block (paste into pad-fusion-v2.js), via the same procedure as
   fit_fusion_v2.py.
2. A calibration report: per-feature separability, ablation, bootstrap intervals
   for weights and threshold, threshold sensitivity, capture-confound check.
3. A readiness verdict against explicit criteria — the answer to "is this fit to
   take test data yet", including what is blocking it.

Usage
-----
    python benchmark/evaluation/calibrate.py --csv calib.csv --out calibration_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = ["sharp_rel", "band_ratio", "chroma_cv", "blue_red", "spec_frac", "depth_rel", "motion_rel"]

MIN_ROWS = 150
MIN_SUBJECTS = 3
MIN_PER_CLASS = 30
MIN_BOOT_AUC_LOW = 0.85
MIN_MARGIN = 0.10
MAX_CONFOUND_AUC = 0.80

RNG = np.random.RandomState(42)


def max_margin_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Widest-gap threshold, computed correctly for separable classes.

    With no overlap the midpoint between the strongest attack and the weakest genuine
    maximises the distance to both classes, so it is returned directly. The band search
    below is only a fallback: on separable data it can return the midpoint of a band
    that lies inside the attack distribution, leaving almost no margin above the
    strongest attack. (Observed 2026-11-04: tau 0.3451, only +0.013 above the top
    attack, when the class midpoint was 0.4686 with +0.137 on both sides.)
    """
    a = np.sort(s[y == 0])
    l = np.sort(s[y == 1])
    if len(a) == 0 or len(l) == 0:
        return 0.5
    if a.max() < l.min():
        return float(np.clip((a.max() + l.min()) / 2.0, 0.0, 1.0))

    best_gap, best_t = -1.0, None
    for attack in a:
        above = l[l > attack]
        if len(above) == 0:
            continue
        gap = above[0] - attack
        if gap > best_gap:
            best_gap, best_t = gap, float((attack + above[0]) / 2.0)
    return 0.5 if best_t is None else float(np.clip(best_t, 0.0, 1.0))


def fit(X: np.ndarray, y: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
    clf.fit((X - mu) / sigma, y)
    return mu, sigma, clf.coef_[0], float(clf.intercept_[0])


def predict(X, mu, sigma, w, w0) -> np.ndarray:
    z = ((X - mu) / sigma) @ w + w0
    return 1.0 / (1.0 + np.exp(-z))


def main() -> int:
    # The report contains Δ and other non-ASCII; a cp1252 console must not crash it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--group-col", default="subject_id")
    ap.add_argument("--out", help="write the markdown report here")
    ap.add_argument("--json", help="write raw diagnostics here")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument(
        "--exclude-device",
        action="append",
        default=None,
        metavar="SUBSTR",
        help="drop rows whose device_label contains SUBSTR (repeatable). Used to keep the "
             "virtual-camera pipeline out of the calibration fit without deleting its rows.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # A device whose rows are excluded here must still be analysed — just not fitted on.
    # The POCO F5 arrives through a Windows Virtual Camera, which resamples the frame and
    # is not the phone's own pipeline: a different system, not a fourth camera.
    excluded_devices = args.exclude_device or []
    dropped_excluded = 0
    if excluded_devices and "device_label" in df.columns:
        mask = df["device_label"].astype(str).apply(
            lambda d: any(s.lower() in d.lower() for s in excluded_devices)
        )
        dropped_excluded = int(mask.sum())
        if dropped_excluded:
            print(f"warning: excluding {dropped_excluded} row(s) from device(s) "
                  f"{sorted(df.loc[mask, 'device_label'].unique())} — not fitted here.", file=sys.stderr)
        df = df[~mask]

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"error: missing feature columns {missing}", file=sys.stderr)
        return 2

    # --- enforce the split ---
    dropped_test = 0
    if "corpus_split" in df.columns:
        dropped_test = int((df["corpus_split"] == "test").sum())
        if dropped_test:
            print(f"warning: dropping {dropped_test} row(s) marked test — calibration must not see them.",
                  file=sys.stderr)
        df = df[df["corpus_split"] != "test"]

    # A row with no subject cannot be grouped or split. Treating "unknown" as a
    # subject would invent a group and inflate the readiness report — a fake subject
    # is worse than a missing one, so these rows are excluded and counted.
    if args.group_col not in df.columns:
        print(f"error: no '{args.group_col}' column; subject-disjoint calibration is impossible.",
              file=sys.stderr)
        return 2

    before_subject = len(df)
    df = df[df[args.group_col].notna() & (df[args.group_col].astype(str).str.strip() != "")]
    dropped_no_subject = before_subject - len(df)
    if dropped_no_subject:
        print(f"warning: {dropped_no_subject} row(s) have no subject id and were excluded.",
              file=sys.stderr)

    df = df.dropna(subset=FEATURES + ["ground_truth_is_live"])
    if df.empty:
        print("error: no attributable calibration rows", file=sys.stderr)
        return 2

    y = df["ground_truth_is_live"].astype(int).values
    groups = df[args.group_col].astype(str).values
    X = df[FEATURES].values.astype(float)
    n_subj = len(np.unique(groups))

    # A mid-session corpus is single-class by construction. Every metric below is
    # undefined then — APCER needs attacks, BPCER needs genuine trials — so say that
    # instead of dying on a sklearn exception halfway through the report.
    if len(np.unique(y)) < 2:
        n_live0, n_att0 = int(y.sum()), int((1 - y).sum())
        msg = (f"one class only: live {n_live0}, attack {n_att0}. "
               f"APCER, BPCER, AUC and the threshold are all undefined until both classes exist.")
        print(f"error: {msg}", file=sys.stderr)
        print("VERDICT: NOT READY", file=sys.stderr)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(f"# Calibration Report\n\n**Corpus:** `{args.csv}`  \n"
                         f"**Rows used:** {len(df)}  \n**Subjects:** {n_subj}\n\n"
                         f"## Verdict: NOT READY\n\n- **{msg}**\n")
            print(f"report written to {args.out}", file=sys.stderr)
        return 1

    # --- diagnostics ---
    checks: list[tuple[str, str, bool, bool]] = []   # name, detail, ok, critical

    checks.append(("calibration rows", f"{len(df)} (min {MIN_ROWS})", len(df) >= MIN_ROWS, True))
    checks.append(("rows attributable to a subject", f"{len(df)} of {before_subject} loaded",
                   dropped_no_subject == 0, True))
    checks.append(("independent subjects", f"{n_subj} (min {MIN_SUBJECTS})", n_subj >= MIN_SUBJECTS, True))

    # Distinct capture days. One session cannot reveal whether a cue's *direction* is
    # stable — and an unstable direction is exactly how 9 screen replays were accepted
    # when the rule kept rewarding sharpness after a sharper display appeared.
    if "created_at" in df.columns:
        n_sessions = int(pd.to_datetime(df["created_at"], errors="coerce").dt.date.nunique())
    else:
        n_sessions = 1
    checks.append(("capture sessions", f"{n_sessions} day(s) (min 2)", n_sessions >= 2, True))

    n_live, n_attack = int(y.sum()), int((1 - y).sum())
    checks.append(("rows per class", f"live {n_live}, attack {n_attack} (min {MIN_PER_CLASS} each)",
                   min(n_live, n_attack) >= MIN_PER_CLASS, True))

    # per-feature separability + degeneracy
    feat_rows = []
    dead = []
    for j, f in enumerate(FEATURES):
        s = X[:, j]
        if np.nanstd(s) < 1e-12:
            feat_rows.append((f, float("nan"), "constant"))
            dead.append(f)
            continue
        auc = roc_auc_score(y, s)
        auc = max(auc, 1 - auc)
        feat_rows.append((f, auc, "weak" if auc < 0.7 else "moderate" if auc < 0.85 else "strong"))

    # fit + threshold (with per-device standardisation if multiple devices exist)
    has_devices = "device_label" in df.columns and df["device_label"].nunique() > 1
    dev_stats = {}
    Z = np.zeros_like(X)

    if has_devices:
        for dev, g in df.groupby("device_label"):
            m_d = g[FEATURES].mean().values
            s_d = g[FEATURES].std().values
            s_d = np.where(s_d < 1e-9, 1.0, s_d)
            dev_stats[dev] = {"mu": m_d, "sigma": s_d, "indices": df.index.get_indexer(g.index)}
            idx = dev_stats[dev]["indices"]
            Z[idx] = (g[FEATURES].values - m_d) / s_d
        mu = df[FEATURES].mean().values
        sigma = df[FEATURES].std().values
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
    else:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        Z = (X - mu) / sigma

    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
    clf.fit(Z, y)
    w = clf.coef_[0]
    w0 = float(clf.intercept_[0])
    s_in = 1.0 / (1.0 + np.exp(-(Z @ w + w0)))
    auc_in = roc_auc_score(y, s_in)

    dev_thresholds = {}
    if has_devices:
        margin_details = []
        all_margins_ok = True
        for dev, st in dev_stats.items():
            idx = st["indices"]
            s_d = s_in[idx]
            y_d = y[idx]
            t_d = max_margin_threshold(y_d, s_d)
            st["tau"] = t_d
            live = s_d[y_d == 1]
            atk = s_d[y_d == 0]
            m_hi = float(live.min() - t_d) if len(live) else float("nan")
            m_lo = float(t_d - atk.max()) if len(atk) else float("nan")
            st["margin_hi"] = m_hi
            st["margin_lo"] = m_lo
            dev_thresholds[dev] = (t_d, m_hi, m_lo)
            dev_short = dev.split('(')[0].strip()
            margin_details.append(f"{dev_short}: +{m_hi:.3f}/+{m_lo:.3f}")
            if m_hi < MIN_MARGIN or m_lo < MIN_MARGIN:
                all_margins_ok = False
        tau = max_margin_threshold(y, s_in)
        checks.append(("threshold margin (per-device)", "; ".join(margin_details), all_margins_ok, True))
    else:
        tau = max_margin_threshold(y, s_in)
        live_min = float(s_in[y == 1].min())
        atk_max = float(s_in[y == 0].max())
        margin_hi = live_min - tau
        margin_lo = tau - atk_max
        checks.append(("threshold margin", f"genuine {margin_hi:+.3f} above tau, attacks {margin_lo:+.3f} below",
                       margin_hi >= MIN_MARGIN and margin_lo >= MIN_MARGIN, True))

    # Every device must contribute both classes, or the fit learns the device instead of
    # the attack. Observed 2026-11-04: a session whose second camera produced 5 genuine
    # trials and no attacks fitted to in-sample AUC 1.000 and 0% ACER, then scored 61.5%
    # APCER on a session captured ninety minutes earlier — every attack on that camera
    # was accepted, because the only way to separate the classes in the fitting corpus was
    # to label that camera's sensor values as genuine.
    if "device_label" in df.columns and df["device_label"].nunique() > 1:
        incomplete = []
        for dev, g in df.groupby("device_label"):
            present = set(g["ground_truth_is_live"].astype(int).unique())
            if len(present) < 2:
                incomplete.append(f"{dev.split('(')[0].strip()} "
                                  f"({'genuine only' if 1 in present else 'attacks only'})")
        checks.append((
            "every device has both classes",
            "; ".join(incomplete) if incomplete else f"{df['device_label'].nunique()} device(s) complete",
            not incomplete,
            True,
        ))

    # bootstrap over rows (subject-level resampling impossible below 2 subjects)
    boot = []
    for _ in range(args.boot):
        idx = RNG.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            clf_boot = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
            clf_boot.fit(Z[idx], y[idx])
            p = clf_boot.predict_proba(Z[idx])[:, 1]
            boot.append((roc_auc_score(y[idx], p), max_margin_threshold(y[idx], p)))
        except Exception:
            continue
    boot_auc = np.array([b[0] for b in boot])
    boot_tau = np.array([b[1] for b in boot])
    auc_lo, auc_hi = (np.percentile(boot_auc, 2.5), np.percentile(boot_auc, 97.5)) if len(boot) else (np.nan, np.nan)
    tau_lo, tau_hi = (np.percentile(boot_tau, 2.5), np.percentile(boot_tau, 97.5)) if len(boot) else (np.nan, np.nan)
    boot_ok = len(boot) > 0 and auc_lo >= MIN_BOOT_AUC_LOW
    checks.append(("bootstrap AUC lower bound", f"{auc_lo:.3f} (min {MIN_BOOT_AUC_LOW})", bool(boot_ok),
                   n_subj >= 2))

    # leave-one-out (stability, not generalisation)
    loo_correct = 0
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        if len(np.unique(y[mask])) < 2:
            continue
        clf_loo = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
        clf_loo.fit(Z[mask], y[mask])
        p = clf_loo.predict_proba(Z[i:i + 1])[:, 1][0]
        t_ = max_margin_threshold(y[mask], clf_loo.predict_proba(Z[mask])[:, 1])
        if (p >= t_) == bool(y[i]):
            loo_correct += 1
    loo_acc = loo_correct / len(y)

    # ablation
    abl = []
    for j, f in enumerate(FEATURES):
        keep = [k for k in range(len(FEATURES)) if k != j]
        try:
            clf_abl = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
            clf_abl.fit(Z[:, keep], y)
            p_abl = clf_abl.predict_proba(Z[:, keep])[:, 1]
            abl.append((f, auc_in - roc_auc_score(y, p_abl)))
        except Exception:
            abl.append((f, float("nan")))

    # Capture confound — measured pooled AND per device.
    #
    # A statistic can cancel out when devices are pooled while remaining a perfect
    # separator inside every device. `crop_min_luma` does exactly that: AUC 1.000 on each
    # of three cameras, with the sign flipping between them (USB live 25 -> print 46,
    # USB32 live 69 -> print 43). Pooled AUC 0.655 therefore looks harmless, but the
    # instrument now standardises per device, and that normalisation removes the
    # inversion — leaving a global weight free to learn "the crop looks like paper".
    # Rows in the corpus are printed photos held up to the camera, not fixed framings,
    # so this is a capture shortcut, not a presentation-attack cue.
    luma_col = "crop_min_luma" if "crop_min_luma" in df.columns else (
        "ambient_lux" if "ambient_lux" in df.columns else None
    )
    covariate_cols = [c for c in (luma_col, "execution_ms") if c]

    confound = []
    for col in covariate_cols:
        v = pd.to_numeric(df[col], errors="coerce").values
        if not np.isnan(v).all() and len(np.unique(y)) == 2:
            try:
                a = roc_auc_score(y, v)
                confound.append((col, max(a, 1 - a)))
            except Exception:
                pass

    per_device = []
    if "device_label" in df.columns and df["device_label"].nunique() > 1:
        for dev, g in df.groupby("device_label"):
            yy = g["ground_truth_is_live"].astype(int).values
            if len(np.unique(yy)) < 2:
                continue
            for col in covariate_cols:
                v = pd.to_numeric(g[col], errors="coerce").values
                if np.isnan(v).all() or np.nanstd(v) < 1e-12:
                    continue
                try:
                    a = roc_auc_score(yy, v)
                    per_device.append((dev, col, max(a, 1 - a)))
                except Exception:
                    pass
    worst_per_device = max((a for _, _, a in per_device), default=None)

    conf_ok = all(a < MAX_CONFOUND_AUC for _, a in confound) if confound else True
    if worst_per_device is not None:
        conf_ok = conf_ok and worst_per_device < MAX_CONFOUND_AUC

    conf_detail = ", ".join(f"{c} AUC {a:.3f}" for c, a in confound) or "not measurable"
    if worst_per_device is not None:
        offenders = [f"{d.split('(')[0].strip()} {a:.3f}" for d, _, a in per_device if a >= MAX_CONFOUND_AUC]
        conf_detail += f" (pooled); worst per-device {worst_per_device:.3f}"
        if offenders:
            conf_detail += " — " + "; ".join(offenders[:4])
    checks.append(("capture confound", conf_detail, conf_ok, True))

    # Was the lighting actually recorded? Without a label the protocol's "illuminant
    # held constant across categories" is an assumption nobody can check later.
    if "illum_condition" in df.columns:
        illum = df["illum_condition"].astype("object")
        unknown = int(illum.isna().sum() + (illum.astype(str).str.strip() == "").sum()
                      + (illum.astype(str).str.lower() == "nan").sum())
        levels = sorted(set(illum.dropna().astype(str).str.strip()) - {"", "nan"})
        checks.append((
            "illumination record",
            f"{len(df) - unknown}/{len(df)} rows labelled, {len(levels)} level(s)"
            + (f": {', '.join(levels[:3])}" if levels else ""),
            unknown == 0 and len(levels) > 0,
            True,
        ))
    else:
        checks.append(("illumination record",
                       "no illum_condition column — the lighting was never recorded", False, True))

    if dead:
        checks.append(("degenerate features", ", ".join(dead), False, False))

    critical_fail = [c for c in checks if c[3] and not c[2]]
    advisory_fail = [c for c in checks if not c[3] and not c[2]]
    ready = not critical_fail

    # ---- report ----
    md = []
    add = md.append
    add("# Calibration Report\n")
    add(f"**Corpus:** `{args.csv}`  \n**Rows used:** {len(df)} (dropped {dropped_test} test rows"
        + (f", excluded {dropped_excluded} row(s) from {len(excluded_devices)} device filter(s)" if dropped_excluded else "")
        + ")  ")
    add(f"**Subjects:** {n_subj}  \n**Generated by:** `benchmark/evaluation/calibrate.py`\n")
    add(f"## Verdict: {'READY FOR TEST DATA' if ready else 'NOT READY'}\n")
    if not ready:
        add("Blocking issues:\n")
        for name, detail, _, _ in critical_fail:
            add(f"- **{name}** — {detail}")
        add("")
    if advisory_fail:
        add("Advisory:\n")
        for name, detail, _, _ in advisory_fail:
            add(f"- {name} — {detail}")
        add("")

    add("## Checks\n")
    add("| Check | Observed | Status |")
    add("|---|---|---|")
    for name, detail, ok, crit in checks:
        status = "pass" if ok else ("**FAIL**" if crit else "warn")
        add(f"| {name} | {detail} | {status} |")
    add("")

    add("## Capture provenance\n")
    if per_device:
        add("Covariate separability inside each device. Anything at or above "
            f"{MAX_CONFOUND_AUC} is a shortcut the standardised model can learn:\n")
        add("| Device | Covariate | AUC |")
        add("|---|---|---|")
        for dev, col, a in sorted(per_device, key=lambda t: -t[2]):
            add(f"| {dev} | `{col}` | {a:.3f} |")
        add("")
    if luma_col:
        add(f"- `{luma_col}` is the darkest 8-bit luma pixel of the 64x64 face crop — a framing "
            "statistic, not an illuminance reading. It does not measure the room.")
    else:
        add("- No crop-brightness column present in this corpus.")
    add("- The corpus contains no lux reading. Lighting control is asserted through the "
        "`illum_condition` label, or not at all.")
    add("")

    add("## Per-feature separability\n")
    add("| Feature | AUC | Weight (z) | Assessment |")
    add("|---|---|---|---|")
    for j, (f, auc, note) in enumerate(feat_rows):
        add(f"| `{f}` | {auc:.3f} | {w[j]:+.3f} | {note} |")
    add("")

    add("## Ablation (ΔAUC when the feature is removed)\n")
    add("| Feature | ΔAUC |")
    add("|---|---|")
    for f, d in sorted(abl, key=lambda t: -(t[1] if t[1] == t[1] else -9)):
        add(f"| `{f}` | {d:+.4f} |")
    add("")

    add("## Fitted constants\n")
    add("```")
    add(f"version   : calib-{len(df)}n-{n_subj}s")
    add(f"tau (global) : {tau:.6f}   (bootstrap 95%: {tau_lo:.3f} .. {tau_hi:.3f})")
    if dev_thresholds:
        add("per-device thresholds & margins:")
        for dev, (td, mh, ml) in dev_thresholds.items():
            add(f"  {dev}: tau={td:.6f} (margin: live {mh:+.3f}, attack {ml:+.3f})")
    add(f"AUC (in-sample) : {auc_in:.4f}")
    add(f"AUC (bootstrap) : {auc_in:.4f} [{auc_lo:.3f}, {auc_hi:.3f}]")
    add(f"leave-one-out   : {loo_acc:.3f} (stability, not generalisation)")
    add(f"mu        : {[round(float(v), 6) for v in mu]}")
    add(f"sigma     : {[round(float(v), 6) for v in sigma]}")
    add(f"w         : {[round(float(v), 6) for v in w]}")
    add(f"w0        : {w0:.6f}")
    add("```")
    add("")
    add("## Interpretation limits\n")
    add(f"- Calibration used **{n_subj} subject(s)**. Any AUC here describes how well the rule fits "
        f"these people, not how it will treat an unseen person.")
    add("- Test rows must be disjoint by subject. If a test subject also appears in calibration, the "
        "result is void.")
    add("- The threshold is chosen from the same rows used to fit, which inflates its apparent margin; "
        "the bootstrap interval is the honest range.")

    text = "\n".join(md)

    print(text)
    print("\n" + "=" * 70)
    print(f"VERDICT: {'READY FOR TEST DATA' if ready else 'NOT READY'}")
    if critical_fail:
        for name, detail, _, _ in critical_fail:
            print(f"  FAIL  {name}: {detail}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nreport written to {args.out}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "rows": int(len(df)), "subjects": int(n_subj), "ready": ready,
                "tau": tau, "auc_in_sample": float(auc_in),
                "bootstrap_auc": [float(auc_lo), float(auc_hi)],
                "bootstrap_tau": [float(tau_lo), float(tau_hi)],
                "loo_accuracy": float(loo_acc),
                "checks": [{"name": n, "detail": d, "ok": bool(o), "critical": bool(c)} for n, d, o, c in checks],
                "features": [{"feature": f, "auc": (None if a != a else float(a)), "note": note,
                              "weight": float(w[j])} for j, (f, a, note) in enumerate(feat_rows)],
                "ablation": [{"feature": f, "delta_auc": (None if d != d else float(d))} for f, d in abl],
            }, fh, indent=2)
        print(f"diagnostics written to {args.json}")

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
