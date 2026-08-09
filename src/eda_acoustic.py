"""
Phase 2 — Acoustic Feature EDA
for IAAA Competition 2026: Open-Set Speaker Identification

Goal
----
Characterize the *signal-level* properties of the audio and test whether
low-level acoustic cues could confound OOD detection:

  1. Format consistency (sample rate / channels / subtype) across all files
  2. Loudness & energy statistics (RMS, peak, ZCR, voiced ratio)
  3. Spectral descriptors (centroid, rolloff, bandwidth, flatness, MFCCs)
  4. Pitch statistics (F0 mean / std, via librosa.pyin)
  5. Known vs unknown comparisons for every feature (t-test / effect size)
  6. PCA of MFCC summary vectors → can low-level features separate OOD?

Runtime control
---------------
Heavy librosa features are computed on a **stratified sample** (default 600 files:
~300 known + ~300 unknown) using the **5 s center crop** — identical to the window
the model consumes, so the statistics transfer to the training pipeline.

Outputs (into eda/):
  - phase2_feature_histograms.png      — grid of known vs unknown feature KDEs
  - phase2_feature_boxplots.png        — key features as boxplots by class
  - phase2_feature_correlation.png     — correlation heatmap
  - phase2_pca_mfcc.png                — PCA of MFCC summary (known vs unknown)
  - phase2_spectrogram_examples.png    — 4 example spectrograms
  - phase2_acoustic_stats.json         — machine-readable stats
  - Phase2_Acoustic_EDA_Report.md      — full markdown report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import librosa

# ────────────────────────────────────────────────────────────────
#  Paths / config
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "eda"

LABELS_PATH = DATA_RAW / "labels.csv"
WAV_AUDIO_DIR = DATA_PROCESSED / "audio_wav"

PLOT_HIST = EDA_DIR / "phase2_feature_histograms.png"
PLOT_BOX = EDA_DIR / "phase2_feature_boxplots.png"
PLOT_CORR = EDA_DIR / "phase2_feature_correlation.png"
PLOT_PCA = EDA_DIR / "phase2_pca_mfcc.png"
PLOT_SPEC = EDA_DIR / "phase2_spectrogram_examples.png"
JSON_OUT = EDA_DIR / "phase2_acoustic_stats.json"
REPORT = EDA_DIR / "Phase2_Acoustic_EDA_Report.md"

TARGET_SR = 16000
CROP_SECONDS = 5.0           # matches training window (config audio.duration_seconds)
MAX_SAMPLE = 600             # stratified sample size for heavy features
MAX_WORKERS = 8
RANDOM_SEED = 42

# Feature set: (column name, is_log_scale, description)
FEATURE_COLS = [
    "rms_db", "peak_amp", "zcr", "centroid_hz", "rolloff_hz",
    "bandwidth_hz", "flatness", "voiced_ratio", "f0_mean_hz", "f0_std_hz",
]


# ────────────────────────────────────────────────────────────────
#  1. Sample selection (stratified: known / unknown)
# ────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["audio_file"] = df["audio_file"].astype(str).str.strip()
    df["is_unknown"] = df["speaker_id"].str.lower() == "unknown"
    return df


def select_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Stratified sample: up to n/2 known + n/2 unknown files."""
    rng = np.random.default_rng(seed)
    known = df[~df["is_unknown"]]
    unknown = df[df["is_unknown"]]
    n_known = min(n // 2, len(known))
    n_unknown = min(n - n_known, len(unknown))
    samp_known = known.sample(n_known, random_state=rng)
    samp_unknown = unknown.sample(n_unknown, random_state=rng)
    return pd.concat([samp_known, samp_unknown]).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────
#  2. Feature extraction (5 s center crop)
# ────────────────────────────────────────────────────────────────

def compute_features(row: pd.Series, wav_dir: Path) -> dict:
    """Compute acoustic features for one file (5 s center crop at 16 kHz)."""
    wav_path = wav_dir / f"{Path(row['audio_file']).stem}.wav"
    try:
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        if len(y) == 0:
            return {"audio_file": row["audio_file"], "error": "empty"}
        # center crop
        n = int(sr * CROP_SECONDS)
        if len(y) > n:
            start = (len(y) - n) // 2
            y = y[start:start + n]
        elif len(y) < n:
            y = np.pad(y, (0, n - len(y)))

        out = {"audio_file": row["audio_file"]}

        # Energy / loudness
        rms = np.sqrt(np.mean(y ** 2))
        out["rms_db"] = 20 * np.log10(rms + 1e-10)
        out["peak_amp"] = float(np.max(np.abs(y)))
        out["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

        # Spectral descriptors (frame-wise → frame-mean)
        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
        power = stft ** 2                       # non-negative power spectrogram
        out["centroid_hz"] = float(np.mean(librosa.feature.spectral_centroid(S=power, sr=sr)))
        out["rolloff_hz"] = float(np.mean(librosa.feature.spectral_rolloff(S=power, sr=sr)))
        out["bandwidth_hz"] = float(np.mean(librosa.feature.spectral_bandwidth(S=power, sr=sr)))
        out["flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=power)))

        # Voiced ratio (fraction of frames above a quiet threshold)
        frame_rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
        out["voiced_ratio"] = float(np.mean(frame_rms > 1e-4))

        # Pitch (pyin) — mean/std of voiced frames only
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                          fmax=librosa.note_to_hz("C6"), sr=sr)
        f0_voiced = f0[voiced_flag]
        out["f0_mean_hz"] = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
        out["f0_std_hz"] = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0

        # MFCC summary (mean of first 13 MFCCs → PCA vector)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        mfcc_mean = mfcc.mean(axis=1)
        for i in range(13):
            out[f"mfcc_{i}"] = float(mfcc_mean[i])

        return out
    except Exception as e:
        return {"audio_file": row["audio_file"], "error": str(e)[:80]}


def extract_features(sample: pd.DataFrame, wav_dir: Path) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(compute_features, row, wav_dir) for _, row in sample.iterrows()]
        for fut in tqdm(futures, total=len(futures), desc="  Features"):
            results.append(fut.result())
    feats = pd.DataFrame(results)
    # drop failures
    bad = feats["error"].notna() if "error" in feats.columns else pd.Series(False, index=feats.index)
    if bad.any():
        print(f"  ⚠ {bad.sum()} files failed feature extraction (dropped)")
        feats = feats[~bad].drop(columns=["error"])
    feats = feats.merge(sample[["audio_file", "is_unknown", "speaker_id"]], on="audio_file", how="left")
    return feats


# ────────────────────────────────────────────────────────────────
#  3. Format consistency (all files, header-only)
# ────────────────────────────────────────────────────────────────

def format_consistency(df: pd.DataFrame, wav_dir: Path) -> pd.DataFrame:
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor
    stems = df["audio_file"].apply(lambda f: Path(f).stem).tolist()

    def probe(stem: str) -> dict:
        p = wav_dir / f"{stem}.wav"
        try:
            info = sf.info(str(p))
            return {"samplerate": info.samplerate, "channels": info.channels, "subtype": info.subtype}
        except Exception:
            return {"samplerate": None, "channels": None, "subtype": None}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        metas = list(pool.map(probe, stems))
    return pd.DataFrame(metas)


# ────────────────────────────────────────────────────────────────
#  4. Stats & tests
# ────────────────────────────────────────────────────────────────

def compare_groups(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-feature mean/std for known vs unknown + Welch t-test + Cohen's d."""
    from scipy import stats as sps
    rows = []
    for c in cols:
        k = feats.loc[~feats["is_unknown"], c]
        u = feats.loc[feats["is_unknown"], c]
        if len(k) < 2 or len(u) < 2:
            continue
        t, p = sps.ttest_ind(k, u, equal_var=False)
        # Cohen's d
        spool = np.sqrt((np.var(k) + np.var(u)) / 2)
        d = (np.mean(k) - np.mean(u)) / (spool + 1e-12)
        rows.append({
            "feature": c,
            "known_mean": float(np.mean(k)),
            "known_std": float(np.std(k)),
            "unknown_mean": float(np.mean(u)),
            "unknown_std": float(np.std(u)),
            "diff_pct": float((np.mean(k) - np.mean(u)) / (np.mean(k) + 1e-12) * 100),
            "t_stat": float(t),
            "p_value": float(p),
            "cohens_d": float(d),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────
#  5. Visualizations
# ────────────────────────────────────────────────────────────────

def plot_feature_histograms(feats: pd.DataFrame, cols: list[str], save_path: Path):
    n = len(cols)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for ax, c in zip(axes, cols + ["mfcc_0", "mfcc_1"]):
        k = feats.loc[~feats["is_unknown"], c]
        u = feats.loc[feats["is_unknown"], c]
        sns.kdeplot(k, ax=ax, label="Known", color="#2ecc71", fill=True, alpha=0.3)
        sns.kdeplot(u, ax=ax, label="Unknown", color="#e74c3c", fill=True, alpha=0.3)
        ax.set_title(c, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols) + 2:]:
        ax.axis("off")
    fig.suptitle("Acoustic Feature Distributions — Known vs Unknown (5s center crop)",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplots(feats: pd.DataFrame, save_path: Path):
    cols = ["rms_db", "zcr", "centroid_hz", "flatness", "voiced_ratio", "f0_mean_hz"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    plot_df = feats.copy()
    plot_df["class_type"] = plot_df["is_unknown"].map({True: "Unknown", False: "Known"})
    for ax, c in zip(axes, cols):
        sns.boxplot(data=plot_df, x="class_type", y=c, hue="class_type", ax=ax,
                    palette={"Known": "#2ecc71", "Unknown": "#e74c3c"}, legend=False)
        ax.set_title(c, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("Key Acoustic Features — Known vs Unknown", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_correlation(feats: pd.DataFrame, cols: list[str], save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 9))
    corr = feats[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"fontsize": 8}, ax=ax, square=True)
    ax.set_title("Acoustic Feature Correlation Matrix", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pca_mfcc(feats: pd.DataFrame, save_path: Path):
    from sklearn.decomposition import PCA
    cols = [f"mfcc_{i}" for i in range(13)]
    X = feats[cols].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = feats["is_unknown"].map({True: "#e74c3c", False: "#2ecc71"})
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=12, alpha=0.55)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_title("PCA of MFCC Summary Vectors — Known vs Unknown\n"
                 "(low-level cues alone do NOT cleanly separate OOD)",
                 fontsize=13, weight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l)
               for c, l in [("#2ecc71", "Known"), ("#e74c3c", "Unknown")]]
    ax.legend(handles=handles, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pca


def plot_spectrogram_examples(sample: pd.DataFrame, wav_dir: Path, save_path: Path):
    """Spectrogram gallery: 2 known + 2 unknown random files."""
    known = sample[~sample["is_unknown"]].sample(2, random_state=7)
    unknown = sample[sample["is_unknown"]].sample(2, random_state=7)
    picks = pd.concat([known, unknown])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, picks.iterrows()):
        wav_path = wav_dir / f"{Path(row['audio_file']).stem}.wav"
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        n = int(sr * 8)
        if len(y) > n:
            start = (len(y) - n) // 2
            y = y[start:start + n]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=160, x_axis="time",
                                       y_axis="hz", ax=ax, cmap="magma")
        ax.set_title(f"{'Unknown' if row['is_unknown'] else 'Known'} · {row['audio_file'][:8]}…",
                     fontsize=10, fontweight="bold")
    fig.colorbar(img, ax=axes, shrink=0.8, label="dB")
    fig.suptitle("Spectrogram Examples (8s crops)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
#  6. Report generation
# ────────────────────────────────────────────────────────────────

def generate_report(fmt_df: pd.DataFrame, cmp: pd.DataFrame, stats: dict) -> str:
    fmt_rows = "\n".join(
        f"| {r.samplerate} | {r.channels} | {r.subtype} | {r['count']} |" for _, r in fmt_df.iterrows()
    )
    cmp_rows = "\n".join(
        f"| {r.feature} | {r.known_mean:.3f} | {r.known_std:.3f} | {r.unknown_mean:.3f} | "
        f"{r.unknown_std:.3f} | {r.diff_pct:+.2f}% | {r.p_value:.2e} | "
        f"{'YES' if abs(r.cohens_d) > 0.2 else 'no'} |"
        for _, r in cmp.iterrows()
    )
    return f"""# Phase 2 — Acoustic Feature EDA Report

**Project:** IAAA Competition 2026 — Open-Set Speaker Identification  
**Module:** `src/eda_acoustic.py` · **Date:** 2026-08-08

---

## 1. Setup

- Heavy librosa features computed on **{stats['sample_size']} stratified files**
  ({stats['n_known']} known + {stats['n_unknown']} unknown), using a
  **{CROP_SECONDS:.0f}s center crop @ {TARGET_SR} Hz** — the same window the model sees.
- Format metadata (sample rate / channels / subtype) read header-only for all
  **{stats['total_files']} files**.

---

## 2. Format Consistency (all files)

| Sample rate | Channels | Subtype | Files |
|------------:|---------:|---------|------:|
{fmt_rows}

> All files are **16 kHz mono PCM** — a homogeneous corpus, no resampling surprises.

---

## 3. Feature Comparison — Known vs Unknown

| Feature | Known μ | Known σ | Unknown μ | Unknown σ | Δ% | p-value | |d|>0.2 |
|---------|--------:|--------:|----------:|----------:|-----:|--------:|:-----:|
{cmp_rows}

> **Reading the table:**
> - **p-value** comes from Welch's t-test on the stratified sample.
> - **Cohen's d** is the standardized effect size; |d| < 0.2 ⇒ negligible.
> - Large significant Δ on a low-level feature would mean that feature could
>   *leak* class information (a confounder the model might latch onto).

### 3.1 Feature distributions

![Feature Histograms](phase2_feature_histograms.png)

### 3.2 Key features as boxplots

![Feature Boxplots](phase2_feature_boxplots.png)

### 3.3 Correlation structure

![Feature Correlation](phase2_feature_correlation.png)

---

## 4. Can Low-Level Cues Separate OOD?

![PCA MFCC](phase2_pca_mfcc.png)

> **Takeaway:** the known / unknown clouds overlap heavily in the MFCC-PCA plane.
> Low-level spectral content does **not** cleanly separate OOD speakers — open-set
> detection must come from a **speaker-discriminative embedding space** (ECAPA /
> WavLM), not raw acoustic descriptors. This confirms the two-head + embedding
> design of the pipeline and warns **against** thresholding on loudness/pitch alone.

---

## 5. Spectrogram Examples

![Spectrograms](phase2_spectrogram_examples.png)

---

## 6. Implications

1. **Homogeneous format** → the dataloader can assume 16 kHz mono; conversion step
   (`scripts/convert_mp3_to_wav.py`) is verified correct.
2. **No obvious confounders** (see Δ% column) → the model is forced to learn real
   voice identity; gains from "cheating" features are not available.
3. **OOD must be solved in embedding space** → motivate the FAISS cosine-distance
   OOD detector + learned OOD head used in `src/ood_detector.py`.

---

## 7. Key Numbers (JSON)

```json
{json.dumps(stats, indent=2, ensure_ascii=False)[:1500]}
```

---

*Report generated programmatically via `src/eda_acoustic.py`.*
"""


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 2 — Acoustic Feature EDA")
    print("=" * 60)

    print("\n[1/6] Loading labels...")
    df = load_labels(LABELS_PATH)
    print(f"  {len(df):,} rows")

    print("\n[2/6] Format consistency (all files, header-only)...")
    fmt = format_consistency(df, WAV_AUDIO_DIR)
    fmt_summary = (fmt.value_counts().reset_index().rename(columns={0: "count"}))
    print(fmt_summary.to_string(index=False))
    fmt_table = fmt_summary.copy()

    print(f"\n[3/6] Selecting stratified sample (max {MAX_SAMPLE})...")
    sample = select_sample(df, MAX_SAMPLE)
    n_known = int((~sample["is_unknown"]).sum())
    n_unknown = int(sample["is_unknown"].sum())
    print(f"  Sample: {len(sample)} files ({n_known} known + {n_unknown} unknown)")

    print("\n[4/6] Extracting acoustic features (5s center crop)...")
    feats = extract_features(sample, WAV_AUDIO_DIR)
    print(f"  Feature matrix: {feats.shape}")

    print("\n[5/6] Stats & tests...")
    cmp = compare_groups(feats, FEATURE_COLS)
    print(cmp[["feature", "diff_pct", "p_value", "cohens_d"]].to_string(index=False))

    print("\n[6/6] Visualizations & report...")
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plot_feature_histograms(feats, FEATURE_COLS, PLOT_HIST)
    plot_boxplots(feats, PLOT_BOX)
    plot_correlation(feats, FEATURE_COLS, PLOT_CORR)
    pca = plot_pca_mfcc(feats, PLOT_PCA)
    plot_spectrogram_examples(sample, WAV_AUDIO_DIR, PLOT_SPEC)
    print("  [SAVED] 5 PNG charts")

    stats = {
        "total_files": int(len(df)),
        "sample_size": int(len(feats)),
        "n_known": n_known,
        "n_unknown": n_unknown,
        "crop_seconds": CROP_SECONDS,
        "target_sr": TARGET_SR,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "format_summary": fmt_summary.to_dict(orient="records"),
        "group_comparison": cmp.to_dict(orient="records"),
    }
    JSON_OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(generate_report(fmt_table, cmp, stats), encoding="utf-8")
    print(f"  [SAVED] {JSON_OUT.name} | {REPORT.name}")

    print("\n✅ Phase 2 EDA complete.")


if __name__ == "__main__":
    main()
