"""
Phase 0 — Exploratory Data Analysis (EDA)
for IAAA Competition 2026: Open-Set Speaker Identification

Generates:
  - class_distribution.png       (pie chart: unknown vs known)
  - known_speakers_dist.png      (bar chart: files per known speaker)
  - Phase0_EDA_Report.md         (full report with stats & plots)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------- paths -------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "labels.csv"
PLOT_PIE = PROJECT_ROOT / "class_distribution.png"
PLOT_BAR = PROJECT_ROOT / "known_speakers_dist.png"
REPORT = PROJECT_ROOT / "Phase0_EDA_Report.md"

# ------------- 1. load & clean -------------
df = pd.read_csv(DATA_PATH)

# strip whitespace from column names (safety net)
df.columns = df.columns.str.strip()

# strip whitespace from cell contents
df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
df["audio_file"] = df["audio_file"].astype(str).str.strip()

print(f"[INFO] Loaded {len(df):,} rows, columns: {list(df.columns)}")

# missing values
missing = df.isnull().sum()
print(f"[INFO] Missing values:\n{missing}")

# duplicates
dup_rows = df.duplicated().sum()
dup_audio = df["audio_file"].duplicated().sum()
print(f"[INFO] Duplicate rows: {dup_rows}")
print(f"[INFO] Duplicate audio_file entries: {dup_audio}")

# drop any duplicates if present
if dup_rows > 0:
    df = df.drop_duplicates()
    print(f"[INFO] Dropped {dup_rows} duplicate row(s).")

# ------------- 2. split known / unknown -------------
unknown_mask = df["speaker_id"].str.lower() == "unknown"

df_unknown = df[unknown_mask]
df_known   = df[~unknown_mask]

total_files = len(df)
n_unknown   = len(df_unknown)
n_known     = len(df_known)

known_speakers = df_known["speaker_id"].unique()
n_known_speakers = len(known_speakers)

print(f"\n{'='*50}")
print(f"Total audio files        : {total_files:,}")
print(f"Unknown files            : {n_unknown:,}")
print(f"Known files              : {n_known:,}")
print(f"Unique known speakers    : {n_known_speakers:,}")
print(f"{'='*50}")

# per-speaker counts
speaker_counts = df_known["speaker_id"].value_counts()

min_files = speaker_counts.min()
max_files = speaker_counts.max()
mean_files = speaker_counts.mean()
median_files = speaker_counts.median()
std_files = speaker_counts.std()

print(f"\n--- Known-speaker statistics ---")
print(f"Min files per speaker    : {min_files}")
print(f"Max files per speaker    : {max_files}")
print(f"Mean files per speaker   : {mean_files:.4f}")
print(f"Median files per speaker : {median_files:.1f}")
print(f"Std  files per speaker   : {std_files:.4f}")

# Imbalance Ratio
imbalance_ratio = n_unknown / mean_files if mean_files > 0 else float("inf")
print(f"\nImbalance Ratio          : {imbalance_ratio:.4f}")
print(f"  (unknown files / avg-files-per-known-speaker)")

# Additional detail: how many speakers have how many files?
print(f"\n--- Distribution of files per known speaker ---")
for val in sorted(speaker_counts.unique()):
    cnt = (speaker_counts == val).sum()
    print(f"  {val:>2d} files  ->  {cnt:>3d} speaker(s)")

# relative frequency (for context)
unknown_frac = n_unknown / total_files * 100
known_frac   = n_known   / total_files * 100
print(f"\nUnknown proportion      : {unknown_frac:.2f}%")
print(f"Known proportion        : {known_frac:.2f}%")

# ------------- 3. visualisations -------------
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["figure.dpi"] = 150

# -- 3a. Pie chart: unknown vs known --
fig1, ax1 = plt.subplots(figsize=(7, 7))
labels_pie = [f"Unknown\n({n_unknown:,} files)", f"Known\n({n_known:,} files)"]
sizes_pie  = [n_unknown, n_known]
colors_pie = ["#e74c3c", "#2ecc71"]
explode_pie = (0.04, 0.04)

wedges, texts, autotexts = ax1.pie(
    sizes_pie,
    labels=labels_pie,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors_pie,
    explode=explode_pie,
    shadow=False,
    textprops={"fontsize": 13, "weight": "bold"},
    pctdistance=0.75,
)

for at in autotexts:
    at.set_fontsize(12)
    at.set_weight("bold")
    at.set_color("white")

ax1.set_title("Class Distribution: Unknown vs Known", fontsize=16, weight="bold", pad=20)
fig1.tight_layout()
fig1.savefig(PLOT_PIE, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\n[SAVED] {PLOT_PIE.name}")
plt.close(fig1)

# -- 3b. Bar chart: distribution of files per known speaker --
fig2, ax2 = plt.subplots(figsize=(12, 6))

# value counts of the per-speaker counts
dist = speaker_counts.value_counts().sort_index()
bars = ax2.bar(
    dist.index.astype(str),
    dist.values,
    color=sns.color_palette("viridis", len(dist)),
    edgecolor="white",
    linewidth=0.8,
)

# add count labels on top of each bar
i = 0
for bar, val in zip(bars, dist.values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        str(val),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )
    i = i + 1

ax2.set_xlabel("Number of audio files per speaker", fontsize=13, weight="bold")
ax2.set_ylabel("Number of speakers", fontsize=13, weight="bold")
ax2.set_title("Distribution of File Counts Among Known Speakers", fontsize=15, weight="bold", pad=12)
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# add a vertical line for the mean
ax2.axvline(
    x=mean_files - 0.5,
    color="red",
    linestyle="--",
    linewidth=2.0,
    alpha=0.7,
    label=f"Mean = {mean_files:.2f}",
)
ax2.legend(fontsize=12)

fig2.tight_layout()
fig2.savefig(PLOT_BAR, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[SAVED] {PLOT_BAR.name}")
plt.close(fig2)


# ------------- 4. Markdown report -------------
report_content = f"""# Phase 0 — Exploratory Data Analysis Report

**Project:** IAAA Competition 2026 — Open-Set Speaker Identification  
**Date:** 2026-07-16  
**Branch:** `feature/dataEDA`

---

## 1. Data Loading & Cleaning

| Metric | Value |
|--------|-------|
| Raw rows loaded | {total_files:,} |
| Missing values | {missing.sum()} |
| Duplicate rows | {dup_rows} |
| Duplicate audio files | {dup_audio} |
| Rows after cleaning | {len(df):,} |

---

## 2. Dataset Statistics

### Global Counts

- **Total audio files** : **{total_files:,}**
- **Unique known speakers (UUIDs)** : **{n_known_speakers:,}**
- **Audio files for `"unknown"` class** : **{n_unknown:,}** ({unknown_frac:.2f}%)
- **Audio files for known classes** : **{n_known:,}** ({known_frac:.2f}%)

### Per-Known-Speaker Distribution

| Statistic | Value |
|-----------|-------|
| Minimum files per speaker | {min_files} |
| Maximum files per speaker | {max_files} |
| Mean files per speaker | {mean_files:.4f} |
| Median files per speaker | {median_files:.1f} |
| Standard deviation | {std_files:.4f} |

### Speaker Frequency Breakdown

| Files per speaker | Number of speakers |
|-------------------|-------------------:|
"""

# build the frequency table
for val in sorted(speaker_counts.unique()):
    cnt = (speaker_counts == val).sum()
    report_content += f"| {val} | {cnt} |\n"

report_content += f"""
### Imbalance Ratio

$$
\\text{{Imbalance Ratio}} = \\frac{{\\text{{Unknown files}}}}{{\\text{{Mean files per known speaker}}}}
= \\frac{{{n_unknown}}}{{{mean_files:.4f}}}
= {imbalance_ratio:.4f}
$$

---

## 3. Visualisations

### 3.1 Class Distribution — Unknown vs Known

![Class Distribution]({PLOT_PIE.name})

### 3.2 Distribution of Files Among Known Speakers

![Known Speakers Distribution]({PLOT_BAR.name})

---

## 4. Analysis & Implications for Model Training

The dataset exhibits a **moderate class imbalance** between the `unknown` (OOD) class and the known-speaker classes:

- **{n_unknown:,} samples ({unknown_frac:.1f}%)** belong to the single `"unknown"` class (class **0**).
- The remaining **{n_known:,} samples ({known_frac:.1f}%)** are spread across **{n_known_speakers:,} known speakers**.
- The **Imbalance Ratio** is **{imbalance_ratio:.2f}**, meaning the unknown class has roughly **{imbalance_ratio:.1f}×** as many samples as the average known speaker.
- Among known speakers, the distribution is **almost perfectly balanced**: the majority of speakers have exactly **{speaker_counts.mode().values[0]}** samples, with only minor deviations.

### Key Challenges

1. **Open-Set Nature:** The model must not only classify {n_known_speakers:,} known speakers but also reliably detect OOD samples as class `0`. A naïve argmax over the softmax output will tend to over-confidently assign known labels to OOD samples.
2. **Imbalance at Class Level:** The `unknown` class is over-represented relative to individual known speakers. Without mitigation, the model may become biased toward predicting `unknown`.
3. **Balanced Among Known Speakers:** The known speakers are nearly uniformly distributed, which is favourable — no sub-sampling or re-weighting is needed *within* the known set.

### Recommended Strategies

| Strategy | Rationale |
|----------|-----------|
| **Weighted Cross-Entropy Loss** | Assign a higher weight to known-speaker classes and a slightly lower weight to the `unknown` class to counteract the imbalance, or use inverse-class-frequency weighting for the 448-class problem. |
| **Focal Loss** | Apply focal loss (Lin et al., 2017) to down-weight easy samples and focus training on hard-to-classify and OOD examples — well suited for open-set problems. |
| **Oversampling / Undersampling** | Since the known-speaker side is already well balanced, consider **undersampling** the `unknown` class during training and controlling OOD detection via an uncertainty threshold. |
| **Two-Head Architecture** | Train one head for known-speaker classification and a separate OOD detector (e.g., an energy-based or Mahalanobis-distance head) — a modern best-practice for open-set recognition. |
| **Data Augmentation** | Use SpecAugment, MixUp, or noise injection to increase effective diversity, especially for speakers with fewer than the modal sample count. |

### Conclusion

The dataset is **highly structured**: known speakers are nearly perfectly balanced, but the OOD (`unknown`) class introduces a **class-level imbalance** with an Imbalance Ratio of **{imbalance_ratio:.2f}**. A standard cross-entropy loss will likely suffice for known-speaker separation, but explicit **imbalance mitigation** (weighted loss, focal loss, or OOD-specific techniques) is essential for robust open-set performance. The recommended starting point is **Weighted Cross-Entropy** combined with **Focal Loss**, with validation against a held-out OOD set to tune the threshold for `unknown` rejection.

---

*Report generated programmatically via `src/eda.py`.*
"""

with open(REPORT, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"\n[SAVED] {REPORT.name}")
print("\n✅ Phase 0 EDA complete — all files generated.")
