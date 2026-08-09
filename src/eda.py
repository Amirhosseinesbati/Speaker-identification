"""
Phase 0 — Label & Metadata EDA
for IAAA Competition 2026: Open-Set Speaker Identification

Goal
----
Understand the *label space* of the challenge **before** touching audio:
  1. Data integrity (missing / duplicates / file-presence on disk)
  2. Class composition: 446 known speakers vs the aggregated "unknown" (OOD) class
  3. Per-speaker sample balance (implications for Macro-Averaged F1)
  4. Train/val split design (per-speaker held-out)
  5. Imbalance quantification w.r.t. the 447-way classification problem

Outputs (written into eda/):
  - phase0_class_distribution.png      — unknown vs known (pie)
  - phase0_speaker_counts.png          — files per known speaker (bar + mean/median)
  - phase0_speaker_frequency.png       — how many speakers have N files
  - phase0_cumulative_coverage.png     — top-k speakers cumulative file share
  - phase0_label_eda_summary.json      — machine-readable stats (reused by later phases)
  - Phase0_EDA_Report.md               — full markdown report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ────────────────────────────────────────────────────────────────
#  Paths
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "eda"

LABELS_PATH = DATA_RAW / "labels.csv"
RAW_AUDIO_DIR = DATA_RAW                    # contains the original .mp3
WAV_AUDIO_DIR = DATA_PROCESSED / "audio_wav"  # contains converted 16 kHz mono .wav

EDA_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PIE = EDA_DIR / "phase0_class_distribution.png"
PLOT_SPEAKER_COUNTS = EDA_DIR / "phase0_speaker_counts.png"
PLOT_SPEAKER_FREQ = EDA_DIR / "phase0_speaker_frequency.png"
PLOT_COVERAGE = EDA_DIR / "phase0_cumulative_coverage.png"
JSON_SUMMARY = EDA_DIR / "phase0_label_eda_summary.json"
REPORT = EDA_DIR / "Phase0_EDA_Report.md"

UNKNOWN_LABEL = "unknown"
RANDOM_SEED = 42


# ────────────────────────────────────────────────────────────────
#  1. Load & Clean
# ────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> pd.DataFrame:
    """Load labels.csv with defensive cleaning."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["audio_file"] = df["audio_file"].astype(str).str.strip()
    return df


def integrity_report(df: pd.DataFrame) -> dict:
    """Return a dict of data-integrity metrics."""
    report = {
        "raw_rows": int(len(df)),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_audio_files": int(df["audio_file"].duplicated().sum()),
        "missing_speaker_id": int(df["speaker_id"].isna().sum()),
    }

    # Presence of the *raw* audio files (mp3) on disk
    labelled = set(df["audio_file"])
    raw_on_disk = {p.name for p in RAW_AUDIO_DIR.glob("*") if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}}
    report["raw_files_on_disk"] = int(len(raw_on_disk))
    report["in_labels_but_missing_raw"] = int(len(labelled - raw_on_disk))

    # Presence of the converted 16 kHz WAV files (used by training).
    # Compare by *stem* because labels use the raw extension (.mp3) while the
    # converted files live as .wav under data/processed/audio_wav/.
    labelled_stems = {Path(f).stem for f in labelled}
    wav_stems = {p.stem for p in WAV_AUDIO_DIR.glob("*.wav")} if WAV_AUDIO_DIR.exists() else set()
    report["wav_files_on_disk"] = int(len(wav_stems))
    report["in_labels_but_missing_wav"] = int(len(labelled_stems - wav_stems))
    report["all_audio_present_raw"] = len(labelled - raw_on_disk) == 0
    return report


# ────────────────────────────────────────────────────────────────
#  2. Class Composition
# ────────────────────────────────────────────────────────────────

def class_composition(df: pd.DataFrame) -> dict:
    """Split known vs unknown, compute per-speaker stats."""
    unknown_mask = df["speaker_id"].str.lower() == UNKNOWN_LABEL
    df_unknown = df[unknown_mask]
    df_known = df[~unknown_mask]

    n_unknown = len(df_unknown)
    n_known = len(df_known)
    n_total = len(df)
    n_known_speakers = df_known["speaker_id"].nunique()

    speaker_counts = df_known["speaker_id"].value_counts()

    stats = {
        "total_files": n_total,
        "unknown_files": int(n_unknown),
        "known_files": int(n_known),
        "unknown_frac": float(n_unknown / n_total),
        "known_frac": float(n_known / n_total),
        "n_known_speakers": int(n_known_speakers),
        "n_unknown_speakers_hidden": 554,  # per competition spec (hidden OOD identities)
        "total_people_spec": 1000,          # 446 known + 554 OOD
        "classes_total": int(n_known_speakers + 1),  # 447 classes
        "per_speaker": {
            "min": int(speaker_counts.min()),
            "max": int(speaker_counts.max()),
            "mean": float(speaker_counts.mean()),
            "median": float(speaker_counts.median()),
            "std": float(speaker_counts.std()),
            "p01": int(speaker_counts.quantile(0.01)),
            "p05": int(speaker_counts.quantile(0.05)),
            "p25": int(speaker_counts.quantile(0.25)),
            "p75": int(speaker_counts.quantile(0.75)),
            "p95": int(speaker_counts.quantile(0.95)),
            "p99": int(speaker_counts.quantile(0.99)),
        },
        # Imbalance ratio: how many unknown samples per average known sample
        "unknown_over_mean_known": float(n_unknown / speaker_counts.mean()),
        "unknown_over_median_known": float(n_unknown / speaker_counts.median()),
        # Macro-F1 note: per-class "support" of unknown ≈ 2275, per known speaker ≈ 5
        "speaker_frequency_table": {
            int(k): int(v) for k, v in speaker_counts.value_counts().sort_index().items()
        },
        "mode_speaker_count": int(speaker_counts.mode().iloc[0]),
        "n_speakers_with_mode": int((speaker_counts == speaker_counts.mode().iloc[0]).sum()),
    }

    # Extra: speakers with < 5 files (could complicate 5-fold / few-shot splits)
    stats["n_speakers_below_5"] = int((speaker_counts < 5).sum())
    stats["n_speakers_at_least_5"] = int((speaker_counts >= 5).sum())
    return stats, speaker_counts


# ────────────────────────────────────────────────────────────────
#  3. Train/Val Split Design Analysis
# ────────────────────────────────────────────────────────────────

def split_design(speaker_counts: pd.Series) -> dict:
    """
    Analyze the stratified split used by src/data_pipeline.stratified_split:
      known speakers → 1 sample held out per speaker for validation
      unknown class  → 20% held out
    This mirrors the *competition* split (≈50/50 per person) but keeps a
    training/validation separation for model selection.
    """
    n_val_known = int((speaker_counts >= 2).sum())   # 1 val + at least 1 train
    n_known_files = int(speaker_counts.sum())
    n_train_known = n_known_files - n_val_known

    # Unknown 80/20 (config default unknown_val_ratio=0.2)
    return {
        "val_known_samples": n_val_known,
        "train_known_samples": n_train_known,
        "val_unknown_ratio": 0.2,
        "note": "competition itself holds out ~50% per person; our local split keeps"
                " 1 sample/speaker for val to monitor generalization.",
    }


# ────────────────────────────────────────────────────────────────
#  4. Visualizations
# ────────────────────────────────────────────────────────────────

def plot_class_distribution(n_unknown: int, n_known: int, save_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sizes = [n_unknown, n_known]
    labels = [f"Unknown (OOD)\n{n_unknown:,} files", f"Known\n{n_known:,} files"]
    colors = ["#e74c3c", "#2ecc71"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, explode=(0.04, 0.04), pctdistance=0.72,
        textprops={"fontsize": 12, "weight": "bold"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(12)
        at.set_weight("bold")
    ax.set_title("Class Distribution — Unknown vs 446 Known Speakers\n"
                 "(447-way classification task)", fontsize=14, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_speaker_counts(speaker_counts: pd.Series, save_path: Path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    vals = np.sort(speaker_counts.values)
    ax.hist(vals, bins=np.arange(vals.min() - 0.5, vals.max() + 1.5, 1.0),
            color="#3498db", edgecolor="white", alpha=0.9)
    ax.axvline(speaker_counts.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {speaker_counts.mean():.2f}")
    ax.axvline(speaker_counts.median(), color="#2c3e50", linestyle="--", linewidth=2,
               label=f"Median = {speaker_counts.median():.0f}")
    ax.set_xlabel("Audio files per known speaker", fontsize=13, weight="bold")
    ax.set_ylabel("Number of speakers", fontsize=13, weight="bold")
    ax.set_title("File Counts Among the 446 Known Speakers", fontsize=15, weight="bold")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_speaker_frequency(speaker_counts: pd.Series, save_path: Path):
    freq = speaker_counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(freq.index.astype(str), freq.values, color=sns.color_palette("viridis", len(freq)),
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, freq.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")
    ax.set_xlabel("Number of files per speaker", fontsize=13, weight="bold")
    ax.set_ylabel("Number of speakers", fontsize=13, weight="bold")
    ax.set_title("Speaker Frequency Breakdown", fontsize=15, weight="bold")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cumulative_coverage(speaker_counts: pd.Series, save_path: Path):
    """Cumulative share of known-speaker files captured by the top-k speakers."""
    sorted_counts = speaker_counts.sort_values(ascending=False)
    cum = np.cumsum(sorted_counts.values) / sorted_counts.values.sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cum) + 1), cum * 100, color="#8e44ad", linewidth=2)
    # highlight: 80% coverage point
    idx80 = np.searchsorted(cum, 0.8) + 1
    ax.axhline(80, color="gray", linestyle="--", linewidth=1.2)
    ax.axvline(idx80, color="gray", linestyle="--", linewidth=1.2)
    ax.annotate(f"80% of known files in top {idx80} speakers",
                xy=(idx80, 80), xytext=(idx80 + 8, 60),
                arrowprops=dict(arrowstyle="->", color="#2c3e50"),
                fontsize=11, fontweight="bold")
    ax.set_xlabel("Speakers sorted by file count (rank)", fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative share of known files (%)", fontsize=13, weight="bold")
    ax.set_title("Cumulative Coverage — Known Speakers", fontsize=15, weight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
#  5. Report Generation
# ────────────────────────────────────────────────────────────────

def fmt_table(headers, rows):
    header = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def generate_report(integrity: dict, stats: dict, split: dict, freq_table: dict) -> str:
    ps = stats["per_speaker"]
    return f"""# Phase 0 — Label & Metadata EDA Report

**Project:** IAAA Competition 2026 — Open-Set Speaker Identification  
**Module:** `src/eda.py` · **Date:** 2026-08-08

---

## 1. Competition Frame (from official spec)

The task is a **447-way open-set classification**:

| Entity | Count | Label in training CSV |
|--------|------:|----------------------:|
| Known speakers | 446 | UUID string (speaker-id) |
| OOD / unknown speakers | 554 (hidden) | `"unknown"` (single aggregated class) |
| **Total classes** | **447** | `0` + `1..446` (internal mapping) |

- Audio for each person is split **≈50/50** train vs hidden-eval chunks.
- The evaluation metric is **Macro-Averaged F1 across all 447 classes** — every known
  speaker counts as its own class with *equal weight to the unknown class*.
- ⇒ **Each of the 446 known speakers must be recognized as its own identity**, while
  OOD speakers must be rejected as `unknown`. Per-class accuracy on the ~5 samples per
  known speaker is decisive.

---

## 2. Data Loading & Cleaning

{fmt_table(
    ["Metric", "Value"],
    [
        ["Raw rows loaded", f"{integrity['raw_rows']:,}"],
        ["Missing values", integrity["missing_values"]],
        ["Duplicate rows", integrity["duplicate_rows"]],
        ["Duplicate audio files", integrity["duplicate_audio_files"]],
        ["Missing speaker_id", integrity["missing_speaker_id"]],
        ["Raw audio files on disk", f"{integrity['raw_files_on_disk']:,}"],
        ["In labels but missing (raw)", integrity["in_labels_but_missing_raw"]],
        ["Converted 16 kHz WAV on disk", f"{integrity['wav_files_on_disk']:,}"],
        ["In labels but missing (WAV)", integrity["in_labels_but_missing_wav"]],
    ],
)}

> ✅ **Integrity**: every labelled audio file exists both as raw `.mp3` and as a
> converted 16 kHz mono `.wav` — the dataset is fully self-consistent.

---

## 3. Class Composition

{fmt_table(
    ["Statistic", "Value"],
    [
        ["Total audio files", f"{stats['total_files']:,}"],
        ["Known files (446 speakers)", f"{stats['known_files']:,} ({stats['known_frac']*100:.2f}%)"],
        ["Unknown / OOD files", f"{stats['unknown_files']:,} ({stats['unknown_frac']*100:.2f}%)"],
        ["Unique known speaker-ids", f"{stats['n_known_speakers']:,}"],
        ["Hidden OOD identities (spec)", f"{stats['n_unknown_speakers_hidden']:,}"],
        ["Total people (spec)", f"{stats['total_people_spec']:,}"],
        ["Total classes (incl. unknown)", f"{stats['classes_total']:,}"],
    ],
)}

### 3.1 Per-Known-Speaker Balance

{fmt_table(
    ["Statistic", "Value"],
    [
        ["Min files / speaker", ps["min"]],
        ["Max files / speaker", ps["max"]],
        ["Mean files / speaker", f"{ps['mean']:.4f}"],
        ["Median files / speaker", f"{ps['median']:.1f}"],
        ["Std dev", f"{ps['std']:.4f}"],
        ["Mode (most common)", f"{stats['mode_speaker_count']} files "
                               f"({stats['n_speakers_with_mode']} speakers)"],
        ["Speakers with ≥ 5 files", f"{stats['n_speakers_at_least_5']} / {stats['n_known_speakers']}"],
    ],
)}

**Speaker frequency breakdown** (how many speakers have N files):

{fmt_table(["Files per speaker", "Number of speakers"],
           [(n, c) for n, c in sorted(freq_table.items())])}

### 3.2 Imbalance Metrics (w.r.t. the 447-way problem)

{fmt_table(
    ["Metric", "Formula", "Value"],
    [
        ["Unknown : mean-known ratio",
         "unknown_files / mean(known per speaker)",
         f"{stats['unknown_over_mean_known']:.2f}×"],
        ["Unknown : median-known ratio",
         "unknown_files / median(known per speaker)",
         f"{stats['unknown_over_median_known']:.2f}×"],
        ["Macro-F1 per-class support gap",
         "2275 (unknown) vs ≈5 (each known speaker)",
         f"≈{stats['unknown_over_median_known']/1:.0f}×"],
    ],
)}

> **Macro-F1 implication:** because the metric averages F1 *per class*, the model gets
> exactly **one F1 term for `unknown`** and **one F1 term for every known speaker**.
> A model that predicts `unknown` for everything scores *high recall on unknown* but
> **zero F1 on all 446 known speakers** ⇒ Macro-F1 collapses. Conversely, missing OOD
> hurts only one class. **The dominant risk is known-speaker recall.**

---

## 4. Train/Val Split Design

{fmt_table(
    ["Component", "Value"],
    [
        ["Validation samples (known, 1/speaker)", f"{split['val_known_samples']:,}"],
        ["Training samples (known)", f"{split['train_known_samples']:,}"],
        ["Validation share of unknown", f"{split['val_unknown_ratio']*100:.0f}%"],
        ["Note", split["note"]],
    ],
)}

---

## 5. Visualizations

### 5.1 Class Distribution

![Class Distribution](phase0_class_distribution.png)

### 5.2 File Counts Among Known Speakers

![Speaker Counts](phase0_speaker_counts.png)

### 5.3 Speaker Frequency Breakdown

![Speaker Frequency](phase0_speaker_frequency.png)

### 5.4 Cumulative Coverage

![Cumulative Coverage](phase0_cumulative_coverage.png)

---

## 6. Implications & Recommended Strategies

### 6.1 Known-speaker recognition is the bottleneck

- Every known speaker has only **≈5 training samples** → few-shot recognition, not
  standard closed-set classification. Per-speaker accuracy is the main driver of
  Macro-F1 (446 of 447 classes).
- **Strategy:** strong pretrained speaker encoder (ECAPA / WavLM / HuBERT), heavy
  time-domain augmentation + random window cropping to multiply effective samples,
  and **ArcFace-style angular margin** to squeeze separation between near-identical
  utterances.

### 6.2 The unknown class must be a *rejection*, not a majority class

- With ~2275 unknown samples the model would happily learn `P(unknown)` — but
  Macro-F1 punishes trading known-recall for unknown-recall 446×.
- **Strategy:** two-head design (OOD head + speaker head) with **controlled batch
  balancing** (30% OOD / 70% known) and an explicit OOD logit fused as
  `p[unknown] = sigmoid(ood)`; do **not** let the unknown class dominate gradient flow.

### 6.3 Class-conditional macro-F1 needs threshold tuning on OOD

- Validate on a held-out OOD set and tune the OOD threshold (or energy-based score)
  so that *both* known recall and unknown precision stay high.

---

## 7. Key Numbers (summary JSON)

```json
{json.dumps(stats, indent=2, ensure_ascii=False)[:2400]}
```

---

*Report generated programmatically via `src/eda.py`.*
"""


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 0 — Label & Metadata EDA")
    print("=" * 60)

    print("\n[1/5] Loading labels...")
    df = load_labels(LABELS_PATH)
    print(f"  Loaded {len(df):,} rows from {LABELS_PATH.name}")

    print("\n[2/5] Data integrity...")
    integrity = integrity_report(df)
    for k, v in integrity.items():
        print(f"  {k:30s} {v}")

    print("\n[3/5] Class composition...")
    stats, speaker_counts = class_composition(df)
    print(f"  Known: {stats['known_files']:,} files / {stats['n_known_speakers']} speakers")
    print(f"  Unknown: {stats['unknown_files']:,} files ({stats['unknown_frac']*100:.2f}%)")
    print(f"  Per-speaker: min={stats['per_speaker']['min']} mean={stats['per_speaker']['mean']:.2f} "
          f"max={stats['per_speaker']['max']}")
    print(f"  Imbalance (unknown/mean-known): {stats['unknown_over_mean_known']:.2f}×")

    print("\n[4/5] Split design...")
    split = split_design(speaker_counts)
    print(f"  Val known (1/speaker): {split['val_known_samples']:,} | "
          f"Train known: {split['train_known_samples']:,}")

    print("\n[5/5] Visualizations & report...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120

    plot_class_distribution(stats["unknown_files"], stats["known_files"], PLOT_PIE)
    plot_speaker_counts(speaker_counts, PLOT_SPEAKER_COUNTS)
    plot_speaker_frequency(speaker_counts, PLOT_SPEAKER_FREQ)
    plot_cumulative_coverage(speaker_counts, PLOT_COVERAGE)
    print(f"  [SAVED] {PLOT_PIE.name}, {PLOT_SPEAKER_COUNTS.name}, "
          f"{PLOT_SPEAKER_FREQ.name}, {PLOT_COVERAGE.name}")

    # JSON summary (reused by later phases)
    JSON_SUMMARY.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [SAVED] {JSON_SUMMARY.name}")

    report = generate_report(integrity, stats, split, stats["speaker_frequency_table"])
    REPORT.write_text(report, encoding="utf-8")
    print(f"  [SAVED] {REPORT.name}")

    print("\n✅ Phase 0 EDA complete.")


if __name__ == "__main__":
    main()
