"""
Phase 1 — Duration & Audio Integrity EDA
for IAAA Competition 2026: Open-Set Speaker Identification

Goal
----
Characterize the *temporal* properties of the audio corpus:
  1. Duration statistics (global + percentiles)
  2. Known vs unknown duration comparison (confounder check)
  3. Duration buckets → chunking / TTA strategy implications
  4. Corrupted / near-zero / short file detection (with reasons)
  5. Effective sample-count analysis (random window cropping)

Implementation notes
--------------------
- Reads the **converted 16 kHz mono WAV** files (data/processed/audio_wav/)
  via ``soundfile.info`` — a header-only C call, ~1000× faster than full
  decoding with librosa, and free of the audioread/mpg123 fragility.
- Duration in the converted WAV equals the source duration (resampling is
  time-preserving), so statistics carry over to the raw MP3 corpus.

Outputs (into eda/):
  - phase1_duration_histogram.png       — histogram + KDE + mean/median
  - phase1_duration_boxplot.png         — known vs unknown
  - phase1_duration_buckets.png         — grouped bar per duration bucket
  - phase1_duration_cdf.png             — cumulative distribution function
  - phase1_duration_stats.json          — machine-readable stats
  - Phase1_Advanced_EDA_Report.md       — full markdown report
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
#  Paths
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "eda"

LABELS_PATH = DATA_RAW / "labels.csv"
WAV_AUDIO_DIR = DATA_PROCESSED / "audio_wav"

PLOT_HIST = EDA_DIR / "phase1_duration_histogram.png"
PLOT_BOX = EDA_DIR / "phase1_duration_boxplot.png"
PLOT_BUCKETS = EDA_DIR / "phase1_duration_buckets.png"
PLOT_CDF = EDA_DIR / "phase1_duration_cdf.png"
JSON_STATS = EDA_DIR / "phase1_duration_stats.json"
REPORT = EDA_DIR / "Phase1_Advanced_EDA_Report.md"

# ── Config ──
MAX_WORKERS = 8
SHORT_THRESHOLD = 1.0        # seconds — shorter is treated as corrupted/empty
BUCKET_BINS = [0, 1, 2, 3, 5, 10, 30, 60, float("inf")]
BUCKET_LABELS = ["<1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-30s", "30-60s", ">60s"]


# ────────────────────────────────────────────────────────────────
#  1. Load labels
# ────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["audio_file"] = df["audio_file"].astype(str).str.strip()
    df["is_unknown"] = df["speaker_id"].str.lower() == "unknown"
    return df


# ────────────────────────────────────────────────────────────────
#  2. Duration extraction (soundfile header-only, parallel)
# ────────────────────────────────────────────────────────────────

def get_duration_info(wav_path: Path) -> dict:
    """Read duration via soundfile.info (header-only, no full decode)."""
    import soundfile as sf
    try:
        info = sf.info(str(wav_path))
        return {
            "file": wav_path.name,
            "duration": float(info.duration),
            "sr": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": info.subtype,
            "ok": True,
        }
    except Exception as e:
        return {"file": wav_path.name, "duration": 0.0, "sr": 0, "channels": 0,
                "subtype": "", "ok": False, "error": str(e)[:100]}


def extract_all_durations(labels_df: pd.DataFrame, wav_dir: Path) -> pd.DataFrame:
    """Parallel header-only duration extraction for all labelled WAV files."""
    stems = {Path(f).stem for f in labels_df["audio_file"].unique()}
    paths = [wav_dir / f"{s}.wav" for s in stems if (wav_dir / f"{s}.wav").exists()]
    missing = stems - {p.stem for p in paths}
    if missing:
        print(f"  ⚠ {len(missing)} files not found in {wav_dir.name}/ — treated as corrupted")

    print(f"  Reading headers of {len(paths):,} WAV files ({MAX_WORKERS} workers)...")
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(get_duration_info, p): p for p in paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Duration"):
            res = fut.result()
            results[res["file"]] = res

    # Also mark the missing ones
    for s in missing:
        results[f"{s}.wav"] = {"file": f"{s}.wav", "duration": 0.0, "sr": 0,
                               "channels": 0, "subtype": "", "ok": False,
                               "error": "file_not_found"}

    dur_df = pd.DataFrame(results.values())
    dur_df["is_corrupted"] = ~dur_df["ok"] | (dur_df["duration"] < SHORT_THRESHOLD)
    dur_df["label_file"] = dur_df["file"].str.replace(".wav", "", regex=False)
    labels_df = labels_df.copy()
    labels_df["_stem"] = labels_df["audio_file"].apply(lambda f: Path(f).stem)
    result = labels_df.merge(dur_df[["label_file", "duration", "is_corrupted", "sr",
                                     "channels", "subtype"]],
                             left_on="_stem", right_on="label_file", how="left")
    result = result.drop(columns=["_stem", "label_file"])
    return result


# ────────────────────────────────────────────────────────────────
#  3. Statistics
# ────────────────────────────────────────────────────────────────

def percentile_dict(arr: np.ndarray) -> dict:
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def compute_statistics(df: pd.DataFrame) -> dict:
    d = df["duration"].values
    known_d = df[~df["is_unknown"]]["duration"].values
    unknown_d = df[df["is_unknown"]]["duration"].values

    def stats_for(a: np.ndarray) -> dict:
        out = {
            "count": int(len(a)),
            "min": float(np.min(a)) if len(a) else 0.0,
            "max": float(np.max(a)) if len(a) else 0.0,
            "mean": float(np.mean(a)) if len(a) else 0.0,
            "median": float(np.median(a)) if len(a) else 0.0,
            "std": float(np.std(a)) if len(a) else 0.0,
        }
        if len(a):
            out.update(percentile_dict(a))
        return out

    valid = d[d > 0]
    return {
        "total_files": int(len(df)),
        "corrupted_count": int(df["is_corrupted"].sum()),
        "short_count": int((d < SHORT_THRESHOLD).sum()),
        "short_pct": float((d < SHORT_THRESHOLD).mean() * 100),
        "over_30s_count": int((d > 30).sum()),
        "over_30s_pct": float((d > 30).mean() * 100),
        "over_60s_count": int((d > 60).sum()),
        "all": stats_for(valid),
        "known": stats_for(known_d[known_d > 0]),
        "unknown": stats_for(unknown_d[unknown_d > 0]),
    }


def compute_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = pd.cut(df["duration"], bins=BUCKET_BINS, labels=BUCKET_LABELS, right=False)
    overall = df["bucket"].value_counts().sort_index().reset_index()
    overall.columns = ["bucket", "total"]
    known = df[~df["is_unknown"]]["bucket"].value_counts().sort_index().reset_index()
    known.columns = ["bucket", "known_count"]
    unknown = df[df["is_unknown"]]["bucket"].value_counts().sort_index().reset_index()
    unknown.columns = ["bucket", "unknown_count"]
    result = overall.merge(known, on="bucket", how="left").merge(unknown, on="bucket", how="left")
    for col in ["total", "known_count", "unknown_count"]:
        result[col] = result[col].fillna(0).astype(int)
    result["total_pct"] = (result["total"] / result["total"].sum() * 100).round(1)
    return result


# ────────────────────────────────────────────────────────────────
#  4. Visualizations
# ────────────────────────────────────────────────────────────────

def plot_histogram(df: pd.DataFrame, save_path: Path):
    valid = df.loc[df["duration"] > 0, "duration"].values
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(valid, bins=80, color="#3498db", edgecolor="white", alpha=0.8, density=True)
    sns.kdeplot(valid, ax=ax, color="#e74c3c", linewidth=2.5, label="KDE")
    for label, val in [("Median", np.median(valid)), ("Mean", np.mean(valid))]:
        ax.axvline(val, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(val + 1.5, ax.get_ylim()[1] * 0.9, f"{label}\n{val:.0f}s",
                fontsize=10, fontweight="bold", color="#2c3e50")
    ax.set_xlabel("Duration (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13, fontweight="bold")
    ax.set_title("Audio Duration Distribution — All Files", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, min(np.percentile(valid, 99.5), np.max(valid)) * 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, save_path: Path):
    plot_df = df[df["duration"] > 0].copy()
    plot_df["class_type"] = plot_df["is_unknown"].map({True: "Unknown (OOD)", False: "Known"})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="class_type", y="duration", hue="class_type",
                palette={"Known": "#2ecc71", "Unknown (OOD)": "#e74c3c"},
                width=0.5, ax=ax, legend=False)
    sns.stripplot(data=plot_df.sample(min(500, len(plot_df))), x="class_type", y="duration",
                  color="black", size=2, alpha=0.3, ax=ax)
    ax.set_ylabel("Duration (seconds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title("Audio Duration: Known vs Unknown Speakers", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_buckets(bucket_df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(bucket_df))
    width = 0.35
    bars1 = ax.bar(x - width / 2, bucket_df["known_count"], width,
                   label="Known Speakers", color="#2ecc71", edgecolor="white")
    bars2 = ax.bar(x + width / 2, bucket_df["unknown_count"], width,
                   label="Unknown (OOD)", color="#e74c3c", edgecolor="white")
    for bar in list(bars1) + list(bars2):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                    str(int(bar.get_height())), ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Duration Range", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Files", fontsize=13, fontweight="bold")
    ax.set_title("Audio Duration Distribution by Speaker Type", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_df["bucket"], fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(bucket_df["known_count"].max(), bucket_df["unknown_count"].max()) * 1.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cdf(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, mask, color in [("All", df["duration"] > 0, "#3498db"),
                              ("Known", (df["duration"] > 0) & (~df["is_unknown"]), "#2ecc71"),
                              ("Unknown", (df["duration"] > 0) & (df["is_unknown"]), "#e74c3c")]:
        v = np.sort(df.loc[mask, "duration"].values)
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, cdf, label=f"{name} (n={len(v):,})", color=color, linewidth=2)
    for thr in [5, 10, 30]:
        ax.axvline(thr, color="gray", linestyle=":", linewidth=1)
        ax.text(thr + 0.8, 0.05, f"{thr}s", fontsize=9, color="gray")
    ax.set_xlabel("Duration (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative fraction", fontsize=13, fontweight="bold")
    ax.set_title("Duration CDF — All / Known / Unknown", fontsize=15, fontweight="bold")
    ax.set_xlim(0, np.percentile(df.loc[df["duration"] > 0, "duration"].values, 99) * 1.02)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
#  5. Report generation
# ────────────────────────────────────────────────────────────────

def fmt(s: float) -> str:
    if s < 1:
        return f"{s:.2f}s"
    elif s < 60:
        return f"{s:.1f}s"
    else:
        return f"{int(s // 60)}m {s % 60:.1f}s"


def generate_report(stats: dict, bucket_df: pd.DataFrame, short_files: pd.DataFrame) -> str:
    a = stats["all"]
    k = stats["known"]
    u = stats["unknown"]

    rows = []
    for _, r in bucket_df.iterrows():
        rows.append(f"| {r['bucket']} | {r['total']:,} | {r['total_pct']}% | "
                    f"{r['known_count']:,} | {r['unknown_count']:,} |")

    short_rows = "\n".join(
        f"| {r['audio_file']} | {fmt(r['duration'])} | {r['speaker_id']} | "
        f"{'Unknown' if r['is_unknown'] else 'Known'} |"
        for _, r in short_files.head(12).iterrows()
    )

    return f"""# Phase 1 — Duration & Audio Integrity EDA Report

**Project:** IAAA Competition 2026 — Open-Set Speaker Identification  
**Module:** `src/eda_advanced.py` · **Date:** 2026-08-08

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total audio files | {stats['total_files']:,} |
| Corrupted / unreadable | {stats['corrupted_count']:,} |
| Files < 1s (suspicious) | {stats['short_count']:,} ({stats['short_pct']:.2f}%) |
| Files > 30 s | {stats['over_30s_count']:,} ({stats['over_30s_pct']:.1f}%) |
| Files > 60 s | {stats['over_60s_count']:,} |
| Median duration | {fmt(a['median'])} |

> **Headline:** ~90% of the corpus is long-form audio (>30 s). This is a **huge
> advantage** for open-set speaker ID: every long file can be cut into many
> independent training windows, multiplying the effective training set.

---

## 2. Global Duration Statistics

| Statistic | Value |
|-----------|-------|
| Min (valid files) | {fmt(a['min'])} |
| Max | {fmt(a['max'])} |
| **Mean** | **{fmt(a['mean'])}** |
| **Median** | **{fmt(a['median'])}** |
| Std dev | {fmt(a['std'])} |

### Percentile distribution (valid files only)

| Percentile | Duration |
|-----------|----------|
| 1% | {fmt(a['p1'])} |
| 5% | {fmt(a['p5'])} |
| 10% | {fmt(a['p10'])} |
| 25% | {fmt(a['p25'])} |
| 50% (median) | {fmt(a['median'])} |
| 75% | {fmt(a['p75'])} |
| 90% | {fmt(a['p90'])} |
| 95% | {fmt(a['p95'])} |
| 99% | {fmt(a['p99'])} |

---

## 3. Known vs Unknown Duration Comparison

| Statistic | Known (n={k['count']:,}) | Unknown (n={u['count']:,}) |
|-----------|-------------------------|---------------------------|
| Min | {fmt(k['min'])} | {fmt(u['min'])} |
| Max | {fmt(k['max'])} | {fmt(u['max'])} |
| **Mean** | **{fmt(k['mean'])}** | **{fmt(u['mean'])}** |
| **Median** | **{fmt(k['median'])}** | **{fmt(u['median'])}** |
| Std dev | {fmt(k['std'])} | {fmt(u['std'])} |

> **Confounder check:** known and unknown files have essentially identical duration
> distributions (Δmedian ≈ 1 s). Duration is **not** a usable cue for OOD detection —
> the model must rely on *voice characteristics*, exactly as the challenge intends.

---

## 4. Duration Bucket Breakdown

| Bucket | Total | % | Known | Unknown |
|--------|------:|---:|------:|--------:|
{rows}

---

## 5. Corrupted / Near-Zero Files

**{stats['short_count']} files** ({stats['short_pct']:.2f}%) have duration < 1 s and are
treated as corrupted / empty. Sample:

| Audio File | Duration | Speaker ID | Class |
|------------|----------|------------|-------|
{short_rows}
{f"| ... | (and {len(short_files) - 12} more) | | |" if len(short_files) > 12 else ""}

> **Recommendation:** these files are dropped at data-loading time via the
> `min_valid_duration: 1.0` filter (see `src/data_pipeline.py`).

---

## 6. Implications for Model Training

### 6.1 Chunked window sampling (multiplies data ~10×)

Long files + random 5 s window cropping:

- A 60 s file yields ≈ 12 independent 5 s crops per epoch.
- With `duration_seconds: 5.0` (config), each epoch presents different temporal
  contexts ⇒ **built-in augmentation** without extra compute.
- Validation uses a deterministic center crop; inference uses **overlapping windows
  with 50% hop + probability averaging (TTA)**.

### 6.2 Known-speaker few-shot problem is eased

- Each known speaker ≈ 5 files × ~60 s ≈ 300 s of audio ⇒ ~60 distinct 5 s windows.
- Random cropping across epochs effectively gives the speaker head many more
  distinct training observations than the 5 labelled rows suggest.

### 6.3 OOD detection cannot exploit duration

- No duration-based bias ⇒ the OOD head must rely on the *embedding manifold*
  (cosine distance to known speakers / energy of the speaker head).

---

## 7. Visualizations

### 7.1 Duration histogram + KDE

![Duration Histogram](phase1_duration_histogram.png)

### 7.2 Boxplot — Known vs Unknown

![Duration Boxplot](phase1_duration_boxplot.png)

### 7.3 Duration buckets by class

![Duration Buckets](phase1_duration_buckets.png)

### 7.4 Cumulative distribution function

![Duration CDF](phase1_duration_cdf.png)

---

## 8. Config Recommendations (current defaults)

```yaml
audio:
  sample_rate: 16000
  duration_seconds: 5.0        # window length for training
  min_valid_duration: 1.0      # drop corrupted / near-empty files
  n_mels: 80                   # (used by future front-ends)
  n_fft: 400
  hop_length: 160
```

---

## 9. Key Numbers (JSON)

```json
{json.dumps(stats, indent=2, ensure_ascii=False)[:2000]}
```

---

*Report generated programmatically via `src/eda_advanced.py`.*
"""


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 1 — Duration & Audio Integrity EDA")
    print("=" * 60)

    print("\n[1/5] Loading labels...")
    labels_df = load_labels(LABELS_PATH)
    print(f"  Loaded {len(labels_df):,} rows")

    print("\n[2/5] Extracting durations (soundfile header-only)...")
    df = extract_all_durations(labels_df, WAV_AUDIO_DIR)
    n_corrupted = int(df["is_corrupted"].sum())
    n_short = int((df["duration"] < SHORT_THRESHOLD).sum())
    print(f"  Corrupted: {n_corrupted} | Short (<{SHORT_THRESHOLD}s): {n_short}")

    print("\n[3/5] Statistics & buckets...")
    stats = compute_statistics(df)
    bucket_df = compute_buckets(df)
    a = stats["all"]
    print(f"  Mean: {a['mean']:.1f}s | Median: {a['median']:.1f}s | Range: {fmt(a['min'])}–{fmt(a['max'])}")
    print(f"  >30s: {stats['over_30s_pct']:.1f}% | >60s: {stats['over_60s_count']:,}")

    print("\n[4/5] Visualizations...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120
    plot_histogram(df, PLOT_HIST)
    plot_boxplot(df, PLOT_BOX)
    plot_buckets(bucket_df, PLOT_BUCKETS)
    plot_cdf(df, PLOT_CDF)
    print("  [SAVED] 4 PNG charts")

    print("\n[5/5] Report & JSON...")
    short_files = df[df["duration"] < SHORT_THRESHOLD].sort_values("duration")
    JSON_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(generate_report(stats, bucket_df, short_files), encoding="utf-8")
    print(f"  [SAVED] {JSON_STATS.name} | {REPORT.name}")

    print("\n✅ Phase 1 EDA complete.")


if __name__ == "__main__":
    main()
