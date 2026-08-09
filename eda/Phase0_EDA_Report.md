# Phase 0 — Label & Metadata EDA Report

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

| Metric | Value |
|---|---|
| Raw rows loaded | 4,529 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate audio files | 0 |
| Missing speaker_id | 0 |
| Raw audio files on disk | 4,529 |
| In labels but missing (raw) | 0 |
| Converted 16 kHz WAV on disk | 4,529 |
| In labels but missing (WAV) | 0 |

> ✅ **Integrity**: every labelled audio file exists both as raw `.mp3` and as a
> converted 16 kHz mono `.wav` — the dataset is fully self-consistent.

---

## 3. Class Composition

| Statistic | Value |
|---|---|
| Total audio files | 4,529 |
| Known files (446 speakers) | 2,254 (49.77%) |
| Unknown / OOD files | 2,275 (50.23%) |
| Unique known speaker-ids | 446 |
| Hidden OOD identities (spec) | 554 |
| Total people (spec) | 1,000 |
| Total classes (incl. unknown) | 447 |

### 3.1 Per-Known-Speaker Balance

| Statistic | Value |
|---|---|
| Min files / speaker | 5 |
| Max files / speaker | 20 |
| Mean files / speaker | 5.0538 |
| Median files / speaker | 5.0 |
| Std dev | 0.7416 |
| Mode (most common) | 5 files (439 speakers) |
| Speakers with ≥ 5 files | 446 / 446 |

**Speaker frequency breakdown** (how many speakers have N files):

| Files per speaker | Number of speakers |
|---|---|
| 5 | 439 |
| 6 | 5 |
| 9 | 1 |
| 20 | 1 |

### 3.2 Imbalance Metrics (w.r.t. the 447-way problem)

| Metric | Formula | Value |
|---|---|---|
| Unknown : mean-known ratio | unknown_files / mean(known per speaker) | 450.16× |
| Unknown : median-known ratio | unknown_files / median(known per speaker) | 455.00× |
| Macro-F1 per-class support gap | 2275 (unknown) vs ≈5 (each known speaker) | ≈455× |

> **Macro-F1 implication:** because the metric averages F1 *per class*, the model gets
> exactly **one F1 term for `unknown`** and **one F1 term for every known speaker**.
> A model that predicts `unknown` for everything scores *high recall on unknown* but
> **zero F1 on all 446 known speakers** ⇒ Macro-F1 collapses. Conversely, missing OOD
> hurts only one class. **The dominant risk is known-speaker recall.**

---

## 4. Train/Val Split Design

| Component | Value |
|---|---|
| Validation samples (known, 1/speaker) | 446 |
| Training samples (known) | 1,808 |
| Validation share of unknown | 20% |
| Note | competition itself holds out ~50% per person; our local split keeps 1 sample/speaker for val to monitor generalization. |

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
{
  "total_files": 4529,
  "unknown_files": 2275,
  "known_files": 2254,
  "unknown_frac": 0.5023183925811437,
  "known_frac": 0.49768160741885625,
  "n_known_speakers": 446,
  "n_unknown_speakers_hidden": 554,
  "total_people_spec": 1000,
  "classes_total": 447,
  "per_speaker": {
    "min": 5,
    "max": 20,
    "mean": 5.053811659192825,
    "median": 5.0,
    "std": 0.7415570018008985,
    "p01": 5,
    "p05": 5,
    "p25": 5,
    "p75": 5,
    "p95": 5,
    "p99": 6
  },
  "unknown_over_mean_known": 450.1552795031056,
  "unknown_over_median_known": 455.0,
  "speaker_frequency_table": {
    "5": 439,
    "6": 5,
    "9": 1,
    "20": 1
  },
  "mode_speaker_count": 5,
  "n_speakers_with_mode": 439,
  "n_speakers_below_5": 0,
  "n_speakers_at_least_5": 446
}
```

---

*Report generated programmatically via `src/eda.py`.*
