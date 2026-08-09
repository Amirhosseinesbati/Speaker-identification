# Phase 1 — Duration & Audio Integrity EDA Report

**Project:** IAAA Competition 2026 — Open-Set Speaker Identification  
**Module:** `src/eda_advanced.py` · **Date:** 2026-08-08

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total audio files | 4,529 |
| Corrupted / unreadable | 70 |
| Files < 1s (suspicious) | 70 (1.55%) |
| Files > 30 s | 4,091 (90.3%) |
| Files > 60 s | 2,217 |
| Median duration | 59.6s |

> **Headline:** ~90% of the corpus is long-form audio (>30 s). This is a **huge
> advantage** for open-set speaker ID: every long file can be cut into many
> independent training windows, multiplying the effective training set.

---

## 2. Global Duration Statistics

| Statistic | Value |
|-----------|-------|
| Min (valid files) | 0.00s |
| Max | 2m 39.4s |
| **Mean** | **58.2s** |
| **Median** | **59.6s** |
| Std dev | 21.3s |

### Percentile distribution (valid files only)

| Percentile | Duration |
|-----------|----------|
| 1% | 0.02s |
| 5% | 6.3s |
| 10% | 32.4s |
| 25% | 50.1s |
| 50% (median) | 59.6s |
| 75% | 1m 10.9s |
| 90% | 1m 21.7s |
| 95% | 1m 28.5s |
| 99% | 1m 40.2s |

---

## 3. Known vs Unknown Duration Comparison

| Statistic | Known (n=2,254) | Unknown (n=2,275) |
|-----------|-------------------------|---------------------------|
| Min | 0.00s | 0.00s |
| Max | 2m 1.9s | 2m 39.4s |
| **Mean** | **58.9s** | **57.5s** |
| **Median** | **1m 0.1s** | **59.1s** |
| Std dev | 20.4s | 22.1s |

> **Confounder check:** known and unknown files have essentially identical duration
> distributions (Δmedian ≈ 1 s). Duration is **not** a usable cue for OOD detection —
> the model must rely on *voice characteristics*, exactly as the challenge intends.

---

## 4. Duration Bucket Breakdown

| Bucket | Total | % | Known | Unknown |
|--------|------:|---:|------:|--------:|
['| <1s | 70 | 1.5% | 22 | 48 |', '| 1-2s | 45 | 1.0% | 20 | 25 |', '| 2-3s | 25 | 0.6% | 11 | 14 |', '| 3-5s | 55 | 1.2% | 26 | 29 |', '| 5-10s | 108 | 2.4% | 58 | 50 |', '| 10-30s | 135 | 3.0% | 56 | 79 |', '| 30-60s | 1,874 | 41.4% | 932 | 942 |', '| >60s | 2,217 | 49.0% | 1,129 | 1,088 |']

---

## 5. Corrupted / Near-Zero Files

**70 files** (1.55%) have duration < 1 s and are
treated as corrupted / empty. Sample:

| Audio File | Duration | Speaker ID | Class |
|------------|----------|------------|-------|
| 05006e09-6c16-4640-8dad-108a558eae85.mp3 | 0.00s | unknown | Unknown |
| 0f2bebd9-4651-4e53-96cd-42dd2acbace1.mp3 | 0.00s | unknown | Unknown |
| 72be17b0-6947-41f7-ba65-9009bcec8d22.mp3 | 0.00s | unknown | Unknown |
| a557ed64-77f7-4942-9a6e-ae20291103a0.mp3 | 0.00s | unknown | Unknown |
| 55efd02e-abd3-444e-80e8-874506eded35.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| 1974c5f7-0cbd-4dfe-8279-30d23313a91f.mp3 | 0.00s | 9eff2f0a-75cf-4589-8049-1c555eb624fd | Known |
| 01ebadf4-5eda-4e3f-a736-ff181c60d40b.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| 0841e792-edc5-4aaf-8ec6-8053094a6f44.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| 5edbc0d1-59ce-4afe-bc5e-f11ff206ceba.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| 1ca9b207-db27-424d-a611-5c2995a2e9d1.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| 286a8f8f-8781-459e-8b7b-dcfa6a91c5dd.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| 52083e6d-3de3-4043-b2d7-9cfc4afd420b.mp3 | 0.00s | 990aa42f-b730-40f8-b720-4f2040b84f73 | Known |
| ... | (and 58 more) | | |

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
{
  "total_files": 4529,
  "corrupted_count": 70,
  "short_count": 70,
  "short_pct": 1.545595054095827,
  "over_30s_count": 4091,
  "over_30s_pct": 90.32899094722897,
  "over_60s_count": 2217,
  "all": {
    "count": 4529,
    "min": 6.25e-05,
    "max": 159.4026875,
    "mean": 58.17783985151247,
    "median": 59.5626875,
    "std": 21.31902167525801,
    "p1": 0.0239325000000001,
    "p5": 6.3146875,
    "p10": 32.39255,
    "p25": 50.0906875,
    "p50": 59.5626875,
    "p75": 70.912,
    "p90": 81.68106250000002,
    "p95": 88.4906875,
    "p99": 100.21889500000005
  },
  "known": {
    "count": 2254,
    "min": 6.25e-05,
    "max": 121.856,
    "mean": 58.88611856699201,
    "median": 60.0746875,
    "std": 20.435575079263845,
    "p1": 1.024,
    "p5": 7.7653125,
    "p10": 38.34028125,
    "p25": 50.6026875,
    "p50": 60.0746875,
    "p75": 71.31734375,
    "p90": 81.664,
    "p95": 87.7226875,
    "p99": 98.07273687499988
  },
  "unknown": {
    "count": 2275,
    "min": 6.25e-05,
    "max": 159.4026875,
    "mean": 57.476099093406596,
    "median": 59.0506875,
    "std": 22.13726878395411,
    "p1": 6.25e-05,
    "p5": 4.753075000000001,
    "p10": 20.8895875,
    "p25": 49.62134375,
    "p50": 59.0506875,
    "p75": 70.4,
    "p90": 81.88587500000001,
    "p95": 89.088,
    "p99": 103.29769124999997
  }
}
```

---

*Report generated programmatically via `src/eda_advanced.py`.*
