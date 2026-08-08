# IAAA Competition 2026 — Open-Set Speaker Identification

**A complete, reproducible solution for the IAAA 2026 Speaker Identification Challenge:**
classify short audio chunks into **447 classes** (446 known speaker-ids + one aggregated `unknown`/OOD class),
optimized for **Macro-Averaged F1 across all 447 classes**.

> This document is the living technical report of the project: competition framing,
> dataset facts, full architecture, every design decision, training strategy, OOD
> handling, inference/submission contract, MLOps setup, and current results.

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)
2. [Key Dataset Facts (measured)](#2-key-dataset-facts-measured)
3. [Exploratory Data Analysis (EDA suite)](#3-exploratory-data-analysis-eda-suite)
4. [Solution Architecture](#4-solution-architecture)
   - 4.1 [End-to-End Data Flow](#41-end-to-end-data-flow)
   - 4.2 [Encoder Backbones](#42-encoder-backbones)
   - 4.3 [Pooling](#43-pooling)
   - 4.4 [Two-Head Design (OOD + Speaker)](#44-two-head-design-ood--speaker)
   - 4.5 [Probability Fusion (the 447-way output)](#45-probability-fusion-the-447-way-output)
5. [Data Pipeline](#5-data-pipeline)
   - 5.1 [MP3 → WAV Conversion](#51-mp3--wav-conversion)
   - 5.2 [Label Cleaning & Class Mapping](#52-label-cleaning--class-mapping)
   - 5.3 [Train/Validation Split](#53-trainvalidation-split)
   - 5.4 [Corrupted/Short File Filtering](#54-corruptedshort-file-filtering)
   - 5.5 [Augmentation](#55-augmentation)
   - 5.6 [Balanced Batch Sampling](#56-balanced-batch-sampling)
   - 5.7 [Window Cropping Strategy](#57-window-cropping-strategy)
6. [Loss Function](#6-loss-function)
7. [Training Strategy](#7-training-strategy)
8. [OOD (Unknown Speaker) Handling](#8-ood-unknown-speaker-handling)
   - 8.1 [Learned OOD Head](#81-learned-ood-head)
   - 8.2 [FAISS Cosine-Distance Detector](#82-faiss-cosine-distance-detector)
   - 8.3 [Score Fusion & Threshold Tuning](#83-score-fusion--threshold-tuning)
9. [Ensembling](#9-ensembling)
10. [Inference & Submission](#10-inference--submission)
11. [Configuration Reference](#11-configuration-reference)
12. [MLOps & Reproducibility](#12-mlops--reproducibility)
13. [Project Structure](#13-project-structure)
14. [Getting Started](#14-getting-started)
15. [Current Results & Roadmap](#15-current-results--roadmap)
16. [Rule Compliance Notes](#16-rule-compliance-notes)

---

## 1. Competition Overview

Source: [`Competition-Guide/iaaa-competition-2026-speaker-identification.pdf`](Competition-Guide/iaaa-competition-2026-speaker-identification.pdf)

### 1.1 Task

Build a model that, given a **short audio chunk**, outputs a probability distribution over
**447 classes**:

| Class | Meaning |
|-------|---------|
| `unknown` | any of the **554 hidden OOD speakers** (aggregated into one class) |
| `[speaker-id]` (×446) | the 446 **known speakers**, each identified by a UUID |

During evaluation the organizers:
1. take your probabilities,
2. apply **argmax** → predicted label,
3. compare against ground truth (`unknown` or a specific speaker-id),
4. compute **Macro-Averaged F1 across all 447 classes**.

### 1.2 Dataset construction (from the spec)

- **1,000 people** with different accents and recording conditions.
- Per person, roughly **50% of the audio is training**, the rest is held out for evaluation.
- **446 known** speakers are labelled with their unique UUID; **554 unknown/OOD** speakers
  are collapsed into the single `unknown` class.
- For the training split only, the organizers ship a metadata CSV: `audio_file,speaker_id`.

### 1.3 Why Macro-F1 makes this hard

Macro-F1 treats every class equally:

- `unknown` contributes **one** F1 term,
- each of the 446 known speakers contributes **one** F1 term.

A degenerate model that predicts `unknown` for everything achieves perfect unknown recall
but **zero F1 on all 446 known classes** → Macro-F1 collapses. The metric therefore
rewards models that keep **known-speaker recall high** (recognizing each specific voice)
*and* reject OOD. Since every known speaker has only ~5 training files, this is a
**few-shot open-set recognition** problem.

### 1.4 Rules that shape the design

| Rule | Implication for our solution |
|------|------------------------------|
| Public pretrained audio models allowed | We use pretrained **ECAPA-TDNN (VoxCeleb)**, **WavLM**, **HuBERT** backbones |
| External speech datasets / self-supervision allowed | Feasible future pretraining on VoxCeleb/LibriSpeech |
| **No** labeled data containing eval speakers | We never add outside labels that could overlap hidden identities |
| **No** access to hidden eval labels | Model selection uses our own local hold-out |
| Ensembling allowed | Multi-encoder ensemble implemented (`src/ensemble.py`) |
| Reproducibility may be required | Config-driven pipeline, fixed seeds, ZenML + MLflow, DVC |

---

## 2. Key Dataset Facts (measured)

Measured directly from `data/raw/` with the project EDA suite (regenerate anytime —
see [Section 3](#3-exploratory-data-analysis-eda-suite)).

### 2.1 Label space

| Metric | Value |
|--------|------:|
| Training rows in `labels.csv` | **4,529** |
| Known speaker files | 2,254 (49.8%) |
| `unknown` (OOD) files | 2,275 (50.2%) |
| Unique known speakers | **446** |
| Total classes (incl. `unknown`) | **447** |
| Hidden OOD identities | 554 (spec) |
| Total people | 1,000 (spec) |

### 2.2 Per-known-speaker balance

| Statistic | Value |
|-----------|------:|
| Min / Mean / Max files per speaker | 5 / 5.05 / 20 |
| Median | 5.0 |
| Speakers with exactly 5 files | 439 / 446 |
| Imbalance (`unknown` : mean known) | ≈ **450×** |

> Every known speaker is a *few-shot class*: ~5 files. Per-file quality and
> augmentation dominate known-speaker accuracy.

### 2.3 Duration & audio format

| Metric | Value |
|--------|------:|
| Duration — mean / median / max | 58.2 s / 59.6 s / 2m39s |
| Files > 30 s | **90.3%** |
| Files > 60 s | 2,217 (49%) |
| Files < 1 s (corrupted/empty) | **70 (1.55%)** |
| Sample rate / channels / bit depth | **16 kHz / mono / PCM_16** (all 4,529) |

### 2.4 Empirical "ceiling" from frozen ECAPA embeddings

| Metric | Value |
|--------|------:|
| Same-speaker mean cosine | 0.588 |
| Cross-speaker mean cosine | 0.123 |
| Verification d′ | 2.72 |
| Argmax-centroid recognition (known files) | **95.5%** |
| Top-5 centroid recognition | 96.9% |
| Centroid-distance OOD detection AUC | **0.953** |

These numbers are measured *before any training* — the frozen encoder already separates
voices and OOD very well, which the learned heads are designed to exploit.

---

## 3. Exploratory Data Analysis (EDA suite)

The EDA lives in [`eda/`](eda/) — four reproducible phases, each a Python script under
[`src/`](src/) that regenerates a Markdown report + PNG charts + a JSON summary.

| Phase | Script | Report | Focus |
|-------|--------|--------|-------|
| 0 | [`src/eda.py`](src/eda.py) | [`eda/Phase0_EDA_Report.md`](eda/Phase0_EDA_Report.md) | Label space, integrity, class imbalance, split design, Macro-F1 implications |
| 1 | [`src/eda_advanced.py`](src/eda_advanced.py) | [`eda/Phase1_Advanced_EDA_Report.md`](eda/Phase1_Advanced_EDA_Report.md) | Durations, buckets, corrupted-file detection, chunking strategy |
| 2 | [`src/eda_acoustic.py`](src/eda_acoustic.py) | [`eda/Phase2_Acoustic_EDA_Report.md`](eda/Phase2_Acoustic_EDA_Report.md) | Signal/spectral features, known-vs-unknown confounder tests, MFCC-PCA |
| 3 | [`src/eda_embeddings.py`](src/eda_embeddings.py) | [`eda/Phase3_Embedding_EDA_Report.md`](eda/Phase3_Embedding_EDA_Report.md) | ECAPA embedding space, **unbiased (LOO)** recognition ceiling, OOD separability, Macro-F1 simulation, t-SNE |

Run any/all of them:

```bash
uv run --no-sync python -m src.eda          # labels
uv run --no-sync python -m src.eda_advanced   # durations (~3s, header-only soundfile)
uv run --no-sync python -m src.eda_acoustic   # ~8 min (stratified 600-file sample)
uv run --no-sync python -m src.eda_embeddings # GPU, several minutes — multi-window ECAPA
```

> **Note on the Phase-3 numbers in §2.4:** the original Phase-3 report (95.5% ceiling,
> AUC 0.953) was **in-sample** (a file contributed ~20% of its own centroid) and embedded
> the 70 corrupted files. The rewritten `src/eda_embeddings.py` is **unbiased**: it drops
> corrupted/MD5-duplicate files, uses multi-window TTA, evaluates with **leave-one-out**
> centroids, and adds a **Macro-F1 simulation** — the only number that decides whether the
> centroid route alone can hit the 0.97 target.

### 3.1 Headline EDA findings that drive the design

1. **Known speakers are nearly perfectly balanced (5 files each)** → no intra-known
   re-weighting needed; the imbalance is *only* between `unknown` and each known class.
2. **90% of files exceed 30 s** → random 8 s window cropping multiplies the effective
   training set ~7× per epoch (a 60 s file yields ~7 independent windows).
3. **70 files are corrupted (< 1 s)** → dropped by `min_valid_duration: 1.0`.
4. **No low-level acoustic confounders** between known and unknown (all Cohen's |d| ≤ 0.20,
   only `zcr` marginally > 0.2) → OOD cannot (and need not) be gamed with loudness/pitch
   thresholds; it must be solved in the embedding manifold.
5. **Frozen ECAPA already reaches 95.5% centroid-recognition and 0.953 OOD-AUC** →
   the learned heads start from a strong manifold and the FAISS detector is a
   principled OOD baseline.

---

## 4. Solution Architecture

The model is a **config-driven, two-headed open-set classifier**:

```
Raw audio (16 kHz mono, 8 s window)
        │
        ▼
┌─────────────────────┐      ┌─────────────────────────────────────────────┐
│  ENCODER            │      │  ECAPA-TDNN 192-d   │ WavLM 768-d │ HuBERT  │
│  (pretrained,       │ ───► │  (frozen)           │ (FE frozen) │ 1024-d  │
│  interchangeable)   │      └─────────────────────────────────────────────┘
└─────────────────────┘
        │  frame-level features  (B, T_frames, D)   [ECAPA: (B, 1, 192)]
        ▼
┌─────────────────────┐
│  POOLING            │  StatisticalPooling / AttentiveStatisticalPooling / Identity
└─────────────────────┘
        │  utterance embedding  (B, pooled_dim)
        ▼
        ├─────────────────────────────┐
        ▼                             ▼
┌─────────────────┐          ┌──────────────────────┐
│   OOD HEAD      │          │   SPEAKER HEAD       │
│  MLP → 1 logit  │          │  ArcFace / Linear    │
│  → P(unknown)   │          │  → logits (446)      │
└─────────────────┘          └──────────────────────┘
        │                             │
        └─────────── FUSION ──────────┘
        p[0]     = σ(ood_logit)
        p[1..446] = (1 − p[0]) · softmax(speaker_logits)
        ▼
  447-way probability vector  (sums to 1)
```

### 4.1 End-to-end data flow

| Stage | Module | What happens |
|-------|--------|--------------|
| Data prep | [`src/data_pipeline.py`](src/data_pipeline.py) | Load labels → clean → class-map → stratified split → filter short files |
| Dataset | `SpeakerDataset` in [`src/data_pipeline.py`](src/data_pipeline.py) | Load WAV, resample, augment, crop/pad to `8 s`, optional MixUp |
| Batching | `get_dataloaders` in [`src/data_pipeline.py`](src/data_pipeline.py) | Balanced OOD/known batch sampler |
| Model | [`src/model_factory.py`](src/model_factory.py) → [`src/model.py`](src/model.py) | Compose encoder + pooling + heads from config |
| Loss | [`src/train.py`](src/train.py) `TwoPartLoss` | Weighted BCE(OOD) + Focal(speaker, masked) |
| Training | [`src/train.py`](src/train.py) `train()` | AMP, separate grad-clip, cosine LR, early stopping, checkpoints |
| OOD | [`src/ood_detector.py`](src/ood_detector.py) | FAISS cosine detector + score fusion |
| Submission | [`submission/inference.py`](submission/inference.py) | Competition CLI → 447-column probability CSV |

### 4.2 Encoder backbones

All encoders implement the same interface (`BaseEncoder` in
[`src/encoders.py`](src/encoders.py)): `forward(waveforms) → (hidden_states, lengths)`,
plus `output_dim`, `freeze()`, `unfreeze()` — so backbones are **hot-swappable via config**
(`model.encoder_type`).

| Encoder | Source / dims | Notes |
|---------|---------------|-------|
| **ECAPA-TDNN** (default) | `speechbrain/spkrec-ecapa-voxceleb`, 192-d | SOTA speaker embedding trained on VoxCeleb1+2 (0.80% EER). Ships its own attentive statistical pooling → use `pooling_type: identity`. Frozen. |
| WavLM | `microsoft/wavlm-base-plus`, 768-d (94M) | HuBERT-style with gated relative position bias; strong speaker transfer. Feature extractor frozen, transformer trainable. |
| HuBERT | `facebook/hubert-large-ls960-ft`, 1024-d (317M) | Large SSL speech model; needs fine-tuning for best speaker transfer. |

Engineering notes baked into `src/encoders.py`:

- **SpeechBrain lazy-module patch** (`_patch_speechbrain_lazy_modules`, [`src/encoders.py:194`](src/encoders.py#L194)):
  SpeechBrain ≥1.0 lazily exports optional modules (e.g. `k2_fsa`). Any attribute access —
  including the `inspect.stack()` walks performed by `librosa`'s lazy loader — triggers an
  `ImportError` for the missing optional dependency. We force-load or stub those modules so
  imports never break.
- **ECAPA forward path** bypasses SpeechBrain's `encode_batch()` (which does implicit
  `wav.to(device)` and `.float()` under AMP) and instead calls
  `compute_features → mean_var_norm → embedding_model` directly, wrapped in
  `torch.no_grad()` — avoiding CPU/CUDA mismatches, AMP half-precision crashes, and
  BatchNorm corruption. The encoder is always kept in `eval()` mode (overridden
  `train()`/`to()`).
- Frozen WavLM/HuBERT **feature extractors** (CNN stem) stay frozen; transformer layers
  remain trainable — a memory/quality trade-off.

### 4.3 Pooling

[`src/pooling.py`](src/pooling.py) turns frame-level features into one utterance embedding:

| Pooling | Output | When to use |
|---------|--------|-------------|
| `StatisticalPooling` | `(B, 2D)` — concat(mean, std) | WavLM/HuBERT frame features |
| `AttentiveStatisticalPooling` | `(B, 2D)` — attention-weighted mean/std | When frame importance varies |
| `IdentityPooling` | `(B, D)` — pass-through | **ECAPA** (already has internal pooling) — the current default |

All pooling layers accept an optional frame `mask` so padded frames do not corrupt the
statistics.

### 4.4 Two-head design (OOD + Speaker)

[`src/model.py`](src/model.py) (`TwoHeadedSpeakerModel`) and [`src/heads.py`](src/heads.py):

**OOD Head** — `OODHead` ([`src/heads.py:22`](src/heads.py#L22)):
```
LayerNorm(D) → Linear(256) → ReLU → Dropout(0.4) → Linear(1)
```
A single logit; `σ(logit) = P(unknown)`. Trained with BCE where target = 1 iff label == 0.

**Speaker Head** — selected by `model.speaker_head_type`:

- **`arcface` (default)** — `ArcFaceHead` ([`src/heads.py:101`](src/heads.py#L101)):
  - projects the pooled embedding to `embedding_dim=192`,
  - L2-normalizes embeddings *and* class weight rows,
  - computes cosine logits and, during training, applies the angular margin:
    `s·cos(θ + m)` with `margin = 0.4`, `scale = 30.0`,
  - on inference (no labels) returns scaled cosine logits.
  - Angular margin pushes embeddings of different speakers apart in cosine space —
    exactly what the OOD/verification machinery needs. Config: `margin 0.3 → 0.4`,
    `scale 15 → 30` (standard for speaker recognition).
- **`linear`** — `LinearSpeakerHead`: `LayerNorm → Linear(D → 446)`.

**Label remapping for ArcFace** ([`src/model.py:86`](src/model.py#L86)): the dataset labels
are `0 = unknown`, `1..446 = known`. ArcFace needs `0..445` for known classes, so labels
are remapped `known k → k−1`. The `unknown` samples map to class `0` — a *harmless
collision* because `TwoPartLoss` masks unknown samples (`ignore_index=-100`) so their
gradient is always zero and ArcFace weight row 0 is only trained by speaker #1.

### 4.5 Probability fusion (the 447-way output)

`TwoHeadedSpeakerModel.predict_proba` ([`src/model.py:108`](src/model.py#L108)):

```python
p_unknown   = sigmoid(ood_logit)                      # P(OOD)
p_known     = softmax(speaker_logits)                 # P(known_i | known)
probs = concat([p_unknown, (1 - p_unknown) * p_known])  # (B, 1 + 446)
probs /= probs.sum(dim=1)                              # numerical safety
```

This guarantees a **valid probability distribution over all 447 classes**, and — crucially —
the OOD probability is *calibrated against the speaker head*: when the model is confident
about a known identity, `p_unknown` is suppressed; when the voice is off-manifold,
`p_unknown` dominates. This is the open-set mechanism.

---

## 5. Data Pipeline

### 5.1 MP3 → WAV conversion

`scripts/convert_mp3_to_wav.py` converts all 4,529 MP3s to **16 kHz mono PCM-16 WAV**
in `data/processed/audio_wav/`, and writes `data/processed/audio_wav_labels.csv`
(labels with `.wav` names). Rationale: on Windows, `librosa/audioread` relies on `mpg123`
for MP3 decoding which fails intermittently during long runs; WAV is read natively by
`soundfile` with zero external dependencies. The ZenML pipeline also contains this as its
step 0 (`convert_audio` in [`src/pipelines/steps.py`](src/pipelines/steps.py)), skipping
if >4000 WAVs already exist.

### 5.2 Label cleaning & class mapping

`prepare_clean_split` / `create_class_mapping` ([`src/data_pipeline.py:44`](src/data_pipeline.py#L44)):

- strip whitespace, drop duplicate rows, drop rows with missing `speaker_id`/`audio_file`;
- build the mapping **`unknown → 0`, known UUIDs → 1..446** (sorted lexicographically for
  determinism);
- persist the labelled CSV (`data/processed/cleaned_labels.csv`) and produce
  `train_df`, `val_df`, `class_map`.

### 5.3 Train/validation split (leak-free)

`stratified_split` ([`src/data_pipeline.py`](src/data_pipeline.py)) is now **leakage-aware**:

- **Known speakers:** exactly **1 sample held out per speaker** for validation, the rest
  trains — with only ~5 files per speaker this is the strictest setting that keeps at least
  one train sample per class.
- **`unknown` class:** 80/20 train/val split (`unknown_val_ratio = 0.2`).
- **No leakage:** every **MD5-duplicate group** (9 groups, 69 files — some with conflicting
  labels) goes **entirely to train**; val is strictly duplicate-free. A known speaker whose
  files are all duplicates is excluded from val (with a warning) to keep it clean.
- Corrupted files (see 5.4) are dropped **before** the split.
- A machine-readable **`data/processed/split_report.json`** records corrupted files
  (known/unknown), duplicate groups (incl. conflicting-label groups), and per-known-speaker
  train/val counts + usable seconds.

This is deliberately **not** the competition's 50/50 per-person split; it exists only to
monitor generalization during model selection while keeping every speaker in the training set.

### 5.4 Corrupted/short file filtering

`min_valid_duration: 1.0`. `prepare_clean_split` scans all files with **`soundfile.info`
(header-only C call)** and drops anything shorter than 1 s or unreadable → **70 files
(22 known + 48 unknown)**. Dropped files are reported in `split_report.json`
(previously also in `checkpoints/corrupted_files.json`).

> Why `soundfile` and not `librosa.get_duration`? `librosa`'s lazy submodule loader calls
> `inspect.stack()`, which trips over SpeechBrain's `LazyModule` (missing `k2`) and raises
> `ImportError` during long runs — a real bug we hit and fixed in `src/encoders.py` with
> `_patch_speechbrain_lazy_modules()`.

### 5.5 Augmentation

`AudioAugmentation` ([`src/data_pipeline.py:150`](src/data_pipeline.py#L150)) wraps
`audiomentations`, applied per training window (train only):

| Transform | Range / params | p |
|-----------|----------------|---|
| AddGaussianNoise | 0.001–0.015 amp | 0.5 |
| PitchShift | **±1 semitone** (was ±4) | **0.3** (was 0.5) |
| TimeStretch | 0.8×–1.25× | **0.2** (was 0.3) |
| Gain | ±6 dB | 0.3 |
| PolarityInversion | — | 0.5 |
| Shift (fractional, rollover) | ±10% | 0.3 |

> **Why gentler pitch shift?** A **frozen** ECAPA encoder cannot adapt to ±4 semitones —
> the old setting caused the inverted train/val gap (train spk acc 0.292 vs val 0.558).

Optional **MixUp** (`mixup_alpha > 0`): with 50% probability a sample is mixed with a
random other sample, `λ ~ Beta(α, α)`; the original label is kept — this acts as OOD
regularization by feeding the model acoustically-ambiguous audio.

### 5.6 Balanced batch sampling (OOD starvation fix)

The old per-class `WeightedRandomSampler` was the **root cause of the OOD-head collapse**:
the unknown class (a single 2,275-sample super-class) got ~1/447 of every batch
(≈0.07 samples per batch of 32) → the BCE target was almost always 0 → the head collapsed
to "always known" (val OOD acc 0.495, threshold sweep F1=0.0).

`make_balanced_batch_sampler` ([`src/data_pipeline.py:652`](src/data_pipeline.py#L652)) fixes
this and is used by **both** `get_dataloaders` and the ZenML `train_model` step:

- `ood_batch_ratio = 0.50` → every batch = `round(0.5·B)` unknown + the rest known, drawn
  from separate OOD/known pools → the OOD head always sees ~50% positives, matching the
  ~50/50 eval mix.
- `ood_pos_weight = 1.0` on the BCE adds an extra safety net (config `training.ood_pos_weight`).

### 5.7 Multi-window strategy (use the whole file)

Files are ~58 s but a single 8 s window wastes ~85% of the signal. `SpeakerDataset` now
returns a **stack of windows `(W, 1, T)`** ([`src/data_pipeline.py`](src/data_pipeline.py)):

- **Training:** `num_train_windows = 3` **random crops** per file per epoch (each
  independently augmented) → implicit augmentation across the whole file.
- **Validation / inference:** **sliding windows** with `eval_hop_ratio = 0.5` (hop = 4 s),
  capped at `max_eval_windows = 8`, **evenly spread** across the full file (last window
  repeated to keep a constant count so batching stays simple).
- The training/eval loops average the **logits** across windows
  (`forward_multi_window` in `src/train.py`); the submission CLI averages the
  **probabilities** (see Section 10).

---

## 6. Loss Function

`TwoPartLoss` ([`src/train.py:159`](src/train.py#L159)):

```
L = w_ood · BCEWithLogits(σ(ood_logit), y_ood)   +   w_spk · CE_focal(spk_logits, y_spk_masked)
```

- **OOD term:** binary BCE (with `ood_pos_weight = 1.0`), target `1` if label==0 else `0`.
- **Speaker term:** only on known samples (`unknown` masked via `ignore_index=-100`);
  uses **Focal Loss** (γ=2.0) to down-weight easy classes and focus on hard known-speaker
  samples, with **label smoothing 0.1** for calibration robustness.
- **Weights:** `ood_loss_weight = 0.3`, `speaker_loss_weight = 0.7` — known-speaker
  recognition is weighted ~2.3× the OOD term because the metric is dominated by the 446
  known classes.
- `FocalLoss` ([`src/train.py:58`](src/train.py#L58)) implements smoothing-compatible focal
  weighting (`(1−p_t)^γ · CE`) with safe handling of the ignore index.

---

## 7. Training Strategy

`src/train.py` `train()` and the ZenML `train_model` step
([`src/pipelines/steps.py`](src/pipelines/steps.py)):

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW with **two param groups** — encoder `encoder_lr = 1e-5`, heads `lr = 1e-4`, `weight_decay = 1e-5` |
| LR schedule | `CosineAnnealingLR` (`T_max = epochs`) — ZenML variant: 3-epoch linear warmup → `CosineAnnealingWarmRestarts` |
| Mixed precision | AMP (`torch.cuda.amp`) with `GradScaler` |
| Gradient clipping | **separate** — `ood_grad_norm = 1.0` for OOD-head params, `max_grad_norm = 5.0` elsewhere (prevents OOD head overfitting) |
| Early stopping | `patience = 10` **on validation Macro-F1** (the competition metric) |
| Checkpoint selection | **best by val Macro-F1** (was: val loss) → `best_model.pt` + `latest_model.pt` every epoch; full state (model, optimizer, scheduler, config, class_map) + `val_macro_f1` |
| Eval forward | **without labels** → no ArcFace margin at eval time (honest metrics) |
| Fine-tuning | `model.encoder_config.ecapa.unfreeze_last_n_blocks: 2` — only the last 2 SE-Res2Blocks trainable (5.4 M encoder params) with `encoder_lr`; `forward()` keeps the graph only when partially unfrozen |
| Hardware profiles | `local` (GTX 1660 Ti 6 GB, batch 8, workers 0) / `vastai` (3090/4090, batch 32) / `vastai_3060` (batch 16) — **batch size is per-profile, everything else is shared** |
| Epochs | 50 (config default) |

Metrics logged per epoch: train/val **loss**, **OOD accuracy**, **known-speaker accuracy**,
and **val Macro-F1** (via `src/metrics.evaluate_macro_f1`, the exact 447-class metric the
organizers score).

---

## 8. OOD (Unknown Speaker) Handling

Three complementary layers, in increasing order of sophistication.

### 8.1 Learned OOD head

The `OODHead` is trained end-to-end with BCE. It learns "does this pooled embedding live
near the known-speaker manifold?" — its sigmoid is fused directly into the 447-way output
([Section 4.5](#45-probability-fusion-the-447-way-output)).

### 8.2 FAISS cosine-distance detector

[`src/ood_detector.py`](src/ood_detector.py) — `FAISSOODDetector`:

- Enrolls all known training embeddings into a **FAISS `IndexFlatIP` with automatic L2
  normalization** (= exact cosine similarity, 100% recall) plus `IndexIDMap` for speaker ids.
- At inference, for each test embedding: `OOD score = 1 − mean(cos sim to k=5 nearest
  known embeddings)`. Higher score → more likely unknown.
- Phase 3 EDA shows this mechanism alone reaches **0.953 OOD-AUC**.

### 8.3 Score fusion & threshold tuning

- `combine_ood_scores(head_score, faiss_score, alpha=0.5)` blends the learned head with the
  FAISS score.
- The evaluation step ([`src/pipelines/steps.py`](src/pipelines/steps.py)) sweeps the OOD
  threshold on the validation set, reports the **competition Macro-F1** (argmax and
  threshold-gated), and **persists `ood_threshold` into `best_model.pt`** so inference can
  reuse it. If the OOD head collapsed (all-F1 = 0) it falls back to the median val
  P(unknown) so a sane threshold is always stored.
- Design note: because the fused output is a full probability vector, the submission path
  simply argmaxes the 447 vector — the threshold tuning happens at the *model-selection*
  stage, not at submission (`--apply-ood-threshold` is off by default).

---

## 9. Ensembling

[`src/ensemble.py`](src/ensemble.py) — `EnsembleModel`:

- Runs `N` trained `TwoHeadedSpeakerModel`s and fuses their 447-way probability vectors.
- **Average fusion** (`fusion_method="average"`): arithmetic mean of probabilities — the
  current default (`config.model.fusion.ensemble_method: none` until models are trained).
- **Learned fusion** (`LearnedFusion`): MLP over concatenated probabilities
  (`N·447 → 512 → ReLU → Dropout → 447 → Softmax`), trained on the validation set while
  base models stay frozen.

Planned ensemble: ECAPA + WavLM + HuBERT models, average-fused — diversity in encoder
biases typically improves Macro-F1 on both known and OOD classes.

---

## 10. Inference & Submission

### 10.1 Competition CLI contract

```bash
uv run --no-sync python -m submission.inference \
    --data-dir <test-set-folder> \
    --predictions-file-path <output.csv>
```

([`submission/inference.py`](submission/inference.py))

Key options (see `--help`):

- `--checkpoint-path` — **repeatable**: pass several checkpoints for an **ensemble**
  (per-window probabilities are averaged across models).
- `--apply-ood-threshold` — hard-gate `P(unknown) > saved threshold` to class 0
  (**default OFF**; the competition scores plain argmax over the 447-way output).
- `--fuse-centroid` — blend with the step-6 centroid classifier (`--centroid-alpha`,
  default 0.5; needs the embedding cache from `python -m src.centroid_baseline`).
- `--id-style stem|filename` — id column format (default `stem`).
- `--max-eval-windows` — override the config value.

### 10.2 Output format

A CSV with columns `id, 0, 1, …, 446`:

- `id` = audio file stem (default),
- columns `0..446` = probabilities for `unknown` (0) and the 446 known speakers,
- **each row sums to exactly 1.0**,
- a sidecar `<output>.class_map.json` maps column index → speaker UUID for reproducibility.

### 10.3 Inference path

1. load `checkpoints/best_model.pt` (+ config) and rebuild the model via
   `create_model_from_config`,
2. restore the exact `class_map` from the checkpoint (so column `i` ↔ speaker-id `i`),
3. for each audio file: load → resample 16 kHz mono → **TTA windowing** (8 s windows,
   50% overlap, up to `max_eval_windows`) → per-window `predict_proba` → **average
   probabilities** → renormalize,
4. optional centroid fusion / OOD-threshold gate,
5. fallback to a **uniform distribution** if a file cannot be decoded (robustness under
   the organizer's controlled environment).

> **Consistency note:** training and inference both use the same window config
> (`audio.duration_seconds`, `eval_hop_ratio`, `max_eval_windows` in
> `configs/default_config.yaml`) — keep these in sync when experimenting.

---

## 11. Configuration Reference

Full file: [`configs/default_config.yaml`](configs/default_config.yaml)

```yaml
hardware:
  mode: local                     # local | vastai | vastai_3060
  profiles:                       # batch_size is per-GPU; everything else shared
    local:   {device: cuda, batch_size: 8,  num_workers: 0, mixed_precision: true}   # GTX 1660 Ti 6 GB
    vastai:  {device: cuda, batch_size: 32, num_workers: 4, mixed_precision: true}   # RTX 3090/4090
    vastai_3060: {device: cuda, batch_size: 16, num_workers: 4, mixed_precision: true}

audio:
  sample_rate: 16000
  duration_seconds: 8.0           # window length (train/eval/inference)
  min_valid_duration: 1.0         # drop corrupted / near-empty files
  ood_batch_ratio: 0.50           # target OOD fraction per training batch
  num_train_windows: 3            # random crops per file in train (multi-window TTA)
  eval_hop_ratio: 0.5             # sliding-window overlap for eval/inference
  max_eval_windows: 8             # cap on eval windows per file
  n_mels: 80                      # future front-end params
  n_fft: 400
  hop_length: 160

model:
  encoder_type: ecapa             # ecapa | wavlm | hubert
  competition_num_known: 446      # 446 known → 447-way output
  encoder_config:
    wavlm:   {base_model: microsoft/wavlm-base-plus, freeze_feature_extractor: true}
    ecapa:   {source: speechbrain/spkrec-ecapa-voxceleb, freeze_encoder: false,
              unfreeze_last_n_blocks: 2}   # fine-tune last 2 SE-Res2Blocks only
    hubert:  {base_model: facebook/hubert-large-ls960-ft, freeze_feature_extractor: true}
  pooling_type: identity          # identity (ECAPA) | statistical | attentive
  speaker_head_type: arcface      # arcface | linear
  speaker_head_config:
    arcface: {embedding_dim: 192, margin: 0.4, scale: 30.0}
  ood_head_config:
    hidden_dim: 256
  fusion: {ensemble_method: none}

training:
  epochs: 50
  learning_rate: 0.0001           # head LR
  encoder_lr: 1.0e-05             # LR for unfrozen encoder blocks (param group)
  weight_decay: 1.0e-05
  warmup_steps: 500
  max_grad_norm: 5.0
  ood_grad_norm: 1.0              # tighter clip for the OOD head
  ood_pos_weight: 1.0             # BCE pos_weight for the OOD head
  ood_loss_weight: 0.3            # OOD : speaker = 3 : 7
  speaker_loss_weight: 0.7
  early_stopping_patience: 10     # on val Macro-F1
  label_smoothing: 0.1

data:
  labels_path: data/processed/audio_wav_labels.csv
  audio_dir:   data/processed/audio_wav
  processed_labels: data/processed/cleaned_labels.csv

logging:
  log_dir: logs
  checkpoint_dir: checkpoints

mlops:
  enabled: true
  experiment_name: speaker-identification
  tracking.uri: https://dagshub.com/<owner>/Speaker-identification.mlflow   # via .env
```

---

## 12. MLOps & Reproducibility

### 12.1 ZenML pipeline

[`src/pipelines/run_pipeline.py`](src/pipelines/run_pipeline.py) orchestrates:

```
convert_audio → prepare_data → build_model → train_model → evaluate_model
```

```bash
python -m src.pipelines.run_pipeline --run all                 # full pipeline
python -m src.pipelines.run_pipeline --run train --no-mlflow   # partial, no tracking
```

### 12.2 Experiment tracking (MLflow / DagsHub)

- `src/mlflow_helper.py` provides a standalone `MLflowTracker` that bypasses ZenML's
  (version-fragile) experiment-tracker integration.
- Runs log: config snapshot, code snapshot (zip of `src/`), params, per-epoch metrics,
  best checkpoint artifact, final summary JSON.
- Credentials via `.env` (`DAGSHUB_REPO_OWNER`, `DAGSHUB_USER_TOKEN`); the pipeline
  resolves `${...}` placeholders from env vars and supports token-based auth on headless
  servers.

### 12.3 Remote GPU (Vast.ai) & UI

- `src/deploy/deploy.py` rents a GPU on Vast.ai, pushes `setup_vast.sh`, and runs the
  pipeline remotely.
- `src/deploy/deploy_app.py` is a Streamlit UI for launching deployments.

### 12.4 Data versioning (DVC)

`data/` is tracked with **DVC** (`.dvc/` cache present) — audio is never committed to git;
`dvc pull` restores it from the DagsHub S3 remote.

### 12.5 Determinism

Fixed random seeds in data splits (42), balanced sampler (42), and the EDA suite;
`class_map` is persisted inside every checkpoint so submission columns are unambiguous.

---

## 13. Project Structure

```
.
├── Competition-Guide/            # official challenge PDF
├── configs/
│   └── default_config.yaml       # single source of truth for hyperparameters
├── data/
│   ├── raw/                      # 4,529 × .mp3 + labels.csv (DVC-tracked)
│   └── processed/                # audio_wav/ (16 kHz mono), label CSVs, split_report.json, embedding caches
├── eda/                          # EDA reports, charts, JSON summaries, .npy
├── checkpoints/                  # best/latest/init models + corrupted_files.json
├── scripts/
│   ├── convert_mp3_to_wav.py
│   └── clean_corrupted.py
├── src/
│   ├── data_pipeline.py          # labels, leak-free split, augmentation, dataset, loaders, balanced sampler
│   ├── encoders.py               # ECAPA / WavLM / HuBERT + factory + partial unfreeze
│   ├── pooling.py                # statistical / attentive / identity
│   ├── heads.py                  # OODHead, LinearSpeakerHead, ArcFaceHead
│   ├── model.py                  # TwoHeadedSpeakerModel + fusion + multi-window predict_proba
│   ├── model_factory.py          # config → model
│   ├── train.py                  # losses, train/val epochs, multi-window forward, training loop
│   ├── metrics.py                # competition Macro-F1 (447-class) + fusion + temperature calibration
│   ├── ood_detector.py           # FAISS cosine OOD detector
│   ├── centroid_baseline.py      # step-6: centroid classifier + embedding cache + fusion
│   ├── ensemble.py               # average / learned fusion
│   ├── ensemble_calibrate.py     # step-9: per-model + ensemble Macro-F1 + temperature report
│   ├── mlflow_helper.py          # standalone MLflow tracker
│   ├── eda*.py                   # 5-phase EDA suite (Phase 3 is unbiased / LOO)
│   ├── pipelines/                # ZenML steps + orchestrator
│   └── deploy/                   # Vast.ai + Streamlit
├── submission/
│   ├── inference.py              # competition CLI → 447-column CSV (TTA + ensemble + fusion)
│   └── __init__.py
├── tests/smoke/                  # pipeline smoke tests
└── setup_vast.sh / setup_project.py / .env.example
```

---

## 14. Getting Started

```bash
# 0. Environment (CRITICAL — see IMPLEMENTATION_PLAN.md section 0)
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#    → must print 2.11.0+cu126 True. ALWAYS use `uv run --no-sync` (a plain
#    `uv sync`/`uv run` can overwrite the CUDA torch build with the CPU wheel).

# 1. Data (DVC remote) or local copy
dvc pull                       # restores data/raw + data/processed

# 2. (Optional) Unbiased EDA — embeddings + LOO centroid + Macro-F1 simulation
uv run --no-sync python -m src.eda_embeddings          # GPU, several minutes
uv run --no-sync python -m src.centroid_baseline       # builds embedding cache + centroid baseline

# 3. Train (local GTX 1660 Ti profile)
uv run --no-sync python -m src.train
#    or: uv run --no-sync python -m src.pipelines.run_pipeline --run train

# 4. Submission CSV on a test folder
uv run --no-sync python -m submission.inference \
    --data-dir <test-set-folder> \
    --predictions-file-path predictions.csv

# 5. (After training ≥2 models) ensemble + temperature report
uv run --no-sync python -m src.ensemble_calibrate \
    --checkpoints checkpoints/best_seed1.pt checkpoints/best_seed2.pt
```

Environment: copy `.env.example` → `.env` and fill DagsHub/Vast credentials.

**Hardware note (MLOps):** `batch_size` is per-GPU in `hardware.profiles` (8 / 16 / 32 for
the 1660 Ti / 3060 / 3090). The window parameters, sampler ratio, and loss weights are
shared — only the profile `mode` needs to change between GPUs.

---

## 15. Current Results & Roadmap

### 15.1 Reproduced run (recorded in `mlruns/`)

| Metric | Value |
|--------|------:|
| Best validation loss | 0.30 |
| Best epoch | 3 |
| Final validation OOD accuracy | 0.85 |

> The recorded run used an early snapshot of the pipeline. **Note:** the competition
> metric is **Macro-F1 across 447 classes**, not OOD binary accuracy — the roadmap below
> is about converting these partial numbers into a strong Macro-F1.

### 15.2 Empirical ceilings to aim at (Phase 3 EDA)

- Frozen ECAPA centroid recognition: **95.5%** known-speaker accuracy.
- Centroid-distance OOD detection: **0.953 AUC**.

### 15.3 Roadmap (ordered by expected Macro-F1 impact)

1. **Full training run with current config** (ECAPA + ArcFace m=0.4/s=30, 8 s windows,
   balanced batches, focal γ=2) and a proper **447-class Macro-F1** evaluation loop
   (per-class precision/recall, unknown threshold tuning).
2. **OOD threshold / calibration tuning** on validation to trade known-recall ↔
   unknown-recall where it maximizes Macro-F1 (the known classes are 446/447 of the F1).
3. **Multi-encoder ensemble** (ECAPA + WavLM + HuBERT), average fusion first, then learned.
4. **Fine-tuning** WavLM/HuBERT (unfreeze transformer) with larger batches on the
   `vastai` profile; compare vs frozen-ECAPA.
5. **External-data pretraining** (allowed by the rules) — e.g. fine-tune on VoxCeleb then
   transfer, while keeping the held-out split free of any eval-speaker labels.
6. **TTA + longer windows** sweeps (8 s → 10 s) and per-class thresholding if validation
   shows it is stable.

---

## 16. Rule Compliance Notes

- ✅ **Pretrained models allowed** — ECAPA/WavLM/HuBERT are public pretrained weights; we
  comply.
- ✅ **No eval-speaker labels** — the model is trained exclusively on `data/raw/labels.csv`;
  any future external data will be filtered to never contain the hidden OOD identities
  (which are unlabelled anyway).
- ✅ **No hidden eval labels** — all model selection uses the local stratified hold-out.
- ✅ **Ensembling allowed** — implemented and planned.
- ✅ **Reproducibility** — single config, fixed seeds, checkpointed class maps, DVC data,
  MLflow runs, smoke tests, and this report.

---

*Generated from the actual project source; EDA numbers are measured on `data/raw/` and
regenerable via the EDA suite.*
