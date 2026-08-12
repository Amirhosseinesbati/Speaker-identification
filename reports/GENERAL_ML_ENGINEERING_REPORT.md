# گزارش جامع مهندسی ML — بهبود مدل Speaker Identification

**پروژه:** IAAA Competition 2026 — Open-Set Speaker Identification  
**شاخه:** `feature/advanced-speaker-id`  
**تاریخ گزارش:** ۲۰۲۶-۰۸-۱۲  
**هدف مسابقه:** Macro-F1 ≥ 0.97 روی ۴۴۷ کلاس (۴۴۶ speaker شناخته‌شده + ۱ کلاس unknown)  
**بهترین امتیاز واقعی فعلی:** **Macro-F1 ≈ 0.9202** (centroid baseline، LOO unbiased)  
**وضعیت فعلی:** کد کامل است، اما هیچ مدل موفقی آموزش داده نشده. تنها run آموزشی با collapsed OOD head شکست خورد. رفع شده ولی دوباره اجرا نشده.

---

> **نحوه استفاده از این گزارش:** هر بخش شامل تحلیل وضعیت موجود، مشکلات شناسایی‌شده، و راهکارهای پیشنهادی با اولویت‌بندی است. جداول انتهای گزارش خلاصه تمام hyperparameter ها و بازه‌های پیشنهادی برای tuning را دارند.
> 
> **مخفف‌های پرکاربرد:** OOD = Out-Of-Distribution (گوینده ناشناس)، TTA = Test-Time Augmentation، LOO = Leave-One-Out، Macro-F1 = معیار نهایی مسابقه


---

## فهرست

1. [خلاصه اجرایی](#1-خلاصه-اجرایی)
2. [تحلیل داده](#2-تحلیل-داده)
3. [تحلیل معماری فعلی](#3-تحلیل-معماری-فعلی)
4. [مشکلات و ناهماهنگی‌های شناسایی‌شده](#4-مشکلات-و-ناهماهنگی‌های-شناسایی‌شده)
5. [راهکارهای بهبود](#5-راهکارهای-بهبود)
   - [A. تنظیمات و Hyperparameter Tuning](#a-تنظیمات-و-hyperparameter-tuning)
   - [B. Regularization و Anti-Overfitting (Few-Shot)](#b-regularization-و-anti-overfitting-few-shot)
   - [C. Augmentation و Data Engineering](#c-augmentation-و-data-engineering)
   - [D. معماری و Heads](#d-معماری-و-heads)
   - [E. Ensemble و Calibration](#e-ensemble-و-calibration)
   - [F. زیرساخت و MLOps](#f-زیرساخت-و-mlops)
   - [G. تکنیک‌های خاص Few-Shot و Open-Set](#g-تکنیک‌های-خاص-few-shot-و-open-set)
   - [H. بهینه‌سازی Inference و Submission](#h-بهینه‌سازی-inference-و-submission)
6. [نقشه راه اولویت‌بندی‌شده](#6-نقشه-راه-اولویت‌بندی‌شده)
7. [جداول مرجع](#7-جداول-مرجع)


---

## 1. خلاصه اجرایی

### ۱.۱ وضعیت فعلی — اعداد واقعی

| متریک | مقدار | توضیح |
|---|---|---|
| **بهترین Macro-F1 واقعی** | **0.9202** | Centroid baseline (ECAPA frozen, LOO unbiased, threshold=0.320) |
| **Best OOD AUC** | 0.9557 | Centroid distance, LOO unbiased |
| **Best known accuracy** | 94.98% | Centroid LOO |
| **EER (cosine verification)** | 0.346 | ECAPA frozen embeddings |
| **Same-speaker cosine** | 0.7675 | ECAPA frozen |
| **Cross-speaker cosine** | 0.1614 | ECAPA frozen |
| **Macro-F1 (pure argmax)** | 0.7013 | بدون OOD threshold gate |
| **تنها run آموزشی** | **شکست خورده** | OOD head collapsed (acc 0.501 ≈ random), Macro-F1=None |
| **مدل‌های آموزش‌دیده** | **۰ از ۵** | هیچکدام از ۵ encoder هنوز train نشده‌اند |

### ۱.۲ وضعیت فعلی — کد و زیرساخت

| جزء | وضعیت | ارزیابی |
|---|---|---|
| ۵ Encoder قابل تعویض | ✅ کامل | ECAPA, CAM++, ERes2NetV2, WavLM, TitaNet — همگی offline-first |
| Two-headed model (OOD + Speaker/ArcFace) | ✅ کامل | Fuse به ۴۴۷-way probability |
| Multi-window TTA (train & eval) | ✅ کامل | Random crops train, sliding windows eval |
| Leak-free split (MD5 duplicate detection) | ✅ کامل | ۹ گروه تکراری / ۶۹ فایل شناسایی و حذف شده |
| Balanced batch sampler | ✅ کامل | رفع مشکل OOD collapse |
| MLflow on DagsHub | ✅ کامل | Tracking params, metrics, artifacts, checkpoints |
| ZenML pipeline | ✅ کامل | Steps: convert → data → build → train → eval |
| FAISS OOD detector | ✅ کامل | k-NN cosine distance، اختیاری در inference |
| Centroid baseline + fusion | ✅ کامل | 0.9202 Macro-F1 — baseline غیرآموزشی |
| Ensemble (sequential avg-probs) | ✅ کامل | ولی هیچ ensemble واقعی train نشده |
| Vast.ai deployment | ✅ کامل | Automated setup, pre-flight checks |
| Streamlit UI | ✅ کامل | Config editing, remote deployment |
| **HPO / Hyperparameter tuning** | ❌ وجود ندارد | نه Optuna، نه Ray Tune، نه W&B sweeps |
| **Config schema/validation** | ❌ وجود ندارد | yaml.safe_load ساده |
| **Augmentation config-driven** | ❌ نیست | پارامترها هاردکدن در AudioAugmentation |
| **حلقه آموزش واحد** | ❌ تکراری | train.py و steps.py دو حلقه متفاوت با scheduler متفاوت |

### ۱.۳ شکاف تا هدف (0.9202 → 0.97)

| منبع بهبود | برآورد افزایش Macro-F1 | توضیح |
|---|---|---|
| Fine-tune encoder (ECAPA) + ArcFace training | ۰.۰۲–۰.۰۴+ | از 0.95 centroid → بالای 0.96 با fine-tuning |
| Ensemble ۵ encoder | ۰.۰۱–۰.۰۲+ | کاهش variance، پوشش بهتر OOD |
| Temperature calibration | ۰.۰۰۵–۰.۰۱ | Softmax sharpening برای speaker head |
| بهبود augmentation + regularization | ۰.۰۰۵–۰.۰۱ | کاهش overfitting، بهبود generalization |
| HPO systematic tuning | ۰.۰۰۵–۰.۰۱۵ | پیدا کردن operating point بهینه |
| **جمع برآوردی** | **۰.۰۴۵–۰.۰۹۵** | مسیر رسیدن به 0.97 ممکن است |


---

## 2. تحلیل داده

### ۲.۱ آمار کلی (Phase 0 — Label EDA)

| متریک | مقدار |
|---|---|
| **کل فایل‌های صوتی** | ۴,۵۲۹ |
| **فایل‌های Known** | ۲,۲۵۴ (۴۹.۸٪) |
| **فایل‌های Unknown** | ۲,۲۷۵ (۵۰.۲٪) |
| **گویندگان شناخته‌شده** | ۴۴۶ |
| **گویندگان ناشناس (hidden)** | ۵۵۴ |
| **کل افراد (مسابقه)** | ۱,۰۰۰ |
| **کل کلاس‌ها** | ۴۴۷ |

### ۲.۲ توزیع نمونه به ازای هر گوینده (Few-Shot Extreme)

| آماره | مقدار |
|---|---|
| **Min** | ۵ |
| **Max** | ۲۰ |
| **Mean** | ۵.۰۵ |
| **Median** | ۵.۰ |
| **Std** | ۰.۷۴ |
| **Mode** | ۵ فایل (۴۳۹ speaker از ۴۴۶) |

| فایل به ازای speaker | تعداد speaker |
|---|---|
| ۵ | ۴۳۹ |
| ۶ | ۵ |
| ۹ | ۱ |
| ۲۰ | ۱ |

> **نکته حیاتی:** ۴۳۹ از ۴۴۶ speaker (۹۸.۴٪) دقیقاً ۵ فایل دارند. این یک مسئله **extreme few-shot** است — هر speaker فقط چند utterance مستقل دارد. برخلاف VoxCeleb که هر speaker صدها utterance دارد، اینجا باید با ۵ نمونه هر speaker را یاد بگیریم.

### ۲.۳ عدم توازن شدید (Extreme Class Imbalance)

| متریک | مقدار |
|---|---|
| **Unknown : mean-Known ratio** | **۴۵۰×** |
| **Unknown : median-Known ratio** | **۴۵۵×** |
| **سهم Unknown در Macro-F1** | **۱/۴۴۷ ≈ ۰.۲۲٪** |

> **پیامد برای Macro-F1:** چون Macro-F1 میانگین F1 هر کلاس را می‌گیرد، مدل یک ترم F1 برای unknown و ۴۴۶ ترم برای speakerهای شناخته‌شده دارد. مدلی که همه را unknown پیش‌بینی کند، recall بالایی روی unknown دارد ولی روی ۴۴۶ کلاس دیگر صفر → Macro-F1 ≈ ۰. **ریسک غالب، recall پایین speakerهای شناخته‌شده است، نه unknown.**

### ۲.۴ مدت زمان فایل‌ها (Phase 1 — Duration EDA)

| آماره | مقدار |
|---|---|
| **Min (valid)** | ~۰s |
| **Max** | ۲m ۳۹s |
| **Mean** | **۵۸.۲s** |
| **Median** | **۵۹.۶s** |
| **Std** | ۲۱.۳s |
| **فایل‌های > ۳۰s** | ۴,۰۹۱ (۹۰.۳٪) |
| **فایل‌های > ۶۰s** | ۲,۲۱۷ (۴۹.۰٪) |

> **فرصت طلایی:** ۹۰٪ فایل‌ها بالای ۳۰ ثانیه هستند. با window cropping تصادفی، هر فایل ≈ ۱۲ پنجره ۵ ثانیه‌ای مستقل تولید می‌کند → **effective dataset ≈ ۱۲× بزرگتر می‌شود.** این بزرگترین مزیت این دیتاست است و باید حداکثر استفاده از آن بشود.

### ۲.۵ فایل‌های خراب و تکراری

| دسته | تعداد |
|---|---|
| **Corrupted (< 1s)** | ۷۰ (۱.۵٪) — ۲۲ known + ۴۸ unknown |
| **MD5 duplicate groups** | ۹ گروه / ۶۹ فایل |
| **فایل‌های معتبر پس از پاکسازی** | ۴,۴۵۹ |

### ۲.۶ مقایسه Known vs Unknown — مدت زمان

| آماره | Known (n=2,254) | Unknown (n=2,275) |
|---|---|---|
| Mean | ۵۸.۹s | ۵۷.۵s |
| Median | ۶۰.۱s | ۵۹.۱s |

> **Confounder check:** توزیع مدت زمان known و unknown تقریباً یکسان است (Δ ≈ ۱s). مدل نمی‌تواند از duration برای تشخیص OOD استفاده کند — باید به ویژگی‌های صوتی تکیه کند. این خوب است چون "تقلب" را غیرممکن می‌کند.

### ۲.۷ ویژگی‌های آکوستیکی (Phase 2 — Acoustic EDA)

**فرمت:** تمام ۴,۵۲۹ فایل **16 kHz / mono / PCM_16** — همگن، بدون نیاز به resampling.

**مقایسه Known vs Unknown — ویژگی‌های low-level:**

| Feature | p-value | Cohen's d | قابل تشخیص؟ |
|---|---|---|---|
| ZCR | 0.013 | -0.203 | مرزی (|d|≈0.2) |
| Centroid Hz | 0.042 | -0.167 | ❌ |
| Rolloff Hz | 0.035 | -0.173 | ❌ |
| Bandwidth Hz | 0.052 | -0.159 | ❌ |
| RMS, Peak, Flatness, Voiced, F0 | > 0.05 | < 0.12 | ❌ |

> **نتیجه:** هیچ ویژگی low-level آکوستیکی نمی‌تواند known را از unknown جدا کند (همه |d| < 0.2، PCA هم overlap کامل). OOD detection **باید** در فضای embedding انجام شود — معماری two-head + FAISS فعلی صحیح است.

### ۲.۸ فضای Embedding (Phase 3 — Unbiased LOO, ECAPA frozen)

| متریک | مقدار |
|---|---|
| Same-speaker mean cosine | **۰.۷۶۷۵** |
| Cross-speaker mean cosine | **۰.۱۶۱۴** |
| **d′ (separability)** | **۳.۳۸** |
| **EER (cosine verification)** | **۰.۳۴۶** |

> **تفسیر:** فاصله خوبی بین same-speaker و cross-speaker وجود دارد (d′=3.38)، ولی EER=0.346 یعنی هنوز overlap قابل توجهی هست. fine-tuning باید d′ را افزایش دهد.

**Known recognition (LOO centroid):**

| متریک | مقدار |
|---|---|
| **Argmax-centroid accuracy (LOO)** | **۹۴.۹۸٪** |
| **Top-5 accuracy (LOO)** | **۹۶.۱۹٪** |

**OOD detection (LOO centroid distance):**

| متریک | مقدار |
|---|---|
| **AUC** | **۰.۹۵۵۷** |
| Best threshold (Youden) | ۰.۲۹۴ |
| TPR @ threshold | ۰.۹۳۶ |
| FPR @ threshold | ۰.۰۵۹ |

**Macro-F1 نهایی (LOO — عددی که مهم است):**

| روش | Macro-F1 |
|---|---|
| Pure argmax (بدون OOD gate) | ۰.۷۰۱۳ |
| **Best OOD threshold (=0.320)** | **۰.۹۲۰۲** |

> **سقف centroid فعلی 0.92 است. شکاف 0.05 تا 0.97 باید با fine-tuning + ensemble + calibration پر شود.**
>
> توجه: سقف known accuracy حتی با centroid کامل 0.95 است. یعنی از هر ۲۰ speaker شناخته‌شده، ۱ نفر اشتباه تشخیص داده می‌شود. این یعنی **حتی مدل بی‌نقص هم اگر speaker head فقط centroid باشد، از 0.95 بالاتر نمی‌رود.** حتماً به fine-tuning encoder + ArcFace head نیاز داریم.

**ساختار داخلی Unknown (KMeans):**

| متریک | مقدار |
|---|---|
| Unknown files embedded | ۲,۲۲۷ |
| KMeans clusters | ۸ |
| Mean cos to cluster centroid | ۰.۵۳۴ |
| Fraction with cos > 0.5 | ۶۴.۵٪ |

> **دلالت:** unknown ها همگن نیستند — ۸ cluster معنادار وجود دارد. این یعنی unknown فقط "هر چیزی غیر از ۴۴۶ speaker" نیست، بلکه ساختار داخلی دارد. می‌توان برای OOD detection از multi-modal unknown distribution استفاده کرد (مثلاً multiple OOD prototypes به جای یک threshold ساده).

### ۲.۹ طراحی Train/Val Split

| جزء | مقدار |
|---|---|
| **Validation samples (known, ۱/speaker)** | ۴۴۶ |
| **Training samples (known)** | ۱,۸۰۸ |
| **Validation share of unknown** | ۲۰٪ |

> **تحلیل:** ۱ نمونه validation به ازای هر speaker یعنی speakerهایی که ۵ نمونه دارند، ۴ نمونه train + ۱ validation. برای speaker با ۲۰ نمونه، split بهینه‌تر است ولی minority هستند. این split conservative است و از overfitting جلوگیری می‌کند. برای speakerهای با ۵ نمونه، این یعنی ۲۰٪ داده در validation است (نسبت بالایی).


---

## 3. تحلیل معماری فعلی

### ۳.۱ معماری مدل — TwoHeadedSpeakerModel

```
Waveform (B, 1, T)
    │
    ▼
┌──────────────┐
│   Encoder    │  ← 5 encoder قابل تعویض (ECAPA/CAM++/ERes2NetV2/WavLM/TitaNet)
│ (frozen یا   │     هرکدام output_dim متفاوت (192/512/1024)
│  fine-tuned) │
└──────┬───────┘
       │ hidden_states (B, seq_len, dim)
       ▼
┌──────────────┐
│   Pooling    │  ← Statistical / Attentive / Identity
│              │     output_multiplier = 2 or 1
└──────┬───────┘
       │ (B, pooled_dim)
       ├──────────────────┐
       ▼                  ▼
┌─────────────┐   ┌──────────────┐
│  OODHead    │   │ SpeakerHead  │
│ (binary)    │   │ (ArcFace/    │
│             │   │  Linear)     │
│ → 1 logit   │   │ → 446 logits │
└──────┬──────┘   └──────┬───────┘
       │                  │
       └────────┬─────────┘
                ▼
    Fuse: p[0] = σ(ood)
          p[i] = (1-p[0]) × softmax(spk)[i]
          → 447-way probability
```

### ۳.۲ Encoderها — ۵ معماری، همه offline-first

| Encoder | Output Dim | Params | Pooling | Source | Weights Size | VRAM (fp32 enc) |
|---|---|---|---|---|---|---|
| ECAPA-TDNN | ۱۹۲ | ۶.۴M enc / ۲۲.۲M total | identity | SpeechBrain | ۸۹ MB | ۰.۲۱ GB |
| CAM++ | ۵۱۲ | ۷.۳M | identity | ModelScope | ۳۰ MB | ۰.۲۱ GB |
| ERes2NetV2 | ۱۹۲ | ۱۷.۹M | identity | Vendored (`sv_arch.py`) | ۷۲ MB | ۰.۱۵ GB |
| TitaNet-L | ۱۹۲ | ۲۵.۳M | identity | NeMo | ۱۰۲ MB | ۰.۲۱ GB |
| WavLM-Large | ۱۰۲۴ | ۳۱۶M | statistical | HuggingFace | ۱,۲۸۳ MB | **۱.۴۳ GB** |

**انتخاب فعلی (default_config.yaml):** `encoder_type: campp` (قبلاً `titanet` بوده در گزارش قبلی — الان `campp` است).

### ۳.۳ Loss و Heads

```
TwoPartLoss = ood_weight × BCE(ood_logit, is_unknown)
            + speaker_weight × (FocalLoss یا CrossEntropy)(speaker_logits, speaker_label)

Unknown ها با ignore_index=-100 در speaker head ماسک می‌شوند.
```

| جزء | جزئیات |
|---|---|
| **OODHead** | LayerNorm → Linear(in, 256) → ReLU → Dropout(0.4) → Linear(256, 1) |
| **ArcFaceHead** | LayerNorm → Linear(in, 192) → Dropout(0.2) → L2-norm → ArcFace margin (m=0.4, s=30) |
| **LinearSpeakerHead** | LayerNorm → Linear(in, 446) |
| **FocalLoss** | γ=2.0, label_smoothing, ignore_index |

### ۳.۴ جریان آموزش

**دو مسیر موازی (با scheduler متفاوت!):**

| | `src/train.py::train()` | `src/pipelines/steps.py::train_model` |
|---|---|---|
| **نوع تابع** | Plain Python | ZenML @step |
| **Config** | مسیر فایل فقط | دیکشنری |
| **Early stopping** | ❌ ندارد | ✅ روی Macro-F1 (patience=15) |
| **Scheduler** | CosineAnnealingLR(T_max=epochs) | LinearLR warmup(3) → CosineAnnealingWarmRestarts(T_0=10) |
| **MLflow** | ❌ صدا نمی‌زند | ✅ per-epoch logging |
| **OOD grad norm** | ❌ استفاده نمی‌کند (default 1.0) | ✅ استفاده می‌کند |
| **Pruning hook** | ❌ وجود ندارد | ❌ وجود ندارد |

### ۳.۵ Multi-Window TTA — دو حالت ناهماهنگ

| حالت | Aggregation | کجا استفاده می‌شود |
|---|---|---|
| **Train/Eval** (`forward_multi_window`) | **میانگین logit ها** (per-head) | train + val |
| **Inference** (`predict_file_probs`) | **میانگین probability های fused** (۴۴۷-way) | submission |

> **⚠️ این دو از نظر ریاضی معادل نیستند!** چون fusion شامل sigmoid و softmax غیرخطی است، `mean(σ(ood)) ≠ σ(mean(ood))`. این inconsistency ظریف است ولی نشان می‌دهد metric ای که موقع validation می‌بینیم، دقیقاً همان چیزی نیست که در inference leaderboard محاسبه می‌شود.

### ۳.۶ OOD Detection — دو مسیر با دو objective متفاوت

| | FAISS k-NN (اختیاری) | Neural OODHead (همیشگی) |
|---|---|---|
| **امتیاز OOD** | `1 - mean cosine to top-k known centroids (k=5)` | `sigmoid(ood_head_logit)` |
| **Combination** | `α × head + (1-α) × faiss` (α=0.5) | — |
| **Threshold tuning objective** | **Macro-F1** (`evaluate_centroid`) | **Binary F1** (`tune_ood_threshold`) |

> **⚠️ ناهماهنگی:** `tune_ood_threshold` در train.py با binary F1 تنظیم می‌شود، در حالی که centroid baseline با Macro-F1 تنظیم می‌شود. objective نهایی مسابقه Macro-F1 است — threshold باید با Macro-F1 tune شود نه binary F1.

### ۳.۷ Ensemble فعلی — Sequential avg-probs

- **روش:** Sequential (یک مدل در VRAM)، avg probability روی ۴۴۷-way output
- **Calibration:** Temperature روی logitهای میانگین‌گیری‌شده (`ensemble_calibrate.py`)
- **Fusion با centroid:** `α × model_probs + (1-α) × centroid_probs` (اختیاری، `--fuse-centroid`)
- **وضعیت:** کد کامل است ولی هیچ ensemble واقعی train نشده (حتی ECAPA هم train نشده)

### ۳.۸ Augmentation — هاردکدن، غیرقابل تنظیم

کد فعلی (`data_pipeline.py:455-463`):
```python
self.pipeline = AA.Compose([
    AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    AA.PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
    AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    AA.Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
    AA.PolarityInversion(p=0.5),
    AA.Shift(min_shift=-0.1, max_shift=0.1, shift_unit="fraction",
             rollover=True, fade_duration=0.005, p=0.3),
])
```

**فعلاً از config نمی‌خواند → قابل sweep نیست.**  
**MixUp هم کدش هست ولی `mixup_alpha` همیشه 0.0 است (غیرفعال).**


---

## 4. مشکلات و ناهماهنگی‌های شناسایی‌شده

### ۴.۱ مشکلات بحرانی (blocking)

| # | مشکل | تأثیر | راه‌حل |
|---|---|---|---|
| **۱** | `submission.py` در ریشه وجود ندارد — مسابقه `submission.py --data-dir ... --predictions-file-path ...` می‌خواهد | Submission reject می‌شود | اضافه کردن `submission.py` wrapper که به `submission/inference.py` delegate کند |
| **۲** | هیچ مدلی train نشده — همه چک‌پوینت‌ها یا fail خورده‌اند یا init هستند | Macro-F1 ≈ 0 | Train مجدد ECAPA با config درست |
| **۳** | Cache embedding قدیمی (بدون suffix encoder) با سیستم جدید encoder-aware ناسازگار است | خطا در centroid/FAISS | پاکسازی cache و rebuild |

### ۴.۲ ناهماهنگی‌های طراحی

| # | ناهماهنگی | شرح | راه‌حل |
|---|---|---|---|
| **۴** | **TTA train vs inference** | Train logit-avg، inference prob-avg | Unify به prob-avg (چون inference است که leaderboard می‌رود) |
| **۵** | **OOD threshold objective** | train.py binary F1، centroid Macro-F1 | یکسان‌سازی به Macro-F1 |
| **۶** | **دو حلقه آموزش** | train.py vs steps.py — scheduler متفاوت، early stopping فقط در یکی | Extract train_core مشترک |
| **۷** | **Temperature در inference فعال نیست** | `predict_proba` بدون T کار می‌کند | Wire temperature calibration به submission |
| **۸** | **`model.fusion.ensemble_method`** | Dead config — نوشته شده ولی هیچ‌جا خوانده نمی‌شود | حذف یا پیاده‌سازی |

### ۴.۳ مشکلات فنی

| # | مشکل | تأثیر | راه‌حل |
|---|---|---|---|
| **۹** | **Augmentation پارامترها هاردکدن** | قابل sweep/optimize نیست | خواندن از config + ساختن pipeline داینامیک |
| **۱۰** | **MixUp غیرفعال (α=0)** | Regularization مؤثر استفاده نمی‌شود | Wiring به config + فعال‌سازی |
| **۱۱** | **Config بدون schema/validation** | خطاها late runtime آشکار می‌شوند | Pydantic/dataclass schema |
| **۱۲** | **`warmup_steps` dead key** | در YAML هست ولی هیچ‌جا خوانده نمی‌شود | حذف یا wiring |
| **۱۳** | **Dropout ها هاردکدن** (OOD 0.4, ArcFace 0.2) | قابل تنظیم نیستند | خواندن از config |

### ۴.۴ شکاف‌های قابلیتی

| # | شکاف | اولویت |
|---|---|---|
| **۱۴** | **هیچ HPO framework وجود ندارد** | بالا |
| **۱۵** | **RIR/MUSAN augmentation موجود نیست** | متوسط |
| **۱۶** | **SpecAugment (time/freq masking) موجود نیست** | متوسط |
| **۱۷** | **Stochastic Weight Averaging (SWA) موجود نیست** | متوسط |
| **۱۸** | **Sub-Center ArcFace / Angular Margin variants موجود نیست** | پایین |
| **۱۹** | **Prototypical/contrastive loss موجود نیست** | پایین |
| **۲۰** | **ONNX export وجود ندارد** | پایین |


---

## 5. راهکارهای بهبود

### A. تنظیمات و Hyperparameter Tuning

#### A.1 پارامترهای ArcFace — بیشترین تأثیر روی speaker separability

| پارامتر | مقدار فعلی | بازه پیشنهادی | توضیح |
|---|---|---|---|
| **`arcface.margin`** | ۰.۴ | ۰.۱–۰.۶ (step ۰.۰۵) | حاشیه زاویه‌ای — higher = separation بیشتر ولی training سخت‌تر. برای few-shot، margin بالاتر ریسک collapse دارد. |
| **`arcface.scale`** | ۳۰.۰ | ۱۵–۴۵ (step ۵) | مقیاس — higher = gradient تیزتر، convergence سریع‌تر ولی ناپایدارتر |
| **`arcface.embedding_dim`** | ۱۹۲ | ۱۲۸, ۱۹۲, ۲۵۶, ۵۱۲ | بعد embedding — higher = capacity بیشتر ولی overfitting ریسک |

> **پیشنهاد:** با margin=0.3, scale=25 شروع کنید (ایمن‌تر از current 0.4/30) و با HPO بالا ببرید. برای encoder با output_dim=512 (مثل CAM++)، embedding_dim=512 منطقی‌تر از 192 است.

#### A.2 نرخ یادگیری — Separate encoder/head LR

| پارامتر | مقدار فعلی | بازه پیشنهادی | توضیح |
|---|---|---|---|
| **`learning_rate`** (heads) | ۱e-۴ | ۱e-۵ → ۱e-۳ (log-uniform) | Headها از صفر train می‌شوند → LR بالاتر |
| **`encoder_lr`** (encoder) | ۱e-۵ | ۱e-۶ → ۱e-۴ (log-uniform) | Encoder pretrained → LR پایین‌تر (۵–۲۰× کمتر از head) |

> **تحلیل:** encoder_lr=1e-5 / lr=1e-4 → ratio=10×. این منطقی است. می‌توان ratio را تا ۲۰× هم افزایش داد (encoder_lr=5e-6) بسته به اینکه encoder چقدر frozen باشد.

#### A.3 Loss weights — تعادل OOD/Speaker

| پارامتر | مقدار فعلی | بازه پیشنهادی | توضیح |
|---|---|---|---|
| **`ood_loss_weight`** | ۰.۳ | ۰.۱–۰.۵ | هرچه کمتر → speaker head سیگنال بیشتری می‌گیرد |
| **`speaker_loss_weight`** | ۰.۷ | ۰.۵–۰.۹ | ۱-ood_weight |
| **`label_smoothing`** | ۰.۱ | ۰.۰–۰.۳ | Smoothing بالاتر → regularization بیشتر، OOD بهتر، speaker accuracy کمی افت |

> **نکته:** با توجه به severe class imbalance (۴۵۰× unknown vs known)، وزن OOD باید low بماند (۰.۲–۰.۳) تا speaker head majority gradient را دریافت کند. اگر OOD weight را زیاد کنید، speaker head underfit می‌شود.

#### A.4 Regularization و optimizer

| پارامتر | مقدار فعلی | بازه پیشنهادی | توضیح |
|---|---|---|---|
| **`weight_decay`** | ۱e-۵ | ۱e-۶ → ۱e-۳ (log-uniform) | مقدار فعلی خیلی کم است. برای few-shot معمولاً ۱e-۴ تا ۱e-۳ بهتر است. |
| **`ood_pos_weight`** | ۱.۰ | ۰.۵–۲.۰ | وزن BCE برای unknown. اگر OOD underfit است، > ۱.۰ شود. |
| **`max_grad_norm`** | ۵.۰ | ۱.۰–۱۰.۰ | Gradient clipping |
| **`ood_grad_norm`** | ۱.۰ (default) | ۰.۵–۳.۰ | Clipping مجزا برای OOD head |

#### A.5 Batch و sampling

| پارامتر | مقدار فعلی | بازه پیشنهادی | توضیح |
|---|---|---|---|
| **`ood_batch_ratio`** | ۰.۵ | ۰.۳–۰.۶ | نسبت unknown در هر batch. 0.5 یعنی batch ۳۲ نفره → ۱۶ unknown + ۱۶ known. |
| **`num_train_windows`** | ۳ | ۲–۸ | تعداد windowهای تصادفی به ازای هر utterance. higher = data بیشتر ولی training کندتر. |

> **تحلیل ood_batch_ratio:** 0.5 متعادل است. اگر OOD head بیش‌فعال است، کمش کنید (0.3). اگر OOD underfit است، زیادش کنید (0.6). توجه: با 0.5 و batch=32، ۱۶ unknown در batch داریم — ۱۶ نمونه مثبت برای OOD head که کافی است.

### B. Regularization و Anti-Overfitting (Few-Shot)

#### B.1 مشکل اصلی: فقط ۵ نمونه برای ۴۳۹ speaker

با توجه به اینکه ۹۸٪ speakerها فقط ۵ فایل دارند، overfitting روی speaker head ریسک اصلی است. راهکارها:

#### B.2 MixUp (فعال‌سازی — کد موجود، غیرفعال)

```yaml
# data_pipeline.py:540 — mixup_alpha همیشه 0.0
# باید از config خوانده شود
data:
  mixup_alpha: 0.2  # Beta(0.2, 0.2) — mild mixing
```

MixUp دو utterance را با وزن Beta(α,α) ترکیب می‌کند → augmentation در سطح embedding، نه waveform. برای few-shot عالی است چون نمونه‌های جدید بین speakerها می‌سازد.

**پیشنهاد:** α = 0.2 (mild)، یا حتی α = 0.4 برای mixing قوی‌تر.

#### B.3 Label Smoothing (افزایش)

مقدار فعلی 0.1 است. برای few-shot extreme (۵ نمونه)، می‌توان smoothing را افزایش داد:

```yaml
training:
  label_smoothing: 0.15  # یا 0.2 برای regularization قوی‌تر
```

#### B.4 Stochastic Weight Averaging (SWA)

SWA میانگین وزن‌های چند checkpoint آخر را می‌گیرد — مانند ensemble رایگان:

```python
# PyTorch built-in: torch.optim.swa_utils
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
# Apply after ~70% of training
```

برای few-shot، SWA variance را کاهش می‌دهد و generalization را بهبود می‌دهد (معمولاً ۰.۵–۱٪ بهبود).

#### B.5 Weight Decay بالاتر

مقدار فعلی ۱e-۵ برای few-shot کم است. وزن‌های ArcFace head نباید خیلی sharp شوند:

```yaml
training:
  weight_decay: 1.0e-4  # ۱۰× بیشتر از فعلی
```

#### B.6 Freeze/Unfreeze Strategy

برای هر encoder باید استراتژی freeze متفاوتی داشت:

| Encoder | استراتژی پیشنهادی | توضیح |
|---|---|---|
| ECAPA | **Unfreeze last 2 blocks** + headها | ECAPA سبک است، fine-tune جزئی safe است |
| CAM++ | **Fully frozen** (فعلی) + headها | 512-d embedding قوی; fine-tune فقط در round 2 |
| ERes2NetV2 | **Fully frozen** (فعلی) + headها | Vendored architecture — fine-tune risky |
| WavLM | **Fully frozen** (فعلی) + headها | ۳۱۶M پارامتر — fine-tune فقط روی ۱۲+ GB GPU |
| TitaNet | **Fully frozen** (فعلی) + headها | NeMo wrapper — fine-tune complex |

> **پیشنهاد:** ECAPA را با unfreeze شروع کنید (بیشترین potential برای بهبود). بقیه encoderها frozen بمانند. اگر ECAPA fine-tuned از centroid 0.92 بهتر شد، CAM++ و ERes2NetV2 را هم unfreeze کنید.

### C. Augmentation و Data Engineering

#### C.1 Config-driven augmentation (پیش‌نیاز هر بهبود)

تمام پارامترهای augmentation باید از config خوانده شوند:

```yaml
augmentation:
  gaussian_noise:
    enabled: true
    min_amplitude: 0.001
    max_amplitude: 0.015
    p: 0.5
  pitch_shift:
    enabled: true
    min_semitones: -1
    max_semitones: 1
    p: 0.3
  time_stretch:
    enabled: true
    min_rate: 0.8
    max_rate: 1.25
    p: 0.2
  gain:
    enabled: true
    min_gain_db: -6
    max_gain_db: 6
    p: 0.3
  polarity_inversion:
    enabled: true
    p: 0.5
  shift:
    enabled: true
    min_shift: -0.1
    max_shift: 0.1
    p: 0.3
  mixup:
    alpha: 0.2  # 0 = disabled
```

این امکان sweep پارامترهای augmentation را در HPO فراهم می‌کند.

#### C.2 Reverberation (RIR) — بیشترین تأثیر برای generalization

افزودن RIR augmentation با `audiomentations.AddImpulseResponse` یا استفاده از کتابخانه‌های RIR:

```python
# audiomentations support
AA.AddImpulseResponse(p=0.3)
# یا اگر audiomentations نسخه قدیمی:
AA.ApplyImpulseResponse(p=0.3)
```

RIR باعث می‌شود مدل نسبت به محیط‌های مختلف (اتاق کوچک، سالن، فضای باز) مقاوم شود. تأثیر آن روی speaker embedding معمولاً **بیشتر از noise/pitch است**.

#### C.3 Background noise (MUSAN / environmental)

افزودن نویز پس‌زمینه واقعی (نه فقط Gaussian):

```python
AA.AddBackgroundNoise(
    sounds_path="data/musan/noise/",
    min_snr_db=0, max_snr_db=15, p=0.3
)
```

MUSAN dataset رایگان است (~۱۲ GB) و استاندارد صنعتی برای speaker recognition. می‌توان از بخش noise و music آن استفاده کرد.

#### C.4 SpecAugment (time/frequency masking روی mel spectrogram)

اگر encoder از mel spectrogram استفاده می‌کند (ECAPA)، می‌توان SpecAugment را به عنوان پیش‌پردازش اعمال کرد:

```python
# Time masking: چند بازه زمانی را صفر می‌کند
# Frequency masking: چند باند فرکانسی را صفر می‌کند
```

SpecAugment روی waveform اعمال نمی‌شود — باید در سطح mel spectrogram باشد. برای encoderهایی که raw waveform می‌گیرند، TimeDomainSpecAugment (time masking + time warping) مناسب‌تر است.

#### C.5 Codec/Compression augmentation

Simulating codec compression (MP3, AMR-WB, etc.):

```python
# با استفاده از ffmpeg یا torchaudio
# Simulate 8kHz, 16kbps compression artifacts
```

این کمک می‌کند مدل نسبت به کیفیت‌های مختلف صدا مقاوم شود.

#### C.6 تعداد پنجره‌های آموزش

```yaml
audio:
  num_train_windows: 5  # از ۳ به ۵ افزایش
```

با ۵ window تصادفی (هر کدام ۸s از نقاط مختلف فایل ۶۰ ثانیه‌ای)، effective epochs ≈ ۱.۶۷× فعلی. برای ۹۰٪ فایل‌های >۳۰s، می‌توان حتی تا ۸ window رفت.

> **توجه:** افزایش num_train_windows → هر epoch طولانی‌تر → باید epochs را متناسب کاهش داد یا early stopping را فعال نگه داشت.

#### C.7 Window duration تدریجی

می‌توان در طول آموزش window duration را تغییر داد:

- Early epochs: ۳-۴s windows (سریع‌تر، اطلاعات زمانی کمتر)
- Mid epochs: ۶-۸s (اطلاعات زمانی بیشتر)
- Late epochs: full file یا طولانی‌ترین window ممکن

این progressive training از coarse-to-fine learning پشتیبانی می‌کند.

### D. معماری و Heads

#### D.1 بهبود ArcFace — margin/scale schedule

به جای margin ثابت، می‌توان margin را در طول آموزش افزایش داد (margin annealing):

```python
# Epoch 1-10: margin = 0.2 (آسان)
# Epoch 11-30: margin = 0.3
# Epoch 31+: margin = 0.4 (هدف نهایی)
```

این strategy از collapse جلوگیری می‌کند و convergence را پایدارتر می‌کند — مخصوصاً برای few-shot با ۵ نمونه.

#### D.2 Sub-Center ArcFace

به جای یک center برای هر کلاس، K center داشته باشیم (K=2 یا ۳):

```python
# هر speaker یک مجموعه K تایی centroid دارد
# Loss = min_k(ArcFace(embedding, center_k))
```

برای speakerهایی که utteranceهای متنوع دارند (مثلاً speaker با ۲۰ فایل)، چندین sub-center به مدل اجازه می‌دهد intra-speaker variation را مدل کند. پیاده‌سازی نسبتاً ساده است: `Linear(in, num_classes × K)` سپس reshape.

#### D.3 OOD Head improvements

**الف) Multi-headed OOD (پیشنهاد):** به جای یه binary head، از N تا binary head با initialisation مختلف ensemble بگیریم (کاهش variance OOD).

**ب) OOD Head با embedding ورودی:** در معماری فعلی OOD head مستقیماً pooled embedding را می‌گیرد. می‌توان speaker logits را هم به OOD head داد (concat یا attention):

```python
ood_input = torch.cat([pooled_embedding, speaker_logits], dim=-1)
```

ایده: speaker logits حاوی اطلاعاتی درباره "چقدر این utterance به speakerهای شناخته‌شده نزدیک است" هستند — این سیگنال قوی برای OOD detection است.

**ج) Energy-based OOD score:** به جای sigmoid(ood_logit)، از انرژی آزاد (free energy) speaker softmax استفاده شود:

```python
ood_score = -temperature * logsumexp(speaker_logits / temperature)
```

این از توزیع speaker head به عنوان OOD detector استفاده می‌کند (بدون نیاز به head جداگانه). محبوب در literature مدرن.

#### D.4 Dropout قابل تنظیم

```yaml
model:
  ood_head_config:
    hidden_dim: 256
    dropout: 0.3  # از 0.4 کاهش (برای few-shot، dropout بالا speaker head را ضعیف می‌کند)
  speaker_head_config:
    arcface:
      dropout: 0.15  # از 0.2 کمی کاهش
```

#### D.5 Pooling استراتژی

WavLM از `statistical` pooling استفاده می‌کند (mean+std). می‌توان برای encoderهای دیگر هم `attentive` pooling را test کرد:

```yaml
model:
  encoder_config:
    ecapa:
      pooling_type: attentive  # از identity به attentive
```

Attentive pooling وزن attention روی frameها اعمال می‌کند — برای utteranceهای بلند (۶۰s) مفید است چون همه frameها به یک اندازه مهم نیستند.

### E. Ensemble و Calibration

#### E.1 استراتژی ensemble — سه سطح

| سطح | روش | هزینه | تأثیر |
|---|---|---|---|
| **سطح ۱: avg-probs** (فعلی) | میانگین probability 447-way | None | Baseline |
| **سطح ۲: weighted avg** | وزن‌های learned یا heuristic (مثلاً بر اساس val Macro-F1 هر مدل) | کم | +۰.۳–۰.۸٪ |
| **سطح ۳: LearnedFusion** | MLP روی concatenated probabilities | متوسط (نیاز به train روی val دارد) | +۰.۵–۱.۵٪ |
| **سطح ۴: Stacking** | Logistic regression / XGBoost روی per-model logits | متوسط | +۱–۲٪ |

> **پیشنهاد:** از weighted avg (سطح ۲) شروع کنید — وزن‌ها را proportional to val Macro-F1 قرار دهید. سپس اگر ensemble نتیجه بهتری از بهترین تک‌مدل نداد، به سطح ۳ بروید.

#### E.2 Temperature Calibration — وصل به submission

کد `calibrate_temperature` در `metrics.py:163` وجود دارد ولی در submission استفاده نمی‌شود. باید:

1. Temperature بهینه را در checkpoint ذخیره کنید
2. در `submission/inference.py`، `predict_proba` را با `speaker_logits / temperature` صدا بزنید

```python
# model.py predict_proba فعلی (بدون T):
speaker_probs = F.softmax(speaker_logits, dim=-1)

# با T:
speaker_probs = F.softmax(speaker_logits / temperature, dim=-1)
```

#### E.3 Centroid fusion — افزایش وزن centroid

در `submission/inference.py`، centroid fusion با `--fuse-centroid` فعال می‌شود و `centroid_alpha=0.5`. این ترکیب برابر مدل و centroid است. می‌توان centroid را قوی‌تر وزن داد (۰.۶–۰.۷) چون centroid LOO unbiased است و overfit نمی‌کند.

```bash
python submission/inference.py ... --fuse-centroid --centroid-alpha 0.7
```

#### E.4 FAISS OOD به صورت پیش‌فرض فعال

در inference فعلی، FAISS OOD با `--faiss-ood <alpha>` اختیاری است. پیشنهاد: با توجه به AUC=0.9557، FAISS OOD را به صورت پیش‌فرض با alpha=0.3 فعال کنید (blend ملایم).

#### E.5 Calibration با استفاده از validation Macro-F1

فعلاً `calibrate_temperature` با grid search دنبال بهترین temperature برای Macro-F1 می‌گردد. می‌توان به جای grid search از binary search/logarithmic search استفاده کرد (دقت بیشتر، cost کمتر). یا از Platt scaling با logistic regression روی logitها.

### F. زیرساخت و MLOps

#### F.1 Hyperparameter Optimization (Optuna) — پیشنهاد اصلی

با توجه به architecture فعلی (بهترین fit = Optuna):

**Search space پیشنهادی (فاز ۱ — scalar only):**

```yaml
# configs/hpo/search_space.yaml
float:
  training.learning_rate:              {low: 1e-5, high: 1e-3, log: true}
  training.encoder_lr:                 {low: 1e-6, high: 1e-4, log: true}
  training.weight_decay:              {low: 1e-6, high: 1e-3, log: true}
  training.label_smoothing:           {low: 0.0,  high: 0.3}
  training.ood_loss_weight:           {low: 0.1,  high: 0.5}
  model.speaker_head_config.arcface.margin: {low: 0.1, high: 0.5}
  model.speaker_head_config.arcface.scale:  {low: 15.0, high: 40.0}
  audio.ood_batch_ratio:              {low: 0.3, high: 0.6}
```

**Study setup:**
- Sampler: `TPESampler` (Bayesian، بهتر از random)
- Pruner: `MedianPruner` (n_warmup_steps=5, n_startup_trials=5)
- Storage: `sqlite:///checkpoints/hpo/study.db` (resumable)
- epoch per trial: ۳۰ (با early stopping patience=8)
- Trials: ۲۰–۵۰

**فاز ۲:** پس از scalar tuning، categorical search روی encoder_type و pooling_type:

```yaml
categorical:
  model.encoder_type: [campp, ecapa, wavlm, titanet, eres2net]
  model.pooling_type: [identity, statistical, attentive]
```

#### F.2 یک‌سازی حلقه‌های آموزش

دو حلقه آموزش (train.py و steps.py) باید یکی شوند. پیشنهاد:

```python
# src/train_core.py
def train_core(config, train_loader, val_loader, class_map,
               callbacks=None, trial=None):
    """حلقه آموزش مشترک — هم train.py هم steps.py هم HPO از آن استفاده می‌کنند"""
    ...
    for epoch in range(epochs):
        train_epoch(...)
        val_metrics = validate_epoch(...)
        macro_f1 = evaluate_macro_f1(...)
        
        # Pruning hook for Optuna
        if trial is not None:
            trial.report(macro_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Callbacks for MLflow, checkpoint, early stopping
        ...
```

#### F.3 Config Schema و Validation

```python
# src/config_schema.py
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    epochs: int = Field(default=200, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0)
    encoder_lr: float = Field(default=1e-5, gt=0)
    ...
```

این validation را به صورت early (pre-training) انجام می‌دهد و از خطاهای late runtime جلوگیری می‌کند.

#### F.4 MLflow experiment مجزا برای HPO

```yaml
mlops:
  experiment_name: speaker-identification-hpo  # جدا از training اصلی
```

#### F.5 DVC برای داده

از DVC که در پروژه تعریف شده برای version کردن:
- MUSAN noise files
- RIR impulse responses
- Checkpointهای train شده

### G. تکنیک‌های خاص Few-Shot و Open-Set

#### G.1 Prototypical Networks / Prototypical Loss

به جای ArcFace classification، از prototypical loss استفاده شود:

- هر speaker یک prototype (مرکز embedding) دارد
- Loss = softmax(similarity(embedding, all_prototypes))
- Prototype = mean embedding of support samples (همان centroid)

```python
# مزیت: کاملاً non-parametric — بدون وزن کلاس
# Speaker head تبدیل به nearest-prototype classifier می‌شود
# این برای ۵ نمونه بسیار طبیعی‌تر از softmax با ۴۴۶ کلاس است
```

> **پیشنهاد:** این می‌تواند speaker head فعلی (ArcFace 446-way softmax) را جایگزین کند. هر speaker یک prototype 192-بعدی دارد. Speaker head = `L2(embedding) @ L2(prototypes).T`. عملاً centroid classifier ای است که gradient از آن عبور می‌کند! این بهترین fusion بین neural training و centroid logic است.

#### G.2 Multi-Task Learning: Speaker ID + Verification

علاوه بر classification speakerها، یه loss verification کمکی (contrastive یا triplet):

```python
# verification loss: same-speaker pair distance < cross-speaker pair distance + margin
L_total = L_speaker + λ * L_verification
```

این embedding را برای unseen speakerها (unknown) هم بهتر می‌کند چون یاد می‌گیرد "چه چیزی یک speaker را unique می‌کند"، نه فقط "این speaker کدام یک از ۴۴۶ تاست".

#### G.3 OOD Augmentation در Embedding Space

در فضای embedding، sampleهای OOD مصنوعی تولید کنید:

```python
# بین speakerهای مختلف interpolation کن
ood_embedding = α * speaker_A_emb + (1-α) * speaker_B_emb
# این embedding به هیچ speaker مشخصی تعلق ندارد → label = unknown
```

این OOD head را با نمونه‌های OOD مصنوعی بیشتر train می‌کند. ساده و مؤثر است (نیاز به داده جدید ندارد).

#### G.4 Large Margin Cosine Loss (LMCL) / CosFace

جایگزین ArcFace (angular margin) با CosFace (cosine margin):

```python
# ArcFace: cos(θ + m)
# CosFace: cos(θ) - m
```

CosFace معمولاً stable تر از ArcFace است (margin در cosine space است نه angular). برای few-shot با margin کوچک (m=0.1-0.2) امتحان شود.

#### G.5 Speaker-specific batch sampling

به جای random balanced batch (که ۵۰٪ unknown و ۵۰٪ random known است)، از PK sampling استفاده شود:

- هر batch: P speaker، هرکدام K utterance (P × K = batch_size)
- مثلاً P=4, K=4 → ۱۶ known + ۱۶ unknown (با balanced sampler)

این باعث می‌شود speaker head utteranceهای متفاوت یک speaker را در یک batch ببیند — برای ArcFace که intra-class compactness را یاد می‌گیرد، حیاتی است.

#### G.6 Data Distillation / Knowledge Distillation

اگر WavLM (316M params) آموزش داده شود، می‌توان از آن به عنوان teacher برای distill کردن دانش به ECAPA (22M) استفاده کرد:

```python
L_distill = KL(softmax(student_logits / T), softmax(teacher_logits / T))
```

این به ECAPA اجازه می‌دهد از قدرت WavLM استفاده کند بدون هزینه inference آن.

### H. بهینه‌سازی Inference و Submission

#### H.1 TTA Consistency Fix

همان‌طور که در بخش ۳.۵ توضیح داده شد، TTA train (logit-avg) و inference (prob-avg) ناهماهنگ هستند.

**پیشنهاد:** inference را مبنا قرار دهید (prob-avg) چون این چیزی است که leaderboard می‌بیند. پس train/val هم باید prob-avg شوند. یا هر دو را به logit-avg تغییر دهید ولی مطمئن شوید eval metrics با هر دو روش محاسبه شده و اختلافشان documented شود.

#### H.2 Temperature در Submission فعال شود

```python
# submission/inference.py → predict_file_probs
# افزودن temperature از checkpoint
probs = model.predict_proba(window, temperature=ood_thresholds.get("temperature", 1.0))
```

#### H.3 Batching در Submission

فعلاً inference هر فایل را جداگانه پردازش می‌کند. می‌توان فایل‌ها را batch کرد:

```python
# به جای:
for file in files:
    probs = predict_file_probs(model, file)  # batch_size=1

# از:
batch_files = files[:batch_size]
batch_probs = predict_batch_probs(model, batch_files)  # efficient GPU utilization
```

با batch_size=8-16، inference time تقریباً ۲-۳× کاهش می‌یابد.

#### H.4 Optional ONNX Export (برای inference سریع‌تر)

ECAPA-TDNN و ERes2NetV2 صرفاً PyTorch native هستند (وابستگی به framework خاص ندارند). می‌توانند به ONNX export شده و با ONNX Runtime اجرا شوند — معمولاً ۱.۵–۲× سریع‌تر.

```python
torch.onnx.export(model, dummy_input, "model.onnx", ...)
```

**برای WavLM/CAM++/TitaNet** که وابستگی به HF/ModelScope/NeMo دارند، ONNX export پیچیده‌تر است و ممکن است ارزشش را نداشته باشد (مگر زمان inference بحرانی شود).

#### H.5 submission.py wrapper (رفع مشکل بحرانی شماره ۱)

فایل `submission.py` در ریشه پروژه اضافه شود که به `submission/inference.py` delegate کند:

```python
# submission.py (در ریشه پروژه)
import sys
sys.path.insert(0, "submission")
from inference import main
# delegate --data-dir --predictions-file-path to Click command
```

#### H.6 Cache Embedding بازسازی

با توجه به IMPLEMENTATION_PLAN2، cache قدیمی (بدون suffix encoder) باید پاک و با سیستم جدید بازسازی شود:

```bash
rm data/processed/embeddings_train.npy  # cache قدیمی
python src/eda_embeddings.py  # بازسازی با suffix encoder-aware
python src/centroid_baseline.py  # بازسازی centroid baseline
```


---

## 6. نقشه راه اولویت‌بندی‌شده

### فاز ۱: رفع مشکلات بحرانی (روز ۱)

| اولویت | اقدام | تأثیر مورد انتظار | زمان |
|---|---|---|---|
| 🔴 P0 | اضافه کردن `submission.py` wrapper در ریشه پروژه | Submission پذیرفته می‌شود | ۳۰ دقیقه |
| 🔴 P0 | پاکسازی cache قدیمی و rebuild با سیستم encoder-aware | Centroid/FAISS کار می‌کنند | ۱ ساعت |
| 🔴 P0 | **Train اولیه ECAPA** با config فعلی (balanced sampler + Macro-F1 + OOD threshold) — اولین run موفق | خروج از Macro-F1=0 | ۱–۲ ساعت |

### فاز ۲: تنظیمات پایه و HPO Infrastructure (روزهای ۲–۴)

| اولویت | اقدام | تأثیر مورد انتظار | زمان |
|---|---|---|---|
| 🟡 P1 | Unify دو حلقه آموزش یا extract train_core مشترک | زیرساخت تمیز برای ادامه | ۳–۴ ساعت |
| 🟡 P1 | نصب Optuna + search space تعریف + objective function + CLI | زیرساخت HPO | ۴–۶ ساعت |
| 🟡 P1 | HPO فاز ۱ (scalar only): LR, margin, scale, label_smoothing, weight_decay, ood_weight, ood_batch_ratio | +۰.۵–۱.۵٪ Macro-F1 | ۱–۲ روز (۲۰-۳۰ trial) |

### فاز ۳: بهبود Regularization و Augmentation (روزهای ۵–۷)

| اولویت | اقدام | تأثیر مورد انتظار | زمان |
|---|---|---|---|
| 🟡 P1 | Config-driven augmentation (خواندن پارامترها از YAML) | پیش‌نیاز sweep augmentation | ۱–۲ ساعت |
| 🟡 P1 | فعال‌سازی MixUp (wiring mixup_alpha به config) | +۰.۲–۰.۵٪ | ۱ ساعت |
| 🟡 P1 | افزایش num_train_windows از ۳ به ۵-۶ | داده بیشتر، generalization بهتر | ۰.۵ ساعت |
| 🟢 P2 | اضافه کردن RIR augmentation | +۰.۳–۰.۸٪ | ۲–۳ ساعت |
| 🟢 P2 | اضافه کردن MUSAN background noise | +۰.۲–۰.۵٪ | ۲–۳ ساعت |
| 🟢 P2 | SWA (Stochastic Weight Averaging) | +۰.۳–۰.۵٪ | ۱–۲ ساعت |

### فاز ۴: Train تمام Encoderها و Ensemble (روزهای ۸—۱۴)

| اولویت | اقدام | تأثیر مورد انتظار | زمان |
|---|---|---|---|
| 🟡 P1 | Train ECAPA با best config از HPO | رسیدن به >0.94 Macro-F1 | ۲–۳ ساعت |
| 🟡 P1 | Build embedding cache برای ECAPA | پیش‌نیاز ensemble | ۱ ساعت |
| 🟡 P1 | Train CAM++ با best config | +۰.۲–۰.۵٪ over ECAPA (بسته به encoder) | ۲–۳ ساعت |
| 🟡 P1 | Train ERes2NetV2 با best config | +۰.۲–۰.۵٪ | ۲–۳ ساعت |
| 🟢 P2 | Train WavLM (نیاز به ≥۱۲GB GPU — Vast.ai RTX 3090/4090) | +۰.۳–۱.۰٪ | ۴–۶ ساعت |
| 🟢 P2 | Train TitaNet (اختیاری — اگر زمان/GPU بود) | +۰.۱–۰.۳٪ | ۲–۳ ساعت |
| 🔴 P0 | **Ensemble calibration** با ۳-۵ مدل train شده | +۱–۲٪ (بزرگترین جهش) | ۲–۳ ساعت |

### فاز ۵: بهینه‌سازی نهایی (روزهای ۱۵—۱۷)

| اولویت | اقدام | تأثیر مورد انتظار | زمان |
|---|---|---|---|
| 🟡 P1 | Temperature calibration در submission فعال شود | +۰.۳–۰.۸٪ | ۱ ساعت |
| 🟡 P1 | Centroid fusion با وزن بهینه (alpha sweep) | +۰.۳–۰.۵٪ | ۱ ساعت |
| 🟡 P1 | FAISS OOD فعال به صورت پیش‌فرض | +۰.۲–۰.۴٪ | ۰.۵ ساعت |
| 🟢 P2 | HPO فاز ۲ (categorical: encoder+pooling combinations) | +۰.۵–۱.۰٪ | ۱–۲ روز |
| 🟢 P2 | TTA consistency fix (prob-avg در train/val) | اطمینان از صحت ارزیابی | ۱–۲ ساعت |
| 🟢 P2 | Prototypical head جایگزین ArcFace (اختیاری) | +۰.۵–۱.۵٪ (potentially) | ۴–۶ ساعت |

### فاز ۶: اعتبارسنجی و ارسال (روز ۱۸)

| اولویت | اقدام | تأثیر مورد انتظار | زمان |
|---|---|---|---|
| 🔴 P0 | Build و verify submission package (idempotent) | Submission آماده | ۱ ساعت |
| 🔴 P0 | Dry-run روی data sample (بدون GPU) | تأیید pipeline submission | ۰.۵ ساعت |
| 🔴 P0 | Measure inference time روی RTX 3090 | تأیید بودجه ۲۰ دقیقه‌ای | ۱ ساعت |
| 🔴 P0 | Zip و ارسال | — | ۰.۵ ساعت |


---

## 7. جداول مرجع

### ۷.۱ خلاصه hyperparameter ها — مقادیر فعلی و بازه پیشنهادی

| مسیر در config | مقدار فعلی | بازه پیشنهادی HPO | نوع | log scale |
|---|---|---|---|---|
| `training.learning_rate` | ۱e-۴ | ۱e-۵ → ۱e-۳ | float | ✅ |
| `training.encoder_lr` | ۱e-۵ | ۱e-۶ → ۱e-۴ | float | ✅ |
| `training.weight_decay` | ۱e-۵ | ۱e-۶ → ۱e-۳ | float | ✅ |
| `training.label_smoothing` | ۰.۱ | ۰.۰ → ۰.۳ | float | ❌ |
| `training.ood_loss_weight` | ۰.۳ | ۰.۱ → ۰.۵ | float | ❌ |
| `training.speaker_loss_weight` | ۰.۷ | ۱-ood_weight | derived | ❌ |
| `training.ood_pos_weight` | ۱.۰ | ۰.۵ → ۲.۰ | float | ❌ |
| `training.max_grad_norm` | ۵.۰ | ۱.۰ → ۱۰.۰ | float | ❌ |
| `training.ood_grad_norm` | ۱.۰ | ۰.۵ → ۳.۰ | float | ❌ |
| `training.epochs` | ۲۰۰ | ۳۰ (HPO) / ۱۰۰ (final) | int | ❌ |
| `training.early_stopping_patience` | ۱۵ | ۸ (HPO) / ۱۵ (final) | int | ❌ |
| `model.speaker_head_config.arcface.margin` | ۰.۴ | ۰.۱ → ۰.۶ | float | ❌ |
| `model.speaker_head_config.arcface.scale` | ۳۰.۰ | ۱۵ → ۴۵ | float | ❌ |
| `model.speaker_head_config.arcface.embedding_dim` | ۱۹۲ | ۱۲۸, ۱۹۲, ۲۵۶, ۵۱۲ | categorical | ❌ |
| `model.ood_head_config.hidden_dim` | ۲۵۶ | ۱۲۸, ۲۵۶, ۵۱۲ | categorical | ❌ |
| `audio.ood_batch_ratio` | ۰.۵ | ۰.۳ → ۰.۶ | float | ❌ |
| `audio.num_train_windows` | ۳ | ۲ → ۸ | int | ❌ |
| `audio.duration_seconds` | ۸.۰ | ۵.۰, ۸.۰, ۱۰.۰ | categorical | ❌ |

### ۷.۲ Encoderها — مقایسه کامل

| ویژگی | ECAPA | CAM++ | ERes2NetV2 | TitaNet | WavLM |
|---|---|---|---|---|---|
| **منبع** | SpeechBrain | ModelScope | Vendored | NeMo | HuggingFace |
| **پارامترها (encoder)** | ۶.۴M | ۷.۳M | ۱۷.۹M | ۲۵.۳M | ۳۱۶M |
| **کل پارامترها (با heads)** | ۲۲.۲M | ~۷.۵M | ~۱۸.۱M | ~۲۵.۵M | ~۳۱۷M |
| **بعد خروجی** | ۱۹۲ | ۵۱۲ | ۱۹۲ | ۱۹۲ | ۱۰۲۴ |
| **وزن فایل** | ۸۹ MB | ۳۰ MB | ۷۲ MB | ۱۰۲ MB | ۱,۲۸۳ MB |
| **VRAM (fp32 enc + fp16 head)** | ۰.۲۱ GB | ۰.۲۱ GB | ۰.۱۵ GB | ۰.۲۱ GB | ۱.۴۳ GB |
| **سرعت inference (ms/file)** | ۵۷۹ | ۴۹۳ | ۳۳۰ | ۲۰۹ | ۱,۲۰۳ |
| **Frozen pooling** | identity | identity | identity | identity | statistical |
| **Fine-tune پیشنهادی** | Last 2 blocks | Fully frozen | Fully frozen | Fully frozen | Fully frozen |
| **اولویت آموزش** | 🥇 اول | 🥈 دوم | 🥉 سوم | ۴⃣ چهارم | ۵⃣ پنجم |

### ۷.۳ تاریخچه امتیازات (واقعی و تخمینی)

| منبع | Macro-F1 | Known Acc | OOD AUC | توضیح |
|---|---|---|---|---|
| Random baseline | ~۰.۰۰۲ | — | — | Uniform 1/447 |
| Centroid LOO (pure argmax) | ۰.۷۰۱۳ | ۹۴.۹۸٪ | — | بدون OOD threshold |
| **Centroid LOO (best threshold)** | **۰.۹۲۰۲** | ۹۴.۹۸٪ | ۰.۹۵۵۷ | **خط مبنا فعلی** (thr=0.320) |
| Centroid LOO (Youden thr) | ~۰.۹۰–۰.۹۲ | ۹۴.۹۸٪ | ۰.۹۵۵۷ | thr=0.294 |
| Centroid biased (in-sample) | ~۰.۹۵ | ۹۵.۵٪ | ۰.۹۵۳ | ⚠️ خوش‌بینانه (در گزارش README) |
| Neural network (تنها run — شکست خورده) | ~۰ | ۵۵.۸٪ | AUC≈۰.۵ | OOD head collapsed |
| **هدف مسابقه** | **≥ ۰.۹۷** | — | — | 🏆 |

### ۷.۴ کلیدهای dead و مشکلات config

| کلید | موقعیت | مشکل | اقدام |
|---|---|---|---|
| `training.warmup_steps` | default_config.yaml:126 | هرگز خوانده نمی‌شود — warmup هاردکد ۳ epoch است | حذف یا wiring به scheduler |
| `model.fusion.ensemble_method` | default_config.yaml:101 | هرگز خوانده نمی‌شود — ensemble inference مستقل است | حذف |
| `mixup_alpha` | SpeakerDataset.__init__ | همیشه ۰.۰ است — از config نمی‌خواند | Wiring به config |
| `augmentation params` | data_pipeline.py:455-463 | هاردکدن — غیرقابل sweep | Config-driven |
| `dropout OOD/ArcFace` | heads.py:37, 152 | هاردکدن — غیرقابل تنظیم | Config-driven |

### ۷.۵ ناهماهنگی‌هایی که باید fix شوند

| ناهماهنگی | شرح | اولویت |
|---|---|---|
| **TTA train (logit-avg) ≠ inference (prob-avg)** | غیرخطی بودن fusion باعث اختلاف می‌شود | P2 |
| **OOD threshold: binary F1 (train) ≠ Macro-F1 (centroid)** | Objective نهایی Macro-F1 است | P1 |
| **Temperature calibration: هست ولی در submission نیست** | `predict_proba` بدون T | P1 |
| **دو حلقه آموزش با scheduler متفاوت** | CosineAnnealingLR vs Warmup+CosineRestart | P1 |
| **warmup_steps dead key** | در YAML هست، در کد نیست | P3 |


---

## پیوست: خلاصه راهکارها به ترتیب اولویت

### اقدامات فوری (این هفته)

1. ✅ `submission.py` wrapper در ریشه
2. ✅ پاکسازی cache و rebuild
3. ✅ **Train اولیه ECAPA** — اولین run موفق
4. ✅ Unify حلقه‌های آموزش (extract train_core)
5. ✅ Config-driven augmentation
6. ✅ فعال‌سازی MixUp
7. ✅ نصب و راه‌اندازی Optuna HPO

### اقدامات میان‌مدت (هفته دوم)

8. ✅ HPO scalar tuning (۲۰-۳۰ trial)
9. ✅ RIR + MUSAN augmentation
10. ✅ Train ECAPA با best HPO config
11. ✅ Train CAM++ و ERes2NetV2
12. ✅ Ensemble calibration با ۳ مدل

### اقدامات بلندمدت (هفته سوم)

13. ✅ Train WavLM (Vast.ai 3090+)
14. ✅ Temperature calibration در submission
15. ✅ Centroid fusion با alpha sweep
16. ✅ Batching در inference
17. ✅ Build + verify + ارسال submission

---

*گزارش تولید شده بر اساس تحلیل کامل کدبیس، EDA چهار فاز، تاریخچه پروژه، و مستندات مسابقه — ۱۲ آگوست ۲۰۲۶*