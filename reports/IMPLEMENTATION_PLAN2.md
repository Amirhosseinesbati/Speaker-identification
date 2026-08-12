# IMPLEMENTATION_PLAN2 — برنامه اجرایی ارتقا پروژه Speaker Identification (IAAA 2026)

> **تاریخ تولید:** ۲۰۲۶-۰۸-۱۱
> **ارزیابی کلان:** **ترمیم و بازسازی (دسته ۳)** — معماری و کد منظم و جامع است، اما اجرای آموزش شکست خورده، پکیج ارسالی ناقص است و نقطه ورود مسابقه وجود ندارد.
> **هدف نهایی:** Macro-F1 ≥ ۰٫۹۷ روی ۴۴۷ کلاس (۴۴۶ known + ۱ unknown).
> **شاخه کاری:** `feature/advanced-speaker-id`. بعد از هر گام یک `git commit` با پیام conventional بزن. `git push` فقط با اجازه کاربر.
> **زبان:** کد و نام متغیرها انگلیسی؛ توضیحات فارسی. مسیرها نسب به ریشه پروژه.
> **مهم برای ایجنت اجراکننده:** همه دستورات اجرا را با `uv run --no-sync python ...` بزن تا uv محیط را sync نکند و خرابش نکند.

---

## فهرست

- [بخش ۰ — خلاصه ارزیابی و یافته‌های بحرانی](#بخش-۰)
- [بخش ۱ — درک مسابقه](#بخش-۱)
- [بخش ۲ — خلاصه EDA و اعداد کلیدی](#بخش-۲)
- [بخش ۳ — استراتژی برنده](#بخش-۳)
- [بخش ۴ — ارزیابی معماری فعلی: نقاط قوت و شکست‌ها](#بخش-۴)
- [بخش ۵ — گام‌های اجرایی (به ترتیب)](#بخش-۵)
- [بخش ۶ — چک‌لیست تأیید نهایی](#بخش-۶)

---

## بخش ۰ — خلاصه ارزیابی و یافته‌های بحرانی <a name="بخش-۰"></a>

### دسته ارزیابی: **ترمیم و بازسازی (۳)**

معماری کد عالی است؛ اجرای آموزش و پکیج ارسالی نیازمند بازسازی است.

### ۵ یافته بحرانی (به ترتیب اولویت)

#### 🔴 یافته ۱ — فایل `submission.py` وجود ندارد (نقص مهلک ساختاری)

مسابقه در `Competition-Guide/submissionforleaderbord.txt` صراحتاً می‌خواهد:
```
python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv
```
پروژه فقط `submission/inference.py` دارد. سیستم ارزیابی `submission.py` را جستجو می‌کند و پیدا نمی‌کند ⇒ **ارسال رد می‌شود قبل از اجرای مدل.**

#### 🔴 یافته ۲ — فقط ۱ مدل آموزش دیده و آن هم شکست خورده

فایل `checkpoints/best_model.pt` بررسی شد:
| فیلد | مقدار | وضعیت |
|---|---|---|
| encoder | `ecapa` | — |
| epoch | 50 (از 200) | توقف زودهنگام |
| `val_macro_f1` | `None` | متریک مسابقه محاسبه/ذخیره نشده |
| `val_ood_acc` | 0.501 | 🔴 تصادفی — OOD head کاملاً collapse شده |
| `val_speaker_acc` | 0.558 | 🔴 55.8% (در مقابل 95% baseline centroid) |
| `ood_threshold` | `None` | آستانه OOD تنظیم و ذخیره نشده |

مدل OOD همه را «known» پیش‌بینی می‌کند (accuracy 0.501 روی split 50/50 = همیشه کلاس majority). این یعنی P(unknown) ≈ 0 همیشه ⇒ argmax روی ۴۴۶ کلاس known ⇒ unknown F1 = 0 ⇒ Macro-F1 فرو می‌ریزد.

#### 🔴 یافته ۳ — هیچ انسمبلی آموزش ندیده است

استراتژی اصلی پروژه انسمبل ۵ encoder است (ECAPA + WavLM + CAM++ + ERes2NetV2 + TitaNet). واقعیت:
- `checkpoints/` فقط `best_model.pt` (ECAPA شکست‌خورده) و `init_model.pt` (WavLM 1.2GB، فقط initialize شده، هرگز آموزش ندیده) دارد.
- هیچ `ecapa_best.pt`، `campp_best.pt`، `wavlm_best.pt`، `eres2net_best.pt`، `titanet_best.pt` وجود ندارد.
- `build_submission.py` با `glob("*_best.pt")` هیچ فایلی پیدا نمی‌کند ⇒ `submission/checkpoints/` ساخته نمی‌شود.

#### 🔴 یافته ۴ — پکیج ارسالی بدون checkpoint و با پیش‌بینی تصادفی

- `submission/checkpoints/` وجود ندارد.
- `submission/sample_predictions.csv` همه مقادیر `0.00223714 = 1/447` است (توزیع یکنواخت = پیش‌بینی تصادفی).
- اگر همین حالا zip ساخته و ارسال شود، Macro-F1 ≈ 0 می‌شود.

#### 🟠 یافته ۵ — embedding cache قدیمی و ناسازگار

- فایل‌های `data/processed/embeddings_train.npy` (بدون پسوند encoder) وجود دارند — از قبل از refactor encoder-aware cache هستند.
- `centroid_baseline.py` اکنون `_cache_paths(encoder)` را فراخوانی می‌کند که دنبال `embeddings_train_ecapa.npy` می‌گردد ⇒ cache قدیمی نادیده گرفته می‌شود ⇒ باید rebuild شود.
- فایل `init_model.pt` (1.2GB) فضای دیسک را هدر می‌دهد — هرگز آموزش ندیده، باید حذف شود.

### آنچه Already Good است (دست نزن — Over-engineering ممنوع)

| مؤلفه | فایل | وضعیت |
|---|---|---|
| معماری دو-هد | `src/model.py` | ✅ عالی — fusion formula درست |
| ۵ encoder با بارگذاری offline | `src/encoders.py` | ✅ عالی — factory pattern، offline-first |
| ArcFace + Linear heads | `src/heads.py` | ✅ عالی |
| Model factory | `src/model_factory.py` | ✅ عالی |
| Data pipeline با balanced sampler | `src/data_pipeline.py` | ✅ عالی — `make_balanced_batch_sampler` مشکل OOD collapse را حل می‌کند |
| Leak-free split | `src/data_pipeline.py` | ✅ عالی — MD5 duplicate detection |
| متریک مسابقه (Macro-F1) | `src/metrics.py` | ✅ عالی — `evaluate_macro_f1`, `calibrate_temperature` |
| FAISS OOD detector | `src/ood_detector.py` | ✅ عالی |
| Centroid baseline + fusion | `src/centroid_baseline.py` | ✅ عالی — encoder-aware cache |
| Sequential ensemble inference | `submission/inference.py` | ✅ عالی — memory-efficient، TTA |
| Ensemble calibration | `src/ensemble_calibrate.py` | ✅ عالی — per-checkpoint config |
| Vendored ERes2NetV2 | `src/sv_arch.py` | ✅ عالی — no modelscope dependency |
| Pooling layers | `src/pooling.py` | ✅ عالی |
| Build submission script | `scripts/build_submission.py` | ✅ عالی — idempotent |
| Training engine | `src/train.py` | ✅ خوب — Macro-F1 checkpoint selection، OOD threshold tuning، separate LR برای encoder |

---

## بخش ۱ — درک مسابقه <a name="بخش-۱"></a>

> منبع: `Competition-Guide/iaaa-competition-2026-speaker-identification.pdf` (۶ صفحه) + `submissionforleaderbord.txt` + `leaderbordpakage.txt`.

### تعریف مسئله
- **Open-Set Speaker Classification** — طبقه‌بندی تکه‌های صوتی به ۴۴۷ کلاس: ۴۴۶ گوینده known (UUID) + ۱ کلاس `"unknown"` (تجمیع ۵۵۴ گوینده OOD).
- هر نفر ~۵۰/۵۰ split train/eval. eval پنهان است.
- مدل باید برای هر chunk یک **توزیع احتمال ۴۴۷‌تایی** خروجی بدهد. پیش‌بینی نهایی = **argmax** روی این ۴۴۷ احتمال.

### متریک
- **Macro-Averaged F1 روی هر ۴۴۷ کلاس** — هر کلاس وزن برابر (1/447). یک F1 برای unknown + یک F1 برای هر known speaker.
- مدل همیشه-unknown ⇒ F1_unknown بالا ولی F1 همه ۴۴۶ known = 0 ⇒ Macro-F1 ≈ 0.002.
- مدل همیشه-known ⇒ F1_unknown = 0 ⇒ Macro-F1 ≈ 0.
- ⇒ **هم known recall و هم unknown recall باید همزمان بالا باشند.**

### قوانین
- ✅ مجاز: مدل pretrained عمومی، دیتاست خارجی، pretraining/SSL، انسمبل.
- ❌ ممنوع: داده برچسب‌دار حاوی گوینده‌های eval، دسترسی به برچسب‌های پنهان.
- خروجی باید ۴۴۷‌تایی باشد (constraint سخت).

### ساختار ارسال (بحرانی)
- **فایل ورودی:** `submission.py` در ریشه zip.
- **آرگومان‌ها:** `--data-dir <test_folder>` و `--predictions-file-path <output.csv>`.
- **خروجی CSV:** ستون `id` + ستون‌های احتمال برای هر کلاس. پروژه فعلی `id,0,1,...,446` (۴۴۸ ستون) تولید می‌کند که درست است.
- **پکیج‌های سرور:** `leaderbordpakage.txt` پین‌های نسخه را تعیین می‌کند: `torch>=2.10`, `transformers>=4.57,<4.58`, `speechbrain>=1.0.3`, `modelscope>=1.38.1`, `nemo-toolkit[asr]>=2.7.3`, `faiss-cpu>=1.14.3` و غیره. پروژه این‌ها را در `pyproject.toml` رعایت کرده.

---

## بخش ۲ — خلاصه EDA و اعداد کلیدی <a name="بخش-۲"></a>

> منابع: `eda/Phase0_EDA_Report.md`، `eda/Phase1_Advanced_EDA_Report.md`، `eda/Phase2_Acoustic_EDA_Report.md`، `eda/Phase3_Embedding_EDA_Report.md`.

### اعداد کلیدی داده
| مورد | مقدار |
|---|---|
| کل فایل‌ها | ۴۵۲۹ |
| فایل‌های known (۴۴۶ speaker) | ۲۲۵۴ (۴۹.۸٪) |
| فایل‌های unknown | ۲۲۷۵ (۵۰.۲٪) |
| فایل per known speaker | حداقل ۵، میانه ۵، حداکثر ۲۰ |
| طول فایل (میانه) | ۵۹.۶ ثانیه |
| فایل‌های > ۳۰s | ۹۰.۳٪ |
| فایل‌های خراب (< ۱s) | ۷۰ (۲۲ known + ۴۸ unknown) |
| فایل‌های MD5 تکراری | ۶۹ (۹ گروه) |
| Sample rate / format | ۱۶kHz / mono / PCM_16 |

### سقف baseline (ECAPA centroid، LOO unbiased)
| متریک | مقدار |
|---|---|
| Known accuracy (LOO) | ۹۴.۹۸٪ |
| OOD detection AUC | ۰.۹۵۵۷ |
| Macro-F1 (pure argmax) | ۰.۷۰۱۳ |
| **Macro-F1 (best OOD threshold)** | **۰.۹۲۰۲** (thr=0.320) |
| EER (cosine verification) | ۰.۳۴۶ |

> این عدد (۰.۹۲) با یک centroid classifier ساده و threshold tuning به‌دست آمده. هدف ۰.۹۷ است. شکاف ۰.۰۵ باید با fine-tuning + ensemble + calibration پر شود.

### نتیجه‌گیری EDA
1. مسئله **few-shot** است (~۵ نمونه per speaker) ⇒ centroid/metric-learning برتر از classifier از-صفر.
2. فایل‌ها بلند (~۶۰s) ولی مدل ۸s می‌بیند ⇒ **multi-window TTA ضروری است** (همان‌طور که پیاده‌سازی شده).
3. هیچ confounder آکوستیکی برای OOD وجود ندارد ⇒ OOD باید در فضای embedding حل شود.
4. کلاس unknown ۵۰٪ داده ولی ۱/۴۴۷ وزن ⇒ **OOD head collapse فاجعه‌بار است** (دقیقاً رخ داده).
5. EDA کافی است — نیازی به EDA جدید نیست. скрипت‌های موجود (`src/eda*.py`) کامل هستند.

---

## بخش ۳ — استراتژی برنده <a name="بخش-۳"></a>

### استراتژی در ۵ محور

#### محور ۱ — آموزش موفق تک‌مدل‌ها
- هر encoder جداگانه آموزش ببیند با **balanced batch sampler** (OOD/known = 50/50) که در کد موجود است.
- **Checkpoint selection بر اساس val Macro-F1** (نه val loss) — در `src/train.py` پیاده‌سازی شده.
- **OOD threshold tuning** در پایان هر epoch و ذخیره در checkpoint — در `src/train.py` پیاده‌سازی شده.
- **Augmentation ملایم** (PitchShift ±1 semitone، نه ±4) — در `src/data_pipeline.py` اصلاح شده.

#### محور ۲ — انسمبل ۵ encoder (یا حداقل ۳)
- ۵ encoder با معماری و pretrained data متفاوت ⇒ diversity بالا ⇒ ensemble gain زیاد.
- Sequential ensemble (یک مدل در GPU در هر لحظه) — در `submission/inference.py` پیاده‌سازی شده.
- Average fusion ساده احتمالاً کافی است (LearnedFusion موجود ولی نیاز به training جداگانه دارد).

#### محور ۳ — Centroid fusion
- Centroid classifier با frozen embeddings به‌عنوان baseline قوی (Macro-F1 = 0.92).
- Fusion: `α · model_probs + (1-α) · centroid_probs` — در `submission/inference.py` با `--fuse-centroid` پیاده‌سازی شده.
- α بهینه روی val set تنظیم شود.

#### محور ۴ — Temperature calibration
- Speaker softmax temperature را روی val set برای Macro-F1 بهینه کن — در `src/metrics.py:calibrate_temperature` پیاده‌سازی شده.

#### محور ۵ — Multi-window TTA
- ۸s windows، ۵۰٪ overlap، max ۸ windows — در config و inference پیاده‌سازی شده.

### اولویت آموزش (اگر زمان/GPU محدود است)
| اولویت | encoder | دلیل |
|---|---|---|
| ۱ | `ecapa` | سبک‌ترین (۲۲M)، سریع‌ترین آموزش، baseline قوی |
| ۲ | `campp` | سبک (۷M)، 512-d embedding، diversity خوب |
| ۳ | `eres2net` | سبک (۱۸M)، 192-d، vendored (no dependency) |
| ۴ | `wavlm` | سنگین (۳۱۶M) ولی قوی‌ترین (1024-d) — نیاز به GPU با VRAM ≥ ۱۲GB |
| ۵ | `titanet` | سنگین (۲۵M)، NeMo dependency سنگین — اختیاری |

> حداقل ۳ مدل (ecapa + campp + eres2net) انسمبل معنی‌دار می‌سازند. ۵ مدل ایده‌آل است.

---

## بخش ۴ — ارزیابی معماری فعلی: نقاط قوت و شکست‌ها <a name="بخش-۴"></a>

### نقاط قوت (آنچه درست است — دست نزن)

1. **معماری مدل (`src/model.py`):** دو-هد (OOD + Speaker) با fusion formula درست:
   ```
   p[0] = sigmoid(ood_logit)
   p[i] = (1 - p[0]) * softmax(speaker_logits)[i]
   ```
   multi-window TTA با average logits. Numerical safety با clamp + renormalize.

2. **۵ Encoder (`src/encoders.py`):** همه offline-first (local_path + allow_hub_download=false). factory pattern تمیز. Batched forward با no_grad برای frozen encoders. BatchNorm safety (eval mode همیشه). رفع باگ‌های پیچیده (WavLM fp16 NaN، NeMo 2.7 API change، ModelScope device mismatch).

3. **Data pipeline (`src/data_pipeline.py`):** balanced batch sampler (`make_balanced_batch_sampler`) که OOD collapse را حل می‌کند. Leak-free split با MD5 duplicate detection. Multi-window dataset (train: random crops، eval: sliding windows). audiomentations augmentation.

4. **Training (`src/train.py`):** TwoPartLoss (BCE + FocalLoss). Separate LR برای encoder vs heads. CosineAnnealingLR. Gradient clipping جداگانه برای OOD head. NaN detection. **Macro-F1 checkpoint selection**. OOD threshold tuning + persistence.

5. **Metrics (`src/metrics.py`):** macro_f1_score با labels=list(range(447)) — دقیقاً متریک مسابقه. Temperature calibration. predict_global_classes با optional OOD threshold.

6. **Inference (`submission/inference.py`):** sequential ensemble (memory-efficient). fp16 autocast. Multi-window TTA. Centroid fusion. FAISS OOD gate. Safe fallback (uniform for undecodable files).

7. **Build submission (`scripts/build_submission.py`):** idempotent. weights/ + checkpoints/ + src/ + configs/ + inference.py.

### نقاط شکست (آنچه باید ترمیم شود)

| # | مشکل | شدت | فایل متاثر |
|---|---|---|---|
| ۱ | `submission.py` وجود ندارد | 🔴 مهلک | `submission/` |
| ۲ | فقط ۱ مدل شکست‌خورده آموزش دیده | 🔴 مهلک | `checkpoints/` |
| ۳ | ۴ encoder دیگر آموزش ندیده‌اند | 🔴 مهلک | `checkpoints/` |
| ۴ | `init_model.pt` ۱.2GB هدر | 🟠 متوسط | `checkpoints/` |
| ۵ | embedding cache قدیمی | 🟠 متوسط | `data/processed/` |
| ۶ | `build_submission.py` با `inference.py` کار می‌کند نه `submission.py` | 🔴 مهلک | `scripts/build_submission.py` |

---

## بخش ۵ — گام‌های اجرایی (به ترتیب) <a name="بخش-۵"></a>

> **قانون:** بعد از هر گام یک `git commit` با پیام conventional بزن. همه دستورات با `uv run --no-sync python ...` اجرا شوند.
> **تأیید محیط قبل از شروع:**
> ```bash
> cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
> uv run --no-sync python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
> ```
> اگر `cuda False` بود، ابتدا بخش ۰ از `IMPLEMENTATION_PLAN.md` قدیمی را اجرا کن (تعمیر torch cu126).

---

### گام ۱ — افزودن `submission.py` (نقطه ورود مسابقه) <a name="گام-۱"></a>

**چرا:** سیستم ارزیابی `submission.py` را در ریشه zip جستجو می‌کند. پروژه فقط `submission/inference.py` دارد. بدون این فایل، ارسال قبل از اجرای مدل رد می‌شود.

**اقدام:**

فایل جدید `submission/submission.py` بساز — یک thin wrapper که `submission.inference.main` را با همان CLI فراخوانی می‌کند:

```python
"""
Competition entry point — delegates to submission.inference.main.

The competition evaluation system calls:
    python submission.py --data-dir <folder> --predictions-file-path <csv>

This file is a thin wrapper so the submission package has BOTH
submission.py (competition-required name) and inference.py (the
full-featured implementation).
"""
from submission.inference import main

if __name__ == "__main__":
    main()
```

**تأیید:**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -m submission.submission --help
# باید همان help دوره inference.py را نشان دهد (گزینه‌های --data-dir و --predictions-file-path)
```

**همچنین `scripts/build_submission.py` را به‌روز کن:**
در تابع `build`، بعد از کپی `inference.py`، مطمئن شو `submission.py` هم کپی می‌شود. بخش کپی entrypoint را پیدا کن (حدود خط ۱۰۴) و این را اضافه کن:

```python
    # ── entrypoint ──
    for f in ("inference.py", "submission.py", "__init__.py"):
        src = ROOT / "submission" / f
        dst = SUB / f
        if src.exists():
            if src.resolve() == dst.resolve():
                print(f"  ✓ {f} (already in place)")
            else:
                shutil.copy2(src, dst)
                print(f"  ✓ {f}")
        else:
            if f == "submission.py":
                print(f"  ⚠ submission.py missing — create it first!")
```

**git commit:** `fix(submission): add submission.py entry point required by competition`

---

### گام ۲ — پاکسازی checkpoint هادر و embedding cache قدیمی <a name="گام-۲"></a>

**چرا:** `init_model.pt` (1.2GB) هرگز آموزش ندیده و فضای دیسک را هدر می‌دهد. embedding cache قدیمی (بدون پسوند encoder) ناسازگار با کد فعلی است.

**اقدام:**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
# حذف checkpoint های هرگزآموزش‌دیده یا شکست‌خورده
rm -f checkpoints/init_model.pt
rm -f checkpoints/best_model.pt
rm -f checkpoints/latest_model.pt

# حذف embedding cache قدیمی (encoder-keyed cache در گام ۴ rebuild می‌شود)
rm -f data/processed/embeddings_train.npy
rm -f data/processed/embeddings_train_labels.npy
rm -f data/processed/embeddings_val.npy
rm -f data/processed/embeddings_val_labels.npy
```

> ⚠️ **هشدار:** `best_model.pt` مدل شکست‌خورده است (OOD acc = 0.501 = تصادفی). حذف آن امن است چون با آموزش مجدد در گام ۳ جایگزین می‌شود.

**git commit:** `chore: remove broken checkpoints and stale embedding cache`

---

### گام ۳ — آموزش مجدد ECAPA با تنظیمات صحیح <a name="گام-۳"></a>

**چرا:** مدل فعلی شکست خورده (OOD collapse). کد آموزش (`src/train.py`) اکنون Macro-F1 checkpoint selection و OOD threshold tuning دارد — فقط باید اجرا شود.

**اقدام:**

۱. بررسی کن که `configs/default_config.yaml` تنظیمات درست دارد:
   - `encoder_type: ecapa`
   - `speaker_head_type: arcface`
   - `audio.duration_seconds: 8.0`
   - `audio.num_train_windows: 3`
   - `audio.ood_batch_ratio: 0.5` (حیاتی برای جلوگیری از OOD collapse)
   - `training.epochs: 200`
   - `training.early_stopping_patience: 15`

۲. آموزش را اجرا کن:
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -m src.train
```

> این روی GPU محلی (GTX 1660 Ti) اجرا می‌شود. با batch_size=8 و num_train_windows=3، هر epoch حدود ۵-۱۰ دقیقه طول می‌کشد. ۲۰۰ epoch با early stopping (patience=15) احتمالاً در ۵۰-۱۰۰ epoch متوقف می‌شود.

۳. **پس از آموزش، تأیید کن:**
```bash
uv run --no-sync python -c "
import torch
ckpt = torch.load('checkpoints/ecapa_best.pt', map_location='cpu', weights_only=False)
print('Epoch:', ckpt.get('epoch'))
print('Val Macro-F1:', ckpt.get('val_macro_f1'))
print('Val OOD Acc:', ckpt.get('val_ood_acc'))
print('Val Speaker Acc:', ckpt.get('val_speaker_acc'))
print('OOD Threshold:', ckpt.get('ood_threshold'))
assert ckpt.get('val_macro_f1') is not None, 'Macro-F1 not saved!'
assert ckpt.get('val_ood_acc') > 0.7, 'OOD head still collapsed!'
assert ckpt.get('val_speaker_acc') > 0.8, 'Speaker accuracy too low!'
print('ECAPA TRAINING OK ✅')
"
```

> **اگر OOD Acc < ۰٫۷ شد:** مشکل احتمالاً در `ood_batch_ratio` است. بررسی کن که `get_dataloaders` واقعاً `make_balanced_batch_sampler` را با `ood_ratio=0.5` فراخوانی می‌کند (در `src/data_pipeline.py` خط ~۷۸۳). اگر استفاده می‌کند و باز هم collapse شد، `ood_loss_weight` را از 0.3 به 0.5 افزایش بده و `ood_pos_weight` را به 2.0 افزایش بده.

**git commit:** `feat(train): retrain ECAPA with balanced sampler + Macro-F1 selection`

---

### گام ۴ — ساخت embedding cache (encoder-aware) برای ECAPA <a name="گام-۴"></a>

**چرا:** centroid fusion و FAISS OOD gate در inference به embedding cache نیاز دارند. cache باید encoder-keyed باشد (`embeddings_train_ecapa.npy` نه `embeddings_train.npy`).

**اقدام:**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -m src.centroid_baseline --encoder-type ecapa
```

این دستور:
1. embedding های train و val را با ECAPA استخراج می‌کند.
2. آن‌ها را در `data/processed/embeddings_train_ecapa.npy` و ... ذخیره می‌کند.
3. centroid classifier را fit می‌کند.
4. OOD threshold را برای Macro-F1 tune می‌کند.
5. fusion با trained model (اگر `best_model.pt` وجود داشته باشد) را ارزیابی می‌کند.

**تأیید:**
```bash
ls -la data/processed/embeddings_*ecapa*
# باید ببینی: embeddings_train_ecapa.npy, embeddings_val_ecapa.npy, embeddings_train_ecapa_labels.npy, embeddings_val_ecapa_labels.npy, embeddings_ecapa_meta.json
```

**git commit:** `feat(centroid): build encoder-aware embedding cache for ECAPA`

---

### گام ۵ — آموزش CAM++ (encoder دوم) <a name="گام-۵"></a>

**چرا:** انسمبل حداقل ۲ مدل متنوع نیاز است. CAM++ (512-d) با ECAPA (192-d) diversity خوبی می‌سازد.

**اقدام:**

۱. Config را موقتاً به CAM++ تغییر بده:
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
# با یک editor یا sed:
# configs/default_config.yaml → model.encoder_type: campp
# configs/default_config.yaml → model.encoder_config.campp.freeze_encoder: true
# configs/default_config.yaml → model.pooling_type: identity (CAM++ 512-d embedding مستقیم)
```

به‌طور دقیق، این تغییرات را در `configs/default_config.yaml` اعمال کن:
- `model.encoder_type`: `ecapa` → `campp`
- `model.pooling_type`: `identity` (بماند — CAM++ identity pooling)
- `model.encoder_config.campp.freeze_encoder`: `true` (بماند)
- `model.encoder_config.campp.local_path`: `weights/campp` (بماند)

> **نکته ArcFace:** ArcFace head با `embedding_dim: 192` تنظیم شده. CAM++ 512-d embedding دارد. `ArcFaceHead` ابتدا به `embedding_dim` project می‌کند (`embedding_proj: Linear(input_dim → 192)`)، پس `input_dim=512` به‌طور خودکار از `encoder.output_dim` گرفته می‌شود. نیازی به تغییر `embedding_dim` در config نیست — projection هر دو را به 192 می‌برد.

۲. آموزش را اجرا کن:
```bash
uv run --no-sync python -m src.train
```

۳. تأیید:
```bash
uv run --no-sync python -c "
import torch
ckpt = torch.load('checkpoints/campp_best.pt', map_location='cpu', weights_only=False)
print('Epoch:', ckpt.get('epoch'))
print('Val Macro-F1:', ckpt.get('val_macro_f1'))
print('Val OOD Acc:', ckpt.get('val_ood_acc'))
assert ckpt.get('val_ood_acc') > 0.7, 'OOD head collapsed!'
print('CAMPP TRAINING OK ✅')
"
```

۴. Config را به `ecapa` برگردان (برای نهفتهای بعدی):
```bash
# configs/default_config.yaml → model.encoder_type: ecapa
```

۵. embedding cache برای CAM++ بساز:
```bash
uv run --no-sync python -m src.centroid_baseline --encoder-type campp
```

**git commit:** `feat(train): train CAM++ encoder + build embedding cache`

---

### گام ۶ — آموزش ERes2NetV2 (encoder سوم) <a name="گام-۶"></a>

**چرا:** انسمبل ۳ مدل متنوع (ECAPA 192-d + CAM++ 512-d + ERes2NetV2 192-d) diversity کافی برای gain معنی‌دار می‌سازد.

**اقدام:**

۱. Config را به ERes2NetV2 تغییر بده:
- `model.encoder_type`: `eres2net`
- `model.pooling_type`: `identity`
- `model.encoder_config.eres2net.freeze_encoder`: `true`
- `model.encoder_config.eres2net.local_path`: `weights/eres2net`

۲. آموزش:
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -m src.train
```

۳. تأیید:
```bash
uv run --no-sync python -c "
import torch
ckpt = torch.load('checkpoints/eres2net_best.pt', map_location='cpu', weights_only=False)
print('Val Macro-F1:', ckpt.get('val_macro_f1'))
print('Val OOD Acc:', ckpt.get('val_ood_acc'))
assert ckpt.get('val_ood_acc') > 0.7
print('ERES2NET TRAINING OK ✅')
"
```

۴. Config را به `ecapa` برگردان.

۵. embedding cache برای ERes2NetV2:
```bash
uv run --no-sync python -m src.centroid_baseline --encoder-type eres2net
```

**git commit:** `feat(train): train ERes2NetV2 encoder + build embedding cache`

---

### گام ۷ — آموزش WavLM (encoder چهارم — اختیاری ولی توصیه‌شده) <a name="گام-۷"></a>

**چرا:** WavLM-Large (1024-d) قوی‌ترین encoder است ولی ۳۱۶M params دارد و نیاز به VRAM ≥ ۱۲GB. اگر GPU محلی کافی نیست، این گام را روی Vast.ai اجرا کن.

**اقدام:**

۱. Config را به WavLM تغییر بده:
- `model.encoder_type`: `wavlm`
- `model.pooling_type`: `statistical` (WavLM به statistical pooling نیاز دارد)
- `model.encoder_config.wavlm.freeze_feature_extractor`: `true`
- `model.encoder_config.wavlm.local_path`: `weights/wavlm_large`
- `hardware.profiles.local.batch_size`: `4` (WavLM سنگین است — اگر OOM شد به 2 کاهش بده)
- `hardware.profiles.local.mixed_precision`: `true`

> ⚠️ **WavLM fp16 NaN:** `WavLMEncoder.forward` اکنون در fp32 اجرا می‌شود (autocast disabled) — این باگ در کد فعلی رفع شده. نیازی به تغییر نیست.

۲. آموزش:
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -m src.train
```

> اگر روی GTX 1660 Ti (6GB) OOM شد:
> - `batch_size` را به 2 کاهش بده.
> - یا این گام را روی Vast.ai با `setup_vast.sh` اجرا کن.
> - یا این گام را رها کن و با ۳ مدل (ecapa + campp + eres2net) ادامه بده.

۳. تأیید و config برگردان و embedding cache (همانند گام‌های قبل).

**git commit:** `feat(train): train WavLM-Large encoder + build embedding cache`

---

### گام ۸ — آموزش TitaNet (encoder پنجم — اختیاری) <a name="گام-۸"></a>

**چرا:** TitaNet-Large (192-d) تنوع NeMo را اضافه می‌کند. اگر زمان/GPU محدود است، رها کن — ۴ مدل کافی است.

**اقدام:** همانند گام‌های قبل با:
- `model.encoder_type`: `titanet`
- `model.pooling_type`: `identity`
- `model.encoder_config.titanet.freeze_encoder`: `true`
- `model.encoder_config.titanet.local_path`: `weights/titanet/titanet_large.nemo`

**git commit:** `feat(train): train TitaNet-Large encoder + build embedding cache`

---

### گام ۹ — Ensemble calibration + انتخاب بهترین fusion <a name="گام-۹"></a>

**چرا:** انسمبل میانگین احتمالات + temperature calibration می‌تواند ۱-۳ درصد Macro-F1 اضافه کند. این گام بهترین ترکیب را پیدا می‌کند.

**اقدام:**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"

# انسمبل calibration را با همه checkpoint های موجود اجرا کن:
uv run --no-sync python -m src.ensemble_calibrate \
    --checkpoints checkpoints/ecapa_best.pt checkpoints/campp_best.pt checkpoints/eres2net_best.pt \
    --config-path configs/default_config.yaml \
    --batch-size 16
```

> اگر WavLM و TitaNet هم آموزش دیده‌اند، آن‌ها را هم اضافه کن:
> ```
> --checkpoints checkpoints/ecapa_best.pt checkpoints/campp_best.pt checkpoints/eres2net_best.pt checkpoints/wavlm_best.pt checkpoints/titanet_best.pt
> ```

این اسکریپت برای هر مدل:
- val logits را جمع می‌کند.
- per-model Macro-F1 را گزارش می‌دهد.
- average-fusion ensemble Macro-F1 را محاسبه می‌کند.
- بهترین temperature را برای speaker softmax پیدا می‌کند.

> خروجی را در یک فایل یادداشت کن (مثلاً `ensemble_results.txt`) — بعداً برای تنظیم inference استفاده می‌شود.

**git commit:** `feat(ensemble): run calibration report for all trained encoders`

---

### گام ۱۰ — ساخت و تأیید پکیج ارسالی <a name="گام-۱۰"></a>

**چرا:** پکیج نهایی باید تمام checkpoint ها، weights، کد و `submission.py` را داشته باشد و offline اجرا شود.

**اقدام:**

۱. `configs/inference_config.yaml` را بررسی کن — مطمئن شو `allow_hub_download: false` است و همه `local_path` ها درست تنظیم شده‌اند. (این فایل از قبل درست است — فقط verify کن.)

۲. پکیج را بساز:
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python scripts/build_submission.py
```

> این اسکریپت:
> - `src/` را در `submission/src/` کپی می‌کند.
> - `weights/` را در `submission/weights/` کپی می‌کند (idempotent — اگر موجود است skip می‌کند).
> - `checkpoints/*_best.pt` را در `submission/checkpoints/` کپی می‌کند.
> - `configs/inference_config.yaml` را کپی می‌کند.
> - `inference.py` و `submission.py` و `__init__.py` را کپی می‌کند.

۳. **تأیید ساختار:**
```bash
# بررسی فایل‌های ضروری
ls submission/submission.py    # باید وجود داشته باشد
ls submission/inference.py     # باید وجود داشته باشد
ls submission/src/encoders.py  # باید وجود داشته باشد
ls submission/configs/inference_config.yaml  # باید وجود داشته باشد

# بررسی checkpoint ها
ls submission/checkpoints/*_best.pt
# باید ببینی: ecapa_best.pt, campp_best.pt, eres2net_best.pt (و optionally wavlm, titanet)

# بررسی weights
ls -d submission/weights/*/    # باید ببینی: ecapa/ campp/ eres2net/ titanet/ wavlm_large/
```

۴. **تأیید offline اجرا:**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"

# شبیه‌سازی محیط offline لیدربورد
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE="$(pwd)/submission/weights/campp"

# تست اجرا روی چند فایل واقعی
uv run --no-sync python -m submission.submission \
    --data-dir data/processed/audio_wav \
    --predictions-file-path /tmp/test_predictions.csv \
    --checkpoint-path submission/checkpoints/ecapa_best.pt \
    --checkpoint-path submission/checkpoints/campp_best.pt \
    --checkpoint-path submission/checkpoints/eres2net_best.pt \
    --max-eval-windows 2

# بررسی خروجی
head -2 /tmp/test_predictions.csv
# باید ببینی: id,0,1,...,446 و مقادیر غیریکنواخت (نه 1/447)
```

> ⚠️ اگر خطای import یا offline loading داشتی:
> - مطمئن شو `submission/src/` کامل کپی شده (به‌خصوص `encoders.py`, `sv_arch.py`).
> - مطمئن شو `MODELSCOPE_CACHE` به مسیر درست `submission/weights/campp` اشاره می‌کند.
> - اگر SpeechBrain خطای symlink داد: `LocalStrategy.COPY` باید در کد باشد (هست — `src/encoders.py` خط ~۳۵۲).

۵. **تأیید probabilities غیریکنواخت:**
```bash
uv run --no-sync python -c "
import pandas as pd
df = pd.read_csv('/tmp/test_predictions.csv')
probs = df.iloc[:, 1:].values  # بدون ستون id
# بررسی: مقادیر نباید همه 1/447 باشند
uniform = 1.0 / 447
row_max = probs.max(axis=1)
print('Max prob per row (first 5):', row_max[:5])
print('Uniform value:', uniform)
assert row_max[0] > uniform * 2, 'Predictions are still uniform — model not working!'
print('PREDICTIONS OK ✅')
"
```

**git commit:** `feat(submission): build and verify complete submission package`

---

### گام ۱۱ — Zip نهایی و آماده ارسال <a name="گام-۱۱"></a>

**اقدام:**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"

# حذف __pycache__ از submission
find submission/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# ساخت zip
cd ..
zip -r speaker-identification-submission.zip Speaker-identification/submission/
ls -lh speaker-identification-submission.zip
# اندازه انتظاری: ~2.1 GB (weights 1.5GB + checkpoints ~500MB + code ~1MB)
```

> ⚠️ **مهم:** zip فقط محتوای `submission/` را باید داشته باشد، نه کل پروژه. ساختار درون zip:
> ```
> submission/
> ├── submission.py
> ├── inference.py
> ├── __init__.py
> ├── src/
> ├── configs/
> ├── weights/
> ├── checkpoints/
> └── README.md
> ```

**git commit:** `chore: finalize submission zip`

---

## بخش ۶ — چک‌لیست تأیید نهایی <a name="بخش-۶"></a>

قبل از ارسال، همه موارد زیر باید ✅ باشند:

### ساختاری
- [ ] `submission/submission.py` وجود دارد و `--data-dir` و `--predictions-file-path` را می‌پذیرد.
- [ ] `submission/inference.py` وجود دارد (implementation کامل).
- [ ] `submission/src/` شامل همه ماژول‌ها است (`encoders.py`, `model.py`, `model_factory.py`, `heads.py`, `pooling.py`, `metrics.py`, `data_pipeline.py`, `train.py`, `centroid_baseline.py`, `ood_detector.py`, `ensemble.py`, `ensemble_calibrate.py`, `sv_arch.py`, `cli_utils.py`, `audio_preprocessing.py`, `pipelines/`).
- [ ] `submission/src/deploy/` حذف شده (نیازی نیست).
- [ ] `submission/configs/inference_config.yaml` با `allow_hub_download: false` وجود دارد.
- [ ] هیچ `__pycache__` در `submission/` نیست.

### Weights
- [ ] `submission/weights/ecapa/` (SpeechBrain savedir — `hyperparams.yaml` + `embedding_model.ckpt` + ...)
- [ ] `submission/weights/campp/` (ModelScope cache — `campplus_voxceleb.bin` + `configuration.json`)
- [ ] `submission/weights/eres2net/` (`eres2netv2.ckpt`)
- [ ] `submission/weights/titanet/` (`titanet_large.nemo`) — اگر آموزش دیده
- [ ] `submission/weights/wavlm_large/` (`pytorch_model.bin` + `config.json`) — اگر آموزش دیده

### Checkpoints
- [ ] `submission/checkpoints/ecapa_best.pt` — با `val_macro_f1 > 0` و `val_ood_acc > 0.7`
- [ ] `submission/checkpoints/campp_best.pt` — همانند بالا
- [ ] `submission/checkpoints/eres2net_best.pt` — همانند بالا
- [ ] (اختیاری) `submission/checkpoints/wavlm_best.pt`
- [ ] (اختیاری) `submission/checkpoints/titanet_best.pt`
- [ ] هر checkpoint دارای: `config`, `class_map`, `model_state_dict`, `ood_threshold`, `val_macro_f1`

### اجرایی
- [ ] `python submission.py --data-dir <folder> --predictions-file-path <csv>` بدون خطا اجرا می‌شود.
- [ ] خروجی CSV: ۴۴۸ ستون (`id` + `0..446`)، هر سطر جمع = ۱.۰.
- [ ] مقادیر احتمال غیریکنواخت هستند (نه 1/447).
- [ ] اجرا با `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` کار می‌کند (zero network calls).
- [ ] فایل‌های خراب (< 1s) به‌طور خودکار با uniform fallback مدیریت می‌شوند.

### کیفیت مدل
- [ ] val Macro-F1 (ECAPA تنها) > 0.90
- [ ] val OOD accuracy (ECAPA) > 0.85
- [ ] val Macro-F1 (ensemble ۳+ مدل) > val Macro-F1 (بهترین تک‌مدل)
- [ ] ensemble calibration report اجرا شده و بهترین temperature ذخیره شده.

### Runtime
- [ ] تخمین زمان inference روی RTX 3090 با batch 32 < ۲۰ دقیقه (برای ~۵۰۰۰ فایل). (مطابق Phase 6 report: ۸.۳ دقیقه بدون safety factor.)

---

## پیوست — جدول مرجع سریع

### فایل‌هایی که نباید دست بزنی (already good)
| فایل | دلیل |
|---|---|
| `src/model.py` | معماری دو-هد درست است |
| `src/encoders.py` | ۵ encoder offline-first، batched forward، BN safety |
| `src/heads.py` | ArcFace + Linear + OOD head درست است |
| `src/model_factory.py` | Factory pattern تمیز |
| `src/pooling.py` | Statistical + Attentive pooling درست است |
| `src/metrics.py` | Macro-F1 + temperature calibration درست است |
| `src/ood_detector.py` | FAISS OOD detector درست است |
| `src/centroid_baseline.py` | Encoder-aware cache + fusion درست است |
| `src/sv_arch.py` | Vendored ERes2NetV2 درست است |
| `src/data_pipeline.py` | Balanced sampler + leak-free split درست است |
| `src/ensemble.py` | Average + LearnedFusion درست است |
| `src/ensemble_calibrate.py` | Per-checkpoint config calibration درست است |
| `submission/inference.py` | Sequential ensemble + TTA + fusion درست است |
| `scripts/build_submission.py` | Idempotent build (فقط submission.py را اضافه کن) |

### فایل‌هایی که باید تغییر کنند
| فایل | تغییر |
|---|---|
| `submission/submission.py` | **جدید** — thin wrapper به `submission.inference.main` |
| `scripts/build_submission.py` | اضافه کردن `submission.py` به لیست کپی entrypoint |
| `configs/default_config.yaml` | فقط `encoder_type` بین آموزش‌ها عوض می‌شود (برای encoder switch) |

### فایل‌هایی که باید حذف شوند
| فایل | دلیل |
|---|---|
| `checkpoints/init_model.pt` | ۱.2GB، هرگز آموزش ندیده |
| `checkpoints/best_model.pt` | شکست‌خورده (OOD acc=0.501) — با آموزش مجدد جایگزین |
| `checkpoints/latest_model.pt` | شکست‌خورده — با آموزش مجدد جایگزین |
| `data/processed/embeddings_*.npy` (قدیمی) | ناسازگار با encoder-aware cache |

---

> **تأیید نهایی:** این سند با بررسی خط‌به‌خط کد، گزارش‌های EDA، قوانین مسابقه و ساختار ارسال نوشته شده. همه گام‌ها قابل اجرا و قابل تأیید هستند. ایجنت اجراکننده باید گام‌ها را به ترتیب اجرا کند و بعد از هر گام تأیید کند.
