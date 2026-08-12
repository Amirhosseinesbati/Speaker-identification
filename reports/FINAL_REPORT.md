# گزارش نهایی — پیاده‌سازی مسابقه Speaker Identification (IAAA 2026)

**شاخه:** `feature/advanced-speaker-id`  |  **تاریخ:** 2026-08-08  |  **هدف:** Macro-F1 ≥ 0.97 روی ۴۴۷ کلاس

این گزارش خلاصه‌ی اجرای گام‌های «[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)» است:
چه چیزی تغییر کرد، کدام باگ‌های بحرانی رفع شد، و چه چیزی برای کاربر باقی مانده است.

---

## ۱. خلاصه‌ی وضعیت گام‌ها

| گام | عنوان | وضعیت کد | اجرای سنگین |
|-----|-------|-----------|-------------|
| 0 | تعمیر محیط (torch cu126 + CUDA) | ✅ انجام شده (محیط سالم است) | — |
| 1 | زیرساخت متریک (`src/metrics.py`) | ✅ تأیید + تست `METRICS OK` | — |
| 2 | split بدون leakage + حذف خراب‌ها/تکراری‌ها + `split_report.json` | ✅ کامیت `40f6f21` | ✅ اجرا شد: ۷۰ خراب + ۹ گروه تکراری (۶۹ فایل، ۲ گروه برچسب متناقض) |
| 3 | multi-window (استفاده از کل طول فایل) | ✅ کامیت `320dc7b` | ✅ تست با مدل واقعی ECAPA |
| 4 | رفع باگ‌های بحرانی sampler/loss/eval | ✅ کامیت `34c4230` | ✅ تست sampler (نسبت unknown = ۰.۵) |
| 5 | EDA بدون bias (LOO + شبیه‌سازی Macro-F1) | ✅ کامیت `54330b6` | ⏳ **با کاربر** |
| 6 | Centroid baseline + fusion + embedding cache | ✅ کامیت `9b21845` | ⏳ **با کاربر** |
| 7 | Fine-tune بخشی از encoder | ✅ کامیت `0878bdd` | ⏳ **با کاربر** (۸-۱۲ ساعت) |
| 8 | بازسازی `submission/inference.py` | ✅ کامیت `0c88419` | ✅ تست smoke با مدل ساختگی |
| 9 | انسمبل + کالیبراسیون دما | ✅ کامیت `d758c64` | ⏳ **با کاربر** (بعد از آموزش چند مدل) |
| 10 | README + گزارش نهایی + نمونه‌ی submission | ✅ این فایل | — |

---

## ۲. باگ‌های بحرانی که رفع شد (علت شکست run قبلی)

1. **🔴 collapse هد OOD** — `WeightedRandomSampler` با وزن per-class، سهم unknown در batch را به ~۱/۴۴۷ می‌رساند
   (≈۰.۰۷ نمونه در batch ۳۲). جایگزین شد با `make_balanced_batch_sampler` (نسبت هدف ۰.۵۰) در
   `src/data_pipeline.py` و هر دو مسیر `get_dataloaders` و ZenML. تست: نسبت unknown کشیده‌شده = **۰.۵** (قبلاً ~۰.۰۰۲).
2. **Macro-F1 (متریک واقعی مسابقه) هیچ‌جا محاسبه نمی‌شد** — اکنون `src/metrics.py` منبع واحد متریک است و
   انتخاب checkpoint / early-stopping / ارزیابی نهایی همگی بر اساس `macro_f1` (۴۴۷ کلاسه) انجام می‌شود.
3. **ArcFace margin در eval اعمال می‌شد** → اعداد acc پایین‌تر از واقع. حالا forward در eval **بدون labels**
   صدا زده می‌شود (`forward_multi_window`).
4. **OOD threshold فقط print می‌شد** → حالا در `best_model.pt` ذخیره می‌شود (با fallback به median در صورت collapse).
5. **criterion ارزیابی نهایی ناسازگار** → با همان وزن‌ها/فocal/pos_weight آموزش ساخته می‌شود.
6. **Augmentation خشن (PitchShift ±4) روی encoder فریزشده** → به ±۱ (p=0.3) کاهش یافت
   (علت فاصله‌ی معکوس train/val acc).
7. **۸۵٪ اتلاف سیگنال** → `SpeakerDataset` حالا چند پنجره برمی‌گرداند: train = `num_train_windows` کراپ تصادفی،
   eval/inference = پنجره‌های لغزان با هم‌پوشانی ۵۰٪ تا `max_eval_windows`؛ logits میانگین گرفته می‌شود.
8. **leakage در split** → گروه‌های MD5-تکراری فقط به train می‌روند؛ ۷۰ فایل خراب قبل از split حذف می‌شوند؛
   `data/processed/split_report.json` تولید می‌شود.
9. **`submission/inference.py` حذف شده بود** → بازسازی شد با TTA چندپنجره‌ای + threshold ذخیره‌شده
   (قابل‌خاموش) + fusion با centroid (قابل‌خاموش) + انسمبل چند checkpoint + fallback یکنواخت + CSV استاندارد.

---

## ۳. تغییرات کلیدی معماری

- **Fine-tune جزئی:** `ECAPAEncoder.unfreeze_last_n_blocks(2)` — فقط ۲ بلوک آخر SE-Res2Block (۵.۴M پارامتر)
  trainable است؛ `forward` فقط وقتی جزئی unfreeze شده گراف را نگه می‌دارد؛ optimizer دو param group دارد
  (`encoder_lr=1e-5` / `learning_rate=1e-4`).
- **مسیر baseline:** `src/centroid_baseline.py` — cache embedding (چندپنجره‌ای) idempotent +
  کلاسifier centroid + threshold بهینه‌شده برای Macro-F1 + fusion.
- **مسیر انسمبل:** `submission/inference.py --checkpoint-path ...` (چند بار) → میانگین احتمالات؛
  `src/ensemble_calibrate.py` → گزارش per-model/ensemble Macro-F1 + بهترین دما.
- **MLOps چند GPU:** همه‌ی پارامترهای محاسباتی از `hardware.profiles` می‌آیند (batch: ۸/۱۶/۳۲ برای
  1660Ti/3060/3090)؛ بقیه‌ی تنظیمات مشترک است.

---

## ۴. دستورهای اجرای سنگین (با کاربر — به ترتیب پیشنهادی)

> ⚠️ همیشه با `uv run --no-sync` اجرا کنید تا uv محیط را دوباره sync نکند و torch cu126 خراب نشود.

```bash
# 1) EDA بدون bias — ~۵-۱۵ دقیقه روی GPU
uv run --no-sync python -m src.eda_embeddings

# 2) Centroid baseline + embedding cache — چند دقیقه (اول بار cache می‌سازد)
uv run --no-sync python -m src.centroid_baseline

# 3) آموزش (حالا با sampler درست + multi-window + checkpoint بر اساس Macro-F1).
#    config فعلی fine-tune جزئی است (unfreeze_last_n_blocks: 2) — ۸-۱۲ ساعت روی 1660 Ti
uv run --no-sync python -m src.train
#    یا از طریق ZenML:
uv run --no-sync python -m src.pipelines.run_pipeline --run train

# 4) (اختیاری) اگر OOM در fine-tune دیدید: batch_size پروفایل را کم کنید یا
#    freeze_encoder را به true برگردانید.

# 5) ساخت submission روی پوشه‌ی تست
uv run --no-sync python -m submission.inference \
    --data-dir <test-set-folder> \
    --predictions-file-path predictions.csv

# 6) (بعد از آموزش ≥۲ مدل) انسمبل + کالیبراسیون دما
uv run --no-sync python -m src.ensemble_calibrate \
    --checkpoints checkpoints/best_seed1.pt checkpoints/best_seed2.pt
```

---

## ۵. اعداد قبل / بعد (وضعیت فعلی)

| مورد | قبل | بعد |
|------|-----|-----|
| سهم unknown در هر batch | ~۰.۰۰۲ (collapse هد OOD) | **۰.۵۰** (تست ✅) |
| OOD val acc (run آخر) | ۰.۴۹۵ ≈ رندوم | باید بعد از آموزش جدید سنجیده شود |
| انتخاب checkpoint | val loss | **val Macro-F1** |
| فایل‌های حذف‌شده (خراب) | ۰ | **۷۰** (۲۲ known + ۴۸ unknown) |
| گروه‌های تکراری MD5 | نادیده (leakage) | **۹ گروه / ۶۹ فایل** (همه به train) |
| پنجره‌ی دیده‌شده از هر فایل | ۸ ثانیه (≈۱۵٪ سیگنال) | **کل فایل** (پنجره‌های لغزان تا ۸) |
| PitchShift | ±۴ نیم‌پرده | **±۱ نیم‌پرده** |
| OOD threshold | فقط چاپ | **در checkpoint ذخیره می‌شود** |
| سقف تشخیص (Phase 3) | ۹۵.۵٪ in-sample (خوش‌بینانه) | ⏳ عدد **unbiased (LOO)** بعد از اجرای گام ۵ |

---

## ۶. بازتولید (Reproducibility)

- همه‌ی splitها / samplerها / t-SNEها seed ثابت (۴۲) دارند؛ `class_map` داخل هر checkpoint ذخیره می‌شود.
- ترتیب ستون‌های CSV با `class_map` تعیین می‌شود: ستون `0` = unknown، ستون‌های `1..446` = UUID گوینده‌های
  شناخته‌شده به ترتیب lexicographic؛ sidecar `.class_map.json` کنار CSV نوشته می‌شود.
- برای ارائه‌ی جزئیات به برگزارکنندگان: README + این گزارش + `IMPLEMENTATION_PLAN.md` کافی است.

---

> ✅ پایان گزارش. گام‌های باقی‌مانده (۵، ۶، ۷، ۹) صرفاً اجرای سنگین با کاربر هستند؛ کد همه‌ی آن‌ها
> نوشته، تست سبک شده و روی شاخه کامیت شده است.
