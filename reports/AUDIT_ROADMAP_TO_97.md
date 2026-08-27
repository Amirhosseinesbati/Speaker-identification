# 🔬 گزارش جامع Audit + نقشه‌ی راه رسیدن به رتبه‌ی ۱ (Score > 0.97)

> **یادداشت 2026-08-27 — این سند اکنون historical است.** گزارش مرجع به‌روز، شامل EDA فایل‌به‌فایل، مرور علمی مستقل، ممیزی implementation فعلی، اشکال‌های P0 در sampler/validation/artifact/HPO و roadmap مرحله‌ای، در
> [`DEEP_COMPETITION_DATA_CODE_AUDIT_AND_WINNING_ROADMAP_2026-08-27.md`](DEEP_COMPETITION_DATA_CODE_AUDIT_AND_WINNING_ROADMAP_2026-08-27.md)
> قرار دارد. اعداد یا توصیه‌های این نسخه فقط وقتی معتبرند که در گزارش جدید تأیید شده باشند.

**مسابقه:** IAAA Competition 2026 — Open-Set Speaker Identification
**نسخه:** ۲.۰ (قطعی — بازنویسی کامل پس از بررسی تمام‌و‌کمال ساختار پروژه، زنجیره‌ی MLOps، گزارش‌های قبلی و مستندات سرور)
**تاریخ:** 2026-08-14
**مبنای تحلیل:** خواندن کامل `Competition-Guide/` (PDF رسمی + قوانین + لیست پکیج‌های سرور)، چهار فاز `eda/`، تمام ماژول‌های `src/` (شامل `pipelines/` و `deploy/`)، `submission/`، `scripts/`، `checkpoints/` (بازخوانی محتوای داخلی checkpointها)، `reports/` قبلی، و تاریخچه‌ی git.

> ⚠️ **تصحیحات نسبت به نسخه‌ی ۱.۰:** (۱) مدل‌های فعلی از مسیر **Pipeline/ZenML** آموزش دیده‌اند (با Warmup و Early-Stopping واقعی)، نه `train.py` مستقل — اثبات: ساختار `scheduler_state_dict` داخل checkpointها؛ (۲) بر اساس تأیید کاربر، **سرور GPU دارد** (سند مستقل: «CUDA Version 12.8» در مستندات رسمی) — تحلیل Inference بر این مبنا بازنویسی شده؛ (۳) محدودیت **۱GB برای ZIP** به قیدهای سخت اضافه شد.

---
---

# بخش ۰ — راهنمای خواندن و واژه‌نامه (پیش از شروع)

این گزارش برای خواندنی‌بودن، هر مفهوم فنی را همان‌جا که استفاده می‌شود به زبان ساده توضیح می‌دهد. واژه‌های مرکزی:

| اصطلاح | تعریف ساده |
|---|---|
| **Open-Set Recognition** | دسته‌بندی وقتی در تست ممکن است کلاس‌های «نادیده» بیایند. اینجا: 446 گوینده‌ی شناخته‌شده + هر گوینده‌ی دیگری باید `unknown` تشخیص داده شود. |
| **OOD (Out-of-Distribution)** | همان نمونه‌های «ناشناس»؛ 554 گوینده‌ی تست که هویتشان را نمی‌دانیم و همه در یک کلاس `unknown` جمع می‌شوند. |
| **Macro-F1** | F1 هر کلاس جدا محاسبه و سپس **میانگین ساده** روی 447 کلاس گرفته می‌شود. اثر: کلاس کوچک ۵نمونه‌ای با کلاس ۲۲۷۵نمونه‌ای وزن برابر دارد. |
| **Embedding** | نمایش بُرداری فشرده‌ی صدا (مثلاً 192 عدد) که «هویت صدای» گوینده را رمز می‌کند. Encoder مدلی است که این بردار را می‌سازد. |
| **Centroid** | «مرکز ثقل» embeddingهای یک گوینده: میانگین بردارهای چند فایل آموزشی او. نزدیک‌ترین Centroid = ساده‌ترین classifier ممکن. |
| **ArcFace** | سر دسته‌بندی که به‌جای شباهت ساده، «فاصله‌ی زاویه‌ای + حاشیه (margin)» را بهینه می‌کند تا embeddingهای هم‌گوینده جمع‌تر و ناهم‌گوینده دورتر شوند. استاندارد صنعت تشخیص چهره/گوینده. |
| **TTA (Test-Time Augmentation)** | در تست، به‌جای یک بار، فایل را چند بار (پنجره‌های هم‌پوشان) به مدل می‌دهیم و خروجی‌ها را میانگین می‌گیریم — دقت بالاتر، هزینه‌ی زمان بیشتر. |
| **OOF (Out-of-Fold)** | با K-Fold، هر نمونه دقیقاً یک بار «validation» می‌شود؛ کنار هم چیدن این پیش‌بینی‌ها = مجموعه‌ی بزرگ و بدون نشت برای تیون نهایی. |
| **EMA / SWA** | نگه‌داشتن میانگین متحرک از وزن‌های مدل در طول آموزش؛ معمولاً مدل نهایی پایدارتر و دقیق‌تر می‌شود (تقریباً رایگان). |
| **HPO** | بهینه‌سازی خودکار هایپرپارامترها (اینجا با Optuna: جست‌وجوی بیزی TPE + حذف زودهنگام trialهای بد). |
| **ICE Score** | Impact × Confidence × Ease — امتیاز اولویت‌بندی: هر اقدام از نظر اثر، اطمینان و سهولت نمره می‌گیرد. |
| **LOO (Leave-One-Out)** | ارزیابی منصفانه‌ی Centroid: centroid هر گوینده بدون فایلی که دارد امتحان می‌شود ساخته می‌شود (حذف bias). |
| **Shakeup** | جهش رتبه‌ها هنگام تعویض Public/Private LB — معمولاً به‌خاطر overfit به public یا validation ناپایدار. |

---

# بخش ۱ — خلاصه‌ی مدیریتی

**وضعیت فعلی (اعداد سنددار):**

| شاخص | مقدار | منبع |
|---|---|---|
| بهترین نتیجه‌ی لوکال (هر روش) | **Macro-F1 = 0.9202** | Centroid روی ECAPA فریز + OOD Gate — `eda/Phase3_Embedding_EDA_Report.md` §6 |
| بهترین Ensemble آموزش‌دیده | **0.9093** (روی Val 891تایی) | `data/processed/ensemble_fusion_results.json` |
| بهترین تک‌مدل آموزش‌دیده | CAM++: 0.8932 (Val) / 0.8671 (در checkpoint) | `ensemble_fusion_results.json` / `campp_best.pt` |
| امتیاز رتبه‌ی ۱ لیدربرد | **0.97** | گفته‌ی کاربر |
| **شکاف** | **≈ ۵ امتیاز** | — |
| Submission موفقِ Scoreدار تا امروز | **۰ عدد** (≥۶ تلاش، همه به crash/timeout شکست خورده‌اند) | git history |

**سه یافته‌ی تعیین‌کننده‌ی این Audit:**

1. **گلوگاه اصلی «ظرفیت مدل» نیست — «سیاست آموزش» است.** سه انکودر از چهارتای آموزش‌دیده کاملاً **Frozen** بوده‌اند (فقط Head یاد گرفته) و ECAPA فقط ۲ بلوک آخرش با LR خیلی کوچک باز بوده. سند قاطع: همان embedding فریزشده‌ی ECAPA با یک **Centroid ساده** known-accuracy=**94.98%** می‌گیرد، در حالی که CAM++ «آموزش‌دیده» (head-only روی embedding ثابت) به **91.5%** قانع است. یعنی آموزش Head روی embedding ثابت نه‌تنها چیزی اضافه نکرده، از روش بدون‌آموزش هم عقب‌تر است.
2. **قوی‌ترین سلاح‌ها استفاده نشده‌اند:** (الف) مسیر تصمیم‌گیری Centroid + OOD Gate که در خود پروژه پیاده شده (`src/centroid_baseline.py`) و در فاز ۳ ثابت کرده **+0.22** ارزش دارد، به Submission راه پیدا نکرده؛ (ب) OOD Thresholdهای تیون‌شده‌ی هر checkpoint (0.25–0.35) در Inference نهایی اصلاً اعمال نمی‌شوند — argmax خام؛ (ج) WavLM-Large به‌خاطر ترس از سقف ۱GB کنار گذاشته شده، در حالی که با بسته‌بندی bf16/fp16 و حذف وزن‌های پایه، در بودجه جا می‌شود (محاسبه در ضمیمه C).
3. **یک باگ مفهومی در آموزش، «ضریب ضرب داده» را خفه کرده:** در Training، ۳ پنجره‌ی رندوم هر فایل پیش از محاسبه‌ی Loss **با هم میانگین** می‌شوند (`train.py::forward_multi_window`) — یعنی به‌جای ۳ نمونه‌ی آموزشی مستقل، مدل ۱ پیش‌بینی «کیسه‌ای» per فایل می‌گیرد. در مسئله‌ی extreme few-shot (۵ فایل per گوینده!) این یعنی دور انداختن بزرگ‌ترین مزیت دیتاست: فایل‌های ۶۰ ثانیه‌ای.

**حکم نهایی: Refactor هدفمند** — زیرساخت (دیتاپایپلاین، EDA، انکودرها، MLOps، پکیجینگ) سالم و ارزشمند است؛ آنچه باید عوض شود: ① Full Fine-Tune انکودرها، ② آموزش per-window (به‌جای میانگین)، ③ Decision Layer هندسی (Centroid + Gate) در Inference، ④ Validation چند-Fold (OOF)، ⑤ تبدیل زنجیره‌ی MLOps به «موتور آزمایش» برای اجرای سناریوها. سقف واقع‌بینانه‌ی این مسیر: **0.95–0.975**.

**۳ اقدام فوری** (جزئیات در بخش ۱۰): ① تولید و ذخیره‌ی OOF/Val probs + ساخت Centroid از checkpointهای موجود و Ship کردن تصمیم Centroid+Gate (بدون Retrain: انتظار 0.9093 → ~0.92-0.935)؛ ② پاک‌سازی لیبل + طراحی 3-Fold OOF؛ ③ رفع باگ میانگین‌پنجره در آموزش و شروع Full Fine-Tune با ۲ انکودر برتر.

---

# بخش ۲ — قواعد بازی (استخراج دقیق از اسناد رسمی)

## ۲.۱ جدول قواعد

| مورد | مقدار دقیق | منبع |
|---|---|---|
| **تسک** | Open-Set Speaker ID: هر chunk صوتی → یکی از 447 کلاس (446 known + `unknown`) | PDF ص۱ |
| **متریک** | **Macro-F1 روی هر 447 کلاس**؛ برچسب نهایی = argmax روی خروجی مدل | PDF ص۴-۵ |
| **جهت بهینه‌سازی** | بالاتر بهتر؛ هدف ما: **> 0.97** | — |
| **داده‌ی Train** | 4,529 فایل: 2,254 known (446 گوینده؛ ۹۸.۴٪ دقیقاً ۵ فایل!) + 2,275 unknown (554 گوینده، بدون هویت) | Phase0 EDA |
| **داده‌ی Test** | **3,604 فایل مخفی** (~50٪ صدای هر شخص)؛ خروجی: CSV با `audio_file,speaker_id` (`unknown` مجاز) | `Speaker.md` |
| **قرارداد اجرا** | `python submission.py --data-dir <dir> --predictions-file-path <csv>` | `submissionforleaderbord.txt` |
| **محیط سرور** | Python **3.12**، **CUDA 12.8** (یعنی GPU دارد)، پکیج‌های از پیش نصب‌شده طبق `leaderbordpakage.txt` | مستندات رسمی |
| **قیدهای کاربر** | GPU=3090، بدون اینترنت در Inference، بودجه‌ی Inference ≈ **۲۰ دقیقه**، بودجه‌ی Train نامحدود، **ZIP ≤ 1GB** | پرامپت کاربر |
| **داده/مدل خارجی** | ✅ مجاز: pretrained models، external speech datasets، pretraining/SSL — ❌ ممنوع: داده‌ی لیبل‌دار شامل گویندگان ارزیابی | PDF ص۵ |
| **Ensemble** | ✅ صریحاً مجاز | PDF ص۶ |
| **ریسک DQ** | استفاده از لیبل eval، دسترسی به داده‌ی مخفی، نقض fair-play، و عملاً: **Crash یا Timeout در اجرای submission** | PDF ص۶ + تاریخچه |

## ۲.۲ تحلیل `leaderbordpakage.txt` — نقشه‌ی واقعی محیط ارزیابی

این فایل `pyproject.toml` محیط سرور است و اطلاعات استراتژیک مهمی دارد:

| بسته‌ی موجود در سرور | نسخه | پیامد برای ما |
|---|---|---|
| torch / torchaudio | ≥2.10 | ✅ نسخه‌ی جدید؛ AMP/bf16 پشتیبانی می‌شود |
| speechbrain | ≥1.0.3 | ✅ لودر ECAPA کار می‌کند (با patchهای موجود در `encoders.py`) |
| nemo-toolkit[asr] | ≥2.7.3 | ✅ TitaNet قابل لود است |
| modelscope | ≥1.38.1 | ⚠️ موجود، ولی **وابستگی‌های فرعی‌اش (addict/easydict/simplejson/yapf) در لیست نیست** — به همین دلیل `submission/vendor/` ساخته شد. درست است، نگهش دارید. |
| transformers | ≥4.57 | ✅ WavLM قابل لود است |
| **faiss-cpu** | ≥1.14.3 | ✅ **FAISS OOD Detector در سرور قابل استفاده است!** (کدش در `src/ood_detector.py` موجود ولی Ship نشده) |
| librosa / soundfile / audiomentations | — | ✅ دیکد/پردازش صوت |
| onnxruntime / onnx / tensorrt | — | ✅ مسیر شتاب‌دهی Inference باز است |
| optuna / lightgbm / xgboost / catboost | — | برای **Stacking/Calibration سبک** در صورت نیاز |
| pydantic / pydantic-settings | — | ✅ برای Config Schema قابل استفاده است |
| CUDA Version | **12.8** | ✅ تأیید GPU در سرور (با تأیید کاربر هم‌خوان) |

**نکته‌ی مهم درباره‌ی Timeoutهای قبلی:** دو Timeout ثبت‌شده (۹۲ و ~۱۷۰ دقیقه) با وجود GPU رخ داده‌اند. با توجه به تأیید کاربر («سیستم GPU داره و احتمالا یک جای کار ما میلنگه»)، محتمل‌ترین علت‌ها در سمت ماست: (الف) حلقه‌ی فایل‌به‌فایل batch-1 با ۳ مدل × ۸ پنجره = ~۸۶هزار forward کوچک به‌جای batching واقعی؛ (ب) احتمال لود نشدن CUDA در venv ارزیابی (مثلاً wheel اشتباه torch) — خط `[diag] cuda_avail=...` که در `inference.py:178-197` هست دقیقاً برای همین تشخیص گذاشته شده و باید در لاگ آینده خوانده شود. در این گزارش، مطابق دستور کاربر، Inference را «مهم ولی حل‌شدنی با مهندسی» فرض می‌کنیم و تمرکز اصلی بر معماری/امتیاز است.

## ۲.۳ نکته‌ی قراردادی طلایی

PDF می‌گوید مدل «توزیع احتمال 447تایی» می‌دهد و ارزیاب argmax می‌گیرد؛ ولی در عمل **خودِ `submission.py` ما CSV نهایی (هارد لیبل) را می‌سازد**. پس قاعده‌ی تصمیم (Decision Rule) کاملاً در اختیار ماست: Gate، Calibration، λ_unknown، فیوژن با Centroid — همه قانونی و همه فعلاً استفاده‌نشده‌اند.

---

# بخش ۳ — اطلس کامل ساختار پروژه (Inventory + ارزیابی کیفی)

| مسیر | نقش | وضعیت | یادداشت Audit |
|---|---|---|---|
| `Competition-Guide/` | اسناد رسمی (PDF، Speaker.md، پکیج سرور، نمونه submission) | ✅ کامل | قلب قواعد بازی |
| `data/raw/` | 4,529 mp3 + `labels.csv` (4530 خط) | ✅ سالم | — |
| `data/processed/` | WAVهای 16kHz، `cleaned_labels.csv`، `split_report.json`، نتایج فیوژن، cache embeddingهای ECAPA (`embeddings_*.npy`) | ⚠️ | cache فقط برای ECAPA موجود؛ OOF probs ذخیره نمی‌شوند |
| `eda/` | ۴ گزارش + JSONهای خلاصه (فاز ۰ تا ۳) | ✅ عالی | فاز ۳ (LOO unbiased) ارزشمندترین سند پروژه است |
| `src/data_pipeline.py` | تمیزکاری، split بدون نشت، Dataset با پنجره‌بندی، Balanced Sampler | ✅ خوب | اشکالات: val_per_known=1 ثابت؛ MixUp غیرفعال؛ Aug هاردکد |
| `src/encoders.py` | ۵ انکودر (ECAPA/CAM++/ERes2NetV2/TitaNet/WavLM) + patchهای سازگاری آفلاین | ✅ خوب | بلوغ بالا؛ سند شکست‌های تاریخی |
| `src/sv_arch.py` | پیاده‌سازی وندورشده‌ی ERes2NetV2 | ✅ | — |
| `src/heads.py` | OODHead / LinearHead / **ArcFaceHead** | ✅ | Sub-center ArcFace ندارد |
| `src/model.py` | `TwoHeadedSpeakerModel` + فیوژن p[0]=σ(ood), p[i]=(1-p[0])·softmax | ✅ | — |
| `src/model_factory.py` | ساخت مدل از Config | ✅ | — |
| `src/pooling.py` | identity / statistical / attentive | ✅ | — |
| `src/train.py` | حلقه‌ی آموزش **مستقل** | ⚠️ **موازی‌کاری خطرناک** | نسخه‌ی ضعیف‌تر: بدون Warmup/Early-Stop (pipeline دارد)؛ باگ میانگین‌پنجره از اینجا آمده |
| `src/pipelines/steps.py` | حلقه‌ی آموزش **ZenML** (۵ Step) | ✅ کامل‌تر | Warmup 3ep + CosineWarmRestarts + Early-Stop + MLflow + persist ood_threshold — **checkpointهای فعلی محصول همین مسیرند** |
| `src/pipelines/run_pipeline.py` | Orchestrator + اتصال DagsHub/MLflow | ✅ | فقط single-config, single-run |
| `src/metrics.py` | Macro-F1 دقیق مسابقه، فیوژن احتمالات، calibrate_temperature | ✅ عالی | Single source of truth برای متریک |
| `src/centroid_baseline.py` | Centroid + LOO + **multi_encoder_centroid_ensemble** | ✅ ولی استفاده‌نشده | بهترین عدد پروژه (0.9202) از اینجاست؛ نسخه‌ی چندانکودری‌اش هرگز اجرا نشده |
| `src/ood_detector.py` | FAISS k-NN OOD + ترکیب با head | ✅ ولی Ship نشده | faiss-cpu در سرور موجود است! |
| `src/ensemble.py` | ۶ روش فیوژن + grid search + LearnedFusion | ✅ | LearnedFusion روی Val کوچک فروپاشید (0.266) — آموزش: no-MLP-on-small-val |
| `src/ensemble_calibrate.py` | جمع‌کردن Val logits هر مدل + فیوژن + temperature | ✅ | خروجی probs ذخیره نمی‌کند (هر بار re-compute) |
| `src/audio_preprocessing.py` | تبدیل MP3→WAV | ✅ | — |
| `src/mlflow_helper.py` | MLflowTracker مستقل از ZenML (snapshot کد+کانفیگ) | ✅ | — |
| `src/deploy/deploy.py` | اجاره‌ی خودکار Vast.ai + تزریق env + self-destruct | ✅ عالی | انتخاب GPU/انکودر/freeze از UI propagate می‌شود |
| `src/deploy/deploy_app.py` | **Streamlit UI چهار تبه**: Config / Cloud / Local / Analysis | ✅ خوب | ⚠️ فقط single-experiment؛ بدون صف/ماتریس آزمایش، بدون HPO |
| `scripts/build_submission.py` | ساخت پکیج ZIP (حذف ماژول‌های اضافی، vendor deps، pruning وزن‌ها) | ✅ | centroids را (هنوز) نمی‌سازد/ship نمی‌کند |
| `scripts/verify_submission.py` | **Replay کامل لیدربرد** روی ZIP (extract → اجرا از cwd متفاوت → چک سکوت → اعتبار CSV) | ✅ عالی | عادت خوب مهندسی؛ قبل از هر آپلود اجباری باشد |
| `scripts/inspect_checkpoints.py`, `clean_corrupted.py`, `convert_mp3_to_wav.py`, `download_all_weights.py`, `phase4_integration.py`, `minimal_ensemble.py` | ابزارهای جانبی | ✅ | — |
| `submission/` | پکیج نهایی: `submission.py` (entry)، `inference.py` (score_ensemble)، `src/` (کپی mirror)، `checkpoints/` (۳ مدل)، `weights/` (۳ انکودر)، `vendor/`، `ensemble_fusion_weights.json` | ✅ کار می‌کند | تصمیم = argmax خام؛ بدون Centroid/Gate |
| `submission_leaderboard.zip` | 386MB — آخرین بیلد | ⚠️ | هنوز Score موفق نگرفته |
| `checkpoints/` | ۴ مدل آموزش‌دیده + خلاصه‌ها | ✅ | اعداد در بخش ۴ |
| `weights/` | وزن‌های پایه‌ی ۵ انکودر (محلی) | ✅ | wavlm_large = 1.28GB |
| `pretrained_models/spkrec-ecapa-voxceleb/` | کش SpeechBrain | ✅ | — |
| `tests/` | فقط `test_audio_preprocessing.py` | ⚠️ ناقص | پوشش تست بسیار کم برای پروژه‌ای به این حساسی |
| `configs/` | `default_config.yaml` + `inference_config.yaml` | ⚠️ | تک‌فایل مشترک؛ بدون experiment profiles؛ چند dead key |
| `.dvc/` | DVC برای versioning داده | ⚠️ بلااستفادهٔ عملی | می‌تواند MUSAN/RIR را مدیریت کند |
| `setup_vast.sh` | Bootstrap سرور Vast (نصب، clone، اجرای pipeline) | ✅ | — |
| `.zcode/plans/` | پلن‌های جلسات قبلی دستیار | — | — |

**جمع‌بندی اطلس:** پروژه از نظر «پوشش زیرساختی» کامل‌تر از حد انتظار یک تیم مسابقه‌ای است (EDA چهارفازی، MLOps واقعی، پکیجینگ+Verifier). نقص‌های ساختاری سه‌تاست: ① دو حلقه‌ی آموزش متفاوت (`train.py` vs `steps.py`)، ② نبود لایه‌ی «مدیریت آزمایش» (named configs / صف / HPO / OOF) بالای این زیرساخت، ③ نبود مرحله‌ی «Decision Tuning» در pipeline (خروجی آموزش = checkpoint؛ ولی τ/λ/α/T و centroids هیچ‌جا ساخته و نگهداری نمی‌شوند).

---

# بخش ۴ — نقشه‌ی End-to-End فعلی (سه مسیر + زنجیره‌ی MLOps)

## ۴.۱ مسیر آموزش (آنچه واقعاً checkpointها را ساخت — ZenML Pipeline)

```
MP3 (4,529) ──convert_audio──▶ WAV 16kHz mono
        │
        ▼  prepare_data (steps.py:162)
تمیزکاری: حذف ۷۰ corrupted (<1s) + تشخیص ۶۹ duplicate (۹ گروه MD5)
Split:   ۱ فایل per known speaker → Val | ۲۰٪ unknown → Val
         Train=3,568 (1,786 known + 1,782 unknown) | Val=891 (446+445)
         ⚠ duplicateها فقط از Val کنار گذاشته می‌شوند؛ در Train می‌مانند
         ⚠ گروه ۴۶تایی با ۴ لیبل متناقض → در Train باقی است
        │
        ▼  train_model (steps.py:285)
Dataset: ۳ کراپ رندوم ۸ثانیه‌ای per فایل + Aug موج‌محور (Gaussian/Pitch±1/Stretch/Gain/Polarity/Shift)
Sampler: هر batch = ۵۰٪ OOD + ۵۰٪ known
Model:   Encoder ─▶ Pooling ─┬─▶ OOD Head (BCE)         ← 0.3 وزن
                             └─▶ ArcFace 192d (m=0.4,s=30) ← 0.7 وزن، Focal γ=2 + LS=0.1
Freeze:  campp/eres2net/titanet = کاملاً Frozen | ecapa = ۲ بلوک آخر باز (lr=1e-5)
Optim:   AdamW (head 1e-4) + LinearLR warmup(3ep) → CosineAnnealingWarmRestarts(T_0=10)
         Early-Stop روی Val Macro-F1 (patience=15) | AMP(fp16) | grad-clip 5 (OOD: 1)
         ⚠ باگ: ۳ پنجره‌ی هر فایل پیش از Loss «میانگین» می‌شوند → ۱ سیگنال per فایل
        │
        ▼  evaluate_model (steps.py:587)
Val Macro-F1 (argmax) + تیون ood_threshold (روی binary F1!) + persist در checkpoint
        │
        ▼
checkpoints/<enc>_best.pt  →  campp 0.8671@ep117 | ecapa 0.8477@ep127
                              eres2net 0.8085@ep138 | titanet 0.8426@ep155
(اثبات مسیر: scheduler ذخیره‌شده در checkpoint = SequentialLR[LinearLR, CosineWarmRestarts])
```

## ۴.۲ مسیر فیوژن آفلاین (ensemble_calibrate.py)

```
۴ مدل ──▶ Val logits (۸ پنجره، میانگین logit) ──▶ 447-probs
──▶ ۶ روش فیوژن:  average 0.8873 | geometric 0.8899 | rank 0.8731 | max_prob 0.8910
                  weighted(grid) 0.9093 ✅ | learned_mlp 0.266 ❌ (فروپاشی)
──▶ انتخاب شده: weighted_average با وزن‌های [0.5 campp, 0.2 eres2net, 0.3 titanet]
──▶ خروجی: ensemble_fusion_weights.json
⚠️ ECAPA با وزن ۰ حذف شد — در حالی‌که قوی‌ترین Centroid-Performer پروژه است
⚠️ probs هیچ‌جا ذخیره نمی‌شوند → هر آزمایش فیوژن = یک Inference کامل مجدد
```

## ۴.۳ مسیر Inference نهایی (submission/)

```
submission.py --data-dir ... ──▶ inference.py::score_ensemble
  ۱. لود ۳ checkpoint (هرکدام config+class_map داخلی خودشان) — آفلاین کامل
  ۲. per فایل: دیکد ۱ بار → ۸ پنجره‌ی ۸ثانیه‌ای (hop 50%) → (8,1,T) batch →
     per مدل: predict_proba → میانگین احتمال پنجره‌ها
  ۳. فیوژن وزنی [0.5, 0.2, 0.3] → normalize → argmax خام
  ۴. نگاشت index→UUID (۰ → "unknown") → CSV
✗ ood_threshold تیون‌شده (0.25-0.35) اعمال نمی‌شود
✗ Centroid/FAISS وجود ندارد (build_submission.py آن ماژول‌ها را حذف می‌کند)
✗ Temperature (T=1.8 تیون شده بود) اعمال نمی‌شود
```

## ۴.۴ زنجیره‌ی MLOps (دارایی بزرگ و نیمه‌بهره‌برداری‌شده)

```
┌─────────────────────────────────────────────────────────────────┐
│  Streamlit UI (deploy_app.py)                                   │
│  ⚙️ Config: encoder / freeze-mode / pooling / ArcFace / audio / │
│      training / loss — ذخیره در default_config.yaml (تک‌فایل!)   │
│  ☁️ Cloud: انتخاب GPU(3090/3060/A4000) + stage + disk           │
│      ─▶ deploy.py ─▶ Vast.ai (ارزان‌ترین آفر) ─▶ setup_vast.sh   │
│      ─▶ clone repo + نصب + اجرای pipeline + self-destruct        │
│      ─▶ استریم زنده‌ی لاگ + دکمه‌ی Destroy                        │
│  💻 Local: اجرای stageها با لاگ زنده و Stop                      │
│  🧪 Analysis: اجرای eda_embeddings / centroid_baseline /         │
│      ensemble_calibrate / submission inference از داخل UI         │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
   ZenML Pipeline (۵ step) ─▶ MLflow/DagsHub (params/metrics/artifacts
   + code snapshot + best checkpoint) per epoch
               ▼
   scripts/build_submission.py ─▶ submission/ ─▶ verify_submission.py
   (replay کامل لیدربرد) ─▶ submission_leaderboard.zip
```

**ارزیابی:** این زنجیره برای «یک آزمایش در هر لحظه» عالی است، ولی برای «کمپین چند ده‌تایی آزمایش» که بردن 0.97 لازم دارد، چهار چیز کم دارد: ① پروفایل‌های نام‌دار آزمایش (الان UI همان یک فایل config را بازنویسی می‌کند)، ② صف/ماتریس اجرا (encoder × recipe × seed × fold)، ③ HPO (Optuna)، ④ مرحله‌ی Decision-Tuning و Ensemble داخل خود Pipeline. طرح ارتقا در **بخش ۱۷**.

## ۴.۵ جدول «هست / نیست / ریسک» برای هر مرحله

| مرحله | چه چیزی هست ✅ | چه چیزی نیست ❌ | ریسک |
|---|---|---|---|
| Cleaning | حذف corrupted + شناسایی duplicate | حذف لیبل‌های متناقض از Train | نویز ArcFace/OOD |
| Split | بدون نشت بایت‌ای | K-Fold/OOF؛ Val=۱ فایل per speaker | واریانس تصمیم‌ها |
| Windows | ۳ کراپ train + ۸ کراپ eval | **آموزش per-window** (الان میانگین می‌شوند) | اتلاف ضریب ضرب داده |
| Augmentation | ۶ افکت موج‌محور | RIR/MUSAN/SpecAugment/Codec؛ config-driven | تعمیم به شرایط تست |
| Model | ۵ انکودر + ArcFace + OOD head | Full FT (۳تایش Frozen)؛ Sub-center؛ WavLM | سقف embedding |
| Loss | Focal+BCE دو سر | objective متریک‌محور؛ تعادل known/unknown | انحراف از متریک |
| Training | warmup+restarts+earlystop (فقط در pipeline) | EMA/SWA؛ حذف حلقه‌ی دوم (train.py) | ناسازگاری runها |
| Metrics | Macro-F1 دقیق + per-class | ذخیره‌ی OOF probs per run | re-compute پرهزینه |
| Decision | threshold در checkpoint ذخیره می‌شود | اعمال واقعی در Submission | امتیاز مجانی از دست می‌رود |
| Ensemble | ۶ روش + grid | فیوژن score-space (cosine)؛ OOF-based | Overfit به ۸۹۱ نمونه |
| Packaging | offline + vendor + verifier | ship centroids؛ strip optimizer؛ برنامه‌ی حجم ۱GB | DQ/Timeout |
| MLOps | UI + Vast + ZenML + MLflow | experiment matrix + HPO + decision-tune step | سرعت تکرار آزمایش |

---

# بخش ۵ — Root Cause Analysis (تشخیص افتراقی با سند)

> دسته‌بندی پرامپت: Metric Mismatch / Validation / Underfitting / Overfitting / Data / Optimization / Post-processing / Backbone / Ensemble-TTA. برای هر مورد: شاهد + شدت + Δmetric قابل بازیابی.

## R1 — سیاست Frozen-Encoder: Underfitting ساختاری ⭐ (ریشه‌ی اصلی)

| | |
|---|---|
| **شرح** | سه انکودر (campp/eres2net/titanet) کاملاً Frozen و ECAPA نیمه‌فریز (۲ بلوک، lr=1e-5) آموزش دیده‌اند. Head روی embedding ثابت فقط «بازترکیب جهت‌های موجود» است؛ نمی‌تواند manifold جدید برای دامنه‌ی مسابقه (لهجه‌ها/کدک/محیط) بسازد. |
| **شاهد عددی** | ① CAM++ head-only: `known_acc=0.9148` (Val) در برابر **همان‌جور embedding فریزشده‌ی ECAPA + Centroid ساده: 0.9498** (LOO، `phase3_embedding_summary.json`). ② `default_config.yaml:87-98` → `freeze_encoder: true` ×3. ③ بهترین Macro-F1 آموزش‌دیده (0.9093) < بیس‌لاین بدون آموزش (0.9202). |
| **شدت** | **۵/۵** |
| **Δ قابل بازیابی** | **+0.03 تا +0.05** (Full FT معمولاً خطای باقی‌مانده‌ی speaker verification را در دامنه‌ی هدف نصف می‌کند) |

## R2 — باگ مفهومی: آموزش با «میانگین‌گیری پنجره‌ها» ⭐ (یافته‌ی جدید نسخه‌ی ۲)

| | |
|---|---|
| **شرح** | `SpeakerDataset._train_windows` برای هر فایل ۳ کراپ رندوم می‌سازد، اما `forward_multi_window` (`train.py:354-364`) لاجیت‌های ۳ پنجره را **میانگین می‌گیرد و بعد** Loss حساب می‌شود. یعنی مدل per فایل ۱ سیگنال گرادیان می‌گیرد، نه ۳ تا. در مسئله‌ای که هر گوینده فقط ۴-۵ فایل دارد، این یعنی دور ریختن ضریب ضرب داده (۴ فایل × ۶۰s باید ~۳۰ پنجره‌ی مستقل می‌داد). |
| **شاهد** | `data_pipeline.py:564-580` + `train.py:325-364`؛ توضیح docstring خودش هم می‌گوید برای VRAM این‌طوری شده (GPU محلی 6GB) — ولی روی 3090 همان باقی مانده. |
| **شاهد عددی تکمیلی** | Phase-3 با ۸ پنجره‌ی «مستقل» per فایل known-acc=94.98% گرفت؛ شبکه با میانگین‌پنجره 87-91%. |
| **شدت** | **۵/۵** |
| **Δ قابل بازیابی** | **+0.01 تا +0.03** (آموزش per-window: flatten به (B×W,1,T) با Loss per پنجره — روی 3090 هزینه‌ی محاسباتی یکسان، سیگنال ۳ برابر) |

## R3 — Decision Layer استفاده‌نشده: Post-processing ⭐

| | |
|---|---|
| **شرح** | Submission فعلی argmax خام است. سه اهرم آماده و استفاده‌نشده: ① OOD Gate (threshold تیون‌شده‌ی هر checkpoint در خود checkpoint ذخیره شده ولی در inference اعمال نمی‌شود)؛ ② Temperature (T=1.8 روی Val بهتر از T=1 بود ولی wire نشده)؛ ③ مسیر Centroid+Gate کدش آماده است و Ship نشده. |
| **شاهد عددی** | فاز ۳: Gate روی مسیر centroid = **0.7013 → 0.9202 (+0.219)**؛ `ensemble_fusion_results.json`: temperature calibration 0.8850→0.8884؛ `submission/inference.py` هیچ منطق threshold ندارد؛ `build_submission.py:140-149` ماژول‌های `centroid_baseline/ood_detector/metrics` را از پکیج حذف می‌کند. |
| **شدت** | **۵/۵** (ارزان‌ترین امتیاز ممکن) |
| **Δ قابل بازیابی** | **+0.01 تا +0.03** روی مدل‌های فعلی؛ بیشتر روی مدل‌های Fine-Tune‌شده |

## R4 — Validation ناپایدار (۱ فایل per speaker)

| | |
|---|---|
| **شرح** | Val = 446 known (دقیقاً ۱ فایل!) + 445 unknown. F1 هر کلاس known عملاً ∈ {0,1}. انتخاب epoch، تیون threshold، و grid-search وزن‌های فیوژن همه روی همین ۸۹۱ نمونه‌ی تک‌seed. |
| **شاهد** | `data_pipeline.py:140` (`val_per_known=1`) + `steps.py:194-201`؛ سند فاجعه: `learned_mlp` fusion با Val-F1=0.0018 فروپاشید (`ensemble_fusion_results.json:88-95`). ضمناً اختلاف حل‌نشده: campp در checkpoint=0.8671 ولی در ensemble_fusion=0.8932 (همان val، همان مدل!) → یعنی مسیر ارزیابی هم تکرارپذیرِ قابل‌اعتماد مستند نشده. |
| **شدت** | **۴/۵** |
| **Δ قابل بازیابی** | غیرمستقیم بزرگ: جلوگیری از ۲-۴ تصمیم غلط ≈ نجات چند امتیاز در Private LB |

## R5 — WavLM-Large بلااستفاده (با راه‌حل بودجه‌ی ۱GB)

| | |
|---|---|
| **شرح** | قوی‌ترین انکودر (317M پارامتر، 1024-d، SOTA روی بسیاری از benchmarkهای speaker پس از FT) به‌خاطر حجم ۱.۲۸GB کنار گذاشته شده. اما: (الف) checkpointهای Fine-Tune‌شده خودکفا‌اند (شامل وزن‌های encoder) — فقط باید loader را طوری کرد که بدون `weights/wavlm_large/` از روی checkpoint بسازد؛ (ب) با ذخیره‌ی **bf16** (~630MB) یا int8 (~320MB) در بودجه‌ی ۱GB جا می‌شود؛ (ج) اگر هیچ‌کدام نشد → Distillation به ECAPA/CAM++. |
| **شاهد** | `weights/wavlm_large/` موجود؛ هیچ `wavlm_best.pt` نیست؛ `build_submission.py` آن را prune می‌کند؛ Phase-6: در fp16، forward خروجی NaN داد → راه‌حل: bf16 یا mixed. |
| **شدت** | **۴/۵** |
| **Δ قابل بازیابی** | **+0.01 تا +0.03** تک‌مدل؛ بیشتر در Ensemble |

## R6 — دو حلقه‌ی آموزش متفاوت (Boronزی معماری)

| | |
|---|---|
| **شرح** | `src/train.py` (بدون warmup/early-stop، Cosine ساده) و `src/pipelines/steps.py::train_model` (warmup 3ep + CosineWarmRestarts + early-stop + MLflow) دو مسیر موازی‌اند. checkpointهای فعلی از مسیر pipeline آمده‌اند (اثبات: `_schedulers=[LinearLR, CosineAnnealingWarmRestarts]` در scheduler_state_dict). وجود نسخه‌ی ضعیف‌ترِ بلااستفاده = تله برای آینده (هر کس ممکن است دوباره از آن استفاده کند). ضمناً `warmup_steps: 500` در config **dead key** است (pipeline warmup را هاردکد ۳ epoch کرده). |
| **شدت** | **۳/۵** |
| **Δ** | غیرمستقیم (تکرارپذیری + اعتماد) |

## R7 — نویز لیبل در Train

| | |
|---|---|
| **شرح** | گروه duplicate ۴۶تایی با MD5 یکسان ولی **۴ لیبل متناقض** (۳ گوینده + unknown) در Train باقی است → همان ورودی بایت-یکسان ۴ supervision متضاد می‌گیرد (هم ArcFace را خراب می‌کند هم OOD head را). + گروه دوم ۲تایی متناقض. + ۷۰ فایل corrupted (درست drop شده‌اند). |
| **شاهد** | `split_report.json` → `duplicate_groups.groups[0]` (`conflicting_labels: true`, n=46). |
| **شدت** | **۲/۵** (واقعی ولی کوچک) |
| **Δ** | +0.001 تا +0.003 (و حذف ریسک متمرکز روی ۳ گوینده) |

## R8 — Augmentation: ناکافی + هاردکد + MixUp خاموش

| | |
|---|---|
| **شرح** | فقط ۶ افکت waveform-level ملایم؛ بدون RIR (پژواک اتاق — مهم‌ترین aug برای generalization صدا)، بدون MUSAN (نویز واقعی)، بدون SpecAugment، بدون Codec-simulation (تست mp3 است!). همه‌ی پارامترها هاردکد (`data_pipeline.py:452-465`) → در HPO قابل sweep نیستند. MixUp پیاده شده ولی `mixup_alpha` همیشه ۰. |
| **شدت** | **۳/۵** |
| **Δ** | +0.005 تا +0.015 |

## R9 — جزئیات Loss/Metric Alignment

| | |
|---|---|
| **شرح** | ① OOD threshold موقع train با **binary F1** تیون می‌شود (`steps.py:684-701`) در حالی که متریک مسابقه Macro-F1 است — objective اشتباه (گزارش قبلی هم گرفته بود، هنوز هست). ② ترکیب Focal(γ=2) + LabelSmoothing(0.1) + ArcFace سه regularizer تداخل‌دار بدون ablation. ③ `ood_batch_ratio=0.5` یعنی نیمی از گرادیان‌ها به ۱ کلاس از ۴۴۷ کلاس می‌رسد؛ با توجه به وزن ۹۹.۸٪ known در متریک، احتمالاً 0.3-0.35 بهتر است. ④ train logit-avg vs inference prob-avg (ناهماهنگی ریاضی: mean(σ(x))≠σ(mean(x))). |
| **شدت** | **۳/۵** |
| **Δ** | +0.003 تا +0.01 مجموعاً |

## R10 — Ensemble: وزن‌دهی روی Val کوچک + حذف ECAPA

| | |
|---|---|
| **شرح** | وزن‌های [0.5, 0.2, 0.3] از grid روی ۸۹۱ نمونه؛ ECAPA (قوی‌ترین centroid) وزن ۰ گرفت. هیچ فیوژنی بین «head-probs» و «centroid-scores» انجام نشده (تابعش موجود: `evaluate_fusion` با α sweep). نتایج multi-encoder centroid هرگز تولید نشده (`centroid_ensemble_results.json` وجود ندارد). |
| **شدت** | **۳/۵** |
| **Δ** | +0.005 تا +0.015 |

## R11 — نبود HPO و لایه‌ی مدیریت آزمایش

| | |
|---|---|
| **شرح** | Optuna در پروژه نیست (در لیست پکیج‌های سرور هست ولی سرور برای eval است). UI فقط یک config را بازنویسی می‌کند؛ نه named experiment، نه صف، نه sweep. نتیجه: هایپرپارامترهای فعلی (m=0.4, s=30, lr=1e-4/1e-5, LS=0.1, ood_w=0.3...) اساساً حدس اولیه‌اند و هرگز جست‌وجو نشده‌اند. |
| **شدت** | **۳/۵** |
| **Δ** | +0.005 تا +0.015 (پس از فیکس شدن پارادایم) |

## R12 — مهندسی اجرا/پکیج (تاریخچه‌ی ۶ شکست LB)

| | |
|---|---|
| **شرح** | سابقه: crash وابستگی‌های modelscope، فرمت CSV اشتباه، نویز stdout، ۲× Timeout. بسیاری فیکس شده‌اند (vendor، سکوت کامل، verifier) ولی: checkpointها optimizer_state دارند (حجم اضافی در ZIP)، batching واقعی cross-file وجود ندارد (file-loop با ۸-prompt batch)، و مسیر تصمیم جدید (centroids) هنوز در بیلد اسکریپت قلاب نشده. |
| **شدت** | **۴/۵** (به‌خاطر حساسیت DQ) |
| **Δ** | حفاظتی (جلوگیری از صفر شدن) |

## جمع‌بندی Root Cause

```
                   سقف فعلی (اثبات‌شده)
   Frozen-head ensemble:        0.9093 ──────────────┐
   Frozen centroid + gate:      0.9202 ──────────┐   │
                                                 ▼   ▼
   ریشه‌ها:  R1 (frozen) + R2 (میانگین‌پنجره) + R3 (تصمیم خام) + R4 (val ناپایدار)
   پتانسیل بازیابی تجمعی (غیرخطی):  0.9093 ──▶ ~0.95-0.975
```

---

# بخش ۶ — Gap Analysis نسبت به SOTA (راه‌حل ما vs الگوی برنده)

| محور | راه‌حل فعلی ما | الگوی برنده در این خانواده (Speaker-ID / Open-Set / Few-Shot) | فاصله و دلیل |
|---|---|---|---|
| **Encoder** | ۴ انکودر pretrained؛ ۳ تای Frozen؛ head-only | **Full/Partial FT با LR تفکیک‌شده** روی داده‌ی هدف؛ Backboneهای رایج برنده: ECAPA-TDNN، CAM++، ResNet-based، WavLM/Hubert FT | R1 — embedding به دامنه‌ی مسابقه تطبیق نیافته |
| **نمونه‌ی آموزشی** | ۱ سیگنال per فایل (میانگین ۳ پنجره) | **Per-window training**: هر کراپ = یک نمونه با لیبل فایل؛ effective dataset ×۳-۱۰ | R2 — باگ مفهومی |
| **Few-Shot Head** | ArcFace تک‌مرکزی (۱ center per speaker) | **Sub-center ArcFace (k=2..3)** + حاشیه‌ی ملایم‌تر؛ در inference: **Nearest-Centroid روی embedding** (پایدارتر از softmax در few-shot) | R1/R3 |
| **OOD** | Head دودویی BCE (در inference بلااستفاده) | **Score-based rejection هندسی**: max-cosine به centroids + τ تیون‌شده **مستقیم روی Macro-F1**؛ ترکیب چند سیگنال (head + cosine + energy) | R3/R9 |
| **Validation** | تک‌split، ۱ فایل per speaker | **K-Fold speaker-aware + OOF**؛ گزارش std بین seedها؛ همه‌ی تیون روی OOF | R4 |
| **Augmentation** | ۶ افکت waveform ملایم، هاردکد | **RIR + MUSAN + SpecAugment + Codec-roundtrip**، config-driven و قابل sweep | R8 |
| **Training Recipe** | AdamW + warmup + warm-restarts + early-stop (pipeline) | + **EMA/SWA**، label-smoothing متعادل، Longer schedule، Gradient accumulation برای بچ بزرگ | — |
| **Ensemble** | Weighted-avg روی prob (وزن از Val) | فیوژن چندسطحی **score-space (cosine) + prob-space**، تنوع architecture × seed × fold، وزن از OOF | R10 |
| **Decision Tuning** | انجام نمی‌شود | مرحله‌ی مجزا در pipeline: ساخت centroids + grid روی (τ, α, λ, T, وزن‌ها) با متریک واقعی | R3 |
| **External Data** | فقط وزن‌های pretrained (VoxCeleb داخل خود مدل‌ها) | FT میانی روی VoxCeleb1+2 با objective مشابه — **مجاز صریح در PDF** | متوسط (چون encoderها از قبل VoxCeleb‌اند) |

**انتقال‌پذیرترین الگوها به شرایط ما (به ترتیب ROI):**
1. **Fine-Tune کامل + Centroid/Gate روی embedding جدید** — ترکیب برنده‌ی few-shot open-set (و سازگار با سند داخلی فاز ۳).
2. **Per-window training** — استاندارد speaker verification؛ فعلاً نقض شده.
3. **OOF-driven everything** — استاندارد Kaggle برای متریک‌های حساس.
4. **Sub-center ArcFace** — برای کلاس‌های ۴-۵ نمونه‌ای با تنوع محیطی.

---

# بخش ۷ — Ceiling Analysis (سقف هر مسیر)

| مسیر | سقف تخمینی | استدلال |
|---|---|---|
| Head-only روی encoder فریز (پارادایم فعلی) | **~0.91–0.93** | بهترین نقطه‌ی مشاهده‌شده 0.9093 (ensemble) و 0.9202 (centroid)؛ embedding ثابت = سقف هندسی ثابت |
| + Decision Layer کامل (Gate+Centroid+T+λ) روی مدل‌های فعلی | **~0.93–0.94** | فاز ۳ با ECAPA خام 0.9202 گرفت؛ فیوژن ۴ مدل + تیون درست کمی بالاتر |
| Full FT تک‌انکودر (ECAPA/CAM++) + Decision کامل | **~0.94–0.96** | known-acc باید از ~95% به ~97%+ برسد؛ AUC از 0.956 به 0.97+ |
| Ensemble ۳-۴ انکودر FT‌شده + WavLM + OOF-tuned fusion | **~0.96–0.975+** | تنوع embeddingها خطاهای decorrelated می‌سازد؛ leader=0.97 اثبات شدنی‌بودن است |

**نتیجه‌ی Ceiling:** با پارادایم فعلی (حتی با تیون کامل) به 0.97 نمی‌رسیم — **Refactor اجباری است**. خبر خوب: داده سقف را محدود نمی‌کند (فایل‌های ۶۰s، embeddingها جدایی‌پذیرند، external data مجاز است).

---

# بخش ۸ — پنج ریسک مرگ‌آور (بازنگری‌شده با ورودی کاربر)

1. **⏱️ اوران هزینه‌ی Inference از بودجه‌ی ۲۰ دقیقه.** GPU وجود دارد (CUDA 12.8 + تأیید شما)، ولی مسیر فعلی file-loop با ۳ فریم‌ورک متوالی است و دو بار timeout خورده. هر فیچر جدید (centroid، پنجره‌ی بیشتر، WavLM) باید با **اندازه‌ی واقعی روی 3,604 فایل در venv لیدربرد** همراه شود، نه حدس. قانون: پیش از هر آپلود، `verify_submission.py` + تست زمان‌بری روی نمونه‌ی ۱۰۰فایلی و برون‌یابی ×۳۶.
2. **🔀 Shakeup از Validation ناپایدار.** وزن‌های فیوژن و thresholdها اگر روی ۸۹۱ نمونه‌ی تک‌seed قفل شوند، روی توزیع واقعی تست (۵۵۴ گوینده‌ی unknown واقعی) می‌توانند چند دهم‌امتیاز تا چند امتیاز جابه‌جا شوند. سند: فروپاشی `learned_mlp`. ضدّت: OOF + seed-std + حاشیه‌ی امنیت در τ.
3. **🧨 شکنندگی پکیج آفلاین (۳ فریم‌ورک در یک پروسه).** شش شکست تاریخی از همین جنس. هر تغییر در submission بدون replay کامل verifier = قمار. WavLM اگر اضافه شود، transformers offline-loading باید دقیقاً همان‌طور که برای speechbrain/nemo انجام شده سخت‌گیرانه تست شود.
4. **🗂️ نویز متمرکز لیبل.** ۴۶ فایل متناقض می‌تواند ۳ گوینده‌ی خاص را تخریب کند؛ اگر آن گویندگان در تست پرتکرار باشند، چند کلاس با هم آسیب می‌بینند (اثر متمرکز، نه پخشیده).
5. **🎚️ Over-tight کردن OOD Gate.** unknown فقط ۱/۴۴۷ وزن دارد ولی known-recall همه‌چیز است؛ اگر τ روی Val سفت تیون شود و در تست چند known اضافی رد شوند، هرکدام یک کلاس F1 را نیمه‌جان می‌کنند. τ باید با OOF و با bias به نفع known-recall تنظیم شود (قانون سرانگشتی: FPR هدف ≤ ۲٪).

---

# بخش ۹ — شکاف اطلاعاتی (برای تصمیم‌های دقیق‌تر)

| مورد | چرا مهم است | راه تهیه |
|---|---|---|
| **Score/لاگ واقعی آخرین submission** | هیچ عدد LB در ریپو ثبت نشده؛ نمی‌دانیم Val ما با LB چقدر gap دارد | دانلود لاگ + شروع `reports/lb_log.md` |
| **مقدار دقیق Timeout سرور** | ۲۰ دقیقه فرض کاربر است؛ PDF خالی گذاشته («TO_BE_FILLED») | پنل مسابقه/برگزارکننده |
| **ذخیره‌ی OOF/Val probs هر مدل** | پیش‌نیاز همه‌ی تیون‌های آفلاین ارزان | اصلاح `ensemble_calibrate.py` برای dump `.npy` |
| **اجرای `multi_encoder_centroid_ensemble`** | تابعش هست، عددش نیست — مرجع تصمیم R10 | ۱ اجرای چندساعته |
| **دلتا‌ی campp (0.8671 در checkpoint vs 0.8932 در fusion)** | اگر مسیر ارزیابی تکرارپذیر نباشد، همه‌ی اعداد زیر سؤال می‌روند | ۱ re-eval کنترل‌شده با seed ثابت |
| **نسبت دقیق known/unknown در تست 3,604تایی** | کالیبراسیون τ و λ | حدس ~50/50 از PDF (split ~50/50 per person) |
| **باگ احتمالی EER=0.346 فاز ۳** | با AUC=0.9557 و d′=3.38 ناسازگار (انتظار EER≈0.03-0.09) | بازبینی `src/eda_embeddings.py` |

---

# بخش ۱۰ — حکم نهایی + ۳ اقدام فوری

## حکم: **Refactor هدفمند** (۵ تغییر محوری، بدون بازنویسی زیرساخت)

| # | چه عوض می‌شود | چه عوض نمی‌شود |
|---|---|---|
| ۱ | Freeze → **Full FT** (با LR تفکیک‌شده) | انکودرها، heads، model_factory |
| ۲ | آموزش bag-level → **per-window** | Dataset و پنجره‌بندی فعلی |
| ۳ | argmax خام → **Centroid + OOD Gate + λ/T تیون‌شده** | قرارداد submission و inference.py اسکلت |
| ۴ | Val تک‌fold → **3-Fold OOF** | منطق تمیزکاری و leak-guard |
| ۵ | UI تک‌آزمایش → **موتور آزمایش (matrix + HPO + decision-tune step)** | Streamlit/Vast/ZenML/MLflow اسکلت |

## ۳ اقدام فوری (بدون نیاز به Retrain — امروز/فردا)

> هدف: بیشترین امتیاز از دارایی‌های موجود + زیرساخت تصمیم‌گیری درست برای ادامه.

1. **آفلاین‌سازی تصمیم (Q2+Q3+Q4 از Roadmap):** ① اصلاح `ensemble_calibrate.py` برای ذخیره‌ی `oof_probs_<enc>.npy` (یا val probs با همان split فعلی)؛ ② ساخت Centroid از embedding هر ۴ checkpoint روی Train (خروجی: `centroids_<enc>.npz`، ~350KB هرکدام)؛ ③ شبیه‌سازی آفلاین: فیوژن `α·probs_head + (1-α)·probs_centroid` + Gate با τ + grid روی (λ_unknown, T, وزن‌ها). **انتظار مستند: 0.9093 → 0.92–0.935** (مبنا: centroid خام ECAPA به‌تنهایی 0.9202 است). هزینه: <۳ ساعت GPU.
2. **پاک‌سازی داده + طراحی OOF:** حذف ۴۶ فایل متناقض + نگه‌داشتن یک نسخه از duplicateهای سالم؛ پیاده‌ی `speaker_aware_kfold(k=3)` کنار split فعلی (بدون شکستن سازگاری). هزینه: ۰ GPU.
3. **فیکس باگ R2 + آماده‌سازی Full-FT:** تغییر `forward_multi_window`/حلقه‌ی train به per-window (flatten B×W) + باز کردن unfreeze برای ecapa/campp + سیو کنترل‌شده‌ی ۲ run آزمایشی (sanity: باید known_acc روی Val از ~91% به 94%+ برسد). هزینه: ~۲×۱۰ ساعت GPU.

---

# بخش ۱۱ — سه اهرم اصلی (با مکانیزم و عدد)

## اهرم ۱ — Full Fine-Tune + آموزش Per-Window ⭐⭐ (بزرگ‌ترین اهرم)

**چرا کار می‌کند (به زبان ساده):** embedding فریزشده برای VoxCeleb ساخته شده، نه برای داده‌ی این مسابقه (لهجه‌ها/کدک/محیط متفاوت — Phase-2 نشان داد هیچ confounder ساده‌ای هم نیست، یعنی کار سخت واقعی است). Head کوچک روی embedding ثابت مثل «مرتب‌کردن مبلمان خانه‌ای است که نقشه‌اش غلط است». Full FT نقشه را اصلاح می‌کند: پنجره‌های هم‌گوینده را در فضای embedding جمع‌تر و جفت‌های قابل‌اشتباه را دورتر می‌کشد. آموزش per-window هم ضریب ضرب داده را احیا می‌کند: ۴ فایل × ~۱۲ پنجره‌ی ۸ثانیه‌ای ≈ **۴۸ نمونه‌ی مؤثر per گوینده** (به‌جای ۴ سیگنال کیسه‌ای فعلی).

**سند داخلی:** frozen-head ≈ frozen-centroid (R1) → گلوگاه embedding است. **سند بیرونی:** در VoxCeleb، full-FT نسبت به frozen-head معمولاً EER را ۲-۴ برابر بهتر می‌کند.

**Δ تخمینی:** +0.03 تا +0.05 | **هزینه:** 8-15h per encoder روی 3090 | **ریسک:** Overfit روی ۴ فایل — پاسخ: augmentation دامنه‌محور + EMA + early-stop روی OOF.

## اهرم ۲ — Decision Layer هندسی: Centroid + OOD Gate + تیون روی OOF ⭐⭐

**چرا کار می‌کند:** (الف) **Centroid در few-shot از softmax-head پایدارتر است** چون «میانگین همه‌ی پنجره‌های آموزشی» را می‌بیند، نه بردار کلاسی که SGD آخرین بار روی ۴ فایل تنظیمش کرده. (ب) **برای OOD، فاصله‌ی هندسی از softmax confidence قابل‌اعتمادتر است:** softmax برای داده‌ی بیرون‌توزیع over-confident می‌شود (شناخته‌شده‌ی ادبیات)، ولی «max cosine به centroids» یک فاصله‌ی واقعی است — فاز ۳ نشان داد AUC=0.9557 و Gate به‌تنهایی +0.219 ارزش داشت. (ج) چون قرارداد نهایی **argmax ماست** (CSV هارد لیبل)، هر knob تصمیم (τ, λ_unknown, T, α فیوژن) مستقیماً روی متریک واقعی قابل تیون است — رایگان، بدون Retrain.

**پیاده‌سازی حیاتی:** centroids باید **پیش‌محاسبه و داخل ZIP** شوند (446×192 float ≈ 350KB per encoder) — سرور به داده‌ی Train دسترسی ندارد. در Inference: embedding فایل تست (میانگین پنجره‌ها + L2-norm) → cosine به centroids → فیوژن با head-probs → Gate → argmax. **هزینه‌ی محاسباتیِ سرور ≈ صفر** (یک ضرب ماتریسی 446×192 per فایل).

**Δ تخمینی:** +0.01 تا +0.03 | **هزینه:** ساعتی | **ریسک:** τ بد تیون شود → پاسخ: OOF + bias به نفع known-recall.

## اهرم ۳ — Validation قابل‌اعتماد (3-Fold OOF) + Ensemble متنوع

**چرا کار می‌کند:** با ۱ فایل per speaker در Val، هر تصمیم (انتخاب epoch، τ، وزن‌ها) روی نویز سوار می‌شود — مثل ساختن خانه روی ماسه. 3-Fold speaker-aware: هر فایل دقیقاً یک بار val می‌شود → **~۲,۷۰۰ نمونه‌ی OOF known + کل unknown** برای تیون؛ هر fold همچنان ~۳-۴ فایل train per speaker دارد (few-shot حفظ می‌شود). Ensemble نهایی: معماری × seed × fold با وزن‌های OOF-optimized. خاصیت جانبی: foldها در زمان inference **هم‌پوشانی مدل‌ها** را هم ممکن می‌کنند (مدل fold متفاوت = تنوع رایگان).

**Δ تخمینی:** مستقیم +0.005 تا +0.015؛ غیرمستقیم بزرگ (جلوگیری از Shakeup و تصمیم‌های غلط پیاپی).

---

# بخش ۱۲ — نقشه‌ی راه در ۳ افق

## 🟢 Quick Wins (۰–۴۸ ساعت — بدون Retrain، ROI بالا)

| # | اقدام | جزئیات اجرایی | Δ انتظار | هزینه |
|---|---|---|---|---|
| Q1 | ثبت `reports/lb_log.md` + خواندن لاگ آخرین run (score + `[diag] cuda_avail`) | از این به بعد: هر submission = یک سطر (تاریخ، کانفیگ، محتوای zip، score) | اعتبار حلقه‌ی تصمیم | ۰ |
| Q2 | ذخیره‌ی Val probs هر ۴ مدل | اصلاح `ensemble_calibrate.py` → `data/processed/val_probs_<enc>.npy` | زیرساخت | <۱h GPU |
| Q3 | تیون آفلاین Decision روی probs ذخیره‌شده | Grid: λ_unknown∈[0.6..1.6]، T∈[0.5..2.5]، وزن‌ها روی simplex | +0.003 تا +0.01 | <۰.۵h |
| Q4 | **Centroid-Fusion Ship شود** | ساخت `centroids_<enc>.npz` از Train با embedding هر checkpoint (خروجی `embedding_proj` پس از L2-norm)؛ فیوژن `α·head + (1-α)·centroid`؛ Gate با τ تیون‌شده روی Val؛ قلاب در `build_submission.py` + replay verifier | **+0.01 تا +0.025** | ۲-۳h |
| Q5 | پاک‌سازی Train | حذف ۴۶+۲ فایل متناقض؛ dedupe غیرمتناقض به یک نسخه | پیش‌نیاز C1 | ۰ |
| Q6 | اجرای `multi_encoder_centroid_ensemble` (موجود در src) | مرجع تصمیم: آیا centroid چندانکودری به‌تنهایی کافی است؟ | دانش | ۲-۴h |
| Q7 | Timing روی venv لیدربرد با ۱۰۰ فایل واقعی + برون‌یابی ×۳۶ | قانون: هیچ فیچری بدون عدد زمان وارد zip نمی‌شود | ضد-DQ | ۱h |

## 🟡 Core Improvements (نیازمند Retrain — هفته‌ی ۱ و ۲)

| # | اقدام | جزئیات | هزینه (3090) |
|---|---|---|---|
| C1 | **فیکس R2 + Full FT**: آموزش per-window + unfreeze کامل برای **ecapa و campp** (۲ انکودر برتر فعلی) | LR: encoder 3e-5 / head 3e-4؛ warmup 10٪ + cosine؛ EMA 0.999؛ early-stop واقعی روی Macro-F1؛ bf16 | ۸-۱۵h ×۲ |
| C2 | **Augmentation دامنه‌محور** | RIR (p=0.4) + MUSAN noise/music (SNR 5-20) + SpecAugment (freq 12×2 / time 40×2) + mp3-codec roundtrip (p=0.3) — config-driven | جذب C1 |
| C3 | **Sub-center ArcFace (k=3) + margin sweep** {0.2,0.3,0.4} | برای کلاس‌های ۴-۵ نمونه‌ای | جذب C1 |
| C4 | **تعادل Loss برای متریک** | speaker:ood = 0.85:0.15؛ CE+LS(0.05) به‌جای Focal+LS (ablation A10)؛ ood_batch_ratio 0.35 | جذب C1 |
| C5 | **3-Fold OOF** برای ۲ انکودر برتر پس از C1 | هر fold ~۳ فایل train per speaker؛ OOF کامل برای تیون نهایی | ×۳ هزینه‌ی C1 |
| C6 | **WavLM-Large FT** (تفنگ بزرگ) | full FT + grad-accum 4 + statistical pooling؛ bf16 (به‌خاطر NaN fp16 در Phase6) | ۲۵-۴۰h |
| C7 | eres2net + titanet FT (تکمیل ensemble) | همان recipe | ۸-۱۵h ×۲ |

## 🔴 High-Risk / High-Reward (فقط اگر زمان ماند)

| # | ایده | مکانیزم | ریسک | پتانسیل |
|---|---|---|---|---|
| H1 | **Transductive centroid update** در inference | با پیش‌بینی‌های خیلی confident تست (p>0.98)، centroidها را on-the-fly به‌روز کن | drift در تست واقعی | +0.005 تا +0.02 |
| H2 | FT میانی روی VoxCeleb1+2 با همین objective، بعد FT نهایی | domain adaptation دو مرحله‌ای | encoderها از قبل VoxCeleb‌اند → ROI نامعلوم | +0.005 تا +0.015 |
| H3 | **Model Soup** (میانگین چند checkpoint پایانی/seedها) | variance کاهش می‌یابد | گاهی صفر اثر | +0 تا +0.005 |
| H4 | Distillation از WavLM-L به ECAPA/CAM++ | اگر WavLM در ۱GB نجاست، دانشش را به مدل کوچک منتقل کن | هزینه‌ی پیاده‌سازی | معماری نهایی سبک |

---

# بخش ۱۳ — جدول اولویت‌بندی ICE

> ICE = Impact × Confidence × Ease (۱ تا ۱۰). مرتب‌شده نزولی. GPU-hour روی 3090.

| رتبه | اقدام | I | C | E | **ICE** | GPU-hour | وابستگی |
|---|---|---|---|---|---|---|---|
| ۱ | Q4: Ship کردن Centroid+Gate روی checkpointهای فعلی | ۸ | ۸ | ۷ | **۴۴۸** | ۲-۳ | Q2 |
| ۲ | Q3: تیون آفلاین (λ, T, وزن‌ها) | ۵ | ۸ | ۹ | **۳۶۰** | <۰.۵ | Q2 |
| ۳ | C1: فیکس R2 (per-window) + Full FT روی ecapa/campp | ۹ | ۸ | ۴ | **۲۸۸** | ۱۶-۳۰ | Q5 |
| ۴ | C5: 3-Fold OOF | ۶ | ۹ | ۵ | **۲۷۰** | ×۳ C1 | C1 |
| ۵ | C2: Augmentation دامنه‌محور (config-driven) | ۶ | ۷ | ۶ | **۲۵۲** | جذب C1 | — |
| ۶ | Q5: پاک‌سازی لیبل | ۴ | ۸ | ۹ | **۲۲۴** | ۰ | — |
| ۷ | C4: تعادل Loss برای متریک | ۴ | ۷ | ۸ | **۲۲۴** | جذب C1 | — |
| ۸ | C3: Sub-center ArcFace + margin sweep | ۵ | ۶ | ۶ | **۱۸۰** | جذب C1 | — |
| ۹ | C6: WavLM-Large FT + حل بسته‌بندی bf16 | ۸ | ۶ | ۳ | **۱۴۴** | ۲۵-۴۰ | C1 |
| ۱۰ | Q6: اجرای multi-encoder centroid موجود | ۴ | ۷ | ۷ | **۱۲۶** | ۲-۴ | — |
| ۱۱ | EMA + warmup config-driven + حذف train.py دوم | ۴ | ۸ | ۶ | **۱۲۸→** (ترکیب با C1) | <۱ | — |
| ۱۲ | TTA بیشتر در inference (۸→۱۵ پنجره) | ۳ | ۷ | ۷ | **۱۴۷** | eval-time | Q7 |
| ۱۳ | H4/H3: Distillation / Model Soup | ۴ | ۵ | ۵ | **۱۰۰** | ۱-۵ | C1/C6 |
| ۱۴ | H2: VoxCeleb intermediate FT | ۶ | ۴ | ۳ | **۷۲** | ۲۰-۴۰ | C1 |
| ۱۵ | H1: Transductive TTA | ۵ | ۴ | ۳ | **۶۰** | eval-time | Q4 |

---

# بخش ۱۴ — Ablation Plan (هر آزمایش = دقیقاً یک متغیر)

> **قرارداد آزمایش:** ① همه‌ی پذیرش/ردها روی **OOF Macro-F1** (نه Val تک‌fold)؛ ② آستانه‌ی پذیرش ≥ **+0.003** (زیر این = نویز، مگر A9 خلافش را نشان دهد)؛ ③ هر run یک سطر در جدول نتایج MLflow + ذخیره‌ی OOF probs به‌عنوان artifact؛ ④ Baseline مرجع = نتیجه‌ی Q4.

| ID | فرضیه | متغیر | ثابت‌ها | شرط پذیرش |
|---|---|---|---|---|
| **A0** | آموزش per-window از bag-level بهتر است | flatten(B×W) در train loop | بقیه‌ی کانفیگ campp فعلی | ΔOOF ≥ +0.005 |
| **A1** | Full-FT از Frozen بهتر است | unfreeze کامل (ecapa) | بقیه ثابت | Δ ≥ +0.010 |
| **A2** | Centroid+Gate از argmax-head بهتر است | decision layer | همان checkpoint فعلی | Δ ≥ +0.005 |
| **A3** | λ_unknown≠1 امتیاز دارد | grid λ∈[0.6,1.6] | فیوژن ثابت | Δ ≥ +0.003 |
| **A4** | RIR+MUSAN+SpecAug کمک می‌کند | +domain aug | بقیه ثابت | Δ ≥ +0.003 |
| **A5** | Sub-center(k=3) بهتر از k=1 | k | margin ثابت | Δ ≥ +0.003 |
| **A6** | EMA بهتر از وزن خام | EMA on/off | — | Δ ≥ +0.002 |
| **A7** | پنجره‌ی بیشتر در eval کمک می‌کند | ۸ vs ۱۵ پنجره | GPU، زمان≤بودجه | Δ ≥ +0.002 |
| **A8** | WavLM-L قوی‌ترین تک‌مدل است | encoder=wavlm | همان recipe | OOF > بهترین موجود |
| **A9** | واریانس seed چقدر است | ۳ seed از کانفیگ برتر | — | گزارش std؛ اگر std>0.005 → همه‌ی آستانه‌ها multi-seed می‌شوند |
| **A10** | CE+LS بهتر از Focal+LS است | loss | — | Δ ≥ +0.003 |
| **A11** | ood_batch_ratio=0.35 بهتر از 0.5 | sampler ratio | — | Δ ≥ +0.003 |
| **A12** | فیوژن head+centroid از هرکدام جدا بهتر است | α∈[0..1] | مدل ثابت | Δ ≥ +0.003 |

**ترتیب اجرا:** A2-A3-A12 (آفلاین، فوری) → A0 → A1 → A9 → A4/A5/A6/A10/A11 → A7/A8.

---

# بخش ۱۵ — Recipe نهایی پیشنهادی + بودجه‌ی بسته

## ۱۵.۱ YAML پیکربندی مدل برنده

```yaml
# ═══════════ Winning Recipe — IAAA 2026 Speaker-ID (v2) ═══════════
data:
  cleaning:
    drop_corrupted: true                 # ۷۰ فایل <1s
    drop_conflicting_duplicates: true    # گروه ۴۶تایی ۴-لیبله + گروه ۲تایی → حذف کامل
    dedupe_nonconflicting: keep_one
  split:
    scheme: speaker_aware_kfold
    folds: 3                             # OOF کامل؛ هر fold ≈ ۳ فایل train + ۱-۲ val per speaker
    unknown_val_ratio_per_fold: 0.33
    seeds: [42, 1337, 2026]

audio:
  sample_rate: 16000
  duration_seconds: 8.0
  train_windows_per_file: 4              # ۴ کراپ رندوم — ولی به‌صورت per-window در Loss (فیکس R2)
  window_loss_mode: per_window           # ← flatten (B×W,1,T)؛ نه میانگین!
  eval: {window_seconds: 8.0, hop_ratio: 0.5, max_windows: 15}   # پوشش کامل ~۶۰s روی GPU

augmentation:                            # ← config-driven (پیش‌نیاز HPO)
  waveform:
    gaussian_noise: {p: 0.4, amp: [0.001, 0.012]}
    gain: {p: 0.3, db: [-6, 6]}
    polarity_inversion: {p: 0.5}
    shift: {p: 0.3, frac: 0.1}
    pitch_shift: {p: 0.25, semitones: [-1, 1]}
    time_stretch: {p: 0.2, rate: [0.85, 1.18]}
  domain:                                # ← جدید (C2)
    rirs_reverb: {p: 0.4}
    musan: {noise_p: 0.4, music_p: 0.2, snr_db: [5, 20]}
    mp3_codec_roundtrip: {p: 0.3}        # تست mp3 است → تطبیق دامنه
  spec: {freq_mask: {p: 0.5, width: 12, n: 2}, time_mask: {p: 0.5, width: 40, n: 2}}

model:
  encoders: [ecapa, campp, eres2net, titanet, wavlm_large]
  freeze_encoder: false                  # ← Full FT (C1)
  speaker_head:
    type: arcface_subcenter              # k=3 (C3)
    embedding_dim: 192
    margin: 0.3                          # sweep {0.2, 0.3, 0.4}
    scale: 32
  ood_head: {hidden_dim: 256, dropout: 0.3}

training:
  optimizer: adamw
  lr: {encoder: 3.0e-5, heads: 3.0e-4}
  weight_decay: 1.0e-4
  schedule: {warmup_ratio: 0.10, type: cosine, min_lr_ratio: 0.05}   # ← warmup واقعاً خوانده شود
  epochs: 120
  early_stopping: {monitor: fold_macro_f1, patience: 15}
  batch_size: 32        # پنجره-level؛ با grad_accum=2 برای WavLM
  amp: bf16             # fp16 روی WavLM ناپایدار بود (Phase6)
  ema: {enabled: true, decay: 0.999}
  grad_clip: {default: 5.0, ood_head: 1.0}
  loss:
    speaker: {type: ce, label_smoothing: 0.05, weight: 0.85}   # focal حذف (A10)
    ood:     {type: bce, weight: 0.15}
  sampler: {type: balanced_batch, ood_ratio: 0.35}               # A11

inference:
  embedding:
    source: arcface_embedding_proj       # فضای ۱۹۲بُعدی نرمال‌شده
    window_aggregation: mean_then_l2norm # میانگین embedding پنجره‌ها → نرمال
  centroids:
    build: all_train_windows             # از کل Train (پس از انتخاب recipe، روی full-data retrain)
    ship_as: centroids_<enc>.npz         # ~۳۵۰KB per encoder — داخل ZIP
  decision:
    fusion: alpha * head_probs + (1-alpha) * softmax(kappa * cosine_to_centroids)
    ood_gate: max_cosine < τ → unknown   # + سیگنال head: p_unknown ترکیبی
    tune_on: oof_only                    # τ, α, κ, λ_unknown, وزن encoderها — همه روی OOF
    grids: {tau: [0.10..0.50, step 0.01], alpha: [0..1, step 0.05],
            lambda_unknown: [0.6..1.6, step 0.05], temperature: [0.5..2.5, step 0.1]}
  ensemble:
    members: [ecapa×3folds, campp×3folds, eres2net×3folds, titanet×3folds, wavlm_large]
    weighting: oof_optimized_simplex
  runtime:
    batching: cross_file                 # فایل‌ها را batch کن، نه فقط پنجره‌ها را
    budget_check: اجباری پیش از آپلود (تست ۱۰۰فایلی × ۳۶)
packaging:
  strip_optimizer_states: true           # حجم checkpoint نصف می‌شود
  verify: verify_submission.py           # اجباری پیش از هر آپلود
  log_rule: "[diag] خط اول stdout"
```

## ۱۵.۲ بودجه‌ی حجم پکیج (سقف ۱GB) — محاسبه‌ی صریح

| گزینه | محتوا | حجم تخمینی | در بودجه؟ |
|---|---|---|---|
| فعلی | ۳ ckpt (~۲۱۰MB) + ۳ weights (~۱۳۰MB) + vendor/code (~۵۰MB) | **۳۸۶MB** ✅ | ✅ |
| +ECAPA (ensemble چهارتایی) | + ckpt ۱۳۴MB (با strip optimizer ≈ ۹۰MB) + weights ۸۹MB | **~۶۰۰MB** | ✅ |
| WavLM-L به‌صورت fp32 | weights ۱,۲۸۳MB + ckpt ~۱,۲۴۰MB | ~۲.۵GB | ❌ |
| **WavLM-L به‌صورت bf16، فقط checkpoint** (بدون weights پایه — نیاز به تغییر loader: ساخت از config و لود state_dict) | ~۶۳۰MB + ۲ مدل کوچک (~۱۵۰MB) + code | **~۸۵۰MB** | ✅ (نیازمند تست NaN — Phase6 با fp16 مشکل داشت، bf16 احتمالاً سالم است چون دامنه‌ی نما بزرگ‌تر است) |
| WavLM-Base-Plus (جایگزین سبک‌تر) | ~۹۴M پارامتر → fp32 ~۳۸۰MB | با ۲-۳ مدل کوچک | ✅ |
| Distillation (H4) | فقط ECAPA/CAM++ دانش‌یافته | ~۲۰۰MB | ✅✅ |

**توصیه:** مسیر A (چهار انکودر کوچک) را اول کامل کنید؛ WavLM-L فقط اگر در OOF برنده شد وارد مسیر بسته‌بندی bf16 شود. قانون طلایی: **هیچ مدلی وارد ZIP نمی‌شود مگر اینکه در OOF سهم مثبت اثبات‌شده داشته باشد.**

---

# بخش ۱۶ — استراتژی HPO و ML-Engineering (نقد گزارش قبلی + طرح من)

## ۱۶.۱ ارزیابی صادقانه‌ی `GENERAL_ML_ENGINEERING_REPORT.md` (۱۲ آگوست)

**نقاط قوتش (هنوز معتبر — تأیید می‌کنم):**
- تشخیص درست زیرساخت: نبود HPO، هاردکد بودن augmentation، dead keyهای config (`warmup_steps`، `fusion.ensemble_method`، `mixup_alpha`)، ناهماهنگی TTA (logit-avg vs prob-avg)، ناهماهنگی objective تیون threshold (binary-F1 vs Macro-F1) — **همه هنوز پابرجاند** و در طرح من پوشش داده شده‌اند.
- جدول بازه‌های HPO (بخش ۷.۱ آن گزارش) نقطه‌ی شروع خوبی برای search space است.
- پیشنهاد Prototypical Head (G.1) دقیقاً هم‌جهت با اهرم ۲ من است.

**اختلافات من با آن گزارش (با سند جدید پس از آموزش ۴ مدل):**

| موضوع | نظر گزارش قبلی (۱۲ آگوست) | نظر من (با سند امروز) |
|---|---|---|
| استراتژی Freeze | «CAM++/ERes2Net/TitaNet کاملاً Frozen بمانند؛ فقط ECAPA جزئی باز شود» (B.6) | ❌ **رد می‌کنم.** همین سیاست اجرا شد و نتیجه‌اش: head-only ≤ centroid خام (R1). Full FT اهرم اصلی است. |
| جایگاه HPO | P1 — هفته‌ی اول، پیش از آموزش کامل | ⚠️ **تأخیری.** HPO روی پارادایم Frozen-Head = تیون سیستمی با سقف 0.92 = اتلاف GPU. ترتیب درست: اول فیکس پارادایم (C1-C4)، بعد HPO روی پارادایم جدید. |
| برآورد Gap | جمع خطی «0.045–0.095» | ⚠️ خوش‌بینانه‌ی خطی؛ اثرها sub-additive‌اند. برآورد واقع‌بینانه‌ی من: 0.95–0.975 با اجرای کامل. |
| Focal Loss | حفظ با γ=2 | پیشنهاد ablation برای حذف (A10) — Focal+LS+ArcFace سه regularizer تداخل‌دار. |
| Priority یادگیری | Train ECAPA اول | درست، ولی با فیکس R2 و Full FT — نه با کانفیگ فعلی. |

## ۱۶.۲ استراتژی HPO پیشنهادی من (طرح دو مرحله‌ای)

**اصل اول: HPO فقط روی پارادایم نهایی.** جست‌وجو وقتی معنا دارد که تپه‌ای که بالا می‌رویم همان کوه باشد.

**اصل دوم: objective = OOF Macro-F1.** هرگز روی Val تک‌fold تیون نکن (R4).

**مرحله‌ی ۱ — Coarse (ارزان):**
- ۳۰-۴۰ trial، هر trial: **۳۰ epoch + ۱ fold + ۲ انکودر برتر**
- Sampler: `TPESampler` (بیزی) + Pruner: `MedianPruner(n_startup=5, n_warmup=5)` — trialهای ضعیف در epoch ۱۰-۱۵ کشته می‌شوند → ~۷۰٪ صرفه‌جویی GPU
- Search space فاز ۱ (با اولویت اثر):

```yaml
space:
  training.learning_rate:      {type: loguniform, low: 5e-5, high: 5e-4}   # head
  training.encoder_lr:         {type: loguniform, low: 1e-6, high: 1e-4}   # encoder
  training.weight_decay:       {type: loguniform, low: 1e-5, high: 1e-3}
  arcface.margin:              {type: uniform, low: 0.2, high: 0.45}
  arcface.scale:               {type: uniform, low: 24, high: 40}
  training.label_smoothing:    {type: uniform, low: 0.0, high: 0.15}
  training.ood_loss_weight:    {type: uniform, low: 0.1, high: 0.3}
  audio.ood_batch_ratio:       {type: uniform, low: 0.3, high: 0.5}
  augmentation.domain_strength:{type: uniform, low: 0.2, high: 0.5}
```

**مرحله‌ی ۲ — Fine (گران):** top-5 کانفیگ × ۳ fold × full 120 epoch → انتخاب نهایی با میانگین OOF و std گزارش‌شده (A9).

**بودجه:** Coarse ≈ ۳۰-۶۰ GPU-hour؛ Fine ≈ ۱۰۰-۱۵۰ GPU-hour — با توجه به «Train نامحدود» قابل‌قبول است.

**نکته‌ی مهم:** پراکسی ارزان برای pruning در ۳۰ epoch: `(known_acc + ood_f1)/2` — چون Macro-F1 در epochهای اولیه نویزی است.

## ۱۶.۳ نظم مهندسی (Engineering Governance)

1. **`reports/lb_log.md`:** هر submission یک سطر: تاریخ | hash کانفیگ | محتوای zip | score LB. بدون این، تصمیم‌گیری کور است.
2. **Seed policy:** هر نتیجه‌ی گزارش‌شده یا ۳ seed دارد یا با برچسب «single-seed, احتمال نویز ±0.005».
3. **Config-as-code:** هر run = یک فایل config نام‌دار در `configs/experiments/` + git commit hash در MLflow (الان `log_code_snapshot` هست — حفظ شود).
4. **One-variable rule:** هر run فقط یک تغییر نسبت به baseline (مگر run نهایی ترکیبی).
5. **تست‌ها:** `tests/` الان فقط audio_preprocessing را پوشش می‌دهد؛ حداقل‌های لازم: تست یکتایی class_map بین train/inference، تست round-trip ساخت مدل از checkpoint بدون weights پایه (پیش‌نیاز WavLM bf16)، تست smoke برای `score_ensemble`.

---

# بخش ۱۷ — ارتقای MLOps به «موتور آزمایش» (Experiment Engine)

> دارایی فعلی: Streamlit UI (۴ تب)، Vast.ai auto-deploy، ZenML ۵-step، MLflow/DagsHub، Verifier. این اسکلت برای «یک run در هر لحظه» خوب است؛ برای بردن 0.97 به «کمپین آزمایشی» نیاز داریم. ارتقاهای پیشنهادی به ترتیب ارزش:

## ۱۷.۱ پروفایل‌های نام‌دار آزمایش (به‌جای بازنویسی config مشترک)

- `configs/experiments/<name>.yaml` — هر آزمایش یک فایل (ارث‌بری از `default_config.yaml` + override).
- در UI: به‌جای «ذخیره در default»، دکمه‌ی **«Save as experiment…»**؛ تب Config خروجی‌اش experiment profile می‌شود.
- چرا مهم است: الان دو کلیک پیاپی در UI، config قبلی را می‌بلعد و reproducibility را می‌شکند.

## ۱۷.۲ ماتریس آزمایش + صف اجرا (Queue)

- در UI تب جدید **«🧬 Experiment Matrix»**: چندانتخابی encoders × recipes (Frozen/Full/Partial) × seeds × folds → تولید N پروفایل → صف اجرا.
- اجرا: محلی (LocalRunner موجود) یا ابری: **یک instance = یک run** (الان هم self-destruct دارد) یا یک instance با حلقه‌ی روی صف (ارزان‌تر).
- هر run خودکار: MLflow run + OOF probs artifact + سطر در جدول نتایج.

## ۱۷.۳ سه Step جدید در Pipeline (قلب تفانتن موجود است)

```
train_model ──▶ [NEW] build_embeddings   → کش embedding train/val per fold
              ─▶ [NEW] decision_tune     → ساخت centroids + grid (τ, α, λ, T) روی OOF
              ─▶ [NEW] ensemble_select   → وزن‌دهی simplex روی OOF + گزارش سهم هر عضو
```

مزیت: خروجی هر run نه‌فقط یک checkpoint، بلکه **یک «بسته‌ی تصمیم» کامل و قابل مقایسه** است.

## ۱۷.۴ اتصال Optuna به UI

- تب Config → بخش «HPO»: انتخاب space preset (بخش ۱۶.۲)، تعداد trials، بودجه‌ی epoch → دکمه‌ی Launch → study در `sqlite:///checkpoints/hpo/study.db` (resumable) + experiment مجزای `speaker-id-hpo` در MLflow.
- Optuna را به `pyproject.toml` محلی اضافه کنید (در لیست سرور هست ولی برای train لازم است محلی باشد).

## ۱۷.۵ تب Analysis ← مقایسه‌ی runها + Promote-to-Submission

- خواندن runها از MLflow API → جدول مرتب‌شده بر OOF Macro-F1 با فیلتر encoder/recipe.
- دکمه‌ی **«Promote»** روی run برنده → `build_submission.py` با checkpoint/centroids/decision-params همان run + اجرای خودکار `verify_submission.py` + ثبت در `lb_log.md`.
- این دکمه، حلقه‌ی «آزمایش → تصمیم → submission» را که الان دستی و خطاپذیر است، می‌بندد.

**هزینه‌ی پیاده‌سازی ۱۷.۱ تا ۱۷.۵:** ~۲-۳ روز کاری — ولی سرعت تکرار آزمایش را چند برابر می‌کند و با بودجه‌ی Train نامحدود شما، **این ارتقا خودش یک اهرم امتیازی غیرمستقیم است.**

---

# بخش ۱۸ — Anti-Patterns (کارهایی که نباید بکنیم)

1. **تیون روی LB** — با 3,604 فایل تست و متریک 447کلاسه، آزمون‌وخطای LB = overfit به public + سوزاندن فرصت. همه‌ی تیون روی OOF.
2. **Stacking/MLP روی Val کوچک** — سند داخلی: `learned_mlp` با 0.266 فروپاشید. اگر stacking، فقط روی OOF بزرگ و با مدل بسیار ساده (LogReg).
3. **عضو جدید در Ensemble بدون اثبات سهم مثبت روی OOF** — eres2net با 0.837 فقط به‌خاطر grid روی Val وزن 0.2 گرفت؛ قانون: leave-one-out باید ضرر کند وگرنه عضو حذف است.
4. **ترکیب سه regularizer بدون ablation** (Focal + LabelSmoothing + ArcFace margin) — یکی‌یکی.
5. **آموزش با مسیر `train.py`** — مسیر pipeline را مرجع نگه دارید و `train.py` را یا حذف/یا به‌همان core واگذار کنید (R6).
6. **افزایش پنجره/مدل در Inference بدون عدد زمان روی venv لیدربرد** — Q7 اجباری.
7. **استفاده‌ی خام از ۴۶ فایل متناقض** — حذف کنید.
8. **اعتماد به Val تک‌فایل برای انتخاب checkpoint** — انتخاب نهایی با OOF/میانگین seedها.
9. **فراموش کردن ship کردن centroids** — سرور داده‌ی Train ندارد؛ centroid باید داخل zip باشد.
10. **Pseudo-Labeling/Transductive بدون threshold بسیار محافظه‌کارانه** (p>0.98) — drift = فاجعه.
11. **HPO پیش از فیکس پارادایم** — تیون سیستم Frozen-Head = پول سوزاندن.
12. **Revert کردن فیکس‌های runtime** (الگوی ۱۴ آگوست) — هر فیکس runtime با تست timing جفت می‌شود و مستند می‌ماند.

---

# ضمیمه A — ریاضیات Metric: 0.97 دقیقاً یعنی چه؟

```
Macro-F1 = ( F1_unknown + Σ_{i=1..446} F1_speaker_i ) / 447
```

- کلاس `unknown` فقط **۱/۴۴۷ ≈ 0.22٪** وزن دارد؛ بازی روی 446 گوینده است.
- با فرض F1_unknown=0.95: `میانگین F1 گویندگان ≥ (447×0.97 − 0.95)/446 = 0.9704`.
- هر گوینده ~۴-۵ فایل تست دارد → یک فایل اشتباه: recall آن کلاس = 0.75-0.80 → F1 آن کلاس ≈ 0.83-0.86 → افت macro ≈ 0.0003 per کلاس.
- **بودجه‌ی خطای کل تست:** برای ماندن ≥0.97، مجموع «F1 از دست‌رفته» ≤ ۱۳.۴ واحد → تقریباً **۱۳ تا ۴۰ فایل known اشتباه** (بسته به الگو) با unknown F1≥0.95.
- ترجمه‌ی عملی: **known per-file accuracy ≥ ~97٪** (از 91.5٪ فعلی campp) و **FPR ≤ ~2٪ در TPR ≥ 95٪** برای unknown (فاز ۳: FPR=5.9٪ — باید ۳ برابر بهتر شود).

**سند امید:** بیس‌لاین frozen ECAPA (بدون یک خط آموزش) با centroid+gate = 0.9202 و known-acc=94.98٪. فاصله‌ی 94.98 → 97+ دقیقاً کارِ Full FT + augmentation + decision tuning است. Leader با 0.97 ثابت کرده شدنی است.

# ضمیمه B — نکته درباره‌ی سرور

`leaderbordpakage.txt` علاوه بر موارد بخش ۲.۲ شامل `optuna`، `faiss-cpu`، `onnxruntime`، `tensorrt` و `pydantic` است — یعنی در صورت نیاز: FAISS-OOD سروری، ONNX/TensorRT برای شتاب، و Config Schema همگی در دسترس‌اند. محدودیت واقعی ما: **ZIP ≤ 1GB + بودجه‌ی زمانی + آفلاین**.

# ضمیمه C — محاسبات بسته‌بندی (۱GB)

| جزء | fp32 | bf16/fp16 | یادداشت |
|---|---|---|---|
| checkpoint مدل (par) | params×4B | params×2B | strip optimizer الزامی (AdamW = +2×params) |
| WavLM-L (317M) | 1.28GB | **0.63GB** | bf16 + loader بدون weights پایه |
| ECAPA (22M) / CAM++ (7.5M) / ERes2Net (18M) / TitaNet (25M) | 89/30/72/102MB | نصف | — |
| سناریوی A: ۴ انکودر کوچک + centroids + code/vendor | **~600-690MB** ✅ | — | امن |
| سناریوی B: WavLM-L bf16 + ۲ کوچک + centroids | **~850MB** ✅ | — | نیازمند تست |
| سناریوی C: distill به ۲ مدل کوچک | ~۲۰۰-۳۰۰MB ✅✅ | — | fallback |

---

*پایان گزارش نسخه‌ی ۲.۰ — در صورت تأیید، فاز اجرا از «۳ اقدام فوری» بخش ۱۰ و سپس Quick Wins آغاز می‌شود.*


