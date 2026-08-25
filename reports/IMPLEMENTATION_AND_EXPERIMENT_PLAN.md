# برنامهٔ پیاده‌سازی و آزمایش‌های مسیر 0.97+

## جمع‌بندی معماری اجراشده

معماری اصلی اکنون یک hybrid واقعی است:

1. یک embedding مشترک از encoder؛
2. هد متریک `446+k` کلاسه برای 446 گویندهٔ شناخته‌شده و pseudo-identityهای unknown؛
3. هد دودویی OOD که target آن از label اصلی مسابقه می‌آید، نه از pseudo-label؛
4. Prototypical loss اختیاری برای هم‌راستا کردن فضای آموزش و centroid readout؛
5. collapse احتمال تمام pseudo-identityها به ستون `unknown` در خروجی 447کلاسه؛
6. windowing گفتارمحور مشترک بین train، validation و submission؛
7. checkpoint خودتوصیف‌گر و bundle قابل تحلیل در MLflow.

این جداسازی خطای قبلی را رفع می‌کند: پس از relabel شدن unknownها به شناسه‌های 447 به بالا، OOD target دیگر ناپدید نمی‌شود. قرارداد ذخیره‌شده در داده چنین است:

- `metric_label`: هدف ArcFace/prototype؛
- `is_ood`: هدف دودویی اصلی؛
- `label`: alias سازگار با کدها و checkpointهای قدیمی؛
- `original_speaker_id`: برچسب پیش از clustering.

## ترتیب اجرای آزمایش‌ها

### P0 — تصمیم معماری اصلی (اجباری، بالاترین اولویت)

سه profile زیر یک recipe یکسان را روی سه fold اجرا می‌کنند:

- `p0-campp-hybrid-k554-oof-f0`
- `p0-campp-hybrid-k554-oof-f1`
- `p0-campp-hybrid-k554-oof-f2`

پیش از ارسال queue، سه نقشهٔ fold-specific را بسازید. اولین فرمان embedding همهٔ فایل‌های سالم را یک‌بار استخراج می‌کند؛ دو فرمان بعد فقط cache را بر اساس فایل‌های train هر fold slice می‌کنند. در نتیجه فایل validation هیچ fold در KMeans همان fold حضور ندارد:

```bash
uv run --no-sync python -m src.unknown_clustering build --checkpoint checkpoints/campp_best.pt --k 554 --split-scheme kfold --folds 3 --fold 0 --split-seed 42 --out data/processed/unknown_clusters_oof_f0.json --force-cache
uv run --no-sync python -m src.unknown_clustering build --checkpoint checkpoints/campp_best.pt --k 554 --split-scheme kfold --folds 3 --fold 1 --split-seed 42 --out data/processed/unknown_clusters_oof_f1.json
uv run --no-sync python -m src.unknown_clustering build --checkpoint checkpoints/campp_best.pt --k 554 --split-scheme kfold --folds 3 --fold 2 --split-seed 42 --out data/processed/unknown_clusters_oof_f2.json
```

این فایل‌ها باید همراه job در همان مسیر وجود داشته باشند یا با نام یکسان در `submission/` قرار گیرند تا fallback سرور آن‌ها را پیدا کند.

وضعیت فعلی: هر چهار map ساخته و در هر دو مسیر `data/processed/` و `submission/` ذخیره شده‌اند. آمار OOF:

| Map | unknown train | اندازهٔ خوشه min/mean/max | coherence margin |
|---|---:|---:|---:|
| f0 | 1482 | 1 / 2.68 / 10 | 0.8201 |
| f1 | 1482 | 1 / 2.68 / 16 | 0.8204 |
| f2 | 1483 | 1 / 2.68 / 13 | 0.8222 |
| full | 2222 | 1 / 4.01 / 18 | 0.8094 |

هر map دقیقاً 554 خوشهٔ غیرخالی دارد. smoke test f0 نیز تأیید کرد `train_pseudo=1482` و `val_pseudo=0`؛ یعنی فایل validation در ساخت یا supervision خوشهٔ fold خودش وارد نشده است.

چرایی: بهترین val تک‌split فعلی به leaderboard نزدیک بوده، اما فاصلهٔ موردنیاز تا 0.97 تقریباً در حد چند فایل است. تصمیم روی یک split می‌تواند همان چند فایل را overfit کند. سه fold باید میانگین، پراکندگی، known accuracy، OOD F1 و best epoch را مشخص کنند.

به‌دلیل اینکه هر گوینده معمولاً ۵ فایل دارد، foldهای K=3 اندازهٔ برابر ندارند (تقریباً 2/2/1 فایل validation برای هر known). امتیاز اصلی OOF باید با concatenation predictionهای هر سه fold و محاسبهٔ یک Macro-F1 واحد ساخته شود؛ میانگین سادهٔ سه Macro-F1 معیار تصمیم نیست.

قاعدهٔ پذیرش:

- بهبود mean OOF Macro-F1 نسبت به recipe فعلی؛
- افت نکردن بیش از 0.002 در هیچ fold بدون علت داده‌ای روشن؛
- بهبود known errors بدون ایجاد unknown→known زیاد؛
- پایداری best epoch برای تعیین retrain نهایی.

فرمان هر اجرا:

```bash
uv run --no-sync python -m src.pipelines.run_pipeline --experiment p0-campp-hybrid-k554-oof-f0 --run eval
```

نام profile را برای foldهای 1 و 2 عوض کنید. در UI نیز می‌توان این سه profile را در Experiment Queue انتخاب و به Vast.ai فرستاد.

هر run فایل `models/bundle/oof_predictions.npz` را در MLflow ذخیره می‌کند. پس از دانلود سه bundle، امتیاز concatenated OOF را چنین بسازید:

```bash
uv run --no-sync python scripts/aggregate_oof_results.py path/f0/oof_predictions.npz path/f1/oof_predictions.npz path/f2/oof_predictions.npz
```

اسکریپت وجود فایل تکراری بین foldها را hard-error می‌کند تا OOF آلوده گزارش نشود.

### P1 — سه ablation برای فهم علت بهبود

1. `p1-campp-metric-only-k554-oof-f0`
   - همان recipe ولی OOD head خاموش است.
   - اختلاف آن با P0 سهم واقعی auxiliary OOD را نشان می‌دهد.

2. `p1-campp-hybrid-no-proto-oof-f0`
   - hybrid حفظ می‌شود اما prototype loss حذف می‌شود.
   - اگر بهتر یا مساوی بود، prototype را حذف می‌کنیم؛ چون افزودن loss بدون سود فقط optimization را سخت‌تر می‌کند.

3. `p1-campp-hybrid-random-windows-oof-f0`
   - speech-aware crop و speech-ranked eval خاموش‌اند.
   - اختلاف آن با P0 اثر واقعی windowing را جدا می‌کند.

این سه اجرا ابتدا فقط روی fold 0 انجام شوند. فقط ablationی که اختلاف معنی‌دار (تقریباً 0.0015 یا بیشتر، یا نجات پایدار خطاهای low-SNR/non-speech) نشان دهد روی foldهای دیگر تکرار شود.

### P2 — تنوع encoder برای ensemble

- `p2-ecapa-hybrid-k554-oof-f0`

چرایی: تحلیل قبلی نشان داد CAM++ و ECAPA با وزن تقریبی 0.8/0.2 بهترین گزینهٔ بدون TitaNet بوده‌اند. هدف این آزمایش شکست دادن CAM++ به‌تنهایی نیست؛ هدف تولید خطاهای غیرهم‌بسته است. پس معیار پذیرش فقط Macro-F1 تکی نیست:

- oracle gain روی خطاهای CAM++؛
- تعداد known errorهایی که ECAPA درست و CAM++ غلط می‌زند؛
- بهبود OOF fusion در چند fold؛
- هزینهٔ inference و اندازهٔ ZIP زیر 1GB.

اگر complementary gain واقعی نبود، ECAPA جدید وارد submission نمی‌شود و وزن صفر می‌گیرد.

### P3 — retrain نهایی روی همهٔ دادهٔ قابل استفاده

- `p3-campp-hybrid-k554-full-data-template`

این profile را قبل از پایان P0 اجرا نکنید. مقدار `training.epochs` باید با یک epoch مقاوم از سه fold جایگزین شود (ترجیحاً median best epoch، نه بیشینهٔ خوش‌بینانه). در حالت `full` همهٔ فایل‌های سالم برای optimization استفاده می‌شوند. validation عمداً overlap دارد و فقط diagnostic است؛ انتخاب checkpoint به آخرین epoch قفل می‌شود.

قبل از retrain نهایی، نقشهٔ کامل را با همهٔ unknownهای سالم بسازید؛ این کار ضعف نقشهٔ قدیمی را رفع می‌کند که فقط 1782 فایل از 2275 unknown را پوشش می‌داد:

```bash
uv run --no-sync python -m src.unknown_clustering build --checkpoint checkpoints/campp_best.pt --k 554 --scope full --split-scheme full --out data/processed/unknown_clusters_full.json --force-cache
```

فرمان پیشنهادی پس از اصلاح epoch:

```bash
uv run --no-sync python -m src.pipelines.run_pipeline --experiment p3-campp-hybrid-k554-full-data-template --run train
```

پس از آن centroidها و decision bundle باید با checkpoint جدید دوباره ساخته شوند:

```bash
uv run --no-sync python -m src.pipelines.run_pipeline --run decision --checkpoints checkpoints/campp_best.pt checkpoints/ecapa_best.pt
```

ECAPA را فقط اگر معیار P2 را پاس کرد در فرمان دوم نگه دارید.

## artifact و MLflow

هر checkpoint جدید این اطلاعات را داخل فایل `.pt` دارد:

- resolved config و class map؛
- schema هدف metric/OOD؛
- architecture و تعداد کلاس‌های competition/pseudo؛
- split و audio policy؛
- تاریخچهٔ کامل epochها و final metrics؛
- git revision و نسخهٔ dependencyهای مهم.

کنار checkpoint، پوشهٔ `<name>_bundle` ساخته و در `models/bundle` روی MLflow ثبت می‌شود؛ شامل `MODEL_CARD.md`، config، class map، history، metadata، manifest، SHA-256 و در صورت وجود cluster map است. بنابراین برای تحلیل سریع نیازی به load کردن PyTorch نیست، ولی خود `.pt` نیز مستقل و کافی باقی می‌ماند.

## ترتیب تصمیم‌گیری نهایی

1. سه اجرای P0؛
2. P1 فقط fold 0؛
3. تکرار ablation برنده روی foldهای 1 و 2؛
4. P2 برای diversity؛
5. انتخاب recipe و epoch با OOF؛
6. P3 full-data؛
7. بازسازی centroid/decision/ensemble؛
8. verify و کنترل اندازه/زمان submission؛
9. فقط یک submission مبتنی بر شواهد پایدار.

هدف واقع‌بینانه این مسیر این است که افزایش دادهٔ مؤثر full-data، کاهش cropهای کم‌گفتار، supervision صحیح OOD و یک encoder مکمل بتوانند دو تا چند known error باقی‌مانده را نجات دهند؛ همان بازه‌ای که val فعلی را از حدود 0.9657 به بالای 0.97 می‌رساند.
