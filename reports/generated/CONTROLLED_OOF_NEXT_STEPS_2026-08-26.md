# مسیر اجرایی کنترل‌شده برای ادامهٔ مسابقه

## تصمیم اصلی

قبل از آزمایش معماری تازه باید دو recipe فعلی را با OOF واقعاً قابل‌مقایسه بازسازی کنیم. چهار اجرای fold 1/2 کافی نیست، چون checkpointهای fold 0 قبلی دو نقص ثبت‌نشده داشتند:

1. فقط `data.split.seed` ثابت بود و seed سراسری model initialization، random crop و augmentation ثبت/اعمال نمی‌شد.
2. معیار best روی raw weights محاسبه می‌شد ولی فایل best شامل EMA weights بود؛ بنابراین metric و artifact متعلق به یک مدل نبودند.

به همین دلیل بلوک صحیح شامل شش اجرای تازه است: سه fold برای no-proto و سه fold برای metric-only.

## کنترل متغیرها

درون هر خانواده فقط `data.split.fold` متغیر علمی است. دو تغییر همراه آن مستقل نیستند:

- `model.unknown_cluster_path`: artifact مشتق‌شده از fold و برای جلوگیری از leakage الزامی است.
- `logging.checkpoint_dir` و `logging.log_dir`: فقط جداسازی خروجی‌ها هستند.

تمام موارد زیر در هر سه fold ثابت‌اند: encoder، initialization seed، augmentation، تعداد window، batch size، optimizer، LRها، schedule، margin، loss weights، epoch budget، patience، EMA decay، split seed، cluster count و محیط dependency.

| خانواده | invariant SHA-256 | checkpoint مبنا |
|---|---|---|
| no-proto | `399a1b1101e0295057a754d91724243aaac2638404aaf1a367e460d555963197` | `92893c7642901dc2e1bc4eb1d70d9b51c8ed7b03c286b0c89f7340a46475ad40` |
| metric-only | `8aa73e90d9aaa029dfb85c02552127b981be6bf1fd2f73ba5eb619671a16a82d` | `ead5d1b7af290271db356c9ecf5e980513693d1f793a0c989e98094e8f0f37e5` |

preflight علاوه بر این hashها، سه cluster map را با hash، تعداد فایل و دقیقاً 554 cluster کنترل می‌کند و با هر drift اجرای آموزش را متوقف می‌کند.

## ترتیب اجرا روی Vast.ai

همهٔ runها را ترجیحاً روی همان instance، همان نوع GPU، همان commit و همان `uv.lock` اجرا کنید. profile سخت‌افزار در هر شش run برابر `vastai_3060` و batch size برابر 16 است؛ حتی روی GPU بزرگ‌تر آن را تغییر ندهید.

### مرحلهٔ صفر: preflight

```bash
uv run --no-sync python -X utf8 scripts/verify_oof_experiments.py
```

خروجی باید `"status": "ok"` باشد. اگر checkpointهای مبنا روی سرور موجود نباشند، بخش baseline برابر `checked: false` می‌شود ولی invariant و cluster mapها همچنان کامل کنترل می‌شوند.

### مرحلهٔ ۱: safety gate با no-proto fold 0

```bash
uv run --no-sync python -X utf8 scripts/run_controlled_oof_block.py --phase no-proto-f0
```

قبل از ادامه باید این موارد برقرار باشند:

- `probability_avg_macro_f1 >= 0.960`؛ مقدار مرجع checkpoint قبلی روی fold 0 تقریباً `0.9674` بود.
- هر دو فایل `campp_best_raw.pt` و `campp_best_ema.pt` موجود باشند و `campp_best.pt` صریحاً variant منتخب را ثبت کرده باشد.
- `oof_predictions.npz` شامل `competition_probs` با عرض 447 و embeddings باشد.
- metadata شامل seed، deterministic policy، git revision، نسخه‌ها و fingerprint داده/cluster map/`uv.lock` باشد.

اگر این gate رد شد، foldهای دیگر را اجرا نکنید و همان directory را برای تحلیل برگردانید.

### مرحلهٔ ۲: تکمیل no-proto OOF

```bash
uv run --no-sync python -X utf8 scripts/run_controlled_oof_block.py --phase no-proto-rest
```

پس از آن runner به‌طور خودکار، در صورت وجود هر سه fold، گزارش زیر را می‌سازد:

`reports/generated/campp_no_proto_controlled_oof.json`

شرط ادامه: 4447 فایل OOF یکتا، بدون overlap، و probability-average OOF Macro-F1 حداقل حدود `0.965`. افت بزرگ یک fold نسبت به دو fold دیگر نشانهٔ instability است و باید قبل از metric-only تحلیل شود.

### مرحلهٔ ۳: safety gate با metric-only fold 0

```bash
uv run --no-sync python -X utf8 scripts/run_controlled_oof_block.py --phase metric-f0
```

مرجع فعلی fold 0 تقریباً `0.9581` است؛ gate محافظه‌کارانه `probability_avg_macro_f1 >= 0.948` است. هدف این مدل شکست دادن no-proto به‌تنهایی نیست، بلکه تولید خطاهای مکمل برای fusion است.

### مرحلهٔ ۴: تکمیل metric-only OOF

```bash
uv run --no-sync python -X utf8 scripts/run_controlled_oof_block.py --phase metric-rest
```

گزارش خودکار:

`reports/generated/campp_metric_only_controlled_oof.json`

گزینهٔ `--phase all` وجود دارد، اما تا عبور safety gateها توصیه نمی‌شود. برای ادامهٔ اجرای قطع‌شده فقط وقتی artifact نهایی موجود است از `--resume` استفاده کنید؛ runner از overwrite کردن directory نیمه‌کاره جلوگیری می‌کند.

## artifactهایی که باید برگردانده شوند

برای هر شش profile کل directory متناظر زیر لازم است:

`checkpoints/<profile>/`

مهم‌ترین فایل‌ها:

- `campp_best.pt`: variant منتخب و canonical
- `campp_best_raw.pt` و `campp_best_ema.pt`: دو candidate مستقل
- `campp_best_bundle/oof_predictions.npz`: logits قدیمی + probability-average دقیق submission + embeddings
- `campp_best_bundle/resolved_config.yaml`
- `campp_best_bundle/metadata.json`
- `campp_best_bundle/training_history.json`
- `campp_best_bundle/manifest.json`
- دو گزارش aggregate خانواده‌ها و runner manifest

این artifactها در MLflow نیز ارسال می‌شوند. دانلود فقط `.pt` برای inference کافی است، اما برای تحلیل OOF و طراحی quality-aware gate باید bundle کامل برگردد.

## تصمیم بعد از دریافت نتایج

1. OOF سه-fold دو خانواده با probability-average ارزیابی می‌شود.
2. وزن fusion و quality-aware OOD gate فقط با cross-fit روی OOF تنظیم می‌شود؛ validation نهایی برای tuning مصرف نمی‌شود.
3. اگر known-to-unknown rejection همچنان خطای غالب باشد، جفت آزمایش معماری known-first تعریف می‌شود: یک control بدون pseudo-tail و یک variant با pseudo-cluster auxiliary confidence-weighted. این دو نیز با یک baseline مشترک و فقط یک عامل معماری متفاوت تعریف خواهند شد.
4. پس از انتخاب recipe/decision layer، یک full-data fit با epoch از پیش تعیین‌شده از OOF انجام می‌شود؛ leaderboard برای تنظیم threshold استفاده نمی‌شود.

پس این شش run «جست‌وجوی hyperparameter» نیستند؛ ابزار اندازه‌گیری تمیز برای تصمیم معماری بعدی‌اند.
