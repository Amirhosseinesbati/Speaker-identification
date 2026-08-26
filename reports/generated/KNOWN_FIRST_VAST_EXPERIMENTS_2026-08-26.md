# Known-first CAM++ — Vast.ai experiment gate

## تصمیم صریح دربارهٔ شش اجرای قبلی

در حال حاضر شش profile زیر را اجرا نکنید:

- `p0-campp-no-proto-repro-oof-f0/f1/f2`
- `p0-campp-metric-only-repro-oof-f0/f1/f2`

دلایل:

1. `no-proto f0` دقیقاً recipe داخل `checkpoints/campp_best (4).pt` است؛ SHA-256
   مدل موجود `92893c...ad40` با baseline ثبت‌شده در profile یکسان است.
2. `metric-only f0` دقیقاً recipe داخل `checkpoints/campp_best (5).pt` است؛ SHA-256
   مدل موجود `ead5d1...37e5` با baseline profile یکسان است.
3. foldهای 1 و 2 فقط variance همان دو معماری را اندازه می‌گیرند. این کار برای
   post-mortem علمی مفید است، اما پس از افت انتقال هر دو معماری روی leaderboard،
   در حال حاضر بازده مناسبی برای رسیدن به امتیاز بالاتر ندارد.
4. equal ensemble مدل‌های مشابه نیز قبلاً diversity کافی نشان نداده است. بنابراین
   چهار آموزش اضافه، فرضیهٔ جدیدی دربارهٔ علت خطای known→unknown آزمایش نمی‌کنند.

نتیجه: این شش اجرا **deferred** هستند، نه اینکه ذاتاً بی‌فایده باشند. فقط اگر gate
جدید شکست بخورد و بخواهیم variance recipeهای قبلی را برای post-mortem دقیق بسنجیم،
به آن‌ها برمی‌گردیم.

## فرضیهٔ جدیدی که واقعاً آزمایش می‌شود

تحلیل خطا نشان داد bottleneck اصلی، اشتباه هویت knownها نیست؛ knownها به unknown
reject می‌شوند. در مدل‌های فعلی، 554 pseudo-class در softmax اصلی با 446 known رقابت
می‌کنند. معماری جدید این رقابت را حذف می‌کند:

```text
audio → CAM++ → shared embedding → ArcFace[446] → known identity
                              ├─ binary OOD head → known / unknown
                              └─ optional EMA metric loss[446+554]
```

- هد اصلی همیشه دقیقاً 446 کلاس دارد.
- pseudo-labelها از CE اصلی ماسک می‌شوند.
- OOD head همچنان pseudo-labelهای بالاتر از 446 را unknown می‌بیند.
- فقط در treatment، pseudo-labelها با وزن 0.05 وارد auxiliary metric loss می‌شوند.
- خروجی مدل مستقیماً 447کلاسه است و هیچ pseudo-tail در softmax submission ندارد.

این decoupling در `model.speaker_target_scope: known` پیاده‌سازی شده و رفتار پیش‌فرض
`metric` برای checkpointها و آزمایش‌های قدیمی بدون تغییر مانده است.

## فاز اول: فقط دو اجرای اجباری

| ترتیب | profile | نقش |
|---:|---|---|
| 1 | `p0-campp-known446-ood-control-oof-f0` | 446-way + OOD، بدون pseudo auxiliary |
| 2 | `p0-campp-known446-ood-auxmetric-oof-f0` | همان مدل، با 5% pseudo metric auxiliary |

دو profile از نظر encoder، fold، augmentation، windows، optimizer، schedule، EMA و
cluster map یکسان‌اند. تنها تفاوت علمی، اختصاص 5% loss به auxiliary metric است.
نسبت speaker:OOD در 95% باقی‌مانده همان 85:15 حفظ شده است.

فرمان پیشنهادی روی instance:

```bash
uv run --no-sync python scripts/run_known_first_block.py --phase gate
```

dry-run محلی/سرور:

```bash
uv run --no-sync python scripts/run_known_first_block.py --phase gate --dry-run
```

runner پیش از آموزش، cluster-mapها و invariant کانفیگ‌ها را verify می‌کند، checkpoint
موجود را overwrite نمی‌کند و هر child را نیز با `uv run --no-sync` اجرا می‌کند.

## قانون Go / No-Go پس از fold 0

مرجع legacy روی همین fold:

- metric-only: `val_macro_f1 = 0.937231`
- no-proto: `macro_f1 = 0.937106`, `known_acc = 0.921769`, `ood_f1 = 0.952949`

برای ادامهٔ campaign، winner باید هم‌زمان این شرایط را داشته باشد:

1. `macro_f1 >= 0.9387`؛ یعنی حداقل حدود 0.0015 بالاتر از بهترین legacy fold-0؛
2. `known_acc >= 0.9238`؛ چون فرضیه دقیقاً باید known→unknown را کم کند؛
3. `ood_f1 >= 0.9500`؛ بهبود known نباید با بازکردن بی‌رویهٔ gate و unknown→known حاصل شود؛
4. raw و EMA هر دو بررسی شوند و انتخاب weight variant از metric خودش انجام شود؛
5. معیار `probability_avg_macro_f1` نیز گزارش شود، چون به inference submission نزدیک‌تر است.

قاعدهٔ انتخاب بین دو run:

- اگر auxmetric حداقل `+0.0015` بهتر از control و شرط‌های بالا را پاس کرد: auxmetric winner؛
- اگر اختلاف کمتر از `0.0015` بود: control برنده است، چون ساده‌تر و کم‌ریسک‌تر است؛
- اگر هیچ‌کدام شرایط بالا را پاس نکردند: **آموزش foldهای 1/2 متوقف شود**. در آن صورت
  known-first با این recipe علت اصلی را حل نکرده و مرحلهٔ بعد باید hard-condition mining
  واقعی باشد، نه افزایش تعداد seed/fold.

## فاز دوم: فقط دو fold خانوادهٔ برنده

اگر control برنده شد:

```bash
uv run --no-sync python scripts/run_known_first_block.py --phase control-rest
```

اگر auxmetric برنده شد:

```bash
uv run --no-sync python scripts/run_known_first_block.py --phase auxmetric-rest
```

runner عمداً `--phase all` ندارد تا هر دو خانواده سهواً اجرا نشوند. پس campaign پیشنهادی
در حالت موفق فقط چهار training run دارد: دو gate + دو confirmation، نه شش اجرای قبلی
و نه شش اجرای جدید.

پس از کامل‌شدن سه fold خانوادهٔ برنده، runner `oof_predictions.npz`ها را concatenate
و گزارش OOF را در `reports/generated/` می‌سازد. تصمیم full-data فقط بعد از مشاهدهٔ
پایداری سه fold گرفته می‌شود.

## artifactهایی که باید برگردانده شوند

برای هر run، ترجیحاً کل این پوشه برگردانده شود:

```text
checkpoints/<profile>/
```

حداقل فایل‌های لازم:

- `campp_best.pt` و در صورت وجود `campp_best_ema.pt`؛
- پوشهٔ `campp_best_bundle/` شامل metadata، config، class map، history، manifest و
  `oof_predictions.npz`؛
- log کامل run؛
- لینک یا export اجرای MLflow.

checkpoint و bundle اکنون `speaker_target_scope`، تعداد metric classها، عرض هد اصلی،
config حل‌شده، split، fingerprints و تاریخچهٔ معیارها را نگه می‌دارند؛ بنابراین تحلیل
بعدی به حدس‌زدن recipe وابسته نیست.

## جمع‌بندی کارشناسی

- اجرای شش profile داخل تصویر الآن توصیه نمی‌شود.
- اجرای دو run gate جدید توصیه می‌شود؛ چون ارزان‌ترین آزمایشی است که مستقیماً فرضیهٔ
  ریشه‌ای known→unknown را رد یا تأیید می‌کند.
- اطمینان به ارزش اطلاعاتی این gate: بالا، حدود 85%.
- اطمینان به اینکه خود این معماری leaderboard را بهتر می‌کند: متوسط، حدود 60%.
- اطمینان به رسیدن مستقیم به بالای 0.97 پیش از دیدن gate و OOF: پایین؛ ادعای بالاتر
  از این با شواهد فعلی صادقانه نیست.
