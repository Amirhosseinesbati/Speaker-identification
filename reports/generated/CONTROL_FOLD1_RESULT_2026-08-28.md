# نتیجه و درگاه علمی Control Fold 1

تاریخ تحلیل: 2026-08-28  
کمپین: `iaaa-speaker-rank1-20260827`  
Run: `p0-campp-known446-ood-control-oof-f1`  
Git اجرای علمی: `f413c48aeb9d1b39807ca6f0de2d157c96c3c74e`  
MLflow Run ID: `38c00cae55164dd08f9d4c640df13169`

## حکم اجرایی

Fold 1 سالم و قابل‌استناد است و trigger خرابی سیستماتیک را فعال نمی‌کند. افت بهترین
`probability-average Raw Macro-F1` نسبت به Fold 0 برابر `0.00534617` است که از
مرز ازپیش‌ثبت‌شدهٔ `0.01` کمتر است. بنابراین اجرای **Control Fold 2 مجاز** است.

Fallback ازپیش‌ثبت‌شدهٔ «ارسال نمونهٔ کم‌انرژی به prior فقط-train» در هر دو Fold
جهت منفی دارد و شرط عدم افت `Known Accuracy`/`OOD-F1` را نقض می‌کند؛ در نتیجه
رد می‌شود و در Fold 2 یا submission اعمال نخواهد شد. Foldهای بعدی AuxMetric نیز
طبق gate قبلی همچنان ممنوع‌اند.

## نتیجهٔ اصلی Fold 1

| معیار | مقدار | ایپاک |
|---|---:|---:|
| بهترین Raw / probability-average Macro-F1 | `0.9415750193` | 60 |
| logit-average در checkpoint منتخب | `0.9365667119` | 60 |
| بهترین EMA / probability-average Macro-F1 | `0.9396334955` | 78 |
| بهترین EMA / logit-average | `0.9328326409` | 66 |
| Known Accuracy در checkpoint منتخب Raw | `0.9503945885` | 60 |
| OOD-F1 در checkpoint منتخب Raw | `0.9411764706` | 60 |

ایپاک 80 نقطهٔ پایان early stopping بود، نه checkpoint منتخب:

| معیار پایان Run | مقدار |
|---|---:|
| Raw probability-average | `0.9381398668` |
| Raw logit-average | `0.9318728360` |
| EMA probability-average | `0.9389621878` |
| EMA logit-average | `0.9315596292` |
| Known Accuracy | `0.9481397971` |
| OOD-F1 | `0.9437585734` |
| train loss | `1.7031481138` |
| validation loss | `1.2104504091` |

مسیر probability-average در checkpoint منتخب `0.00500831` از logit-average
بهتر است؛ بنابراین همچنان مسیر واقعی انتخاب checkpoint و submission می‌ماند.

## مقایسهٔ بین Foldها

| Fold | بهترین Raw | بهترین EMA | بهترین Known | بهترین OOD-F1 |
|---:|---:|---:|---:|---:|
| 0 | `0.9469211906` (e112) | `0.9444993028` (e131) | `0.9573991031` | `0.9586206897` |
| 1 | `0.9415750193` (e60) | `0.9396334955` (e78) | `0.9526493799` | `0.9466576637` |

در ده ایپاک آخر Fold 1، میانگین و انحراف معیار Raw به‌ترتیب `0.9378621` و
`0.002449` و شیب آن `+0.000259/epoch` بود. EMA با انحراف معیار کمتر
`0.001489` و شیب `+0.000477/epoch` هنوز هموارتر رشد می‌کرد. هم‌زمان validation
loss شیب اندک مثبت داشت. هم‌بستگی validation loss و Raw Macro-F1 پس از ایپاک
20 برابر `+0.9574` بود؛ این رفتار نشان می‌دهد cross-entropy با معیار رقابت و
کالیبراسیون open-set هم‌راستا نیست و افزایش loss را نباید به‌تنهایی معادل
بیش‌برازش دانست. بااین‌حال انتخاب پس‌نگرانهٔ epoch یا امتداد Run ممنوع است و همان
checkpoint ازپیش‌تعریف‌شده حفظ می‌شود.

## ممیزی OOF، split و provenance

- OOF شامل `1627` فایل یکتا، `447` خروجی رقابت، embedding با بعد `192` و
  speaker logits با عرض `446` است؛ تمام آرایه‌های عددی finite هستند.
- شمار validation دقیقاً با Fold 1 گزارش split منطبق است: `887` known و `740`
  unknown. train پاک نیز `2824 = 1342 known + 1482 unknown` فایل دارد.
- `16` ردیف برچسب موجود در CSV با فهرست corrupted تقاطع داشت و پیش از محاسبهٔ
  prior حذف شد؛ کل مجموعهٔ پاک `4451` فایل است.
- SHA-256 فایل OOF:
  `f6a9c90efea242dc68ee261e26d15372bebf0413758ee0d81047165a7ec29e7d`
- SHA-256 گزارش split:
  `381d48dbd07261b881f5189a005328068e98d74c7a1a4a83708dc22986ad49c0`
- هر `26/26` artifact ثبت‌شده در campaign receipt موجود بود و اندازه و SHA-256
  آن دوباره تطبیق داده شد. هش مدل منتخب با manifest و receipt منطبق بود.
- class map شامل `1001` ورودی مدل metric-space است و قرارداد خروجی رقابت
  `unknown + 446 known = 447` در metadata ثبت شده است.

## ممیزی DagsHub/MLflow

- Run با وضعیت `FINISHED` ثبت شده است.
- هر `80/80` نقطهٔ سری‌های زنده finite و دارای stepهای پیوستهٔ 1 تا 80 است.
- `41` پارامتر، `47` کلید متریک و `31` artifact با حجم شناخته‌شدهٔ مجموع
  `213,738,574` بایت ثبت شده است.
- سه مدل `campp_best.pt`، `campp_best_raw.pt` و `campp_best_ema.pt`، سه bundle
  متناظر، config حل‌شده، history، OOF، class map، metadata، manifest، clustering
  و training summary حاضرند. recovery یا backfill لازم نبود.

## ممیزی کم‌انرژی ازپیش‌ثبت‌شده

تعریف‌ها بدون تغییر حفظ شدند: سکوت دقیق `PCM peak == 0` و کم‌انرژی
`PCM RMS < 1e-4`. prior فقط از train پاک همان Fold محاسبه شد و در هر دو Fold
کلاس `unknown` بود.

| Fold / Gate | نمونه | Δ Macro-F1 | Δ Known Acc | Δ OOD-F1 | حکم |
|---|---:|---:|---:|---:|---|
| F0 exact silence | 10 | `-0.00041531` | `-0.00224215` | `-0.00382617` | رد |
| F0 RMS < 1e-4 | 23 | `-0.00101569` | `-0.00448430` | `-0.00552756` | رد |
| F1 exact silence | 5 | `-0.00000108` | `0.00000000` | `-0.00048117` | رد؛ جهت Macro-F1 منفی |
| F1 RMS < 1e-4 | 15 | `-0.00224214` | `-0.00225479` | `-0.00223482` | رد |

درست است که overall accuracy روی gate کم‌انرژی اندکی بالا رفت، اما Macro-F1،
Known Accuracy و OOD-F1 افت کردند. این دقیقاً نمونه‌ای از نامناسب‌بودن accuracy
برای تصمیم open-set و نامتوازن این مسابقه است.

خروجی‌های بازتولیدپذیر:

- `diagnostics/control_fold0_low_energy_prior.json` با SHA-256
  `1fccda950145ac74cb5b9669ef500a1cbda07bd464c094c224365ee0734ca2f3`
- `diagnostics/control_fold1_low_energy_prior.json` با SHA-256
  `0ea77f48fcd845dc034e50be54bd4f455dcd3e2bdef7ffcfb5b8f16eb6ea6259`
- ابزار: `scripts/analyze_low_energy_prior.py`، commit `ad99e23`

## گام بعدی قفل‌شده

1. فقط `p0-campp-known446-ood-control-oof-f2` از Campaign Supervisor اجرا شود.
2. هیچ threshold، fallback، blend یا AuxMetric جدید در طول Fold 2 تغییر نکند.
3. پس از پایان Fold 2، سه OOF بدون overlap تجمیع، variance و error topology بررسی
   و سپس epoch/full-data policy و decision layer بدون leaderboard tuning تعیین شود.
