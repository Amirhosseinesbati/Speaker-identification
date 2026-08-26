# گزارش ریشه‌ای اختلاف Validation و Leaderboard و Candidate نهایی Top-5

تاریخ: 2026-08-26

## جمع‌بندی اجرایی

اختلاف نتیجه‌ی مدل‌های جدید با لیدربرد ناشی از خرابی ZIP یا تفاوت مسیر inference نبود. مسیر واقعی package بازتولید و روی شواهد خام کنترل شد. مسئله این است که مدل‌های جدید روی split تصادفی/OOF بهبود بزرگی نشان می‌دهند، اما روی نمونه‌های دارای session/channel/quality shift تقریباً به سطح کنترل تاریخی برمی‌گردند. خطا نیز تقریباً کاملاً در مرز known/unknown است، نه در تشخیص شناسه‌ی known.

برای submission بعدی، قانون ساده و پایدار `tail_top5_mean_plus_ood` انتخاب شد. این قانون به‌جای جمع‌کردن احتمال هر 554 pseudo-cluster ناشناخته، میانگین پنج شاهد قوی را با OOD head ترکیب می‌کند. در هفت calibration seed، تنها خانواده‌ی top-k/logmeanexp بود که در همه‌ی seedها هم Random و هم Hard را بدون افت recall ناشناخته بهبود داد. نسخه‌ی بسته‌بندی‌شده از Top-5 استفاده می‌کند چون کمترین حساسیت به seed را داشت.

## 1. قفل‌کردن کنترل تاریخی 0.9625

- checkpoint واقعی: `checkpoints/modelrigestry/campp_best (5).pt`
- SHA-256 کوتاه: `ff5108b0e037`
- زمان checkpoint: 2026-08-19 21:39
- commit map: `4a47c98` در 2026-08-19 22:21
- ساختار مدل: CAM++ قدیمی 447-way با 446 known و OOD head
- تصمیم لیدربرد: `alpha=0.35, kappa=24, tau=0, lambda_unknown=0.5`
- 554 centroid ناشناخته در inference به مدل 447-way اضافه می‌شدند؛ checkpoint تاریخی یک مدل 1001-way نبود.

artifactهای centroid و map جاری پروژه با ارسال 19 اوت یکسان نبودند. به همین دلیل map مستقیماً از commit بازیابی شد، split با کد همان commit بازسازی شد، 4459 embedding استخراج شد و centroidها در فضای دقیق checkpoint تاریخی دوباره ساخته شدند. ترتیب 891 فایل validation با cache قدیمی نیز دقیقاً یکسان بود.

## 2. نتایج بازتولیدشده

| سیستم | Validation تصادفی مشترک | Hard | Leaderboard Macro-F1 |
|---|---:|---:|---:|
| کنترل تاریخی 0.9625 | 0.95781 | 0.92344 | 0.96250 |
| package فعلی no-proto/metric 60/40 | 0.97238 | 0.92311 | 0.96060 |
| candidate ثابت Top-5 | 0.97536 | میانگین 0.93153 | هنوز ارسال نشده |

نکته‌ی کلیدی: ترکیب جدید روی Random حدود 0.0146 بهتر از کنترل تاریخی است، اما روی Hard تقریباً هیچ برتری ندارد. این دقیقاً توضیح می‌دهد چرا ranking محلی روی hidden leaderboard حفظ نشده است.

## 3. ماهیت خطا

### package فعلی روی Random

- known accuracy: 0.97309
- unknown recall: 0.99775
- known → unknown: 12
- known → wrong known: 0
- unknown → known: 1

### package فعلی روی Hard

- known accuracy: 0.92377
- unknown recall: 0.99775
- known → unknown: 34
- known → wrong known: 0
- unknown → known: 1

در نتیجه classifier هویت شناخته‌شده تقریباً بی‌خطاست؛ افت امتیاز از reject شدن knownهای سخت می‌آید. افزایش ظرفیت classifier یا ensemble برابر foldها، بدون حل rejection boundary، مسیر اصلی نیست.

### تحلیل کانال و کیفیت روی validation مشترک

| bucket | تعداد | known accuracy package | known accuracy Top-5 |
|---|---:|---:|---:|
| clean speech | 859 | 0.9883 | 0.9883 |
| low-SNR speech | 13 | 0.8750 | 0.8750 |
| non-speech/very-low-energy | 19 | 0.4000 | 0.6000 |

بخش عمده‌ی شکست‌ها به فایل‌های کم‌گفتار، بسیار آرام یا channel-shifted تعلق دارد. نتیجه با EDA قبلی سازگار است: duration و VAD پایین، RMS بسیار کم و flatness بالا در knownهای ردشده فراوان‌تر بود.

## 4. چرا `sum(554)` مسئله‌دار است

در مدل 1001-way، منطق قبلی تمام probability mass مربوط به 554 pseudo-cluster را جمع و به unknown منتقل می‌کرد. این score با تعداد clusterها رشد می‌کند و حتی شواهد کوچک و پخش‌شده می‌توانند یک known سخت را رد کنند. این همان cardinality bias است.

قانون جدید:

1. probabilityهای 446 known دو مدل با وزن 0.6/0.4 ترکیب می‌شوند.
2. از tail ناشناخته فقط میانگین پنج مقدار بزرگ استفاده می‌شود.
3. log-ratio این مقدار به قوی‌ترین known محاسبه می‌شود.
4. نصف logit خروجی OOD مدل no-proto به score اضافه می‌شود.
5. اگر score از threshold ثابت `-2.279` بزرگ‌تر باشد، نمونه unknown است؛ در غیر این صورت شناسه‌ی قوی‌ترین known انتخاب می‌شود.

## 5. Ablation قوانین تصمیم

اعداد زیر مربوط به seed 42 هستند؛ threshold فقط روی calibration جدا انتخاب شده است.

| قانون | Random Macro-F1 | Hard Macro-F1 | Random unknown recall |
|---|---:|---:|---:|
| package argmax | 0.97238 | 0.92311 | 0.99775 |
| sum با threshold مجزا | 0.97536 | 0.93729 | 0.99326 |
| max-tail + OOD | 0.97536 | 0.93057 | 0.99326 |
| top-3 mean + OOD | 0.97461 | 0.93430 | 0.99101 |
| top-5 mean + OOD | 0.97536 | 0.93505 | 0.99326 |
| logmeanexp/mean-tail + OOD | 0.97686 | 0.93654 | 0.99775 |
| linear evidence | 0.97536 | 0.93803 | 0.99326 |

Linear evidence روی یک split کمی بهتر بود، اما در چند calibration seed افت کرد و رد شد. logmeanexp بهترین Random تک-seed را داشت، ولی Top-5 در هفت seed پایدارتر بود.

## 6. پایداری candidate ثابت

thresholdهای بهینه‌ی هفت seed بین `-2.794` و `-2.000` بودند و median آن‌ها `-2.280059` بود. تنها فایل در فاصله‌ی 0.01 از median یک known بود که تغییر batch-size در FP16 score آن را حدود 0.00016 جابه‌جا می‌کرد. threshold نهایی به `-2.279` منتقل شد؛ این margin هیچ نمونه‌ی دیگری را تغییر نمی‌دهد.

- Random Macro-F1 ثابت: 0.97536
- Random known accuracy: 0.97758
- Random unknown recall: 0.99326
- Hard Macro-F1 میانگین هفت split: 0.93153
- حداقل بهبود Hard نسبت به package: +0.00597
- میانگین بهبود Hard: +0.01056
- بیشترین بهبود Hard: +0.01493

## 7. Candidate آماده‌ی ارسال

- فایل: `submission_no-proto_metric-only_60-40_top5.zip`
- حجم: 146.68 MiB
- SHA-256: `567eef9f74ace94af07e757ba2e78b42085395371c0cdf4b9340456c79ee16d8`
- سقف مسابقه: 1 GiB؛ بسته حدود 14.3% سقف است.
- smoke test ZIP: پاس
- تطبیق inference واقعی با cache روی 24 نمونه‌ی مرزی: 0 mismatch
- تست پروژه: 250 passed، 41 skipped

این candidate از نظر مهندسی آماده‌ی ارسال است، اما رسیدن آن به بالای 0.97 روی leaderboard تضمین نمی‌شود. شواهد محلی می‌گویند احتمال بهبود نسبت به submission فعلی خوب است؛ با این حال hidden shift هنوز متغیر تعیین‌کننده است.

## 8. مسیر واقعی رسیدن به 0.97

### اولویت P0 — ارسال Top-5 برای گرفتن یک مشاهده‌ی leaderboard

هدف این ارسال اندازه‌گیری مستقیم اثر حذف cardinality bias است. اگر Macro-F1 از 0.96060 به‌طور معنادار رشد کند، لایه‌ی تصمیم جدید تأیید می‌شود. اگر ثابت بماند، bottleneck تقریباً کامل در representation/condition robustness است.

### اولویت P1 — Known-first CAM++ با hard-condition training

معماری پیشنهادی:

- ArcFace اصلی فقط 446 known را طبقه‌بندی کند.
- OOD head باینری و مستقل باقی بماند.
- pseudo-clusterهای unknown یک auxiliary metric branch با وزن کم و confidence weighting باشند؛ وارد softmax اصلی 446-way نشوند.
- checkpoint selection فقط با Random val انجام نشود؛ معیار ترکیبی Random Macro-F1، Hard Macro-F1 و حداقل unknown recall استفاده شود.
- sampling برای knownهای VAD پایین، RMS پایین، codec/channel shift و window disagreement وزن بیشتری بگیرد.

علت: pseudo-labelهای unknown purity حدود 0.935 و ARI حدود 0.731 دارند. قرار دادن آن‌ها به‌عنوان 554 کلاس هم‌ارز known، noise را به مرز اصلی classifier منتقل می‌کند.

### اولویت P2 — دوهدی 446 + OOD بدون pseudo-tail در softmax

این آزمایش کنترل معماری P1 است:

- 446-way ArcFace
- OOD binary head
- metric projection/contrastive loss برای unknownها بدون class id سخت
- همان augmentation و hard sampling آزمایش P1

اگر P2 از P1 بهتر شود، pseudo-cluster auxiliary هنوز نویزی است. اگر P1 بهتر شود، cluster supervision مفید است ولی باید از output competition جدا بماند.

### اولویت P3 — condition-aware validation و checkpoint selection

- Random: val تاریخی ثابت
- Hard: یک فایل با بیشترین leave-one-out distance برای هر known + unknownهای val
- Channel: گزارش جدا برای clean، low-SNR و non-speech
- هر threshold فقط روی calibration جدا tune شود.
- هیچ ensemble یا checkpoint بر اساس fold score تنها انتخاب نشود.

### اولویت P4 — ensemble پس از بهبود diversity واقعی

ensemble برابر سه P0 فعلی دوباره اجرا نشود. foldها known accuracyهای متفاوت دارند و خطاهای rejection هم‌بسته‌اند. ensemble بعدی باید بر اساس OOF/Hard و با وزن‌های بهینه یا gating شرطی ساخته شود؛ یک مدل CAM++ condition-robust و یک encoder واقعاً متفاوت مثل ECAPA، در صورت تأیید Hard، ارزش بیشتری از سه CAM++ مشابه دارند.

## 9. میزان اطمینان

- خرابی‌نبودن package و ریشه‌ی validation shift: زیاد، حدود 0.9
- تشخیص known-rejection به‌عنوان bottleneck اصلی: بسیار زیاد، بیش از 0.95
- بهتر بودن Top-5 از package فعلی روی توزیع‌های محلی: زیاد، حدود 0.85
- بهتر شدن Top-5 روی leaderboard: متوسط رو به زیاد، حدود 0.65؛ hidden labels در دسترس نیست.
- رسیدن همین candidate بدون retraining به بالای 0.97 leaderboard: پایین تا متوسط؛ این submission یک اصلاح تصمیم است، نه درمان کامل robustness.

## 10. خروجی‌های قابل بازتولید

- `reports/generated/forensic_decision_audit.json`
- `reports/generated/open_set_decision_ablation.json`
- `reports/generated/open_set_decision_errors.csv`
- `scripts/forensic_decision_audit.py`
- `scripts/dump_decision_evidence.py`
- `scripts/analyze_open_set_decisions.py`
- `scripts/verify_open_set_rule_equivalence.py`

تمام اجرای تحلیل و verification با `uv run --no-sync` انجام شده است.
