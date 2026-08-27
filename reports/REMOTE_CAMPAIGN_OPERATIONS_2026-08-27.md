# گزارش عملیاتی کمپین Remote-First برای عبور از 0.973

تاریخ شروع: 2026-08-27  
Instance: `48886926`  
Branch مرجع: `feature/Improvement_of_recent_changes`  
هدف: ساخت یک مسیر علمی، قابل‌بازتولید و بودجه‌دار برای عبور پایدار از Macro-F1 برابر 0.973.

## قرارداد اجرایی

- تمام نصب، تست سنگین، preprocessing، inference و train روی worker انجام می‌شود.
- سیستم محلی فقط source code، Git و انتقال artifact ضروری را مدیریت می‌کند؛ GPU محلی استفاده نمی‌شود.
- هر تغییر علمی ابتدا در workspace اصلی اعمال، commit و push می‌شود و سپس worker آن را با `git pull --ff-only` می‌گیرد.
- اجرای علمی با checkout کثیف یا branch عقب/جلوی upstream به‌صورت پیش‌فرض ممنوع است.
- اعلان‌های مهم تلگرام فقط به فارسی و شامل وضعیت، عدد کلیدی، اقدام بعدی و هشدار لازم هستند.
- هیچ مدل فقط به‌دلیل یک val score انتخاب نمی‌شود؛ OOF، exact inference parity، runtime، اندازهٔ package و artifact hash همگی gate هستند.

## وضعیت worker و هزینه

| مورد | مقدار |
|---|---:|
| GPU | RTX 3090، VRAM قابل‌استفاده 23.56 GiB |
| CPU | Ryzen 9 5950X، 32 vCPU مؤثر |
| RAM | حدود 35 GiB |
| Disk | 100 GiB |
| نرخ | 0.171111 دلار در ساعت |
| سقف نرخ | 0.22 دلار در ساعت |
| سقف کل کمپین | 20 دلار |
| سقف هر run | 12 ساعت |
| WAITING keep-alive | 12 ساعت، سپس تصمیم lifecycle |

زمان واقعی شروع billing برابر `2026-08-27T11:56:06Z` در ledger ثبت شده است؛ بنابراین برآورد هزینه از زمان اتصال Codex شروع نمی‌شود و کم‌شماری ندارد.

## acceptance اندازه‌گیری‌شده

- FP16 matmul: median برابر 77.50 TFLOPS و minimum برابر 76.79 TFLOPS.
- H2D: 12.58 GB/s.
- دانلود خارجی: 216.88 Mbps برای payload صد مگابایتی.
- تست‌های P0 اولیه روی خود worker: 19/19 موفق.
- تست‌های state و batch-selection پس از افزودن supervisor: 6/6 موفق.
- تست کامل repository پیش از data gate: 87/87 موفق، با دو warning شناخته‌شدهٔ scheduler در تست‌ها.

نتیجه: worker برای CAM++ gate پذیرفته شده است. انتخاب batch همچنان با step واقعی forward/backward و حداقل 10% headroom انجام می‌شود، نه بر مبنای حدس از VRAM اسمی.

## اصلاحات P0 قبل از آموزش

### 1. sampler واقعی در سطح batch

پیاده‌سازی قدیمی یک لیست تخت از indexها می‌ساخت و آن را به `SubsetRandomSampler` می‌داد. shuffle سراسری sampler مرز batchهای طراحی‌شده را از بین می‌برد؛ بنابراین نسبت OOD/known ادعاشده الزاماً در batch واقعی وجود نداشت.

اکنون `BalancedOODBatchSampler` مستقیماً batchها را yield می‌کند، نسبت دقیق را در هر batch نگه می‌دارد، برای هر epoch shuffle قطعی مستقل دارد و pseudo-idهای خارج از 446 را در known-first به‌درستی OOD می‌بیند.

### 2. یکسان‌سازی معیار checkpoint با submission

مسیر قبلی checkpoint selection ابتدا logits پنجره‌ها را میانگین و سپس sigmoid/softmax را اعمال می‌کرد؛ submission واقعی probability هر پنجره را محاسبه و سپس probabilityها را میانگین می‌گیرد. به‌علت غیرخطی بودن sigmoid/softmax این دو یکی نیستند.

اکنون validation هر دو خروجی را نگه می‌دارد:

- `logit_avg_macro_f1`: فقط diagnostic؛
- `macro_f1`: probability-average دقیق و معیار authoritative برای checkpoint، EMA و early stopping.

### 3. HPO label smoothing

HPO فقط `training.label_smoothing` را تغییر می‌داد، درحالی‌که loss فعال ابتدا `training.loss.speaker.label_smoothing` را می‌خواند. هر دو مسیر اکنون هم‌زمان update می‌شوند؛ بنابراین trial دیگر پارامتر ظاهری و بی‌اثر ندارد.

### 4. hash مستقل از Windows/Linux

preflight cluster-map در Windows موفق و روی Linux fail می‌شد، زیرا hash روی byteهای CRLF/LF محاسبه شده بود. hash متن اکنون ابتدا newline را به LF normalize می‌کند. count فایل، تعداد دقیق 554 خوشه و hash محتوای normalize‌شده همگی کنترل می‌شوند. preflight کامل شش profile پس از اصلاح `status=ok` داد.

Commits مرتبط:

- `dd723c9`: validation parity، batch sampler و HPO.
- `2528261`: state/budget/Telegram supervisor و DVC repair guard.
- `b14e157`: دانلود انتخابی وزن‌ها؛ bootstrap فقط CAM++.
- `568722f` و `e7b5deb`: probe واقعی batch و ساختار تست‌پذیر آن.
- `4bfaf43`: secret loader امن MLflow و snapshot عملیاتی کمپین.

## اتصال امن DagsHub/MLflow

پس از تأیید صریح کاربر، فقط چهار مقدار allowlistشدهٔ `DAGSHUB_USER_TOKEN`،
`DAGSHUB_REPO_OWNER`، `DAGSHUB_REPO_NAME` و `DAGSHUB_TRACKING_URI` از stdin
رمزشدهٔ SSH به worker منتقل شدند. مقدارها هرگز در command argument، stdout، Git یا
گزارش ثبت نشدند. فایل مقصد `/root/.iaaa_mlflow.env` با مالک `root:root` و mode
برابر `0600` ساخته شد. یک درخواست واقعی `MlflowClient.search_experiments` نیز بدون
چاپ URI حاوی credential یا token موفق شد.

اجرای نهایی data gate پس از پوشش همهٔ خرابی‌ها در DagsHub با run id زیر ثبت شد:

`c48774e617494b21ab2c12e9b179750b`

## state machine و guardها

ledger runtime در `data/experiments/campaign_state.json` و event ledger در JSONL نگهداری می‌شود. state به‌صورت atomic جایگزین و transitionها append-only ثبت می‌شوند.

مسیر اصلی:

`PREFLIGHT → READY → RUNNING_EXPERIMENT → ANALYZING → WAITING_FOR_LEADERBOARD`

در خطا، timeout یا عدم یکپارچگی، مسیر به `CAMPAIGN_BLOCKED` یا `STOPPED_FOR_BUDGET` می‌رود. supervisor فقط نام profile ثبت‌شده را می‌پذیرد و arbitrary shell را از کانال کنترل آزمایش اجرا نمی‌کند.

هنگام پایان موفق run، checkpoint/NPZ/JSON/MODEL_CARDهای منتخب hash می‌شوند و receipt در state ذخیره می‌شود. ورود به `WAITING_FOR_LEADERBOARD` فقط با artifact موجود و SHA256 ممکن است.

## رخداد یکپارچگی داده و تصمیم

نسخهٔ اولیهٔ worker شامل 4529 مسیر صوتی بود، اما 87 فایل برخلاف سیستم مرجع صفر‌بایتی بودند. conversion نتیجهٔ `4438 موفق / 91 ناموفق` داد. شمارش سیستم مرجع ثابت کرد فایل صفر‌بایتی وجود ندارد؛ بنابراین آموزش متوقف شد.

ابزار `repair_dvc_zero_files.py` دقیقاً workspace و cache objectهای صفر را با manifest دایرکتوری DVC تطبیق داد و آن‌ها را، بدون حذف، به quarantine منتقل کرد. `dvc pull` هر 87 object را دوباره fetch کرد، اما remote همان objectهای صفر را بازگرداند؛ پس remote data store نیز برای این subset ناسالم است.

بازیابی در سه لایه انجام شد:

1. فقط filenameهای همان 87 object از manifest گرفته شد.
2. نسخهٔ سالم محلی آن‌ها 0.332 GiB بود؛ کل dataset دوباره منتقل نمی‌شود.
3. برای کاهش فشار و زمان، WAVهای deterministic ازقبل‌پردازش‌شدهٔ همان subset به archive 131.5 MiB تبدیل شدند.
4. archive با انتقال resumable ارسال شد و فقط پس از تطابق SHA256 برابر
   `007f38542f83e176856a25344b0389e410527f429bce3eb4960ba8af4301919d`
   استخراج شد؛ تعداد entryها دقیقاً 87 بود.
5. چهار WAV باقیمانده جداگانه منتقل و SHA256 تک‌تک آن‌ها با مبدأ تطبیق داده شد؛
   در نتیجه count به 4529 و missing label path به صفر رسید.

اولین اجرای data gate با وجود conversion ظاهراً سالم، 100 فایل زیر یک ثانیه دید؛
این با EDA مرجع 70 فایل سازگار نبود و gate رد شد. بررسی duration و manifest نشان داد:

- 46 فایل header-only واقعی دیتاست و 24 فایل کوتاه واقعی‌اند؛
- 29 raw غیرصفر در گروه WAVهای یک‌نمونه‌ای MD5 متفاوت از manifest DVC داشتند؛
- یک raw خراب دیگر (`2ec04e18-...`) WAV ناقص 0.996 ثانیه‌ای ساخته بود، درحالی‌که
  مرجع سالم آن 66.389 ثانیه است؛
- در مجموع علاوه بر 87 zero-byte، تعداد 30 raw غیرصفر با checksum غلط روی worker
  کشف شد.

سی WAV سالم در archive دوم با 30 entry و SHA256 برابر
`811cc22b91d1f03c5ac9616ce63f20877289429aa7ade76c3f4e345b648f70a5`
منتقل شد؛ این بسته شامل 29 مورد خراب و یک فایل کوتاه سالم برای تطبیق گروهی بود.
آخرین مورد نزدیک مرز نیز جداگانه با SHA256 برابر
`aec175877a27a330963d4b9aaa3a1c10d069a67135a2c4c66e396d487f42c1d4`
جایگزین شد.

نتیجهٔ data gate نهایی:

| کنترل | نتیجه |
|---|---:|
| WAV برچسب‌خورده | 4529/4529 |
| `<1s` | 70 |
| `>=1s` | 4459 |
| duplicate group | 9 گروه / 69 فایل |
| حذف duplicate/conflict | 62 فایل: 48 conflicting + 14 repeated |
| cleaned labels | 4467 |
| fold-0 train / val | 2819 / 1632 |
| train known / unknown | 1337 / 1482 |
| val known / unknown | 892 / 740 |

ممیزی کامل checksum روی 4530 entry و 4470 cache object یکتا، طی 45 ثانیه حدود
32.45 میلیارد بایت را hash کرد و 152 mismatch در workspace و 152 mismatch در
cache گزارش داد. 87 مورد zero-byte، تعداد 30 مورد non-zero که با duration gate
ریشه‌یابی شدند، چهار WAV مفقودِ جداگانه و 31 mismatch پنهان باقی‌مانده همگی با
WAV سالم محلی پوشش داده شدند. archive سوم 31 entry داشت و SHA256 آن برابر
`302539c82933a8c4d60ee2bb4c3f4ef20aea30dbda9be6e0d83d04bc8ee0fb8e` بود.

نکتهٔ provenance: raw DVC remote همچنان منبع قابل‌اعتماد این subset نیست؛ سلامت
مسیر آموزش فعلی از طریق WAVهای پردازش‌شدهٔ hash-verified برقرار شده است. ابزار جدید
`audit_dvc_integrity.py` برای کنترل read-only checksum همهٔ workspace/cache objectها
نسبت به manifest اضافه شد تا خرابی non-zero در پیش‌پروازهای بعدی پنهان نماند.

## آماده‌سازی augmentation

- MUSAN: تعداد 1606 فایل noise/music دانلود و استخراج شد.
- RIR: تعداد 60000 فایل پاسخ ضربهٔ اتاق دانلود و استخراج شد.
- presence-check قبلی MUSAN فقط `*.wav` مستقیم را می‌دید، درحالی‌که layout رسمی
  زیرپوشه‌ای است؛ به همین علت هر data run استخراج را بیهوده تکرار می‌کرد. check به
  `rglob` تغییر یافت و تست regression برای layout nested اضافه شد.
- dataloader به مسیرهای `musan/noise`، `musan/music` و `rirs` واقعی اشاره دارد؛ پس
  augmentationهای domain دیگر به‌علت نبود داده silently skip نمی‌شوند.
- پس از گیت، archiveهای cache و انتقالیِ تأییدشده حذف شدند؛ دادهٔ استخراج‌شده و
  نسخه‌های محلی حفظ شدند و فضای آزاد worker از 31 به 43 GiB رسید.

## پروفایل batch اندازه‌گیری‌شدهٔ RTX 3090

probe از forward، backward و optimizer step واقعی با 8 window هشت‌ثانیه‌ای و
balanced batch sampler استفاده کرد. خلاصهٔ حالت configured (دو block آخر CAM++
قابل‌آموزش) چنین بود:

| batch | فایل/ثانیه | peak VRAM GiB |
|---:|---:|---:|
| 16 | 26.64 | 2.15 |
| 24 | 28.88 | 3.15 |
| 32 | 29.77 | 4.12 |
| 40 | 30.10 | 5.13 |
| 48 | **30.89** | 6.11 |
| 64 | 28.54 | 8.08 |

در حالت encoder کاملاً frozen، batch 48 برابر 52.86 و batch 64 برابر 53.42
فایل/ثانیه بود؛ سود 64 کمتر از 1.1% است، اما بعد از unfreeze افت throughput دارد.
پس batch پویا ارزش پیچیدگی و تغییر ناگهانی noise scale را ندارد. profile جدید
`vastai_3090_campp` با batch 48، هشت worker و mixed precision به هر شش config
کنترل/aux foldهای 0 تا 2 متصل شد. learning rate در gate اول ثابت می‌ماند تا
مقایسهٔ control/aux با confound هم‌زمان LR آلوده نشود؛ LR scaling فقط برای خانوادهٔ
برنده یک ablation جدا خواهد بود.

پیش‌پرواز invariant نشان داد تعریف این profile در `default_config.yaml` حتی برای
آزمایش‌های قدیمی که از آن استفاده نمی‌کنند، resolved-config و در نتیجه hash آن‌ها را
تغییر می‌دهد. برای جلوگیری از blast radius، تعریف profile فقط داخل همین شش config
هدف نگه داشته شد؛ تنظیمات سراسری و invariant خانواده‌های legacy بدون تغییر باقی
می‌مانند. این خطا پیش از هر train شناسایی شد و هیچ نتیجهٔ آزمایشی را آلوده نکرد.

هیچ train پیش از پایان این gate آغاز نمی‌شود.

## برنامهٔ آزمایش نزدیک

1. تکمیل data-integrity gate و کنترل split-report.
2. model-load smoke برای CAM++ و یک forward روی RTX 3090.
3. batch probe در دو حالت encoder frozen و configured partial-unfreeze با candidateهای 16/24/32/40/48/64.
4. انتخاب operational batch با بیشترین files/s و حداقل 10% VRAM headroom؛ اثر تغییر تعداد optimizer step جداگانه در recipe ثبت می‌شود.
5. fold-0 control و auxmetric با تنها یک تفاوت علمی مجاز.
6. تحلیل OOF exact-path، known/OOD breakdown، calibration و artifact alignment.
7. ادامهٔ فقط خانوادهٔ برنده روی foldهای 1 و 2.
8. fusion cross-fitted، full-data fit ازپیش‌تعریف‌شده، runtime/size verifier و ساخت submission.
9. انتقال package منتخب به سیستم کاربر، ثبت SHA256 و ورود به `WAITING_FOR_LEADERBOARD`.
10. دریافت score واقعی، تحلیل gap و طراحی دور بعد در صورت نیاز.

## وضعیت جاری در زمان این snapshot

- supervisor فعال و state برابر `PREFLIGHT` است.
- وزن CAM++ روی worker حاضر است؛ وزن encoderهای بی‌استفاده دانلود نشده‌اند.
- preflight config/cluster-map موفق است.
- دادهٔ WAV کامل و data gate نهایی با اعداد مرجع موفق است.
- MUSAN و RIR کامل‌اند؛ اصلاح idempotency محلی آمادهٔ commit/push است.
- DagsHub/MLflow احراز هویت شده و data run پاک ثبت شده است.
- auditor کامل DVC اجرا شده و تمام 152 mismatch در مسیر WAV پوشش داده شده‌اند.
- batch probe configured/frozen کامل و operational batch برابر 48 انتخاب شده است.
- GPU هنوز وارد آموزش نشده است.
