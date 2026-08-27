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

## state machine و guardها

ledger runtime در `data/experiments/campaign_state.json` و event ledger در JSONL نگهداری می‌شود. state به‌صورت atomic جایگزین و transitionها append-only ثبت می‌شوند.

مسیر اصلی:

`PREFLIGHT → READY → RUNNING_EXPERIMENT → ANALYZING → WAITING_FOR_LEADERBOARD`

در خطا، timeout یا عدم یکپارچگی، مسیر به `CAMPAIGN_BLOCKED` یا `STOPPED_FOR_BUDGET` می‌رود. supervisor فقط نام profile ثبت‌شده را می‌پذیرد و arbitrary shell را از کانال کنترل آزمایش اجرا نمی‌کند.

هنگام پایان موفق run، checkpoint/NPZ/JSON/MODEL_CARDهای منتخب hash می‌شوند و receipt در state ذخیره می‌شود. ورود به `WAITING_FOR_LEADERBOARD` فقط با artifact موجود و SHA256 ممکن است.

## رخداد یکپارچگی داده و تصمیم

نسخهٔ اولیهٔ worker شامل 4529 فایل صوتی بود، اما 87 فایل برخلاف سیستم مرجع صفر‌بایتی بودند. conversion نتیجهٔ `4438 موفق / 91 ناموفق` داد. شمارش سیستم مرجع ثابت کرد فایل صفر‌بایتی وجود ندارد؛ بنابراین آموزش متوقف شد.

ابزار `repair_dvc_zero_files.py` دقیقاً workspace و cache objectهای صفر را با manifest دایرکتوری DVC تطبیق داد و آن‌ها را، بدون حذف، به quarantine منتقل کرد. `dvc pull` هر 87 object را دوباره fetch کرد، اما remote همان objectهای صفر را بازگرداند؛ پس remote data store نیز برای این subset ناسالم است.

راه بازیابی انتخاب‌شده:

1. فقط filenameهای همان 87 object از manifest گرفته شد.
2. نسخهٔ سالم محلی آن‌ها 0.332 GiB بود؛ کل dataset دوباره منتقل نمی‌شود.
3. برای کاهش فشار و زمان، WAVهای deterministic ازقبل‌پردازش‌شدهٔ همان subset به archive 131.5 MiB تبدیل شدند.
4. archive با انتقال resumable ارسال می‌شود؛ extract فقط پس از تطابق SHA256 مبدأ/مقصد مجاز است.
5. پس از extract، count/metadata/clean-split و preflight دوباره اجرا می‌شوند.

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
- بازیابی subset داده در حال انتقال resumable است.
- GPU هنوز وارد آموزش نشده است.
