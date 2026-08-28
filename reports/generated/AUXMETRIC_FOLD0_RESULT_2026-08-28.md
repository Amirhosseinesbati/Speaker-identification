# گزارش نهایی آزمایش AuxMetric — Fold 0

تاریخ تحلیل: 2026-08-28  
کمپین: `iaaa-speaker-rank1-20260827`  
Control: `p0-campp-known446-ood-control-oof-f0`  
Candidate: `p0-campp-known446-ood-auxmetric-oof-f0`

## جمع‌بندی تصمیم

AuxMetric در Fold 0 به‌عنوان مدل مستقل رد شد و بدون شاهد تأییدی به Foldهای بعد
توسعه نمی‌یابد. بهترین Macro-F1 مسیر واقعی submission آن `0.9433028273` است؛
درحالی‌که Control به `0.9469211906` رسیده است. افت مستقل `-0.0036183633`، افت
Known Accuracy برابر `-0.0044843049` و افت OOD-F1 برابر `-0.0070187810` است.
هیچ blend ثابت بررسی‌شده‌ای نیز Control را شکست نداد.

بااین‌حال فرضیهٔ AuxMetric کاملاً بی‌اطلاعات نیست: ۱۱ خطای Control را نجات
می‌دهد که همگی OOD و عمدتاً کوتاه‌اند. این اثر فقط یک سرنخ برای طراحی یک
متخصص کوتاه‌گفتار یا Gate ازپیش‌ثبت‌شده است؛ انتخاب Gate از همین Fold ممنوع
است و نتیجهٔ فعلی مجوز اجرای مستقیم AuxMetric Fold 1 نیست.

## هویت و بازتولیدپذیری Run

| مورد | مقدار |
|---|---:|
| commit علمی Run | `40bdc5bb9d233d03cee3ef1fa309662a766bb757` |
| شروع | `2026-08-27T21:48:11Z` |
| پایان | `2026-08-28T02:24:21.889818Z` |
| وضعیت پایان | early-stop موفق، exit code صفر |
| آخرین epoch | 95 |
| علت توقف | ۲۰ epoch بدون بهبود بهترین Raw |
| checkpoint منتخب | EMA، epoch 93 |
| MLflow Run ID | `dd93cd84fffd4e40991fe4182d81d2ee` |
| OOF Aux SHA-256 | `efaefa6861838f611a7de8879de4937579c951ae23ef8e8ea415ac9d610c550b` |
| OOF Control SHA-256 | `221aa9edb00e13cbf98a629a4dddede8eb5835460fafe513fb5ffc9d1ea89394` |

OOF هر دو Run شامل ۱۶۳۲ فایل، ۴۴۷ کلاس رقابت و split یکسان
`kfold/fold0/folds3/seed42` است. filename، label، class map و ابعاد احتمال
به‌صورت مستقل کنترل شده‌اند.

## منحنی و checkpointها

| وزن/تجمیع | بهترین Macro-F1 | epoch |
|---|---:|---:|
| Aux Raw probability-average | `0.9431993659` | 75 |
| Aux Raw logit-average | `0.9406446072` | 92 |
| Aux EMA probability-average | **`0.9433028273`** | **93** |
| Aux EMA logit-average | `0.9406158899` | 94 |
| آخرین Aux Raw probability-average | `0.9426341589` | 95 |
| آخرین Aux EMA probability-average | `0.9430013405` | 95 |
| Control Raw probability-average | **`0.9469211906`** | **112** |
| Control EMA probability-average | `0.9444993028` | 131 |

Probability-average در Aux و Control بهتر از logit-average است. EMA در Aux
قلهٔ Raw را فقط حدود `+0.0001035` بهبود می‌دهد؛ پس مزیت اصلی Run از هموارسازی
بسیار کوچک می‌آید، نه تغییر پایدار representation. افت آخرین epoch نسبت به قله
کوچک است و early stopping به‌درستی از ادامهٔ آموزش بدون شاهد جلوگیری کرده است.

## معیارهای checkpoint منتخب

| معیار | Aux EMA e93 | Control Raw e112 | اختلاف Aux-Control |
|---|---:|---:|---:|
| probability-average Macro-F1 | `0.9433028273` | `0.9469211906` | `-0.0036183633` |
| Known Accuracy | `0.9529147982` | `0.9573991031` | `-0.0044843049` |
| OOD-F1 | `0.9516019087` | `0.9586206897` | `-0.0070187810` |
| Overall Accuracy | `0.9485294118` | — | — |

Aux در هر دو مؤلفهٔ Known و OOD افت دارد؛ بنابراین اختلاف Macro-F1 فقط ناشی
از trade-off یک طرف نیست و بهبود مستقیم معماری تأیید نمی‌شود.

## تفسیر صحیح Loss

logger تاریخی Aux جزء prototype را داخل `train_loss` ثبت نمی‌کرد و مجموع وزن
اجزای primary آن `0.95` بود. بنابراین Loss خام Aux با Control قابل‌مقایسه نیست.
پس از بازگرداندن مقیاس primary به واحد مشترک در epoch 95:

| Loss primary تصحیح‌شده | Aux | Control | اختلاف |
|---|---:|---:|---:|
| Train | `1.5839071408` | `1.5799668908` | `+0.0039402500` |
| Validation | `1.2725205097` | `1.2715652042` | `+0.0009553055` |

میانگین اختلاف Validation Primary Loss در ۱۰ epoch پایانی حدود `+0.001875`
به زیان Aux است. این اختلاف کوچک است، اما با افت F1 هم‌جهت است و شاهدی برای
برتری objective کمکی نمی‌دهد. اصلاح logger برای Runهای بعدی ثبت شده است تا
اجزای primary/prototype جدا و قابل‌ممیزی باشند.

## مکمل‌بودن خطاهای OOF

| شاخص | مقدار |
|---|---:|
| خطای Aux | 84 |
| خطای Control | 83 |
| خطای مشترک | 72 |
| فقط Aux صحیح | 11 |
| فقط Control صحیح | 12 |
| Jaccard مجموعهٔ خطا | `0.757895` |
| سهم خطاهای مشترک از مجموعهٔ خطای کوچک‌تر | `0.86747` |
| بازیابی خطاهای Control توسط Aux | `13.253%` |
| Oracle Macro-F1 | `0.9486057691` |
| Oracle gain نسبت به Control | `+0.0016845785` |

هر ۱۱ نجات Aux مربوط به OOD است؛ Aux هیچ خطای Known مدل Control را نجات
نمی‌دهد. از ۱۲ نجات Control، هشت مورد OOD و چهار مورد Known است. Oracle gain
واقعی اما کمتر از شرط `+0.0020` سیاست پیش‌ثبت‌شده است و مدل Aux نیز بیش از
حد مجاز `-0.0010` از Control عقب است؛ درنتیجه شرط توسعهٔ Aux برقرار نیست.

Blendهای ثابت و صرفاً توصیفی نیز Control را شکست ندادند:

| وزن Aux/Control | Macro-F1 |
|---|---:|
| 25/75 | `0.9461026380` |
| 50/50 | `0.9448318570` |
| 75/25 | `0.9440607900` |

## تشخیص فایل‌به‌فایل

در ۲۳ نمونه‌ای که فقط یکی از دو مدل صحیح است، تفاوت‌ها ساختاری‌اند:

| گروه | تعداد | میانگین/میانهٔ مدت | نکتهٔ اصلی |
|---|---:|---:|---|
| فقط Aux صحیح | 11 | `6.10s / 5.04s` | همگی OOD؛ delta احتمال OOD مثبت |
| فقط Control صحیح | 12 | `37.72s / 50.22s` | ۸ OOD و ۴ Known؛ عمدتاً بلندتر |
| هر دو غلط | 72 | `31.94s / 13.91s` | خطاهای سخت و مشترک |
| هر دو صحیح | 1537 | `60.65s / 60.16s` | بخش پایدار داده |

برای گروه فقط Aux صحیح، میانگین delta احتمال OOD برابر `+0.0312` و میانگین
delta حاشیهٔ unknown برابر `+0.2536` است. برای گروه فقط Control صحیح این دو
عدد به‌ترتیب `-0.0474` و `-0.0384` هستند. embedding cosine نیز در نجات‌های Aux
کمتر است (`0.9375` در برابر `0.9674`)؛ یعنی objective کمکی در همان نمونه‌های
کوتاه representation را بیشتر جابه‌جا کرده است.

AUCهای تشخیصی فقط روی همین ۲۳ نمونه، نه به‌عنوان تخمین تعمیم:

| سیگنال | separability | جهت متمایل به Aux |
|---|---:|---|
| embedding cosine | `0.8485` | کمتر |
| unknown-margin delta | `0.7955` | بیشتر |
| OOD-probability delta | `0.7879` | بیشتر |
| duration | `0.7689` | کوتاه‌تر |
| candidate entropy | `0.7121` | بیشتر |
| probability L1 | `0.6970` | بیشتر |

نگاشت cluster ناشناختهٔ train-only برای فایل‌های validation پوشش نداشت؛ خالی
بودن خلاصهٔ cluster به معنی نبود اثر cluster نیست، بلکه این سیگنال در Fold 0
قابل‌دسترسی نبوده است.

Gate ثابت «Aux فقط وقتی Aux=unknown، Control=known و delta احتمال OOD ≥ 0»
روی همین Fold به `0.9471778442` رسید؛ فقط `+0.0002566536` بالاتر از Control.
این مقدار بسیار کوچک، پس‌نگرانه و همراه با Known/OOD ضعیف‌تر از Control است.
بنابراین evidence قابل‌استفاده برای submission یا انتخاب threshold نیست.

## سلامت Artifact و MLflow

- ۲۶ artifact رسید کمپین مستقلاً re-hash شدند؛ mismatch برابر صفر است.
- MLflow دارای ۴۱ parameter، ۴۰ کلید metric و history کامل ۹۵ مرحله است.
- Backfill هدفمند ۱۲ artifact گمشده با حجم `240,739,271` بایت را اضافه کرد.
- پس از Backfill، ۴۴ artifact با حجم `454,324,788` بایت روی MLflow وجود داشت.
- هر ۱۲ فایل Backfill دوباره download و SHA-256 آن با منبع تطبیق داده شد؛
  `missing_paths`، `size_mismatches` و `hash_mismatches` همگی خالی‌اند.
- مدل‌های `best/latest/init`، config/profile، campaign state، لاگ pipeline و
  supervisor، تحلیل‌های OOF/training، سیاست gate و marker تلگرام ثبت شده‌اند.

پیوند Run:
`https://dagshub.com/amiresbati52/Speaker-identification.mlflow/#/experiments/0/runs/dd93cd84fffd4e40991fe4182d81d2ee`

## تصمیم بودجه و گام بعدی

1. AuxMetric فعلی به Fold 1 گسترش نمی‌یابد.
2. پیش از هر Run GPU، تحلیل کل ۱۶۳۲ نمونه با باکت‌های ثابت مدت‌زمان، هش فایل و
   PCM، و گروه‌های probability کاملاً یکسان انجام می‌شود.
3. Gateهای مدت‌زمان `<2s`، `<5s` و `<10s` صرفاً exploratory هستند؛ چون پس از
   مشاهدهٔ Fold 0 اضافه شده‌اند، انتخاب بهترین آن‌ها از همین Fold ممنوع است.
4. فقط اگر یک قاعدهٔ سادهٔ کوتاه‌گفتار بدون نشانهٔ duplicate/leakage پیدا شود،
   قاعده پیش از Fold 1 freeze و یک Fold تأییدی واحد اجرا می‌شود.
5. در غیر این صورت Aux بازنشسته و بودجه به Control چند-Fold و سپس encoder مکمل
   دارای error decorrelation واقعی منتقل می‌شود.

## وضعیت تحلیل تکمیلی

ابزار بازتولیدپذیر تحلیل مدت‌زمان و identity در commit `a92550c` اضافه و با ۱۰
تست محدود اعتبارسنجی شده است. اجرای آن روی Worker و الحاق hash خروجی به MLflow
پس از بازیابی دسترسی CLI محیط Vast انجام می‌شود؛ این بخش تا آن زمان باز است و
هیچ نتیجهٔ Gate مدت‌زمانی در این گزارش نهایی تلقی نمی‌شود.

