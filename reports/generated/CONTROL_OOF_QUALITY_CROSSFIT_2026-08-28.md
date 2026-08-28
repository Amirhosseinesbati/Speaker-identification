# تحلیل کیفیت OOF سه‌Fold و پیش‌ثبت تنها آزمایش بعدی

تاریخ تحلیل: 2026-08-28

خانوادهٔ مرجع: `p0-campp-known446-ood-control-oof`

وضعیت کمپین هنگام تحلیل: `ANALYZING`، بدون Run فعال

پروتکل: leave-one-control-fold-out، بدون استفاده از leaderboard

## حکم اجرایی

هیچ decision rule کیفیت‌محور مجاز نشد و سیاست submission دست‌نخورده می‌ماند:
**Raw probability-average با argmax مستقیم**. افزودن کیفیت صوت به boundary به‌تنهایی
امتیاز را کاهش داد. مدل ترکیبی confidence+quality در هر سه Fold Macro-F1 را بالا
برد، اما Known Accuracy تجمیعی را `0.00403950` پایین آورد که چهار برابر guardrail
ازپیش‌ثبت‌شدهٔ `0.001` است؛ بنابراین با وجود gain ظاهری، رد شد.

این نتیجه نشان می‌دهد کیفیت پایین یک شاخص قوی برای نمونه‌های سخت است، اما یک
post-processing سراسری نمی‌تواند false rejectهای known و false acceptهای unknown
را هم‌زمان و بدون هزینه متوازن کند. هیچ threshold، fallback، blend یا training
جدید به‌طور خودکار فعال نمی‌شود.

## داده و سلامت تحلیل

- هر `4447` نمونهٔ OOF یکتا و بدون overlap تحلیل شد.
- استخراج ویژگی صوت برای `4447/4447` فایل موفق بود و failure برابر صفر است.
- metadata مستقیم session/channel در دادهٔ در دسترس وجود نداشت؛ duration، RMS،
  peak، سهم نمونه‌های nonzero، clipping، speech-active fraction و تعداد window
  یکتای مؤثر به‌عنوان proxyهای condition استفاده شدند.
- runtime همیشه هشت tensor برمی‌گرداند؛ `effective_unique_windows` تعداد windowهای
  واقعاً متمایز پیش از padding را می‌شمارد.
- هیچ محاسبهٔ GPU، تغییر checkout یا استفاده‌ای از leaderboard انجام نشد.

## امضای کیفیت خطاها

| گروه | تعداد | median duration | median RMS | median active fraction | median window یکتا |
|---|---:|---:|---:|---:|---:|
| صحیح | 4210 | `60.629s` | `0.092440` | `0.8286` | `8` |
| known → unknown | 70 | `13.909s` | `0.018409` | `0.7070` | `3` |
| unknown → known | 128 | `52.480s` | `0.067772` | `0.8108` | `8` |
| known → known اشتباه | 39 | `5.633s` | `0.000112` | `0.0238` | `1` |

knownهای false-reject به‌طور محسوسی کوتاه‌تر، کم‌انرژی‌تر و دارای window مؤثر
کمترند. خطاهای اشتباه میان دو known حتی شدیدتر روی صوت تقریباً خالی متمرکزند.
در مقابل، unknownهای false-accept از نظر duration و RMS به نمونه‌های صحیح نزدیک‌ترند؛
پس کیفیت خام برای اصلاح جهت دوم مرز اطلاعات کافی ندارد.

در تفکیک known false-reject از known صحیح، separation-AUC برای duration برابر
`0.82188`، RMS برابر `0.81235`، peak برابر `0.79938` و تعداد window مؤثر برابر
`0.77415` بود. برای unknown false-accept، بهترین proxy کیفیت فقط تعداد window با
`0.65720` بود، در حالی که scoreهای confidence مانند `max_known` و `known_margin`
به‌ترتیب `0.98230` و `0.92856` separation داشتند. این عدم تقارن علت علمی شکست
quality-only gate است.

## ارزیابی cross-fit ازپیش‌تعریف‌شده

در هر تکرار، مدل و threshold فقط روی دو Fold دیگر fit/انتخاب شد و سپس روی Fold
نگه‌داشته‌شده ارزیابی شد. شرط پذیرش: gain تجمیعی حداقل `0.003`، gain نامنفی در
هر سه Fold و افت حداکثر `0.001` برای هر کدام از Known Accuracy و OOD-F1.

| Candidate | ΔF0 | ΔF1 | ΔF2 | Δ Macro تجمیعی | Δ Known | Δ OOD-F1 | حکم |
|---|---:|---:|---:|---:|---:|---:|---|
| boundary-only | `0` | `0` | `0` | `0` | `0` | `0` | رد؛ بدون gain |
| boundary + quality | `-0.001214` | `-0.003064` | `-0.004295` | `-0.003014` | `-0.000449` | `-0.003945` | رد |
| confidence + quality | `+0.004325` | `+0.003002` | `+0.003740` | `+0.005852` | `-0.004039` | `+0.007134` | رد؛ نقض Known guardrail |

Candidate سوم Macro-F1 را از `0.94388860` به `0.94974012` و OOD-F1 را از
`0.95479452` به `0.96192836` رساند، اما Known Accuracy از `0.95107720` به
`0.94703770` افت کرد. نرخ unknown false-accept از `5.768%` به `3.785%` کم شد،
ولی known reject از `3.142%` به `3.815%` بالا رفت. thresholdهای منتخب سه تکرار
نیز `0.575`، `0.530` و `0.625` بودند؛ این پراکندگی همراه با نقض guardrail اجازهٔ
استفاده از candidate را نمی‌دهد.

## tail کلاس‌های known

کلاس‌های `414` با support چهار و `360` با support پنج دارای F1 صفر بودند. کلاس‌های
`148` و `377` نیز به‌ترتیب F1 برابر `0.1143` و `0.1818` داشتند. این tail هم
support بسیار کوچک دارد و هم در چند مورد median duration/RMS بسیار پایین است؛
در نتیجه class-specific fallback یا prior می‌تواند به‌سادگی overfit کند و مجاز
نیست. راه‌حل بعدی باید representation مکمل بسازد، نه جدول استثنا برای کلاس‌ها.

## پیش‌ثبت تاریخی و لغوشده ـ TitaNet known-first Fold 0

> وضعیت 2026-08-28: این پیشنهاد با تصمیم صریح کاربر لغو و با پیش‌ثبت
> `ERES2NET_FOLD0_PREREGISTRATION_2026-08-28.md` جایگزین شد. TitaNet نباید
> train شود. جزئیات زیر فقط برای حفظ سابقهٔ تصمیم علمی باقی مانده‌اند.

### فرضیه

یک **TitaNet-Large کاملاً frozen** با head جدید `446-way ArcFace + binary OOD`
می‌تواند خطاهایی متفاوت از CAM++ تولید کند و بدون انتقال pseudo-clusterها به
softmax اصلی، منبع representation مکمل بسازد. دلیل انتخاب از شواهد موجود است:
در benchmark خام، TitaNet بهترین OOD AUC (`0.9626`) و بهترین Macro-F1 (`0.9331`)
را میان encoderهای frozen داشت؛ معماری 192بعدی آن نیز با CAM++ متفاوت است.
WavLM خام با Macro-F1 کمتر از `0.50` رد شده و ECAPA/ERes2Net در همان benchmark
از TitaNet ضعیف‌تر بودند.

### قرارداد ثابت Run

- فقط Fold 0 و همان split سه‌Fold با seed `42`؛ هیچ Fold دیگری پیشاپیش مجاز نیست.
- encoder محلی `titanet_large.nemo` کاملاً frozen؛ تنها ArcFace 446کلاسه و OOD
  head آموزش می‌بینند.
- مسیر known-first، همان augmentation، window policy و loss weights کنترل؛ بدون
  AuxMetric، proto loss، low-energy fallback یا pseudo-class در softmax اصلی.
- `Raw probability-average + argmax` تنها تصمیم مستقل است.
- تنها ensemble diagnostic مجاز، میانگین احتمال ثابت `50/50` با checkpoint Raw
  کنترل Fold 0 است؛ هیچ وزن یا threshold جست‌وجو نمی‌شود.
- batch با preflight سیستمی از مجموعهٔ ثابت `48، 32، 24` انتخاب می‌شود: بزرگ‌ترین
  مقدار بدون OOM و با peak VRAM حداکثر `21.5 GiB`. این انتخاب علمی نیست.
- سقف `120` ایپاک، patience برابر `15`، حداکثر زمان `6h` و هزینهٔ افزوده حداکثر
  `$2.50`.

### gate ادامه به Foldهای بعد

همهٔ شروط زیر باید هم‌زمان برقرار باشند:

1. Run، OOF، class map، config، history، checkpointها و MLflow کامل و hash-verified باشند.
2. Macro-F1 مستقل TitaNet روی Fold 0 حداقل `0.93692119` باشد؛ یعنی بیش از `0.010`
   از کنترل CAM++ (`0.94692119`) عقب نماند.
3. fixed 50/50 probability ensemble نسبت به CAM++ Fold 0 حداقل `+0.002` Macro-F1
   gain بسازد، gain مستقل از threshold باشد و Known Accuracy یا OOD-F1 را بیش از
   `0.001` پایین نیاورد.
4. TitaNet حداقل `25%` خطاهای CAM++ را به‌تنهایی نجات دهد تا diversity واقعی،
   نه فقط نوسان score، ثابت شود.

اگر حتی یکی از شروط 2 تا 4 شکست بخورد، TitaNet Foldهای 1 و 2 اجرا نمی‌شوند و
candidate رد می‌شود. عبور Fold 0 فقط مجوز تحلیل و پیش‌ثبت Fold 1 است، نه مجوز
خودکار اجرای آن یا ساخت submission.

### stop rule

- NaN، خرابی split/provenance یا mismatch artifact: توقف فوری.
- OOM پس از یک بار کاهش batch طبق grid ثابت: توقف و گزارش؛ grid جدید ساخته نمی‌شود.
- اگر تا پایان ایپاک 40 بهترین Raw Macro-F1 کمتر از `0.90` باشد، توقف futility.
- عبور از `6h` یا `$2.50` هزینهٔ افزوده: توقف امن و حفظ artifactها.

## تصمیم فعلی

تحلیل quality-aware تمام شد و همهٔ candidateهای decision-layer رد شدند. پیشنهاد
TitaNet بعداً با تصمیم کاربر لغو شد. گام مجاز فعلی فقط ERes2NetV2 Fold 0 تحت
قرارداد مستقل `ERES2NET_FOLD0_PREREGISTRATION_2026-08-28.md` است. Instance برای
ادامهٔ مصوب روشن می‌ماند.
