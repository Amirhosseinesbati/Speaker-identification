# سیاست پیش‌ثبت‌شدهٔ تصمیم Control در برابر AuxMetric — Fold 0

تاریخ پیش‌ثبت: 2026-08-28  
کمپین: `iaaa-speaker-rank1-20260827`  
Run فعال: `p0-campp-known446-ood-auxmetric-oof-f0`

## هدف

این سند پیش از پایان Run فعال ثبت شده است تا تصمیم توسعهٔ خانوادهٔ Control یا
AuxMetric به Foldهای 1 و 2 بر اساس یک قلهٔ خوش‌شانس یا تفسیر پس‌نگرانه گرفته
نشود. دو Run از split، seed، encoder، optimizer، window policy و بودجهٔ epoch
یکسان استفاده می‌کنند. تنها عامل علمی AuxMetric، انتقال ۵٪ جرم loss اصلی به
یک prototype objective روی ۱۰۰۰ هویت metric است.

## داده‌های معتبر برای gate

1. معیار اصلی، `probability-average Macro-F1` دقیق مسیر submission است.
2. Raw و EMA دو مدل مستقل‌اند؛ هرکدام فقط با checkpoint و OOF متعلق به همان
   وزن‌ها ارزیابی می‌شوند.
3. مقایسهٔ نهایی فقط پس از early stop طبیعی یا پایان epoch budget انجام می‌شود؛
   مقایسهٔ same-horizon در heartbeat صرفاً diagnostic است.
4. `oof_predictions.npz` دو Run باید دقیقاً همان ۱۶۳۲ فایل Fold 0، labelها،
   class map، split seed و عرض ۴۴۷ را پوشش دهد. اختلاف یا تکرار فایل hard-error
   است.
5. وزن fusion روی همین Fold تنظیم نمی‌شود. وزن‌های ۰٫۲۵، ۰٫۵ و ۰٫۷۵ فقط
   diagnostic هستند؛ انتخاب وزن نیازمند nested/cross-fold evidence است.
6. Loss خام بین دو recipe معیار gate نیست. مجموع وزن‌های Primary در Aux برابر
   ۰٫۹۵ و در Control برابر ۱ است و logger قدیمی Proto را در `train_loss` ثبت
   نمی‌کند. فقط Primary Loss نرمال‌شده و اجزای جداگانه برای عیب‌یابی استفاده
   می‌شوند؛ تصمیم معماری از F1 و الگوی خطا می‌آید.

## مرجع Control

- بهترین Raw probability-average: `0.9469211906` در epoch 112
- بهترین EMA probability-average: `0.9444993028` در epoch 131
- Known Accuracy متناظر بهترین Raw: `0.9573991031`
- OOD-F1 متناظر بهترین Raw: `0.9586206897`
- OOF Fold 0: ۱۶۳۲ فایل، SHA-256 برابر
  `221aa9edb00e13cbf98a629a4dddede8eb5835460fafe513fb5ffc9d1ea89394`

## شاخص‌های نهایی

برای هر خانواده این موارد استخراج می‌شود:

- بهترین Raw و بهترین EMA probability-average و epoch متناظر؛
- بهترین/آخرین logit-average به‌عنوان diagnostic؛
- میانهٔ پنج epoch برتر probability-average برای کاهش حساسیت به spike؛
- Known Accuracy و OOD-F1 همان checkpoint منتخب؛
- تعداد خطاهای مشترک، فقط-Control-صحیح و فقط-Aux-صحیح؛
- نرخ بازیابی خطاهای Control توسط Aux؛
- oracle Macro-F1 و gain آن نسبت به بهترین مدل مستقل؛
- fixed 50/50 blend صرفاً به‌عنوان شاهد توصیفی، نه وزن انتخاب‌شده؛
- سلامت artifact، hashها، MLflow metrics/params و بازتولیدپذیری Git/config.

## قانون تصمیم

### A — توسعهٔ AuxMetric به Fold 1

Aux ابتدا فقط به Fold 1 می‌رود، نه هم‌زمان Foldهای 1 و 2، اگر همهٔ شروط سلامت
artifact برقرار باشد و یکی از دو مسیر زیر محقق شود:

1. **برد مستقل:** checkpoint منتخب Aux حداقل `+0.0015` Macro-F1 بهتر از
   checkpoint منتخب Control باشد، میانهٔ پنج قلهٔ آن از Control عقب نباشد و
   افت Known Accuracy یا OOD-F1 بیش از `0.003` ایجاد نکند؛ یا
2. **مکمل واقعی:** اختلاف مستقل Aux حداکثر `-0.0010` باشد، اما oracle gain نسبت
   به بهترین standalone حداقل `+0.0020` شود، Aux حداقل ۱۰٪ خطاهای Control را
   نجات دهد و این نجات فقط از یک زیرگروه Known/OOD نیاید.

پس از Fold 1، اجرای Fold 2 فقط با تکرار جهت اثر یا مکمل‌بودن خطاها مجاز است.

### B — انتخاب Control برای Foldهای بعدی

اگر Aux حداقل `0.0015` ضعیف‌تر باشد و oracle gain کمتر از `0.0015` بماند، یا
بیش از ۹۰٪ مجموعهٔ خطای مدل بهتر را مشترک داشته باشد، Aux رد می‌شود و بودجه
صرف Foldهای Control و سپس full-data fit می‌شود.

### C — وضعیت مبهم

اگر نتیجه میان دو ناحیهٔ بالا باشد، هر دو خانواده کورکورانه گسترش نمی‌یابند.
ابتدا تحلیل per-file و per-subgroup OOF انجام می‌شود. فقط اگر complementarity
واقعی ولی نامطمئن باشد یک Fold تأییدی Aux اجرا می‌شود؛ در غیر این صورت Control
کم‌هزینه‌ترین انتخاب است.

## snapshot هنگام پیش‌ثبت

تا epoch 81:

- بهترین Raw Aux: `0.9431993659` در epoch 75؛
- بهترین Raw Control در همین افق: `0.9431409972`؛
- بهترین EMA Aux: `0.9402397735` در epoch 81؛
- بهترین EMA Control در همین افق: `0.9424803972`؛
- قلهٔ Raw تقریباً مساوی است، اما EMA Aux هنوز عقب است؛ gate باز می‌ماند.
- پس از تصحیح مقیاس، Validation Primary Loss در epoch 81 برای Aux و Control
  به‌ترتیب `1.2644906291` و `1.2652039318` است؛ تفاوت عملی ناچیز است.

## ابزار و ثبت

- `scripts/compare_training_histories.py`: مقایسهٔ same-horizon/full و تصحیح
  خودکار مقیاس Primary Loss از config checkpoint.
- `scripts/compare_oof_predictions.py`: کنترل یکپارچگی، خطاهای مکمل، oracle و
  blendهای fixed توصیفی.
- commit ابزار OOF: `311938a`
- commit اصلاح ثبت Loss: `9266115`

این سیاست فقط gate معماری Fold 0 را تعیین می‌کند. امتیاز لیدربرد برای تنظیم
وزن، threshold یا انتخاب post-hoc checkpoint استفاده نمی‌شود.
