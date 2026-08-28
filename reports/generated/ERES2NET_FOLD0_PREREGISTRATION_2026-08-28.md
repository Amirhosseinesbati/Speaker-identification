# پیش‌ثبت ERes2NetV2 known-first ـ Fold 0

تاریخ: 2026-08-28

پروفایل اجرایی: `p1-eres2net-known446-ood-complement-oof-f0`

مرجع مقایسه: `p0-campp-known446-ood-control-oof-f0`

وضعیت مجوز: کاربر اجرای فقط Fold 0 را صریحاً تأیید کرده است. TitaNet لغو شده
و نباید train شود. عبور از این gate مجوز خودکار Fold بعدی یا submission نیست.

## فرضیه

ERes2NetV2 با معماری BDFF/BLFF می‌تواند خطاهایی متفاوت از CAM++ تولید کند و
representation مکمل بسازد. شواهد frozen benchmark برای ERes2NetV2 شامل known
LOO top-1 برابر `0.9578853`، top-5 برابر `0.9663978` و OOD AUC برابر
`0.9599373` است. هدف این Run جایگزین‌کردن policy قفل‌شده نیست؛ هدف سنجش diversity
واقعی روی همان Fold و با همان recipe کنترلی است.

## قرارداد ثابت Run

- فقط Fold 0 از split سه‌Fold، seed داده و آموزش برابر `42`.
- مسیر strict known-first: ArcFace 446کلاسه و binary OOD مستقل.
- همان duplicate cleaning، augmentation، هشت window هشت‌ثانیه‌ای، crop policy،
  loss weightها و Raw probability averaging کنترل.
- checkpoint محلی و offline با اندازهٔ `71768231` بایت و SHA256 زیر:
  `0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c`.
- encoder برای 20 ایپاک اول frozen است؛ سپس فقط آخرین block باز می‌شود و
  encoder LR برابر `5e-6` است. head LR برابر `3e-4` باقی می‌ماند.
- سقف 120 ایپاک، patience برابر 15 و deterministic algorithms فعال است.
- تصمیم مستقل فقط `Raw probability-average + argmax` است. EMA و logit-average
  صرفاً گزارش می‌شوند و حق انتخاب مدل یا submission ندارند.
- تنها ensemble تشخیصی مجاز، میانگین احتمال ثابت `50/50` با Raw CAM++ Fold 0
  است؛ هیچ weight، threshold یا epoch با OOF یا leaderboard جست‌وجو نمی‌شود.
- حداکثر زمان Run شش ساعت و هزینهٔ افزوده حداکثر `$2.50` است.

## پیش‌پرواز سخت‌افزار

preflight روی همان RTX 3090 و مسیر واقعی forward/backward با encoder در حالت
partial fine-tune انجام شد. این probe از profile محافظه‌کارانه‌تر 1000کلاسه
استفاده کرد و بنابراین head آن از head 446کلاسهٔ Run نهایی بزرگ‌تر بود.

| Batch | وضعیت | peak VRAM | files/s | windows/s |
|---:|---|---:|---:|---:|
| 24 | ok | `3.2552 GiB` | `9.6563` | `77.2501` |
| 32 | ok | `4.2922 GiB` | `13.8745` | `110.9962` |
| 48 | ok | `6.3603 GiB` | `14.0598` | `112.4782` |

طبق grid ازپیش‌ثابت `24، 32، 48`، batch 48 انتخاب شد. peak بسیار پایین‌تر از
guard `21.5 GiB` است و این انتخاب فقط عملیاتی است، نه hyperparameter tuning.

## gate علمی

همهٔ شروط زیر باید هم‌زمان برقرار باشند:

1. OOF، class map، split، config، history، log، checkpointها، manifest و receipt
   کامل، هم‌تراز و hash-verified باشند؛ MLflow نیز باید FINISHED و کامل باشد.
2. Macro-F1 مستقل Raw probability-average حداقل `0.93692119` باشد؛ یعنی بیش از
   `0.010` از CAM++ Fold 0 با امتیاز `0.94692119` عقب نماند.
3. ensemble ثابت 50/50 نسبت به CAM++ Fold 0 حداقل `+0.002` Macro-F1 gain بسازد
   و افت Known Accuracy و OOD-F1 هر کدام حداکثر `0.001` باشد.
4. ERes2NetV2 به‌تنهایی حداقل `25%` خطاهای CAM++ Fold 0 را درست کند.

شکست هر شرط، candidate را رد می‌کند و Foldهای 1 و 2 اجرا نمی‌شوند. عبور از gate
فقط مجوز تحلیل و نوشتن پیش‌ثبت جداگانهٔ Fold 1 است.

## stop rule

- NaN، mismatch split/provenance، artifact corruption یا خطای MLflow: توقف فوری.
- OOM با batch 48: فقط یک کاهش به batch 32؛ OOM دوم توقف کامل است.
- اگر تا پایان ایپاک 40 بهترین Raw Macro-F1 کمتر از `0.90` باشد، Run به‌عنوان
  futility متوقف می‌شود.
- عبور از شش ساعت یا `$2.50` هزینهٔ افزوده: توقف امن و حفظ artifactها.
- در تمام مدت، Campaign Supervisor تنها مالک Run است و اجرای تکراری ممنوع است.
