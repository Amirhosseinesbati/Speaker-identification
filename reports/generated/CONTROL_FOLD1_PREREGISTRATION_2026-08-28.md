# پیش‌ثبت Control Fold 1 و ممیزی کم‌انرژی

تاریخ پیش‌ثبت: 2026-08-28  
کمپین: `iaaa-speaker-rank1-20260827`  
پروفایل مجاز: `p0-campp-known446-ood-control-oof-f1`

## علت اجرا

Control در Fold 0 با probability-average Macro-F1 برابر `0.9469211906` از
AuxMetric با مقدار `0.9433028273` بهتر بود. Aux در Known Accuracy و OOD-F1 نیز
افت داشت و طبق سیاست gate رد شد. بنابراین فقط خانوادهٔ Control برای Fold 1
تأیید شده است؛ AuxMetric Fold 1 اجرا نمی‌شود.

## قرارداد علمی Run

- config، encoder، optimizer، window policy و معیار checkpoint همان Control
  Fold 0 می‌مانند؛ تنها split از Fold 0 به Fold 1 تغییر می‌کند.
- معیار اصلی، probability-average Macro-F1 مسیر واقعی submission است.
- Raw و EMA مستقل ارزیابی می‌شوند؛ logit-average فقط diagnostic است.
- early stopping طبیعی، seed 42 و deterministic algorithms حفظ می‌شوند.
- هیچ threshold، blend یا checkpoint با مشاهدهٔ نتیجهٔ Fold 1 پس‌نگرانه انتخاب
  نمی‌شود.

## ممیزی کم‌انرژی ازپیش‌ثابت

Fold 0 دارای ۱۰ WAV سکوت دقیق و ۲۳ WAV با `RMS < 1e-4` بود. تعریف‌ها قبل از
مشاهدهٔ Fold 1 ثابت می‌شوند:

- سکوت دقیق: `PCM peak == 0`؛
- کم‌انرژی: `PCM RMS < 1e-4` روی WAV شانزده‌کیلوهرتز پردازش‌شده؛
- prior مجاز: فقط از برچسب‌های train همان Fold محاسبه می‌شود؛ برچسب validation
  هرگز در ساخت fallback یا threshold استفاده نمی‌شود.

خروجی Control دست‌نخورده معیار اصلی می‌ماند. سه diagnostic ازپیش‌تعریف‌شده
گزارش می‌شوند: عملکرد روی سکوت دقیق، عملکرد روی کم‌انرژی و یک fallback مبتنی بر
prior صرفاً train-only. این diagnosticها به‌تنهایی مجوز تغییر submission نیستند.

## Gate تصمیم بعد از Fold 1

1. ابتدا سلامت OOF، class map، split metadata، artifact hash، MLflow history و
   مدل‌های Raw/EMA کنترل می‌شود.
2. اگر Control Fold 1 بدون خرابی سیستماتیک کامل شود، Fold 2 برای تکمیل OOF
   خانوادهٔ برنده مجاز است؛ افت بیش از `0.01` نسبت به انتظار split، trigger
   تحلیل خطا پیش از Fold 2 است.
3. fallback کم‌انرژی فقط وقتی نامزد ادامه می‌شود که روی Fold 1 جهت اثر Fold 0
   را تأیید کند و در Known Accuracy یا OOD-F1 افت بیش از `0.001` نسازد.
4. تصمیم نهایی فقط روی OOF تجمیعی و بدون تنظیم روی leaderboard گرفته می‌شود.

## الزامات عملیاتی

- اجرا فقط از `Campaign Supervisor`، با checkout تمیز و همگام آغاز می‌شود.
- سقف زمان ۱۲ ساعت و سقف کل کمپین ۲۰ دلار حفظ می‌شود.
- مدل، config، history، OOF، log، state و manifest باید در DagsHub/MLflow ثبت و
  hash-verified شوند.
- گزارش‌های تلگرام فارسی و شامل آخرین epoch، متریک‌ها، GPU و تفسیر علمی هستند.
