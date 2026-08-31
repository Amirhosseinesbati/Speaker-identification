# پیش‌ثبت اصلاحی بازتولید no-proto، Fold 0

## علت نسخهٔ اصلاحی

اجرای اولیهٔ `p0-campp-no-proto-repro-oof-f0` پس از دو ایپاک کامل عمداً
متوقف شد. بررسی کد نشان داد `early_stopping_start_epoch` در profile تعیین نشده
و مقدار پیش‌فرض 1 بود، درحالی‌که encoder تا پایان ایپاک 20 frozen می‌ماند. بنابراین
patience می‌توانست پیش از آنکه encoder فرصت adaptation داشته باشد مصرف شود. معیارهای
دو ایپاک نخست برای انتخاب هیچ پارامتر یا نتیجه‌ای استفاده نمی‌شوند و اجرای اولیه یک
engineering-invalid run است، نه شکست علمی recipe.

## تغییر یگانه و ازپیش‌قفل‌شده

- profile جدید: `p0-campp-no-proto-repro-es21-oof-f0`؛ مسیر checkpoint مستقل است.
- تنها تغییر رفتاری نسبت به پیش‌ثبت قبلی:
  `training.early_stopping_start_epoch=21`.
- `freeze_epochs=20`، patience=`20` و حداکثر `200` ایپاک ثابت‌اند. در نتیجه
  شمارش patience دقیقاً از اولین ایپاک post-unfreeze آغاز می‌شود و حداقل 20 ایپاک
  adaptation در دسترس است؛ ادامه تا 200 اجباری نیست.
- split، cluster map، seed، deterministic mode، architecture، augmentation، loss،
  LR، batch، window policy، Raw probability-average و تمام gateها بدون تغییرند.

## gate و بودجه

- safety gate: Raw probability-average Macro-F1 روی Fold 0 حداقل `0.960`.
- OOF، class-map، split، Raw/EMA checkpoints، config، history و manifest باید کامل،
  خوانا و hash-valid باشند.
- NaN، OOM دوم، split/provenance mismatch یا artifact corruption توقف فوری است.
- timeout دوازده ساعت؛ هزینهٔ افزوده حداکثر حدود `$2.06` و سقف کل کمپین `$50`.
- شکست gate خانوادهٔ no-proto و metric-only را می‌بندد. عبور فقط مجوز پیش‌ثبت جداگانهٔ
  Foldهای 1 و 2 است و به‌تنهایی مجوز submission نیست.
