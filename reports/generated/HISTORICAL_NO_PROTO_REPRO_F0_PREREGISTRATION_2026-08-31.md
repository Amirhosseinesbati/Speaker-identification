# پیش‌ثبت بازتولید تمیز no-proto، Fold 0

## پرسش و شواهد مبنا

بستهٔ تاریخی no-proto/metric-only با fusion ثابت 60/40 روی holdout قدیمی 891فایلی
Macro-F1 برابر `0.9723763719` داشت، اما روی لیدربرد به `0.9605963611`
رسید. ممیزی provenance نشان داد تمام 891 فایل holdout واقعاً خارج از train تاریخی
بودند، ولی 3556 فایل از OOF جدید 4447فایلی در train همان دو checkpoint قرار
داشتند؛ بنابراین ارزیابی مستقیم checkpointهای قدیمی روی OOF جدید ممنوع است.

در stress set تاریخیِ leave-one-out، همان package به `0.9231068162` افت کرد.
فرضیهٔ واحد این است که recipe بدون prototype، با آموزش deterministic روی split سه‌Fold
جدید، ممکن است مزیت واقعی خود را حفظ کند؛ اگر حفظ نشود، عدد محلی 0.97 را یک اثر
split/channel-specific می‌دانیم و خانواده را می‌بندیم.

## قرارداد اجرا

- تنها Run مجاز در این مرحله: `p0-campp-no-proto-repro-oof-f0`.
- split: `kfold/folds3/fold0/seed42`؛ seed آموزش 42 و deterministic algorithms فعال.
- CAM++، 554 pseudo-unknown، OOD head و ArcFace؛ prototype خاموش.
- encoder دو block آخر trainable، freeze اولیه 20 epoch، encoder LR=`1e-5`.
- حداکثر 200 epoch با early stopping patience=`20`؛ توقف metric زودهنگام مجاز و
  اجرای اجباری تا horizon ممنوع است.
- batch size=`16` و هشت window، مطابق recipe بازتولید و بدون تغییر برای RTX 3090.
- Raw probability-average معیار اصلی است؛ EMA و logit فقط diagnostic هستند.
- هیچ threshold، blend، epoch یا submission با leaderboard انتخاب نمی‌شود.

## gate و stop rule

- safety gate: `Raw probability-average Macro-F1 >= 0.960` روی Fold 0.
- artifactهای Raw/EMA/canonical، OOF با 447 probability، embedding، class-map، split،
  history، config و manifest باید کامل و hash-valid باشند.
- NaN، OOM دوم، split/provenance mismatch یا artifact corruption توقف فوری است.
- timeout حداکثر 12 ساعت؛ هزینهٔ افزوده حداکثر حدود `$2.06` با نرخ
  `$0.1711111111/h`؛ سقف کل کمپین `$50`.
- شکست gate یعنی no-proto Foldهای 1/2 و metric-only خودکار شروع نمی‌شوند و مسیر به
  encoder مکمل frozen بعدی می‌رود.
- عبور gate فقط مجوز پیش‌ثبت جداگانهٔ Foldهای 1/2 است، نه اجرای خودکار آن‌ها.

## رفع drift پیش از اجرا

`default_config.yaml` بعداً capability غیرفعال `inter_class` را اضافه کرده بود و
hash invariant را تغییر می‌داد، هرچند رفتار علمی آن خاموش بود. preflight اکنون hash
جدید را قفل می‌کند و صریحاً `inter_class.enabled == false` را کنترل می‌کند؛ هر تغییر
رفتاری همچنان fail-closed است.
