# نتیجهٔ نهایی Control سه‌Fold و سیاست مرحلهٔ بعد

تاریخ تحلیل: 2026-08-28  
خانواده: `p0-campp-known446-ood-control-oof`  
Fold 2 Git: `bdaeab43b81e8c29225decdb814e0b78bb12518d`  
Fold 2 MLflow Run ID: `30905f3517874c19924549e9116ec3ad`

## حکم اجرایی

سه Fold کنترل با موفقیت کامل شدند و OOF تجمیعی سالم، بدون overlap و قابل‌استناد
است. مسیر صحیح تصمیم همچنان **Raw probability-average با argmax مستقیم** است.
logit-average در OOF تجمیعی `0.00623053` ضعیف‌تر است و EMA در بهترین نقطهٔ هر
سه Fold از Raw عقب می‌ماند. calibration تک‌بعدی leave-one-fold-out نیز در هر
سه تکرار دقیقاً آستانهٔ بومی `0.5` را انتخاب کرد و هیچ gainی نساخت؛ بنابراین
هیچ threshold، lambda-unknown یا OOD gate جدیدی مجاز نیست.

فعلاً full-data training کور با ایپاک ثابت اجرا نمی‌شود. بهترین ایپاک Raw در
Foldها `112، 60، 101` است؛ دامنهٔ `52` ایپاکی نشان می‌دهد حذف validation و
انتخاب یک ایپاک واحد ریسک قابل‌توجهی دارد. دارایی معتبر فعلی همان سه checkpoint
Raw و مسیر probability-average است. گام بعدی باید ابتدا تحلیل CPU-only خطاهای
مرزی و کیفیت صوت باشد؛ هیچ GPU Run جدیدی تا پایان آن تحلیل آغاز نمی‌شود.

## نتیجهٔ Fold 2

Fold 2 با early stopping در ایپاک `121/200` پایان یافت. checkpoint منتخب Raw
مربوط به ایپاک `101` است.

| معیار | مقدار | ایپاک |
|---|---:|---:|
| بهترین Raw probability-average Macro-F1 | `0.9231124034` | 101 |
| بهترین Raw logit-average Macro-F1 | `0.9162787727` | 114 |
| بهترین EMA probability-average Macro-F1 | `0.9201280858` | 112 |
| بهترین EMA logit-average Macro-F1 | `0.9155696123` | 110 |
| Known Accuracy در checkpoint منتخب Raw | `0.9398663697` | 101 |
| OOD-F1 در checkpoint منتخب Raw | `0.9645776567` | 101 |

در آخرین ایپاک، Raw probability-average به `0.9145258620` رسیده بود، در حالی
که EMA برابر `0.9190062431` بود. فاصلهٔ checkpoint منتخب Raw تا انتهای Run
حدود `0.00858654` است و عملکرد صحیح early stopping را تأیید می‌کند؛ checkpoint
آخر نباید جایگزین checkpoint منتخب شود.

## OOF سه‌Fold

| Fold | نمونه | Raw probability | Raw logit | Known Accuracy | OOD-F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1632 | `0.9469211906` | `0.9433423184` | `0.9573991031` | `0.9586206897` |
| 1 | 1627 | `0.9415750193` | `0.9365667119` | `0.9503945885` | `0.9411764706` |
| 2 | 1188 | `0.9231124034` | `0.9146715672` | `0.9398663697` | `0.9645776567` |

تجمیع filename-aligned سه Fold:

| معیار | probability-average | logit-average |
|---|---:|---:|
| Macro-F1 | `0.9438885981` | `0.9376580667` |
| Known Accuracy | `0.9510771993` | `0.9506283662` |
| OOD-F1 | `0.9547945205` | `0.9477800782` |
| Overall Accuracy | `0.9467056443` | `0.9395097819` |

- میانگین Macro-F1 Foldها: `0.9372028711`
- انحراف معیار جمعیتی Foldها: `0.0101997172`
- دامنهٔ Foldها: `0.0238087872`
- در `43` اختلاف تصمیم probability/logit، مسیر probability روی `33` نمونه
  تنها مسیر درست بود و logit فقط روی `1` نمونه برنده شد.

تفاوت score تجمیعی با میانگین سادهٔ Foldها طبیعی است: Macro-F1 غیرخطی است و
پس از کنار هم گذاشتن همهٔ فایل‌ها، support هر کلاس کامل‌تر می‌شود.

## توپولوژی خطا

در مسیر منتخب probability-average، از `4447` نمونه، `4210` تصمیم صحیح و `237`
خطا ثبت شد:

| نوع خطا | تعداد | نرخ شرطی |
|---|---:|---:|
| known → unknown | 70 | `3.1418%` از known |
| unknown → known | 128 | `5.7684%` از unknown |
| known → known اشتباه | 39 | `1.7504%` از known |

در نتیجه `198/237 = 83.54%` خطاها در مرز known/unknown رخ می‌دهند. مشکل غالب
false accept نمونه‌های unknown است، اما کاهش آن با threshold سفت‌تر می‌تواند
false reject کلاس‌های known را بالا ببرد و به‌دلیل Macro-F1 کلاس‌محور زیان‌بار
باشد. همین trade-off علت ممنوع‌بودن threshold tuning ساده است.

در OOF کامل، فقط کلاس‌های known شمارهٔ `360` و `414` دارای F1 صفر بودند.
چارک اول F1 کلاس‌های known `0.9091`، میانه `1.0` و صدک دهم `0.8333` است؛ بنابراین
بخش عمدهٔ افت در یک tail کوچک از کلاس‌ها و مرز open-set متمرکز است، نه در شکست
عمومی شناسهٔ گوینده.

## آزمون cross-fit آستانه

score آستانه‌گذاری به‌صورت
`p_unknown / (p_unknown + max_known)` تعریف شد؛ argmax فعلی دقیقاً معادل آستانهٔ
`0.5` است. برای هر Fold، آستانه فقط روی دو Fold دیگر از grid ثابت
`0.35..0.65` با گام `0.0025` انتخاب و سپس روی Fold نگه‌داشته‌شده ارزیابی شد.

| Fold نگه‌داشته‌شده | آستانهٔ منتخب از دو Fold دیگر | Δ Macro-F1 |
|---:|---:|---:|
| 0 | `0.5` | `0.000000` |
| 1 | `0.5` | `0.000000` |
| 2 | `0.5` | `0.000000` |

نتیجه روشن است: decision rule فعلی یک baseline تصادفی یا موقت نیست؛ همان نقطه‌ای
است که هر سه cross-fit مستقل انتخاب می‌کنند. آستانه‌های OOD داخلی گزارش‌شده در
evaluation (`0.40، 0.35، 0.20`) برای binary OOD-F1 تیون شده‌اند و نباید وارد
مسیر submission شوند؛ برای نمونه در Fold 2، threshold `0.20` Macro-F1 را تا
`0.887447` پایین آورد.

## ممیزی provenance و artifactها

- union سه OOF شامل `4447` filename یکتا و overlap آن دقیقاً صفر است.
- هر Fold split metadata صحیح `kfold / 3 / seed=42` و fold متناظر `0،1،2` دارد.
- خروجی رقابت عرض `447`، speaker logits عرض `446` و embedding بعد `192` دارد؛
  تمام آرایه‌های عددی finite هستند.
- هر class map دارای `1001` ورودی با دامنهٔ metric label از `0` تا `1000` است،
  در حالی که `num_unknown_clusters=0` در OOF قرارداد known-first را درست منعکس
  می‌کند.
- `cleaned_labels.csv` دارای `4467` ردیف است. `20` ردیف وارد OOF نشده‌اند:
  `16` فایل corrupted/short و `4` فایل non-corrupt از گروه‌های MD5 duplicate که
  طبق leak guard عمداً همیشه train-only می‌مانند. هیچ فایل OOF بیرون از labels
  وجود ندارد.
- هر `26/26` artifact ثبت‌شده در receipt Fold 2 موجود بود و size و SHA-256 آن
  دوباره تطبیق داده شد.
- SHA-256 OOF Fold 2:
  `69aeb460046b77402731baaf827fe81f4b5dcd79da56e6bf36df48e0e3f4166d`
- SHA-256 latest checkpoint Fold 2:
  `38aca4051e8c006bbe2e7f5c52800555e7455fc00cf4bf7d6898fdae4a10977d`

## ممیزی DagsHub/MLflow Fold 2

- Run وضعیت `FINISHED` دارد.
- هر `121/121` نقطهٔ تمام سری‌های زنده finite، پیوسته و همگام است.
- `47` کلید متریک، `41` پارامتر و `31` artifact با مجموع حجم شناخته‌شدهٔ
  `212,739,714` بایت ثبت شده است.
- مدل منتخب، دو variant Raw/EMA، سه bundle متناظر، configهای base/profile/run،
  training history، class map، metadata، manifest، unknown-cluster metadata،
  OOF و training summary حاضرند. backfill لازم نیست.

## سیاست قفل‌شدهٔ آینده

1. **Decision layer:** Raw probability-average + argmax مستقیم؛ بدون threshold،
   logit replacement، fallback کم‌انرژی یا AuxMetric.
2. **Full-data:** فعلاً اجرا نشود. median ایپاک Raw برابر `101` فقط یک prior برای
   طراحی است، نه مجوز آموزش کور. اگر full-data بعداً لازم شد، policy باید پیش از
   اجرا ثبت و checkpoint selection بدون استفاده از leaderboard تعریف شود.
3. **Ensemble Foldها:** میانگین سه مدل روی test یک candidate طبیعی است، اما OOF
   فعلی gain ensemble را بی‌طرفانه اندازه نمی‌گیرد، چون برای هر فایل فقط یک مدل
   واقعاً out-of-fold است. وزن‌های دلخواه یا ادعای gain پیش از شواهد مستقل ممنوع.
4. **کم‌هزینه‌ترین گام بعدی:** خطاهای `198` مورد boundary و tail کلاس‌های hard با
   duration، RMS/peak، confidence، margin، entropy، تعداد window و session/channel
   بررسی شوند. فقط اگر یک signal ازپیش‌تعریف‌شده در leave-one-fold-out جهت یکسان
   و guardrail known/unknown سالم داشت، یک decision candidate جدید مجاز است.
5. اگر quality-aware decision نیز رد شد، آزمایش GPU بعدی باید به‌جای threshold
   بیشتر، یک منبع خطای واقعاً مکمل یا recipe مقاوم به channel/session shift را
   هدف بگیرد و ابتدا روی یک Fold پیش‌ثبت شود.
