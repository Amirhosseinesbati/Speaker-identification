# گزارش مرجع درک عمیق داده - IAAA 2026 Speaker Identification

**تاریخ ممیزی:** 2026-08-27  
**منبع:** `data/raw` + EDAهای Phase 0 تا 3 + embedding منجمد ECAPA  
**هدف:** این فایل مرجع canonical مرحله Data Understanding است؛ اعداد مهم نباید دوباره از روی حدس یا گزارش‌های قدیمی استخراج شوند.

## 1. نتیجه اجرایی

1. مسئله واقعاً یک **open-set identification با 446 هویت known و یک خروجی تجمیعی unknown** است. راهنما می‌گوید 554 گوینده unknown نیز بین train/eval تقسیم شده‌اند، پس طبق spec صدای آن جمعیت کاملاً ندیده نیست؛ اما شمارش مشاهده‌شده train با این ادعا سازگار نیست و باید از برگزارکننده تأیید شود.
2. `labels.csv` شامل 4,529 ردیف است: 2,254 known و 2,275 unknown. هر 446 گوینده known دقیقاً در یک بلوک پیوسته در ترتیب CSV قرار دارد (446/446). این ساختار، شاهد قوی حفظ ترتیب هویت‌های اصلی است.
3. هر 4,529 فایل پسوند `.mp3` دارند، ولی 4,528 فایل واقعاً `RIFF/WAVE`، 16kHz و stereo هستند؛ فقط یک فایل MP3 واقعی 48kHz/mono است. دو کانال WAV عملاً یکسان‌اند (median correlation=1.0، side/mid≈-105.0dB)، پس branch دوکاناله ارزش آزمایش GPU ندارد و mono امن است.
4. 70 فایل کوتاه‌تر از 1 ثانیه‌اند؛ فقط 46 مورد header خالی 48 بایتی‌اند و 24 فایل کوتاه ولی non-empty هستند. نامیدن همه آن‌ها به‌عنوان «corrupted» دقیق نیست.
5. بعد از فیلتر 1 ثانیه، 4,459 فایل باقی می‌ماند. 4 گوینده known کمتر از 5 فایل معتبر و 0 گوینده کمتر از 4 فایل معتبر دارند؛ validation باید این ناهمگنی را صریحاً لحاظ کند.
6. ارزیابی تصحیح‌شده frozen ECAPA: top-1 LOO=0.9525، closed-set macro-F1=0.9563، و 21 هویت F1 زیر 0.8 دارند. مسئله اصلی در یک زیرمجموعه کوچک از speakerهای سخت متمرکز است.
7. EER صحیح verification برابر 0.0621 در threshold=0.3457 است. عدد 0.346 گزارش Phase 3 قدیمی threshold بود، نه EER.
8. segmentation صرفاً با ترتیب ردیف و cosine روی knownها، با تعداد مرز صحیح، boundary precision=0.7393 و ARI=0.6747 می‌دهد. سیگنال واقعی است، اما برای اعلام ground truth کافی نیست؛ به‌ویژه چون 2275 فایل unknown دقیقاً `455×5` است و با عدد 554 راهنما ناسازگاری دارد.

## 2. صحت برچسب و ساختار فایل

| شاخص | مقدار |
|---|---:|
| ردیف برچسب | 4,529 |
| فایل صوتی روی دیسک | 4,529 |
| فایل label‌شده مفقود | 0 |
| فایل بدون label | 0 |
| نام فایل تکراری در CSV | 0 |
| گوینده known | 446 |
| runهای unknown در CSV | 204 |
| بزرگ‌ترین run پیوسته unknown | 60 |

ترتیب CSV random row order نیست. تمام نمونه‌های هر known speaker کنار هم آمده‌اند و runهای unknown زمانی طولانی می‌شوند که چند هویت unknown پشت سر هم قرار گرفته‌اند. این property باید در split، pseudo-label و تحلیل leakage حفظ و مستند شود.

## 3. کیفیت و فرمت صوت

| شاخص | مقدار |
|---|---:|
| RIFF/WAVE واقعی | 4,528 |
| خطای decode/header | 0 |
| کمتر از 1s | 70 |
| header خالی 48B | 46 |
| کوتاه ولی non-empty | 24 |
| exact duplicate group | 9 |
| duplicate group دارای فایل معتبر | 5 |
| duplicate group با label متناقض (کل) | 2 |
| duplicate conflict دارای صوت معتبر | 1 |
| median duration | 59.56s |
| p95 duration | 88.49s |
| median stereo correlation | 1.0000 |
| stereo correlation < 0.9 | 0 |
| channel RMS delta > 6dB | 0 |

بزرگ‌ترین اختلاف low-level بین known/unknown مربوط به `peak` با Cohen's d=0.109 است. این اندازه اثر برای تصمیم هویتی قوی نیست؛ featureهای کیفیت باید برای QA/robustness استفاده شوند، نه به‌عنوان OOD shortcut. فهرست کامل exact duplicateها در `deep_exact_duplicate_groups.csv` است؛ مهم‌ترین conflict یک صوت 3.669s است که دقیقاً یکسان، یک‌بار known و یک‌بار unknown برچسب خورده و باید از scoring محلی/آموزش supervised پاک یا quarantine شود.

![quality](deep_quality_distributions.png)

## 4. هندسه embedding و سختی واقعی کلاس‌ها

| شاخص unbiased/corrected | مقدار |
|---|---:|
| known LOO top-1 | 0.9525 |
| known LOO top-5 | 0.9619 |
| known closed-set macro-F1 | 0.9563 |
| speaker با F1 کامل | 363 / 446 |
| speaker با F1 < 0.8 | 21 |
| verification EER (نرخ، نه threshold) | 0.0621 |
| OOD AUC | 0.9585 |
| FPR در TPR>=0.95 | 0.0923 |
| corrected best direct Macro-F1 | 0.9198 |

فایل `deep_known_speaker_diagnostics.csv` هویت‌های سخت را بر اساس F1، margin و نزدیک‌ترین impostor رتبه‌بندی می‌کند. `deep_nearest_speaker_pairs.csv` نیز نزدیک‌ترین جفت centroidها را ثبت می‌کند. این دو فایل باید مبنای hard-negative mining و ارزیابی per-speaker باشند.

![embedding](deep_embedding_margin_ood.png)

## 5. بازسازی هویت‌های unknown از ترتیب داده

در راهنمای مسابقه صریحاً 554 گوینده unknown ذکر شده، اما train دقیقاً 2275 فایل unknown (`455×5`) دارد و 193 مورد از 204 run آن مضرب پنج‌اند. بنابراین سه hypothesis جدا نگه داشته شده‌اند: (الف) تحمیل 554 گروه مطابق spec، (ب) threshold مرز کالیبره‌شده روی known، و (ج) بلوک‌های پنج‌تایی داخل runهای unknown. قبل از اعمال threshold روی unknown، همان الگوریتم بدون label روی knownها سنجیده شد:

| اعتبارسنجی order-constrained segmentation | مقدار |
|---|---:|
| مرز واقعی known | 445 |
| boundary precision با تعداد مرز صحیح | 0.7393 |
| Adjusted Rand Index | 0.6747 |
| calibrated boundary F1 | 0.8602 |
| calibrated precision / recall | 0.7573 / 0.9955 |
| mean cosine داخل بلوک known | 0.7504 |
| mean cosine روی مرز known | 0.1638 |

خروجی سه hypothesis:

| hypothesis | گروه | median size | singleton | within cosine | boundary cosine |
|---|---:|---:|---:|---:|---:|
| forced spec=554 | 554 | 5.0 | 102 | 0.7965 | 0.1235 |
| known-calibrated threshold | 587 | 5.0 | 126 | 0.8042 | 0.1398 |
| run-local blocks of five | 460 | 5.0 | 10 | 0.7413 | 0.1981 |

هر سه ستون در `deep_unknown_pseudo_speakers.csv` ثبت شده‌اند. هیچ‌کدام فعلاً ground truth نیست. استفاده آموزشی باید hypothesis-specific ablation و confidence weighting داشته باشد؛ چیزی که EDA قطعی می‌کند فقط این است که فرض «unknown یک کلاس بدون ساختار داخلی است» نادرست است.

![order](deep_order_similarity.png)

![groups](deep_unknown_pseudo_group_sizes.png)

## 6. اصلاحات لازم نسبت به EDA قبلی

1. **Container mismatch:** گزارش قبلی raw را MP3 می‌نامید؛ 4528 فایل WAVE/PCM stereo با پسوند اشتباه و یک MP3 واقعی داریم.
2. **Short != corrupt:** threshold یک‌ثانیه یک policy است؛ 24 فایل non-empty کوتاه را نباید بدون سنجش contribution حذف قطعی نامید؛ 46 فایل 48B عملاً خالی‌اند.
3. **Duplicate wording:** Phase 3 ادعا می‌کرد duplicateها حذف شده‌اند، اما `clean_labels` فقط `corrupted` را در drop-set می‌گذارد. آمار duplicate باید جداگانه گزارش شود.
4. **EER bug:** کد قدیمی مقدار threshold را با نام EER گزارش می‌کرد.
5. **Macro-F1 leakage:** شبیه‌سازی قدیمی OOD score را LOO می‌ساخت ولی speaker prediction را با centroid کامل (شامل خود نمونه) انجام می‌داد. عدد corrected این گزارش از prediction کاملاً LOO برای known استفاده می‌کند.
6. **KMeans=8 بدون مبنا:** specification تعداد 554 هویت unknown را می‌دهد و ترتیب CSV سیگنال مرز قوی دارد؛ 8 خوشه representation مناسبی از ساختار واقعی نیست.
7. **Validation risk:** یک holdout تصادفی از هر speaker برای انتخاب threshold کافی نیست؛ file/session/group-aware folds و گزارش dispersion بین foldها لازم است.

## 7. قرارداد استفاده در مراحل بعد

- برای integrity و کیفیت، `deep_audio_inventory.csv` مرجع است.
- برای hard speakers و hard negatives، دو CSV تشخیصی embedding مرجع‌اند.
- برای unknown identity-aware sampling، فقط `deep_unknown_pseudo_speakers.csv` با confidence/ablation استفاده شود.
- هر ادعای بهبود باید **OOF per-speaker Macro-F1، known recall، unknown precision/recall، و fold variance** را هم‌زمان گزارش کند.
- threshold نهایی نباید روی یک split یا روی train in-sample انتخاب شود.

## 8. محدودیت‌های این گزارش

- embedding ECAPA یک ابزار اندازه‌گیری است و ceiling معماری‌های بهتر نیست.
- pseudo-identityهای unknown ground truth رسمی ندارند؛ اعتبار آن‌ها از کنترل known و جدایی edgeها می‌آید.
- ویژگی active/silent یک energy proxy است، نه neural VAD.
- hidden eval قابل مشاهده نیست؛ هر نتیجه محلی باید در چند split گروه‌محور و سپس روی leaderboard تأیید شود.
