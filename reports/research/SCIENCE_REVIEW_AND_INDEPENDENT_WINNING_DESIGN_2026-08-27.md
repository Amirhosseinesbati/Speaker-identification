# مرور علمی و طراحی مستقل راهکار برنده

**مسابقه:** IAAA 2026 Speaker Identification  
**تاریخ:** 2026-08-27  
**وضعیت سند:** این طراحی عمداً قبل از ممیزی کد و استراتژی فعلی پروژه نوشته شده است تا مقایسه بعدی دچار confirmation bias نشود.  
**ورودی‌ها:** راهنمای رسمی مسابقه، پکیج‌های سرور، EDA عمیق جدید، جست‌وجوی SciSpace و منابع اولیه مقالات/چالش‌ها.

## 1. صورت دقیق مسئله

- خروجی نهایی 447 کلاس دارد: 446 شناسه known و یک `unknown`.
- معیار، Macro-F1 روی 447 کلاس است؛ بنابراین کیفیت تشخیص تک‌تک 446 هویت known عامل اصلی امتیاز است.
- با این حال خطای unknown فقط «یک جمله از میان 447 جمله Macro-F1» نیست. وقتی unknown به یک known اشتباه نسبت داده شود، علاوه بر F1 unknown، precision همان known نیز خراب می‌شود. پس rejection باید دقیق باشد، اما نباید recall شناسه‌های known را قربانی کند.
- تقریباً همه knownها پنج فایل بلند دارند. واحد مستقل آماری **فایل** است، نه window؛ هزار window هم‌بسته از یک فایل نباید وزن آن فایل را در prototype یا validation بیشتر کند.
- guide تعداد 554 هویت unknown را اعلام می‌کند، ولی train دقیقاً 2275 فایل unknown (`455×5`) و order structure متفاوتی دارد. طراحی باید در برابر دو سناریو robust باشد:
  1. unknownهای eval همان background identities شنیده‌شده در train هستند؛
  2. بخشی از unknownهای eval واقعاً novel identity هستند.

## 2. جمع‌بندی شواهد علمی

### 2.1 Few-shot speaker recognition

- Angular Margin Centroid و Prototypical/centroid losses مستقیماً compactness درون-speaker و separation بین-speaker را بهینه می‌کنند و برای enrollment کم مناسب‌اند. برای این مسابقه، prototype/metric objective از 447-way cross-entropy ساده هم‌راستاتر است. [Angular Margin Centroid, Interspeech 2020](https://www.isca-archive.org/interspeech_2020/wei20b_interspeech.html)
- winning recipe در VoxSRC-20 یک مرحله **large-margin fine-tuning** با utteranceهای بلندتر و margin بزرگ‌تر، سپس quality-aware calibration بود. این دقیقاً با فایل‌های حدود 60 ثانیه‌ای این داده هم‌خوان است. [IDLAB VoxSRC-20](https://arxiv.org/abs/2010.11255)
- ECAPA-TDNN با Res2Net، SE و attentive statistics pooling baseline تاریخی قدرتمندی است؛ اما literature جدید نشان می‌دهد معماری‌های مکمل بهتر است در score-level fusion ترکیب شوند، نه اینکه تنها یک encoder را بیش‌ازحد fine-tune کنیم. [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- CAM++ با D-TDNN، context-aware masking و multi-granularity pooling، کیفیت/سرعت خوبی دارد و برای inference محدود سرور مناسب است. [CAM++](https://arxiv.org/abs/2303.00332)
- ERes2NetV2 به‌طور خاص multi-scale fusion برای short-duration را تقویت می‌کند و روی trialهای 2-3 ثانیه‌ای افت کمتری دارد؛ این مکمل خوبی برای CAM++ روی windowهای کوتاه است. [ERes2NetV2](https://www.isca-archive.org/interspeech_2024/chen24l_interspeech.html)
- سیستم Microsoft در VoxSRC-22 نشان می‌دهد fusion مدل‌های supervised مانند ECAPA/Res2Net با SSL encoderهای WavLM/wav2vec، همراه score normalization و calibration، می‌تواند از هر خانواده منفرد بهتر شود. [Microsoft VoxSRC-22](https://arxiv.org/abs/2209.11266)

### 2.2 Open-set identification و calibration

- VoxWatch نشان می‌دهد با بزرگ شدن watchlist، max impostor score بالا می‌رود و false alarm بدتر می‌شود. adaptive score normalization همیشه کمک نمی‌کند، اما calibration و fusion بهبود معنادارتری دارند. برای 446 هویت، raw max-cosine threshold کافی نیست. [VoxWatch](https://arxiv.org/abs/2307.00169)
- پژوهش‌های کلاسیک open-set speaker ID نیز cohort normalization را مفید می‌دانند، ولی نتیجه VoxWatch هشدار می‌دهد آن را باید ablate کرد، نه dogma دانست. [Verification effectiveness in OSI](https://doi.org/10.1049/IP-VIS:20050273)
- calibration روی embeddingهای عمیق در داده in-domain ساده‌تر است؛ در domain mismatch، adaptive S-norm گاهی score distribution را پایدار می‌کند اما trade-off دارد. نتیجه عملی: calibration باید OOF و quality-aware باشد و fold variance گزارش شود. [Deep speaker score calibration](https://arxiv.org/abs/2203.15106)
- ساختار «closed-set identification + outlier detector» در OSI از نظر تجربی قوی‌تر از تلاش برای یادگیری یک unknown logit ناهمگن است. [Open-set speaker identification with i-vectors](https://doi.org/10.21437/Odyssey.2020-58)

### 2.3 Long audio، VAD، domain adaptation و label noise

- soft VAD به‌جای حذف سخت frameها، pooling را با posterior گفتار وزن می‌دهد و در محیط واقعی robustتر است. در این داده، 60 ثانیه audio با حدود 20% silent-frame ratio داریم، پس aggregation speech-aware ارزشمند است. [Self-adaptive soft VAD](https://arxiv.org/abs/1909.11886)
- pseudo-label adaptation در NIST SRE و VoxSRC موفق بوده است، اما کیفیت pseudo-label تعیین‌کننده است. در این پروژه unknown ordering یک prior قوی می‌دهد، ولی ناسازگاری 455/554 باید در ablation حفظ شود. [Unlabeled-data robustness/adaptation](https://www.isca-archive.org/interspeech_2017/castan17_interspeech.html)
- در حضور pseudo-label noise، mixup، sub-centers و margin-based losses از memorization بهتر جلوگیری می‌کنند. این نتیجه استفاده کور از pseudo-identityهای unknown را رد و استفاده confidence-weighted/sub-center را حمایت می‌کند. [Label-noise robust SV, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/fathan24_interspeech.html)
- adapterهای محدود SE/BN می‌توانند با تغییر حدود 1% پارامترها domain adaptation انجام دهند و ریسک overfit داده کم را کاهش دهند. [SE/BN Adapter](https://arxiv.org/abs/2406.07832)

## 3. معماری مستقل پیشنهادی

نام پیشنهادی: **File-Balanced Multi-Encoder Open-Set Metric Ensemble**

### 3.1 Front-end و encoderها

سه خانواده را مستقل نگه می‌داریم:

1. **CAM++**: encoder اصلی با نسبت accuracy/latency مناسب.
2. **ERes2NetV2 یا ResNet speaker model**: مکمل multi-scale برای windowهای کوتاه و hard speakers.
3. **WavLM-based speaker encoder**: فقط اگر OOF نشان دهد خطاهایش با دو مدل اول decorrelated است و runtime/size سرور اجازه دهد.

هدف، داشتن دو مدل واقعاً مکمل است؛ ensemble سه مدل مشابه با error overlap بالا فقط هزینه می‌سازد.

### 3.2 Sampling و pooling

- train stage اول: cropهای 3-5 ثانیه‌ای با speed perturb، noise و RIR ملایم.
- large-margin stage دوم: cropهای 8-12 ثانیه‌ای، learning rate پایین و margin بزرگ‌تر.
- inference: windowهای چندمقیاسی 3/5/8/12 ثانیه از بخش‌های speech-rich.
- soft/energy VAD برای وزن‌دادن frame/window، نه حذف کور.
- هر فایل ابتدا یک embedding فایل تولید می‌کند؛ سپس پنج فایل speaker وزن برابر می‌گیرند. windowهای بیشتر از یک فایل نباید enrollment آن فایل را غالب کنند.
- وزن window بر اساس speech ratio، embedding consistency با median فایل، clipping/noise proxy و disagreement بین encoderها.

### 3.3 Objective

دو stage supervised روی knownها:

1. **AAM-Softmax/CosFace با class-balanced batches**؛ unknown به‌عنوان یک class عظیم وارد این head نشود.
2. **AM-Centroid یا episodic prototypical fine-tuning**؛ در هر episode support/query از فایل‌های جدا انتخاب شوند.

برای pseudo-unknown:

- سه hypothesis موجود (`forced_554`, `calibrated`, `block5`) جداگانه آزمایش شوند.
- pseudo-labelها فقط با confidence بالا وارد metric learning شوند.
- sub-center loss یا mixup برای مقاومت به مرز اشتباه.
- هیچ pseudo-unknownی مستقیماً به‌عنوان evidence برای یکی از 446 knownها استفاده نشود.

### 3.4 Enrollment و prototype

برای هر encoder و speaker:

- embedding هر فایل از robust weighted mean windowها ساخته شود.
- prototype پایه = میانگین L2-normalized پنج embedding فایل.
- outlier file با leave-one-file-out consistency وزن کمتر بگیرد.
- prototype نهایی، blend تنظیم‌شده OOF بین centroid داده و class weight شبکه AAM باشد.
- برای speakerهای چندحالته، حداکثر دو sub-centroid با shrinkage به centroid اصلی؛ شرط فعال‌شدن sub-center باید OOF باشد.

### 3.5 Known identity scoring

برای هر test file و encoder:

- score هر window به تمام 446 prototype محاسبه شود.
- aggregation فقط mean score نباشد؛ trimmed mean + top-quality mean و consistency ثبت شود.
- featureهای candidate top-1 شامل:
  - top-1 cosine؛
  - top-1 minus top-2 margin؛
  - cohort z-score یا percentile؛
  - agreement مدل‌ها؛
  - window vote share؛
  - score variance؛
  - quality/duration/speech ratio؛
  - candidate-specific OOF reliability.

score-level fusion با logistic regression منظم یا مدل بسیار کوچک monotonic/GBM روی predictionهای OOF انجام شود. embedding concatenation اولویت پایین‌تری دارد، چون calibration و missing-model handling را سخت می‌کند.

### 3.6 Unknown rejection

به‌جای unknown logit منفرد، دو evidence مستقل بسازیم:

1. **Known confidence:** آیا top known candidate واقعاً target-like است؟
2. **Background evidence:** similarity به unknown pseudo-centroids/cohort و distance از known manifold.

مدل calibration نهایی روی این featureها، احتمال known-correct را تخمین می‌زند. خروجی `unknown` وقتی انتخاب می‌شود که expected Macro-F1 risk آن کمتر باشد، نه صرفاً وقتی max cosine از threshold عمومی پایین‌تر است.

دو validation regime لازم است:

- unknown-file holdout داخل pseudo-identity برای سناریوی background identity seen؛
- whole-pseudo-identity holdout برای سناریوی truly novel unknown.

threshold/weights نهایی باید روی mixture این دو regime robust باشند تا ناسازگاری guide/data باعث leaderboard collapse نشود.

## 4. پروتکل ارزیابی که اجازه خودفریبی نمی‌دهد

### 4.1 Known OOF

- 5-fold leave-one-file-out per known speaker؛ هر fold تقریباً یک فایل از هر speaker را query می‌کند.
- exact duplicates در یک fold بمانند.
- قطعه exact-duplicate با conflict known/unknown quarantine شود.
- 46 header خالی حذف؛ 24 فایل کوتاه non-empty به‌صورت ablation `drop` در برابر `short-window include` بررسی شوند.

### 4.2 Metrics

هر run باید ثبت کند:

- 447-way Macro-F1؛
- known-only macro-F1؛
- mean و worst-decile per-speaker F1؛
- unknown precision/recall/F1؛
- known→unknown و unknown→known spillover؛
- top-1/top-5 known accuracy؛
- fold mean/std و bootstrap CI؛
- latency، peak VRAM/RAM و package size.

### 4.3 Selection discipline

- threshold tuning و fusion training فقط روی OOF.
- final full-data model بعد از freeze شدن recipe آموزش ببیند.
- leaderboard فقط برای تأیید hypothesisهای بزرگ استفاده شود، نه تنظیم ده‌ها threshold.
- هر آزمایش باید یک control و یک تغییر داشته باشد.

## 5. برنامه آزمایش کم‌اتلاف

### Tier 0 - بدون GPU یا با embedding cached

1. بازسازی دقیق OOF برای تمام encoderهای موجود.
2. file-balanced robust prototypes در برابر centroid ساده.
3. candidate-specific calibration و quality features.
4. score-level fusion دو encoder بر اساس error complementarity.
5. unknown pseudo-centroid evidence برای هر سه hypothesis.
6. quarantine duplicate conflict و ablation فایل‌های کوتاه.

این tier باید قبل از هر training جدید تمام شود؛ اگر calibration/fusion بهبود OOF ندهد، GPU مشکل را حل نخواهد کرد.

### Tier 1 - یک GPU

1. large-margin fine-tuning قوی‌ترین encoder موجود با crop بلند.
2. episodic AM-Centroid روی همان encoder.
3. frozen-backbone SE/BN adapter در برابر full fine-tune.

### Tier 2 - دو یا سه GPU فقط در صورت نیاز

1. ERes2NetV2 complementary model.
2. WavLM speaker fine-tuning اگر error correlation پایین باشد.
3. دو seed فقط برای recipe برنده، نه برای همه gridها.

### Tier 3 - بسته نهایی leaderboard

- دو مدل مکمل + calibration کوچک.
- offline weights، بدون دانلود runtime.
- smoke test دقیق روی Python 3.12، CUDA 12.8، Torch 2.10 و package versions سرور.
- fallback CPU-safe برای post-processing؛ encoder inference روی CUDA.

## 6. انتظار واقع‌بینانه سود

این اعداد guarantee نیستند و فقط برای اولویت‌بندی‌اند:

| تغییر | سود محتمل Macro-F1 | ریسک/هزینه |
|---|---:|---|
| OOF درست + file-balanced prototype + hard-speaker fixes | 0.002 تا 0.005 | کم |
| quality-aware candidate calibration | 0.001 تا 0.004 | کم |
| large-margin long-crop fine-tuning | 0.002 تا 0.006 | متوسط |
| fusion دو encoder واقعاً مکمل | 0.002 تا 0.005 | متوسط |
| pseudo-unknown competing prototypes | 0.0005 تا 0.003 | متوسط، label-risk |
| encoder سوم SSL | 0 تا 0.003 | زیاد، runtime-risk |

برای عبور از 0.972، محتمل‌ترین مسیر جمع چند بهبود مستقل کوچک است؛ نه یک architecture replacement پرریسک.

## 7. تصمیم‌های از پیش ردشده

- **stereo branch:** کانال‌ها عملاً کپی‌اند؛ ارزش compute ندارد.
- **447-way cross-entropy با unknown عظیم:** objective با Macro-F1 و ساختار heterogeneous unknown هم‌راستا نیست.
- **یک threshold روی raw max cosine:** watchlist-size و speaker-specific difficulty را نادیده می‌گیرد.
- **random window split:** leakage شدید داخل فایل و score خوش‌بینانه می‌سازد.
- **تحمیل کور 554 pseudo-cluster:** 102 singleton و ناسازگاری train count هشدار واضح است.
- **ensemble مدل‌های هم‌بسته بدون OOF fusion:** هزینه زیاد، gain کم.
- **جست‌وجوی بزرگ hyperparameter قبل از اصلاح validation:** منابع Vast.ai را هدر می‌دهد.

## 8. معیار مقایسه با پیاده‌سازی فعلی

در ممیزی کد، این سؤال‌ها پاسخ داده خواهند شد:

1. split واقعاً file-disjoint و OOF است؟
2. prototypeها file-balanced هستند یا window-balanced؟
3. known head از unknown class جداست؟
4. speaker prediction در calibration in-sample نیست؟
5. hard speakers و candidate-specific errors دیده می‌شوند؟
6. calibration روی scoreهای OOF و چند regime unknown انجام می‌شود؟
7. fusion بر اساس complementarity است یا وزن‌های دستی؟
8. runtime package دقیقاً با server lock سازگار است؟
9. submission هیچ دانلود شبکه‌ای ندارد؟
10. UI experiment manager lineage، config، seed، fold و artifact را غیرقابل‌ابهام ثبت می‌کند؟

---

### یادداشت روش تحقیق

جست‌وجوی SciSpace با پرسش‌های semantic کامل درباره few-shot speaker recognition، open-set calibration، VoxSRC system recipes و long-audio/domain adaptation انجام شد. یافته‌های SciSpace با صفحات اصلی arXiv/ISCA تطبیق داده شدند؛ توصیه‌های این سند استنتاج برای این داده خاص‌اند و ادعای نقل مستقیم هیچ مقاله نیستند.
