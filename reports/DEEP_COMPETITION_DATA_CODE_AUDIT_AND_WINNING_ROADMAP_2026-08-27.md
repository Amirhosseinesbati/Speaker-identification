# گزارش مرجع مسابقه، داده، کد و نقشه‌ی عبور از 0.972

تاریخ ممیزی: 2026-08-27  
وضعیت: ممیزی تشخیصی کامل + EDA تکمیل‌شده + طراحی علمی مستقل؛ کد production در این مرحله عمداً تغییر نکرده است.

## 1. پاسخ کوتاه مدیریتی

هدف 0.972 دست‌یافتنی است، اما شواهد فعلی اجازه نمی‌دهد بگوییم candidate محلی 0.975 واقعاً به همان امتیاز روی leaderboard منتقل خواهد شد. بهترین نتیجه‌ی رسمی ثبت‌شده در خود پروژه 0.9625 است؛ گزارش بعدی پروژه یک package جدید را با 0.96060 روی leaderboard ثبت کرده و candidate جدید Top-5 هنوز در گزارش به‌عنوان «ارسال‌نشده» معرفی شده است. بنابراین امتیازهای محلی 0.972 تا 0.975 باید evidence آزمایشگاهی تلقی شوند، نه رکورد رسمی.

مهم‌ترین نتیجه‌ی این ممیزی این است:

> گلوگاه فعلی عمدتاً تشخیص شناسه‌ی known نیست؛ گلوگاه، رد شدن فایل‌های known سخت به‌عنوان unknown در اثر تغییر session/channel/quality و calibration نامطمئن است.

با این حال، قبل از خرج GPU باید چند اشکال measurement اصلاح شود. sampler موسوم به balanced عملاً batch-balanced نیست؛ checkpoint با logit-average انتخاب می‌شود ولی submission از probability-average استفاده می‌کند؛ artifactهای validation چند مدل با filename هم‌تراز نشده‌اند؛ HPO فعلی همه‌ی trialها را صفر خوانده؛ centroid builder دقیقاً همان cleaning/audio policy زمان آموزش را بازسازی نمی‌کند؛ و بعضی نتایج post-hoc روی همان validation تنظیم و گزارش شده‌اند. تا رفع این موارد، HPO گسترده یا ensemble بزرگ می‌تواند فقط خطای measurement را بهینه کند.

استراتژی پیشنهادی نهایی سه لایه دارد:

1. ابتدا measurement و provenance را قابل اعتماد کنیم.
2. سپس بدون training گران، calibration را با OOF/cross-fit و evidenceهای quality-aware بازسازی کنیم.
3. فقط اگر نتیجه‌ی cross-fit نشان داد representation محدودکننده است، دو encoder مکمل و یک معماری known-first را با gateهای Go/No-Go اجرا کنیم.

## 2. سلسله‌مراتب شواهد و جلوگیری از قاطی‌شدن واقعیت با فرضیه

در این گزارش شواهد در چهار سطح تفکیک شده‌اند:

| سطح | معنی | نمونه |
|---|---|---|
| رسمی | مستند مسابقه یا نتیجه‌ی leaderboard ثبت‌شده | 446 known، unknown تجمیع‌شده، Macro-F1، رکورد 0.9625 |
| مشاهده‌شده | مستقیماً از فایل صوتی/کد/artifact استخراج شده | 4529 فایل، 46 header خالی، sampler تخت، HPO=0 |
| بازتولید محلی | نتیجه‌ی آزمایش پروژه با protocol مشخص | CAM++ frozen LOO، Random/Hard، Top-5 candidate |
| فرضیه | نیازمند آزمایش مستقل | تعداد واقعی identityهای unknown، انتقال Top-5 به leaderboard، برتری WavLM |

این تفکیک حیاتی است. برای مثال «554 هویت unknown» در guide آمده، اما خود داده‌ی train شامل 2275 فایل unknown است که دقیقاً `455×5` می‌شود و ترتیب فایل‌ها نیز ساختار block-like نشان می‌دهد. پس k=554 یک prior رسمی ارزشمند است، ولی ground truth اثبات‌شده‌ی train نیست.

## 3. قرارداد واقعی مسابقه و سرور leaderboard

### 3.1 مسئله

- 446 شناسه‌ی known و یک کلاس تجمیع‌شده‌ی unknown؛ فضای خروجی مؤثر 447 کلاس است.
- معیار تصمیم نهایی argmax و معیار رتبه‌بندی Macro-F1 روی تمام 447 کلاس است.
- داده‌ی بیرونی عمومی، مدل pretrained عمومی، self-supervised learning و ensemble مجازند.
- داده‌ی labelدار مربوط به speakerهای evaluation مجاز نیست.
- package باید offline و self-contained باشد و entrypoint/CSV مورد انتظار سرور را رعایت کند.
- فایل راهنما تقریباً 50/50 train/evaluation برای هر هویت را ذکر می‌کند؛ فایل package server تعداد 3604 فایل evaluation را ثبت کرده است.

### 3.2 محیط سرور

`leaderbordpakage.txt` محیط Python 3.12 و CUDA 12.8 را همراه با torch/torchaudio 2.10، SpeechBrain 1.0.3، Transformers 4.57، ModelScope 1.38+، NeMo 2.7+، FAISS CPU، ONNX Runtime و TensorRT مشخص می‌کند. نتیجه‌ی عملی:

- CAM++، ECAPA، ERes2Net و TitaNet از نظر dependency قابل بسته‌بندی‌اند.
- مدل باید بدون network و بدون دانلود hub اجرا شود.
- checkpoint load و importها باید در محیط leaderboard واقعی تست شوند، نه فقط venv توسعه.
- استفاده از FAISS/TensorRT ممکن است، ولی تا وقتی bottleneck runtime اندازه‌گیری نشده، پیچیدگی آن توجیه ندارد.

راهنمای PDF در بخش packageها placeholder سازمان‌دهنده دارد؛ بنابراین `leaderbordpakage.txt` منبع اجرایی دقیق‌تر این repository است.

## 4. EDA جدید و درک عمیق داده

گزارش canonical داده در `eda/DEEP_DATA_UNDERSTANDING_REPORT.md` و خروجی ماشین‌خوان آن در `eda/deep_data_summary.json` است. این EDA همه‌ی فایل‌ها را inventory کرده، format واقعی، quality، duplicate، ترتیب داده، pseudo-speaker hypothesis و baseline embedding صحیح را بررسی می‌کند.

### 4.1 موجودی و قالب صوتی

| مورد | مقدار |
|---|---:|
| کل فایل‌ها | 4529 |
| known | 2254 |
| unknown | 2275 |
| speaker known یکتا | 446 |
| توزیع فایل known | 439 نفر×5، 5 نفر×6، 1 نفر×9، 1 نفر×20 |
| فایل معتبر پس از حذف `<1s` | 4459 |
| known معتبر | 2232 |
| unknown معتبر | 2227 |

نام همه‌ی فایل‌ها `.mp3` است، اما 4528 فایل عملاً RIFF/WAVE PCM16، 16kHz و stereo هستند. فقط یک فایل MP3 واقعی، 48kHz mono و 43.89 ثانیه است. بنابراین decoder باید بر اساس header/content مقاوم باشد و suffix را حقیقت codec فرض نکند.

دو کانال تقریباً کاملاً یکسان‌اند: median correlation برابر 1 و median side/mid حدود -105 dB است. mono conversion امن است؛ stereo branch یا spatial feature هیچ ارزش معناداری ندارد.

### 4.2 فایل‌های خراب و کوتاه

- 70 فایل کوتاه‌تر از یک ثانیه وجود دارد.
- 46 مورد header خالی 48-byte هستند.
- 24 فایل کوتاه اما non-empty هستند.
- ترکیب: 22 known و 48 unknown.
- پس از filtering هنوز هر 446 speaker known نماینده دارد؛ حداقل فایل معتبر هر speaker برابر 4 و median برابر 5 است.

این 70 فایل نباید به encoder خورانده شوند. در evaluation اگر decode شکست بخورد، fallback به unknown منطقی است؛ در train باید از split و prototype حذف شوند، ولی گزارش provenance آن‌ها حفظ شود.

### 4.3 duplicate و conflict

9 گروه byte-identical شامل 69 فایل یافت شد. یک conflict معتبر و جدی وجود دارد: همان صوت 3.669 ثانیه‌ای یک‌بار با speaker known `148618d7...` و یک‌بار با unknown برچسب خورده است:

- `7ce572ed...mp3` — known
- `ccecf1d3...mp3` — unknown

هر دو باید quarantine شوند؛ نگه‌داشتن یکی از دو label ریسک تزریق خطای قطعی به boundary را دارد. duplicateهای non-conflicting باید به یک نماینده کاهش یابند و هیچ duplicate family نباید بین train/validation پخش شود.

### 4.4 نبود shortcut ساده

اثر ویژگی‌های low-level بین known و unknown بسیار کوچک است؛ بیشترین Cohen's d حدود 0.109 برای peak amplitude است. بنابراین duration/RMS/codec classifier به‌تنهایی راه‌حل نیست. quality featureها باید برای reliability/gating استفاده شوند، نه به‌عنوان هویت یا OOD shortcut.

### 4.5 ترتیب ردیف‌ها و هویت‌های پنهان unknown

هر speaker known دقیقاً در یک block پیوسته آمده است. unknownها 204 run پیوسته دارند؛ طول runها از 5 تا 60، median برابر 10 و 193 run از 204 run مضرب 5 هستند. این ساختار تصادفی نیست و احتمالاً اطلاعات collection/session دارد.

سه partition فرضی بررسی شد:

| فرضیه | تعداد گروه | singleton | within adjacent cosine | boundary cosine |
|---|---:|---:|---:|---:|
| forced k=554 | 554 | 102 | 0.7965 | 0.1235 |
| threshold کالیبره‌شده با known | 587 | 126 | 0.8042 | 0.1398 |
| blockهای run-local پنج‌تایی | 460 | 10 | 0.7413 | 0.1981 |

کنترل روی speakerهای known برای threshold کالیبره‌شده precision مرزی 0.7393، recall 0.9955 و ARI 0.6747 داد. این برای استفاده‌ی weak/self-supervised امیدوارکننده است، اما برای تبدیل همه‌ی unknownها به 554 label سخت کافی نیست.

### 4.6 baseline embedding صحیح

با ECAPA frozen و leave-one-file-out واقعی:

| معیار | مقدار |
|---|---:|
| known top-1 | 0.952509 |
| known top-5 | 0.961918 |
| closed-set known Macro-F1 | 0.956278 |
| speakerهای کاملاً صحیح | 363/446 |
| speakerهای زیر 0.8 | 21 |
| known با margin منفی | 106 فایل |
| EER | 0.062107 |
| threshold EER | 0.345732 |
| OOD AUC | 0.958488 |
| FPR@TPR95 | 0.092294 |
| direct 447-way Macro-F1 | 0.919801 |

نتیجه: embedding pretrained از ابتدا قوی است، اما فاصله‌ی لازم برای رتبه‌ی اول عمدتاً در tail فایل‌های سخت و open-set calibration قرار دارد.

### 4.7 ایرادهای EDA قدیمی

- duplicateها گزارش می‌شدند ولی واقعاً حذف نمی‌شدند.
- EER و threshold آن در متن با هم اشتباه گرفته شده بودند.
- Macro-F1 از ترکیب LOO برای known و in-sample برای unknown ساخته شده بود.
- KMeans=8 مبنای مسئله نداشت.
- format صوتی بر اساس suffix اشتباه گزارش شده بود.

از این پس گزارش deep جدید مرجع است و Phase0 تا Phase3 قدیمی فقط سابقه‌ی تاریخی‌اند.

## 5. طراحی علمی مستقل قبل از مشاهده‌ی معماری فعلی

طراحی مستقل در `reports/research/SCIENCE_REVIEW_AND_INDEPENDENT_WINNING_DESIGN_2026-08-27.md` ثبت شد و قبل از ممیزی implementation فعلی freeze شد. مرور SciSpace و منابع اولیه شامل ECAPA-TDNN، CAM++، ERes2NetV2، recipes مسابقات VoxSRC، WavLM، angular-margin/centroid objectives، soft-VAD، robust training در حضور label noise و open-set speaker identification بود.

### 5.1 معماری مستقل پیشنهادی

```text
audio
  ├─ content-aware windows / soft VAD / quality features
  ├─ CAM++ encoder ─────────────┐
  ├─ ERes2NetV2 یا ResNet ──────┼─ calibrated score fusion
  └─ WavLM فقط در صورت diversity ┘

هر encoder:
  embedding unit-norm
  ├─ 446-way AAM/CosFace head برای identity known
  ├─ episodic / angular-margin centroid objective
  └─ OOD evidence مستقل و quality-aware

decision:
  known score + top1/top2 margin + cohort z-score
  + window agreement + quality + prototype reliability
  + pseudo-background evidence
  → regularised cross-fitted calibrator
  → 447-way decision
```

اصل راهبرد: identity head را با صدها pseudo-class نویزی مجبور به رقابت مستقیم نکنیم. unknownها برای background/metric/contrastive evidence مفیدند، اما hard 554-way CE فقط در صورت confidence بسیار بالا مجاز است.

### 5.2 training schedule مستقل

1. stage کوتاه 3 تا 5 ثانیه با augmentation ملایم و encoder freeze/partial-unfreeze.
2. stage 8 تا 12 ثانیه با large-margin fine-tuning.
3. file-disjoint support/query episode؛ هیچ window از یک فایل نباید هم support و هم query باشد.
4. prototypeهای robust و quality-weighted، نه mean ساده‌ی همه‌ی فایل‌ها.
5. calibration فقط روی OOF و به‌صورت cross-fit.

### 5.3 چیزهایی که طراحی مستقل رد کرد

- 447-way CE بزرگ به‌عنوان تنها objective.
- threshold خام max-cosine.
- split تصادفی window-level.
- پذیرفتن کورکورانه‌ی k=554.
- ensemble چند مدل بسیار هم‌بسته.
- HPO قبل از تثبیت protocol.

## 6. کالبدشکافی ساختار و flow فعلی پروژه

سطح قابل مشاهده‌ی repository شامل 186 فایل non-ignored در `configs/`، `src/`، `scripts/`، `reports/`، `submission/` و `tests/` است؛ raw data و artifactهای ignored جداگانه بررسی شدند. مسیرهای اجرایی بحرانی و dependency آن‌ها به‌صورت زیرند.

```text
configs/default_config.yaml + configs/experiments/*.yaml
             │
             ├─ src/experiment_config.py → resolved profile
             │
             ├─ src/pipelines/run_pipeline.py
             │      ├─ data: prepare/split/dataloader
             │      ├─ train: model/loss/EMA/checkpoint/bundle
             │      ├─ eval: Macro-F1 + OOD threshold diagnostics
             │      └─ decision/ensemble stages
             │
             ├─ src/deploy/deploy_app.py
             │      ├─ config editor
             │      ├─ local runner / matrix / queue
             │      ├─ Vast launcher / HPO
             │      └─ analysis / package promotion
             │
checkpoint + class_map + config
             ├─ OOF/val artifacts
             ├─ centroids / decision tuning
             └─ scripts/build_submission.py
                    └─ submission/submission.py
                           └─ submission/inference.py
                                  └─ CSV leaderboard
```

### 6.1 نقاط قوی implementation

- profileهای نام‌دار با deep-merge و resolved config وجود دارند.
- checkpointهای جدید config، class map، history، versions، git revision و fingerprintهای داده را همراه خود نگه می‌دارند.
- raw و EMA جداگانه امتیازدهی و ذخیره می‌شوند.
- OOF bundle می‌تواند filenames، labels، logits، competition probabilities و embeddings را نگه دارد.
- submission offline است، مدل‌ها را از config خود checkpoint می‌سازد و class-map known را بین ensembleها کنترل می‌کند.
- package builder مدل‌های zero-weight را حذف و ZIP root/size را کنترل می‌کند.
- verifier اجرای package از cwd متفاوت، silence policy و schema CSV را تست می‌کند.
- UI برای config، profile، queue، local/Vast، MLflow analysis و package promotion یک flow عملی فراهم کرده است.
- 257 تست پاس و 41 تست skip شدند؛ دو warning scheduler وجود داشت و failure ثبت نشد.

این زیرساخت برای experiment management خوب است، اما تست‌ها چند invariant علمی بحرانی را پوشش نمی‌دهند.

## 7. یافته‌های کد با اولویت و اثر علمی

### P0 — قبل از هر training یا ادعای بهبود باید اصلاح شود

#### P0.1 sampler متوازن، batch متوازن تولید نمی‌کند

`src/data_pipeline.py:1248-1307` یک لیست تخت از indexها می‌سازد. سپس `src/data_pipeline.py:1448-1467` و `src/pipelines/steps.py:464-477` آن را به `SubsetRandomSampler` می‌دهند. `SubsetRandomSampler` کل لیست را دوباره shuffle می‌کند؛ مرز batchهای اولیه از بین می‌رود. در نتیجه:

- نسبت OOD/known در هر batch تضمین نمی‌شود.
- همان multiset از indexها در همه‌ی epochها تکرار می‌شود.
- sampling با replacement می‌تواند بعضی فایل‌های 5-shot را در تمام training هرگز نبیند و بعضی را بارها تکرار کند.

راه اصلاح: یک `BatchSampler` epoch-aware با seed=`base_seed+epoch`، نسبت دقیق هر batch، گردش بدون replacement تا exhaustion و test برای coverage/ratio.

#### P0.2 mismatch انتخاب checkpoint و مسیر واقعی submission

`src/train.py:507-548` و `src/train.py:698-755` logits پنجره‌ها را average می‌کنند. `src/train.py:931-954` همین logit-average را برای Macro-F1 و انتخاب best checkpoint استفاده می‌کند. اما `src/model.py:269-318` و submission، احتمال هر window را جدا محاسبه و سپس probability-average می‌کنند.

به‌علت nonlinearity sigmoid/softmax این دو یکسان نیستند. بنابراین ممکن است epoch/EMA انتخاب‌شده بهترین مدل برای forward نهایی نباشد.

راه اصلاح: validation در همان encoder pass هم logit-average diagnostic و هم exact competition probability-average را برگرداند؛ early-stop/checkpoint selection فقط با مسیر دوم انجام شود.

#### P0.3 artifactهای validation چند checkpoint هم‌تراز نیستند

`src/decision_engine.py:41-145` برای هر encoder `val_probs_<enc>.npy` و `val_emb_<enc>.npy` می‌نویسد، اما یک فایل مشترک `val_labels.npy` دارد و filename manifest ذخیره نمی‌کند. `src/decision_engine.py:148-211` آرایه‌ها را صرفاً stack می‌کند.

اگر checkpointها fold/config متفاوت داشته باشند، row i یک فایل واحد نیست. fusion/tuning حاصل در این حالت نامعتبر است، حتی اگر shapeها برابر باشند.

راه اصلاح: هر artifact باید `files`, `labels`, `split`, checkpoint SHA و config hash داشته باشد؛ loader باید بر اساس filename join کند و mismatch را hard error کند. برای ensemble OOF باید predictionهای هر مدل روی همان فایل‌ها تولید شوند، نه validation disjoint هر fold.

#### P0.4 ensemble calibrator نیز fold و config را مخلوط می‌کند

`src/ensemble_calibrate.py:223-267` config checkpoint اول را برای validation همه‌ی مدل‌ها استفاده می‌کند و labels هر iteration را overwrite می‌کند. هیچ equality check برای filenames/labels نیست. همچنین اگر OOD head غیرفعال باشد، `None` وارد مسیر temperature calibration می‌شود.

علاوه بر آن `src/ensemble_calibrate.py:460-480` ممکن است `best_method` را geometric/max/MLP ثبت کند، ولی `submission/submission.py:231-243` همیشه `weighted_average` را hard-code می‌کند. پس «best fusion» گزارش‌شده الزاماً چیزی نیست که leaderboard اجرا می‌کند.

#### P0.5 HPO فعلی نتیجه‌ی معتبر ندارد

`src/hpo.py:47` فقط عبارت `Best val Macro-F1:` را parse می‌کند، در حالی‌که pipeline در `src/pipelines/steps.py:903-904` عبارت `Training complete! Selected ... Macro-F1:` چاپ می‌کند. فایل `checkpoints/hpo/best_params.json` برای 30 trial مقدار `best_value: 0.0` دارد؛ آن پارامترها قابل استفاده نیستند.

همچنین HPO فقط `training.label_smoothing` را تغییر می‌دهد، ولی loss builder مقدار nested `training.loss.speaker.label_smoothing` را ترجیح می‌دهد؛ بنابراین این dimension عملاً dead است. تغییر وزن OOD بدون حفظ مجموع/نسبت speaker نیز objective را ناخواسته تغییر می‌دهد.

نتیجه: profile `speaker-hpo-best` و هر استنتاج مبتنی بر آن باید invalid علامت بخورد.

#### P0.6 centroidهای checkpoint همان data/audio policy آموزش را بازسازی نمی‌کنند

`src/centroid_baseline.py:232-252` split checkpoint را می‌گیرد، اما `split_args_from_config` فقط scheme/fold/folds/seed را منتقل می‌کند و `data.clean_duplicates` را منتقل نمی‌کند. در نتیجه checkpointی که با duplicate cleaning آموزش دیده می‌تواند centroidهایی از داده‌ی تمیزنشده بگیرد.

همچنین `SpeakerDataset` در centroid builder گزینه‌های `eval_speech_aware`, `speech_relative_db` و `short_audio_mode` را از config منتقل نمی‌کند. پس embedding centroid و embedding submission می‌توانند policy متفاوت داشته باشند.

#### P0.7 گزارش split پس از cleaning اطلاعات conflict را خراب می‌کند

duplicate groups روی dataframe خام پیدا می‌شوند، سپس dataframe پاک می‌شود، اما `src/data_pipeline.py:773-775` همان dataframe پاک‌شده را به `_write_split_report` می‌دهد. در `src/data_pipeline.py:536-546` labels گروه حذف‌شده دوباره از dataframe پاک‌شده lookup می‌شوند؛ بنابراین conflict واقعی می‌تواند با labels خالی و `conflicting=false` ثبت شود. شمار known/unknown فایل‌های corrupted نیز به همین علت با total سازگار نیست.

راه اصلاح: یک `audit_df_raw` immutable برای report و یک `model_df_clean` جدا نگه داشته شود.

### P1 — ریسک زیاد برای تعمیم و calibration

#### P1.1 tuning و گزارش روی همان validation

`src/decision_engine.py:214+` پارامترهای alpha/kappa/tau/lambda را روی همان validation بهینه و امتیاز نهایی را همان‌جا گزارش می‌کند. grid search وزن ensemble در `src/ensemble.py` نیز همین مشکل را دارد. این اعداد selection-biased هستند.

راه اصلاح: nested cross-fit روی OOF؛ هر row فقط prediction مدلی را بگیرد که آن فایل را در training ندیده و threshold آن row در fold دیگری fit شده باشد.

#### P1.2 منطق centroid و open-set با هم ناسازگار است

در `submission/inference.py:513-559` open-set rule labels را تعیین می‌کند. سپس اگر centroid decision نیز فعال باشد، `submission/inference.py:561-600` labels را کامل overwrite می‌کند. بنابراین فعال‌بودن هم‌زمان دو rule به معنی composition نیست؛ rule اول عملاً اثر نهایی ندارد.

همچنین max-cosine برای hard gate روی centroidهای known و pseudo-unknown ادغام‌شده محاسبه می‌شود. نزدیک بودن به یک pseudo-unknown centroid می‌تواند `max_cosine` را بالا ببرد و gate «دور بودن از known» را غیرفعال کند. اگر هدف gate فاصله از known است، max-known-cosine باید جدا محاسبه شود.

#### P1.3 mean centroid ساده و sample-kNN bias

prototype known میانگین مساوی فایل‌هاست؛ فایل کم‌گفتار/خراب وزن برابر دارد. `src/ood_detector.py` نیز هر sample train را index می‌کند، نه speaker prototype را؛ speakerهای پرتعداد یا duplicate وزن بیشتری در kNN می‌گیرند.

راه اصلاح: robust/quality-weighted centroid، leave-one-file reliability، cohort normalization و speaker-balanced background index.

#### P1.4 cache provenance ناکافی

چند cache embedding/cluster فقط با نام encoder یا نام checkpoint کلید می‌خورند. overwrite کردن checkpoint با همان filename می‌تواند cache stale را بی‌صدا reuse کند. cache frozen centroid نیز split/config/weight SHA را در key ندارد.

راه اصلاح: content-addressed key شامل checkpoint SHA256، config hash، file-manifest hash، audio policy و code revision؛ metadata mismatch باید hard error باشد.

#### P1.5 فرض k=554 بیش از حد سخت استفاده شده است

artifactهای فعلی cluster countهای 554 و 1000 و نقشه‌های fold-specific دارند. در نقشه‌های موجود singleton زیاد است؛ برای مثال mapهای fold حدود 112 تا 133 singleton دارند و k=1000 حدود 534 singleton. این نشان می‌دهد cluster count بیش از resolution داده است.

نتیجه‌ی قدیمی phase1 نیز 447-centroid را حدود 0.96343 و 1000-centroid را حدود 0.96045 گزارش کرده بود؛ یعنی همان آزمایش محلی برتری pseudo-centroid را ثابت نکرد.

#### P1.6 PrototypicalLoss از prototype غیروحد و initialization نامطمئن استفاده می‌کند

در `src/train.py:380-415` prototypeها تصادفی unit-norm شروع می‌شوند، اما EMA update دوباره normalize نمی‌شود. cosine به‌صورت dot-product با prototypeهایی با norm متغیر محاسبه می‌شود. کلاس unseen نیز random prototype را به‌عنوان negative دارد و update قبل از loss همان batch انجام می‌شود.

این می‌تواند علت بخشی از افت variantهای proto باشد. قبل از نتیجه‌گیری «proto بد است»، implementation باید first-observation initialization، count tracking و renormalization داشته باشد.

### P2 — کیفیت مهندسی و هزینه

#### P2.1 config صوتی ensemble کنترل نمی‌شود

submission config audio checkpoint اول را برای تمام مدل‌ها استفاده می‌کند و equality check ندارد. ensemble دو checkpoint با duration/VAD policy متفاوت silently نادرست اجرا می‌شود.

#### P2.2 short audio چند بار بیهوده تکرار می‌شود

window builder برای فایل کوتاه، همان window padded/tiled را تا سقف W تکرار می‌کند؛ ممکن است 8 forward یکسان انجام شود. dedup windowها runtime را کم می‌کند بدون تغییر prediction.

#### P2.3 augmentation احتمالاً بیش از حد قوی است

config پایه هم‌زمان RIR=0.4، MUSAN noise=0.4، music=0.2، MP3=0.3، time mask=0.5 و چند waveform transform دارد. در few-shot این ترکیب می‌تواند identity cue را خراب کند. باید به‌صورت cumulative ablation و condition-matched بررسی شود، نه all-on.

#### P2.4 رفتار train/eval encoderها متفاوت و نیازمند ثبت است

برای بعضی wrapperهای pretrained، encoder حتی هنگام unfreeze در eval mode نگه داشته می‌شود تا BN ثابت باشد؛ gradient عبور می‌کند ولی dropout خاموش است. این الزاماً bug نیست، اما یک hyperparameter پنهان است و باید در metadata/ablation ثبت شود.

#### P2.5 package verifier smoke محدود دارد

verifier فقط 8 فایل اول sorted را می‌سنجد. decode failure، MP3 واقعی 48k، فایل کوتاه، batch-size متفاوت، همه‌ی 3604 فایل و سقف زمانی/VRAM را پوشش نمی‌دهد. packageهای نهایی باید adversarial smoke set و full-runtime rehearsal داشته باشند.

#### P2.6 READMEها و reportهای قدیمی stale هستند

`submission/README.md` هنوز ensemble قدیمی را توصیف می‌کند، در حالی‌که package جاری دو CAM++ و Top-5 rule است. README ریشه و EDA Phase3 نیز به یافته‌های اکنون ابطال‌شده تکیه دارند. این گزارش و deep EDA باید canonical شوند.

## 8. اعتبار نتایج و artifactهای قدیمی

### 8.1 نتایج رسمی

| تاریخ/منبع | سیستم | Macro-F1 | اعتبار |
|---|---|---:|---|
| `reports/lb_log.md`، 2026-08-15 | CAM++ تک‌مدل | 0.9505 | رسمی ثبت‌شده |
| `reports/lb_log.md`، 2026-08-19 | CAM++ + 554 unknown centroids | 0.9625 | بهترین رسمی ثبت‌شده در log |
| گزارش root-cause | no-proto/metric 60/40 | 0.96060 | گزارش‌شده به‌عنوان leaderboard؛ به lb_log اضافه نشده |
| Top-5 candidate | no-proto/metric 60/40 + top5 | 0.97536 local | هنوز در گزارش ارسال‌نشده |

### 8.2 checkpoint metrics

ممیزی checkpointها نشان داد CAM++های اصلی val حدود 0.932 تا 0.943 دارند؛ ECAPA حدود 0.921، ERes2Net حدود 0.8085 و TitaNet حدود 0.8426. امتیاز 0.975 یک metric خام checkpoint نیست؛ حاصل post-hoc fusion/decision روی protocol محلی است.

پس فایل‌های قدیمی بی‌ارزش نیستند، اما استفاده‌ی درست آن‌ها چنین است:

- checkpointهای CAM++ تاریخی: anchor و source diversity/teacher.
- ECAPA: تنها در صورت error decorrelation روی OOF/hard.
- ERes2Net/TitaNet قدیمی: evidence منفی برای recipe اجراشده، نه حکم نهایی درباره‌ی معماری.
- cacheهای قدیمی بدون SHA/manifest: فقط exploratory، نه calibration نهایی.
- HPO best_params: invalid.
- Phase3 embedding score: historical only.

### 8.3 چرا Random 0.975 قابل اعتماد کافی نیست

گزارش موجود نشان می‌دهد:

| سیستم | Random | Hard | Leaderboard |
|---|---:|---:|---:|
| کنترل تاریخی | 0.95781 | 0.92344 | 0.96250 |
| no-proto/metric 60/40 | 0.97238 | 0.92311 | 0.96060 |
| Top-5 | 0.97536 | میانگین 0.93153 | نامشخص |

بهبود بزرگ Random به Hard منتقل نشده است. همچنین hard set ساخته‌شده از همان dataset است و distribution واقعی evaluation را تضمین نمی‌کند. بنابراین Top-5 یک candidate مهندسی خوب برای یک مشاهده‌ی leaderboard است، ولی درمان representation shift نیست.

## 9. معماری برنده‌ی نهایی پس از تطبیق طراحی مستقل با وضعیت پروژه

طراحی مستقل و ممیزی کد به یک نسخه‌ی عملی مشترک می‌رسند:

### 9.1 لایه‌ی representation

- مدل اصلی: CAM++ condition-robust، چون تنها خانواده‌ی فعلی با evidence leaderboard قوی است.
- مدل مکمل: ERes2NetV2/ResNet speaker encoder جدید با recipe استاندارد؛ ECAPA فقط در صورت OOF complementarity.
- WavLM large فقط وقتی اضافه شود که gain شرطی و decorrelation آن هزینه‌ی runtime/ZIP را توجیه کند.
- stereo branch حذف؛ mono 16k canonical.

### 9.2 head و objective

- primary head دقیقاً 446-way AAM/CosFace.
- OOD head باینری مستقل.
- metric projection مشترک با supervised contrastive/AM-centroid episodic objective.
- pseudo-unknownها به‌عنوان background pair یا auxiliary prototype با confidence weight؛ نه classهای هم‌ارز known در softmax اصلی.
- loss برای unknown identity فقط زمانی hard می‌شود که cluster stability بین encoder/augmentation/run بالا باشد.

### 9.3 prototype و scoring

- prototype هر speaker از فایل‌های train معتبر و file-balanced ساخته شود.
- estimator: quality-weighted trimmed mean یا medoid+mean؛ clipهای کم‌گفتار weight کمتر.
- scoreها:

  - top-1 known cosine/logit
  - top1-top2 margin
  - adaptive cohort z/t normalization
  - local density یا distance به background
  - binary OOD probability
  - pseudo-background top-k evidence
  - window agreement/variance
  - speech ratio، RMS، clipping/flatness، duration
  - prototype reliability و speaker-specific tail

### 9.4 calibrator

یک logistic regression regularized یا GBDT کوچک روی evidenceها، با cross-fitting group-aware. خروجی calibrator فقط احتمال known/unknown را تعیین می‌کند؛ در صورت known، شناسه از fused known scores گرفته می‌شود. این separation جلوی bias جمع 554 tail را می‌گیرد.

### 9.5 fusion

- score normalization per encoder قبل از fusion.
- وزن‌ها با OOF و objective شامل mean Macro-F1، worst-fold و hard-known recall انتخاب شوند.
- مدل جدید فقط اگر gain marginal یا error decorrelation واقعی دارد وارد package شود.
- سه fold یک معماری می‌توانند variance را کم کنند، اما جای encoder مکمل را نمی‌گیرند.

## 10. protocol اعتبارسنجی که باید معیار تصمیم باشد

### 10.1 سه regime

1. **OOF standard**: file-disjoint، سه fold، exact submission probability path.
2. **Known-hard**: برای هر speaker فایل با کمترین LOO margin/بیشترین disagreement، بدون استفاده در fit calibrator همان fold.
3. **Novel-background**: گروه‌های pseudo-unknown به‌طور کامل بین fit/calibration جدا؛ یک pseudo identity نباید دو طرف باشد.

### 10.2 قواعد عدم leakage

- duplicate family یک واحد split است.
- همه‌ی windowهای یک فایل یک واحدند.
- pseudo-speaker group یک واحد split است.
- prototype هیچ query file را نمی‌بیند.
- threshold هر row در fold دیگری fit می‌شود.
- final score از concatenation predictionهای cross-fitted محاسبه می‌شود.

### 10.3 metric board

برای هر run حداقل این‌ها ثبت شوند:

- 447-way Macro-F1 mean/std/min fold
- known macro recall و known accuracy
- unknown precision/recall/F1
- known→unknown، known→wrong-known، unknown→known
- low-speech/low-SNR/long/short/real-MP3 buckets
- calibration ECE/Brier برای OOD
- runtime، peak VRAM، ZIP size
- checkpoint/config/data/code SHA

یک run فقط با یک scalar «بهتر» نیست؛ باید failure mode مورد هدف را بهبود دهد.

## 11. برنامه‌ی آزمایش مرحله‌ای و کم‌هزینه

### Tier 0 — بدون training GPU؛ اجباری

1. اصلاح sampler، exact validation path، artifact alignment، HPO parser، centroid policy و split report.
2. regenerate OOF برای checkpointهای موجود با filename manifest؛ فقط inference لازم است.
3. ساخت robust prototype و feature table quality/evidence.
4. cross-fit calibration: head-only، Top-5، known-centroid، cohort-normalized و logistic/GBM.
5. ارزیابی Random/Hard/Novel-background و تعیین confidence interval.

شرط عبور: gain حداقل +0.003 در OOF mean، عدم افت worst-fold بیش از 0.001 و بهبود known-hard recall بدون کاهش unknown recall زیر guardrail از پیش تعیین‌شده.

### Tier 1 — دو training run gate

فقط اگر Tier0 کافی نبود:

| run | تفاوت علمی |
|---|---|
| A | known-first CAM++: 446-way + binary OOD، بدون pseudo hard labels |
| B | همان A + confidence-weighted auxiliary background/metric loss |

همه‌چیز جز یک عامل ثابت: split، seed، data manifest، augmentation، batch sampler، epochs، optimizer و environment.

Go اگر treatment حداقل +0.002 OOF mean و +0.004 known-hard Macro-F1 بدهد، unknown recall guardrail را نگه دارد و در دو fold جهت gain یکسان باشد. در غیر این صورت rest folds اجرا نشوند.

### Tier 2 — encoder مکمل

یک ERes2NetV2/ResNet speaker encoder با همان protocol. شرط ورود به ensemble:

- standalone قابل قبول؛
- حداقل 20% از خطاهای CAM++ را روی OOF/hard اصلاح کند؛
- fusion cross-fitted حداقل +0.002 بدهد؛
- runtime package زیر budget بماند.

### Tier 3 — full-data fit و package

- epoch count از OOF از پیش تعیین شود؛ full-data validation diagnostic برای tuning استفاده نشود.
- همه‌ی 4459 فایل معتبر استفاده شوند، conflict quarantine شود.
- prototype full-data با policy ثابت ساخته شود.
- calibrator از OOF freeze شود.
- full 3604-file dry run در leaderboard venv با اندازه‌گیری time/VRAM.

## 12. استفاده‌ی درست از Vast.ai

اکنون اجرای HPO وسیع روی Vast توصیه نمی‌شود. علت کمبود GPU نیست؛ objective فعلی measurable نیست. پس از patch P0:

- ابتدا Tier0 را local/یک GPU انجام دهید.
- Tier1 را روی یک instance و یک image/commit/lock ثابت، sequential اجرا کنید.
- gate پس از هر run artifact را بررسی کند؛ شکست gate باید queue را متوقف کند.
- full directories شامل checkpoint، bundle، OOF، logs و manifest برگردند؛ تنها `.pt` کافی نیست.
- HPO فقط پس از یک baseline تمیز و با search space کوچک روی LR/margin/augmentation strength انجام شود.

محیط Vast فعلی PyTorch 2.13 در lock و driver CUDA≥13 را هدف می‌گیرد، در حالی‌که leaderboard torch 2.10/CUDA12.8 است. این الزاماً خطا نیست، ولی package compatibility باید در venv leaderboard مستقل verify شود. base image CUDA12.1 نیز با wheel جدید از طریق setup مدیریت شده؛ هر تغییر این stack باید به‌عنوان تغییر experiment environment ثبت شود.

## 13. ترتیب patch پیشنهادی

این ممیزی production code را تغییر نداده است. ترتیب implementation پیشنهادی:

1. `BalancedOODBatchSampler` + tests ratio/coverage/epoch determinism.
2. validation exact probability-average + best checkpoint selection consistency.
3. artifact schema v2 با filenames/labels/SHA و hard alignment checks.
4. HPO regex/structured JSON result و همگام‌سازی nested loss fields.
5. raw audit dataframe برای split report.
6. centroid builder با clean_duplicates و audio policy دقیق checkpoint.
7. جداسازی `max_known_cosine` از pseudo-background cosine و منع overwrite دو decision rule.
8. content-addressed cache keys.
9. prototype normalization/initialization و quality weighting.
10. verifier adversarial/full-runtime و README canonical.

بعد از هر patch test اختصاصی و یک regression replay روی checkpoint تاریخی 0.9625 لازم است تا anchor تغییر نکند.

## 14. تصمیم درباره‌ی candidateهای فعلی

### Top-5 فعلی

از نظر مهندسی package آماده و smoke-tested گزارش شده است. ارزش ارسال آن «اندازه‌گیری یک hypothesis مشخص» است: آیا حذف cardinality bias روی leaderboard کمک می‌کند؟ اگر quota ارسال اجازه می‌دهد، یک ارسال کنترل‌شده مفید است؛ اما نتیجه‌ی آن نباید برای tune چند threshold متوالی استفاده شود.

### کنترل تاریخی

باید همیشه به‌عنوان anchor reproducibility نگه داشته شود. هر تغییر builder/inference باید روی cache و package آن prediction-equivalent باشد.

### CAM++/ECAPA centroid

local 0.9649 به‌تنهایی کافی نیست. ECAPA باید با OOF filename-aligned و exact submission path دوباره سنجیده شود؛ artifact فعلی امکان mismatch دارد.

### known-first profiles فعلی

فرضیه‌ی معماری درست و همسو با طراحی مستقل است، اما sampler و checkpoint-selection mismatch قبل از اجرای Vast باید اصلاح شوند. در غیر این صورت run نتیجه‌ی خود معماری را تمیز اندازه نمی‌گیرد.

## 15. معیار موفقیت واقعی برای عبور از 0.972

هدف نباید «local score ≥0.98» باشد. معیار آمادگی submission:

- OOF exact-path mean حداقل 0.972 با std پایین؛
- worst-fold حداقل 0.968؛
- known-hard بهبود معنادار نسبت به anchor؛
- novel-background بدون collapse؛
- gain در حداقل دو seed/fold هم‌جهت؛
- full runtime/VRAM/ZIP pass؛
- تمام artifactها content-addressed و قابل بازسازی؛
- فقط یک hypothesis اصلی در هر leaderboard submission.

برای اختلاف 0.01 در Macro-F1، چند known rejection می‌توانند تعیین‌کننده باشند؛ بنابراین calibration محافظه‌کار و prototype reliability احتمالاً ارزشمندتر از افزودن کورکورانه‌ی صدها میلیون پارامتر است.

## 16. مواردی که نباید تکرار شوند

- استفاده از `speaker-hpo-best` فعلی.
- مقایسه‌ی checkpointها روی validationهای نامختلف بدون filename join.
- گزارش score tune‌شده روی همان data به‌عنوان estimate تعمیم.
- فرض اینکه 554 cluster ground truth است.
- جمع probability همه‌ی pseudo-classها بدون normalization.
- اضافه‌کردن encoder صرفاً چون standalone مشهور است.
- ensemble برابر چند fold هم‌بسته قبل از error-diversity analysis.
- تغییر هم‌زمان augmentation، loss، sampler و architecture در یک run.
- overwrite base config یا checkpoint بدون hash/manifest.

## 17. خروجی‌های canonical این مرحله

- `eda/DEEP_DATA_UNDERSTANDING_REPORT.md` — مرجع درک داده.
- `eda/deep_data_summary.json` — خلاصه‌ی ماشین‌خوان.
- `eda/deep_audio_inventory.csv` — inventory فایل‌به‌فایل.
- `eda/deep_known_speaker_diagnostics.csv` — speaker/file diagnostics.
- `eda/deep_unknown_pseudo_speakers.csv` — partition فرضی unknown.
- `eda/deep_exact_duplicate_groups.csv` — duplicate/conflict.
- `scripts/deep_data_eda.py` — بازتولید EDA.
- `reports/research/SCIENCE_REVIEW_AND_INDEPENDENT_WINNING_DESIGN_2026-08-27.md` — مرور علمی و طرح مستقل.
- این فایل — مرجع یکپارچه‌ی مسابقه، کد، اعتبار نتایج و roadmap.

## 18. جمع‌بندی نهایی

پروژه از نظر infrastructure جلوتر از یک baseline معمولی است: CAM++ قوی، package offline، UI/queue/Vast، checkpoint bundle، OOF و decision tooling دارد. مشکل این نیست که «همه‌چیز باید از صفر نوشته شود». مشکل این است که چند mismatch کوچک در sampler، forward، split/artifact alignment و calibration دقیقاً در محدوده‌ی همان 0.01 مورد نیاز برای رتبه‌ی اول قرار گرفته‌اند.

بهترین حرکت بعدی training گسترده نیست. بهترین حرکت، patch شش invariant P0 و بازسازی یک OOF exact-path است. اگر پس از آن Top-5/quality-aware calibration به 0.972 پایدار نرسید، known-first CAM++ و سپس یک encoder مکمل واقعی منطقی‌ترین سرمایه‌گذاری GPU هستند. این مسیر کم‌ریسک‌ترین راه برای تبدیل امتیازهای محلی جذاب به یک submission واقعاً leaderboard-grade است.
