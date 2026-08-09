# 📘 سند راهنمای پیاده‌سازی کامل — مسابقه Speaker Identification (IAAA 2026)

> **هدف این سند:** این فایل یک نقشه‌ی پیاده‌سازی بسیار دقیق و self-contained است تا یک مدل هوش مصنوعی (مثلاً DeepSeek) یا یک برنامه‌نویس انسانی بتواند **بدون هیچ حدس و گمانی** معماری راه‌حل را پیاده‌سازی یا اصلاح کند.
>
> **هدف نهایی مسابقه:** رسیدن به **Macro-F1 ≥ ۰/۹۷** (۰.۹۷).
>
> **قرارداد زبان:** کدها و نام متغیرها/توابع/کامیت‌ها به **انگلیسی**؛ توضیحات به **فارسی**. همه‌ی مسیرهای فایل نسبت به ریشه‌ی پروژه (`Speaker-identification/`) است.
>
> **شاخه‌ی کاری:** `feature/advanced-speaker-id` (همان شاخه‌ی فعلی). بعد از **هر گام** یک `git commit` با پیام مناسب بزنید. `git push` فقط با اجازه‌ی کاربر.

---

## 🧭 فهرست

- [بخش ۰ — پیش‌نیاز: تعمیر محیط (بحرانی)](#بخش-۰)
- [بخش ۱ — درک مسئله (از PDF مسابقه)](#بخش-۱)
- [بخش ۲ — درک داده (EDA)](#بخش-۲)
- [بخش ۳ — طرح ذهنی راه‌حل ایده‌آل](#بخش-۳)
- [بخش ۴ — نقد معماری فعلی + علل شکست](#بخش-۴)
- [بخش ۵ — گام‌های پیاده‌سازی (۱۰ گام، هر کدام یک کامیت)](#بخش-۵)
- [بخش ۶ — دستورالعمل دقیق برای مدل DeepSeek](#بخش-۶)
- [بخش ۷ — چک‌لیست تأیید (Acceptance Criteria)](#بخش-۷)
- [پیوست A — فرهنگ نام‌گذاری و قراردادها](#پیوست-a)

---

## بخش ۰ — پیش‌نیاز: تعمیر محیط (بحرانی) <a name="بخش-۰"></a>

> ⚠️ **این بخش را حتماً اول انجام بده.** محیط پایتون پروژه در حال حاضر **شکسته است** و بدون تعمیر آن هیچ کدی (حتی تست‌ها) اجرا نمی‌شود.

### علت خرابی
- `import torch` با این خطا می‌شکند:
  ```
  OSError: [WinError 127] The specified procedure could not be found.
  Error loading "...\.venv\Lib\site-packages\torch\lib\c10_cuda.dll" or one of its dependencies.
  ```
- **root cause:** در گذشته یک دستور `uv run --with pypdf ...` داخل محیط پروژه اجرا شده که باعث یک sync ناخواسته شد. این sync، نسخه‌ی **torch 2.11.0+cu126 (CUDA)** را به اشتباه با **torch 2.13.0 (نسخه‌ی PyPI خالص، که روی ویندوز CPU-only است)** جایگزین/مخلوط کرد و پوشه‌ی DLLهای CUDA را خراب کرد. نتیجه: `c10_cuda.dll` لود نمی‌شود.
- **شواهد:** در `.venv/Lib/site-packages/` دو dist-info ناسازگار دیده می‌شود: `torch-2.11.0+cu126.dist-info` (خالی/یتیم) و `torch-2.13.0.dist-info`. پوشه‌ی `nvidia/` وجود ندارد.
- **نکته‌ی مهم:** `uv.lock` نسخه‌ی `torch = "2.13.0"` از PyPI را pin کرده — این برای GPU **نادرست** است. نسخه‌ی درست برای GPU شما (GTX 1660 Ti، درایور 610.62) **`2.11.0+cu126`** است که از index رسمی PyTorch می‌آید.

### راه‌حل (به ترتیب ترجیح)

**روش A (توصیه‌شده، بدون دانلود):** بازسازی لینک‌های torch/torchaudio از wheelهای سالمِ cu126 که در cache محلی uv هستند.

wheelهای سالم در این مسیرهای cache هستند (از قبل تأیید شده‌اند و `c10_cuda.dll`شان سالم لود می‌شود):
- torch cu126 (کامل، شامل `torch/lib/c10_cuda.dll`):
  `C:/Users/AmirhosseinEsbati/AppData/Local/uv/cache/archive-v0/7DzPcEVbudebZsLl/`
  (محتوا: `torch/`, `torchgen/`, `functorch/`, `torch-2.11.0+cu126.dist-info/`)
- torchaudio cu126:
  `C:/Users/AmirhosseinEsbati/AppData/Local/uv/cache/archive-v0/9KrSw1-r7IOCnwdg/`
  (محتوا: `torchaudio/`, `torchaudio-2.11.0+cu126.dist-info/`)

کار: محتوای این پوشه‌های cache را **روی** `.venv/Lib/site-packages/` کپی کن (overwrite) تا `torch/`, `torchgen/`, `functorch/`, `torchaudio/` و dist-infoهای درست جایگزین نسخه‌های خراب شوند. قبل از کپی، پوشه‌های خراب `torch`, `torchgen`, `functorch`, `torchaudio` و **هر دو** dist-info اضافی (`torch-2.11.0+cu126.dist-info` و `torch-2.13.0.dist-info`) را از `site-packages` حذف کن.

> نکته: cache در فرمت «unpacked archive» است (نه فایل `.whl`)، پس کپی مستقیم فایل‌ها معادل نصب است. نیازی به دانلود چندگیگابایتی نیست.

**روش B (اگر روش A نشد):** نصب مجدد torch cu126 با uv و pin کردن index رسمی PyTorch:
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv pip install --python .venv/Scripts/python.exe \
  torch==2.11.0+cu126 torchaudio==2.11.0+cu126 torchcodec==0.15.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

**روش C (سختگیرانه و کند):** اگر خواستی دقیقاً از `uv.lock` پیروی شود، باید index مخصوص PyTorch به `pyproject.toml` اضافه شود تا resolver به‌جای PyPI، نسخه‌ی cu126 را بگیرد؛ در غیر این صورت `uv sync` دوباره CPU-only نصب می‌کند. این روش را فقط در صورت آشنایی با uv/lockfile انجام بده.

### تأیید تعمیر (حتماً اجرا کن)
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
✅ انتظار: چاپ `torch 2.11.0+cu126`، `cuda True`، و نام GPU. اگر `cuda False` شد، یعنی هنوز CPU-only نصب است — برگرد و روش B را اجرا کن.

> 📌 از این به بعد، **همه‌ی دستورهای اجرا/تست** با پیشوند زیر اجرا می‌شوند (تا uv دوباره محیط را sync نکند و خرابش نکند):
> `uv run --no-sync python ...`

---

## بخش ۱ — درک مسئله (از PDF مسابقه) <a name="بخش-۱"></a>

> منبع: `Competition-Guide/iaaa-competition-2026-speaker-identification.pdf` (۶ صفحه، استخراج کامل انجام شد).

### تعریف دقیق مسئله
- **Open-Set Speaker Classification.** هدف: طبقه‌بندی تکه‌های صوتی کوتاه بر اساس **هویت گوینده**، با مدیریت صریح گوینده‌های **خارج از توزیع (OOD)**.
- صدا از **۱۰۰۰ نفر** با لهجه‌ها و شرایط ضبط متفاوت جمع شده است. برای هر نفر، **تقریباً ۵۰٪** صدا برای train و ۵۰٪ برای eval (پنهان) کنار گذاشته شده.
- **۴۴۶ گوینده‌ی «شناخته‌شده»** (هرکدام یک UUID یکتا) + **۵۵۴ گوینده‌ی «ناشناخته/OOD»** که همه در **یک کلاس واحد با برچسب `"unknown"`** ادغام شده‌اند.
- مدل باید برای هر chunk یک **توزیع احتمال ۴۴۷‌تایی** خروجی بدهد: `p_unknown` + `p_id1 … p_id446`. پیش‌بینی نهایی = **argmax** روی این ۴۴۷ احتمال.

### معیار ارزیابی (Metric) — بسیار مهم
- **متریک اصلی: Macro-Averaged F1 روی هر ۴۴۷ کلاس.**
- فرمول: برای هر کلاس `c` مقدار `F1_c` محاسبه می‌شود، سپس **میانگین ساده (macro)** روی هر ۴۴۷ کلاس گرفته می‌شود. هیچ وزنی بین زیروظیفه‌ها (شناسایی known در برابر تشخیص unknown) وجود ندارد؛ هر کلاس وزن برابر دارد.
- منطق مسابقه (نقل): چون ۴۴۶ گوینده‌ی شناخته‌شده و ۱ کلاس unknown تجمیعی داریم، macro-F1 باعث می‌شود مدل با پیش‌بینی بیش از حد کلاس اکثریت امتیاز بالا نگیرد و تعادل precision/recall هم برای گوینده‌های known و هم برای کلاس unknown حفظ شود.
- **نحوه‌ی ارزیابی:** برگزارکنندگان مدل/کد ارسالی را **روی دیتاست پنهان در محیط کنترل‌شده** اجرا می‌کنند (یعنی احتمالاً code submission است، نه فقط CSV). جزئیات دقیق submission هنوز `TO_BE_FILLED` است.

### قوانین و محدودیت‌ها
- ✅ **مجاز:** استفاده از مدل‌های pretrained صوتی عمومی؛ دیتاست‌های گفتار خارجی؛ pretraining یا self-supervised learning؛ **انسمبل چند مدل**.
- ❌ **ممنوع:** استفاده از داده‌ی برچسب‌دار حاوی گوینده‌های eval؛ دسترسی به برچسب‌های پنهان eval.
- مدل باید خروجی ۴۴۷‌تایی تولید کند (constraint سخت).
- تیم‌های برتر ممکن است ملزم به ارائه‌ی جزئیات آموزش/معماری شوند (reproducibility).
- جزئیات submission (API، بسته‌بندی، محدودیت زمان اجرا) و تاریخ‌ها **هنوز اعلام نشده** (`TO_BE_FILLED BY ORGANIZERS`).

### فرمت داده (طبق PDF)
```
audio/
  train/   <uuid>.mp3 ...
  eval/    (پنهان — تحویل داده نمی‌شود)
metadata/
  train_labels.csv      # ستون‌ها: audio_file,speaker_id
```
- نمونه‌ی `train_labels.csv`:
  ```
  audio_file,speaker_id
  e9105299-....mp3,3943d8f3-....
  57e9178b-....mp3,unknown
  ```
- فایل‌ها `.mp3` با نام UUID هستند. در PDF هیچ مشخصاتی از sample rate / طول دقیق chunk داده نشده.

### اعداد کلیدی مسابقه
| مورد | مقدار |
|---|---|
| کل افراد | 1000 |
| گوینده‌های known | 446 (UUID یکتا) |
| گوینده‌های unknown/OOD | 554 (در ۱ کلاس `"unknown"`) |
| تعداد کلاس‌های خروجی | **447** |
| متریک | **Macro-F1 روی 447 کلاس** |
| پیش‌بینی | argmax روی 447 احتمال |
| تقسیم train/eval | ~۵۰/۵۰ per-speaker (eval پنهان) |

---

## بخش ۲ — درک داده (EDA) <a name="بخش-۲"></a>

> منابع: کاوش مستقیم داده + مرور گزارش‌های موجود در `eda/` (`Phase0..3`) و JSONهای خلاصه‌ی آن‌ها.

### ساختار داده روی دیسک
```
data/
  raw/                     4529 فایل .mp3 + labels.csv
  processed/
    audio_wav/             4529 فایل .wav (16kHz, mono, PCM_16)
    audio_wav_labels.csv   4529 سطر (speaker_id, audio_file → نام .wav)
    cleaned_labels.csv     4529 سطر (speaker_id, audio_file, label int)
```

### یافته‌های کلیدی (با عدد)
- **۴۵۲۹ فایل**؛ **۴۴۷ برچسب یکتا** (۴۴۶ UUID + `"unknown"`).
- **سهم unknown = ۲۲۷۵ / ۴۵۲۹ ≈ ۵۰/۲۳٪** (یک ابرکلاس که ۵۵۴ هویت پنهان را پوشش می‌دهد).
- **توزیع per-speaker:** برای گوینده‌های known: حداقل ۵، حداکثر ۲۰، میانگین ~۵٫۰۵، **میانه ۵** فایل. (۴۳۹ گوینده × ۵ فایل؛ ۵ گوینده × ۶؛ ۱ × ۹؛ ۱ × ۲۰).
  → این یک مسئله‌ی **few-shot** واقعی است (~۵ نمونه per known speaker).
- **طول فایل‌ها:** میانگین ~۵۸٫۲ ثانیه، میانه ~۵۹٫۶، حداکثر ~۱۵۹٫۴ ثانیه؛ **۹۰/۳٪ فایل‌ها > ۳۰ ثانیه**.
  ⚠️ **نکته‌ی بحرانی:** مدل فعلی فقط **۸ ثانیه‌ی مرکزی** هر فایل را می‌بیند ⇒ **~۸۵٪ از سیگنال دور ریخته می‌شود.**
- **فایل‌های خراب:** ۷۰ فایل با طول < ۱ ثانیه (۲۲ known + ۴۸ unknown). این‌ها باید حذف شوند.
- **فایل‌های تکراری (MD5-identical):** ۹ گروه شامل ۶۹ فایل. از جمله یک گروه‌ی ۴۶تایی از فایل‌های خالی/خرابِ یکسان با برچسب‌های مخلوط (known و unknown)، و یک گوینده‌ی known (`4e3c2da0-…`) که **۳ فایل byte-identical** دارد.
  ⚠️ **ریسک leakage:** اگر split روی فایل‌ها تصادفی باشد، نسخه‌های تکراری یک فایل می‌توانند هم در train و هم در val بیفتند ⇒ متریک val خوش‌بینانه و غیرواقعی می‌شود.
- **نرخ نمونه‌برداری:** خام‌ها 16kHz **استریو**؛ پردازش‌شده‌ها 16kHz **مونو** (downmix انجام شده).
- **سکوت:** نمونه‌ی ~۶۰ فایل: میانگین ~۱۶٪ فریم‌های ساکت (RMS<0.005)؛ known و unknown تقریباً یکسان ⇒ سکوت نشانه‌ی OOD نیست.

### مرور و نقد EDAهای موجود (`eda/`)
- **Phase 0 (Labels):** صحیح و دقیق. ترکیب کلاس‌ها، عدم‌توازن (~۴۵۰×)، طرح split (۱ فایل per known برای val + ۲۰٪ unknown) — اعدادش با محاسبه‌ی مستقل مطابقت دارد. ⚠️ ادعای «Duplicate audio files: 0» فقط درباره‌ی **ردیف‌های CSV** است، نه محتوای فایل‌ها (که ۹ گروه تکراری پیدا شد).
- **Phase 1 (Duration):** صحیح. آمار طول، ۷۰ فایل خراب، معادل‌بودن طول known/unknown. ⚠️ جدول bucket در markdown بدشکل است؛ «خراب» از روی طول < ۱s استنباط شده نه خطای decode واقعی.
- **Phase 2 (Acoustic):** کافی ولی **سطحی**. ۱۰ ویژگی low-level روی ۶۰۰ فایل با **۵ ثانیه‌ی مرکزی**؛ فقط ZCR معنادار (p=0.013، اثر ناچیز |d|=0.20). نتیجه: تفکیک OOD باید در فضای embedding باشد. ⚠️ کراپ مرکزی >۹۰٪ فایل را دور می‌ریزد؛ تحلیل SNR/سکوت/session انجام نشده.
- **Phase 3 (Embeddings):** ارزشمندترین، ولی **دو عدد اصلی‌اش خوش‌بینانه و in-sample است:**
  - «سقف تشخیص ۹۵٫۴۷٪» و «OOD AUC ۰٫۹۵۲۹» با centroid روی **همان داده‌ای** محاسبه شده که centroid از آن ساخته شده (train==test). با ~۵ فایل per speaker، هر نقطه ~۲۰٪ centroid خودش است ⇒ سوگیری خوش‌بینانه.
  - **۷۰ فایل خراب هم embed شده‌اند** (کراپ مرکزی به ۵ ثانیه‌ی صفر pad می‌کند) ⇒ centroidها و توزیع OOD را آلوده می‌کند.
  - هیچ **تحلیل leakage/تکرار**، هیچ **آمار usable-seconds per speaker**، و هیچ **شبیه‌سازی Macro-F1** وجود ندارد.
  - جفت‌های «cross-speaker» در واقع cross-**label** هستند؛ ساختار درونی ابرکلاس unknown (که ۵۵۴ نفر است) بررسی نشده.

### شکاف‌هایی که EDA فعلی ندارد (و باید در گام ۵ اضافه شود)
1. ارزیابی **out-of-sample** (split-half / leave-one-out) به‌جای in-sample.
2. حذف فایل‌های خراب/تکراری از تحلیل embedding.
3. چک near-duplicate (cosine > 0.99) و cross-file similarity درون هر speaker برای کمی‌کردن leakage.
4. آمار **usable speech seconds per known speaker** (بعد از حذف خراب‌ها).
5. تحلیل ساختار درونی unknown (خوشه‌بندی/ consistency) برای ارزیابی امکان pseudo-labelling.
6. **شبیه‌سازی Macro-F1** از operating point (تنها عدد واقعاً مهم برای تصمیم‌گیری).

---

## بخش ۳ — طرح ذهنی راه‌حل ایده‌آل <a name="بخش-۳"></a>

> این بخش «معیار مرجع» است. قبل از نگاه به کد فعلی طراحی شده و باید در هر تصمیم به آن رجوع شود.

### اصول طراحی بر اساس مسئله و داده
1. **مسئله few-shot است (~۵ نمونه per known speaker).** ⇒ یک دسته‌بند (classifier) سنگینِ از-صفر روی داده‌ی کم، ضعیف است. **روش metric-learning / centroid** روی embeddingهای قوی pretrained معمولاً برتر است. EDA هم این را تأیید می‌کند (سقف centroid ~۹۵٪ حتی با bias).
2. **متریک Macro-F1 روی ۴۴۷ کلاس است.** ⇒ همه‌ی ارزیابی، انتخاب checkpoint و threshold tuning باید مستقیماً همین را هدف بگیرد (نه loss).
3. **کلاس unknown اهمیت برابر دارد (۱/۴۴۷ وزن) ولی ۵۰٪ داده است.** ⇒ تشخیص OOD باید قوی باشد؛ collapse آن (پیش‌بینی همیشه «known») فاجعه است — دقیقاً همان چیزی که در run اخیر رخ داد.
4. **فایل‌ها بلند (~۵۸s) ولی مدل ۸s می‌بیند.** ⇒ باید از **چند پنجره‌ی هم‌پوشان + میانگین‌گیری (TTA)** استفاده کرد تا کل سیگنال دیده شود.
5. **split باید بدون leakage باشد.** ⇒ فایل‌های MD5-تکراری باید در یک سمت split بمانند.
6. **انسمبل و داده‌ی خارجی مجاز است.** ⇒ برای عبور از ۰/۹۷، انسمبل چند encoder/seed و کالیبراسیون دما ابزار نهایی است.

### معماری ایده‌آل (مرجع)
```
                ورودی: waveform (کل فایل، چند پنجره‌ی ۸s هم‌پوشان)
                              │
        ┌─────────────────────┴─────────────────────┐
        │   Encoder قوی pretrained (ECAPA-TDNN /     │
        │   WavLM-Large / HuBERT) — frozen یا جزئی   │
        │   fine-tune (unfreeze آخرین بلوک‌ها)        │
        └─────────────────────┬─────────────────────┘
                              │ embedding 192d (میانگین روی پنجره‌ها)
            ┌─────────────────┼──────────────────┐
            │                 │                  │
     (الف) Speaker ID   (ب) OOD detection   (ج) Centroid / FAISS
     ArcFace head      OOD head (BCE با      similarity به centroidها
     (metric loss)     pos_weight متعادل)    → argmax + OOD-threshold
            │                 │                  │
            └─────────────────┴──────────────────┘
                       fusion → توزیع ۴۴۷‌تایی
                       (p[0]=P(unknown)، p[i]=(1−p[0])·softmax(speaker))
                       + کالیبراسیون دما + انسمبل (اختیاری)
```
- **(الف) شناسایی speaker:** ArcFace روی embedding؛ ولی با توجه به few-shot، **centroid/FAISS (ج) به‌عنوان baseline قوی** و ترکیبش با head آموزش‌دیده.
- **(ب) تشخیص OOD:** head دودویی با **نمونه‌برداری متعادل** (تا collapse نشود) + امتیاز مبتنی بر فاصله‌ی centroid (FAISS).
- **انتخاب مدل:** بر اساس **val Macro-F1**، نه loss.
- **Inference:** TTA چندپنجره‌ای + threshold ذخیره‌شده + fusion با centroid.

---

## بخش ۴ — نقد معماری فعلی + علل شکست <a name="بخش-۴"></a>

### معماری فعلی (خلاصه)
- Encoder: ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)، **کاملاً frozen**، خروجی 192d.
- Pooling: identity. Headها: `OODHead (192→1)`، `ArcFaceHead (192→446)`، `margin=0.4, scale=30`.
- Loss: `TwoPartLoss = 0.3·BCE(OOD) + 0.7·Focal(speaker, γ=2.0, smoothing=0.1)`.
- 173,121 پارامتر trainable (بقیه frozen). ۵۰ epoch، AdamW lr=1e-4.
- Fusion: `p[0]=σ(ood)`، `p[i]=(1−p[0])·softmax(speaker)`.

### نتیجه‌ی run اخیر (از لاگ کاربر)
```
Final Val:  Loss=5.7424 | OOD Acc=0.495 (≈رندوم) | Speaker Acc=0.555
OOD Threshold tuned: 0.50 (F1=0.0000)  ← OOD head کاملاً collapse شده
Train Spk Acc در epoch 50 = 0.292  ولی  Val = 0.558  (فاصله‌ی معکوس!)
```

### علل شکست (با مرجع دقیق کد)

**🔴 علت اصلی collapse هد OOD — باگ sampler در `src/pipelines/steps.py:421-429`:**
```python
class_counts = np.bincount(train_labels, minlength=len(class_map))  # 447 کلاس
weights = 1.0 / (class_counts + 1e-8)
sample_weights = weights[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, ...)
```
- این sampler **جرم احتمال هر کلاس را برابر می‌کند، نه هر ابرکلاس را.** unknown یک کلاس با ~۱۸۲۰ نمونه است (وزن هر نمونه ۱/۱۸۲۰ ⇒ جرم کل ≈ ۱). ۴۴۶ گوینده‌ی known هرکدام ~۴ نمونه دارند (وزن هر نمونه ~۱/۴ ⇒ جرم هر کلاس ≈ ۱، جمعاً ≈ ۴۴۶).
- ⇒ سهم unknown از کل drawها ≈ **۱/۴۴۷ ≈ ۰/۲۲٪** ⇒ در batch با اندازه‌ی ۳۲ فقط ~۰٫۰۷ نمونه‌ی unknown!
- BCE target `(labels==0)` تقریباً همیشه ۰ است و `pos_weight` هم ندارد (در کامیت `f256b0a` حذف شده) ⇒ کمینه‌ساز loss همیشه logit منفی بزرگ خروجی می‌دهد (همیشه «known»).
- **اثبات:** val OOD Acc = ۴۴۶/(۴۴۶+۴۵۵) ≈ ۰٫۴۹۵ — دقیقاً همان عدد لاگ. sweep آستانه در `steps.py:686-693` هیچ threshold با F1>0 پیدا نمی‌کند ⇒ F1=0.0000.
- sampler متعادلِ درست (`ood_batch_ratio=0.3`) در `src/data_pipeline.py:410-445` **وجود دارد ولی در مسیر ZenML استفاده نمی‌شود** — مسیر ZenML در `steps.py` یک dataloader جداگانه با WeightedRandomSampler ساده می‌سازد.

**🟠 سایر علل/مسائل:**
1. **Macro-F1 (متریک واقعی) هیچ‌جا محاسبه نمی‌شود.** انتخاب checkpoint بر اساس **val loss** است (`steps.py:527`)، نه متریک مسابقه.
2. **Margin در زمان ارزیابی اعمال می‌شود:** `train.py:384` و `steps.py:668` به forward، `labels=labels` پاس می‌دهند ⇒ ArcFace حتی هنگام eval مارجین `cos(θ+m)` را روی کلاس هدف اعمال می‌کند ⇒ accهای گزارش‌شده پایین‌تر از واقع.
3. **Threshold بهینه فقط print می‌شود و ذخیره نمی‌شود** (`steps.py:686-708`) ⇒ هیچ‌کجای inference از آن استفاده نمی‌شود.
4. **criterion ارزیابی نهایی ناسازگار است:** `steps.py:654` از `TwoPartLoss(ignore_index=-100)` با وزن‌های پیش‌فرض ۱/۱ و **بدون** label smoothing استفاده می‌کند، در حالی که آموزش با ۰٫۳/۰٫۷ و smoothing=0.1 بود ⇒ عدد `final_val_loss` قابل مقایسه با loss آموزش نیست.
5. **Augmentation خشن روی encoder فریزشده:** `PitchShift ±4 semitones (p=0.5)` + TimeStretch + ... در `data_pipeline.py:168-176` ویژگی‌های صوتی را آن‌قدر تغییر می‌دهد که encoder فریزشده نمی‌تواند سازگار شود ⇒ فاصله‌ی معکوس train/val acc (0.292 در برابر 0.558) را توضیح می‌دهد.
6. **۸۵٪ اتلاف سیگنال:** فقط ۸ ثانیه‌ی مرکزی/تصادفی از فایل‌های ~۵۸ ثانیه‌ای (`duration_seconds: 8.0` در config).
7. **leakage در split:** `stratified_split` (`data_pipeline.py:58-102`) گروه‌های MD5-تکراری را در نظر نمی‌گیرد.
8. **`submission/inference.py` و `tests/` حذف شده‌اند** (در git status: deleted) ⇒ مسیر ساخت submission از بین رفته.
9. **`src/ood_detector.py` (FAISS، AUC=0.953) و `src/ensemble.py` موجودند ولی هیچ‌کدام به مسیر inference/submission متصل نیستند.**

---

## بخش ۵ — گام‌های پیاده‌سازی (۱۰ گام) <a name="بخش-۵"></a>

> **قانون:** بعد از **هر گام** یک `git commit` با پیام conventional مناسب بزن (push فقط با اجازه). هر گام باید self-contained و قابل تأیید باشد. گام‌های محاسباتی (۵، ۶، ۷) را فقط **کدش را بنویس** و دستور اجرایش را بده — اجرای سنگین با کاربر است.

> **تأیید محیط قبل از شروع هر کد:** ابتدا بخش ۰ را کامل کن و `import torch` + `cuda True` را ببین.

---

### گام ۱ — زیرساخت صحت‌سنجی (متریک واقعی مسابقه)

**چرا:** متریک واقعی (Macro-F1 روی ۴۴۷ کلاس) در کد نیست. همه‌چیز (train/eval/checkpoint/submission) باید به همین یک منبع ارجاع دهد.

**وضعیت:** ✅ `src/metrics.py` **از قبل نوشته شده است.** فقط باید **تأیید** شود (تست زیر). اگر فایل را ندیدی، دقیقاً با همین مشخصات بسازش.

**محتوای `src/metrics.py` (موجود — مشخصات مرجع):**
- قرارداد فضای برچسب: کلاس `0` = unknown؛ کلاس‌های `1..446` = known. خروجی head گوینده ۴۴۶ logit با اندیس `0..445` دارد که به کلاس سراسری `j+1` نگاشت می‌شود.
- `macro_f1_score(y_true, y_pred, num_classes=447)` → از `sklearn.metrics.f1_score` با `labels=list(range(447))`, `average="macro"`, `zero_division=0`.
- `per_class_f1(...)` → آرایه‌ی 447تایی برای تشخیص.
- `fused_probs_from_logits(ood_logits, speaker_logits, temperature=1.0)` → همان فرمول fusion مدل: `p[0]=σ(ood)`، `p[i]=(1−p[0])·softmax(speaker/T)`، خروجی (batch,447) با مجموع سطر = ۱.
- `predict_global_classes(ood_logits, speaker_logits, ood_threshold=None, temperature=1.0)` → argmax روی توزیع fused؛ اگر `ood_threshold` داده شود، نمونه‌هایی که `P(unknown)>threshold` دارند به کلاس ۰ forced می‌شوند (فقط برای تحلیل محلی OOD؛ مسابقه argmax ساده است، پس برای گزارش `None` بگذار).
- `evaluate_macro_f1(all_ood_logits, all_speaker_logits, all_labels, num_classes=447, ood_threshold=None, temperature=1.0)` → dict با کلیدهای `macro_f1` (اصلی)، `ood_f1` (F1 دودویی کلاس unknown)، `known_acc`، `overall_acc`.

**تست (self-verify):**
```bash
cd "D:\Projects\My projects\IAAA_Compet\Speaker-identification"
uv run --no-sync python -c "
import numpy as np, torch
from src.metrics import macro_f1_score, fused_probs_from_logits, predict_global_classes, evaluate_macro_f1
y = np.arange(447)
assert macro_f1_score(y, y.copy()) == 1.0
assert macro_f1_score(np.zeros(10,int), np.ones(10,int)) < 0.01
p = fused_probs_from_logits(torch.randn(8,1), torch.randn(8,446))
assert p.shape==(8,447) and torch.allclose(p.sum(1), torch.ones(8), atol=1e-5)
assert (predict_global_classes(torch.full((4,1),10.0), torch.randn(4,446), ood_threshold=0.5)==0).all()
r = evaluate_macro_f1(torch.randn(4,1), torch.randn(4,446), torch.tensor([0,1,2,3]))
assert set(r)=={'macro_f1','ood_f1','known_acc','overall_acc'}
print('METRICS OK ✅')
"
```
✅ انتظار: `METRICS OK ✅`.

**کامیت ۱:** `feat(metrics): add competition macro-F1 (447-class) + fused-prob evaluation helpers`

---

### گام ۲ — پاک‌سازی داده و split بدون leakage

**چرا:** ۹ گروه فایل MD5-تکراری (بعضی با برچسب متناقض) و ۷۰ فایل خراب وجود دارد. split فعلی (`src/data_pipeline.py:58-102`) leakage دارد و فایل خراب را حذف نمی‌کند.

**تغییرات در `src/data_pipeline.py`:**

1. **تابع جدید `find_duplicate_groups(labels_df, audio_dir) -> Dict[str, List[str]]`:**
   - برای هر `audio_file` هش MD5 محتوای فایل را حساب کن (فایل‌ها در `data/processed/audio_wav/` هستند؛ از `.wav` استفاده کن چون mono/16k است و پایدارتر).
   - گروه‌بندی فایل‌ها بر اساس MD5؛ هر MD5 با بیش از یک فایل = یک گروه تکراری.
   - خروجی: `dict[md5] = [file1, file2, ...]`.
   - ⚠️ کارایی: ۴۵۲۹ فایل ~ چند صد مگابایت؛ MD5 خواندن یک‌باره قابل‌قبول است (چند ثانیه تا یک دقیقه). از `hashlib.md5` با خواندن chunkای (مثلاً 1MB) استفاده کن.

2. **تابع جدید `find_corrupted_files(labels_df, audio_dir, min_valid_duration=1.0) -> List[str]`:**
   - با `soundfile.info(path).duration` (header-only، سریع) فایل‌هایی با `duration < min_valid_duration` یا خطای خواندن یا missing را پیدا کن و لیست برگردان. (همان منطق موجود در `steps.py:367-404` را این‌جا به‌صورت reusable بیاور.)

3. **بازنویسی `stratified_split` برای leakage-awareness:**
   - امضای فعلی: `stratified_split(labels_df, val_per_known=1, unknown_val_ratio=0.2, random_seed=42)`.
   - **قبل از تقسیم:** گروه‌های تکراری را پیدا کن. برای هر گروه، **همه‌ی اعضایش باید به یک سمت (train یا val) بروند** تا leakage پیش نیاید. چون val برای known دقیقاً ۱ فایل per speaker می‌خواهد، اگر گروهی متعلق به یک known speaker است و قرار است ۱ فایل به val برود، باید مطمئن شوی هیچ duplicate آن فایل در train نیست.
     - راه‌حل ساده و امن: **همه‌ی فایل‌هایی که در یک گروه تکراری هستند را از val حذف و فقط به train بفرست** (تا val هرگز duplicate نداشته باشد)، **به‌شرطی** که هر known speaker حداقل ۱ فایل غیرتکراری برای val داشته باشد. اگر speaker‌ای همه‌ی فایل‌هایش تکراری بود (مثل `4e3c2da0-…` با ۳ فایل یکسان)، آن speaker را کامل به train بفرست و در val برایش از یکی از همان گروه استفاده نکن — یا بهتر: هشدار بده و آن speaker را با یک نسخه در val و بقیه در train نگه دار **فقط اگر** هش یکسان دقیقاً یک فایل باشد. (تصمیم امن: val را کاملاً duplicate-free نگه دار.)
   - گروه‌هایی با **برچسب‌های متناقض** (مثل گروه ۴۶تایی که هم known هم unknown دارد) را کامل گزارش کن و از val حذف کن (به train بفرست) تا val تمیز بماند.
   - رفتار فعلی را برای بقیه حفظ کن: `val_per_known=1` فایل per known speaker برای val، و `unknown_val_ratio=0.2` از unknown برای val، با seed ثابت ۴۲.

4. **حذف فایل‌های خراب** از train و val (با `find_corrupted_files`) **قبل** از ساخت dataset.

5. **گزارش:** در انتها یک `data/processed/split_report.json` بنویس با:
   - تعداد فایل‌های حذف‌شده (خراب) به تفکیک known/unknown.
   - تعداد گروه‌های تکراری و تعداد فایل‌های درگیر، و تعداد گروه‌های با برچسب متناقض.
   - آمار per-known-speaker: تعداد فایل train/val و **usable seconds** (مجموع طول فایل‌های غیرخراب آن speaker).

**نکته‌ی مهم:** این تغییر روی `prepare_data` در `src/pipelines/steps.py:212-271` هم اثر می‌گذارد — مطمئن شو `prepare_data` از split جدید و فیلتر خراب استفاده می‌کند و `split_report.json` را می‌نویسد.

**تست:**
```bash
uv run --no-sync python -c "
import pandas as pd
from src.data_pipeline import create_class_mapping, stratified_split, find_duplicate_groups, find_corrupted_files
df = pd.read_csv('data/processed/audio_wav_labels.csv')
df.columns = df.columns.str.strip()
cm = create_class_mapping(df); df['label'] = df['speaker_id'].map(cm)
tr, va = stratified_split(df, val_per_known=1, unknown_val_ratio=0.2)
# leakage check: no audio_file should appear in both train and val
assert set(tr.audio_file).isdisjoint(set(va.audio_file)), 'file leakage!'
# known coverage: each known speaker should appear in train
known = df[df.label!=0]
print('train', len(tr), 'val', len(va))
print('OK ✅')
"
```
✅ انتظار: بدون `AssertionError`، و چاپ تعداد train/val. (توجه: با حذف duplicateها ممکن است چند speaker کمتر در val باشند — قابل‌قبول است به‌شرط تمیزبودن val.)

**کامیت ۲:** `fix(data): leak-free stratified split + drop corrupted/MD5-duplicate files + split report`

---

### گام ۳ — استفاده از کل طول فایل (چند پنجره + میانگین‌گیری)

**چرا:** فایل‌ها ~۵۸ ثانیه‌اند ولی مدل فقط ۸ ثانیه می‌بیند ⇒ ~۸۵٪ اتلاف. با چند پنجره‌ی هم‌پوشان و میانگین‌گیری embedding، کل سیگنال دیده می‌شود و دقت/OOD بهتر می‌شود.

**تغییرات در `src/data_pipeline.py` (کلاس `SpeakerDataset`, خطوط 198-300):**

1. **پارامترهای جدید `__init__`:**
   - `num_train_windows: int = 1` (تعداد پنجره‌های تصادفی در train؛ شروع با ۱ تا سریع بماند، بعداً ۳-۴).
   - `eval_hop_ratio: float = 0.5` (هم‌پوشانی پنجره‌های eval؛ ۰٫۵ یعنی hop = نصف طول پنجره).
   - `max_eval_windows: int = 8` (سقف تعداد پنجره در eval/inference برای محدودکردن محاسبه).

2. **رفتار `__getitem__`:**
   - به‌جای برگرداندن یک waveform `(1, T)`، یک **لیست/تنسور پنجره‌ها** `(num_windows, 1, T)` برگردان:
     - **train:** `num_train_windows` کراپ تصادفی (در صورت augment) یا مرکزی.
     - **val/inference:** پنجره‌های ۸ ثانیه‌ای با hop = `eval_hop_ratio * window_len` که کل فایل را می‌پوشانند (تا `max_eval_windows` پنجره؛ اگر فایل کوتاه‌تر از یک پنجره بود، pad کن).
   - ⚠️ سازگاری با batch: چون تعداد پنجره‌ها بین فایل‌ها متفاوت است، یا (الف) همیشه تعداد ثابت `max_eval_windows` برگردان (با pad/تکرار) و یک mask هم بده، یا (ب) در collate با طول متغیر کنار بیا. **توصیه‌ی ساده:** در train دقیقاً `num_train_windows` (ثابت) و در eval دقیقاً `max_eval_windows` (ثابت، با pad صفر) برگردان تا batching ساده بماند.

3. **میانگین‌گیری embedding در مدل:** چون encoder (ECAPA) روی هر پنجره یک embedding 192d می‌دهد، باید خروجی روی پنجره‌ها **میانگین** شود. این کار در forward مدل انجام می‌شود — به گام ۴ و تغییر `src/model.py` رجوع کن. ساده‌ترین راه بدون تغییر مدل: در `SpeakerDataset` برای **eval** همه‌ی پنجره‌ها را بده و در حلقه‌ی eval روی آن‌ها حلقه بزن و logits را میانگین بگیر؛ ولی برای تمیزی، بهتر است مدل `(B, W, 1, T)` را بپذیرد و روی W میانگین بگیرد.

> 📌 **توصیه‌ی پیاده‌سازی (کم‌ریسک):** فعلاً شکل خروجی dataset را `(num_windows, 1, T)` نگه دار و در حلقه‌ی train/eval روی `num_windows` حلقه‌ی forward بزن و logits را **میانگین** بگیر (معادل TTA درون‌مدلی). این از تغییر ساختار batch جلوگیری می‌کند و روی GPU 6GB امن است. در `train.py`/`steps.py` جایی که `model(waveforms, ...)` صدا زده می‌شود، این منطق میانگین‌گیری را اضافه کن.

**پیکربندی در `configs/default_config.yaml` (بخش `audio`):**
```yaml
audio:
  duration_seconds: 8.0        # طول هر پنجره (بدون تغییر)
  num_train_windows: 3         # جدید: کراپ‌های تصادفی per فایل در train
  eval_hop_ratio: 0.5          # جدید: هم‌پوشانی پنجره‌های eval
  max_eval_windows: 8          # جدید: سقف پنجره در eval/inference
```

**تست:**
```bash
uv run --no-sync python -c "
import pandas as pd, torch
from src.data_pipeline import SpeakerDataset
df = pd.read_csv('data/processed/audio_wav_labels.csv'); df.columns=df.columns.str.strip()
df['label']=0
ds = SpeakerDataset(df.head(4), audio_dir='data/processed/audio_wav', sample_rate=16000, duration_seconds=8.0, augment=False)
w, l = ds[0]
print('window tensor shape:', tuple(w.shape))   # انتظار: (W, 1, 128000)
assert w.dim()==3 and w.shape[-1]==128000
print('OK ✅')
"
```

**کامیت ۳:** `feat(data): multi-window sampling (train random crops / eval sliding windows) to use full file length`

---

### گام ۴ — اصلاح sampler / loss / eval (رفع باگ‌های بحرانی)

> مهم‌ترین گام. collapse هد OOD و متریک غلط این‌جا رفع می‌شود.

#### ۴‑الف) اصلاح sampler در `src/pipelines/steps.py:421-429`

**مشکل:** `WeightedRandomSampler` با وزن per-class ⇒ سهم unknown ≈ ۱/۴۴۷.

**راه‌حل:** از **balanced batch sampler** استفاده کن که سهم unknown در هر batch را به مقدار هدف (پیش‌فرض ۰٫۵ برای تطابق با eval-mix ~۵۰/۵۰) تضمین کند. منطق آماده در `src/data_pipeline.py:410-445` هست ولی باید درست و reusable شود.

**تابع جدید در `src/data_pipeline.py`:**
```python
def make_balanced_batch_sampler(train_labels, batch_size, ood_ratio=0.5, seed=42):
    """
    برگرداندن یک sampler که هر batch را با نسبت ood_ratio از کلاس unknown (label==0)
    و (1-ood_ratio) از کلاس‌های known پر می‌کند.
    - ood_indices = where(train_labels==0), known_indices = where(train_labels!=0)
    - در هر batch: n_ood = round(batch_size*ood_ratio) از ood_indices (با جایگذاری در صورت نیاز)
      و بقیه از known_indices. known را طوری نمونه بگیر که پوشش همه‌ی speakerها حفظ شود.
    - خروجی: لیست flat از اندیس‌ها (قابل استفاده در SubsetRandomSampler یا BatchSampler).
    """
```
**استفاده در `steps.py`:** جایگزین کن بلاک `WeightedRandomSampler` (خطوط 421-429) را با:
```python
from src.data_pipeline import make_balanced_batch_sampler
ood_ratio = audio_cfg.get("ood_batch_ratio", 0.5)   # ⚠️ پیش‌فرض را به 0.5 تغییر بده
balanced_indices = make_balanced_batch_sampler(train_labels, hw_profile["batch_size"], ood_ratio=ood_ratio)
sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)
```
⚠️ **مقدار `ood_batch_ratio` در config را از 0.30 به 0.50 تغییر بده** تا با eval-mix ~۵۰/۵۰ هم‌خوان شود و هد OOD به‌اندازه‌ی کافی نمونه‌ی مثبت ببیند.

#### ۴‑ب) `pos_weight` در BCE (`src/train.py`)

`TwoPartLoss.__init__` (خطوط 170-198) از `nn.BCEWithLogitsLoss()` بدون `pos_weight` استفاده می‌کند. برای استحکام بیشتر (حتی با sampler متعادل)، `pos_weight` را قابل‌تنظیم کن:
- پارامتر جدید `ood_pos_weight: float = 1.0` به `TwoPartLoss` اضافه کن.
- `self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ood_pos_weight))`.
- در `steps.py` مقدار را از config بخوان: `training.ood_pos_weight` (پیش‌فرض ۱٫۰؛ با sampler متعادل ۰٫۵ معمولاً ۱٫۰ کافی است).

#### ۴‑ج) ارزیابی بدون margin

- در `src/train.py` تابع `validate_epoch` (خط 384) و در `src/pipelines/steps.py` تابع `evaluate_model` (خط 668)، forward را **بدون labels** صدا بزن تا ArcFace مارجین اعمال نکند:
  ```python
  ood_logits, speaker_logits = model(waveforms)   # labels=None → بدون margin
  ```
- ⚠️ ولی loss هنوز به labels نیاز دارد. راه‌حل: در val، forward را **دو بار** نزن؛ بلکه:
  - برای **متریک** (accuracy/F1) از logits بدون margin استفاده کن.
  - برای **loss** می‌توانی همان logits بدون margin را به criterion بدهی (criterion خودش remap می‌کند) — ولی چون ArcFace بدون labels خروجی‌اش `cosine*s` است (نه margin)، loss محاسبه‌شده کمی متفاوت از train خواهد بود. **قابل‌قبول است** چون هدف اصلی val، متریک Macro-F1 است نه loss. اگر خواستی loss val دقیقاً مثل train باشد، forward دوم با labels بزن فقط برای loss (هزینه‌ی محاسباتی ۲×). **توصیه:** در val فقط یک forward بدون labels بزن و checkpoint selection را بر اساس Macro-F1 انجام بده (نه loss).

#### ۴‑د) انتخاب checkpoint بر اساس val Macro-F1

در `steps.py` حلقه‌ی آموزش (خطوط 497-568):
- هر epoch، روی val علاوه‌بر loss، **Macro-F1** را با `evaluate_macro_f1` از `src/metrics.py` حساب کن. برای این کار باید در `validate_epoch` علاوه‌بر accuracy، **logits و labels را جمع‌آوری و برگردانی** (مثل کاری که `evaluate_model` در خطوط 660-677 می‌کند). سپس:
  ```python
  val_m = evaluate_macro_f1(all_ood_logits, all_speaker_logits, all_labels, num_classes=len(class_map))
  val_metrics["macro_f1"] = val_m["macro_f1"]
  ```
- معیار best/early-stopping را از `val_metrics["loss"]` به `val_metrics["macro_f1"]` تغییر بده (بیشتر=بهتر). یعنی:
  ```python
  if val_metrics["macro_f1"] > best_val_f1:  # به‌جای val_loss < best_val_loss
      best_val_f1 = val_metrics["macro_f1"]; patience_counter = 0; save best
  else: patience_counter += 1
  ```
- `best_val_f1` را با `-inf` مقداردهی اولیه کن.

#### ۴‑ه) ذخیره‌ی tuned OOD threshold

در `evaluate_model` (خطوط 686-708):
- بعد از یافتن `best_threshold` و `best_f1`، آن‌ها را **در metrics dict و در checkpoint** ذخیره کن:
  ```python
  metrics["ood_threshold"] = float(best_threshold)
  metrics["ood_threshold_f1"] = float(best_f1)
  # و در checkpoint ذخیره شود:
  # ckpt["ood_threshold"] = best_threshold
  ```
- ⚠️ رفع باگ fallback: اگر همه‌ی F1ها صفر بودند، `best_threshold` روی ۰٫۵ می‌ماند؛ در این حالت threshold را بر اساس نرخ مثبت val (مثلاً میانه‌ی `ood_probs`) تنظیم کن تا همیشه یک مقدار معقول ذخیره شود.

#### ۴‑و) یکسان‌سازی criterion ارزیابی نهایی

در `steps.py:654`، `criterion = TwoPartLoss(ignore_index=-100)` را با **همان وزن‌ها و smoothing آموزش** بساز تا عدد loss قابل مقایسه باشد:
```python
criterion = TwoPartLoss(
    ignore_index=-100, use_focal=True, focal_gamma=2.0,
    ood_weight=train_cfg.get("ood_loss_weight",1.0),
    speaker_weight=train_cfg.get("speaker_loss_weight",1.0),
    label_smoothing=0.0,   # در eval smoothing را 0 نگه دار (استاندارد)
)
```

#### ۴‑ز) کاهش Augmentation

در `src/data_pipeline.py` کلاس `AudioAugmentation` (خطوط 168-176):
- `PitchShift(min_semitones=-4, max_semitones=4, p=0.5)` → `PitchShift(min_semitones=-1, max_semitones=1, p=0.3)`.
- (اختیاری) `TimeStretch` p=0.3→0.2. بقیه دست نخورده.
- دلیل: encoder فریزشده نمی‌تواند با pitch shift شدید سازگار شود و این باعث فاصله‌ی معکوس train/val acc شده.

**پیکربندی در `configs/default_config.yaml`:**
```yaml
audio:
  ood_batch_ratio: 0.50    # ← از 0.30 (تطابق با eval-mix ~50/50)
training:
  ood_pos_weight: 1.0      # ← جدید (با sampler متعادل معمولاً 1.0 کافی است)
```

**تست:**
```bash
uv run --no-sync python -c "
import numpy as np, torch
from src.data_pipeline import make_balanced_batch_sampler
labels = np.array([0]*1820 + list(range(1,447))*4)  # شبیه‌سازی توزیع واقعی
idx = make_balanced_batch_sampler(labels, batch_size=32, ood_ratio=0.5)
idx = np.array(idx)
# بررسی نسبت unknown در نمونه‌ی کشیده‌شده
frac_unknown = (labels[idx]==0).mean()
print('fraction unknown drawn:', round(float(frac_unknown),3))  # انتظار: نزدیک 0.5 نه 0.002
assert frac_unknown > 0.3, 'sampler still starves OOD!'
from src.train import TwoPartLoss
c = TwoPartLoss(ignore_index=-100, ood_pos_weight=2.0)
print('pos_weight set OK')
print('OK ✅')
"
```
✅ انتظار: `fraction unknown drawn` نزدیک ۰٫۵ (نه ۰٫۰۰۲) و بدون خطا.

**کامیت ۴:** `fix(train): balanced OOD/known batch sampler, BCE pos_weight, no-margin eval, Macro-F1 checkpoint selection, persist OOD threshold, gentler augmentation`

---

### گام ۵ — بازنویسی EDA فاز ۳ بدون bias (اسکریپت — اجرا با کاربر)

**چرا:** اعداد کلیدی فاز ۳ (سقف ۹۵٫۵٪ و AUC ۰٫۹۵۳) **in-sample** و با فایل‌های خراب محاسبه شده ⇒ خوش‌بینانه. برای تصمیم‌گیری درست به سقف **out-of-sample** و **شبیه‌سازی Macro-F1** نیاز داریم.

**فایل:** `src/eda_embeddings.py` را بازنویسی کن (یا تابع جدید `unbiased_centroid_eval` اضافه کن).

**الزامات:**
1. **حذف فایل‌های خراب و تکراری** قبل از embed کردن (از توابع گام ۲ استفاده کن).
2. **ارزیابی out-of-sample:** به‌جای centroid روی همه‌ی فایل‌ها و امتیاز روی همان‌ها:
   - **split-half:** برای هر known speaker، فایل‌هایش را نصف کن (مثلاً ۴ train / ۱ test، یا k-fold با k=تعداد فایل‌ها). centroid را از بخش train بساز و روی test امتیاز بگیر. میانگین روی foldها = سقف واقعی‌تر.
   - **leave-one-out (LOO):** برای هر فایل، centroid speakerش را بدون آن فایل بساز و امتیاز بگیر.
3. **ساختار درونی unknown:** چون unknown ۵۵۴ نفر است ولی برچسب واحد دارد، نمی‌توانی per-person consistency را مستقیم بسنجی؛ ولی می‌توانی **خوشه‌بندی** (مثلاً KMeans روی embeddingهای unknown) کنی و ببینی چقدر جمع‌وجور است — این برای pseudo-labelling مفید است.
4. **شبیه‌سازی Macro-F1:** با operating point مشخص (TPR/FPR برای OOD و دقت known)، و با فرض ترکیب ~۵۰/۵۰ eval، **Macro-F1 تخمینی** را محاسبه و گزارش کن (این تنها عدد واقعاً مهم است).
5. خروجی: به‌روزرسانی `eda/phase3_embedding_summary.json` و گزارش markdown با **اعداد out-of-sample** و برچسب واضح «unbiased».

> 📌 **اجرا با کاربر:** این اسکریپت روی ~۴۵۰۰ فایل embed می‌گیرد (GPU، چند دقیقه تا ~۱۵ دقیقه). کد را بنویس و دستور اجرا را بده:
> ```bash
> uv run --no-sync python -m src.eda_embeddings
> ```

**کامیت ۵:** `refactor(eda): unbiased split-half/LOO centroid eval + drop corrupted/dupes + macro-F1 simulation`

---

### گام ۶ — Centroid baseline + fusion با embedding cache (اسکریپت — اجرا با کاربر)

**چرا:** مسئله few-shot است؛ روش centroid روی embeddingهای قوی معمولاً از head آموزش‌دیده قوی‌تر است (EDA سقف ~۹۵٪ را نشان می‌دهد). این baseline هم یک مرجع قوی می‌دهد و هم برای fusion.

**فایل جدید:** `src/centroid_baseline.py`

**الزامات:**
1. **Embedding cache:** یک‌بار embedding همه‌ی فایل‌های train/val را با encoder frozen استخراج و روی دیسک ذخیره کن (`data/processed/embeddings_train.npy`، `..._val.npy` + فایل labels متناظر). این کار را idempotent کن (اگر cache هست، دوباره حساب نکن). از منطق embedding موجود در `src/eda_embeddings.py` (که از ECAPA استفاده می‌کند) بهره ببر، ولی **چند پنجره‌ای** (گام ۳) با میانگین‌گیری.
2. **Centroid classifier:**
   - برای هر known speaker، centroid = میانگین embeddingهای trainاش (L2-normalized).
   - پیش‌بینی known = argmax cosine similarity به centroidها.
   - **OOD score = 1 − max_cosine_similarity** (یا 1 − mean top-k). threshold را روی val طوری tune کن که **Macro-F1** بیشینه شود (نه فقط binary F1).
3. **ارزیابی:** روی val، Macro-F1 را با `src/metrics.py` حساب کن و با نتیجه‌ی مدل آموزش‌دیده مقایسه کن.
4. **Fusion (اختیاری ولی توصیه):** ترکیب احتمالات مدل آموزش‌دیده و centroid (مثلاً میانگین موزون، یا استفاده از `combine_ood_scores` موجود در `src/ood_detector.py` برای OOD و میانگین softmax برای speaker).
5. خروجی: گزارش Macro-F1 centroid به‌تنهایی، مدل به‌تنهایی، و fusion، + ذخیره‌ی threshold بهینه.

> 📌 `src/ood_detector.py` (`FAISSOODDetector`) از قبل این منطق را دارد — می‌توانی از آن استفاده کنی یا یک پیاده‌ی ساده‌ی numpy بنویسی. مهم: threshold باید برای **Macro-F1** بهینه شود.

**دستور اجرا (با کاربر):**
```bash
uv run --no-sync python -m src.centroid_baseline
```

**کامیت ۶:** `feat(baseline): centroid classifier + OOD threshold tuned for macro-F1 + embedding cache + fusion`

---

### گام ۷ — Fine-tune بخشی از encoder (اسکریپت — اجرا با کاربر، ۸-۱۲ ساعت)

**چرا:** encoder کاملاً frozen است (فقط ۱۷۳هزار پارامتر trainable). برای عبور از سقف فعلی به ۰/۹۷، احتمالاً باید بخشی از encoder را fine-tune کرد تا embeddingها برای این ۱۰۰۰ گوینده‌ی خاص تطبیق پیدا کنند.

**تغییرات:**

1. **در `src/encoders.py` کلاس `ECAPAEncoder`:** متد `unfreeze` فعلی (خطوط 369-373) همه‌چیز را باز می‌کند. متد جدید اضافه کن:
   ```python
   def unfreeze_last_n_blocks(self, n: int = 2):
       """فقط n بلوک آخر ECAPA-TDNN را unfreeze کن (بقیه frozen).
       SpeechBrain ECAPA modules در self.classifier.mods.embedding_model هستند.
       باید submoduleهای embedding_model را پیمایش کنی و فقط n تای آخر را requires_grad=True کنی.
       ابتدا همه را freeze کن، سپس n بلوک آخر را باز کن. batch normها را در eval نگه دار."""
   ```
   ⚠️ **احتیاط:** forward فعلی (خط 345) در `torch.no_grad()` اجرا می‌شود — برای fine-tune باید این `no_grad` **فقط وقتی** encoder کاملاً frozen است فعال باشد. منطق را اصلاح کن: اگر encoder partially-trainable است، `no_grad` را بردار (یا conditional کن).
   ⚠️ **eval mode:** متد `train()` (خط 314) encoder را همیشه eval نگه می‌دارد تا BatchNorm خراب نشود — این را حفظ کن (fine-tune با BN در eval mode امن‌تر است).

2. **در config (`model.encoder_config.ecapa`):**
   ```yaml
   ecapa:
     source: speechbrain/spkrec-ecapa-voxceleb
     freeze_encoder: false          # ← برای fine-tune
     unfreeze_last_n_blocks: 2      # ← جدید
   ```

3. **در `steps.py` (train_model):** اگر encoder partially-trainable است، **learning rate متفاوت** برای encoder در برابر headها بگذار (param groups):
   ```python
   encoder_params = [p for n,p in model.named_parameters() if 'encoder' in n and p.requires_grad]
   head_params    = [p for n,p in model.named_parameters() if 'encoder' not in n and p.requires_grad]
   optimizer = torch.optim.AdamW([
       {'params': encoder_params, 'lr': train_cfg.get('encoder_lr', 1e-5)},   # LR کوچک‌تر برای encoder
       {'params': head_params,    'lr': train_cfg['learning_rate']},          # 1e-4 برای headها
   ], weight_decay=train_cfg['weight_decay'])
   ```
   ⚠️ VRAM: GTX 1660 Ti (۶GB). fine-tune بخشی از ECAPA با batch کوچک (مثلاً ۴-۸) و AMP انجام شود. اگر OOM شد، `batch_size` را کم کن یا `duration_seconds`/`num_train_windows` را کاهش بده.

4. **config:**
   ```yaml
   training:
     encoder_lr: 1.0e-05    # ← جدید
   ```

**دستور اجرا (با کاربر — سنگین، ۸-۱۲ ساعت):**
```bash
uv run --no-sync python -m src.pipelines.run_pipeline --run train
```

**کامیت ۷:** `feat(encoder): partial unfreeze (last-n blocks) + separate encoder LR for fine-tuning`

---

### گام ۸ — بازسازی مسیر inference/submission

**چرا:** `submission/inference.py` حذف شده (در git status: deleted). باید بازسازی شود با TTA + threshold ذخیره‌شده + fusion با centroid.

**فایل جدید:** `submission/inference.py` (و `submission/__init__.py` خالی).

**الزامات:**
1. **CLI با click:** پارامترهای `--data-dir` (پوشه‌ی فایل‌های صوتی تست) و `--predictions-file-path` (خروجی CSV).
2. **بارگذاری مدل:** از `checkpoints/best_model.pt` + config؛ با `create_model_from_config` مدل را بساز و `state_dict` و `class_map` را از checkpoint بخوان.
3. **TTA چندپنجره‌ای:** برای هر فایل تست، پنجره‌های ۸ ثانیه‌ای با هم‌پوشانی ۵۰٪ (hop=4s) بساز (تا `max_eval_windows`)، `predict_proba` را روی هر پنجره بزن و **میانگین احتمالات** را بگیر (renormalize کن).
4. **استفاده از threshold ذخیره‌شده:** اگر در checkpoint `ood_threshold` هست، می‌توانی (اختیاری) نمونه‌هایی با `P(unknown)>threshold` را به کلاس ۰ نزدیک‌تر کنی — ولی چون مسابقه argmax ساده است، این را به‌صورت قابل‌خاموش‌کردن (flag) بگذار و پیش‌فرض **خاموش** (argmax خالص). مهم‌ترین کاربرد threshold برای تحلیل محلی است.
5. **Fusion با centroid (اختیاری):** اگر embedding cache و centroidهای گام ۶ موجود است، خروجی مدل را با خروجی centroid ترکیب کن (flag-controlled).
6. **Fallback امن:** اگر فایل decode نشد یا مدل لود نشد، یک توزیع **یکنواخت ۱/۴۴۷** برای آن فایل خروجی بده تا فرمت CSV همیشه معتبر بماند.
7. **فرمت خروجی CSV:** ستون اول `id` (نام فایل بدون پسوند، یا همان نام فایل — بسته به spec مسابقه که هنوز TBD است)، سپس ستون‌های `0,1,...,446` که مقادیر احتمال هستند و **مجموع هر سطر = ۱٫۰**.
   - ⚠️ ترتیب ستون‌ها باید با `class_map` سازگار باشد: ستون `0` = unknown، ستون `1..446` = known UUIDها به ترتیب مرتب‌شده‌ی lexicographic (همان قرارداد `create_class_mapping`). چون spec دقیق submission هنوز TBD است، ترتیب و نام‌گذاری را قابل‌تنظیم نگه دار و در کامنت توضیح بده.

**تست smoke:**
```bash
uv run --no-sync python -c "
# ساخت چند فایل wav تستی و اجرای inference رویشان، سپس چک فرمت CSV
import pandas as pd, numpy as np
# ... (اجرای inference روی چند فایل نمونه) ...
# df = pd.read_csv('predictions.csv'); assert df.shape[1]==448; assert np.allclose(df.iloc[:,1:].sum(1),1,atol=1e-3)
print('INFERENCE SMOKE OK ✅')
"
```

**کامیت ۸:** `feat(submission): rebuild inference CLI with multi-window TTA + persisted OOD threshold + optional centroid fusion + safe CSV output`

---

### گام ۹ — انسمبل و کالیبراسیون (فقط در صورت نیاز برای رسیدن به ۰/۹۷)

**چرا:** اگر بعد از گام‌های ۱-۸ هنوز به ۰/۹۷ نرسیدیم، انسمبل و کالیبراسیون آخرین اهرم است (هر دو در مسابقه مجازند).

**تغییرات:**
1. **`src/ensemble.py` (`EnsembleModel`)** از قبل موجود است (average fusion و LearnedFusion). آن را به مسیر inference متصل کن:
   - چند مدل را با **seedهای متفاوت** یا **encoderهای متفاوت** (ECAPA + WavLM-Large + HuBERT) آموزش بده (هر کدام یک checkpoint).
   - در inference، احتمالات همه‌ی مدل‌ها را میانگین بگیر (average fusion) یا LearnedFusion را روی val آموزش بده.
2. **کالیبراسیون دما (temperature scaling):** روی val، دمای `T` را برای softmax speaker طوری انتخاب کن که Macro-F1 (یا NLL) بهینه شود. `fused_probs_from_logits` در `src/metrics.py` پارامتر `temperature` دارد — از آن استفاده کن و `T` بهینه را ذخیره و در inference اعمال کن.
3. گزارش: Macro-F1 هر مدل به‌تنهایی و انسمبل، + دمای بهینه.

> 📌 این گام **فقط در صورت نیاز** و بعد از دیدن نتایج گام‌های قبلی اجرا می‌شود. اجرای آموزش چند مدل سنگین است و با کاربر هماهنگ می‌شود.

**کامیت ۹:** `feat(ensemble): multi-model average/learned fusion + temperature calibration for macro-F1`

---

### گام ۱۰ — به‌روزرسانی README + گزارش نهایی

1. **`README.md`:** معماری نهایی، نحوه‌ی اجرا، نتایج (Macro-F1 محلی)، تصمیمات کلیدی (centroid fusion، fine-tune، انسمبل)، و نحوه‌ی تولید submission را مستند کن.
2. **گزارش نهایی:** خلاصه‌ی مسیر، اعداد قبل/بعد، و دستورالعمل بازتولید (reproducibility) — چون تیم‌های برتر باید جزئیات را ارائه دهند.
3. **نمونه‌ی submission:** یک `predictions.csv` نمونه (روی val) با فرمت صحیح.

**کامیت ۱۰:** `docs: final architecture, results, reproduction guide + sample submission`

---

## بخش ۶ — دستورالعمل دقیق برای مدل DeepSeek <a name="بخش-۶"></a>

> این بخش را به‌همراه کل سند به DeepSeek بده. این دستورالعمل، نحوه‌ی استفاده از سند را مشخص می‌کند.

### نقش تو
تو یک مهندس یادگیری ماشین ارشد هستی که باید گام‌های بخش ۵ را **به ترتیب** و با دقت پیاده‌سازی کنی. هر گام را کامل کن، تستش را اجرا کن، سپس کامیت بزن و سراغ گام بعدی برو.

### قوانین سخت
1. **اول بخش ۰ (تعمیر محیط) را کامل کن** و با دستور تأیید، `cuda True` را ببین. بدون این، هیچ‌کدام از تست‌ها کار نمی‌کند.
2. **همیشه با `uv run --no-sync python ...` اجرا کن** (پرچم `--no-sync` حیاتی است تا uv محیط را دوباره نشکند).
3. **کد را مطابق سبک موجود پروژه بنویس:** بانرهای چاپی با `───` و ایموجی، docstringهای سه‌خطی `"""..."""`، type hints، و نام‌گذاری snake_case. به کامنت‌های موجود در `src/train.py` و `src/data_pipeline.py` نگاه کن و همان لحن را حفظ کن.
4. **از نوشتن کد تکراری خودداری کن:** قبل از ساخت تابع جدید، بررسی کن آیا معادل آن در `src/data_pipeline.py`، `src/train.py`، `src/metrics.py`، `src/ood_detector.py`، یا `src/ensemble.py` وجود دارد (مثلاً منطق sampler متعادل و embedding و FAISS OOD از قبل هست).
5. **به فایل:خط ارجاع بده** و فقط همان بخش را تغییر بده؛ بازنویسی کل فایل‌ها را به حداقل برسان.
6. **بعد از هر گام `git commit` بزن** (پیام دقیقاً مثل آنچه در انتهای هر گام نوشته شده). **هرگز `git push` نکن** مگر کاربر صراحتاً بگوید.
7. **تست هر گام را اجرا کن و خروجی را گزارش بده.** اگر تستی fail شد، آن را **پنهان نکن** — خطای کامل را بنویس، علت را تحلیل کن، و اصلاحش کن.
8. **گام‌های محاسباتی (۵، ۶، ۷):** فقط کد را بنویس و تستِ سبکِ واحد را اجرا کن؛ **اجرای کاملِ سنگین را به کاربر بسپار** و دستور دقیق اجرا را به فارسی به او بده. (مثلاً آموزش ۸-۱۲ ساعته‌ی گام ۷ را خودت اجرا نکن.)

### ترتیب و وابستگی گام‌ها
- گام ۱ (metrics) → پیش‌نیاز همه. (فایلش موجود است؛ فقط تأیید کن.)
- گام ۲ (split پاک) → پیش‌نیاز ۳، ۵، ۶.
- گام ۳ (multi-window) → پیش‌نیاز ۴، ۶، ۸.
- گام ۴ (sampler/loss/eval) → **مهم‌ترین گام**؛ پیش‌نیاز ۷.
- گام ۵ (EDA unbiased) → مستقل، ولی به ۲ وابسته است.
- گام ۶ (centroid baseline) → به ۲ و ۳ وابسته؛ برای fusion در ۸ لازم است.
- گام ۷ (fine-tune) → به ۴ وابسته؛ سنگین، با کاربر.
- گام ۸ (submission) → به ۳، ۴، (۶) وابسته.
- گام ۹ (انسمبل) → فقط در صورت نیاز، بعد از ۷ و ۸.
- گام ۱۰ (مستندات) → آخر.

### آنچه باید به کاربر گزارش بدهی
- در پایان هر گام: چه چیزی تغییر کرد، نتیجه‌ی تست، و (اگر گام محاسباتی بود) دستور اجرای سنگین.
- اگر جایی به **تصمیمی** رسیدی که در سند نیامده (مثلاً مقدار دقیق یک hyperparameter که باید tune شود)، آن را با کاربر مطرح کن و حدس نزن.
- همه‌ی گفت‌وگو با کاربر به **فارسی**.

---

## بخش ۷ — چک‌لیست تأیید (Acceptance Criteria) <a name="بخش-۷"></a>

قبل از اینکه بگویی «تمام شد»، همه‌ی موارد زیر باید ✅ باشند:

**محیط:**
- [ ] `import torch` بدون خطا؛ `torch.cuda.is_available()` == `True`؛ نسخه = `2.11.0+cu126`.

**گام ۱ (metrics):**
- [ ] `src/metrics.py` موجود و تست آن `METRICS OK ✅` می‌دهد.

**گام ۲ (split):**
- [ ] هیچ `audio_file` هم‌زمان در train و val نیست (no leakage).
- [ ] فایل‌های خراب (۷۰ مورد) حذف شده‌اند.
- [ ] `data/processed/split_report.json` ساخته شده و گروه‌های تکراری/متناقض را گزارش می‌کند.

**گام ۳ (multi-window):**
- [ ] `SpeakerDataset` در eval چند پنجره‌ی `(W,1,128000)` برمی‌گرداند.
- [ ] مدل روی پنجره‌ها میانگین می‌گیرد (TTA).

**گام ۴ (sampler/loss/eval):**
- [ ] تست sampler نشان می‌دهد سهم unknown در batch ≈ ۰٫۵ است (نه ۰٫۰۰۲).
- [ ] در val، forward **بدون labels** (بدون margin) صدا زده می‌شود.
- [ ] checkpoint selection بر اساس `macro_f1` است (نه loss).
- [ ] `ood_threshold` در metrics و checkpoint ذخیره می‌شود.
- [ ] PitchShift به ±۱ کاهش یافته.

**گام ۵ (EDA):**
- [ ] `phase3_embedding_summary.json` شامل اعداد **out-of-sample** (split-half یا LOO) است.
- [ ] شبیه‌سازی Macro-F1 گزارش شده.

**گام ۶ (centroid):**
- [ ] `src/centroid_baseline.py` embedding cache می‌سازد و Macro-F1 centroid را گزارش می‌کند.
- [ ] threshold برای Macro-F1 بهینه شده.

**گام ۷ (fine-tune):**
- [ ] `unfreeze_last_n_blocks` کار می‌کند و فقط n بلوک آخر trainable است.
- [ ] optimizer دو param group با LR متفاوت دارد.

**گام ۸ (submission):**
- [ ] `submission/inference.py` با click کار می‌کند.
- [ ] CSV خروجی ۴۴۸ ستون (`id` + `0..446`) دارد و مجموع هر سطر = ۱.
- [ ] TTA چندپنجره‌ای و fallback یکنواخت پیاده شده.

**گام ۹ و ۱۰ (در صورت اجرا):**
- [ ] انسمبل/کالیبراسیون مستند و Macro-F1 گزارش شده.
- [ ] README و گزارش نهایی به‌روز شده.

---

## پیوست A — فرهنگ نام‌گذاری و قراردادها <a name="پیوست-a"></a>

### قرارداد فضای برچسب (بسیار مهم — منبع اصلی باگ‌های off-by-one)
| فضا | بازه | معنا |
|---|---|---|
| **Global / dataset** | `0` | unknown |
| | `1..446` | known speaker (مرتب lexicographic بر اساس UUID) |
| **Speaker head (ArcFace)** | `0..445` | known speaker؛ اندیس `j` ↔ global `j+1` |

- نگاشت global→head: `head = global - 1` (فقط برای known). نگاشت head→global: `global = head + 1`.
- در ArcFace هنگام train، unknown (global 0) به کلاس ۰ head نگاشت می‌شود که با speaker #1 برخورد دارد — **بی‌ضرر است** چون loss با `ignore_index=-100` آن را mask می‌کند (`src/model.py:86-100`).

### نام‌گذاری
- متغیرها/توابع: `snake_case` (مثل `make_balanced_batch_sampler`).
- کلاس‌ها: `PascalCase` (مثل `FAISSOODDetector`).
- ثابت‌ها: `UPPER_SNAKE`.
- کلیدهای config: `snake_case` در YAML.

### قرارداد چاپ (سبک موجود پروژه)
- بانر بخش: `print("=" * 55)` + عنوان.
- مرحله: ایموجی + متن، مثل `print(f"  ✓ Train: {n} samples")`.
- هشدار: `⚠`، موفقیت: `✓`/`✅`، مدل: `📊`، OOD: `🎯`.

### قرارداد فایل‌های خروجی
| فایل | محتوا |
|---|---|
| `data/processed/split_report.json` | گزارش split، خراب‌ها، تکراری‌ها |
| `data/processed/embeddings_{train,val}.npy` | cache embedding برای گام ۶ |
| `checkpoints/best_model.pt` | بهترین مدل (بر اساس macro_f1) + `ood_threshold` |
| `checkpoints/corrupted_files.json` | لیست فایل‌های خراب |
| `predictions.csv` | خروجی submission (`id, 0..446`) |

### محدودیت‌های سخت‌افزاری (GTX 1660 Ti — ۶GB VRAM)
- `batch_size` کوچک (۸ محلی / ۱۶-۳۲ سرور) + AMP (`mixed_precision: true`).
- در fine-tune (گام ۷) احتمال OOM هست ⇒ batch کوچک‌تر یا پنجره‌های کمتر.
- برای CPU-bound بودن dataloader، `num_workers=0` روی ویندوز محلی.

---

> ✅ **پایان سند.** این سند self-contained است؛ با دنبال‌کردن بخش ۵ به ترتیب و رعایت قوانین بخش ۶، راه‌حل به‌سمت Macro-F1 ≥ ۰/۹۷ پیش می‌رود.
