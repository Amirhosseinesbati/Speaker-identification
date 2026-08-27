# طرح عملیاتی Campaign ایجنتیک روی Vast.ai

تاریخ: 2026-08-27  
هدف: اجرای کنترل‌شده‌ی آزمایش‌های Speaker Identification تا رسیدن به submission پایدار بالاتر از 0.972، همراه با Telegram، resume، budget guard و توقف برای نتیجه‌ی leaderboard.

## 1. تصمیم

استفاده از یک RTX 3090 روی Vast.ai تصمیم مناسبی است، به شرط آنکه آن را به یک «ماشین اجرای deterministic campaign» تبدیل کنیم، نه یک حلقه‌ی HPO بدون نظارت. orchestration باید شواهد را ثبت کند، هر آزمایش را با gate علمی بسنجد، artifactها را بیرون از Instance پایدار کند و فقط پس از تصمیم روشن به مرحله‌ی بعد برود.

هیچ launch یا هزینه‌ای قبل از تعیین سقف قیمت، سقف هزینه‌ی campaign و سیاست `WAITING_FOR_LEADERBOARD` انجام نمی‌شود.

## 2. وضعیت فعلی و blockerها

### 2.1 CLI میزبان

plugin رسمی Vast.ai موجود است، اما اجرای `vastai --version` روی میزبان فعلی با `command not found` شکست خورد. طبق قواعد plugin نباید CLI جایگزین نصب، binary با مسیر غیررسمی صدا زده یا REST API مستقیماً فراخوانی شود. ابتدا باید binary رسمی `vastai` در همان shell روی `PATH` قرار گیرد.

`pyproject.toml` dependency مربوط به Vast را دارد؛ محتمل است با فعال‌کردن environment پروژه binary روی PATH قرار گیرد، ولی تا زمانی که `vastai --version` موفق نشود هیچ account query یا launch انجام نمی‌شود.

### 2.2 flow فعلی deploy

- `setup_vast.sh` روی `EXIT/ERR` Instance را destroy می‌کند؛ این با `WAITING_FOR_LEADERBOARD` ناسازگار است.
- cleanup داخل Instance تلاش می‌کند Vast CLI را با pip نصب کند؛ این flow باید حذف و lifecycle فقط از control plane مدیریت شود.
- `src/deploy/deploy.py` هنگام create، `--ssh --direct --cancel-unavail` را ارسال نمی‌کند.
- پاسخ create باید از `new_contract` خوانده شود؛ کد فعلی `new_instance` یا offer id را استفاده می‌کند.
- secrets در command string و per-instance `--env` ساخته می‌شوند؛ باید از secret store/account env-vars یا environment امن process استفاده شود.
- SSH key قبل از create بررسی/ثبت نمی‌شود.
- polling چهار وضعیت `actual_status/intended_status/cur_state/next_state` و timeout ندارد.
- queue فقط چهار status ساده دارد و gate علمی/Telegram/budget/artifact sync ندارد.
- `setup_vast.sh` تمام frameworkها و تمام weights را دانلود می‌کند، حتی اگر campaign فقط CAM++ باشد.
- workspace فعلی حدود 27.6 GiB داده/processed/weights/checkpoints دارد؛ disk فعلی 60GB برای venv، DVC cache، artifactهای چند run و checkpointها کم‌حاشیه است. پیشنهاد اولیه 100GB است.
- تغییرات محلی هنوز روی branch remote موجود نیستند؛ clone روی Instance آن‌ها را نخواهد دید.

## 3. معماری پیشنهادی

```text
Codex thread + Vast CLI on local host
               │
               ├─ create/status/SSH/copy/stop/destroy
               ├─ periodic heartbeat monitor
               └─ research decisions + new experiment profiles

RTX 3090 Vast instance
  /workspace/project
    ├─ campaign supervisor
    ├─ deterministic experiment runner
    ├─ evaluator / gate engine
    ├─ Telegram notifier + restricted command inbox
    └─ local campaign state + logs + artifacts
               │
               ├─ DagsHub MLflow: metrics/config/logs/model bundles
               ├─ Git: source/profile revisions
               └─ DVC/object storage: immutable data and selected large artifacts
```

Codex control plane تصمیم علمی می‌گیرد. Instance فقط plan نسخه‌بندی‌شده را اجرا می‌کند. Bot اجازه‌ی اجرای shell دلخواه ندارد؛ فقط commandهای محدود state machine را می‌پذیرد.

## 4. state machine

```text
CREATED
  → BOOTSTRAPPING
  → PREFLIGHT
  → READY
  → RUNNING_EXPERIMENT
  → ANALYZING
       ├─ GATE_FAILED → READY یا CAMPAIGN_BLOCKED
       ├─ NEXT_EXPERIMENT → RUNNING_EXPERIMENT
       └─ PACKAGE_READY → WAITING_FOR_LEADERBOARD

WAITING_FOR_LEADERBOARD
  ├─ /leaderboard <score> → ANALYZING
  ├─ /pause → PAUSED
  ├─ TTL exceeded → STOPPED_FOR_BUDGET
  └─ host failure → RECOVERY_REQUIRED

terminal:
  CAMPAIGN_COMPLETE | BUDGET_EXHAUSTED | FAILED | DESTROYED
```

state در یک JSON atomic و append-only event log نگهداری می‌شود. هر transition شامل timestamp، commit SHA، config hash، data manifest hash، run id، هزینه‌ی تخمینی و دلیل تصمیم است.

## 5. Telegram

### 5.1 eventهای خروجی

- Instance created/running و مشخصات GPU/host/price.
- bootstrap و preflight pass/fail.
- شروع/پایان هر run و مدت/هزینه.
- best metric فقط هنگام improvement معنادار؛ نه spam هر epoch.
- gate passed/failed با مقایسه‌ی control.
- artifact sync success/failure.
- package ready با SHA256/size/runtime.
- ورود به `WAITING_FOR_LEADERBOARD`.
- budget warning و timeout.
- exception با آخرین log lines و مسیر artifact.

### 5.2 commandهای ورودی محدود

- `/status`
- `/leaderboard 0.9662 optional-note`
- `/pause`
- `/resume`
- `/stop_after_run`
- `/budget`

فقط `TELEGRAM_CHAT_ID` مجاز پذیرفته می‌شود. هیچ command عمومی، Python expression یا shell execution از Telegram پشتیبانی نمی‌شود.

### 5.3 secrets

- token از BotFather گرفته و مانند password نگهداری شود.
- token هرگز در Git، report، log یا chat این task نوشته نشود.
- `TELEGRAM_BOT_TOKEN` و در صورت تمایل `TELEGRAM_CHAT_ID` در Vast account env-var یا secret mechanism قرار گیرند.
- notifier خطای Telegram را log می‌کند ولی training را fail نمی‌کند.
- Bot ابتدا باید از طرف کاربر یک پیام دریافت کند؛ bot نمی‌تواند conversation خصوصی را خودش شروع کند.

راهنمای رسمی: `https://core.telegram.org/bots/tutorial` و `https://core.telegram.org/bots/api`.

## 6. سیاست هزینه و WAITING

درخواست روشن‌ماندن Instance قابل اجراست، اما unlimited keep-alive خطر هزینه‌ی بدون سقف دارد. policy پیشنهادی:

- `MAX_PRICE_USD_PER_HOUR`: سقف offer.
- `MAX_CAMPAIGN_COST_USD`: سقف کل campaign.
- `MAX_RUN_HOURS`: timeout هر training.
- `WAITING_KEEPALIVE_HOURS`: مدت روشن ماندن پس از package؛ پیشنهاد اولیه 6 ساعت.
- هشدار Telegram در 50%، 75%، 90% budget.
- پس از TTL: ابتدا artifact sync، سپس `vastai stop instance <id>`؛ disk حفظ و GPU billing قطع می‌شود.
- destroy فقط با تصمیم صریح یا پس از تأیید sync کامل، با syntax اجباری `vastai destroy instance <id> -y`.

اگر کاربر روشن‌ماندن تا مدت نامحدود را انتخاب کند، باید سقف هزینه‌ی دلاری همچنان hard stop باشد.

## 7. persistence و recovery

Vast shared/network volume ندارد. local disk تنها copy قابل اعتماد نیست. بعد از هر run:

1. metrics/config/history/logs به MLflow.
2. checkpoint منتخب، OOF bundle، manifest و analysis JSON به artifact store.
3. SHA256 تمام artifactهای promoted.
4. state/event log به storage خارجی.
5. یک Telegram receipt شامل run id و sync status.

Instance تنها وقتی destroy می‌شود که آخرین sync receipt موفق باشد. در host failure، campaign از آخرین run کامل resume می‌شود؛ run نیمه‌کاره به‌عنوان aborted ثبت و overwrite نمی‌شود.

## 8. campaign علمی اولیه

### Phase 0 — اصلاح measurement؛ بدون اجاره‌ی طولانی

- balanced BatchSampler واقعی.
- checkpoint selection با exact probability-average.
- artifact schema همراه filenames/SHA و alignment hard check.
- HPO metric/parser و nested config fix.
- centroid/split-report consistency.
- regression tests و replay کنترل تاریخی.

بدون pass این phase، GPU campaign آغاز نمی‌شود.

### Phase 1 — cached/inference experiments

- OOF exact-path برای checkpointهای موجود.
- robust/quality-weighted prototypes.
- cross-fitted calibration و Top-5 evidence.
- Random/Hard/Novel-background dashboard.

### Phase 2 — دو gate training روی 3090

1. known-first CAM++ control.
2. همان recipe با auxiliary confidence-weighted background/metric loss.

فقط یک متغیر علمی تغییر می‌کند. اگر gate از پیش تعریف‌شده پاس نشود، foldهای دیگر اجرا نمی‌شوند.

### Phase 3 — confirmation

- فقط family برنده روی foldهای باقی‌مانده.
- cross-fitted fusion/calibration.
- full-data fit با epoch از پیش تعیین‌شده.

### Phase 4 — diversity

ERes2NetV2 فقط اگر CAM++ هنوز representation-limited باشد و error complementarity آن gain قابل اندازه‌گیری بدهد. WavLM پس از آن و فقط با runtime budget.

## 9. انتخاب offer و launch ایمن

پس از آماده شدن CLI و budget:

- GPU: RTX 3090، 24GB، یک GPU.
- verified/rentable و direct SSH.
- reliability پیشنهادی حداقل 0.98.
- disk پیشنهادی 100GB.
- price cap از کاربر.
- image و lock کاملاً pinned؛ automatic floating environment برای campaign علمی مناسب نیست.
- قبل از create: account، balance، instanceهای موجود و SSH key بررسی شوند.
- create همیشه با `--disk 100 --ssh --direct --cancel-unavail`.
- create response از `new_contract` parse شود.
- provisioning حداکثر 10 دقیقه poll و چهار status field بررسی شود.
- `exited/unknown/offline` terminal است؛ logs خوانده، سپس Instance با `-y` destroy و offer دیگری انتخاب شود.

## 10. یک‌بار آماده‌سازی توسط کاربر

1. در PowerShell محیطی را فعال کند که binary رسمی `vastai` روی PATH قرار دهد و این دستور موفق شود:

   ```powershell
   vastai --version
   ```

2. API key را در terminal محلی تنظیم کند، نه در chat:

   ```powershell
   vastai set api-key <KEY>
   vastai show user --raw
   ```

3. اگر 2FA فعال است، login را خودش در terminal انجام دهد.
4. SSH public key را قبل از create ثبت کند؛ محتویات key باید ارسال شود، نه مسیر فایل.
5. از BotFather با `/newbot` bot بسازد، به bot پیام `/start` بدهد و token را فقط در secret store قرار دهد.
6. چهار مقدار را تعیین کند:

   - سقف قیمت ساعتی RTX 3090.
   - سقف کل هزینه‌ی campaign.
   - حداکثر ساعت روشن ماندن در `WAITING_FOR_LEADERBOARD`.
   - آیا پس از TTL Instance `stop` شود یا با sync کامل `destroy`.

## 11. ترتیب implementation

1. patchهای P0 measurement.
2. `campaign_state.py` و schema/event log.
3. `telegram_notifier.py` و allowlisted command polling.
4. `campaign_supervisor.py` با budget/gate/resume.
5. بازنویسی lifecycle deploy و حذف self-destroy trap از worker.
6. preflight و unit/integration tests با fake executor؛ بدون اجاره.
7. dry-run کامل locally.
8. read-only Vast account/offer preflight.
9. تأیید offer/cost توسط کاربر.
10. create، bootstrap، sync test و شروع Phase 1/2.

## 12. معیار آمادگی launch

- CLI رسمی روی PATH و auth سالم.
- SSH key ثبت‌شده.
- budget policy کامل.
- Telegram send/receive test موفق.
- state resume test موفق.
- artifact sync test موفق.
- patchهای P0 و test suite پاس.
- commit/branch remote شامل تمام code/profileهای campaign.
- هیچ secret در Git یا logs.
- یک dry-run تمام transitionها تا `WAITING_FOR_LEADERBOARD`.

تا قبل از این checklist، launch یک RTX 3090 زودهنگام است.
