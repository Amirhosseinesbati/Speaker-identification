# 📋 Leaderboard Submission Log

> یک سطر برای هر submission — منبع حقیقت برای «آزمایش → تصمیم → submission».
> قبل از هر آپلود `scripts/verify_submission.py` را اجرا کنید و سطر را همین‌جا ثبت کنید.

## قواعد

- هر submission = **یک سطر** در جدول زیر.
- `config` = فایل کانفیگ نام‌دار / commit hash که run از آن ساخته شده.
- `zip contents` = کدام checkpoints + centroids + decision params داخل بسته است.
- `score` = عدد Leaderboard (Macro-F1). اگر failed/timeout/crash، همان را بنویسید.
- `[diag]` = خروجی خط `[diag] cuda_avail=...` از لاگ سرور (تشخیص علت timeout).

## Log

| تاریخ | commit/config | zip contents | score | نتیجه / یادداشت | [diag] cuda_avail |
|---|---|---|---|---|---|
| 2026-08-14 | (baseline قبل از Phase 1) | 3 ckpt (campp/eres2net/titanet), argmax خام، بدون centroid/gate | — | ≥۶ تلاش قبلی همگی crash/timeout — score ثبت نشده | ? |
| 2026-08-15 | تکمدله campp — `feature/single-model-submission` | campp_best.pt + centroids_campp.npz + decision (α=0.2, κ=24, τ=0, λ=1.05) | **f1_macro 0.9505** (acc 0.9451) | ✅ اولین submission موفق — بدون crash/timeout | ? |

## TODO برای خواندن لاگ آخرین run

- [ ] دانلود لاگ آخرین submission و خواندن `[diag] cuda_avail=...` (آیا CUDA در venv لیدربرد لود شده؟).
- [ ] مقدار دقیق Timeout سرور را از پنل مسابقه تأیید کنید (PDF خالی بود).
