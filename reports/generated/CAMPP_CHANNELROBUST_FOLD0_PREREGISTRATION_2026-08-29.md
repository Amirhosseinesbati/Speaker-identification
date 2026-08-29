# CAM++ channel/session-robust complement Fold 0 preregistration — 2026-08-29

## Motivation

The externally validated CAM++ LME20 package scored `0.9667174285` on the
leaderboard versus `0.9633564052` on locked three-Fold OOF.  Post-processing
audits, LMFT, ERes2Net, and selected Raw/EMA snapshot fusion have not produced
a safe improvement.  The residual EDA points to known files rejected under
session/channel/quality shift, while the Raw/EMA audits show that highly
correlated snapshots do not provide usable representation diversity.

Recent speaker-recognition work and challenge systems consistently motivate
multi-condition channel augmentation, duration-aware adaptation, and fusion of
genuinely different systems.  This experiment tests the smallest trainable
step in that direction while holding the architecture and scientific protocol
fixed.

## Single treatment

Relative to Control Fold 0, only the augmentation policy changes.  The
candidate keeps the same CAM++ initialisation, seed `42`, split
`kfold/folds3/fold0/seed42`, known-first 446-way ArcFace plus binary OOD head,
loss weights, optimiser, schedule, freeze/unfreeze plan, eight 8-second train
windows, and exact evaluation path.

The candidate replaces broad voice-changing augmentation with a
speaker-preserving channel/session policy:

- MUSAN noise `0.60`, music `0.10`, SNR `3..20 dB`;
- RIR convolution `0.60`;
- codec probability unchanged at `0.30`;
- pitch shift disabled;
- time stretch reduced to `0.10` and constrained to `0.95..1.05`;
- Gaussian noise and time masking reduced;
- gain, shift, and polarity controls otherwise unchanged.

MUSAN and RIR assets are already present on the worker (934 noise files, 672
music files, and 20,000 RIR files).  No augmentation data will be downloaded
during the run.

## Locked evaluation and gate

Only Fold 0 is authorised.  The selected candidate will be evaluated under the
same fixed LME20 policy as Control, both standalone and in one fixed `50/50`
probability-evidence fusion.  No weight, threshold, epoch, beta, or prototype
rule is selected from the Fold or leaderboard.

The candidate passes only if all conditions hold:

1. standalone LME20 is no more than `0.010` below Control Fold 0 LME20
   (`0.9611456663`);
2. fixed fusion improves Macro-F1 by at least `+0.002`;
3. Known Accuracy and OOD-F1 each fall by at most `0.001`;
4. the candidate independently rescues at least 20% of Control errors;
5. provenance, split, class map, OOF file order, and artifacts validate.

At epoch 40, a best Raw probability-average Macro-F1 below `0.90` stops the
run.  NaN, provenance mismatch, artifact corruption, or a second OOM also stop
it.  Runtime is capped at six hours and incremental cost at `$1.10`.  A pass
authorises analysis and a separate three-Fold preregistration only.  It does
not authorise a submission or automatic Fold expansion.

