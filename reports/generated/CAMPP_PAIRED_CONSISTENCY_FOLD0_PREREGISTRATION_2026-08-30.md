# CAM++ paired clean/aug consistency — conditional Fold-0 preregistration

Date: 2026-08-30
Status: **conditional; do not run while Channel-Robust source/continuation is active or accepted**

## Why this exists

The active Channel-Robust treatment is being given its full preregistered
patience/time budget and, if mechanically eligible, a stateful continuation.
This document does not shorten or replace that treatment.  It prepares a
single next representation hypothesis only if the source plus continuation
fails the original LME20/fixed-fusion gate.

Strong waveform/channel augmentation can improve mismatch robustness while
also leaving augmentation-specific residuals.  The proposed treatment asks
CAM++ to preserve its speaker embedding for the *same sampled speech crop*
before and after the already locked channel/noise augmentation.  The primary
446-way ArcFace and binary OOD losses remain active, so the auxiliary term is
not allowed to become a self-supervised replacement for identity learning.

The fixed auxiliary loss is

`0.1 * mean(1 - cosine(z_aug, stopgrad(z_clean)))`.

The coefficient was fixed before Fold-0 evidence.  Since cosine loss lies in
`[0, 2]`, its weighted contribution is bounded by `0.2`, well below the recent
primary training loss near `1.5`; this limits the chance that invariance
overwhelms the supervised identity/OOD objective.

## Matched A/B design

Two sequential Fold-0 branches are required; neither may run in parallel with
another scientific training process.

1. `p4-campp-known446-ood-channelrobust-paired-control-oof-f0`: warm-start the
   selected Channel-Robust Raw checkpoint, reset optimiser/scheduler/EMA, use
   the existing strong augmentation, and train for exactly 40 epochs with no
   consistency loss.
2. `p4-campp-known446-ood-channelrobust-consistency-c01-oof-f0`: identical
   warm start, seed, split, model, augmentation, crops/windows, optimiser,
   schedule, LR, fixed horizon and inference; enable only the fixed cosine
   consistency term.

The reset is deliberate and matched across both branches; it prevents a loss
change from masquerading as a stateful resume.  `early_stopping_patience=0`
means metric early stopping is disabled.  Immediate stops are limited to NaN,
provenance mismatch, artifact corruption or a second OOM.  The no-consistency
branch is capped at 2.5 hours / `$0.45`; the paired branch at 4.5 hours /
`$0.80`.  The campaign-wide `$20` cap remains binding.

The paired data path samples the crop once, clones it before augmentation and
returns clean/augmented tensors with identical shape.  Mixup is rejected.  The
clean view is a stop-gradient target evaluated without dropout; the augmented
branch retains the original supervised forward and its training-time dropout.
The default non-paired data and training paths remain unchanged.

## Locked decision rule and gate

All decisions use Raw probability-average plus direct argmax.  Logit-average
and EMA remain diagnostic.  No threshold, blend weight, epoch, coefficient or
submission may be tuned from Fold 0 or the leaderboard.

The candidate is rejected unless all of the following hold:

1. paired consistency improves terminal Fold-0 LME20 Macro-F1 by at least
   `+0.002` over the matched no-consistency branch;
2. its fixed 50/50 probability-evidence fusion with externally validated
   Control Fold 0 improves Control LME20 by at least `+0.002`;
3. Known Accuracy and OOD-F1 each fall by no more than `0.001` relative to
   Control LME20;
4. it rescues at least `20%` of Control LME20 errors;
5. mean logged augmented embedding standard deviation is at least `95%` of
   the matched no-consistency branch, and no non-finite/collapse signal occurs;
6. split, class map, warm-start SHA, history, selected checkpoint, OOF and
   MLflow receipt all validate.

Passing Fold 0 permits only a separately preregistered cross-fold test.  It is
not permission to create a submission.

## Implementation verification

The implementation is default-off and fail-fast.  It logs the raw/weighted
consistency loss, pair cosine, and clean/augmented batch feature standard
deviation.  The focused training, config, warm-start and stateful-resume suite
passed `33/33` CPU tests before this preregistration was committed.

## Literature basis and limits

- Cai et al., *Within-sample Variability-invariant Loss for Robust Speaker
  Recognition under Noisy Environments*, explicitly pairs identical clean and
  corrupted speech and combines embedding invariance with supervised speaker
  classification: https://arxiv.org/abs/2002.00924
- *Clean Segment Guided Speaker Embedding* uses asymmetric clean guidance to
  reduce speaker-information loss from augmentation:
  https://arxiv.org/abs/2309.04265
- NAW-SV combines clean-teacher/noisy-student consistency with speaker metric
  learning and cautions that generic enhancement can distort speaker cues:
  https://www.isca-archive.org/interspeech_2024/lim24_interspeech.pdf
- Speaker-oriented VICReg work motivates explicitly monitoring feature spread
  rather than assuming an invariance objective cannot collapse:
  https://www.isca-archive.org/interspeech_2022/lepage22_interspeech.pdf

These papers support the hypothesis, not its success on this competition.
That is why the coefficient is fixed, the primary losses remain active, a
matched A/B is mandatory, and all downstream gates are held out from the
leaderboard.
