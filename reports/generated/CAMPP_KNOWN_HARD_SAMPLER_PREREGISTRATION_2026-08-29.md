# CAM++ known-hard sampler — conditional preregistration

Date: 2026-08-29  
Status: **conditional draft; not authorised while the channel-robust Fold-0 run is active**

## Trigger and scientific question

This candidate may run only if
`p3-campp-known446-ood-channelrobust-oof-f0` fails its terminal LME20,
fixed-fusion, Known/OOD, rescue-rate, or provenance gate.  It asks one question:

> Does increasing exposure to low-quality **known-speaker training files** reduce
> the dominant Known-to-Unknown boundary errors without weakening OOD rejection?

The aggregated Control OOF error analysis found that Known-to-Unknown files were
shorter and lower-energy than correct Known files (separation AUC: duration
`0.82188`, RMS `0.81235`, effective-window count `0.77415`).  Known-to-wrong-
Known errors were even more concentrated in near-silent audio.  This directly
motivates a sampling intervention; it does not authorise a decision threshold.

## Locked control

- Profile family: CAM++ strict known-first `446-way ArcFace + binary OOD`.
- Target: Fold 0, `kfold/folds3/fold0/seed42`.
- Architecture, checkpoint initialisation, split, augmentation, 8-second crop,
  eight train/eval windows, loss, optimiser, scheduler, freeze/unfreeze,
  batch size, mixed precision, inference, and LME20 decision policy remain
  identical to `p0-campp-known446-ood-control-oof-f0`.
- Baseline Raw probability-average Macro-F1: `0.9469211906147802`.
- Baseline locked LME20 Macro-F1: `0.9611456662793696`.
- The exact `24 OOD + 24 known` composition of every batch remains unchanged.

## The single treatment

Only selection **within the known half of each training batch** changes.

1. Compute duration, full-file RMS, and active-frame fraction from Fold-0
   training audio only.  Validation audio must not contribute thresholds,
   weights, or normalization statistics.
2. Estimate the 25th percentile of each feature from the known training pool.
3. A known file is `hard` only if at least two of its three features are at or
   below their training-pool 25th percentiles.
4. Assign sampling weight `2.0` to hard known files and `1.0` to every other
   known file.  The OOD pool remains uniform.
5. Draw the known stream deterministically with replacement using
   `seed + epoch`; preserve the existing number of batches and OOD/known ratio.

No weight, percentile, feature set, threshold, crop duration, or decision
parameter may be selected from Fold 0 or from the leaderboard.

## Required implementation evidence before launch

- Unit test: exact per-batch OOD/known count is preserved.
- Unit test: fixed seed and epoch reproduce the same weighted stream.
- Unit test: increasing a known sample's weight increases its draw frequency in
  a deterministic large-sample check and never changes the OOD stream.
- Provenance artifact: training-only feature table SHA256, percentile values,
  hard-file count, per-class hard-file counts, and realised exposure ratio.
- Preflight: one real batch on the worker, no validation indices in the feature
  table, and no material VRAM regression.

## Fold-0 gate and stop rules

The candidate is accepted only if all terminal checks pass:

1. Candidate standalone LME20 is no more than `0.010` below Control LME20.
2. Fixed 50/50 probability-evidence fusion with Control gains at least `+0.002`
   Macro-F1 over Control LME20.
3. Known Accuracy and OOD-F1 each drop by at most `0.001`.
4. Candidate rescues at least `20%` of Control LME20 errors.
5. Split, class-map, train-only feature provenance, checkpoint, receipt, and
   MLflow artifacts all validate.

Immediate stop: NaN, split/provenance mismatch, feature leakage, artifact
corruption, or second OOM.  Futility stop: best Raw probability-average
Macro-F1 below `0.90` at the end of epoch 40.  Maximum runtime: six hours;
maximum incremental cost: `$1.10`.

Passing Fold 0 authorises analysis and a separate Fold-1/2 preregistration only.
It never authorises automatic leaderboard submission.  Failure rejects this
sampling rule; its weight or quantiles must not be tuned on Fold 0.

## Literature boundary

Short-duration challenge systems have reduced duration mismatch using matched
training chunks with online MUSAN/RIR augmentation, while channel-adversarial
work motivates invariant representations when trustworthy domain labels exist.
Here, direct channel labels do not exist and generic channel augmentation is
already under test.  The sampler is therefore the cheapest causal intervention
that follows the local error topology without inventing pseudo-domain labels.

- SJTU SdSV 2021 system: https://www.isca-archive.org/interspeech_2021/han21c_interspeech.pdf
- Channel-adversarial speaker recognition: https://arxiv.org/abs/1902.09074

