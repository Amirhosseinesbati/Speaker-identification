# CAM++ paired consistency — long-horizon Fold-0 amendment

Date: 2026-08-30
Status: **preregistered before any paired-consistency training evidence**

## Reason for the amendment

The original matched experiment used a fixed 40-epoch horizon.  The user
correctly noted that the paired clean/aug objective creates a harder adaptation
problem and may need more optimisation time.  At the time of this amendment,
only two epochs of the no-consistency engineering run had completed and the
paired-consistency treatment had never started.  No treatment metric or
leaderboard result was available or used to select the new horizon.

Simply changing only the treatment from 40 to 80 epochs would confound the
auxiliary loss with a different cosine schedule and twice the optimisation
budget.  Therefore the completed 40-epoch no-consistency run is classified as
an engineering/runtime pilot and is excluded from the acceptance gate.

## Locked long-horizon design

The decision experiment is a new sequential 80/80 matched pair:

1. `p4-campp-known446-ood-channelrobust-paired-control-long80-oof-f0`
2. `p4-campp-known446-ood-channelrobust-consistency-c01-long80-oof-f0`

The raw profile files are locked before either long80 branch starts:

- matched control SHA256:
  `29fad79221ef180cdd7eb35102dc75cd488e8505f4de0d8e44eb20d3cd144562`;
- consistency treatment SHA256:
  `1d0625d1c4311dbe0544775cfeb8db91c2f07d0335eba748a7c87240ef4ba860`.

Both start from the same selected Channel-Robust Raw checkpoint and reset
optimizer, scheduler and EMA.  They share seed, Fold-0 split, model,
augmentation, windows/crops, optimizer, 80-epoch cosine schedule and direct
Raw probability-average inference.  The only scientific treatment difference
is `training.loss.consistency.enabled=false` versus `true`; its type and fixed
weight remain cosine/0.1 in both configs.  Metric early stopping is disabled.
Both branches explicitly set `training.selection_variant=raw`; EMA remains a
fully logged diagnostic candidate but cannot silently replace the canonical
checkpoint or its OOF bundle.

Each branch preserves an exact Raw epoch-40 milestone from inside the same
80-epoch schedule.  The milestone is diagnostic learning-curve evidence; the
primary A/B decision compares the selected Raw models over the complete
80-epoch horizons.  This avoids comparing a 40-epoch cosine schedule to the
halfway point of an 80-epoch cosine schedule.

The control is capped at 5 hours / `$0.90`; the paired treatment at 9.5 hours /
`$1.65`.  The campaign-wide `$20` cap and 12-hour per-run ceiling remain
binding.  Immediate stop rules remain limited to NaN, provenance mismatch,
artifact corruption or a second OOM.  A weak intermediate metric is not a stop
rule.

## Locked gate

The original six-part gate is unchanged at the 80-epoch horizon:

1. selected Raw treatment LME20 Macro-F1 is at least `+0.002` above the
   selected Raw long80 matched control;
2. fixed 50/50 probability-evidence fusion with externally validated Control
   Fold 0 is at least `+0.002` above external Control LME20;
3. Known Accuracy and OOD-F1 each fall by no more than `0.001` relative to
   external Control LME20;
4. treatment rescues at least `20%` of external Control LME20 errors;
5. no non-finite or representation-collapse signal occurs;
6. split, class map, warm-start SHA, history, selected checkpoint, OOF and
   MLflow receipt all validate.

The original logged-standard-deviation ratio is mechanically undefined for
the no-consistency branch because that branch intentionally does not request a
paired clean view and therefore logs zero for paired-view spread.  Before any
treatment evidence, the collapse measurement is clarified as the mean
coordinate standard deviation of deterministically extracted Fold-0 training
embeddings from the selected Raw checkpoints.  The treatment/control ratio
must be at least `0.95`, and both values must be finite and positive.  This
uses the same files, extraction policy and statistic for both branches.

Passing Fold 0 still permits only a separate cross-fold preregistration.  It
does not authorize threshold tuning, a tuned blend or a submission.

## Literature cross-check during the engineering pilot

This amendment is a testable hypothesis, not a claim that consistency must
help this competition.  Three primary sources support testing invariance while
also warning against inferring a universal recipe:

- Huh et al., *Augmentation adversarial training for self-supervised speaker
  recognition* (2020), explicitly target speaker-discriminative embeddings
  that are invariant to simulated acoustic/channel augmentation and report
  gains on VoxCeleb and VOiCES:
  <https://arxiv.org/abs/2007.12085>.
- Sang et al., *Self-Supervised Speaker Verification with Simple Siamese
  Network and Self-Supervised Regularization* (ICASSP 2022), apply a
  positive-pair latent regularizer with online time/frequency augmentation and
  report a 23.4% relative improvement in their self-supervised VoxCeleb setup:
  <https://arxiv.org/abs/2112.04459>.
- Sadhu and Wang, *Improving Audio Event Recognition with Consistency
  Regularization* (2025), train paired-view supervised audio models for 60
  epochs and show that useful consistency coefficient and augmentation count
  depend on dataset scale; their ablations therefore argue for an explicitly
  matched control rather than assuming that a stronger or longer auxiliary
  objective is automatically better:
  <https://arxiv.org/html/2509.10391v1>.

The evidence supports giving the harder paired objective a real optimisation
horizon.  It does **not** justify changing only the treatment duration or
tuning its weight on Fold 0.  The locked 80/80 design therefore doubles the
original horizon for both branches, retains a single fixed coefficient, and
uses the full learning curves plus the collapse/identity guardrails above.  If
the treatment remains positively sloped at epoch 80, that is evidence for a
separately preregistered matched extension; it is not permission to continue
only the treatment after seeing its held-out metric.

## Predeclared under-training diagnostic

Before either long80 branch starts, the phrase "positively sloped" is made
mechanical.  The terminal audit compares epochs 61--70 with epochs 71--80 and
fits a linear slope over epochs 61--80.  A separate matched extension may be
preregistered only when all of the following hold: the treatment's final
10-epoch mean improves by at least `0.0005`; its 20-epoch slope is positive;
its best Raw epoch lies in 71--80; its improvement relative to the matched
control accelerates by at least `0.0005`; the final-window Known Accuracy and
OOD-F1 each fall by no more than `0.002`; and the deterministic embedding
spread ratio remains at least `0.95`.  This diagnostic does not change the
acceptance gate and never authorises a treatment-only continuation.  It only
prevents a genuinely still-learning, healthy treatment from being rejected
merely because the preregistered horizon ended.
