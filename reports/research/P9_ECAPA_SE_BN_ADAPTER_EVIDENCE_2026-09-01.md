# P9 ECAPA SE/BN adapter: evidence and conditional preregistration

Date: 2026-09-01

Status: **conditional only**. This document does not authorize a run while
`p8-ecapa-frozen-known446-ood-complement-oof-f0` is active. P9 is eligible only
if P8 terminates below its preregistered standalone gate
`0.9269211906147802`, with a selected Raw Macro-F1 of at least `0.90`.

## Scientific question

P8 tests a fully frozen ECAPA representation with a trained competition head.
If it learns a useful but sub-gate representation, the next single-variable
hypothesis is that target-domain mismatch lies partly in channel weighting and
activation statistics. P9 therefore adapts only ECAPA squeeze/excitation (SE)
parameters and batch-normalization (BN) affine/statistical state while keeping
the embedding core and the already-trained competition heads fixed.

This is not generic full-model fine-tuning and is not a hyperparameter sweep.

## Primary evidence

The Interspeech 2024 paper *SE/BN Adapter: Parametric Efficient Domain
Adaptation for Speaker Recognition* freezes the core speaker encoder and adapts
SE blocks plus BN layers:

- ISCA paper: https://www.isca-archive.org/interspeech_2024/wang24ma_interspeech.pdf
- arXiv record: https://arxiv.org/abs/2406.07832
- referenced Sunine repository: https://gitlab.com/csltstu/sunine

The paper provides four points that transfer to this campaign:

1. SE reweights channel maps and is explicitly treated as domain-specific.
2. BN adjusts activation shift/scale; SE and BN were complementary.
3. Adapting SE/BN in all groups was stronger than adapting one group.
4. The limited-data experiment replaced a newly initialized Softmax head with
   GE2E because the random head could absorb the error signal.

Point 4 does **not** directly require GE2E here. P9 warm-starts the complete P8
model, including a trained 446-way ArcFace head and binary OOD head. Freezing
those trained heads preserves the decision geometry while still allowing
gradients to propagate through them into the adapter. This distinction is
recorded to avoid falsely claiming an exact reproduction of the paper.

The paper does not publish an adapter-specific learning rate or epoch schedule,
and the current public Sunine checkout does not contain a clearly identifiable
SE/BN-adapter recipe. Consequently, the optimization values below are campaign
preregistration choices, not values attributed to the authors.

## Implemented adapter invariant

Commit `e125ce6` implements the structural hypothesis against the pinned
SpeechBrain ECAPA revision used by P8:

- every non-adapter ECAPA parameter is frozen;
- parameters inside every native SpeechBrain `SEBlock` are trainable;
- affine weight/bias in every PyTorch BatchNorm layer are trainable;
- during training only SE and BN modules enter train mode, so BN running
  statistics adapt to the target domain;
- during validation/inference the complete encoder is in eval mode, preventing
  validation-statistic mutation;
- missing SE or BN modules are a hard structural failure.

Measured on the actual pinned ECAPA embedding model, the adapter contains
`828,288` trainable encoder parameters (`3.9883757%` of that embedding model),
distributed across 3 SE blocks and 31 BN modules. The paper's reported `1%`
belongs to its 8M-parameter ResNet34SE and must not be copied as an ECAPA claim.

## Locked P9 design (except terminal P8 receipt fields)

- profile: `p9-ecapa-sebn-known446-ood-adapter-oof-f0`
- split: `kfold/folds3/fold0/seed42`
- encoder source/revision, audio, augmentation, 8-window evaluation, known-first
  446-way ArcFace head, binary OOD head, losses and Raw probability-average
  argmax: identical to P8
- warm start: P8 selected `ecapa_best_raw.pt`; full model weights restored;
  optimizer, scheduler and EMA reset
- trainable encoder subset: all SE + BN adapter parameters only
- competition heads: loaded from P8 and fixed (`head LR = 0`); an invariant test
  must prove their tensors do not change after an optimizer step
- adapter LR: `1e-5`
- optimizer/schedule: existing AdamW, cosine schedule, `warmup_ratio=0.05`,
  `min_lr_ratio=0.05`, weight decay `1e-4`
- maximum horizon: 45 epochs
- early stopping: start at epoch 5, patience 12 on Raw probability-average
  Macro-F1
- immediate stop: NaN, provenance/split/class-map mismatch, artifact
  corruption, or second OOM
- futility stop: by epoch 12, selected best below terminal P8 Raw by more than
  `0.003`
- budget: at most 2.5 hours and USD 0.50 incremental
- no threshold, blend, epoch, LR, adapter-location or loss sweep

The terminal P8 selected epoch, metrics, checkpoint path/SHA/size and MLflow
receipt must be inserted into the P9 preregistration before any worker launch.

## Decision gates

P9 standalone passes only if all are true:

1. selected Raw Macro-F1 is at least
   `max(0.9269211906147802, terminal_P8_raw + 0.002)`;
2. Known Accuracy and OOD-F1 each drop by no more than `0.001` from the terminal
   P8 selected checkpoint;
3. the Raw improvement of at least `0.002` is supported by two completed
   epochs, not a single spike;
4. artifact, split, class-map, history and MLflow provenance are complete.

Only a standalone pass authorizes generating P9 Fold0 OOF and applying the
already-locked, exactly 50/50 probability-average evaluator against CAM++
Fold0. The fusion gate remains:

- Macro-F1 gain over locked CAM++ Fold0 at least `+0.002`;
- Known Accuracy and OOD-F1 drops each at most `0.001`;
- at least 25% of CAM++ errors rescued;
- rescued errors strictly greater than introduced errors.

A Fold0 pass authorizes a separate Fold1/Fold2 preregistration; it never
authorizes a submission by itself.

## Falsification interpretation

If P9 fails, the evidence will reject this particular domain-statistics
adaptation mechanism for the pinned ECAPA representation and competition loss.
It will not be re-run with an LR sweep. The next family must add genuinely new
information (for example a different frontend or condition-robust objective),
rather than another small optimizer variation.
