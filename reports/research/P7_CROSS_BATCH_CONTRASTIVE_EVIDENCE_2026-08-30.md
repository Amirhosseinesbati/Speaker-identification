# P7 Cross-Batch Contrastive Evidence Note — 2026-08-30

## Status

This is a research backlog note, not an active experiment and not a
preregistration. It was written while the P5 matched-sampler control was still
running, before any terminal P5 control or treatment result was available.
Nothing in this note authorizes a worker checkout change, a new run, a queue
size, a loss coefficient, or a leaderboard submission.

## What P5 does and does not test

P5 is a deliberately narrow positive-pair ablation. Every selected known
speaker contributes exactly two distinct files. The auxiliary objective is
`1 - cosine(anchor, detached_target)` and the heterogeneous OOD pool is not
treated as a same-speaker positive set. The matched control uses the identical
paired sampler with the auxiliary term disabled.

This is scientifically useful, but it is not the complete supervised
contrastive objective described in the speaker-recognition literature. It has
no contrastive denominator and no explicit cross-speaker negatives inside the
auxiliary term; inter-class separation is left to the ordinary ArcFace head.

## Primary literature evidence

### AAMSupCon

Li and Mak combine ArcFace with supervised contrastive learning over an
original-plus-augmentation batch. All same-label examples are positives and
other examples are negatives. Their reported configuration uses ArcFace
margin `0.2`, contrastive temperature `0.07`, and a training batch size of
`3072`. Their batch-size ablation for the SupCon component improves as the
batch grows from `128` to `512` to `1024`, supporting the claim that the number
of available negatives matters. The reported CN-Celeb result improves ECAPA
AAMSoftmax from `8.79%` EER to `8.49%` EER with AAMSupCon.

Primary source: <https://arxiv.org/pdf/2210.16636>

### Prototypical momentum contrastive speaker learning

Xia et al. use a momentum encoder and a queue of `10000` negatives with
momentum `0.999`. Their system is trained on more than one million VoxCeleb2
utterances with batch size `4096` for `150` epochs. A prototypical memory bank
is introduced only after 60 epochs; its ProtoNCE weight is `0.25`. On their
self-supervised speaker-verification setup, MoCo gives `15.11%` EER,
waveform augmentation gives `8.63%`, and ProtoNCE further gives `8.23%`.

These numbers show that a queue can address the small-negative problem, but
they do not validate copying `10000`, `0.999`, or `0.25` into this much smaller
closed-set-plus-OOD competition.

Primary source: <https://arxiv.org/pdf/2012.07178.pdf>

### Cross-Batch Memory

Wang et al. show that detached embeddings from recent mini-batches can supply
many negatives with low memory cost after a warm-up period. Their original
evidence is image retrieval rather than speaker identification. They warm up
for 1000 iterations and select memory ratios per dataset; the paper therefore
does not supply a universal queue size for this campaign.

Primary source: <https://arxiv.org/pdf/1912.06798v3>

## Competition-specific label hazard

The competition label `unknown` aggregates 554 different OOD speakers. A
naive supervised contrastive implementation that uses the final 447-way label
would incorrectly treat every OOD file as a positive for every other OOD
file, explicitly collapsing distinct speakers into one embedding cluster.

Any future contrastive candidate must therefore satisfy one of these two
predeclared semantics:

1. known-only positives: OOD examples may act as negatives for known anchors,
   but never as positives for one another; or
2. train-only 1000-identity positives: the validated 554 OOD identity groups
   are used during training, then collapsed to the single competition
   `unknown` output exactly as in the existing provenance-safe backend.

The target Fold must not choose between these semantics after seeing its own
metric.

## Conditional decision

1. Do not alter or reinterpret P5. Finish the locked control/treatment pair and
   audit its OOF, Known/OOD balance, rescue topology, embedding spread,
   receipts, and MLflow artifacts.
2. Preserve the already preregistered P6 activation rule. P6 is a cheap,
   source-matched test of explicit inter-class separation and must not be
   replaced post hoc because this literature review occurred during P5.
3. Consider a P7 cross-batch supervised-contrastive pair only after P5/P6
   decisions and only if the remaining campaign budget can cover a complete
   matched control and treatment.
4. Before preregistering P7, measure embedding drift on train-only batches,
   establish a queue-memory feasibility bound, and choose exactly one queue
   policy and one coefficient without using the target Fold or leaderboard.
5. P7 must retain the authoritative Raw probability-average argmax path and
   the existing Known Accuracy/OOD-F1 guardrails. It is rejected after one
   Fold if Macro-F1 gain is below `+0.002`, either guardrail drops by more than
   `0.001`, representation spread falls below `0.95`, or the queue contains an
   invalid OOD-positive pair.

## Current conclusion

P5 remains a valid low-cost test of cross-file invariance, but a negative P5
result would not prove that supervised contrastive learning is ineffective.
The literature-supported missing ingredient is a sufficiently rich and
correctly labelled negative set. That candidate is retained as a conditional
future direction, not silently promoted ahead of the locked P5/P6 sequence.
