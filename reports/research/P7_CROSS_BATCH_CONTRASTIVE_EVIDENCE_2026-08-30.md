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

The full-text methodology also sharpens an important transfer limitation. The
paper forms each view from an utterance and one stochastic augmentation of
that same utterance; it does not isolate the harder distinct-recording
positive-pair question that P5 tests. Its ablation reports progressively lower
EER as the contrastive batch grows from `128` to `512` to `1024`, while the
main configuration uses batch size `3072`. A direct SupCon treatment at this
campaign's batch size `48` would therefore remove one of the mechanism's
experimentally supported ingredients and is not a defensible next run.

### Identical-condition angular metric-learning evidence

Chung et al. compare classification, triplet, prototypical, GE2E, and angular
prototypical objectives under a shared speaker-recognition training setup.
Their metric-learning batches contain `M` utterances from each of `N`
speakers. Triplet mining uses `M=2`; prototypical and angular-prototypical
losses classify a query against same-batch speaker centroids, so the softmax
over multiple centroids supplies hard negatives without selecting one triplet
per anchor. Angular prototypical performs best with `M=2` in their reported
table and improves as total batch size increases from `200` to `800`.

This is stronger support for P6's *combination* of cross-file compactness and
explicit inter-class angular separation than for positive-only consistency.
It is not a transferable coefficient or a licence to replace P6: the paper
uses VoxCeleb verification EER, thousands of identities, much larger batches,
and a test-time verification protocol. It also reports that difficult hard
negative mining is delayed until epoch 100 because enabling it early can make
training diverge. Any later P7 hard-negative mechanism must consequently have
a preregistered warm-up and must not mine the aggregate `unknown` label as if
it were one speaker.

Primary source: <https://arxiv.org/pdf/2003.11982>

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

### Masked Proxy / Multinomial Masked Proxy

Lian et al. combine entity-to-centroid comparisons for speakers represented
inside the mini-batch with entity-to-proxy comparisons for every speaker not
represented in that batch. This is relevant because it supplies inter-class
negatives without requiring every training identity to occur in one batch.
Their balanced variant uses two samples per represented speaker, which is
structurally close to the cross-file sampler already being tested in P5.

The transfer evidence is nevertheless weak for this competition. The paper
trains Thin-ResNet34 on 5994 VoxCeleb2 identities with expected batch sizes of
400 or 800, evaluates speaker-verification EER, and reports the best values
after exploring loss weights and other hyperparameters. It does not validate
a coefficient for CAM++ with a 447-way Macro-F1 objective or a heterogeneous
aggregated OOD class. MP/MMP is therefore a batch-efficient research option,
not a licensed P7 recipe.

Primary source: <https://arxiv.org/pdf/2011.04491>

### Reciprocal points with real unknown negatives

Chen et al. propose SRPL+ for open-set speaker identification. Their negative
samples are not collapsed into a same-speaker positive class: 1000 or more
real or synthesised unknown-speaker samples are pushed toward high entropy
relative to learnable reciprocal points. This semantic treatment is better
aligned with the competition's 554 heterogeneous OOD identities than naive
447-label supervised contrastive learning.

Direct transfer is still unjustified. Their experiments use a WavLM frontend,
a three-layer few-shot adapter, ten target speakers, text-dependent datasets,
100 training epochs, and AUC/OSCR rather than 447-class Macro-F1. The useful
hypothesis is the *role* of real OOD identities as diverse negatives, not the
reported architecture, coefficient, epoch count, or reciprocal-point count.

Primary source: <https://arxiv.org/pdf/2409.15742>

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
   establish a queue/proxy-memory feasibility bound, and choose exactly one
   negative mechanism (cross-batch queue, masked proxies, or reciprocal-point
   entropy) and one coefficient without using the target Fold or leaderboard.
   The choice must be made from source-Fold evidence and engineering bounds;
   the target Fold cannot select among the three mechanisms.
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
