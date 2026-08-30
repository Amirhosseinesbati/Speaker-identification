# Cross-File Positive-Pair Implementation Readiness — 2026-08-30

## Status

This is a dormant conditional implementation, not an authorised experiment.
It may be preregistered only if the active same-crop Long-120 paired test is
terminal and shows neutral or harmful consistency without representation
collapse.  No configuration, campaign transition or training launch is part
of this change.

## Scientific isolation

The proposed fallback uses the existing 48-file balanced batch:

- 24 OOD rows remain on the existing OOD/primary path;
- 24 known rows are emitted as 12 speaker-balanced pairs;
- each pair contains two different `audio_file` values for one known speaker;
- both rows retain the normal ArcFace/OOD supervised objective;
- treatment adds only fixed positive-pair cosine alignment with a detached
  target; and
- the matched control uses the identical two-file sampler with no alignment
  loss.

The pair loss reuses embeddings already produced for supervised training.  It
does not load an extra file, add a second audio forward, invent an OOD positive,
or introduce contrastive negatives, a memory queue, clustering or an
adversarial head.  This makes `consistency.enabled` the intended eventual
single objective change between the matched branches.

## Exact Fold-0 feasibility evidence

The audit recreated `kfold/folds3/fold0/seed42` from 4,529 labelled files with
the production one-second duration floor and duplicate cleaning:

- 70 corrupted/short files were excluded;
- nine duplicate groups covering 69 files were detected;
- 62 rows were removed by duplicate cleaning (48 conflicting and 14 repeated);
- train/validation sizes were 2,819/1,632;
- the training fold contained 1,337 known files and 1,482 OOD files;
- all 1,337 known training rows had unique `audio_file` values;
- all 446 known speakers were present;
- distinct training files per known speaker were min/median/max `2/3/5`; and
- no speaker had fewer than two training files.

At batch 48 and OOD ratio 0.5, the deterministic sampler emitted 58 batches.
Every batch satisfied 24 OOD + 24 known, exactly 12 distinct known speakers and
two distinct file ids per selected speaker.  Across one epoch all 446 speakers
were exposed; per-speaker known-row exposure was min/median/max `2/4/4`.

## Literature boundary

A focused SciSpace retrieval found speaker-specific support for treating the
recording identity of a positive pair as a real experimental variable.  SSPS
(Interspeech 2025, DOI `10.21437/Interspeech.2025-183`) identifies standard
same-utterance positive sampling as a source of retained recording-channel
information and reports lower intra-speaker variance when positives come from
different recording conditions.  Augmentation-adversarial speaker training
(arXiv `2007.12085`) makes the related observation that two segments from one
utterance share acoustic conditions, while asymmetric clean/noisy pairing
(arXiv `2309.04265`) supports a stable clean target for augmented views.

These papers motivate the dormant branch but do not validate its competition
decision layer: none demonstrates preservation of a joint 446-way known
classifier and binary unknown-speaker detector under this dataset's split.
Consequently the evidence cannot waive the preregistered Known Accuracy and
OOD-F1 guardrails, cannot select a consistency coefficient, and cannot justify
launching this branch before the active same-crop pair is terminal.

## Safeguards implemented

- fail if known samples per batch are odd;
- fail if a known speaker has fewer than two rows;
- fail if two rows silently share an `audio_file` id;
- fail if pair sampling is combined with per-file known weights;
- fail if cross-file consistency is enabled without the paired sampler;
- fail if a treatment batch does not contain exactly two known rows per
  selected speaker;
- ignore OOD rows in the pair loss; and
- keep target embeddings stop-gradient while both files retain supervised
  gradients.

The targeted sampler, consistency, configuration, training/resume, audit and
supervisor regression suites passed `79/79` CPU tests.  The two emitted PyTorch
scheduler warnings come from existing tests that intentionally step a
scheduler without an optimiser step; they are unrelated to this change.

## Activation boundary

If activated later, the matched control and treatment require immutable raw
config hashes, the same source checkpoint/split/augmentation/horizon/seed and
a separate preregistration.  The current same-crop control or treatment must
never be mutated, resumed into this sampler, or used to select its coefficient.
