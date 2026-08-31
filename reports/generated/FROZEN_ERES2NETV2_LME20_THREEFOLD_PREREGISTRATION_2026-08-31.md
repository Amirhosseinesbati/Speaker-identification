# Frozen ERes2NetV2 + LME20 three-Fold preregistration — 2026-08-31

## Motivation and immutable anchor

The externally validated anchor remains CAM++ + LME20 + PCM recovery:

- leaderboard Macro-F1 `0.9667174284505605`;
- leaderboard accuracy `0.9667036625971143`;
- aggregate three-Fold OOF Macro-F1 `0.9633564052154656`;
- package SHA256
  `3653d0f4e54433f4096a521d814d5e606c5b9314fefa986b315b7623143a7494`.

The previous task-head ERes2NetV2 training recipe failed Fold 0.  That result
does not isolate the pretrained encoder: the older raw-embedding diagnostic
ranked frozen ERes2NetV2 above frozen CAM++ for known leave-one-out top-1 and
OOD AUC, but it lacked exact Fold filenames and was not valid OOF evidence.
This experiment tests the pretrained representation directly, with no task
head training and no parameter search.

## Single scientific question

Does the official, fully frozen ERes2NetV2 representation provide prototype
evidence that improves or usefully complements the locked CAM++ LME20 policy
under the exact clean three-Fold split?

## Data and leakage controls

- Folds: exact `kfold/folds3/seed42` Control folds.
- Expected held-out union: exactly `4447` unique files with zero overlap.
- Enrollment membership, known labels and 554 pseudo-unknown group ids come
  from the already hashed Control train artifacts for the matching Fold.
- The exact held-out union contains `4447` filenames.  The three immutable
  enrollment artifacts additionally share four train-only anchors that are
  deliberately never held out, so the frozen embedding cache contains exactly
  `4451` unique filenames.  Evaluation remains restricted to the `4447` OOF
  rows; each target Fold uses only its matching train-side enrollment rows.
- The waveform/window policy is read from the immutable Control checkpoints
  and must be identical across all three Folds.
- The official ERes2NetV2 checkpoint must match SHA256
  `0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c`.
- No leaderboard labels, score or submission output select a parameter.

## Locked scoring policy

All variants use `beta=20`, `kappa=16`, `tau=0.50`, unknown multiplier
`0.75`, and the existing 554 pseudo-unknown groups.  No value is searched.

The primary candidate is fixed before inference:

1. keep the immutable CAM++ 447-way head evidence;
2. average CAM++ and frozen ERes2NetV2 LME20 prototype probabilities `50/50`;
3. average their raw maximum LME scores `50/50` for the unchanged OOD gate;
4. retain head/prototype alpha `0.15/0.85` and direct argmax.

Two mechanism-only diagnostics are also recorded: ERes prototype-only and
CAM++ head + ERes prototype.  They cannot be promoted post hoc if the primary
candidate fails.

## Acceptance gate

The primary candidate passes only if all conditions hold:

- aggregate Macro-F1 gain at least `+0.001` over locked CAM++ LME20;
- aggregate Known Accuracy and OOD-F1 drops each no worse than `-0.001`;
- at least two held-out Folds have positive Macro-F1 direction;
- the worst Fold Macro-F1 delta is at least `-0.001`;
- at least `15%` of locked-baseline errors are rescued;
- rescued errors outnumber introduced errors;
- every cache, config, checkpoint, OOF and output receipt passes SHA/shape/
  finite-value validation.

Passing authorises packaging engineering analysis, not leaderboard tuning.
Failure retires this frozen ERes2NetV2 hypothesis and activates the previously
declared honest audit of historical P0/no-proto/metric-only models, followed
by frozen ECAPA if those representations are not complementary.

## Runtime and stop rules

- GPU: existing Vast.ai RTX 3090 worker only.
- Encoder: fully frozen; no optimizer, scheduler, EMA or epoch selection.
- Batch size `48`, eight-window production path, at most `1.5` hours and
  `$0.30` incremental campaign cost.
- Immediate failure: checkpoint/input hash mismatch, Fold overlap, missing
  enrollment group, non-finite embedding, OOM, or cache corruption.
- P6 remains dormant and TitaNet remains prohibited.
