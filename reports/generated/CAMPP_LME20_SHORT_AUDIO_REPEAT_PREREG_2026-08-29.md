# CAM++ LME20 short-audio repeat preregistration — 2026-08-29

## Motivation

The locked three-fold CAM++ + LME20 OOF policy reaches Macro-F1
`0.9633564052154656`.  Its residual audit contains 131 errors: 87
known-to-unknown, 19 known-to-wrong-known and 25 unknown-to-known.  Previous
quality analysis found that the hard known-error tail is much shorter than the
correct population, while the current eight-second inference policy zero-pads
all files shorter than one window.  Zero padding can dilute time statistics and
creates a train/inference representation mismatch for very short speech.

## Single hypothesis

For audio shorter than eight seconds, tile the waveform until it fills one
eight-second window.  Apply the same operation to enrollment and validation
audio.  Do not change weights, folds, augmentation, speech-aware window
selection, number of windows, prototype grouping, aggregation or decision
parameters.

Locked decision policy:

- selected Raw CAM++ checkpoint for Control Fold 0/1/2;
- fixed train-only KMeans-554 grouping per Fold;
- log-mean-exp enrollment aggregation with beta `20`;
- probability fusion alpha `0.15`;
- prototype softmax kappa `16`;
- raw maximum threshold tau `0.50`;
- unknown probability weight `0.75`;
- direct argmax after the fixed rule.

There is no candidate grid and no leaderboard tuning.

## Provenance and invariants

The audit must:

1. reproduce the locked pad-mode OOF Macro-F1 exactly;
2. validate 4447 unique, non-overlapping OOF files;
3. hash every source checkpoint, OOF bundle, pad artifact and repeat cache;
4. preserve the enrollment and OOF file order and group membership;
5. show maximum long-file differences no greater than `2e-5` for enrollment
   embeddings, validation probabilities and validation embeddings;
6. reject on non-finite values, shape mismatch, provenance mismatch, OOM or a
   failed long-file equivalence check.

## Acceptance gate

The repeat candidate is accepted only when all conditions hold:

- Macro-F1 gain is strictly positive in each of Fold 0, Fold 1 and Fold 2;
- aggregate three-fold Macro-F1 gain is at least `+0.001`;
- Known Accuracy does not fall by more than `0.001` in any Fold or aggregate;
- OOD-F1 does not fall by more than `0.001` in any Fold or aggregate.

Passing this gate only authorizes building and rehearsing a separate repeat
submission package.  Failure rejects the hypothesis and leaves the current
local LME20 PCM-recovery ZIP unchanged.

## Resource bound

The audit runs on the existing Vast.ai RTX 3090 worker, reuses idempotent
per-Fold caches, and has a two-hour timeout.  At the current hourly rate this is
at most about `$0.35`, well inside the campaign ceiling of `$20`.
