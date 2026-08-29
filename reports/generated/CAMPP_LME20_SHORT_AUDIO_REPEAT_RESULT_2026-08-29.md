# CAM++ LME20 short-audio repeat result — 2026-08-29

## Decision

**Rejected.**  Repeating short audio to fill the eight-second inference window
does not generalise across the three fixed OOF folds.  The existing pad-mode
LME20 package remains the locked candidate.

## Verified result

| Scope | Pad Macro-F1 | Repeat Macro-F1 | Delta Macro-F1 | Delta Known Acc. | Delta OOD-F1 |
|---|---:|---:|---:|---:|---:|
| Fold 0 | — | — | -0.0015336 | -0.0011211 | -0.0033669 |
| Fold 1 | — | — | +0.0021074 | +0.0033822 | +0.0039098 |
| Fold 2 | — | — | -0.0048502 | -0.0022272 | -0.0013876 |
| Aggregate | 0.9633564 | 0.9618203 | -0.0015361 | +0.0004488 | -0.0002725 |

Across all 4447 unique OOF files, repeat changed 26 predictions, rescued 6
pad-mode errors and introduced 10 errors.  Of the 182 validation files shorter
than eight seconds, pad classified 112 correctly and repeat classified 107
correctly; repeat rescued 4 but introduced 9.

## Provenance checks

- The locked pad baseline reproduced exactly at Macro-F1
  `0.9633564052154656`.
- Enrollment and validation used matching repeat preprocessing.
- The existing train-only speaker and KMeans-554 group memberships were held
  fixed.
- CAM++ weights and locked LME20 beta/alpha/kappa/tau/unknown weight were held
  fixed.
- There was no candidate grid, threshold selection or leaderboard feedback.
- Long-file enrollment embeddings and validation probabilities were bitwise
  identical to pad mode; the largest long-file embedding difference was only
  `2.980232238769531e-08`, far below the preregistered `2e-5` tolerance.
- Repeat cache SHA256 values:
  - Fold 0: `51d0cda25c9b051686967cc113ad08e7b62a3d3bbab418844b51c65f994c879b`
  - Fold 1: `50d9190ed83fec4b73f81989c3b1a3f50b91e7f93f2bc6adebcd7b0c735f5468`
  - Fold 2: `2cbb1f89dd221840dd178e3a6b73f75b6c187985831f2f3fa27cb5a8c2a397e9`
- Result JSON SHA256:
  `1dd2ce09720d65f677daa6c1f285faeb8e84a965056085fb66349282f2d7e1b2`.

## Interpretation

The Fold-1 gain confirms that zero padding can be harmful in one domain
slice, but the two negative held-out folds show that naive periodic repetition
also creates a strong synthetic boundary/periodicity artefact.  Fold 2 is the
clearest failure and both its Known Accuracy and OOD-F1 guardrails are broken.
This is not a parameter issue: the experiment changed one preprocessing rule
and used no tunable values.  Consequently, selecting repeat only for Fold 1,
choosing a duration threshold from OOF, or blending pad/repeat after seeing
these results would be post-hoc tuning and is not allowed.

The next representation-level candidate should preserve real speech duration
without either eight-second zero dilution or periodic tiling.  A scientifically
distinct option is duration-aware embedding pooling (or speech-only segment
pooling) evaluated as another fixed three-fold transformation, but it requires
a separate preregistration and must not inherit a threshold from this rejected
audit.
