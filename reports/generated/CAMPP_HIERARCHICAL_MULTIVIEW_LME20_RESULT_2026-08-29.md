# CAM++ hierarchical multi-view LME20 result — 2026-08-29

## Decision

Rejected.  Equal-file hierarchical aggregation did not repair the flat-view
failure and must not replace the official `0.9667174285` leaderboard package.
No ZIP was built.

## Provenance

- preregistered code commit: `0e7d184`;
- result SHA256: `2670d041f73c42dbd395c136106c835f4cb5b1e235b2c174b897515e41170225`;
- reused per-view caches whose checkpoint, OOF, file order and aggregate
  reproduction hashes had already been validated;
- locked 4447-file OOF baseline reproduced exactly at Macro-F1
  `0.9633564052154656`;
- fixed beta `20` at every hierarchy level and unchanged alpha/kappa/tau/
  unknown weight; no search or leaderboard feedback.

## Metrics

| Scope | Candidate Macro-F1 | Delta Macro-F1 | Delta Known | Delta OOD-F1 |
|---|---:|---:|---:|---:|
| Fold 0 | 0.9566626487 | -0.0044830175 | -0.0067264574 | -0.0039088350 |
| Fold 1 | 0.9560125981 | -0.0029019302 | -0.0033821871 | -0.0019247067 |
| Fold 2 | 0.9421640278 | +0.0014913666 | +0.0022271715 | -0.0000257778 |
| Aggregate | 0.9604218640 | -0.0029345413 | -0.0035906643 | -0.0019582378 |

The candidate rescued 5 errors, introduced 14 and changed 19 predictions.
Fourteen of the changed files had all eight real query views, so this is not a
short-audio-only failure.  Fold2 was again weakly positive while Fold0 and
Fold1 were clearly negative; treating Fold2 as a selected regime would be
post-hoc Fold tuning.

## Interpretation and closed branch

Giving every enrollment file equal mass did not restore transferability.
Therefore the flat-view regression was not explained only by duration/view
count weighting.  Replacing the locked aggregate score shifts the known/OOD
boundary and remains unstable across Folds.  Both full replacement variants
(flat-view and hierarchical equal-file) are closed.

The next pre-registered hypothesis keeps every baseline known/unknown decision
bit-for-bit and permits temporal-view evidence only to rerank known identities
inside rows already accepted as known.  This directly isolates the useful
signal seen in the flat-view audit (known-to-wrong-known errors fell from 19 to
12) while structurally preventing its harmful known-to-unknown increase.
