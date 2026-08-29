# CAM++ view-level LME20 result — 2026-08-29

## Decision

Rejected.  The fixed view-pair LME20 candidate regressed aggregate Macro-F1 and
failed both Known Accuracy and OOD-F1 guardrails.  It must not replace the
locked CAM++ LME20 submission, and no leaderboard package was built.

## Reproducibility

- code commit used on the worker: `a8ba5bea50474434786bae4e5eac18a57ceb29ac`;
- result SHA256: `f9791426b2e03594e70fd535db5363227bc33ab9bc5763a4717cd35f0e94966c`;
- 4447 unique three-Fold OOF files, with no overlap;
- locked baseline reproduced exactly at Macro-F1 `0.9633564052154656`;
- enrollment aggregate reproduction maximum absolute difference:
  `2.9802322387695312e-08` in every Fold;
- validation aggregate reproduction maximum absolute difference: exactly `0`
  in every Fold;
- cache SHA256 values: Fold0
  `4296a9eaa9a69c2d6cc52534839e61bb4ff7937d28a1786de18fe78c7587a1e7`,
  Fold1 `fd11c5f7083efc276a1b3014c59947db723f7d41280fb3ca95cebf11f720e380`,
  Fold2 `57bcd5a8d0b4539e594000be84fb6e2177697b3723e13658b4ae53d41aa9db6f`.

Two earlier engineering attempts stopped before metrics because the embedding
batching path did not reproduce locked artifacts.  They produced no candidate
result.  The final audit explicitly matched production: window-major batches
for enrollment and one file's windows per batch for validation/query.

## Metrics

| Scope | Baseline Macro-F1 | Candidate Macro-F1 | Delta Macro-F1 | Delta Known | Delta OOD-F1 |
|---|---:|---:|---:|---:|---:|
| Fold 0 | 0.9611456663 | 0.9586984534 | -0.0024472128 | -0.0056053812 | -0.0050152421 |
| Fold 1 | 0.9589145283 | 0.9575054558 | -0.0014090725 | -0.0022547914 | -0.0012839887 |
| Fold 2 | 0.9406726612 | 0.9421654923 | +0.0014928312 | +0.0022271715 | +0.0006288642 |
| Aggregate | 0.9633564052 | 0.9619709820 | -0.0013854233 | -0.0026929982 | -0.0018988021 |

The candidate rescued 11 baseline errors, introduced 13 errors and changed 35
predictions.  Aggregate known-to-wrong-known errors fell from 19 to 12 and
unknown-to-known errors fell from 25 to 21, but known-to-unknown errors rose
from 87 to 100.  This is a useful error-topology signal: retaining temporal
views can correct fine speaker and OOD false-accept decisions, but equal weight
per view over-penalises known files and is not calibrated to the locked raw
maximum threshold.

## Scientific interpretation

The Fold2-only gain is not transferable evidence; Fold0 and Fold1 both
regressed, so selecting Fold2 behaviour or retuning `tau`, `beta`, `alpha` or a
duration cutoff would be post-hoc tuning and is forbidden.  The next candidate
may test a parameter-free hierarchical aggregation that gives each enrollment
file equal total mass while retaining its real temporal views.  That is a new,
pre-registered structural hypothesis motivated by the observed view-count
weighting failure—not a parameter sweep of this rejected candidate.
