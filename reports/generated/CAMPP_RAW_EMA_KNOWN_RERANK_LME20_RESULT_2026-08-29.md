# CAM++ Raw/EMA binary-locked known reranker result — 2026-08-29

## Decision

Rejected.  Keeping the locked Raw known/unknown decisions and using the fixed
`50/50` Raw/EMA LME20 evidence only to rerank classes `1..446` does not improve
the externally validated Raw policy.  No submission package is authorised.

## Three-fold result

| Fold | Macro-F1 delta | Accuracy delta | Known delta | OOD-F1 delta |
|---:|---:|---:|---:|---:|
| 0 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| 1 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| 2 | -0.0003728561 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| **OOF** | **-0.0000635085** | **0.0000000000** | **0.0000000000** | **0.0000000000** |

The aggregate Raw Macro-F1 was `0.9633564052`; the candidate produced
`0.9632928968`.  It changed one Fold-2 prediction, but that change moved one
wrong known identity to another wrong known identity.  It rescued zero errors,
introduced zero additional errors, and preserved all 4,447 binary Known/OOD
decisions exactly.

## Interpretation

The preceding full Raw/EMA audit showed that EMA could reduce some
`known->wrong-known` and `unknown->known` errors only by rejecting additional
known files.  This narrower experiment removed that binary trade-off and found
no useful identity gain.  Therefore the selected Raw and EMA snapshots are too
correlated for productive fusion, and the complete CAM++ snapshot branch is
closed.  The next representation experiment must create genuine diversity
through a different training policy or encoder rather than another snapshot
weight or decision threshold.

## Provenance

- preregistration/implementation commit: `6cbe828100f5`
- result JSON: `reports/generated/campp_raw_ema_known_rerank_lme20.json`
- result JSON SHA256:
  `f2d0811cf2d2a4eb0d0072d1dc26f66bed4fc2dc2e8728465be5992b73b456e0`
- OOF files: 4,447 unique and non-overlapping
- local focused tests: 5 passed
- local full suite: 161 passed, 8 skipped
- worker focused tests: 5 passed
- EMA inference caches: reused; no new extraction or parameter selection
- leaderboard used for parameter selection: no

