# CAM++ Raw/EMA binary-locked known reranker preregistration — 2026-08-29

## Motivation

The fixed full Raw/EMA LME20 ensemble was rejected at aggregate Macro-F1
`0.9630353626` versus the Raw `0.9633564052`.  It reduced
`known->wrong-known` errors from 19 to 10 and `unknown->known` from 25 to 19,
but increased `known->unknown` from 87 to 99.  This follow-up isolates the
potentially useful identity complementarity without permitting the harmful
binary decision change.

## Locked candidate

- The baseline prediction and its known/unknown status come exclusively from
  the locked Raw CAM++ LME20 policy.
- Files predicted unknown by Raw remain unknown.
- Files predicted known by Raw are reranked only among classes `1..446` using
  the fixed `50/50` Raw/EMA LME20 fused probability vector.
- The already hash-verified EMA caches are reused; no model is re-extracted.
- Folds, snapshots, weights, LME20 beta, alpha, kappa, tau, and unknown weight
  are unchanged and not selected.
- No leaderboard observation is used to select any parameter.

## Acceptance gate

1. The known/unknown decision vector must be exactly identical to Raw in every
   Fold; OOD-F1 must therefore be identical to `1e-12`.
2. No Fold may have negative Macro-F1 delta.
3. Aggregate Macro-F1 gain must be at least `+0.001`.
4. Aggregate Known Accuracy gain must be at least `+0.001`.

Failure closes the Raw/EMA snapshot branch and produces no submission.  Passing
permits a separate deployment-equivalence implementation and package audit,
not automatic leaderboard submission.
