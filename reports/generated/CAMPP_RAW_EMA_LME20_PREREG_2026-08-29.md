# CAM++ Raw/EMA LME20 snapshot ensemble preregistration — 2026-08-29

## Motivation

The externally validated CAM++ Raw + LME20 + PCM-recovery package scored
`0.9667174284505605` leaderboard Macro-F1.  Recent score-normalisation,
quality, short-audio, temporal-view, and known-reranking branches did not pass
the three-fold OOF guardrails.  The next candidate therefore tests model
evidence diversity rather than another decision heuristic.

Each Control fold retained three binary-distinct CAM++ snapshots.  The best EMA
snapshot is individually weaker than the selected Raw snapshot in all three
folds, but it was selected at a different epoch and may rescue a different
subset of Raw errors.  That complementarity is not assumed; it is the fixed
hypothesis under test.

## Locked candidate

- Control Fold0/1/2 and `kfold/folds3/seed42` remain unchanged.
- For each target fold, use its already selected `campp_best_raw.pt` and its
  already selected `campp_best_ema.pt`; no epoch is selected in this audit.
- Extract enrollment embeddings with the production `model.embed` cache path.
- Extract validation head probabilities and embeddings with the production
  per-file `predict_proba_and_embed` path.
- Build independent Raw and EMA LME20 prototype evidence using beta `20`.
- Average Raw and EMA head probabilities, prototype probabilities, and maximum
  prototype scores with fixed weights `[0.5, 0.5]`.
- Apply the unchanged policy: alpha `0.15`, kappa `16`, tau `0.50`, and unknown
  weight `0.75`.
- `campp_latest.pt` is provenance-only and is not evaluated or substituted.
- No threshold, epoch, fusion weight, or submission is selected on OOF or the
  leaderboard.

## Provenance and failure rules

All EMA caches must bind the EMA and Raw checkpoint SHA256, Raw OOF and
enrollment-artifact SHA256, ordered train/validation filename digests, class
map, cluster ids, and finite array shapes.  A Raw self-ensemble must reproduce
the locked Raw decisions and probabilities to `1e-8`.  NaN, OOM, class-map or
config mismatch, file-order mismatch, OOF overlap, hash drift, or failure to
reproduce the locked `0.9633564052154656` Raw OOF baseline aborts the audit.

## Acceptance gate

The candidate is accepted only when all conditions hold:

1. Macro-F1 gain is strictly positive in every held-out Fold.
2. Aggregate Macro-F1 gain is at least `+0.001`.
3. Known Accuracy and OOD-F1 decline by no more than `0.001` in every Fold.
4. Aggregate Known Accuracy and OOD-F1 decline by no more than `0.001`.

Failure closes this snapshot-ensemble branch and produces no submission ZIP.
Passing permits a separate deployment-equivalence implementation and audit; it
does not automatically authorise leaderboard submission.
