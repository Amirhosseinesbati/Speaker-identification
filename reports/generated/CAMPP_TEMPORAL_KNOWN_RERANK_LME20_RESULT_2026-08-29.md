# CAM++ temporal known-only rerank result — 2026-08-29

## Decision

Rejected as neutral.  The candidate preserved every prediction exactly, so it
cannot improve the official `0.9667174285` package and no ZIP was built.

## Evidence

- preregistered code commit: `c53b64d`;
- result SHA256: `620b609fdd28f2a1f61f61ea7a755dce13727f234d2de71036806459ec98f89c`;
- 4447 unique OOF files and validated Fold-specific view caches;
- candidate/baseline unknown masks exactly equal in every row;
- OOD-F1 exactly equal in every Fold and aggregate;
- zero changed predictions, zero rescues and zero introduced errors;
- Macro-F1, Known Accuracy, OOD-F1 and accuracy deltas all exactly `0` in all
  three Folds and aggregate.

## Interpretation

The temporal fused known argmax already equals the locked file-aggregate known
argmax for every row accepted as known.  Therefore the apparent reduction of
known-to-wrong-known counts in the rejected full flat-view replacement was not
speaker-identity correction: those rows moved across the known/unknown boundary
instead.  Temporal view scoring supplies no usable complementary known-class
ranking under the fixed fusion.

The entire temporal-view decision branch is now closed: flat replacement,
equal-file hierarchical replacement and binary-locked known reranking all fail.
Further post-hoc temporal thresholds or selective rules would be tuning the same
rejected evidence.  The next work should move to independent model evidence
(available Raw/EMA/latest checkpoints or a new leakage-free model replicate),
while retaining the externally validated LME20 + PCM stack as the baseline.
