# CAM++ temporal known-only rerank preregistration — 2026-08-29

## Evidence and fixed hypothesis

The rejected flat temporal-view replacement reduced aggregate
known-to-wrong-known errors from 19 to 12 and unknown-to-known errors from 25
to 21, but raised known-to-unknown errors from 87 to 100.  The hierarchical
variant also failed.  This indicates potentially useful identity-ranking
evidence alongside a non-transferable known/unknown score shift.

The fixed candidate preserves every locked file-aggregate LME20 binary decision
bit-for-bit.  Rows predicted unknown remain unknown.  Rows predicted known
remain known, but their known identity is the argmax over classes 1..446 after
the already fixed temporal-view LME20 prototype/head fusion.  This can only
affect known-identity ranking; it cannot rescue or introduce an OOD boundary
error.

## Locked variables and invariants

- selected Raw CAM++ Control Fold0/1/2 checkpoints and 4447-file OOF;
- existing train-only known/KMeans-554 enrollment groups and validated view
  caches;
- flat real-view LME beta `20`, alpha `0.15`, kappa `16` and unchanged head
  probability average;
- baseline binary decision from the current LME20 tau `0.50` and unknown weight
  `0.75`;
- no threshold, blend, duration rule, learned gate, Fold selection, epoch
  selection or leaderboard feedback.

For every row and Fold, `(candidate == 0)` must equal `(baseline == 0)` exactly;
therefore OOD-F1 must be exactly unchanged.  The only changed variable is the
known-class argmax on baseline-known rows.

## Acceptance and resource bound

Acceptance requires positive Macro-F1 gain in every Fold, aggregate gain at
least `+0.001`, no Known Accuracy loss beyond `0.001`, exact OOD-F1 equality and
the standard aggregate/fold guardrails.  Passing would authorize a full-data
implementation/equivalence audit, not automatic leaderboard submission.

Existing view caches make this a scoring-only GPU audit with a 20-minute /
`$0.06` ceiling.  Any binary mismatch, cache/provenance mismatch, non-finite
score or OOM stops immediately.
