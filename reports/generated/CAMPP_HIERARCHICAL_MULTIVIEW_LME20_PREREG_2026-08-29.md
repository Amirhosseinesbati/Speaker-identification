# CAM++ hierarchical multi-view LME20 preregistration — 2026-08-29

## Evidence and hypothesis

The flat view-pair LME20 audit retained temporal evidence but regressed Macro-F1
by `0.0013854`: it reduced known-to-wrong-known and unknown-to-known errors, yet
raised known-to-unknown errors from 87 to 100.  Its equal-per-view aggregation
also gave a longer enrollment file more total mass than a short file.

The fixed candidate retains the same real temporal views but applies beta-20
normalised log-mean-exp hierarchically: first across views within each
enrollment file, then across enrollment files within each speaker group, and
finally across real query views.  Each enrollment file therefore has equal
total mass regardless of duration or view count.  This follows the general
multi-enrollment principle of preserving utterance-level structure without a
learned backend:

- https://arxiv.org/abs/2203.05642
- https://www.isca-archive.org/interspeech_2014/pelecanos14_interspeech.html
- https://www.isca-archive.org/interspeech_2004/cheung04_interspeech.html

## Locked variables

- selected Raw CAM++ Control Fold0/1/2 checkpoints and fixed OOF;
- existing train-only known and KMeans-554 enrollment membership;
- production-matched window-major enrollment and file-major query extraction;
- exclusion of repeated last-window padding views;
- beta `20` at all three levels, alpha `0.15`, kappa `16`, tau `0.50`, unknown
  weight `0.75`, head probability average and direct decision rule;
- no grid, duration cutoff, calibration fit, learned attention, threshold/blend
  selection, epoch selection or leaderboard feedback.

The only changed variable relative to the rejected flat-view audit is the
predefined equal-file hierarchy.  Existing per-view caches are reused only if
checkpoint, OOF, file order and aggregate-reproduction hashes all match.

## Acceptance and resource bound

Acceptance requires positive Macro-F1 gain in every Fold, aggregate gain at
least `+0.001`, and no Known Accuracy or OOD-F1 loss beyond `0.001` in any Fold
or aggregate.  Failure leaves the current local ZIP unchanged.  The cached
three-Fold GPU scoring has a 20-minute / `$0.06` ceiling; OOM, non-finite scores
or provenance mismatch stops immediately.
