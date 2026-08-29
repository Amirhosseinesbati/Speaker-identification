# CAM++ LME20 native-length short-audio preregistration — 2026-08-29

## Motivation and single hypothesis

The fixed repeat-padding candidate failed because periodic tiling introduced
more errors than it rescued.  It nevertheless improved Fold 1, confirming that
zero dilution can matter for one domain slice.  The next single candidate
removes both artefacts: audio shorter than eight seconds is passed to CAM++ at
its true duration, with neither zero padding nor repetition.

Only the short-file forward representation changes.  Files at least eight
seconds long are copied bit-for-bit from the locked pad caches.  Enrollment and
validation are transformed together.

## Locked policy

- selected Raw CAM++ Control checkpoints for fixed Fold 0/1/2 OOF;
- unchanged train-only speaker and KMeans-554 group memberships;
- LME beta `20`, alpha `0.15`, kappa `16`, tau `0.50`, unknown weight `0.75`;
- probability-average head plus mean-then-L2-normalised embedding;
- direct fixed decision rule with no candidate grid or leaderboard tuning.

## Invariants and gate

The pad baseline must reproduce Macro-F1 `0.9633564052154656`; all 4447 OOF
files must remain unique and non-overlapping; cache/checkpoint/OOF SHA256 values
must validate; and every long-file row must be exactly equal to its pad source.

Acceptance requires all of the following:

- strictly positive Macro-F1 delta in all three folds;
- aggregate Macro-F1 delta at least `+0.001`;
- no per-fold or aggregate Known Accuracy drop beyond `0.001`;
- no per-fold or aggregate OOD-F1 drop beyond `0.001`.

Failure leaves the existing local LME20 PCM-recovery ZIP unchanged.  Passing
only authorizes a separately reviewed full-data prototype rebuild and package
rehearsal.

## Resource and stop rule

Only short rows are recomputed on the existing RTX 3090; long rows are reused.
Timeout is one hour (about `$0.18` at the current rate).  Non-finite outputs,
shape/provenance mismatch, native-length incompatibility, OOM or failure to
reproduce the locked baseline stop the audit immediately.
