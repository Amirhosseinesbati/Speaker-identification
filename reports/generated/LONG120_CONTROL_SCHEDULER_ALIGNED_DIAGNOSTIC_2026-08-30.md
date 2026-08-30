# Long-120 Control Scheduler-Aligned Diagnostic — 2026-08-30

## Scope

This is an interim monitoring diagnostic for
`p4-campp-known446-ood-channelrobust-paired-control-long120-oof-f0`.  It does
not select a checkpoint, modify the locked 120-epoch control/treatment pair,
or authorise a submission.  Its purpose is to distinguish a slower scheduler
from an intrinsically worse trajectory when comparing the 120-epoch control
with the earlier 40-epoch paired-control pilot.

## Authoritative state

At the audit point, the campaign and the original supervisor/pipeline process
tree were live and unique.  The latest readable checkpoint contained 59
complete epochs with scheduler state at epoch 59, Fold 0 of 3 and seed 42.
MLflow Run `8c771e5cb91c45b79e85672f9d901d01` was `RUNNING` with 22 parameters,
33 metric series and 59/59 contiguous points.  No OOM, NaN or traceback token
was present in the stable log.

The selected interim Raw checkpoint remained epoch 51:

- probability-average Macro-F1: `0.9437080966861129`;
- Known Accuracy: `0.9517937219730942`; and
- OOD-F1: `0.9549795361527967`.

These values are observations, not a terminal gate result.

## Alignment method

Raw epoch-number matching is inappropriate because the long control uses six
warm-up epochs followed by 114 cosine epochs, while the pilot uses two warm-up
epochs followed by 38 cosine epochs.  For each completed long-control epoch,
the diagnostic computed post-warm-up cosine progress

`(epoch - warmup) / (total_epochs - warmup)`

and mapped that progress to the nearest pilot epoch.  It then checked the
actual logged head learning rates.  This is a descriptive trajectory
comparison; repeated nearest-neighbour pilot epochs make adjacent deltas
correlated, so they must not be treated as independent significance samples.

## Result through epoch 59

- mean aligned Macro-F1 delta over all 59 long-control points: `+0.00077708`;
- aligned delta was positive for `67.80%` of long-control points;
- mean aligned delta over the latest ten points: `+0.00227444`; and
- at long epoch 59, normalized cosine progress was `0.4649123`, mapping to
  pilot epoch 20.

The epoch-59 learning rates were closely aligned:

- long epoch 59 head LR: `5.7725390e-5`;
- pilot epoch 20 head LR: `5.6422519e-5`.

At that aligned point, Long-120 had Raw probability-average Macro-F1
`0.9400059247` versus pilot `0.9379373081`, a delta of `+0.0020686166`.

## Interpretation boundary

The previously observed epoch-number-matched deficit does not establish that
Long-120 is worse: after scheduler alignment, the sign reverses and the recent
window favours the longer schedule.  This is evidence for continuing the
preregistered horizon rather than stopping early.  It is not evidence that the
final model will beat the original Fold-0 control, that consistency treatment
will help, or that any particular intermediate epoch should be selected.

The only admissible next action remains unchanged: finish and audit the locked
120-epoch control, then run the matched fixed-0.1 treatment after its separate
VRAM/throughput preflight.  Final conclusions must use the full trajectories,
the preregistered Known/OOD guardrails and the fixed ensemble/rescue gates.
