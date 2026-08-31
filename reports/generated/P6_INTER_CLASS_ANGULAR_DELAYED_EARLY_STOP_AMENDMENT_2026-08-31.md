# P6 delayed early-stopping amendment — 2026-08-31

## Decision status

`PREREGISTERED / CONTROL NOT YET STARTED`

The original P6 pair used a mandatory 120-epoch horizon. The user subsequently
required a defensible early-stopping policy unless a fixed horizon is genuinely
necessary. This amendment changes the shared stopping policy of both P6 arms;
it does not change the inter-class hypothesis, coefficient, data, split,
augmentation, sampler, model, optimiser, schedule, decision policy or gates.

## Why ordinary patience is unsafe

The policy was chosen from completed predecessor histories, before any P6
result exists. Replaying conventional patience from epoch one produced the
following counterfactuals:

- P5 matched control: patience 15/20/25/30 would stop at epoch 16/21/26/31 and
  retain epoch 1 (`0.9356361`), missing the actual epoch-87 best (`0.9409893`)
  by `0.0053532`.
- P5 treatment: patience 25 would stop at epoch 50 and miss its epoch-75 best
  by `0.0027030`; patience 30 was the first tested ordinary value that retained
  the best within the observed 104 epochs.
- P3 source: patience 15 already retained its epoch-81 best, showing that the
  failure is recipe-dependent rather than a general reason to disable stopping.

The learning dynamics therefore contain a long low-amplitude adaptation phase.
Starting patience at epoch one is scientifically unsafe, while forcing all 120
epochs spends budget on a tail that predecessor runs show can be non-improving.

## Locked adaptive horizon

- Maximum epochs: `120`; this is an upper bound, not a required horizon.
- Raw probability-average Macro-F1 remains the sole selection and stopping
  metric.
- Patience starts at epoch `80`.
- Patience: `20` consecutive epochs without a strict Raw improvement.
- The best Raw checkpoint is tracked from epoch 1, including before epoch 80.
- On a stateful continuation, consumed post-start patience is reconstructed
  from checkpoint history; a timeout cannot grant a fresh patience window.
- Milestones at epochs 40 and 80 remain diagnostic only.

Historical replay of this exact policy preserves every observed predecessor
best: P5 control stops at 107 and selects epoch 87; P5 treatment stops at 99 and
selects epoch 75; P3 stops at 101 and selects epoch 81. The maximum observed
score loss relative to each complete available history is exactly zero.

## Locked profiles and hashes

- Control:
  `p6-campp-known446-ood-crossfile-consistency-interclass-control-es80p20-oof-f0`
  SHA256 `1f041abb3035af1f39de1714f389a6b9aadb04fb0a0c34e737c5ca363d92b3ab`.
- Treatment:
  `p6-campp-known446-ood-crossfile-consistency-interclass-e01-es80p20-oof-f0`
  SHA256 `25dd1f1a52e4908c47e66b234de8e58671b82cb78fa528453a3865490f01ba5c`.

After normalising profile identity and the shared stopping amendment, the
control reproduces the historical P5 treatment recipe. Within the new P6 pair,
the only scientific difference remains
`training.loss.speaker.inter_class.enabled: false -> true`; weight `0.01` is
unchanged and is not tunable.

## Execution and gates

The control must run first. The treatment is authorised only after a clean
control terminal, complete receipt/OOF/MLflow provenance, healthy embedding
spread and a fresh budget check. Each arm retains the original 8-hour / `$1.40`
hard cap, though delayed stopping is expected to shorten it. Immediate stops
remain NaN, provenance mismatch, artifact corruption, second OOM or invalid
pairing.

All original P6 score, fixed-fusion, Known/OOD, rescue, spread and exclusive
energy gates remain unchanged. Passing Fold 0 authorises only a separately
preregistered multi-fold evaluation, never automatic submission or leaderboard
tuning.
