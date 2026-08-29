# CAM++ Paired Consistency Long-120 Preregistration — 2026-08-30

## Decision

The decision experiment uses a fresh, matched 120-epoch horizon for both the
no-consistency control and the fixed-weight consistency treatment. The earlier
40-epoch run remains an engineering pilot and is excluded from the scientific
gate. The previously prepared 80-epoch pair is superseded before either member
starts; its scheduler cannot be cleanly extended after epoch 80 without
changing the stateful-resume contract.

## Locked pair

- Control: `p4-campp-known446-ood-channelrobust-paired-control-long120-oof-f0`
- Treatment: `p4-campp-known446-ood-channelrobust-consistency-c01-long120-oof-f0`
- Source Raw checkpoint: `checkpoints/p3-campp-known446-ood-channelrobust-oof-f0/campp_best_raw.pt`
- Source SHA256: `a46715e603173201a35bf20d9b43f6ad27f0352561b4c834ce7a2b3a3ae67a06`
- Control config SHA256: `88eed2d8f3ab1a4e37f72ae1955ded78887e84332308fc965c66b777cae0b5e1`
- Treatment config SHA256: `823891d4aa396b02d21563efc487acbe71f3bcff84572b96eb8a2d1554826f77`
- Fold/split: `kfold/folds3/fold0/seed42`
- Fixed horizon: 120 epochs; metric early stopping disabled
- Diagnostic milestones: epochs 40 and 80; neither milestone may stop or
  alter the recipe
- Selection: Raw probability-average with argmax
- Only scientific difference: `training.loss.consistency.enabled`
- Consistency term: cosine, fixed weight 0.1, identical clean/aug speech crop

The learning rates, cosine schedule, augmentation, model, batch size, windows,
loss weights, seed, and deterministic policy are identical between branches.
The longer scheduler is chosen before observing either long-run result.

## Runtime and budget

- Control timeout: 8 hours; maximum incremental cost $1.40
- Treatment timeout: 15 hours; maximum incremental cost $2.60
- Pair ceiling: $4.00
- Immediate stops only: NaN, provenance mismatch, artifact corruption, or a
  second OOM
- The campaign-wide $20 ceiling remains binding and is checked before each run

## Evidence and gate

Epoch-40 and epoch-80 checkpoints are diagnostics only. The decision uses the
full terminal epoch-120 receipts, contiguous histories, bound Raw OOF bundles,
class maps, split provenance, and MLflow artifacts. The final 20 epochs are
audited as two locked 10-epoch windows (101–110 and 111–120), including Raw
Macro-F1 slope, Known Accuracy, OOD-F1, and embedding-spread collapse checks.

The candidate is accepted only if all existing preregistered requirements
hold: treatment LME-20 gain over its matched control at least +0.002; fixed
50/50 probability-evidence fusion gain over the externally validated CAM++
Fold-0 baseline at least +0.002; Known and OOD-F1 drops each no worse than
-0.001; CAM++ baseline-error rescue rate at least 20%; and embedding-spread
ratio at least 0.95. Passing authorises analysis only, never automatic later
folds or leaderboard tuning.
