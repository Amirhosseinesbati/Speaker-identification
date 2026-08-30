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

## Literature rationale and counter-evidence

The hypothesis is plausible but not assumed true. Chen, Guo, and Gu reported
speaker-verification gains from positive clean/augmented pairs with an added
contrastive objective ([Interspeech 2021](https://www.isca-archive.org/interspeech_2021/chen21f_interspeech.html)).
Sang et al. likewise reported substantial relative improvement from
positive-pair speaker regularisation with strong online augmentation
([arXiv:2112.04459](https://arxiv.org/abs/2112.04459)). Conversely, the TalTech
SdSV 2020 system found no significant benefit from its JSD consistency loss
and reported roughly 2.5x slower training, so it omitted that term from the
final system
([Interspeech 2020](https://www.isca-archive.org/interspeech_2020/alumae20_interspeech.pdf)).
More recent negative evidence is also directly relevant: Wang, Fang, and He
reported that adding a consistency loss on top of WavAugment changed EER from
5.21% to 5.35% (minDCF 0.55 to 0.54), and concluded that the extra term did not
improve their stage-1 speaker representation
([Interspeech 2024](https://www.isca-archive.org/interspeech_2024/wang24z_interspeech.html)).
The successful VoxSRC large-margin fine-tuning recipe is not evidence that
arbitrary extra epochs are sufficient either: its gains jointly used longer
training segments and a more aggressive angular margin
([arXiv:2010.11255](https://arxiv.org/abs/2010.11255)).

This mixed evidence motivates the long matched horizon: the treatment receives
enough optimisation time to compensate for its harder/slower objective, but it
must still beat a compute-matched control and pass collapse, Known, and OOD
guardrails. Runtime alone is not evidence of quality.

## Runtime and budget

- Control timeout: 8 hours; maximum incremental cost $1.40
- Treatment timeout: 15 hours; maximum incremental cost $2.60
- Pair ceiling: $4.00
- Immediate stops only: NaN, provenance mismatch, artifact corruption, or a
  second OOM
- The campaign-wide $20 ceiling remains binding and is checked before each run

### Runtime-policy compatibility discovered before treatment launch

The durable campaign state currently caps every supervisor Run at 12 hours,
while the treatment profile declares a 15-hour maximum.  The supervisor uses
the smaller value, so passing `--timeout-hours 15` without addressing policy
would silently impose a 12-hour ceiling.  This does not affect the active
eight-hour control and no live state is changed while that Run is active.

Before treatment launch, the required batch-48/eight-window GPU preflight must
project end-to-end completion with at least 20% wall-time headroom inside the
12-hour campaign ceiling.  If it passes, the stricter 12-hour ceiling remains
and a normal completion preserves the scientific pair.  If it does not pass,
the treatment must not be launched under a knowingly truncating ceiling.  An
atomic, evented and active-Run-forbidden policy-update command is available,
but raising the ceiling is a separate operational decision and may occur only
after the control is terminal, with the $2.60 treatment and $20 campaign cost
guards still satisfied.  A timeout is incomplete evidence, never a scientific
rejection of consistency.

The 20% rule is executable rather than judgement-based.  After the control is
terminal, run the existing real forward/backward batch-48 probe once for the
matched control and once for the treatment, on the same worker and checkout.
Then run `scripts/project_paired_training_runtime.py` with the terminal control
log, both probe JSON files, the authoritative control wall time, 120 treatment
epochs, the effective supervisor timeout, current hourly price and the $2.60
incremental cap.  The verifier parses every complete Train/Raw-Val/EMA-Val
timing sequence from the control log, charges validation/checkpoint overhead at
no less than the observed control rate, and scales only the training component
by the measured treatment/control throughput ratio.  Launch is forbidden
unless both `time_gate_pass` and `cost_gate_pass` are true.  Its JSON receipt is
part of treatment provenance; a failed projection is an operational no-launch,
not evidence against the scientific hypothesis.

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

The terminal audit additionally runs two fixed-seed paired Monte Carlo
prediction-swap tests (20,000 replicates; seed `20260830`) for the primary
447-class Macro-F1 delta: treatment versus matched control, and fixed fusion
versus external Control.  Their one- and two-sided p-values, null quantiles and
paired win/loss counts are uncertainty diagnostics only.  They cannot replace
the locked minimum effect size, rescue, Known/OOD or collapse guardrails.

The terminal report must also expose the fixed error topology for the external
Control, matched Control, treatment, and fixed fusion: `known_to_unknown`,
`known_to_wrong_known`, and `unknown_to_known`.  These counts are descriptive
diagnostics, not new gates.  In particular, they may explain whether the
representation changes the dominant Known-to-Unknown residual, but they may
not authorise a post-hoc threshold, blend weight, or alternate checkpoint.

## Conditional longer-horizon path

Epoch 120 is a minimum complete comparison, not an unconditional claim that
the harder consistency objective has converged.  A later experiment may use a
longer horizon only when the already-implemented matched-extension diagnostic
passes every one of the following locked checks:

- the treatment mean Raw Macro-F1 over epochs 111--120 exceeds its mean over
  epochs 101--110 by at least `0.0005`;
- the treatment Raw Macro-F1 slope over epochs 101--120 is positive;
- the treatment's best Raw epoch lies in epochs 111--120;
- the treatment's tail improvement exceeds the matched control's tail
  improvement by at least `0.0005`;
- treatment tail Known Accuracy and OOD-F1 each decline by no more than
  `0.002` relative to the preceding ten-epoch window; and
- treatment-to-control embedding-spread ratio remains at least `0.95`.

Passing these checks authorises preregistration of a **new matched extension**
only.  It never authorises continuing the treatment alone, changing the
consistency weight from Fold-0 feedback, or interpreting extra epochs as an
acceptance-gate pass.  The extension must preserve equal horizons and all
scientific settings for both branches; its scheduler/resume contract and cost
ceiling must be fixed before either extension branch starts.

## Control diagnostic at epoch 40

The first diagnostic milestone completed without intervention.  The Raw
checkpoint `campp_milestone_epoch040_raw.pt` is readable, is `69,185,377`
bytes, and has SHA256
`8e5edec9853f992d788deaa7d81e01de311596e286887a52fdb37faa142323c8`.
It binds epoch 40 to a 40-row history, scheduler `last_epoch=40`, 948 model
tensors, optimizer state, RNG state, split `fold0/folds3/seed42`, the fixed
120-epoch horizon, and milestones `[40, 80]`.  The worker remained on commit
`e1059813cc974f269c9e4ec717b1210786197731`; the on-worker config SHA256
remained
`88eed2d8f3ab1a4e37f72ae1955ded78887e84332308fc965c66b777cae0b5e1`.
The milestone intentionally snapshots the Raw `latest` checkpoint: validation
metrics live in its embedded history and EMA is diagnostic rather than
statefully resumed.  Their absence as top-level/EMA state fields is therefore
not artifact corruption.

At epoch 40, Raw probability-average Macro-F1 was
`0.9380598877266396`, logit-average was `0.9359668513192003`, Known
Accuracy was `0.9517937219730942`, and OOD-F1 was
`0.9508650519031142`.  EMA Macro-F1 was `0.9401857297300604` with Known
Accuracy `0.9506726457399103` and OOD-F1 `0.9487354750512645`.  The long
control's best Raw point through this milestone was epoch 34 at
`0.9424689925372944`, only `+0.0004976466519496` over the 40-epoch pilot's
best.  Conversely, the current matched delta was `-0.0013302587084196`, the
mean matched delta was `-0.0008078365415769`, the last-five delta was
`-0.0021131896081753`, and the last-ten slope was
`-0.0001506433522042` per epoch.  Thus the longer schedule has demonstrated a
slightly higher isolated peak but not a sustained advantage by epoch 40.

Operational integrity remained healthy: zero NaN, OOM, or traceback events;
all 33 MLflow metric series were contiguous at 40/40 points; the Run stayed
`RUNNING` with 22 parameters and five initial artifacts; GPU memory was
`7,276/24,576 MiB`, temperature `55 C`, and workspace use `63%`.  These mixed
scientific signals neither accept nor reject the control.  Per preregistration,
training continues unchanged through epoch 120 and epoch 80 remains diagnostic
only.
