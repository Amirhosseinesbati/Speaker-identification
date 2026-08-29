# CAM++ channel-robust continuation — conditional preregistration

Date: 2026-08-29  
Status: **conditional; written while the source Fold-0 run is active at epoch 36**

## Motivation and boundary

The active `p3-campp-known446-ood-channelrobust-oof-f0` treatment is harder
than Control because it exposes the encoder to substantially more channel,
noise, reverberation and mild codec variability.  Post-unfreeze learning has
not plateaued by epoch 36: Raw reached `0.9251`, essentially matching Control's
matched epoch, and EMA continues to rise.  This makes an automatic rejection at
an arbitrary early epoch scientifically weak.  It does **not** imply that more
epochs must help; strong augmentation can also leave augmentation residuals.

The active process, config and checkout must remain unchanged.  This document
only defines when a separate state-preserving continuation may be launched
after the source supervisor reaches a terminal state.

## Eligibility gate fixed before terminal evidence

A continuation is allowed only when all provenance/artifact checks pass, no
NaN or repeated OOM occurred, and the source ended because of its six-hour
timeout or metric patience rather than corruption.  In addition, at least two
of these three trend tests over complete source epochs must pass:

1. The selected Raw probability-average checkpoint lies within the final eight
   complete epochs.
2. Ordinary-least-squares slope of EMA Macro-F1 over the final ten complete
   epochs is at least `+0.0002` per epoch.
3. Mean training loss over the final five epochs is at least `0.05` below the
   preceding five-epoch mean, while mean validation loss worsens by no more
   than `0.03`.

The source must also have best Raw probability-average Macro-F1 at least
`0.93` by epoch 60.  These rules are evaluated once, mechanically, after the
source terminates; their constants may not be changed using Fold 0.

## Continuation treatment

- Resume the selected **Raw** source checkpoint with its model, optimiser and
  scheduler states.  Preserve the source epoch offset and append to, rather
  than replace, its history.
- If the source's latest checkpoint lacks a scheduler state, use the selected
  best-Raw checkpoint, which already stores both optimiser and scheduler.  Do
  not silently reconstruct or reset either state.
- Keep architecture, Fold-0 split, class map, augmentation probabilities,
  objective, batch composition, crop/window policy, encoder/head learning
  rates, unfreeze scope and Raw LME20 decision policy identical.
- EMA may be restarted only as a labelled diagnostic because its shadow state
  is not currently present in the resumable Raw checkpoint.  EMA cannot select
  the final model or satisfy a gate.
- Permit at most three additional wall-clock hours and `$0.55` incremental
  cost.  The campaign total remains capped at `$20`.

Immediate stops remain NaN, provenance mismatch, artifact corruption or a
second OOM.  During continuation, stop after 12 consecutive epochs without a
new Raw probability-average best.  No learning rate, augmentation probability,
checkpoint, patience or epoch count may be tuned from the leaderboard.

## Scientific decision after continuation

The candidate is still judged by the original terminal contract:

1. standalone LME20 no more than `0.010` below Control LME20 `0.9611456663`;
2. fixed 50/50 probability-evidence fusion gains at least `+0.002` over
   Control LME20;
3. Known Accuracy and OOD-F1 each drop by at most `0.001`;
4. at least `20%` of Control LME20 errors are rescued; and
5. split, class-map, checkpoint, combined history, receipt and MLflow
   provenance validate.

Continuation failure rejects this augmentation recipe.  Passing only permits
analysis and a separate Fold-1/2 preregistration; it never authorises a
leaderboard submission.

## Implementation readiness

Commit `ad9c570` implements a separate `training.resume_checkpoint` path that
restores model, optimizer and scheduler state, continues global epoch numbers,
preserves/truncates a contiguous source history at the selected checkpoint,
and hard-fails on class-map or scientific-contract changes.  A separately
specified latest `.pt` checkpoint can supply the complete terminal history;
both the resumed model checkpoint and history source are hashed in the resume
receipt.  EMA is explicitly restarted from Raw weights and remains diagnostic.
The implementation also writes scheduler state and the complete current epoch
into all future latest/best checkpoints.  The focused training/data/audit
regression suite passed `50/50` tests.

The live source checkpoint was inspected without modifying the worker.  Its
epoch-42 Raw best contains optimizer state and scheduler state with
`last_epoch=42`; because the source run predates the implementation, its
embedded history ends at epoch 41.  Its separately updated latest checkpoint
therefore remains the mandatory history source if that epoch is selected for a
continuation.  The worker checkout must not pull `ad9c570` until the source
supervisor is terminal.

### Live evidence update — epoch 56

The source run remained active and produced a new Raw probability-average best
of `0.9338324913327573` at epoch 56, superseding the earlier epoch-42 best.
The selected Raw checkpoint was read without modifying the worker: it is tagged
`weight_variant=raw`, stores optimizer state, stores scheduler state with
`last_epoch=56`, and has SHA256
`c11f886f8635df26ef09dd48e8de880ebce553a8e20bd1f609522d5fff28fcbb`.
Its embedded history ends at epoch 55 due to the source commit's save ordering,
so the terminal `campp_latest.pt` remains the mandatory external history source
for any continuation.  MLflow Run `557c7d5542e64cb6b21637e2d69a1f14`
was independently verified `RUNNING` with 56 contiguous points for the four
queried core metrics.  This update changes no gate constant or treatment; it
only records that additional time yielded a genuine new Raw best and reset the
source patience counter.

### Live evidence update — epoch 65 gate preview

The source remained healthy and active through complete epoch 65.  A
non-terminal preview of the already locked mechanical continuation evaluator
used `campp_latest.pt` as the contiguous history source and the unchanged
epoch-56 Raw checkpoint as the selected model.  The by-epoch-60 Raw floor
passed (`0.9338324913327573 >= 0.93`).  Two of the three trend tests already
passed: the final-ten EMA OLS slope was `+0.0003928554` per epoch, and mean
training loss fell by `0.1051542` between the preceding and final five-epoch
windows while validation loss rose by only `0.0115912`, within the locked
`0.03` guardrail.  The selected checkpoint was no longer inside the final-eight
window, so that third trend test failed.

The preview intentionally supplied a non-eligible `complete` termination and
therefore returned `eligible=false`; the source is not terminal and no
continuation decision has been made.  If the same evidence is present after an
allowed patience or six-hour timeout termination, the locked two-of-three
trend requirement would be satisfied.  No gate constant, model, learning rate,
augmentation probability, stopping rule or active worker checkout changed.

### Live evidence update — epoch 70 new Raw best

Five additional complete epochs produced another genuine Raw improvement.
Epoch 70 reached probability-average Macro-F1 `0.9342659636973467`, exceeding
the epoch-56 selected value `0.9338324913327573`; EMA simultaneously reached
`0.9339465636520022`.  The active patience counter therefore reset again rather
than terminating the harder treatment.  At the matched epoch, candidate Raw
remained `0.0038122818` below Control, while candidate training loss was
`0.1364814` lower and validation loss only `0.0045074` higher.  This remains
evidence of slower adaptation, not terminal superiority.

The newly selected checkpoint was read independently from the active worker.
It is tagged `weight_variant=raw`, stores optimiser state, stores scheduler
state with `last_epoch=70`, and has SHA256
`c52b0462f3dfebbfb387d6348c9c3aed7e372f2fd98417600d1b9b9049dc0b7f`.
Its embedded history ends at epoch 69 due to the source save ordering, so the
terminal latest checkpoint remains the mandatory external history source.
MLflow Run `557c7d5542e64cb6b21637e2d69a1f14` was independently verified
`RUNNING` with 70 contiguous points for all four queried core metrics.  This
update changes no continuation constant, terminal gate or worker checkout.

### Live evidence update — epoch 73 gate preview

The source remained healthy and active through complete epoch 73.  Raw
probability-average Macro-F1 was `0.9327862045982851`, while diagnostic EMA
was `0.9344044471561251`; the epoch-70 Raw checkpoint therefore remained the
selected model.  At matched epoch 73 the candidate was only `0.0017522979`
below Control Raw, with training loss `0.1166554` lower and validation loss
only `0.0051964` higher.  EMA OOD-F1 also increased to `0.9455`, so the late
adaptation signal had not collapsed even though no newer Raw best appeared.

A second non-terminal execution of the locked continuation evaluator passed
all three trend tests: the epoch-70 selected Raw checkpoint was within the
final-eight window, the final-ten EMA OLS slope was `+0.0003315311` per epoch,
and mean training loss improved by `0.0700048` between the preceding and final
five-epoch windows while validation loss rose by `0.0172550`, below the locked
`0.03` guardrail.  The by-epoch-60 Raw floor remained satisfied at
`0.9338324913327573`.  MLflow independently contained 73 contiguous points
for each queried core metric, and the selected checkpoint SHA256 remained
`c52b0462f3dfebbfb387d6348c9c3aed7e372f2fd98417600d1b9b9049dc0b7f`.

The preview still returned `eligible=false` solely because `complete` is not
an allowed terminal reason; only an actual patience or six-hour timeout may
trigger the stateful continuation.  This update strengthens the preregistered
case for additional time without changing any threshold, budget, stopping
rule, active worker checkout or downstream treatment.

### Live evidence update — epoch 75 new Raw best

Two more complete epochs produced consecutive Raw improvements.  Epoch 74
reached `0.9357969937539253`, and epoch 75 then established a new selected Raw
probability-average best of `0.9361443624241722`; its diagnostic logit average
was `0.9344`.  Training loss fell to `1.572686648702827`, which was `0.1216758`
below matched Control epoch 75, while validation loss was only `0.0077095`
higher.  The active patience counter therefore reset at epoch 75.  MLflow Run
`557c7d5542e64cb6b21637e2d69a1f14` independently contained 75 contiguous
points for each queried core metric.

The new selected Raw checkpoint is tagged `weight_variant=raw`, stores
optimizer state, stores scheduler state with `last_epoch=75`, and has SHA256
`6570e3661c390bd4a1777ace58a13c3b28c3fb31571d8af480acf9774a149bcc`.
Its embedded history ends at epoch 74 because the source commit still predates
the save-order fix, so terminal `campp_latest.pt` remains the mandatory full
history source.

The locked non-terminal continuation preview still passed the required two of
three trend tests: the selected checkpoint was inside the final-eight window,
and mean training loss improved by `0.0796404` while validation loss rose by
only `0.0173507`, within the fixed `0.03` guardrail.  The final-ten EMA slope
was `+0.0001978008` per epoch, just `0.0000021992` below its individual
`0.0002` threshold, so that one test did not pass.  The by-epoch-60 Raw floor
remained satisfied.  As before, `eligible=false` at this non-terminal preview
only because `complete` is not an allowed terminal reason; the underlying
two-of-three requirement was met.  No gate constant, budget, active checkout
or treatment changed.

### Live evidence update — epochs 78–79

The source remained healthy after the epoch-75 Raw selection.  Although Raw
probability-average fluctuated at epochs 76–78, diagnostic EMA continued to
establish new late-stage highs: `0.9361222766526638` at epoch 76,
approximately `0.9367` at epoch 77, and `0.9379` at epoch 78.  At epoch 78 the
EMA Known accuracy and OOD-F1 were `0.9462` and `0.9479`, respectively.  This
is consistent with a slowly adapting representation under the harder channel
augmentation treatment and is not evidence that the selected Raw decision
rule has already beaten the control.

The latest checkpoint contained a contiguous 79-epoch history when inspected.
A non-terminal rerun of the locked continuation evaluator passed all three
trend tests: the epoch-75 selected Raw checkpoint was inside the final-eight
window, the final-ten EMA OLS slope was `+0.0004305290` per epoch, and mean
training loss improved by `0.0669493` between the preceding and final
five-epoch windows while validation loss rose by only `0.0120992`.  The
by-epoch-60 Raw floor remained satisfied.  The preview returned
`eligible=false` solely because the source was still running and `complete`
is intentionally not an allowed terminal reason.  If an actual patience or
six-hour timeout termination preserves the locked provenance and safety
conditions, this evidence mechanically authorizes the preregistered stateful
continuation of at most three additional hours and `$0.55`; it does not alter
the terminal scientific acceptance gate.

## Literature rationale

- Curriculum learning for speaker verification reports benefits from gradually
  increasing augmentation difficulty rather than assuming the hardest regime
  converges immediately: https://arxiv.org/abs/2203.14525
- The NIST SRE21 overview identifies augmentation and long-duration fine-tuning
  among the ingredients of strong mismatched-domain systems:
  https://doi.org/10.21437/Odyssey.2022-45
- Adversarial data augmentation warns that vanilla augmentation can leave
  transformation-specific residuals, which is why the eligibility test is
  trend-gated rather than an unconditional extension:
  https://doi.org/10.1145/3638884.3638917
