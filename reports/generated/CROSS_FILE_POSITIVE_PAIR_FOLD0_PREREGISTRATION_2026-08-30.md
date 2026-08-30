# CAM++ cross-file positive-pair Fold-0 preregistration — 2026-08-30

## Status and activation boundary

This document fixes the scientific contract before any cross-file outcome is
observed.  It does not create a configuration, alter the worker checkout or
authorise a concurrent Run.  The candidate becomes eligible only after the
active same-crop Long-120 pair is terminal and one of these mutually exclusive
branches is established:

1. if same-crop consistency passes its complete locked gate, keep this
   candidate dormant;
2. if same-crop consistency is neutral or harmful while the treatment/control
   embedding-spread ratio remains at least `0.95`, activate this matched
   cross-file test;
3. if the same-crop treatment is still improving at epoch 120 and passes every
   predeclared matched-extension check, extend both same-crop branches instead;
4. if same-crop consistency loses spread or exhibits late degradation, retire
   invariance-pressure variants and do not activate this candidate.

No milestone, selected checkpoint or leaderboard result may rewrite these
branches.  Only one campaign Run may be active at a time.

## Hypothesis

The dominant residual error after the externally validated LME20 package is
Known-to-Unknown (`87` files), not Unknown-to-Known (`25` files).  A clean and
augmented view of one crop can retain recording identity.  Two different files
from the same known speaker provide a closer available proxy for session/channel
variation and may reduce intra-speaker spread without weakening the binary OOD
boundary.

SciSpace retrieval identified SSPS (Interspeech 2025, DOI
`10.21437/Interspeech.2025-183`) as direct speaker evidence that positives from
different recordings reduce intra-speaker variance relative to same-utterance
positive sampling.  A second objective-specific retrieval found no controlled
study matching the complete local contract of ArcFace, a positive-only
cross-file cosine term and joint 446-known/binary-OOD guardrails.  Therefore no
clustering, memory queue, contrastive negatives, adaptive margin, centroid loss,
extra teacher or paper-derived threshold is imported.

A follow-up SciSpace query found speaker evidence for symmetric contrastive
sampling, angular margins and explicit inter-class separability, but did not
return a reusable fixed coefficient or training schedule for a positive-only
cross-utterance term beside an existing ArcFace classifier.  Consequently
`0.1` below is not claimed as a literature-optimal weight: it is retained only
to make pair identity the single changed variable relative to the active
same-crop consistency experiment.  No coefficient or schedule search is
authorised from Fold-0 outcomes.

## Exact paired recipes

Both branches will be created only after activation and will use:

- profile family `p5-campp-known446-ood-crossfile-paired-long120-oof-f0`;
- CAM++ warm-started from the immutable selected P3 Raw checkpoint
  `checkpoints/p3-campp-known446-ood-channelrobust-oof-f0/campp_best_raw.pt`,
  SHA256 `a46715e603173201a35bf20d9b43f6ad27f0352561b4c834ce7a2b3a3ae67a06`;
- split `kfold/folds3/fold0/seed42`, duplicate cleaning and the existing
  446-known plus binary-OOD target contract;
- the locked channel-robust augmentation, 8 windows, batch size 48, OOD ratio
  `0.5`, deterministic algorithms and seed 42;
- ArcFace margin `0.4`, scale `30`, label smoothing `0.05`, speaker/OOD weights
  `0.85/0.15`, last two CAM++ blocks open, encoder LR `5e-6`, head LR `1e-4`;
- cosine schedule, warmup ratio `0.05`, minimum-LR ratio `0.05`, EMA diagnostic
  only, fixed horizon 120 and metric early stopping disabled; and
- diagnostic milestones 40 and 80 that cannot select, stop or modify a branch.

The shared sampler emits 24 ordinary OOD rows and 12 known-speaker pairs in
every batch.  Each pair contains two distinct `audio_file` values from one
known speaker.  Both files retain all ordinary supervised gradients.  OOD rows
never receive a fabricated positive.

The exact branch difference is one Boolean:

- matched sampler control: `consistency.enabled=false`;
- cross-file treatment: `consistency.enabled=true`.

Both retain `consistency.type=cosine`, `consistency.pairing=cross_file_batch`
and the declared coefficient `0.1`; the treatment uses the first shuffled file
as the student and the second as a detached target.  The deterministic sampler
must be tested to rotate every known file through both roles across the fixed
horizon.  Raw config hashes and a structural single-difference test are
mandatory before either launch.

## Immediate stop rules and budget

Stop only for NaN/non-finite values, split or provenance mismatch, artifact
corruption, a second OOM, a missing/distorted known pair, or failure of the
declared OOD/known batch composition.  A weak metric is not an early-stop rule.

Each branch has an 8-hour timeout and a maximum incremental budget of `$1.40`;
the pair is capped at `$2.80`, subject also to the campaign-wide `$20` ceiling.
If the remaining campaign budget cannot cover both branches and their terminal
audits, launch neither.  Treatment is never run without its matched sampler
control.

## Terminal evaluation and immutable gate

Canonical decisions use selected Raw probability-average plus direct argmax.
Logit-average and EMA are diagnostic only.  The selected checkpoint of each
branch is audited with the fixed LME20/PCM backend; no threshold, coefficient,
epoch, blend or submission is tuned on OOF or leaderboard outcomes.

The treatment passes Fold 0 only if all conditions hold:

1. selected-treatment LME20 Macro-F1 minus matched-sampler-control LME20
   Macro-F1 is at least `+0.002`;
2. the one declared fixed `50/50` probability-evidence fusion with the external
   CAM++ Control Fold-0 anchor gains at least `+0.002` Macro-F1 over that anchor;
3. fusion Known Accuracy and OOD-F1 each decline by no more than `0.001`;
4. the treatment rescues at least `20%` of the anchor's errors;
5. treatment/control deterministic train-embedding spread is at least `0.95`;
6. OOF, class map, split, checkpoint and receipt hashes are valid; and
7. complete Raw/EMA/logit, Known/OOD, loss, pair-cosine and spread trajectories
   show no non-finite value, representation collapse or provenance gap.

A fixed-seed paired randomisation statistic is descriptive and cannot override
the effect-size or guardrail gate.  Failure of any item rejects this recipe and
forbids automatic Fold-1/Fold-2 expansion.  Passing Fold 0 authorises only a
separate leakage-safe multi-fold preregistration; it does not authorise a
submission or leaderboard probe.

## Readiness evidence

The exact Fold-0 audit found all 446 known training speakers pairable after
duration filtering and duplicate cleaning, with `2/3/5` minimum/median/maximum
distinct files.  The implementation reuses already-computed supervised
embeddings and adds no second forward pass.  Targeted tests cover deterministic
speaker exposure, distinct files, invalid contracts, known-only loss,
stop-gradient and full file/gradient-role rotation across an extended horizon.
See `CROSS_FILE_PAIRING_READINESS_2026-08-30.md` for the feasibility receipt.
