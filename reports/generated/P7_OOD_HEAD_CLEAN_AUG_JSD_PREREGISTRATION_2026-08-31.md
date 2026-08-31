# P7 OOD-head clean/aug JSD preregistration — 2026-08-31

## Status and ordering

`DORMANT FALLBACK / DO NOT RUN BEFORE P6 TERMINAL DECISION`

P7 is a single, fixed transfer hypothesis for the existing CAM++ model. It may
be activated only if P6 fails its locked Fold-0 gate, the P6 receipt is complete,
no other run is active, and the campaign has at least `$1.10` remaining. It is
not a coefficient search, an alternative interpretation of P4, or permission to
run later folds automatically.

## Why another clean/aug experiment is not a repeat of P4

P4 already tested cosine consistency between clean and augmented **speaker
embeddings** for 120 matched epochs. It improved Fold-0 Macro-F1 by only
`+0.0009816106` over its matched control, remained `-0.0028155228` below the
immutable external CAM++ reference, rescued only `10.42%` of that reference's
errors, and failed the Known/OOD/fusion gates. Its terminal extension diagnostic
also failed. P4 is rejected and must not be rerun with a different coefficient.

The residual error topology points to a different bottleneck. The immutable
CAM++ Fold-0 reference has 24 known-to-unknown, 13 known-to-wrong-known and 11
unknown-to-known errors. The later P5 treatment has 37 known-to-unknown but only
4 known-to-wrong-known errors. The dominant failure is therefore the binary OOD
boundary, not insufficient same-speaker embedding invariance or inter-speaker
separation. P7 regularises only the binary OOD predictive distribution and
freezes the speaker representation path.

## Literature transfer and limits

SciSpace semantic search was run with two full methodological questions. It
retrieved primary speaker-recognition work showing that clean/noisy paired
training can improve robustness when the noisy representation is constrained by
its clean counterpart:

- Cai, Cai and Li, ICASSP 2020, *Within-sample variability-invariant loss for
  robust speaker recognition under noisy environments*,
  <https://arxiv.org/abs/2002.00924>.
- MohammadAmini et al., Odyssey 2022, *Learning Noise Robust ResNet-Based
  Speaker Embedding for Speaker Recognition*,
  <https://www.isca-archive.org/odyssey_2022/mohammadamini22_odyssey.html>.

SciSpace also retrieved Suzuki and Matsuzawa's 2021 JSD consistency study. The
primary paper uses Jensen-Shannon divergence among the hard target and two
augmented predictive distributions, with the fixed AugMix multiplier `12`, and
reports a better in-distribution/OOD trade-off on most evaluated image OOD
sets: <https://doi.org/10.15344/2456-4451/2021/165>.

These papers do not establish efficacy for this competition's binary speaker
OOD head. The domains, architectures and metrics differ, and the JSD paper also
reports exceptions. P7 therefore transfers exactly one coefficient and treats
failure as rejection; it must not trigger a Fold-0 weight grid.

## Locked matched pair

- Control profile:
  `p7-campp-known446-ood-cleanaug-oodjsd-control-lmft-oof-f0`.
- Treatment profile:
  `p7-campp-known446-ood-cleanaug-oodjsd-w12-lmft-oof-f0`.
- Source model: selected Raw CAM++ Fold-0 checkpoint at epoch 112,
  `checkpoints/p0-campp-known446-ood-control-oof-f0/campp_best_raw.pt`.
- Source SHA256:
  `f50f67f549b913b57111043b43daca1ff8bcbbf49bebe5dccab91ade8b19ae0d`.
- Split: `kfold/folds3/fold0/seed42`.
- Decision rule: Raw probability-average, LME20, direct argmax.
- Encoder and speaker head are frozen and kept in evaluation mode. Only the
  existing binary OOD head is trainable.
- The OOD head also remains in evaluation mode while its weights receive
  gradients, disabling its `0.7` dropout for both views. Thus the measured
  clean/aug discrepancy is caused by the paired audio transformation rather
  than two unrelated dropout masks; this mode is identical in both arms.
- Optimizer, scheduler and EMA are reset; the model weights alone are warm
  started. The control and treatment use identical batches, paired crops,
  augmentations, windows, seed, OOD sampling, optimizer and stopping policy.
- The sole trainable parameter group uses AdamW with learning rate `5e-5`,
  weight decay `1e-4`, the existing cosine schedule, warmup ratio `0.05` and
  minimum-LR ratio `0.05`. These values are transferred unchanged from the P2
  head group and are not selected from P7 results.
- Both branches compute and log the same clean/aug JSD. The single scientific
  difference is its multiplier: `0` in the control and `12` in the treatment.
- The existing OOD branch weight `0.15` remains outside the auxiliary term, so
  the effective contribution is `0.15 * 12 * JSD` in the treatment.
- No threshold, temperature, class prior, blend, epoch or post-processing
  parameter may be selected from Fold 0 or the leaderboard.

For an augmented OOD logit `z_a`, a no-gradient clean-view logit `z_c`, and
binary target `y`, define Bernoulli distributions

`P_a=[1-sigmoid(z_a), sigmoid(z_a)]`,
`P_c=[1-sigmoid(z_c), sigmoid(z_c)]`, and `P_y=[1-y,y]`.

With `M=(P_a+P_c+P_y)/3`, the raw auxiliary loss is

`JSD=(KL(P_a||M)+KL(P_c||M)+KL(P_y||M))/3`.

Probabilities are clamped only for finite logarithms. `P_c` and `P_y` are
detached; gradients flow through the augmented OOD prediction only. The
ordinary augmented-view BCE remains the supervised anchor.

## Horizon, early stopping and budget

- Maximum: 60 epochs.
- Raw probability-average Macro-F1 is the sole selection/stopping metric.
- Patience begins after epoch 20.
- Patience: 12 complete epochs without a strict Raw improvement.
- Immediate stop: NaN/Inf, provenance mismatch, non-finite JSD, artifact
  corruption, or second OOM.
- Per-arm cap: 3 hours / `$0.52`.
- Complete pair cap: 6 hours / `$1.04`; activation requires at least `$1.10`
  available to preserve rounding headroom.

The delayed start gives the warm-started OOD head time to adapt under its reset
optimizer/scheduler while avoiding another unjustified 120-epoch tail. Passing
early is not allowed to terminate a branch; only the locked patience or
immediate safety rules may do so.

## Acceptance and mechanism gates

P7 passes Fold 0 only if every condition holds:

1. Treatment Raw LME20 Macro-F1 is at least `+0.002` over its source-matched
   control and at least `+0.002` over the immutable external CAM++ reference.
2. The gain threshold is met at two or more complete epochs, not one isolated
   peak.
3. Known Accuracy and OOD-F1 each decline by no more than `0.001` against both
   controls.
4. Known-to-unknown errors fall by at least `20%` from the immutable reference
   (24 to at most 19), while unknown-to-known errors do not increase.
5. The mean absolute clean/aug OOD-probability gap on a fixed deterministic
   audit set is at most `0.80` times the matched-control gap.
6. Speaker logits from the source and selected treatment are byte-for-byte
   identical on the deterministic audit set, proving that the frozen speaker
   path did not drift.
7. Fixed 50/50 probability fusion with the immutable external CAM++ reference
   gains at least `+0.002` Macro-F1 and respects both guardrails. No other fusion
   weight is evaluated.
8. Receipt hashes, OOF uniqueness, split/class-map identity, complete histories,
   MLflow series and all selected artifacts are mutually consistent.

Failure of any condition rejects P7 and forbids folds 1/2. Passing authorises
only a separately preregistered multi-fold replication; it is not automatic
submission permission and cannot change the leaderboard-independent decision
policy.
