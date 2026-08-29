# Speaker channel-mismatch literature ledger — 2026-08-29

## Purpose

This ledger records which literature findings change the IAAA campaign decision
tree.  Papers motivate hypotheses; locked OOF evidence and Known/OOD guardrails
remain authoritative.  Leaderboard results are external anchors and are never
used to choose thresholds, fusion weights, epochs, or augmentation strength.

## Evidence mapped to local decisions

| Evidence | Relevant finding | Local evidence | Campaign decision |
|---|---|---|---|
| VoxWatch open-set benchmark ([DOI](https://doi.org/10.48550/arXiv.2307.00169)) | Adaptive score normalization is not guaranteed to help open-set speaker identification, while calibration and genuinely complementary score fusion can help. | AS-Norm was rejected by locked OOF; fixed LME20/PCM transferred from `0.9633564052` OOF to `0.9667174285` leaderboard Macro-F1. | Keep LME20/PCM as the operational baseline.  Do not revive score normalization without a new, leakage-free hypothesis. |
| Comprehensive speaker augmentation study ([DOI](https://doi.org/10.21437/Interspeech.2024-2478)) | Speed and vocal-tract perturbations can create speaker variation rather than merely preserve channel variation. | The residual error topology is dominated by Known/OOD boundary errors, so identity-corrupting augmentation can increase known-speaker rejection. | In the active treatment, pitch shift is disabled and time stretch is rare and restricted to `0.95..1.05`. |
| Noise/reverberation robustness via training ([DOI](https://doi.org/10.1109/EISIC.2015.20)) and multi-channel training ([DOI](https://doi.org/10.21437/INTERSPEECH.2019-1437)) | Training on additive noise and RIR/channel variability can reduce mismatch degradation. | MUSAN and RIR assets are already local to the worker; no data download or provenance expansion is required. | Test one single-variable CAM++ Fold-0 treatment with stronger speaker-preserving MUSAN/RIR exposure. |
| Domain-weighted low-resource transfer ([DOI](https://doi.org/10.1186/s13636-024-00385-z)) | Uncontrolled fine-tuning under domain mismatch can degrade performance; domain-weighted adaptation can be safer. | The fixed-30 CAM++ LMFT experiment did not beat its warm-start checkpoint. | Do not repeat generic LMFT.  A future adaptation experiment must define a domain proxy, weighting rule, held-out gate, and stop rule before training. |
| Curriculum learning for speaker verification ([arXiv](https://arxiv.org/abs/2203.14525)) and the NIST SRE21 overview ([DOI](https://doi.org/10.21437/Odyssey.2022-45)) | Gradually increasing augmentation difficulty can improve speaker representations, and long-duration fine-tuning was part of strong mismatched-domain systems. | The active channel-robust treatment is deliberately harder than Control; after unfreezing, its EMA and Raw curves are still rising rather than plateauing. | Do not stop the active run merely because early post-unfreeze epochs lag.  Let its preregistered patience/timeout act, then use the terminal slope and saved optimiser/scheduler state to decide whether a state-preserving continuation is scientifically warranted. |
| Adversarial data augmentation ([DOI](https://doi.org/10.1145/3638884.3638917)) and ASVspoof5 augmentation ablations ([paper](https://www.isca-archive.org/asvspoof_2024/xie24_asvspoof.pdf)) | Vanilla augmentation can leave augmentation-specific residuals; overly broad or overly frequent transforms do not improve every task, while moderate frequency masking can outperform stronger combinations. | The current treatment uses several simultaneous channel transforms and has not yet beaten Control terminal performance. | Extra epochs are evidence collection, not a presumption that more is always better.  If terminal performance fails, prefer a preregistered curriculum or single-factor ablation over increasing all augmentation probabilities again. |
| Extended variability modelling ([DOI](https://doi.org/10.21437/INTERSPEECH.2017-1586)) and open-set mismatch work ([DOI](https://doi.org/10.21437/Interspeech.2009-395)) | Duration/session/channel-aware backends and condition-adjusted normalization can help under mismatch. | Direct session/channel metadata is absent; quality-aware and normalization candidates already violated local guardrails. | Treat condition-aware prototype banks as a backup research direction, not an authorised run.  Any parameters must be selected leave-one-fold-out from training folds only. |
| Contrastive adversarial domain adaptation ([DOI](https://doi.org/10.1109/TNNLS.2020.3044215)) | Separating speaker-discriminative and domain-invariant objectives can improve mismatched speaker recognition. | No reliable channel labels exist and this is a multi-component architectural change. | Keep as a later high-cost hypothesis only after cheaper channel augmentation and prototype conditioning are resolved. |

## Active experiment and interpretation boundary

The active profile is `p3-campp-known446-ood-channelrobust-oof-f0`, commit
`eb433629b85a07a0665056d1bf6fcd84694cf1ca`.  It changes only the augmentation
policy relative to Control Fold 0.  Its early epoch-8 advantage did not persist
through the frozen-encoder phase, but the post-unfreeze trajectory is converging
toward Control.  At matched epoch 33, treatment probability-average Macro-F1
was `0.9210` versus Control `0.9222834` (about `-0.0013`), while logit-average
was slightly higher (`0.9160` versus `0.9157215`).  Treatment EMA Macro-F1 was
`0.9106` versus Control `0.9073612` (about `+0.0032`), and training loss was
lower (`3.1711` versus `3.3714`); validation loss remained nearly equal but
slightly higher (`1.1514` versus `1.1499`).  This is evidence of continuing
adaptation, not a terminal win: Raw is still oscillatory, and neither selected
checkpoint nor LME20 complementarity is known.  The run remains healthy and
active, so checkpoint selection and downstream gates stay locked until terminal
evaluation.  The epoch-40 futility rule only rejects a best Raw score below
`0.90`; it is not a reason to stop this trajectory after that threshold has
already been exceeded.

The treatment still passes only if the preregistered standalone LME20,
fixed-50/50 complementarity, Known/OOD, rescue-rate, and provenance gates all
pass after the run.  Failure rejects the candidate and forbids automatic Fold
expansion.

The later matched-epoch observation strengthens the case for letting the
harder treatment converge.  At epoch 42, treatment probability-average
Macro-F1 reached a new best of `0.9335` versus Control `0.9282921` (about
`+0.0052`), and treatment logit-average was `0.9277` versus Control
`0.9223749` (about `+0.0053`).  Training loss was lower (`2.4337` versus
`2.5685`) while validation loss remained nearly matched (`1.1946` versus
`1.1920`).  EMA still lagged (`0.9225` versus `0.9268546`), which is consistent
with a slow shadow-model response but prevents calling epoch 42 a broad or
terminal win.  Agreement of both Raw aggregation rules is nevertheless a
material signal that the channel-robust representation may now be overtaking
Control; the selected checkpoint, LME20 and complementarity gates remain
locked until the source run is terminal.

## Decision-selector oracle ceiling

The locked three-Fold residual audit contains `4447` OOF files and `131` LME20
errors.  Relative to the raw head, LME20 corrected `120` head errors while the
head was correct on only `14` LME20 errors.  Therefore an impossible oracle
that chooses perfectly between the head and LME20 can save at most 14 additional
files, an accuracy ceiling of `14 / 4447 = 0.00314819`; this is not a guaranteed
Macro-F1 gain.  A learned selector would also need to preserve the 120 LME20-only
rescues.  The error-feature distributions overlap: mean fused confidence is
`0.4372` for Known-to-Unknown and `0.4302` for Unknown-to-Known errors.

This evidence deprioritises another generic head/LME selector or global
threshold sweep.  The remaining high-value hypothesis class is representation
or condition modelling that changes the residual topology itself.  Any later
selector must first demonstrate cross-fit separability and must remain subject
to simultaneous Known/OOD guardrails.

## Prototype-aggregation cross-fit update

The completed leave-one-fold-out aggregation audit tested 11 train-only
prototype aggregation rules.  The best fixed family under per-target-fold
selection was `logmeanexp_b20`: aggregate Macro-F1 `0.9629589450`, minimum
held-out Fold gain `+0.0136101890`, aggregate Known Accuracy delta
`+0.0017953321`, and OOD-F1 delta `+0.0200328352`.  This independently confirms
that log-mean-exp prototype evidence is stable across all three Folds.

It does **not** justify a new submission: the locked operational LME20/PCM OOF
score is `0.9633564052`, so the new cross-fit policy is lower by about
`0.00039746`.  The leaderboard is not used to choose another aggregation or
decision parameter.  Generic aggregation sweeps are therefore deprioritised;
their scientific value is confirmation of the LME mechanism, not a replacement
for the current record-setting package.

## Backup decision order

1. Finish and audit the active channel-robust Fold-0 treatment.
2. If it passes, preregister Fold 1/2 and require consistent three-Fold gain.
3. If it fails, analyse whether the failure is representation quality or lack
   of complementary errors; do not tune a fusion weight on Fold 0.
4. Because the generic aggregation family is now exhausted, the cheapest next
   candidate is a training-fold-only **condition-aware**
   prototype bank using predefined acoustic/channel proxies and leave-one-fold-
   out selection.  It is not authorised until its exact contract is written.
5. Domain-adversarial training is a later option only if the proxy/prototype
   route cannot explain the residual topology and the remaining budget permits
   a controlled ablation.
