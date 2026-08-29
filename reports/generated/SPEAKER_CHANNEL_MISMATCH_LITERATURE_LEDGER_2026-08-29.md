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
| Extended variability modelling ([DOI](https://doi.org/10.21437/INTERSPEECH.2017-1586)) and open-set mismatch work ([DOI](https://doi.org/10.21437/Interspeech.2009-395)) | Duration/session/channel-aware backends and condition-adjusted normalization can help under mismatch. | Direct session/channel metadata is absent; quality-aware and normalization candidates already violated local guardrails. | Treat condition-aware prototype banks as a backup research direction, not an authorised run.  Any parameters must be selected leave-one-fold-out from training folds only. |
| Contrastive adversarial domain adaptation ([DOI](https://doi.org/10.1109/TNNLS.2020.3044215)) | Separating speaker-discriminative and domain-invariant objectives can improve mismatched speaker recognition. | No reliable channel labels exist and this is a multi-component architectural change. | Keep as a later high-cost hypothesis only after cheaper channel augmentation and prototype conditioning are resolved. |

## Active experiment and interpretation boundary

The active profile is `p3-campp-known446-ood-channelrobust-oof-f0`, commit
`eb433629b85a07a0665056d1bf6fcd84694cf1ca`.  It changes only the augmentation
policy relative to Control Fold 0.  Its early epoch-8 advantage did not persist
through the frozen-encoder phase.  At the matched epoch-21 snapshot, immediately
after progressive encoder unfreezing, the treatment had probability-average
Macro-F1 `0.8893` versus Control `0.8942` (`-0.0049`), logit-average `0.8787`
versus `0.8930` (`-0.0143`), and validation loss `1.0794` versus `1.0734`.
The run remained healthy and active, so this is evidence to watch the post-
unfreeze trajectory, not a terminal rejection.

The treatment still passes only if the preregistered standalone LME20,
fixed-50/50 complementarity, Known/OOD, rescue-rate, and provenance gates all
pass after the run.  Failure rejects the candidate and forbids automatic Fold
expansion.

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

## Backup decision order

1. Finish and audit the active channel-robust Fold-0 treatment.
2. If it passes, preregister Fold 1/2 and require consistent three-Fold gain.
3. If it fails, analyse whether the failure is representation quality or lack
   of complementary errors; do not tune a fusion weight on Fold 0.
4. The cheapest next candidate is a training-fold-only condition-aware
   prototype bank using predefined acoustic/channel proxies and leave-one-fold-
   out selection.  It is not authorised until its exact contract is written.
5. Domain-adversarial training is a later option only if the proxy/prototype
   route cannot explain the residual topology and the remaining budget permits
   a controlled ablation.
