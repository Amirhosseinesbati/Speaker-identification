# Open-Set Speaker Backends: Evidence Mapped to the IAAA Campaign (2026-08-31)

## Purpose

This note maps a focused SciSpace and primary-source review onto experiments that already exist in the campaign. It prevents attractive literature names from reopening rejected branches or bundling multiple untested changes. It does not activate a Run while P5 is active.

## Evidence and campaign mapping

### Two-stage closed-set identity plus outlier detection

Wilkinghoff (Odyssey 2020) decomposes open-set speaker identification into closed-set identity classification and outlier detection. The reported i-vector system uses LDA/PLDA, Adaptive Symmetric Normalization and linear alignment, with cohort sizes selected on a development set. This is strong evidence for the decomposition, but it does not establish that its backend transfers unchanged to CAM++ probabilities and the IAAA Macro-F1 objective.

The campaign has already performed leakage-free three-Fold tests of the closest available backends on the locked LME20 evidence:

- AS-Norm cross-fit: aggregate `ΔMacro-F1 = -0.0094302638`, `ΔKnown = -0.0076301616`, `ΔOOD-F1 = -0.0089981543`; all held-out Fold gates failed.
- LDA cross-fit: aggregate `ΔMacro-F1 = -0.0038690176`, `ΔKnown = -0.0004488330`, `ΔOOD-F1 = -0.0038736529`; only one of three held-out Fold gates passed.
- NAP cross-fit: aggregate `ΔMacro-F1 = -0.0013184902`, `ΔKnown = +0.0008976661`, `ΔOOD-F1 = -0.0024992207`; only one of three held-out Fold gates passed.
- shrinkage-WCCN with known plus fixed pseudo-unknown groups: aggregate `ΔMacro-F1 = +0.0001572191`, `ΔKnown = +0.0004488330`, `ΔOOD-F1 = -0.0011516186`; only one of three held-out Fold gates passed.
- known-only shrinkage-WCCN: aggregate `ΔMacro-F1 = +0.0007247944`, `ΔKnown = +0.0008976661`, `ΔOOD-F1 = +0.0003900025`; despite the positive aggregate, only one of three held-out Fold gates passed.

Therefore cohort-size, projection-dimension, shrinkage-strength or threshold sweeps would tune rejected evidence. AS-Norm, LDA, NAP and both WCCN variants remain closed unless a genuinely different representation supplies independent evidence.

### Linear alignment is distinct, but not an immediate backend candidate

Wilkinghoff's linear alignment should not be collapsed into the rejected LDA branch. It fits one affine transform from each known-speaker i-vector to that speaker's training mean, thereby targeting within-class contraction without explicitly maximizing or suppressing between-class separation. In the MCE study it improved most development- and test-set EERs, outperformed WCCN as preprocessing, and avoided the severe open-set degradation observed when LDA itself was used as preprocessing.

The transfer evidence is nevertheless incomplete for IAAA. The reported transform is a 600-dimensional i-vector/PLDA preprocessing stage trained for 400 epochs; the paper's AS-Norm cohort sizes and several model choices were selected on its development set, and the authors explicitly report optimistic development-set results relative to test. They also identify evaluation with x-vectors and audio as future work. IAAA instead uses nonlinear CAM++ embeddings, only two to five distinct recordings per known identity, and Macro-F1 guardrails rather than EER. A direct full-dimensional affine fit would therefore have high overfitting risk and would confound representation contraction with the already rejected LDA/AS-Norm decision branch.

Linear alignment is retained only as a separate research hypothesis. It may be reconsidered after P5/P6 if and only if a closed-form or strongly regularized transform can be fitted strictly inside each training split, every hyperparameter is selected from the other Folds, and a no-transform matched control plus embedding-rank/spread checks are preregistered. It is not authorized during the active P5 run and it does not reopen AS-Norm, LDA, WCCN or score-threshold tuning.

### Discriminative normalization flows

Cai et al. model non-Gaussian, speaker-dependent embedding distributions with class-specific Gaussian priors in a discriminative normalization flow. The method is relevant to domain mismatch, but it estimates a nonlinear class-conditional transform. In the current IAAA training Fold every known speaker has only a few distinct recordings after filtering (minimum two, median three, maximum five in the P5 feasibility audit). That is weak support for estimating a new flow without strong regularization and would add a large training/objective confound. DNF is not a low-risk next experiment under the remaining campaign budget.

### Reciprocal points and negative samples

Chen et al. propose Speaker Reciprocal Points Learning (SRPL) and SRPL+ with real or synthesized unknown samples. The later SpeakerRPL v2 adds LogitNorm, adaptive anchors, score fusion and model selection. These papers provide direct evidence that explicitly reserving open space and using negative speakers can help open-set SID.

The transfer boundary is important:

- reported tasks are few-shot 5-way or 10-way enrollment settings, whereas IAAA has 446 known identities plus one aggregated unknown class;
- SRPL uses a pretrained WavLM/TDNN frontend and a new adapter/backend;
- SRPL+ and v2 combine several changes, including synthetic negatives, anchor learning, fusion and selection;
- IAAA already supplies real unknown training recordings, so synthesizing unknown voices would not isolate the useful factor.

Consequently these results do not authorize importing SRPL+ wholesale. They do reinforce the already preregistered P6 principle: test one angular/open-space constraint against a matched control, with fixed coefficient, Known/OOD guardrails and no leaderboard tuning. P6 remains dormant until P5 is terminal and the full matched-pair budget is still available.

### Prototype learning

Centroid/prototypical speaker losses improve seen and unseen-speaker embedding quality in verification and identification experiments. The campaign has nevertheless already shown that post-hoc centroid, entropy-reliability, AS-Norm and LDA decision layers do not pass its three-Fold guardrails. Representation training and post-hoc prototype correction are different hypotheses; the positive prototypical-loss literature cannot be used to rescue the rejected post-hoc rules.

## Locked consequences

1. Do not rerun or tune AS-Norm, LDA, scalar thresholds or entropy fusion.
2. Do not bundle reciprocal points, LogitNorm, synthetic negatives, adaptive anchors and fusion into one experiment.
3. Keep the active P5 treatment unchanged through its fixed horizon.
4. If P5 fails and budget permits, P6 remains the next representation experiment because it isolates one angular factor with a matched control.
5. A future reciprocal-point or normalization-flow experiment requires a new preregistration, a transfer argument for 446-way low-shot classes, and a complete matched control; it is not automatically activated by this review.
6. Treat linear alignment as a distinct low-rank/regularized representation hypothesis, not as evidence for rerunning LDA or AS-Norm; it remains dormant until a leakage-free three-Fold contract is written.

## Primary sources

- Kevin Wilkinghoff, “On Open-Set Speaker Identification with I-Vectors,” Odyssey 2020, DOI: [10.21437/Odyssey.2020-58](https://doi.org/10.21437/Odyssey.2020-58), [author PDF](https://wilkinghoff.com/publications/odyssey2020_open_set.pdf).
- Yunqi Cai et al., “Deep Normalization for Speaker Vectors,” IEEE/ACM TASLP 2021, DOI: [10.1109/TASLP.2020.3039573](https://doi.org/10.1109/TASLP.2020.3039573), [arXiv:2004.04095](https://arxiv.org/abs/2004.04095).
- Zhiyong Chen et al., “Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Sample,” [arXiv:2409.15742](https://arxiv.org/abs/2409.15742).
- Zhiyong Chen et al., “SpeakerRPL v2: Robust Open-set Speaker Identification through Enhanced Few-shot Foundation Tuning and Model Fusion,” [arXiv:2604.13605](https://arxiv.org/abs/2604.13605).
- Jixuan Wang et al., “Centroid-based Deep Metric Learning for Speaker Recognition,” ICASSP 2019, DOI: [10.1109/ICASSP.2019.8683393](https://doi.org/10.1109/ICASSP.2019.8683393), [arXiv:1902.02375](https://arxiv.org/abs/1902.02375).

## Retrieval note

SciSpace supplied the semantic paper shortlist. Method and transfer claims were checked against the primary Odyssey PDF and complete arXiv records. Recent SpeakerRPL v2 evidence is treated as preprint evidence and does not override the campaign's preregistered gates.
