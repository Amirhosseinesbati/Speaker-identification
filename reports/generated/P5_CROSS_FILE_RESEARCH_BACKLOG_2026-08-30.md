# P5 cross-file research backlog — 2026-08-30

## Boundary

This note was created after the P5 matched-control launch.  It cannot change
the active cross-file recipes, their fixed 120-epoch horizon, coefficient,
sampler, gate or ordering.  It only ranks mutually exclusive follow-up
hypotheses if the complete P5 pair fails.

## Evidence retrieved with SciSpace

1. **Different-recording positives.**  SSPS (Interspeech 2025, DOI
   `10.21437/Interspeech.2025-183`) directly supports using different
   recordings of one speaker rather than two views of one utterance to reduce
   intra-speaker variance.  This is the isolated P5 treatment already running.
   The paper is self-supervised speaker verification rather than this
   closed-set-plus-OOD task, so it supports the mechanism but cannot by itself
   establish the P5 decision policy or expected gain.
2. **Explicit inter-speaker angular separation.**  Chen, Ren and Xu (APSIPA
   2019, DOI `10.1109/APSIPAASC47483.2019.9023165`) report that an exclusive
   inter-class regularizer complements angular-margin embedding learning.  This
   is the lowest-complexity anti-collapse guard returned by the search.
3. **Angular-margin centroid training.**  Wei, Du and Liu (Interspeech 2020,
   DOI `10.21437/INTERSPEECH.2020-2538`) jointly contract same-speaker
   embeddings and separate speaker centroids.  This is a training objective,
   not the already rejected post-hoc centroid decision rule.
4. **Discriminant variance objective.**  Gao, Song and McLoughlin
   (Interspeech 2019, DOI `10.21437/INTERSPEECH.2019-1489`) explicitly optimize
   small intra-speaker and large inter-speaker variance and report gains with a
   simple cosine backend.
5. **Channel adversarial learning.**  Chen, Wang and Qian (ICASSP 2020, DOI
   `10.1109/ICASSP40776.2020.9053905`) suppress device/environment information
   with joint multi-task and adversarial training.  It requires reliable
   channel labels or proxies that the local dataset does not currently expose.
6. **Cross-channel meta-learning.**  Zhang, Wang and Lee (ICASSP 2021, DOI
   `10.1109/ICASSP39728.2021.9413978`) align support/query embeddings under
   unseen-channel evaluation.  This adds an episodic optimizer and a second
   distribution objective, so it is not a one-variable continuation.
7. **Angular supervised contrastive separation.**  Li and Mak (APSIPA 2022,
   DOI `10.23919/APSIPAASC55919.2022.9980014`) combine label-aware positive and
   negative pairs with an additive angular margin.  Their result supports the
   specific follow-up of coupling cross-file compactness to explicit
   inter-speaker separation, rather than increasing the positive-only weight.
8. **Hard-negative and objective-balance failure modes.**  Li, Mak and Meng
   (ICASSP 2023, DOI `10.1109/ICASSP49357.2023.10096230`) report that plain
   softmax contrastive learning may lack discrimination and that hard negatives
   can dominate learning; they use class-aware attention and gradient-based
   multi-objective balancing.  Those additions are intentionally not eligible
   as an immediate follow-up because they change more than one variable, but
   they are a warning against interpreting positive-pair contraction alone as
   sufficient evidence.

## Checkpoint-averaging evidence boundary

A targeted SciSpace search during the locked P5 treatment retrieved Model
Soups (Wortsman et al., ICML 2022, DOI `10.48550/arXiv.2203.05482`) as the
closest primary evidence.  The paper averages independently fine-tuned models
that remain in one low-error basin and reports that the resulting single model
can improve robustness without ensemble inference cost.  The search did not
retrieve a speaker-identification experiment establishing that the same claim
holds for a Raw/EMA midpoint from one CAM++ trajectory under open-set
channel/session shift; SciSpace's methodology-column lookup also returned no
additional record for the paper.

This makes parameter averaging a distinct but weakly transferred hypothesis,
not evidence for changing P5.  The campaign has already rejected fixed Raw/EMA
probability fusion on all three Control folds, so neither that negative result
nor the general Model Soups result can be silently reinterpreted as proof for
or against a weight-space midpoint.  If the hypothesis is ever activated, it
must be a separately preregistered, fixed `50/50` Raw/EMA parameter midpoint
evaluated on all three locked Control folds with no coefficient search and the
same Macro-F1, Known and OOD guardrails.  It cannot outrank the already locked
P6 pair merely because it is cheap, and it cannot be selected from P5's
observed trajectory.

## Locked decision order after P5

- If P5 passes its complete gate, do not add any of these losses; preregister
  leakage-safe multi-fold confirmation first.
- If P5 is neutral and embedding spread remains healthy, the first eligible
  follow-up is **one** matched Fold-0 ablation adding exclusive inter-class
  angular separation to the cross-file treatment.  This is now locked as the
  dormant P6 profile with the literature-fixed convex coefficient `0.01` and
  the score/guardrail/mechanism gates in
  `reports/generated/P6_INTER_CLASS_ANGULAR_PREREGISTRATION_2026-08-30.md`;
  no grid search is permitted.
- A neutral P5 result must not be answered by tuning the positive-consistency
  coefficient.  The literature-backed next variable is separation, while
  class-aware hard-negative weighting or automatic multi-objective balancing
  remain later, separately preregistered hypotheses.
- If P5 collapses or degrades Known Accuracy, retire positive-only invariance.
  Angular-margin centroid training is then the preferred alternative because
  it couples compactness to explicit separation; it must start from the same
  immutable source checkpoint and use a matched control.
- Channel-adversarial or meta-learning variants remain ineligible unless EDA
  first establishes leakage-safe channel/session groups.  Acoustic proxies may
  diagnose shift but may not be silently promoted to channel labels.
- Triplet, memory-queue, teacher-student, PLDA and multi-loss combinations are
  not concurrent fallbacks.  Each would require a separate hypothesis,
  budget, single-variable ablation and Known/OOD guardrail.

The leaderboard cannot select among these branches.  Only OOF evidence and the
predeclared error-rescue/guardrail criteria may activate the next one.
