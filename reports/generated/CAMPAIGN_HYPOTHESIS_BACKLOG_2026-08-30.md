# IAAA speaker-identification hypothesis backlog — 2026-08-30

## Operational anchor and target

- Externally validated package:
  `data/artifacts/iaaa_campp_lme20_pcm_recovery_20260829.zip`.
- Package SHA256:
  `3653d0f4e54433f4096a521d814d5e606c5b9314fefa986b315b7623143a7494`.
- Package size: `94,231,494` bytes, safely below the `1 GiB` competition cap.
- Official user-reported Macro-F1: `0.9667174284505605`; accuracy:
  `0.9667036625971143`.
- Gap to the stated `0.973` research target: `0.0062825715494395`.
- Locked three-Fold LME20/PCM OOF Macro-F1: `0.9633564052154656`.

The leaderboard result is an external anchor, never a parameter-selection
set.  A new ZIP is built locally only after a candidate passes its internal
OOF and provenance gates.

## Closed branches

| Branch | Evidence | Status |
|---|---|---|
| Global unknown threshold / scalar calibration | Three leave-one-fold-out fits selected native threshold `0.5`; quality-aware gain violated Known guardrail. | Closed. |
| Low-energy fallback | Failed independent Fold checks. | Closed. |
| AuxMetric | Fold-0 standalone and fixed blends lost; oracle gain below gate. | Closed. |
| ERes2NetV2 recipe | Best Fold-0 Raw `0.9124381`, below the `0.9369212` gate. | Closed; no Fold 1/2. |
| Generic CAM++ LMFT | Fixed-30 warm-start did not beat its source checkpoint. | Closed unless a genuinely different, preregistered LMFT mechanism is proposed. |
| Raw/EMA snapshot fusion and known rerank | No aggregate gain; snapshots too correlated. | Closed. |
| Temporal multiview replacement / hierarchical aggregation / known rerank | Aggregate regressions or exactly neutral decisions. | Closed. |
| Short-audio repeat and native-duration replacement | Inconsistent Fold direction and OOD regressions. | Closed. |
| Same-crop paired consistency Long-120 (P4) | Treatment/control LME20 delta only `+0.0009816106`, fixed-fusion delta `-0.0023666564`, rescue `5/48`; spread remained healthy and every extension check failed. | Closed as neutral; activated cross-file P5. |
| Speaker-specific enrollment threshold | Exact LME20 baseline reproduced, but aggregate Macro-F1 changed by `-0.0325577308`; every Fold was negative and Known/OOD guardrails failed. | Closed. |
| Post-hoc prototype/threshold/blend sweeps | Would tune rejected evidence. | Forbidden. |

## Active decision experiment

The active experiment is the matched Fold-0 cross-file Long-120 pair:

1. `p5-campp-known446-ood-crossfile-paired-control-long120-oof-f0`, raw config
   SHA256 `ceae8376e4bf6963063295e2e7d0a44a64aa492988fde9caa091989ea464726e`;
2. `p5-campp-known446-ood-crossfile-consistency-c01-long120-oof-f0`, raw config
   SHA256 `5243b42eebf82d5f2fb75588ec0040072137a44ee9d811ee50cacff6ac98d5ec`.

Both branches start from the immutable P3 Raw checkpoint and have identical
split, seed, model, augmentation, optimiser, learning rates, cosine schedule,
windows, batch composition, speaker-balanced two-file sampler and fixed
120-epoch horizon.  Every batch contains 24 OOD rows plus 12 distinct known
speaker pairs.  The only scientific difference is whether the fixed `0.1`
cross-file cosine-consistency term is active.  Metric early stopping is off;
epochs 40 and 80 are diagnostics only.  The matched control launched first;
the treatment may start only after the control is terminal and fully audited.

The primary gate remains treatment LME20 gain over matched control at least
`+0.002`, fixed `50/50` fusion gain over the external Control Fold-0 evidence at
least `+0.002`, Known and OOD-F1 drops each no worse than `-0.001`, Control-error
rescue at least `20%`, and treatment/control embedding-spread ratio at least
`0.95`.  Paired randomisation tests describe uncertainty but cannot override
these effect-size and safety gates.

## Historical activation record for P4 (completed)

The P4-specific branches below are retained as an audit trail.  Their decision
has already been made: same-crop consistency was neutral without collapse,
the extension gate failed, and that evidence activated P5.  They are not the
current launch policy.

Before P5 activation, the preregistered CPU-only enrollment test in
`CAMPP_LME20_SPEAKER_SPECIFIC_THRESHOLD_PREREG_2026-08-30.md` was run.  It
reproduced the exact locked baseline, then failed every Fold and both safety
guardrails as recorded in the closed-branches table.  This completed the
historical prerequisite; the diagnostic must not be repeated or repaired with
a quantile, offset or leaderboard result.

### A. Treatment passes every gate

Write a separate Fold-1/2 preregistration with unchanged scientific settings.
Do not tune the consistency weight, horizon, checkpoint or fusion from Fold 0.
Only a leakage-free aggregate OOF result with consistent Fold direction can
authorise a local submission package.

### B. Treatment is still improving at epoch 120 but has not converged

Permit a new matched extension only if every predeclared late-tail check passes:
treatment epochs 111--120 improve over 101--110 by at least `0.0005`, the
101--120 slope is positive, the best epoch lies in 111--120, treatment tail
improvement exceeds control tail improvement by `0.0005`, Known/OOD tail drops
are each no worse than `-0.002`, and spread ratio remains at least `0.95`.
The extension must use equal horizons and a newly fixed scheduler/resume
contract for both branches; never continue treatment alone.

### C. Same-crop consistency is neutral or harmful without collapse

The next representation hypothesis is a known-only, speaker-balanced
**cross-file positive-pair** treatment, compared with a sampler-matched control.
Its immutable activation, horizon, budget, stop rules and terminal gate are
fixed in `CROSS_FILE_POSITIVE_PAIR_FOLD0_PREREGISTRATION_2026-08-30.md` before
any cross-file outcome exists; implementation feasibility is recorded in
`CROSS_FILE_PAIRING_READINESS_2026-08-30.md`.
After the exact Fold-0 split, duration filtering and duplicate cleaning, all
446 known speakers retain at least two distinct training files (median three,
maximum five), so different files are a feasible declared recording proxy.
Keep the OOD objective unchanged.
Do not import SSPS clustering, memory queues and a new loss together.  This
choice follows SSPS evidence that same-utterance positives retain channel cues
([Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-183)).

A focused SciSpace pass adds an implementation boundary.  Supervised
speaker-contrastive learning with labels has evidence for same-speaker
positives, different-speaker negatives and an angular margin (APSIPA 2022,
DOI `10.23919/APSIPAASC55919.2022.9980014`), while simple Siamese speaker
regularisation reports gains using positive pairs alone (ICASSP 2022, DOI
`10.1109/ICASSP43922.2022.9747526`).  Because CAM++ already has a supervised
ArcFace separation loss, adding a second contrastive negative/margin system
would confound pair identity with a new classifier objective.  Therefore the
cleanest conditional test is positive-only cosine alignment between two
different files of the same known speaker, using the same declared coefficient
as the current consistency test.  Its control must use the identical
speaker-balanced two-file sampler and primary losses but zero cross-file
alignment weight.  Both files are already ordinary supervised rows in the
same batch, so treatment reuses their existing embeddings and does not add a
second audio forward pass.  Unknown examples retain the existing OOD path and
never receive a fabricated cross-file positive.  A deterministic sampler and
loss path are implemented and CPU-tested, but this remains a conditional
backlog item: no config or Run is authorised before the active same-crop pair
is terminal.

The latest SciSpace conclusion audit reinforces this order.  SSPS supplies
direct speaker evidence for cross-recording positives, while the stage-wise
fixed-anchor strategy (arXiv `2510.18530`) supplies only indirect evidence for
an asymmetric clean-anchor teacher.  Therefore a fixed-anchor/stop-gradient
variant remains a later contingency, not a reason to change the running pair
or to bundle teacher asymmetry with the cross-file sampler test.

Channel-adversarial speaker training (ICASSP 2020, DOI
`10.1109/ICASSP40776.2020.9053323`) provides a second, supervised indication
that same-speaker **recording identity** is the relevant nuisance granularity.
It does not authorise copying its adversary here: this dataset has no reliable
session/channel labels and different files are only a declared proxy.  Keep
the conditional test positive-only, and do not bundle an adversarial head with
the sampler change.

### D. Consistency causes spread loss or late degradation

Retire invariance-pressure variants.  Prefer one isolated training-side factor:

1. a sampler-matched known-hard exposure ablation using only train-fold
   duration/RMS/active-fraction statistics; or
2. a single fixed bandwidth/resampling augmentation against an otherwise
   matched control.

The order is selected from the terminal error topology: excess
Known-to-Unknown errors favour the known-hard sampler; codec/bandwidth-sensitive
residuals favour bandwidth augmentation.  Neither may be selected using the
leaderboard.

### E. Fixed-hard augmentation adapts slowly but consistency adds no value

Test one literature-fixed augmentation-frequency curriculum against a
compute-matched fixed-hard control.  Breakpoints and probabilities must be
chosen before Fold-0 results.  Do not combine curriculum, a new sampler and a
new loss in one Run.

## Locked decision tree after the active P5 pair

### A. P5 treatment passes every gate

Write a separate Fold-1/Fold-2 preregistration with the exact same sampler,
coefficient, horizon, scheduler and decision policy.  Only consistent
multi-Fold OOF direction with the same Known/OOD, rescue, spread and provenance
guardrails may authorise a local package.  Fold 0 alone never authorises a
submission.

### B. P5 is neutral or harmful while spread remains healthy

Reject positive-only cross-file consistency and do not tune its coefficient or
schedule.  The next eligible GPU hypothesis is one matched Fold-0 ablation that
adds explicit inter-speaker angular separation to the P5 treatment.  Its loss,
coefficient, budget and gate must be fixed before outcome observation.  This is
supported by APSIPA 2022 supervised angular contrastive evidence (DOI
`10.23919/APSIPAASC55919.2022.9980014`) and the exclusive inter-class angular
regulariser evidence already recorded in the research ledger.  Class-aware
hard-negative weighting and automatic multi-objective balancing change more
than one variable and remain later hypotheses.

### C. P5 loses embedding spread or violates the Known guardrail

Retire positive-pair invariance.  Select exactly one alternative from terminal
error topology: angular-margin centroid training if inter-speaker separation
collapsed, known-hard exposure if Known-to-Unknown errors dominate, or one
fixed bandwidth transform if codec/bandwidth sensitivity dominates.  The
previous post-hoc centroid decision rule and AuxMetric result are not reopened;
any centroid branch must be a new training objective with a matched control.

### D. Either branch is incomplete or fails provenance

Repair only with idempotent project recovery.  Do not infer a scientific result
from a partial history, incomplete OOF, missing receipt or mismatched MLflow
run, and do not launch a successor until the pair is terminal and auditable.

## Research-only later directions

- A focused SciSpace search found angular-margin centroid and episodic
  prototypical evidence for few-shot speaker recognition.  The former is not a
  new branch here: the rejected Fold-0 AuxMetric already used an EMA
  data-centroid AM-softmax loss and lost on Macro-F1, Known accuracy, OOD-F1,
  fixed blends and the oracle-gain gate.  Do not reopen it.  A future episodic
  support/query objective is scientifically distinct only if prototypes are
  constructed inside speaker-balanced episodes rather than kept as global EMA
  centroids; it remains research-only behind the preregistered cross-file pair.
- A second SciSpace pass found OpenFEAT and cross-channel meta-learning as the
  closest published matches to the dominant LME20 residual (`87`
  known-to-unknown errors).  Their common useful factor is
  enrolment-conditioned support/query adaptation, with guest/unknown examples
  retained during training.  If the paired consistency and cross-file branches
  both fail without collapse, this is a more distinct later hypothesis than
  another global prototype loss.  PLDA/cohort/T-Norm papers from the same
  search do not reopen score normalisation: our NAP, LDA, WCCN and adaptive
  AS-Norm cross-fit audits already rejected that family.
- Duration-aware embedding objectives such as DAME are relevant to the hard
  short-utterance tail, but are a larger representation change and remain
  research-only until cheaper paired hypotheses fail
  ([arXiv:2601.13999](https://arxiv.org/abs/2601.13999)).
- Sharpness-aware optimisers and SWITCH-EMA have recent speaker-verification
  support, but changing optimiser plus representation policy would not be a
  clean next ablation
  ([Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-2361)).
- A scheduled-off consistency term is plausible because embedding-level
  contrastive learning was reported useful mainly early in MeMo training, but
  it requires a new preregistration and cannot be invented from the active
  treatment curve
  ([Interspeech 2024](https://doi.org/10.21437/Interspeech.2024-360)).
- SciSpace also retrieved a speaker-verification weight-space ensemble study
  reporting gains under cross-channel shift by interpolating a base model and
  its fine-tuned descendant (IEEE/ACM TASLP, DOI
  `10.1109/TASLP.2022.3233231`).  The active treatment is architecturally
  compatible with that idea, but its epoch curve was already visible when the
  paper was retrieved.  Therefore no interpolation coefficient may be fitted
  to this Fold-0 curve.  Keep this research-only unless a future multi-Fold
  contract fixes the coefficient (for example `0.5`) before outcomes and
  compares it against both endpoints with the same Known/OOD and rescue
  guardrails.
- A speaker-specific SciSpace search found support for channel/recording and
  clean/noisy invariance, but no controlled evidence that delaying the
  consistency coefficient is better than applying it from the start.  Keep
  schedule changes out of the locked pair; if needed, evaluate one as a new
  matched experiment rather than a post-hoc rescue.

## Resource and execution rules

- Recompute campaign cost before every launch; the campaign ceiling is `$20`.
- Run only one scientific training process at a time on Instance `48886926`.
- Heavy inference, embedding extraction and training remain worker-only.
- Never change the worker checkout during a scientific Run.
- Hash OOF, class map, split, config, history, selected model, logs, receipts
  and MLflow artifacts before using any terminal result.
- No candidate reaches Fold expansion or submission through a single metric;
  effect size, Fold direction, Known/OOD balance, rescue, collapse and
  provenance must agree.
