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
| Post-hoc prototype/threshold/blend sweeps | Would tune rejected evidence. | Forbidden. |

## Active decision experiment

The active experiment is the matched Fold-0 long-120 pair:

1. `p4-campp-known446-ood-channelrobust-paired-control-long120-oof-f0`;
2. `p4-campp-known446-ood-channelrobust-consistency-c01-long120-oof-f0`.

Both branches start from the same P3 Raw checkpoint and have identical split,
seed, model, augmentation, optimiser, learning rates, cosine schedule, windows,
batch size and fixed 120-epoch horizon.  The only scientific difference is the
fixed `0.1` clean/aug cosine-consistency term.  Metric early stopping is off;
epochs 40 and 80 are diagnostics only.  The treatment must not start until the
control is terminal and fully audited.

The primary gate remains treatment LME20 gain over matched control at least
`+0.002`, fixed `50/50` fusion gain over the external Control Fold-0 evidence at
least `+0.002`, Known and OOD-F1 drops each no worse than `-0.001`, Control-error
rescue at least `20%`, and treatment/control embedding-spread ratio at least
`0.95`.  Paired randomisation tests describe uncertainty but cannot override
these effect-size and safety gates.

## Locked decision tree after the pair

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
Every known speaker has at least five distinct local files, so different files
are a feasible declared recording proxy.  Keep the OOD objective unchanged.
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
speaker-balanced two-file sampler, primary losses and two forward passes but
zero cross-file alignment weight.  Unknown examples retain the existing OOD
path and never receive a fabricated cross-file positive.  This design is only
a conditional backlog item; no config or Run is authorised before the active
same-crop pair is terminal.

The latest SciSpace conclusion audit reinforces this order.  SSPS supplies
direct speaker evidence for cross-recording positives, while the stage-wise
fixed-anchor strategy (arXiv `2510.18530`) supplies only indirect evidence for
an asymmetric clean-anchor teacher.  Therefore a fixed-anchor/stop-gradient
variant remains a later contingency, not a reason to change the running pair
or to bundle teacher asymmetry with the cross-file sampler test.

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

## Research-only later directions

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
