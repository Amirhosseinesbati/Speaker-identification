# IAAA leaderboard result — CAM++ LME20 PCM recovery

Date reported by the user: 2026-08-29

## Immutable package receipt

- Local package: `data/artifacts/iaaa_campp_lme20_pcm_recovery_20260829.zip`
- Size: `94,231,494` bytes (`89.866 MiB`), below the `1 GiB` competition limit
- SHA256: `3653d0f4e54433f4096a521d814d5e606c5b9314fefa986b315b7623143a7494`
- Leaderboard task: `scoring`
- Leaderboard created time reported by the user: `2026-08-29 18:13:05`
- Accuracy: `0.9667036625971143`
- Macro-F1: `0.9667174284505605`
- Rank snapshot reported by the user: seventh; first place was `0.972643`

The ZIP receipt was re-hashed locally after the leaderboard result was
received.  The leaderboard values are user-provided external evidence; hidden
labels or per-file predictions are not available to the project.

## Internal evidence that preceded the submission

The packaged decision is the already locked PCM/LME20 rule with
`alpha=0.15`, `kappa=16`, `tau=0.5`, `lambda_unknown=0.75`, and `beta=20`.
Submission equivalence was exact on all three OOF folds (`0` prediction
mismatches).  Its aggregate three-fold evidence was:

| Metric | Raw CAM++ probability average | PCM/LME20 | Delta |
|---|---:|---:|---:|
| Macro-F1 | `0.9438885981` | `0.9633564052` | `+0.0194678072` |
| Accuracy | `0.9467056443` | `0.9705419384` | `+0.0238362941` |
| Known Accuracy | `0.9510771993` | `0.9524236984` | `+0.0013464991` |
| OOD-F1 | `0.9547945205` | `0.9751111111` | `+0.0203165906` |

The rule rescued `120/237` baseline OOF errors while introducing `14`, a
`50.63%` rescue rate.  It sharply reduced unknown-to-known errors (`128` to
`25`) while also reducing known-to-wrong-known errors (`39` to `19`).

The hidden leaderboard Macro-F1 is `+0.0033610232` above the aggregate OOF
estimate.  This is evidence that the cross-fitted rule transferred; it is not
permission to retune any parameter against the leaderboard.

## Decision-layer exhaustion check

Existing leave-one-fold-out alternatives were re-read after the leaderboard
result rather than rerun or retuned.  None exceeds the already packaged
PCM/LME20 rule:

| Cross-fitted decision family | Aggregate Macro-F1 | Known Accuracy | OOD-F1 |
|---|---:|---:|---:|
| Packaged PCM/LME20 | `0.9633564052` | `0.9524236984` | `0.9751111111` |
| Joint prototype aggregation/parameters | `0.9612296580` | `0.9501795332` | `0.9718216108` |
| Fixed mean-centroid, 1000-centroid parameters | `0.9604604563` | `0.9506283662` | `0.9720000000` |
| Unknown-cluster hypothesis plus parameters | `0.9594993754` | `0.9497307002` | `0.9706796979` |

The prototype search already included mean centroids, medoids, centrality
weights, log-mean-exp, maximum exemplar and top-2 exemplar evidence.  The
unknown-cluster search included KMeans sizes `455/500/554/600/700` and average
linkage sizes `455/554/700`, with every target Fold held out from selection.
These negative results close the obvious post-processing branch for now.  A
new decision rule is not justified unless a new representation produces new
OOF evidence; the active budget should target representation robustness.

## Consequence for the active research path

PCM/LME20 remains the fixed deployment decision layer and external baseline.
The rejected Channel-Robust Fold-0 candidate scored `0.9566779064` with the
same LME20 rule versus Control's `0.9611456663`; its fixed 50/50 fusion also
lost `0.0020774947` and violated Known/OOD guardrails.  More generic channel
augmentation alone therefore did not improve the representation feeding the
successful decision layer.

The active paired clean/aug consistency experiment tests a narrower mechanism:
retain the strong ArcFace/OOD representation while explicitly aligning the
same speech crop before and after channel augmentation.  The engineering
40-epoch no-consistency pilot is excluded from the decision.  The locked
long-horizon decision remains a 120/120 matched A/B with no metric early stop,
followed by exact LME20, error-rescue, Known/OOD and embedding-collapse audits.
No new leaderboard package is warranted unless that internal gate passes.
