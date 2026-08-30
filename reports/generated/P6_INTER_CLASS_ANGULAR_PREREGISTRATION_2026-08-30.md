# P6 Inter-Class Angular Preregistration — 2026-08-30

## Status

`DORMANT / NOT AUTHORIZED TO RUN YET`

This document and its implementation were fixed while the P5 matched-sampler
control was still running. Therefore no terminal P5 treatment metric was
available to influence the P6 formulation, weight, activation rule, or gates.

## Research evidence

SciSpace semantic search was used with a full methodological question rather
than keywords. It returned the primary APSIPA 2019 paper *A Study on Angular
Based Embedding Learning for Text-independent Speaker Verification* (DOI
`10.1109/APSIPAASC47483.2019.9023165`, SciSpace id `2kw1bjnl2h`) and the
Interspeech 2020 *Angular Margin Centroid Loss* paper (DOI
`10.21437/INTERSPEECH.2020-2538`, SciSpace id `4vfuhfrfu7`). SciSpace's
methodology-column endpoint was retried with those native ids, but reported no
methodology data for either paper. Exact equations and ablations were therefore
checked against the primary PDFs:

- APSIPA 2019: <https://www.apsipa.org/proceedings/2019/pdfs/414.pdf>
- Interspeech 2020: <https://www.isca-archive.org/interspeech_2020/wei20b_interspeech.pdf>

The APSIPA treatment defines, for normalized class-weight vectors, the
exclusive positive-cosine energy

`L_inter = (1/C) || relu(W_n^T W_n) - I ||_F^2`

and combines it with an angular classification objective as

`L = (1 - lambda_inter) L_angular + lambda_inter L_inter`.

It fixes `lambda_inter=0.01` rather than selecting it from the evaluation set.
Its VoxCeleb ablation reports small directionally consistent improvements when
the term is added to AM-Softmax and AAM-Softmax. This is useful mechanism-level
evidence, but it is not direct evidence for this competition's 446-known plus
binary-OOD Macro-F1 objective. P6 therefore treats `0.01` as one fixed transfer
hypothesis, not as a promised gain and not as a tunable grid.

Our `ArcFaceHead.weight` stores classes by row, so the mathematically equivalent
implementation uses `W_n W_n^T`. Negative inter-class cosine values remain
unpenalized exactly as in the source. The computation is float32 even under AMP.

## Exact candidate

- Profile:
  `p6-campp-known446-ood-crossfile-consistency-interclass-e01-long120-oof-f0`
- Raw config SHA256:
  `d30c5631b8fd8499a4f2655f7dc41c5e3d5f6b0194ec4cfdcdf40628a5a2dbdc`
- Matched control: terminal P5 cross-file consistency treatment
  `p5-campp-known446-ood-crossfile-consistency-c01-long120-oof-f0`
- Single scientific difference: `training.loss.speaker.inter_class.enabled`
  changes from `false` to `true`.
- Fixed type: `exclusive_angular_energy`.
- Fixed convex weight: `0.01`.
- Same P3 Raw warm start, Fold 0 split, paired-file sampler, augmentations,
  eight windows, 120-epoch cosine schedule, optimizer, EMA, ArcFace, binary OOD
  head, loss weights, seeds, hardware profile, and immediate stop rules as P5.
- Maximum incremental runtime/cost if activated: 8 hours / `$1.40`.

The default configuration explicitly disables this regularizer. Runtime code
rejects non-scalar or non-finite energy, invalid weights, missing class weights,
and sub-center/3-D class-weight tensors. It logs raw speaker loss, effective
speaker loss, raw inter-class energy, and its exact weighted contribution.

## Activation rule

Activate P6 only if all of the following are true after a complete P5 audit:

1. P5 treatment is neutral or fails its score gate.
2. Its embedding-spread ratio is at least `0.95` relative to the P5 sampler
   control, so the invariance hypothesis did not collapse representation.
3. P5 Known Accuracy and OOD-F1 each remain within `0.001` of their matched
   control.
4. All P5 OOF, split, class-map, model, history, receipt, and MLflow provenance
   checks pass.
5. Remaining total campaign budget safely covers the preregistered P6 cap.

Do not activate P6 after a passing P5, representation collapse, guardrail
failure, artifact/provenance failure, or insufficient budget. Do not modify the
weight after observing P5 or leaderboard results.

## P6 acceptance gates

P6 is accepted only if every condition below holds on untouched Fold 0 OOF:

1. Raw probability-average Macro-F1 improves by at least `+0.002` over the
   terminal P5 treatment.
2. It also improves by at least `+0.002` over the terminal P5 matched-sampler
   control, preventing a weak relative win from hiding a failed P5 objective.
3. Fixed 50/50 probability fusion with the immutable external CAM++ reference
   improves Macro-F1 by at least `+0.002`; no blend search is allowed.
4. Known Accuracy and OOD-F1 each fall by no more than `0.001` against both
   relevant controls.
5. It rescues at least `20%` of matched-control errors.
6. Embedding-spread ratio is at least `0.95`.
7. The selected Raw checkpoint's exact exclusive class-weight energy is at most
   `0.95` times the corresponding terminal P5-treatment energy. This is the
   mechanism gate; a score fluctuation without geometric movement is rejected.
8. All receipt hashes, OOF uniqueness, class-map/split provenance, histories,
   and MLflow artifacts are complete and mutually consistent.

Passing Fold 0 authorizes only a separately preregistered multi-fold evaluation.
It does not authorize a submission, threshold search, blend search, epoch
selection against the leaderboard, or automatic full-data training.

## Verification fixed before activation

- Pure energy is zero for orthogonal and antipodal class weights.
- Positive-cosine energy matches the closed-form expectation.
- Gradients are finite and nonzero.
- The exact convex speaker-loss mixture is numerically tested.
- Default-disabled behavior remains unchanged.
- The training loop logs the actual backpropagated contribution.
- The fully resolved P6 profile becomes byte-for-byte equivalent to P5 after
  changing only `inter_class.enabled` back to `false`.
- The raw P6 config hash is locked in a regression test.
