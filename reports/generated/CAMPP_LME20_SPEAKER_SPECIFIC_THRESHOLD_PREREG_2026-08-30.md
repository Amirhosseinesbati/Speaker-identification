# CAM++ LME-20 enrollment-only speaker-specific rejection preregistration

**Status:** preregistered before running the rule on any CAM++ OOF labels
**Execution:** CPU-only on the Vast worker, after the active Long-120 paired
consistency Run reaches its terminal audit
**Leaderboard tuning:** forbidden

## Why this is a distinct candidate

The record package uses the locked CAM++ LME-20 prototype backend and achieved
Macro-F1 `0.9667174284505605` on the leaderboard.  Its three-Fold OOF anchor is
`0.9633564052`.  Global thresholding, AS-Norm, LDA/WCCN/NAP and generic
centroid blending have already been rejected, so none of those searches may be
reopened under a new name.

SciSpace retrieved Chaubey, Sinha and Ghose, *Meta-Learning Framework for
End-to-End Imposter Identification in Unseen Speaker Recognition*
([arXiv:2306.00952](https://arxiv.org/abs/2306.00952)).  Equations 4 and 7 give
a deterministic speaker-specific rejection rule: the threshold for enrolled
speaker `j` is the maximum cosine between any enrollment utterance of `j` and
any enrollment utterance from another enrolled speaker.  A query assigned to
`j` is retained only when its query-to-`j` enrollment similarity is strictly
greater than that threshold.  The rule therefore uses only train enrollment
data and is not a validation-fitted global threshold or cohort normalisation.

## Locked implementation

- Baseline: fixed CAM++ Control Fold 0/1/2 embeddings, fixed train-only
  KMeans-554 unknown partition, LME beta `20`, and the historical decision
  parameters already selected by leave-one-Fold-out calibration.
- Threshold evidence: only the target Fold's training embeddings from the 446
  competition-known speakers.  Pseudo-unknown groups and held-out OOF rows do
  not enter the threshold.
- Threshold for speaker `j`: maximum pairwise cosine between its enrollment
  files and every other known speaker's enrollment files.
- Gate score: maximum cosine between the query and enrollment files of the
  known speaker predicted by the locked LME-20 baseline.
- Boundary: retain known only for `score > threshold`; equality is rejected,
  matching the paper's strict decision equation.
- Unknown predictions produced by the LME-20 anchor remain unknown.
- No tunable multiplier, quantile, offset, temperature, cohort size, threshold
  search or target-Fold selection is permitted.

The executable audit is
`scripts/analyze_lme20_speaker_specific_threshold.py`.  Its focused regression
tests are in `tests/test_lme20_speaker_specific_threshold.py`.

## Acceptance gate

The rule is eligible only if all conditions hold:

1. the locked LME-20 aggregate baseline is reproduced at `0.9633564052`;
2. held-out Macro-F1 gain is non-negative in every Fold;
3. aggregate Macro-F1 gain is at least `+0.001`;
4. aggregate Known Accuracy delta is at least `-0.001`;
5. aggregate OOD-F1 delta is at least `-0.001`.

Any failure closes the branch.  The result must not be repaired by selecting a
per-Fold threshold, relaxing the maximum to a post-hoc quantile, or choosing a
leaderboard-dependent offset.

## Critical limitation

The source paper evaluates watchlists of only 5 and 10 enrolled speakers, each
with five enrollment clips.  This competition has 446 enrolled identities.
Taking a maximum over far more cross-speaker comparisons can inflate the
speaker-specific threshold and over-reject genuine known queries.  That is why
this is a cheap falsification test, not an assumed improvement and not a reason
to interrupt or alter the active representation-learning experiment.
