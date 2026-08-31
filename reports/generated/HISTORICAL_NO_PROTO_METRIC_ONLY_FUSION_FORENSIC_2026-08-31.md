# Historical no-proto / metric-only fusion forensic

## Decision

The historical local `0.9734066888749531` value is not evidence that either
checkpoint reached that score and is not a submission-consistent Fold-0 OOF
measurement. It was produced by an in-sample weight search under a different
multi-window aggregation policy from the one shipped to the leaderboard.

The historical `0.6/0.4` result is therefore quarantined as diagnostic-only.
It cannot select a model, authorize Fold 1/2, or authorize a submission.

## Immutable artifacts

| Artifact | SHA-256 | Embedded epoch | Embedded validation Macro-F1 |
|---|---|---:|---:|
| `campp_no_proto_best.pt` | `92893c7642901dc2e1bc4eb1d70d9b51c8ed7b03c286b0c89f7340a46475ad40` | 133 | `0.9393457427357391` |
| `campp_metric_only_best.pt` | `ead5d1b7af290271db356c9ecf5e980513693d1f793a0c989e98094e8f0f37e5` | 45 | `0.9372307578530886` |
| `submission_no-proto_metric-only_60-40.zip` | `0c7a3c417360c08a9120c6a3f5148e7eb18650f9c292c4da6d6f2251dac2b612` | — | manifest claim `0.9734066888749531` |

Both checkpoints use the same Fold-0 seed-42 split and an internal 1001-way
class map. They are scientifically different:

- no-proto: binary OOD head enabled, prototype loss disabled;
- metric-only: binary OOD head disabled, metric prototype loss enabled.

This makes error complementarity plausible, but it does not validate the old
fusion number.

## Root cause 1: aggregation-policy mismatch

`src/ensemble_calibrate.py` calls `forward_multi_window`, averages logits over
windows, and only then calls `logits_to_probs`. Its fusion values therefore use:

`mean(window logits) -> softmax/collapse -> model fusion`

The packaged `submission/inference.py` calls `predict_proba_and_embed`, which
creates competition probabilities per window and averages those probabilities:

`softmax/collapse per window -> mean(window probabilities) -> model fusion`

Softmax is nonlinear, so these policies are not equivalent. The historical
local number did not measure the actual shipped inference path.

## Root cause 2: in-sample Fold-0 weight selection

`src/ensemble_calibrate.py` calls `grid_search_weights(..., step=0.05)` on the
same validation labels later used to quote the best score. The selected
`[0.6, 0.4]` weight is consequently tuned on Fold 0. It is not a clean estimate
of transfer performance and cannot be reused as a Fold-0 gate.

## External evidence

The shipped ZIP scored on the leaderboard:

- Accuracy: `0.9689234184239733`
- Macro-F1: `0.9605963610873509`

The `-0.0128103277876022` Macro-F1 gap from the local manifest claim is
consistent with the two mismatches above. The leaderboard result is used here
only as transfer evidence; it must not tune a weight, threshold, checkpoint, or
epoch.

## Correct follow-up

1. Finish the active deterministic no-proto Fold-0 run under its original
   early-stopping contract.
2. Evaluate its selected checkpoint with raw probability averaging and locked
   LME20 without tuning.
3. Produce probability-average OOF for the immutable historical metric-only
   checkpoint with exact filename alignment and split/SHA validation.
4. Use only fixed `0.5/0.5` probability averaging as the Fold-0 gate-bearing
   pair. Report historical `0.6/0.4` only as a quarantined diagnostic.
5. If the fixed pair passes its preregistered Macro-F1, Known, OOD-F1 and rescue
   gates, reproduce both recipes on Fold 1/2 and select any weight only by
   leave-one-fold-out cross-fit.

The executable contract is
`configs/analyses/no-proto-metric-only-paired-f0-prereg.json`; the audited
evaluators are `scripts/dump_checkpoint_oof_single_fold.py` and
`scripts/evaluate_fixed_oof_pair.py`.

## Packaging hygiene

The historical ZIP contains `weights/campp/credentials/session`, which is not
required for offline inference. Its contents were neither read nor printed.
The current submission builder and verifier already remove/reject credential,
session, token and secret paths, so future candidate ZIPs must pass that guard.
The immutable historical and best-baseline ZIPs are not modified because their
SHA receipts are part of provenance.
