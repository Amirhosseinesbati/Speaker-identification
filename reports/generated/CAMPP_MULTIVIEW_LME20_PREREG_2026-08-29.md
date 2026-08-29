# CAM++ view-level LME20 preregistration — 2026-08-29

## Evidence and hypothesis

The locked CAM++ backend represents each audio file by one mean-then-normalised
embedding before multi-enrollment scoring.  Multi-session speaker-verification
work reports benefits from retaining relationships among enrollment utterances,
and parameter-free attentive scoring has shown that multiple enrollment
statistics can improve over a single cosine representation:

- https://arxiv.org/abs/2203.05642
- https://www.isca-archive.org/interspeech_2014/pelecanos14_interspeech.html
- https://www.isca-archive.org/interspeech_2004/cheung04_interspeech.html

The fixed candidate keeps every real CAM++ temporal-window embedding and
computes one beta-20 log-mean-exp score over all valid query/enrollment view
pairs for each speaker group.  The score is normalised by the exact pair count,
which keeps score scale comparable across query and enrollment view counts.
Within a group, every real temporal view is deliberately one observation, so a
longer enrollment file can contribute more views than a short file; this is part
of the single pre-registered scoring change.  Repeated last-window padding
views are excluded.

## Locked variables

- selected Raw CAM++ Control checkpoints and fixed Fold0/1/2 OOF;
- existing train-only known and KMeans-554 group membership;
- unchanged waveform preprocessing, eight-second windows, speech-aware window
  selection and head probability average;
- LME beta `20`, fusion alpha `0.15`, kappa `16`, tau `0.50`, unknown weight
  `0.75`, and direct fixed decision rule;
- no grid, learned attention, threshold selection, blend selection or
  leaderboard feedback.

The only changed variable is scoring before the fixed decision layer:
file-mean embeddings become all valid temporal-view pairs.

## Provenance and acceptance

Per-window caches must hash checkpoint, OOF, pad artifact and file order.  The
mean of raw window embeddings must reproduce every locked aggregate embedding
within `2e-5`; all 4447 OOF files must remain unique and non-overlapping; and
all scores must be finite.

Extraction preserves the two batching paths already used in production:
enrollment uses window-major `model.embed` batches, while each validation/query
file uses its own window batch as in `predict_proba_and_embed`.  No candidate
metric is computed unless both paths reproduce their locked aggregates.

Acceptance requires strictly positive Macro-F1 gain in each Fold, aggregate
gain at least `+0.001`, and no Known Accuracy or OOD-F1 loss beyond `0.001` in
any Fold or aggregate.  Failure leaves the current local submission ZIP
unchanged.  Passing only authorizes a full-data cache/build equivalence audit.

## Resource bound

The existing RTX 3090 worker performs one deterministic extraction per Fold
and idempotently caches it.  Timeout is two hours (about `$0.35` at the current
rate); OOM, non-finite values, provenance mismatch or aggregate-reproduction
failure stops the experiment immediately.
