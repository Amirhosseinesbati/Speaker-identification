# Phase 7 — Smoke Verification Record (2026-08-10)

Re-generated the self-contained submission package and ran the verification
pass after the Phase 7 fixes (commits `eb7d2e3` … `7c824e1`).

## Commands run

| check | result |
|-------|--------|
| `python scripts/build_submission.py` | ✅ package rebuilt (1576 MB; weights dirs skipped, idempotent) |
| `diff src/<mod>.py submission/src/<mod>.py` | ✅ byte-identical for encoders, ensemble_calibrate, centroid_baseline, eda_embeddings, train, model_factory, pipelines/steps |
| `python -m src.encoders` | ✅ all 5 registry keys resolve |
| `python -m src.ood_detector` | ✅ all FAISS tests pass |
| `python -m src.ensemble_calibrate --help` | ✅ CLI parses |
| `pytest tests/` | ✅ 8 passed (41.7 s) |

## Per-step unit / smoke checks (run during implementation)

- **encoders** (`eb7d2e3`): offline guard still raises loudly with
  `allow_hub_download=false`; registry smoke passes.
- **setup_vast.sh** (`9f111af`): `bash -n` OK; YAML-edit logic tested for
  titanet (hub on, full fine-tune, local_path override), ecapa (partial),
  wavlm (freeze_feature_extractor only) and the no-env fallback.
- **deploy.py** (`a5a9f50`): `read_model_selection()` returns `ENCODER_TYPE`,
  `ALLOW_HUB_DOWNLOAD` and all 5 `LOCAL_PATH_<ENC>` from the real config.
- **deploy_app.py** (`72de0f1`): `_encoder_save_config()` unit-tested — Full
  fine-tune now honored for campp/eres2net/titanet, per-encoder pooling,
  stale-key cleanup.
- **ensemble_calibrate** (`9344e89`): synthetic ecapa + campp checkpoints each
  load their **own** embedded config; no-config error + `--config-path`
  fallback paths verified.
- **centroid / eda** (`3cdd917`): CentroidFuser metadata validation (match /
  encoder-mismatch / dim-mismatch all behave); real 2-file ECAPA extraction →
  `(2, 192)` on CPU.
- **inference `--faiss-ood`** (`07d5b0f`): combine + renormalise math verified
  (p0 = 0.55, row sums 1.0; extreme case keeps the row well-formed).
- **train / steps** (`2fb23ba`): `tune_ood_threshold` verified for a separable
  head and the collapsed-head median fallback; encoder-named filenames present
  in both training paths.

## Still open (not run here by design)

- Training the 5 models on Vast.ai to produce `checkpoints/<enc>_best.pt`.
- Full GPU ensemble dry-run and the real 3090 runtime measurement.
- Short-file batching in `inference.py` (deferred until the 3090 numbers exist).
