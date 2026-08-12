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

## General dependency-closure verification (2026-08-11)

After the stale `uv.lock` was regenerated (`f583b26` — it previously missed
modelscope/nemo entirely and even violated the transformers/huggingface-hub
pins), the closure was verified **generally** so that starting training of
ANY of the 5 encoders works on a fresh machine, not just campp:

- **Static scan:** the import chains used by the encoder build paths
  (speechbrain, modelscope `models`+`hub`+`utils.config`, nemo
  `collections.asr.models`, transformers `models.wavlm`) were checked against
  the lock. The only packages present on the dev machine but absent from the
  lock (`funasr`, `jieba`, `oss2`, `tiktoken`, `umap-learn`, `kaldiio`,
  `tensorboardx`, `whisper_normalizer`) are OPTIONAL / off-path — `umap` lives
  only in modelscope's `cluster_backend` which the CAM++ path never imports
  (verified via `sys.modules` after `from modelscope.models import Model`).
- **Definitive test:** a throwaway venv installed from the lock
  (`uv sync --no-dev` → 355 packages) and ALL 5 encoders (ecapa, wavlm, campp,
  eres2net, titanet) were built with `create_model_from_config` →
  **5/5 build OK**. The venv was removed afterwards.
- **setup_vast.sh pre-flight** now imports ALL five frameworks (not just the
  active encoder) right after `uv sync`, so any missing dep fails early with a
  clear message. speechbrain is imported LAST because its broken LazyModules
  (`integrations.k2_fsa`) break lazy_loader's `inspect.stack` in other
  framework imports.

## Still open (not run here by design)

- Training the 5 models on Vast.ai to produce `checkpoints/<enc>_best.pt`.
- Full GPU ensemble dry-run and the real 3090 runtime measurement.
- Short-file batching in `inference.py` (deferred until the 3090 numbers exist).
