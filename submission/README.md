# IAAA 2026 Speaker Identification — Submission Package

Open-set speaker identification: classify audio into **447 classes** (446
known speaker-ids + aggregated `unknown`), scored on Macro-F1 over the 447
classes (argmax of the output probabilities).

This folder is **fully self-contained**: all code, pretrained encoder weights
and trained checkpoints are shipped inside it. No internet and no external
files are needed at inference time.

## Leaderboard entry point

```bash
python submission.py --data-dir <test-set-folder> --predictions-file-path predictions.csv
```

`submission.py` implements the mandatory contract from
`submissionforleaderbord.txt` (`load_data`, `predict`, `save_predictions` +
standard-library CLI) and:

- auto-discovers the trained checkpoints in `checkpoints/` in the order
  recorded in `ensemble_fusion_weights.json`;
- runs the selected raw CAM++ Control Fold-0 checkpoint with multi-window TTA
  (8 s windows, 50% overlap, at most 8 windows);
- applies the locked LME-20 multi-enrollment backend: every usable enrollment
  embedding remains available, similarities are pooled per identity with
  `log(mean(exp(20*cosine)))/20`, and the 554 train-only KMeans groups are
  collapsed into the single competition `unknown` class;
- uses fixed decision parameters selected by leave-one-fold-out cross-fit on
  all three Control folds; no leaderboard result selected a threshold, blend,
  epoch or clustering parameter;
- sets the offline env vars itself (`HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `MODELSCOPE_CACHE=weights/campp`), so the
  evaluation environment needs zero configuration.

## Output (competition CSV)

Exactly the competition format — columns `audio_file,speaker_id`:

```csv
audio_file,speaker_id
e9105299-285b-4df7-8c66-e4b3b721e8c8.mp3,3943d8f3-d820-44ff-aba6-23c796fda87b
016a2324-dc96-4f59-8d96-fb387afb1fff.mp3,845cbfc6-af92-4fa0-8b33-4e4e0e3b002a
57e9178b-7153-4e43-8fe4-6aacea3c9118.mp3,unknown
```

- `audio_file` — the test audio file's full name (with extension).
- `speaker_id` — the predicted speaker **UUID**, or `unknown` for out-of-set
  speakers. The model outputs a 447-way probability distribution and the
  argmax class index is mapped back to the UUID via the class map.

## Encoder ensemble (4 models)

| key | framework | emb dim | pooling | weights (local) |
|-----|-----------|---------|---------|-----------------|
| `ecapa`    | SpeechBrain | 192 | identity | `weights/ecapa/` |
| `campp`    | ModelScope  | 512 | identity | `weights/campp/` |
| `eres2net` | vendored arch + ckpt | 192 | identity | `weights/eres2net/eres2netv2.ckpt` |
| `titanet`  | NeMo        | 192 | identity | `weights/titanet/titanet_large.nemo` |

All weights are shipped here; every encoder loads from its `local_path` and
`allow_hub_download` is `false` everywhere. (`wavlm_large` weights are *not*
shipped because no `wavlm_best.pt` checkpoint was trained.)

## File map

- `submission.py` — competition entry point (required name)
- `inference.py` — inference core (`score_ensemble` + model/audio loading)
- `src/` — runtime code only (offline encoders, model factory, heads, pooling,
  fusion functions)
- `vendor/` — pure-Python deps absent from the leaderboard env (modelscope's
  runtime imports: addict/easydict/simplejson/yapf)
- `weights/` — pretrained encoder weights for the 4 used encoders
- `checkpoints/` — one trained `TwoHeadedSpeakerModel` per encoder
- `centroids/` — per-encoder speaker centroids (`centroids_<enc>.npz`, 192-d
  ArcFace space) for the cosine centroid + OOD-gate decision layer
- `decision_config.json` — tuned decision params (`alpha`, `kappa`, `tau`,
  `lambda_unknown`, `temperature`); absent → plain argmax fallback
- `ensemble_fusion_weights.json` — best fusion config (weights + encoder order)
- `README.md` — this file

## Offline guarantees

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` set automatically; no code
  path fetches from a hub with `allow_hub_download: false`.
- ModelScope models resolve via `MODELSCOPE_CACHE` pointed at `weights/campp`
  (shipped cache, never downloads).
- SpeechBrain ECAPA, NeMo TitaNet and the vendored ERes2NetV2 load purely from
  local files.
- Undecodable files fall back to a uniform `1/447` row, so every file still
  receives a valid prediction.
