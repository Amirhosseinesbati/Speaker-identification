# IAAA 2026 Speaker Identification — Submission Package

Open-set speaker identification: classify audio into **447 classes** (446
known speaker IDs plus the aggregated `unknown` class), scored with Macro-F1
over the 447 argmax predictions.

This folder is self-contained. All runtime code, pretrained encoder weights,
trained checkpoints, and any enabled decision-layer artifacts are shipped in
the package. Inference does not require network access or external files.

## Leaderboard entry point

```bash
python submission.py --data-dir <test-set-folder> --predictions-file-path predictions.csv
```

`submission.py` implements the required `load_data`, `predict`, and
`save_predictions` contract. The exact deployed recipe is pinned by
`ensemble_fusion_weights.json`; do not infer the package contents from this
README. At runtime it:

- loads only the encoder/checkpoint entries listed in the manifest;
- applies each model's pinned multi-window inference policy;
- combines models with the fixed manifest weights, if more than one model is
  present;
- applies the shipped decision-layer configuration only when the manifest
  enables it (for example the locked CAM++ LME-20 prototype backend);
- maps the final 447-way argmax back to a speaker UUID or `unknown`;
- forces Hugging Face, Transformers, and ModelScope into offline mode.

All thresholds, fusion weights, epochs, and prototype parameters must come
from the committed manifest/config artifacts. They are not selected from a
leaderboard result at inference time.

## Output format

The output CSV contains exactly `audio_file,speaker_id`:

```csv
audio_file,speaker_id
e9105299-285b-4df7-8c66-e4b3b721e8c8.mp3,3943d8f3-d820-44ff-aba6-23c796fda87b
016a2324-dc96-4f59-8d96-fb387afb1fff.mp3,845cbfc6-af92-4fa0-8b33-4e4e0e3b002a
57e9178b-7153-4e43-8fe4-6aacea3c9118.mp3,unknown
```

- `audio_file` is the test file's full name, including its extension.
- `speaker_id` is a known-speaker UUID or the literal `unknown`.

## Deployed model and backend

`ensemble_fusion_weights.json` is the authoritative deployment receipt. Its
`encoder_names`, `checkpoints`, `weights`, and decision-layer flags identify
the exact models in a particular ZIP. A package may contain a single CAM++
model or a fixed ensemble; unused experimental encoders are not implied.

When the locked CAM++ LME-20 backend is enabled, the package additionally
contains full train-only enrollment embeddings in `prototypes/` and a fixed
decision configuration. Similarities are pooled per identity with
`log(mean(exp(20*cosine)))/20`; all train-only OOD identities are then
collapsed into the competition's single `unknown` class. Packages that do not
enable this backend use the manifest's declared argmax/open-set path instead.

## File map

- `submission.py` — required competition entry point.
- `inference.py` — audio/model loading and scoring runtime.
- `ensemble_fusion_weights.json` — authoritative model order, checkpoint
  names, fusion weights, and decision-layer flags.
- `decision_config.json` — optional fixed decision parameters; its absence
  means no such decision layer is enabled.
- `src/` — runtime model, pooling, and fusion code.
- `vendor/` — pure-Python dependencies absent from the leaderboard image.
- `weights/` — only the pretrained encoder assets required by the manifest.
- `checkpoints/` — only the trained checkpoints required by the manifest.
- `prototypes/` or `centroids/` — optional train-only backend artifacts when
  the manifest enables them.
- `README.md` — this runtime contract.

## Offline guarantees

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set automatically.
- ModelScope caches resolve to shipped local directories.
- Every manifest-listed checkpoint and encoder asset is packaged locally.
- Undecodable inputs fall back to a valid uniform `1/447` probability row, so
  every input still receives a prediction.
