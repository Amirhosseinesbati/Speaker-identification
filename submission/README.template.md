# IAAA 2026 Speaker Identification — Submission Package

Open-set speaker identification: classify audio into **447 classes** (446
known speaker-ids + aggregated `unknown`), scored on Macro-F1.

## Encoder ensemble (5 models)

| key | framework | emb dim | pooling | weights (local) |
|-----|-----------|---------|---------|-----------------|
| `ecapa`    | SpeechBrain  | 192  | identity    | `weights/ecapa/` |
| `campp`    | ModelScope   | 512  | identity    | `weights/campp/` |
| `eres2net` | vendored arch + ckpt | 192 | identity | `weights/eres2net/eres2netv2.ckpt` |
| `titanet`  | NeMo         | 192  | identity    | `weights/titanet/titanet_large.nemo` |
| `wavlm`    | transformers | 1024 | statistical | `weights/wavlm_large/` |

All weights are shipped in this package; **no internet is required** at
inference time. Every encoder loads from its `local_path` and
`allow_hub_download` is `false` everywhere.

## Run

```bash
python -m submission.inference \
    --data-dir <test-set-folder> \
    --predictions-file-path predictions.csv \
    --checkpoint-path checkpoints/ecapa_best.pt \
    --checkpoint-path checkpoints/campp_best.pt \
    --checkpoint-path checkpoints/eres2net_best.pt \
    --checkpoint-path checkpoints/titanet_best.pt \
    --checkpoint-path checkpoints/wavlm_best.pt
```

(Each checkpoint embeds its own training config, so no shared config file is
needed; `--config-path configs/inference_config.yaml` is only a fallback.)

Output: a CSV with columns `id,0,1,...,446` — each row sums to 1.0 — plus a
sidecar `<output>.class_map.json` mapping column index → speaker-id.

## File map

- `src/` — project code (`encoders.py` with the 5 offline encoders,
  `model_factory.py`, `model.py`, `pooling.py`, `heads.py`, `train.py`,
  `sv_arch.py` [vendored ERes2NetV2], `data_pipeline.py`, `metrics.py`, …)
- `configs/inference_config.yaml` — offline inference config
- `weights/` — the 5 pretrained encoder weight sets (see table)
- `checkpoints/` — one trained `TwoHeadedSpeakerModel` per encoder
- `inference.py` — competition entrypoint (sequential ensemble, fp16 autocast,
  TTA, per-model wall-time logging)
- `README.md` — this file

## Offline guarantees

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` can be set as belt-and-braces;
  no code path performs a hub fetch with `allow_hub_download: false`.
- ModelScope models resolve via `MODELSCOPE_CACHE` pointed at `weights/campp`
  (the pipeline reads from the shipped cache, never downloads).
- SpeechBrain ECAPA, NeMo TitaNet and the vendored ERes2NetV2 load purely from
  local files (`torch.load` / `restore_from`).
