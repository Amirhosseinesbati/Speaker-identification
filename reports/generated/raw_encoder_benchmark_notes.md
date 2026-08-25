# Raw encoder benchmark

Scope: 4,459 clean cached files (2,232 known and 2,227 unknown); 70 corrupt files were excluded. Known-speaker accuracy is leave-one-out centroid scoring. Pair separation uses 2,231 same-speaker and 8,907 different-speaker sampled pairs.

| Encoder | dim | LOO top-1 | LOO top-5 | OOD AUC | best Macro-F1 | d-prime |
|---|---:|---:|---:|---:|---:|---:|
| ERes2NetV2 (existing) | 192 | 0.9579 | 0.9664 | 0.9599 | 0.9258 | 3.456 |
| TitaNet (existing) | 192 | 0.9539 | 0.9659 | **0.9626** | **0.9331** | 3.860 |
| ECAPA (existing) | 192 | 0.9525 | 0.9619 | 0.9585 | 0.9197 | **3.675** |
| CAM++ (existing) | 512 | 0.9503 | 0.9597 | 0.9569 | 0.9249 | **3.996** |
| WavLM-base-SV (8-sec center crop) | 512 | 0.5995 | 0.8011 | 0.6797 | 0.4442 | 1.630 |
| WavLM-base-plus-SV (8-sec center crop) | 512 | 0.6510 | 0.8289 | 0.7144 | 0.4932 | 1.597 |

The WavLM-SV results are directly reproducible from `scripts/benchmark_wavlm_sv.py`; their embeddings and JSON reports are cached alongside the existing arrays. The WavLM-SV models are English VoxCeleb fine-tunes, while this challenge data is not guaranteed to match that domain; the large gap is therefore evidence against using them raw here, not evidence that WavLM is intrinsically weak.

Downloaded official ModelScope variants (ready for an adapter benchmark): `speech_eres2net_sv_zh-cn_16k-common`, `speech_eres2net_base_sv_zh-cn_3dspeaker_16k`, and `speech_campplus_sv_zh_en_16k-common_advanced` under `weights/modelscope_variants/`. They use ModelScope/3D-Speaker wrapper formats and are not interchangeable with the current checkpoint loaders without a small adapter.
