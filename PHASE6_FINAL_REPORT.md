# Phase 6 — Final Report: Encoder Stack Refactoring

## 1. Encoder table (measured on this machine)

All shapes from the real `forward(waveforms[B,1,T]) → (hidden, lengths)` path.
Per-file numbers: GTX 1660 Ti, fp32 encoder + fp16 heads, real 58 s file,
8 × 8 s TTA windows (the worst case — `max_eval_windows=8`).

| encoder  | params (model) | emb dim | pooling    | load (offline) | VRAM peak* | ms/file (58s→8 win) | weights size |
|----------|---------------:|--------:|------------|----------------|-----------:|--------------------:|-------------:|
| `ecapa`    | 22.2 M (6.4 M enc) | 192 | identity | 1.0 s | 0.21 GB | 579 | 89 MB |
| `campp`    | 7.3 M         | 512 | identity | 0.7 s | 0.21 GB | 493 | 30 MB |
| `eres2net` | 17.9 M        | 192 | identity | 0.2 s | 0.15 GB | 330 | 72 MB |
| `titanet`  | 25.3 M        | 192 | identity | 6.5 s | 0.21 GB | 209 | 102 MB |
| `wavlm`    | 316.0 M       | 1024 | statistical | 0.2 s | 1.43 GB | 1203 | 1283 MB |
| **ensemble** | ~389 M      | —   | —        | ~9 s  | **1.43 GB** | **2814** | **1576 MB** |

\* VRAM peak = max concurrent model + activations (batch 2, 4 s) — all fit in
the 6.4 GB card; on a 24 GB 3090 there is plenty of headroom for batch 32.

Pooling is **config-driven per encoder** (`model.encoder_config.<type>.pooling_type`
overrides the global), so the OOD head input dim is always derived correctly:
192 / 512 / 192 / 192 / 2048 (= 1024×2 statistical).

## 2. Offline loading — verified

| encoder  | mechanism | hub fetch at inference |
|----------|-----------|:---:|
| `ecapa`    | SpeechBrain `from_hparams(savedir=weights/ecapa, COPY)` | ❌ none |
| `campp`    | ModelScope `Model.from_pretrained` + `snapshot_download(local_files_only=True)` | ❌ none |
| `eres2net` | vendored `src/sv_arch.py` + `torch.load` ckpt (no modelscope/3dspeaker!) | ❌ none |
| `titanet`  | NeMo `restore_from(weights/titanet/titanet_large.nemo)` | ❌ none |
| `wavlm`    | HF `from_pretrained(weights/wavlm_large, local_files_only=True)` | ❌ none |

Offline simulation passed: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODELSCOPE_CACHE=<pkg>/weights/campp`, 2-model ensemble → 448-column CSV,
row sums = 1.0, all finite.

## 3. Runtime estimate vs 20-min budget

Measured (1660 Ti, batch 1, 8 windows/file): **2 814 ms/file** for the 5-model
ensemble. The target is the RTX 3090 (24 GB, fp16, bs=32).

Scaling to the 3090: the 1660 Ti has no tensor cores (~2.2 TFLOPS fp16
effective); the 3090 reaches ~142 TFLOPS fp16. A documented, conservative
**10×** factor (real-world Ampere speedup + better memory bandwidth), plus a
**2× safety factor** as the prompt requires. Windows/file = 4 for 15 s audio
(8 s windows, 50 % overlap), 1 window if the eval chunk ≤ 8 s.

| N files | windows | 3090 est. (batch 32) | ×2 safety | vs 20 min | verdict |
|--------:|--------:|---------------------:|----------:|:---------:|:-------:|
| 5 000 | 20 k | 8.3 min | 16.7 min | < 20 | ✅ **PASS** |
| 20 000 | 80 k | 33 min | 67 min | > 20 | ❌ **FAIL** (short chunks: PASS) |
| 25 000 | 100 k | 42 min | 83 min | > 20 | ❌ **FAIL** (short chunks: borderline) |

> **Short-chunk scenario:** the competition evaluates *short audio chunks*
> (per the official guide). If eval files are ≤ 8 s, they need **1 window**
> instead of 4 → all numbers ÷4 → **20 k and 25 k PASS**.

**Levers if 25 k × 15 s must fit in 20 min (ranked):**
1. **Batch files** (bs=32, sort-by-length) — the single biggest win; the
   measured numbers are batch-1.
2. fp16 tensor-core autocast on Ampere (planned; WavLM encoder itself must
   stay fp32 — see §4).
3. Trim `max_eval_windows` (8 → 4) for the 15 s case.
4. Drop WavLM (≈43 % of the ensemble time) only if Macro-F1 suffers too much.

## 4. Known issues found & fixed

- **WavLM-Large NaN under fp16 autocast** (attention-logit overflow →
  softmax NaN — reproduced on noise AND real speech). Fix: WavLM forward runs
  in fp32 (`autocast(enabled=False)`); heads still fp16. Verified: valid
  probs, and it's *faster* than the NaN path on this GPU.
- **NeMo 2.7 dropped tensor `get_embedding(input_signal, …)`** — now file-path
  only. Fix: run `preprocessor → encoder → decoder` directly; `length` must be
  in **samples** (passing ones() → 0 frames → NaN).
- **ModelScope `SpeechEmbedding` class doesn't exist in modelscope 1.39** —
  the pipeline wrapper is `SpeakerVerificationCAMPPlus` (registered class)
  loaded via `Model.from_pretrained`. Offline snapshot needs the explicit
  `revision` (cache lookup otherwise targets `snapshots/None`).
- **`nn.Module.to()` bypasses `to()` overrides** (`_apply` path) → wrapper
  `.device` stayed CPU → CUDA/cpu conv mismatch. Fix: re-sync device from the
  actual parameter device in `_modelscope_forward`.
- **ERes2NetV2 is 192-dim, not 512** (official VoxCeleb checkpoint, 17.86 M
  params, `seg_1.weight (192, 20480)`). The prompt's "512 (V)" is resolved:
  it's 192.
- SpeechBrain `from_hparams` symlinks fail on Windows → `LocalStrategy.COPY`
  (also makes the weights dir zip-portable).

## 5. Remaining risks / manual verification (on Vast.ai)

1. **3090 runtime** — measure `inference.py` wall-time on the real 24 GB GPU
   with bs=32; confirm the §3 estimate. The local card is not representative.
2. **WavLM training smoke** — skipped on the 6 GB card (311 M trainable
   params would OOM). Run `scripts/phase4_integration.py --gpu` on Vast.ai
   with the full profile (it auto-detects VRAM ≥ 8 GB and enables the WavLM
   training smoke).
3. **NeMo on Linux** — the `restore_from` path was validated on Windows; run
   the same test on the Linux leaderboard env.
4. **Checkpoints** — the submission package expects one trained
   `checkpoints/<enc>_best.pt` per encoder; train all 5 and re-run
   `scripts/build_submission.py` (idempotent, skips existing weights).
5. **`leaderbordpakage.txt` pins** — the local venv now matches the server
   (transformers 4.57, hf-hub 0.36, modelscope 1.39, nemo 2.7.3). On the
   server, `pip install` from that file must not upgrade torch's CUDA build
   (use `uv run --no-sync` semantics).

## 6. Zip size estimate

| component | size |
|-----------|-----:|
| weights/ (5 encoders) | 1 576 MB |
| checkpoints/ (5 trained models, ~100 MB each) | ~500 MB |
| src/ + configs + inference.py + README | ~1 MB |
| **submission.zip** | **~2.1 GB** (≈ 1.9 GB zipped) |
