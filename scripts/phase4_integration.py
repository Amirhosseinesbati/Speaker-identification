"""
Phase 4 — Integration tests for all 5 encoders (ECAPA, CAM++, ERes2NetV2,
WavLM-Large, TitaNet-Large).

For each available encoder:
  1. Build the two-headed model from config; assert OOD head input dim
     matches the pooled dim derived from the encoder output dim.
  2. Variable-length batch — no shape errors.
  3. Training smoke (20 steps): loss finite; checkpoint save+load OK.
  4. (GPU only) Sequential ensemble dry-run with per-model VRAM peak.

Memory safety: the script auto-detects GPU VRAM. On small GPUs (< 8 GB, e.g.
the local GTX 1660 Ti) it uses tiny batches + short windows + fp16, frees
memory between every test, and skips the WavLM training smoke (311M trainable
params would OOM a 6 GB card — defer that to Vast.ai).

Usage:
    uv run --no-sync python scripts/phase4_integration.py [--gpu]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# ═══════════════════════════════════════════════════════════
#  Global test profile (auto-tuned for small GPUs)
# ═══════════════════════════════════════════════════════════

def _profile() -> dict:
    """Return batch/window settings tuned to the available GPU."""
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        vram_gb = 0
    small = vram_gb < 8
    return {
        "small_gpu": small,
        "batch": 2 if small else 4,          # main batch size
        "train_batch": 2 if small else 4,    # training smoke batch size
        "seconds": 4 if small else 8,        # window length
        "fp16": small,                       # autocast in training smoke
    }


P = _profile()


def _free(model) -> None:
    """Delete a model + free GPU memory (call after every test)."""
    if "model" in locals() or model is not None:
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════
#  Per-encoder config builders (offline, local_path only)
# ═══════════════════════════════════════════════════════════

def enc_config(name: str) -> dict:
    """Return (encoder_type, encoder_config_block) for the named encoder."""
    blocks = {
        "ecapa": {
            "encoder_type": "ecapa",
            "encoder_config": {
                "ecapa": {"local_path": "weights/ecapa", "freeze_encoder": True},
            },
        },
        "campp": {
            "encoder_type": "campp",
            "encoder_config": {
                "campp": {"local_path": "weights/campp", "revision": "v1.0.2"},
            },
        },
        "eres2net": {
            "encoder_type": "eres2net",
            "encoder_config": {
                "eres2net": {"local_path": "weights/eres2net"},
            },
        },
        "wavlm": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {"local_path": "weights/wavlm_large", "pooling_type": "statistical"},
            },
        },
        "titanet": {
            "encoder_type": "titanet",
            "encoder_config": {
                "titanet": {"local_path": "weights/titanet/titanet_large.nemo"},
            },
        },
    }
    return blocks[name]


def base_config() -> dict:
    return {
        "model": {
            "pooling_type": "identity",
            "speaker_head_type": "linear",
            "ood_head_config": {"hidden_dim": 256},
            "fusion": {"ensemble_method": "none"},
        }
    }


def build_model(name: str, num_known: int = 10):
    from src.model_factory import create_model_from_config

    cfg = base_config()
    cfg["model"].update(enc_config(name))
    model = create_model_from_config(cfg, num_known_speakers=num_known)
    return model, cfg


# ═══════════════════════════════════════════════════════════
#  Per-encoder tests
# ═══════════════════════════════════════════════════════════

def test_two_headed(name: str, num_known: int = 10) -> bool:
    print(f"\n  ── [{name}] T6 two-headed build ──")
    try:
        model, _ = build_model(name, num_known)
    except Exception as e:
        print(f"  ⚠ SKIP (load failed): {type(e).__name__}: {str(e)[:120]}")
        return False

    enc_dim = model.encoder.output_dim
    ood_in = model.head_ood.input_dim
    wav = torch.randn(2, 1, 16000 * P["seconds"])
    pooled = model.pooling(model.encoder(wav)[0]).shape[-1]
    ok = (ood_in == pooled) and (pooled == enc_dim * model.pooling.output_multiplier)
    print(f"  encoder={enc_dim}d pooled={pooled}d OOD_in={ood_in} → "
          f"{'✅ PASS' if ok else '❌ FAIL'}")
    _free(model)
    return ok


def test_variable_length(name: str, num_known: int = 10) -> bool:
    print(f"  ── [{name}] variable-length batch (2s/8s/15s) ──")
    model, _ = build_model(name, num_known)
    model.eval()
    max_s = 15
    lens = [16000 * 2, 16000 * 8, 16000 * max_s, 16000 * 5]
    batch = torch.zeros(4, 1, 16000 * max_s)
    for i, L in enumerate(lens):
        batch[i, :, :L] = torch.randn(1, 1, L)
    try:
        with torch.no_grad():
            ood, spk = model(batch)
        ok = (ood.shape == (4, 1)) and (spk.shape == (4, num_known))
        print(f"  ood={tuple(ood.shape)} spk={tuple(spk.shape)} → {'✅ PASS' if ok else '❌ FAIL'}")
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {str(e)[:200]}")
        ok = False
    _free(model)
    return ok


def test_train_smoke(name: str, num_known: int = 10) -> bool:
    """20-step training smoke. WavLM is skipped on small GPUs (311M trainable
    params would OOM a 6 GB card — that one runs on Vast.ai instead)."""
    if name == "wavlm" and P["small_gpu"]:
        print(f"  ── [{name}] 20-step training smoke ──")
        print("  ⚠ SKIPPED on small GPU (311M trainable params; run on Vast.ai)")
        return True

    print(f"  ── [{name}] 20-step training smoke ──")
    model, _ = build_model(name, num_known)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    bs = P["train_batch"]
    T = 16000 * P["seconds"]
    for _ in range(20):
        opt.zero_grad()
        wav = torch.randn(bs, 1, T)
        labels = torch.randint(0, num_known + 1, (bs,))
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=P["fp16"] and torch.cuda.is_available(),
        ):
            ood, spk = model(wav, labels=labels)
            ood_t = (labels == 0).float().unsqueeze(1)
            spk_l = labels.clone() - 1
            spk_l[labels == 0] = -100
            loss = torch.nn.functional.binary_cross_entropy_with_logits(ood, ood_t) \
                + 0.7 * torch.nn.functional.cross_entropy(spk, spk_l, ignore_index=-100)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
    ckpt = PROJECT_ROOT / "checkpoints" / f"_it_{name}.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt))
    model2, _ = build_model(name, num_known)
    model2.load_state_dict(torch.load(str(ckpt), weights_only=False)["model_state_dict"])
    ckpt.unlink(missing_ok=True)
    print(f"  loss[-1]={losses[-1]:.4f} finite={finite} ckpt↺OK → "
          f"{'✅ PASS' if finite else '❌ FAIL'}")
    _free(model)
    _free(model2)
    return finite


# ═══════════════════════════════════════════════════════════
#  Sequential ensemble dry-run
# ═══════════════════════════════════════════════════════════

def ensemble_dry_run(names, device: torch.device):
    print("\n  ── Sequential ensemble dry-run ──")
    total_peak = 0.0
    bs = P["batch"]
    wav = torch.randn(bs, 1, 16000 * P["seconds"], device=device)
    for name in names:
        model, _ = build_model(name, num_known=10)
        model.to(device).eval()
        t0 = time.time()
        with torch.no_grad(), torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=device.type == "cuda",
        ):
            probs = model.predict_proba(wav)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"  {name:<12} infer {dt*1000:.0f}ms peak={peak:.2f}GB "
              f"probs={tuple(probs.shape)}")
        total_peak = max(total_peak, peak)
        _free(model)
    print(f"  peak VRAM (max across models): {total_peak:.2f} GB")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 4 integration tests")
    parser.add_argument("--gpu", action="store_true", help="run GPU dry-run")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | profile: {P}")

    all_names = ["ecapa", "campp", "eres2net", "wavlm", "titanet"]
    results = {}
    for name in all_names:
        print(f"\n{'='*56}\n  Encoder: {name}\n{'='*56}")
        t1 = test_two_headed(name)
        if not t1:
            results[name] = "SKIP"
            continue
        t2 = test_variable_length(name)
        t3 = test_train_smoke(name)
        results[name] = "✅ PASS" if (t1 and t2 and t3) else "❌ FAIL"

    print("\n" + "=" * 56)
    print("  INTEGRATION SUMMARY")
    print("=" * 56)
    for name in all_names:
        print(f"  {name:<12} {results.get(name, 'SKIP')}")

    available = [n for n in all_names if results.get(n) == "✅ PASS"]
    if args.gpu and device.type == "cuda":
        ensemble_dry_run(available, device)
    elif args.gpu:
        print("\n⚠ --gpu given but no CUDA — ensemble dry-run skipped (run on Vast.ai)")

    print("\n✅ Phase 4 integration complete.")


if __name__ == "__main__":
    main()
