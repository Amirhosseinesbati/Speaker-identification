"""
ZenML Pipeline Steps for Open-Set Speaker Identification.

Each step wraps existing functions from src/data_pipeline.py, src/model.py,
and src/train.py, adding MLflow experiment tracking via the ZenML
MLflowExperimentTracker.

Pipeline flow:
  1. prepare_data      → config, class_map, train_df, val_df
  2. build_model       → model_checkpoint_path
  3. train_model       → training_history, best_model_path
  4. evaluate_model    → evaluation_metrics
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from zenml import step
from torch.utils.data import DataLoader

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (
    load_config,
    get_active_profile,
    create_class_mapping,
    prepare_clean_split,
    make_balanced_batch_sampler,
    SpeakerDataset,
)
from src.metrics import evaluate_macro_f1
from src.train import (
    train_epoch,
    validate_epoch,
    TwoPartLoss,
    forward_multi_window,
    compute_ood_accuracy,
    compute_speaker_accuracy,
    setup_device,
)


# ─────────────────────────────────────────────────────────
#  MLflow Helper — standalone tracking (not dependent on ZenML)
# ─────────────────────────────────────────────────────────

from src.mlflow_helper import get_tracker, MLflowTracker

def _mlflow_active() -> bool:
    """Check if MLflow has an active run."""
    return get_tracker().is_active


def _mlflow_log_params(params: dict):
    if get_tracker().is_active:
        get_tracker().log_params(params)


def _mlflow_log_metrics(metrics: dict, step: int = None):
    if get_tracker().is_active:
        get_tracker().log_metrics(metrics, step=step)


def _mlflow_log_artifact(path: str, artifact_path: str = None):
    if get_tracker().is_active:
        get_tracker().log_artifact(path, artifact_path)


# ─────────────────────────────────────────────────────────
#  Step 0: Convert Audio (MP3 → WAV)
# ─────────────────────────────────────────────────────────

@step
def convert_audio(
    config_path: str = "configs/default_config.yaml",
) -> Dict:
    """
    Convert raw MP3 files to WAV (mono 16kHz) for reliable dataloading.

    This step:
    1. Loads config from config_path
    2. Checks if WAV files already exist (>4000 = skip)
    3. Converts all MP3 → WAV using FFmpeg (fast) or librosa (fallback)
    4. Generates updated labels CSV pointing to .wav files
    5. Returns updated config dict with WAV paths

    Returns:
        Updated config dict with audio_dir and labels_path pointing to WAV
    """
    import subprocess
    from pathlib import Path
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor

    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import yaml
    from tqdm import tqdm

    print("=" * 55)
    print("  [ZenML Step 0/5] Converting MP3 → WAV")
    print("=" * 55)

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = Path("data/raw")
    wav_dir = Path("data/processed/audio_wav")
    wav_labels_path = Path("data/processed/audio_wav_labels.csv")

    # ── Check if already converted ──
    wav_dir.mkdir(parents=True, exist_ok=True)
    existing_wavs = list(wav_dir.glob("*.wav"))
    if len(existing_wavs) > 4000:
        print(f"  ✅ {len(existing_wavs)} WAV files already exist. Skipping conversion.")
        config["data"]["audio_dir"] = str(wav_dir)
        config["data"]["labels_path"] = str(wav_labels_path)
        return config

    # ── Load raw labels ──
    raw_labels = raw_dir / "labels.csv"
    if not raw_labels.exists():
        raise FileNotFoundError(f"Raw labels not found: {raw_labels}")

    df = pd.read_csv(raw_labels)
    df.columns = df.columns.str.strip()
    print(f"  Raw labels: {len(df)} rows")

    # ── Find MP3 files to convert ──
    mp3_files = list(raw_dir.glob("*.mp3"))
    print(f"  MP3 files found: {len(mp3_files)}")
    print(f"  Converting to: {wav_dir}")

    # ── Convert (parallel) ──
    def _convert_one(mp3_path):
        wav_path = wav_dir / (mp3_path.stem + ".wav")
        if wav_path.exists():
            return "skip"
        try:
            # Try FFmpeg first (fast)
            result = subprocess.run(
                ["ffmpeg", "-i", str(mp3_path), "-ac", "1", "-ar", "16000",
                 "-sample_fmt", "s16", "-v", "error", "-y", str(wav_path)],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and wav_path.stat().st_size > 1000:
                return "ok"
            # FFmpeg failed — fallback to librosa
            wav_path.unlink(missing_ok=True)
            y, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
            sf.write(str(wav_path), y, 16000, subtype="PCM_16")
            return "ok"
        except Exception:
            return "fail"

    converted, skipped, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(tqdm(
            pool.map(_convert_one, mp3_files),
            total=len(mp3_files),
            desc="  Converting",
        ))
    for r in results:
        if r == "ok": converted += 1
        elif r == "skip": skipped += 1
        else: failed += 1

    print(f"  ✅ Converted: {converted} | Skipped: {skipped} | Failed: {failed}")

    # ── Build updated labels CSV ──
    df["audio_file"] = df["audio_file"].apply(lambda x: Path(x).stem + ".wav")
    df.to_csv(wav_labels_path, index=False)
    print(f"  Labels saved: {wav_labels_path} ({len(df)} rows)")

    # ── Update config and save to file ──
    config["data"]["audio_dir"] = str(wav_dir)
    config["data"]["labels_path"] = str(wav_labels_path)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Config updated: audio_dir → {wav_dir}")

    # ── Start MLflow run ──
    tracker = get_tracker(config)
    encoder_type = config.get("model", {}).get("encoder_type", "unknown")
    run_name = f"{encoder_type}-{datetime.now().strftime('%m%d-%H%M')}"
    tracker.start_run(run_name=run_name)
    tracker.log_params({
        "encoder_type": encoder_type,
        "pipeline_stage": "convert_audio",
        "wav_files_converted": converted,
    })
    tracker.log_code_snapshot()
    tracker.log_config_snapshot()

    return config


# ─────────────────────────────────────────────────────────
#  Step 1: Prepare Data
# ─────────────────────────────────────────────────────────

@step
def prepare_data(
    config_path: str = "configs/default_config.yaml",
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Load config, prepare labels, and perform leak-free stratified split
    (corrupted / MD5-duplicate files filtered; split_report.json written).

    Returns:
        config, class_map, train_df, val_df
    """
    print("=" * 55)
    print("  [ZenML Step 1/4] Preparing Data")
    print("=" * 55)

    config = load_config(config_path)
    data_cfg = config["data"]
    audio_cfg = config["audio"]

    # ── Verify audio paths exist ──
    labels_path = data_cfg["labels_path"]
    audio_dir = data_cfg["audio_dir"]

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}\n"
            f"Run: python scripts/convert_mp3_to_wav.py"
        )
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    # Leak-free split with corrupted/duplicate filtering + split_report.json
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=labels_path,
        audio_dir=audio_dir,
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", 1.0),
    )

    print(f"  ✓ Train: {len(train_df)} samples | Val: {len(val_df)} samples")
    print(f"  ✓ Classes: {len(class_map)} (0=unknown, 1..{len(class_map)-1}=known)")

    # Log data params to MLflow
    _mlflow_log_params({
            "num_classes": len(class_map),
            "num_known": len(class_map) - 1,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_known": int((train_df["label"] != 0).sum()),
            "train_unknown": int((train_df["label"] == 0).sum()),
        })

    return config, class_map, train_df, val_df


# ─────────────────────────────────────────────────────────
#  Step 2: Build Model
# ─────────────────────────────────────────────────────────

@step
def build_model(
    config: Dict,
    class_map: Dict,
) -> str:
    """
    Instantiate the TwoHeadedSpeakerModel via factory and save its initial state.

    Returns:
        checkpoint_path: path to the saved initial model checkpoint (state_dict only).
    """
    print("=" * 55)
    print("  [ZenML Step 2/4] Building Model")
    print("=" * 55)

    num_known = config.get("model", {}).get("competition_num_known", len(class_map) - 1)
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)

    model.print_summary()

    # Save initial state dict to a temp checkpoint
    log_cfg = config.get("logging", {})
    ckpt_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    init_path = str(ckpt_dir / "init_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "class_map": class_map,
    }, init_path)
    print(f"  ✓ Initial model saved to {init_path}")

    # Log model architecture params (backward-compat config reading)
    model_cfg = config.get("model", {})
    encoder_type = model_cfg.get("encoder_type", "wavlm")
    if "encoder_config" in model_cfg:
        enc_cfg = model_cfg["encoder_config"].get(encoder_type, {})
        base_model = enc_cfg.get("base_model", enc_cfg.get("source", "unknown"))
        freeze_fe = enc_cfg.get("freeze_feature_extractor",
                                enc_cfg.get("freeze_encoder", True))
    else:
        base_model = model_cfg.get("base_model", "unknown")
        freeze_fe = model_cfg.get("freeze_feature_extractor", True)

    _mlflow_log_params({
        "encoder_type": encoder_type,
        "base_model": base_model,
        "freeze_encoder": freeze_fe,
        "pooling_type": model_cfg.get("pooling_type", "statistical"),
        "speaker_head_type": model_cfg.get("speaker_head_type", "linear"),
        "num_known_speakers": num_known,
        "total_params": model.get_trainable_params(),
    })

    return init_path


# ─────────────────────────────────────────────────────────
#  Step 3: Train Model
# ─────────────────────────────────────────────────────────

@step
def train_model(
    config: Dict,
    class_map: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_checkpoint_path: str,
) -> Tuple[str, Dict]:
    """
    Full training loop with MLflow autologging.
    Saves best and latest checkpoints.

    Returns:
        (best_model_path, training_summary_dict)
    """
    print("=" * 55)
    print("  [ZenML Step 3/4] Training Model")
    print("=" * 55)

    # ── Device ──
    device = setup_device(config)

    # ── DataLoaders (from train_df/val_df directly) ──
    audio_cfg = config["audio"]
    data_cfg = config["data"]
    hw_profile = get_active_profile(config)

    # ── Filter short/corrupted files (min_valid_duration) ──
    min_valid_duration = audio_cfg.get("min_valid_duration", 0.0)
    if min_valid_duration > 0:
        # NOTE: use soundfile.info (header-only, C-extension) instead of
        # librosa.get_duration. librosa's lazy submodule loader calls
        # inspect.stack() which trips over speechbrain's LazyModule
        # (`speechbrain.integrations.k2_fsa` → missing `k2` package) and
        # raises an ImportError. soundfile has no such interaction.
        import soundfile as sf
        all_files = set(train_df["audio_file"].unique()) | set(val_df["audio_file"].unique())
        short_files = set()
        audio_dir_path = Path(data_cfg["audio_dir"])
        corrupted_log = []
        for fname in all_files:
            fpath = audio_dir_path / fname
            if fpath.exists():
                try:
                    dur = sf.info(str(fpath)).duration
                    if dur < min_valid_duration:
                        short_files.add(fname)
                        corrupted_log.append({"file": fname, "duration": round(dur, 4), "reason": "too_short"})
                except Exception:
                    short_files.add(fname)
                    corrupted_log.append({"file": fname, "duration": 0, "reason": "load_error"})
            else:
                short_files.add(fname)
                corrupted_log.append({"file": fname, "duration": 0, "reason": "missing"})
        if short_files:
            print(f"  ⚠ Filtering {len(short_files)} short/corrupted files (< {min_valid_duration}s)")
            train_df = train_df[~train_df["audio_file"].isin(short_files)].reset_index(drop=True)
            val_df = val_df[~val_df["audio_file"].isin(short_files)].reset_index(drop=True)
            # Save corrupted list for debugging
            import json
            ckpt_dir = Path(config.get("logging", {}).get("checkpoint_dir", "checkpoints"))
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "corrupted_files.json", "w") as f:
                json.dump(corrupted_log, f, indent=2)
            print(f"    Corrupted list saved to {ckpt_dir / 'corrupted_files.json'}")

    train_dataset = SpeakerDataset(
        df=train_df,
        audio_dir=data_cfg["audio_dir"],
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=True,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    val_dataset = SpeakerDataset(
        df=val_df,
        audio_dir=data_cfg["audio_dir"],
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )

    train_labels = train_df["label"].values
    ood_ratio = audio_cfg.get("ood_batch_ratio", 0.5)
    balanced_indices = make_balanced_batch_sampler(
        train_labels, hw_profile["batch_size"], ood_ratio=ood_ratio,
    )
    sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hw_profile["batch_size"],
        sampler=sampler,
        num_workers=hw_profile["num_workers"],
        pin_memory=(hw_profile["device"] == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hw_profile["batch_size"],
        shuffle=False,
        num_workers=hw_profile["num_workers"],
        pin_memory=(hw_profile["device"] == "cuda"),
        drop_last=False,
    )

    # ── Load model ──
    num_known = config.get("model", {}).get("competition_num_known", len(class_map) - 1)
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # ── Optimizer, Loss, Scheduler, Scaler ──
    train_cfg = config["training"]
    # Separate LR for unfrozen encoder blocks (fine-tuning) vs the heads
    encoder_params = [p for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "encoder" not in n and p.requires_grad]
    param_groups = [{"params": head_params, "lr": train_cfg["learning_rate"]}]
    if encoder_params:
        encoder_lr = train_cfg.get("encoder_lr", 1e-5)
        param_groups.insert(0, {"params": encoder_params, "lr": encoder_lr})
        print(f"  🔓 Encoder LR: {encoder_lr:.2e} ({len(encoder_params):,} tensors) | "
              f"Head LR: {train_cfg['learning_rate']:.2e}")
    else:
        print(f"  🔒 Encoder fully frozen — single LR {train_cfg['learning_rate']:.2e}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg["weight_decay"],
    )
    # Linear warmup (3 epochs) + CosineAnnealingWarmRestarts
    warmup_epochs = 3
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs,
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )
    criterion = TwoPartLoss(
        ignore_index=-100,
        use_focal=True,
        focal_gamma=2.0,
        ood_weight=train_cfg.get("ood_loss_weight", 1.0),
        speaker_weight=train_cfg.get("speaker_loss_weight", 1.0),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
        ood_pos_weight=train_cfg.get("ood_pos_weight", 1.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=hw_profile["mixed_precision"])

    # ── Training Loop with MLflow autologging ──
    log_cfg = config.get("logging", {})
    checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -float("inf")
    best_epoch = -1
    patience_counter = 0
    early_stop_patience = train_cfg.get("early_stopping_patience", 10)
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, train_cfg["max_grad_norm"],
            ood_grad_norm=train_cfg.get("ood_grad_norm", 1.0),
        )
        # Validate + competition metric (Macro-F1 over all 447 classes)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_m = evaluate_macro_f1(
            val_metrics["ood_logits"], val_metrics["speaker_logits"],
            val_metrics["labels"], num_classes=len(class_map),
        )
        val_metrics["macro_f1"] = val_m["macro_f1"]
        val_metrics["ood_f1"] = val_m["ood_f1"]
        val_metrics["known_acc"] = val_m["known_acc"]
        val_metrics["overall_acc"] = val_m["overall_acc"]
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics per epoch
        _mlflow_log_metrics({
            "train_loss": train_metrics["loss"],
            "train_ood_acc": train_metrics["ood_acc"],
            "train_speaker_acc": train_metrics["speaker_acc"],
            "val_loss": val_metrics["loss"],
            "val_ood_acc": val_metrics["ood_acc"],
            "val_speaker_acc": val_metrics["speaker_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
        }, step=epoch)

        # Print progress
        print(f"\n  Epoch {epoch:2d}/{train_cfg['epochs']} — "
              f"Loss: {train_metrics['loss']:.4f} / {val_metrics['loss']:.4f}  |  "
              f"OOD: {train_metrics['ood_acc']:.3f} / {val_metrics['ood_acc']:.3f}  |  "
              f"Spk: {train_metrics['speaker_acc']:.3f} / {val_metrics['speaker_acc']:.3f}  |  "
              f"MacroF1: {val_metrics['macro_f1']:.4f}")

        # Save best model (based on val Macro-F1) + Early stopping (also Macro-F1)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "class_map": class_map,
                "val_loss": val_metrics["loss"],
                "val_ood_acc": val_metrics["ood_acc"],
                "val_speaker_acc": val_metrics["speaker_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }, best_path)
        else:
            patience_counter += 1

        # Record history (BEFORE early stopping check — prevents crash on break)
        # NOTE: exclude tensors (collect for Macro-F1) from the history dict
        val_history = {k: v for k, v in val_metrics.items()
                       if not isinstance(v, torch.Tensor)}
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_history.items()},
            "lr": current_lr,
        })

        # Save latest checkpoint (BEFORE early stopping check)
        latest_path = checkpoint_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "class_map": class_map,
        }, latest_path)

        # Early stopping based on val Macro-F1 (no improvement for N epochs)
        if patience_counter >= early_stop_patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} "
                  f"(val_macro_f1 not improved for {early_stop_patience} epochs)")
            break

    # ── Final logging ──
    final_best_path = str(checkpoint_dir / "best_model.pt")

    # Safety: if early stopping happened before any improvement, use last epoch
    safe_idx = best_epoch - 1
    if safe_idx < 0 or safe_idx >= len(history):
        safe_idx = len(history) - 1  # fallback to last recorded epoch

    summary = {
        "best_epoch": best_epoch if best_epoch > 0 else (safe_idx + 1),
        "best_val_macro_f1": round(best_val_f1, 6),
        "best_val_ood_acc": round(history[safe_idx]["val_ood_acc"], 4),
        "best_val_speaker_acc": round(history[safe_idx]["val_speaker_acc"], 4),
        "total_epochs_run": len(history),
        "total_epochs_configured": train_cfg["epochs"],
    }

    _mlflow_log_params({
        "epochs": train_cfg["epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "batch_size": hw_profile["batch_size"],
    })
    _mlflow_log_metrics(summary)
    _mlflow_log_artifact(final_best_path, artifact_path="models")

    print(f"\n  ✓ Training complete! Best val Macro-F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"  ✓ Best model: {final_best_path}")

    return final_best_path, summary


# ─────────────────────────────────────────────────────────
#  Step 4: Evaluate Model
# ─────────────────────────────────────────────────────────

@step
def evaluate_model(
    config: Dict,
    class_map: Dict,
    val_df: pd.DataFrame,
    best_model_path: str,
) -> Dict:
    """
    Load the best checkpoint and run final evaluation on validation set.

    Returns:
        metrics dict with final OOD and Speaker accuracy.
    """
    print("=" * 55)
    print("  [ZenML Step 4/4] Evaluating Model")
    print("=" * 55)

    device = setup_device(config)
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    # Validation dataset (no augmentation)
    val_dataset = SpeakerDataset(
        df=val_df,
        audio_dir=data_cfg["audio_dir"],
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    hw_profile = get_active_profile(config)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hw_profile["batch_size"],
        shuffle=False,
        num_workers=hw_profile["num_workers"],
        drop_last=False,
    )

    # Load model
    num_known = config.get("model", {}).get("competition_num_known", len(class_map) - 1)
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Evaluate with threshold tuning (criterion mirrors training weights)
    train_cfg = config["training"]
    criterion = TwoPartLoss(
        ignore_index=-100,
        use_focal=True,
        focal_gamma=2.0,
        ood_weight=train_cfg.get("ood_loss_weight", 1.0),
        speaker_weight=train_cfg.get("speaker_loss_weight", 1.0),
        label_smoothing=0.0,  # eval: no label smoothing (standard)
        ood_pos_weight=train_cfg.get("ood_pos_weight", 1.0),
    )
    total_loss = 0.0
    total_ood_acc = 0.0
    total_speaker_acc = 0.0
    num_batches = len(val_loader)

    all_ood_logits = []
    all_speaker_logits = []
    all_labels = []

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # No labels → no ArcFace margin → honest eval logits
            ood_logits, speaker_logits = forward_multi_window(model, waveforms, labels=None)
            loss, loss_dict = criterion(ood_logits, speaker_logits, labels)

            total_loss += loss_dict["loss_total"]
            total_ood_acc += compute_ood_accuracy(ood_logits, labels)
            speaker_acc, _ = compute_speaker_accuracy(speaker_logits, labels)
            total_speaker_acc += speaker_acc

            all_ood_logits.append(ood_logits.cpu())
            all_speaker_logits.append(speaker_logits.cpu())
            all_labels.append(labels.cpu())

    all_ood = torch.cat(all_ood_logits)
    all_spk = torch.cat(all_speaker_logits)
    all_lbl = torch.cat(all_labels)

    # ── Competition metric (plain argmax — exactly what the organizers score) ──
    argmax_metrics = evaluate_macro_f1(
        all_ood, all_spk, all_lbl, num_classes=len(class_map),
    )

    # ── Tune OOD threshold on validation (binary unknown-class F1) ──
    ood_probs = torch.sigmoid(all_ood.squeeze(1)).numpy()
    ood_targets = (all_lbl == 0).numpy().astype(int)

    from sklearn.metrics import f1_score
    best_threshold = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.9, 0.05):
        preds = (ood_probs > thr).astype(int)
        f1 = f1_score(ood_targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    # Fallback: if the OOD head collapsed (all F1 == 0), use the median positive
    # rate of the val set so a sane threshold is always persisted.
    if best_f1 == 0.0:
        best_threshold = float(np.median(ood_probs))
        print(f"  ⚠ OOD head collapsed (F1=0) — falling back to median "
              f"P(unknown)={best_threshold:.3f}")

    # Macro-F1 at the tuned threshold (local OOD operating-point analysis)
    thr_metrics = evaluate_macro_f1(
        all_ood, all_spk, all_lbl, num_classes=len(class_map),
        ood_threshold=best_threshold,
    )

    tuned_ood_acc = ((ood_probs > best_threshold).astype(int) == ood_targets).mean()

    print(f"  🎯 OOD Threshold tuned: {best_threshold:.2f} (binary F1={best_f1:.4f})")
    print(f"     Default (0.5): OOD Acc = {total_ood_acc/num_batches:.4f}")
    print(f"     Tuned ({best_threshold:.2f}): OOD Acc = {tuned_ood_acc:.4f}")
    print(f"     Macro-F1 (argmax):   {argmax_metrics['macro_f1']:.4f}")
    print(f"     Macro-F1 (thr={best_threshold:.2f}): {thr_metrics['macro_f1']:.4f}")

    metrics = {
        "final_val_loss": round(total_loss / num_batches, 6),
        "final_val_ood_acc": round(total_ood_acc / num_batches, 4),
        "final_val_speaker_acc": round(total_speaker_acc / num_batches, 4),
        "macro_f1": round(argmax_metrics["macro_f1"], 6),
        "ood_f1": round(argmax_metrics["ood_f1"], 6),
        "known_acc": round(argmax_metrics["known_acc"], 6),
        "overall_acc": round(argmax_metrics["overall_acc"], 6),
        "macro_f1_at_ood_threshold": round(thr_metrics["macro_f1"], 6),
        "ood_threshold": float(best_threshold),
        "ood_threshold_f1": float(best_f1),
        "num_val_batches": num_batches,
    }

    print(f"  ✓ Final Validation Results:")
    print(f"    Loss:        {metrics['final_val_loss']:.4f}")
    print(f"    OOD Acc:     {metrics['final_val_ood_acc']:.3f}")
    print(f"    Speaker Acc: {metrics['final_val_speaker_acc']:.3f}")
    print(f"    Macro-F1:    {metrics['macro_f1']:.4f}")

    # ── Persist the tuned OOD threshold into the checkpoint ──
    try:
        checkpoint["ood_threshold"] = float(best_threshold)
        checkpoint["macro_f1"] = argmax_metrics["macro_f1"]
        torch.save(checkpoint, best_model_path)
        print(f"  ✓ OOD threshold persisted to {best_model_path}")
    except Exception as e:
        print(f"  ⚠ Could not persist OOD threshold: {e}")

    # ── Final MLflow logging ──
    tracker = get_tracker()
    if tracker.is_active:
        tracker.log_metrics(metrics)
        tracker.log_best_checkpoint(best_model_path)
        tracker.log_summary(
            {"final_val_loss": metrics["final_val_loss"],
             "final_ood_acc": metrics["final_val_ood_acc"],
             "final_speaker_acc": metrics["final_val_speaker_acc"]},
            metrics,
        )
        # Log model
        try:
            tracker.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"  ⚠ Could not log model to MLflow: {e}")
        tracker.end_run()
        print(f"  ✅ MLflow run completed with all artifacts.")

    return metrics
