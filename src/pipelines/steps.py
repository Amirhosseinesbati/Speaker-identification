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

import mlflow
import numpy as np
import pandas as pd
import torch
from zenml import step
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (
    load_config,
    get_active_profile,
    create_class_mapping,
    stratified_split,
    SpeakerDataset,
)
from src.model import TwoHeadedWavLM
from src.train import (
    train_epoch,
    validate_epoch,
    TwoPartLoss,
    compute_ood_accuracy,
    compute_speaker_accuracy,
    setup_device,
)


# ─────────────────────────────────────────────────────────
#  Helper: MLflow active check
# ─────────────────────────────────────────────────────────
# ZenML automatically activates an MLflow run before each step
# when the stack has an MLflow experiment tracker configured.
# We just need to check if it's active before logging.

def _mlflow_active() -> bool:
    """Check if MLflow has an active run (set up by ZenML pipeline)."""
    return mlflow.active_run() is not None


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

    return config


# ─────────────────────────────────────────────────────────
#  Step 1: Prepare Data
# ─────────────────────────────────────────────────────────

@step
def prepare_data(
    config_path: str = "configs/default_config.yaml",
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Load config, prepare labels, and perform stratified split.

    Returns:
        config, class_map, train_df, val_df
    """
    print("=" * 55)
    print("  [ZenML Step 1/4] Preparing Data")
    print("=" * 55)

    config = load_config(config_path)
    data_cfg = config["data"]

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

    # Load raw labels
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["speaker_id", "audio_file"]).reset_index(drop=True)

    # Create class mapping
    class_map = create_class_mapping(df)
    df["label"] = df["speaker_id"].map(class_map)

    # Save cleaned labels
    os.makedirs(os.path.dirname(data_cfg["processed_labels"]), exist_ok=True)
    df.to_csv(data_cfg["processed_labels"], index=False)

    # Stratified split
    train_df, val_df = stratified_split(df, val_per_known=1, unknown_val_ratio=0.2)

    print(f"  ✓ Train: {len(train_df)} samples | Val: {len(val_df)} samples")
    print(f"  ✓ Classes: {len(class_map)} (0=unknown, 1..{len(class_map)-1}=known)")

    # Log data params to MLflow (ZenML auto-activates MLflow run)
    if _mlflow_active():
        mlflow.log_params({
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

    num_known = len(class_map) - 1
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

    if _mlflow_active():
        mlflow.log_params({
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

    train_dataset = SpeakerDataset(
        df=train_df,
        audio_dir=data_cfg["audio_dir"],
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=True,
    )
    val_dataset = SpeakerDataset(
        df=val_df,
        audio_dir=data_cfg["audio_dir"],
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=False,
    )

    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels, minlength=len(class_map))
    weights = 1.0 / (class_counts + 1e-8)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

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
    num_known = len(class_map) - 1
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # ── Optimizer, Loss, Scheduler, Scaler ──
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"],
    )
    criterion = TwoPartLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=hw_profile["mixed_precision"])

    # ── Training Loop with MLflow autologging ──
    log_cfg = config.get("logging", {})
    checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    # MLflow is auto-activated by ZenML — log directly
    mlflow_on = _mlflow_active()

    for epoch in range(1, train_cfg["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, train_cfg["max_grad_norm"],
        )
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics per epoch (directly to the active MLflow run)
        if mlflow_on:
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_ood_acc": train_metrics["ood_acc"],
                "train_speaker_acc": train_metrics["speaker_acc"],
                "val_loss": val_metrics["loss"],
                "val_ood_acc": val_metrics["ood_acc"],
                "val_speaker_acc": val_metrics["speaker_acc"],
                "learning_rate": current_lr,
            }, step=epoch)

        # Print progress
        print(f"\n  Epoch {epoch:2d}/{train_cfg['epochs']} — "
              f"Loss: {train_metrics['loss']:.4f} / {val_metrics['loss']:.4f}  |  "
              f"OOD: {train_metrics['ood_acc']:.3f} / {val_metrics['ood_acc']:.3f}  |  "
              f"Spk: {train_metrics['speaker_acc']:.3f} / {val_metrics['speaker_acc']:.3f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
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
            }, best_path)

        # Save latest
        latest_path = checkpoint_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "class_map": class_map,
        }, latest_path)

        # Record history
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": current_lr,
        })

    # ── Final logging ──
    final_best_path = str(checkpoint_dir / "best_model.pt")
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_ood_acc": round(history[best_epoch - 1]["val_ood_acc"], 4),
        "best_val_speaker_acc": round(history[best_epoch - 1]["val_speaker_acc"], 4),
        "total_epochs": train_cfg["epochs"],
    }

    if mlflow_on:
        mlflow.log_params({
            "epochs": train_cfg["epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "batch_size": hw_profile["batch_size"],
        })
        mlflow.log_metrics(summary)
        # Log the best model artifact
        mlflow.log_artifact(final_best_path, artifact_path="models")

    print(f"\n  ✓ Training complete! Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
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
    num_known = len(class_map) - 1
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Evaluate
    criterion = TwoPartLoss(ignore_index=-100)
    total_loss = 0.0
    total_ood_acc = 0.0
    total_speaker_acc = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            ood_logits, speaker_logits = model(waveforms)
            loss, loss_dict = criterion(ood_logits, speaker_logits, labels)

            total_loss += loss_dict["loss_total"]
            total_ood_acc += compute_ood_accuracy(ood_logits, labels)
            speaker_acc, _ = compute_speaker_accuracy(speaker_logits, labels)
            total_speaker_acc += speaker_acc

    metrics = {
        "final_val_loss": round(total_loss / num_batches, 6),
        "final_val_ood_acc": round(total_ood_acc / num_batches, 4),
        "final_val_speaker_acc": round(total_speaker_acc / num_batches, 4),
        "num_val_batches": num_batches,
    }

    print(f"  ✓ Final Validation Results:")
    print(f"    Loss:       {metrics['final_val_loss']:.4f}")
    print(f"    OOD Acc:    {metrics['final_val_ood_acc']:.3f}")
    print(f"    Speaker Acc: {metrics['final_val_speaker_acc']:.3f}")

    # Log to MLflow (directly to the active run)
    if _mlflow_active():
        mlflow.log_metrics(metrics)
        # Register the model in MLflow model registry
        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name="speaker-identification",
            )
        except Exception as e:
            print(f"  ⚠ Could not log model to MLflow registry: {e}")

    return metrics
