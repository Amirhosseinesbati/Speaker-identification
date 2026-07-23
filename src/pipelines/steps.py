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

    # Load raw labels
    df = pd.read_csv(data_cfg["labels_path"])
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
    Instantiate the TwoHeadedWavLM model and save its initial state.

    Returns:
        checkpoint_path: path to the saved initial model checkpoint (state_dict only).
    """
    print("=" * 55)
    print("  [ZenML Step 2/4] Building Model")
    print("=" * 55)

    num_known = len(class_map) - 1
    model = TwoHeadedWavLM(config, num_known_speakers=num_known)

    # Note: freeze/unfreeze is handled inside TwoHeadedWavLM.__init__
    # based on config["model"]["freeze_feature_extractor"]
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

    # Log model architecture params
    if _mlflow_active():
        mlflow.log_params({
            "base_model": config["model"]["base_model"],
            "freeze_feature_extractor": config["model"].get("freeze_feature_extractor", True),
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
    model = TwoHeadedWavLM(config, num_known_speakers=num_known)
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
    model = TwoHeadedWavLM(config, num_known_speakers=num_known)
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
