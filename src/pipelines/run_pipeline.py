"""
ZenML Pipeline Orchestrator for Open-Set Speaker Identification.

Usage:
    # Full pipeline (local)
    python -m src.pipelines.run_pipeline --run all

    # Partial runs
    python -m src.pipelines.run_pipeline --run data     # prepare_data + build_model only
    python -m src.pipelines.run_pipeline --run train    # data + build + train
    python -m src.pipelines.run_pipeline --run eval     # data + build + train + evaluate

    # With custom config
    python -m src.pipelines.run_pipeline --config configs/default_config.yaml --run all

Environment variables (for DagsHub MLflow tracking):
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import mlflow
try:
    import dagshub
    _HAS_DAGSHUB = True
except ImportError:
    _HAS_DAGSHUB = False

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenml import pipeline
from zenml.client import Client
from zenml.config import DockerSettings

from src.pipelines.steps import convert_audio, prepare_data, build_model, train_model, evaluate_model
from src.data_pipeline import load_config


# ─────────────────────────────────────────────────────────
#  ZenML Stack Initialization
# ─────────────────────────────────────────────────────────

def ensure_mlflow_stack(config: dict) -> bool:
    """
    Ensure a ZenML stack with MLflow experiment tracker exists and is active.
    Creates the stack and tracker if they don't exist.

    Returns True if the stack is ready, False if MLflow tracking is unavailable.
    """
    mlops_cfg = config.get("mlops", {})
    tracking_cfg = mlops_cfg.get("tracking", {})

    # Resolve tracking URI (env var takes priority)
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or tracking_cfg.get("uri", "")
    )
    # Strip literal ${...} patterns that weren't resolved
    dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER") or os.getenv("DAGSHUB_USERNAME", "")
    dagshub_tok = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN", "")
    tracking_uri = tracking_uri.replace("${DAGSHUB_REPO_OWNER}", dagshub_owner)
    tracking_uri = tracking_uri.replace("${DAGSHUB_USER_TOKEN}", dagshub_tok)

    if not tracking_uri or "dagshub" not in tracking_uri.lower():
        print("  ⚠ No valid DagsHub MLflow tracking URI. Metrics will not be logged.")
        return False

    # Explicitly set MLflow tracking URI BEFORE dagshub.init
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Authenticate with DagsHub using dagshub.init()
    # Support both naming conventions (deploy.py uses DAGSHUB_USERNAME/DAGSHUB_TOKEN,
    # local .env may use DAGSHUB_REPO_OWNER/DAGSHUB_USER_TOKEN)
    dagshub_user = (
        os.getenv("DAGSHUB_REPO_OWNER")
        or os.getenv("DAGSHUB_USERNAME")
        or tracking_cfg.get("username", "").replace("${DAGSHUB_REPO_OWNER}", "")
    )
    dagshub_token = (
        os.getenv("DAGSHUB_USER_TOKEN")
        or os.getenv("DAGSHUB_TOKEN")
        or tracking_cfg.get("password", "").replace("${DAGSHUB_USER_TOKEN}", "")
    )

    if dagshub_user and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        # MLflow env vars are sufficient for DagsHub tracking.
        # We skip dagshub.init() on headless servers (no OAuth browser).
        # Only call dagshub.init() if running interactively with a display.
        if _HAS_DAGSHUB and os.environ.get("DISPLAY") or os.environ.get("SSH_TTY"):
            try:
                dagshub.init(repo_owner=dagshub_user, repo_name=os.getenv("DAGSHUB_REPO_NAME", "Speaker-identification"), mlflow=True)
                print(f"  ✓ DagsHub authenticated: {dagshub_user}")
            except Exception as e:
                print(f"  ⚠ DagsHub init skipped (headless server): {e}")
        else:
            print(f"  ✓ MLflow env vars set for DagsHub (token-based auth)")

    try:
        client = Client()

        # Register MLflow experiment tracker (API varies by ZenML version)
        tracker_name = "dagshub_tracker"
        try:
            client.get_experiment_tracker(tracker_name)
            print(f"  ✓ Using existing experiment tracker: {tracker_name}")
        except (KeyError, AttributeError):
            print(f"  ➕ Creating MLflow experiment tracker: {tracker_name}")
            try:
                client.create_experiment_tracker(
                    name=tracker_name,
                    flavor="mlflow",
                    tracking_uri=tracking_uri,
                )
            except AttributeError:
                print(f"  ⚠ ZenML version doesn't support experiment trackers. "
                      f"MLflow will work via env vars directly.")

        # Register stack
        stack_name = "speaker_stack"
        try:
            client.get_stack(stack_name)
            print(f"  ✓ Using existing stack: {stack_name}")
        except (KeyError, AttributeError):
            print(f"  ➕ Creating stack: {stack_name}")
            try:
                client.create_stack(
                    name=stack_name,
                    components={
                        "orchestrator": "default",
                        "artifact_store": "default",
                        "experiment_tracker": tracker_name,
                    },
                )
            except Exception:
                print(f"  ⚠ Could not create ZenML stack. MLflow tracking via env vars.")

        try:
            client.activate_stack(stack_name)
            print(f"  ✓ Active stack: {stack_name}")
        except Exception:
            print(f"  ⚠ Could not activate stack. Using default.")

        # AWS/DagsHub S3 credentials for artifact logging
        if dagshub_token:
            os.environ["AWS_ACCESS_KEY_ID"] = dagshub_token
            os.environ["AWS_SECRET_ACCESS_KEY"] = dagshub_token
            os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
                f"https://dagshub.com/{dagshub_user}/"
                f"{os.getenv('DAGSHUB_REPO_NAME', 'Speaker-identification')}.s3"
            )

        # Create experiment if it doesn't exist
        experiment_name = mlops_cfg.get("experiment_name", "speaker-identification")
        try:
            mlflow.set_experiment(experiment_name)
            print(f"  ✓ MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"  ⚠ Could not set MLflow experiment: {e}")

        return True

    except Exception as e:
        print(f"  ⚠ Could not configure ZenML MLflow stack: {e}")
        print("  Running without experiment tracking.")
        return False


# ─────────────────────────────────────────────────────────
#  Pipeline Definition
# ─────────────────────────────────────────────────────────

@pipeline(
    name="speaker_id_pipeline",
    enable_cache=False,  # Always re-run to get fresh results
)
def speaker_id_pipeline(config_path: str = "configs/default_config.yaml"):
    """
    Full training pipeline for open-set speaker identification.

    Steps:
        0. convert_audio  → config (with WAV paths)
        1. prepare_data    → config, class_map, train_df, val_df
        2. build_model     → model_checkpoint_path
        3. train_model     → best_model_path, summary
        4. evaluate_model  → metrics
    """
    # Step 0: Convert MP3 → WAV (skips if already converted)
    config = convert_audio(config_path=config_path)
    config, class_map, train_df, val_df = prepare_data(config_path=config_path)
    model_path = build_model(config=config, class_map=class_map)
    best_path, summary = train_model(
        config=config,
        class_map=class_map,
        train_df=train_df,
        val_df=val_df,
        model_checkpoint_path=model_path,
    )
    metrics = evaluate_model(
        config=config,
        class_map=class_map,
        val_df=val_df,
        best_model_path=best_path,
    )
    return metrics


# ─────────────────────────────────────────────────────────
#  Partial Pipeline Runners
# ─────────────────────────────────────────────────────────

def run_data_stage(config_path: str):
    """Run only data preparation and model building."""
    print("  ▶ Running data preparation + model building stage")
    config, class_map, train_df, val_df = prepare_data(config_path=config_path)
    model_path = build_model(config=config, class_map=class_map)
    print(f"\n  ✅ Data stage complete. Model checkpoint: {model_path}")
    return model_path


def run_train_stage(config_path: str):
    """Run data prep + model building + training."""
    print("  ▶ Running training stage (data → build → train)")
    config, class_map, train_df, val_df = prepare_data(config_path=config_path)
    model_path = build_model(config=config, class_map=class_map)
    best_path, summary = train_model(
        config=config,
        class_map=class_map,
        train_df=train_df,
        val_df=val_df,
        model_checkpoint_path=model_path,
    )
    print(f"\n  ✅ Training stage complete. Best model: {best_path}")
    print(f"     Summary: {summary}")
    return best_path, summary


def run_eval_stage(config_path: str):
    """Run the full pipeline including evaluation."""
    print("  ▶ Running full pipeline (data → build → train → evaluate)")
    config, class_map, train_df, val_df = prepare_data(config_path=config_path)
    model_path = build_model(config=config, class_map=class_map)
    best_path, summary = train_model(
        config=config,
        class_map=class_map,
        train_df=train_df,
        val_df=val_df,
        model_checkpoint_path=model_path,
    )
    metrics = evaluate_model(
        config=config,
        class_map=class_map,
        val_df=val_df,
        best_model_path=best_path,
    )
    print(f"\n  ✅ Full pipeline complete!")
    print(f"     Training summary: {summary}")
    print(f"     Evaluation metrics: {metrics}")
    return metrics


# ─────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Speaker-Identification MLOps Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML config file (default: configs/default_config.yaml)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "data", "train", "eval"],
        help="Which pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking",
    )

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    print("=" * 55)
    print("  Speaker-Identification MLOps Pipeline")
    print("=" * 55)
    print(f"  Config: {config_path}")
    print(f"  Stage:  {args.run}")
    print()

    # Configure MLflow stack (unless disabled)
    if not args.no_mlflow:
        config = load_config(config_path)
        ensure_mlflow_stack(config)
    else:
        print("  MLflow tracking disabled via --no-mlflow")

    print()

    # Run the requested stage
    if args.run == "data":
        run_data_stage(config_path)
    elif args.run == "train":
        run_train_stage(config_path)
    elif args.run == "eval":
        run_eval_stage(config_path)
    else:  # "all"
        # Use the full ZenML pipeline
        print("  🚀 Executing full ZenML pipeline...\n")
        speaker_id_pipeline(config_path=config_path)

    print("\n  ✅ Pipeline finished successfully!")


if __name__ == "__main__":
    main()
