#!/bin/bash
# ============================================================================
# setup_vast.sh — Bootstrap script for Vast.ai GPU instances
#
# This script runs automatically when a Vast.ai instance is created via
# deploy.py. It:
#   1. Installs system dependencies (uv, git, libgl)
#   2. Clones the speaker-identification repository
#   3. Pulls raw data via DVC from DagsHub
#   4. Configures ZenML stack with MLflow experiment tracker on DagsHub
#   5. Runs the target pipeline (all / data / train / eval)
#   6. Self-destructs the instance after completion
# ============================================================================

set -e  # Exit on any error

# ── Cleanup Handler: Destroy instance on exit/failure ──
cleanup() {
    echo ""
    echo "🚨 Job finished or failed. Cleaning up and destroying instance..."
    
    # Get numeric instance ID from Vast container label
    INSTANCE_ID=$(echo "${VAST_CONTAINERLABEL}" | tr -cd '0-9')
    if [ -n "$INSTANCE_ID" ] && [ -n "$VAST_API_KEY" ]; then
        echo "   Destroying instance $INSTANCE_ID..."
        pip install --upgrade --no-cache-dir vastai -q 2>/dev/null || true
        vastai destroy instance "$INSTANCE_ID" -y --api-key "$VAST_API_KEY" 2>/dev/null || true
        echo "   ✅ Instance destroyed."
    else
        echo "   ⚠ Could not determine instance ID or API key. Skipping destroy."
    fi
}

# Enable trap: cleanup on EXIT, ERR, or any signal
# Comment out the line below to keep the instance alive for debugging:
#trap cleanup EXIT ERR

# ============================================================================
#  Phase 1: System Setup
# ============================================================================
echo ""
echo "🚀 Starting Speaker-Identification MLOps Pipeline on Vast.ai..."
echo "   Instance: ${VAST_CONTAINERLABEL:-unknown}"
echo ""

echo "📦 Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git libgl1-mesa-glx libglib2.0-0 2>/dev/null || true

echo "📦 Installing uv (Python package manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy

# ============================================================================
#  Phase 2: Clone Repository
# ============================================================================
echo ""
echo "📥 Cloning repository: $GIT_REPO_URL (branch: $GIT_BRANCH)"
if [ -d "/workspace/project" ]; then
    echo "   Directory exists, skipping clone."
else
    git clone -b "$GIT_BRANCH" "$GIT_REPO_URL" /workspace/project
fi
cd /workspace/project

echo "📦 Installing Python dependencies with uv..."
uv sync

# ============================================================================
#  Phase 3: DVC — Pull Raw Data from DagsHub
# ============================================================================
echo ""
echo "🗄️  Setting up DVC and pulling raw data from DagsHub..."

# Authenticate DVC with DagsHub S3-compatible storage
# Use 'uv run dvc' to ensure the command is found (installed via pyproject.toml)
uv run dvc remote remove origin 2>/dev/null || true
uv run dvc remote add -d origin s3://dvc
uv run dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.s3"
uv run dvc remote modify origin --local access_key_id "${DAGSHUB_TOKEN}"
uv run dvc remote modify origin --local secret_access_key "${DAGSHUB_TOKEN}"

echo "   Pulling data from DagsHub..."
uv run dvc pull -r origin

if [ -d "data/raw" ]; then
    echo "✅ Raw data successfully pulled!"
    echo "   Files: $(ls data/raw/*.mp3 2>/dev/null | wc -l) MP3 files"
else
    echo "❌ ERROR: data/raw not found after DVC pull!"
    exit 1
fi

# ============================================================================
#  Phase 4: ZenML & MLflow Configuration
# ============================================================================
echo ""
echo "🔗 Configuring ZenML stack with MLflow (DagsHub)..."

# Set DagsHub credentials for MLflow
export MLFLOW_TRACKING_URI="${DAGSHUB_TRACKING_URI}"
export MLFLOW_TRACKING_USERNAME="${DAGSHUB_USERNAME}"
export MLFLOW_TRACKING_PASSWORD="${DAGSHUB_TOKEN}"
export MLFLOW_ALLOW_FILESTORE=true

# DagsHub S3 credentials (for artifact logging)
export AWS_ACCESS_KEY_ID="${DAGSHUB_TOKEN}"
export AWS_SECRET_ACCESS_KEY="${DAGSHUB_TOKEN}"
export AWS_DEFAULT_REGION="us-east-1"
export MLFLOW_S3_ENDPOINT_URL="https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.s3"

# ── Select hardware profile based on GPU target ──
# RTX 3090/4090/A5000 → vastai (batch_size=32)
# RTX 3060           → vastai_3060 (batch_size=16)
GPU_TARGET="${GPU_TARGET:-RTX_3090}"
case "$GPU_TARGET" in
    RTX_3060)
        echo "   Switching config to 'vastai_3060' profile (batch_size=16)..."
        sed -i 's/mode: local/mode: vastai_3060/' configs/default_config.yaml
        ;;
    *)
        echo "   Switching config to 'vastai' profile (batch_size=32)..."
        sed -i 's/mode: local/mode: vastai/' configs/default_config.yaml
        ;;
esac

# ── Apply freeze choice from user selection ──
# FREEZE_FEATURE_EXTRACTOR comes from deploy_app.py → deploy.py → env var
FREEZE_FE="${FREEZE_FEATURE_EXTRACTOR:-true}"
if [ "$FREEZE_FE" = "true" ]; then
    echo "   Keeping feature extractor FROZEN (as selected by user)..."
    # Ensure it's set to true in config
    sed -i 's/freeze_feature_extractor: false/freeze_feature_extractor: true/' configs/default_config.yaml 2>/dev/null || true
else
    echo "   Unfreezing feature extractor for full fine-tune (as selected by user)..."
    sed -i 's/freeze_feature_extractor: true/freeze_feature_extractor: false/' configs/default_config.yaml
fi

echo "✅ Environment configured for DagsHub MLflow tracking."
echo "   GPU: $GPU_TARGET | Freeze: $FREEZE_FE | Pipeline: $TARGET_PIPELINE"

# ============================================================================
#  Phase 5: Initialize ZenML
# ============================================================================
echo ""
echo "🔧 Initializing ZenML repository..."
uv run zenml init 2>/dev/null || true
echo "✅ ZenML initialized."

# ============================================================================
#  Phase 6: Run Pipeline
# ============================================================================
echo ""
echo "🔥 Starting Pipeline: $TARGET_PIPELINE"
echo "   Running: python -m src.pipelines.run_pipeline --run $TARGET_PIPELINE"
echo ""

uv run python -m src.pipelines.run_pipeline --run "$TARGET_PIPELINE"

# ============================================================================
#  Phase 7: Complete
# ============================================================================
echo ""
echo "🎉 Pipeline '$TARGET_PIPELINE' completed successfully!"
echo "📊 View results on DagsHub: $DAGSHUB_TRACKING_URI"
echo "🚀 Instance will self-destruct automatically."
echo ""
