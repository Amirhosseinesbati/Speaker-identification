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

# Ensure /tmp exists and is writable (some Vast.ai images don't have it)
if mkdir -p /tmp 2>/dev/null && touch /tmp/.dvc_test_write 2>/dev/null; then
    rm -f /tmp/.dvc_test_write
    chmod 1777 /tmp
    echo "   ✅ /tmp directory ready and writable"
else
    echo "   ⚠ /tmp not writable, using fallback TMPDIR in workspace..."
    export TMPDIR="/workspace/project/.tmp"
    mkdir -p "$TMPDIR"
    chmod 1777 "$TMPDIR"
    echo "   ✅ Fallback TMPDIR set to $TMPDIR"
fi

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
#  Phase 2.5: Ensure CUDA-enabled PyTorch in the venv (adaptive)
# ============================================================================
# uv.lock pins torch 2.13.0+cu126 (CUDA 12.6, needs host driver >= 560).
# Rented instances have different driver versions, so if the lock torch cannot
# initialise CUDA we pick a wheel whose CUDA level is compatible with the HOST
# DRIVER (detected via nvidia-smi): cu124 (>= 550) → cu121 (>= 530) → cu118
# (>= 515). Each candidate is verified; the first that works is kept. You can
# force a specific level with TORCH_CUDA_LEVEL, e.g. "cu118" or "cu124:2.6.0".
echo ""
echo "🖥️  Verifying CUDA-enabled PyTorch in the venv..."
cuda_ok() { uv run --no-sync python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

install_cuda_torch() {
    # $1 = level (e.g. cu121), $2 = torch/torchaudio version (e.g. 2.2.0)
    echo "   Trying ${1} (torch ${2})..."
    if uv pip install --python .venv/bin/python \
            "torch==${2}+${1}" "torchaudio==${2}+${1}" \
            --index-url "https://download.pytorch.org/whl/${1}" >/tmp/torch_install.log 2>&1; then
        if cuda_ok; then
            echo "   ✅ CUDA-enabled PyTorch: $(uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" 2>/dev/null)"
            return 0
        fi
    fi
    echo "   (${1} failed; log tail: $(tail -2 /tmp/torch_install.log 2>/dev/null | tr '\n' ' '))"
    return 1
}

if cuda_ok; then
    echo "   ✅ CUDA available: $(uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" 2>/dev/null)"
else
    echo "   ⚠ torch.cuda.is_available()=False (torch $(uv run --no-sync python -c "import torch; print(torch.__version__)" 2>/dev/null))"

    # ── Detect host driver version ──
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    DVER=$(echo "${DRIVER:-0}" | cut -d. -f1)
    echo "   Host NVIDIA driver: ${DRIVER:-not detected}"
    if [ -z "${DRIVER:-}" ] || [ "${DVER:-0}" -eq 0 ] 2>/dev/null; then
        echo "   ❌ nvidia-smi failed — GPU is not visible in this container."
        echo "   → Check the Vast.ai instance (GPU must be exposed). Aborting."
        exit 1
    fi

    # ── Pick candidate wheels by driver version (best first) ──
    if [ -n "${TORCH_CUDA_LEVEL:-}" ]; then
        # Manual override, e.g. TORCH_CUDA_LEVEL="cu118" or "cu124:2.6.0"
        LVL="${TORCH_CUDA_LEVEL%%:*}"; TV="${TORCH_CUDA_LEVEL##*:}"
        [ "$TV" = "$LVL" ] && TV="2.2.0"
        if install_cuda_torch "$LVL" "$TV"; then :; else echo "   ❌ Override ${TORCH_CUDA_LEVEL} failed."; exit 1; fi
    elif [ "${DVER:-0}" -ge 550 ] 2>/dev/null; then
        for pair in "cu124 2.6.0" "cu121 2.2.0" "cu118 2.2.0"; do
            read -r LVL TV <<< "$pair"
            if install_cuda_torch "$LVL" "$TV"; then break; fi
        done
    elif [ "${DVER:-0}" -ge 530 ] 2>/dev/null; then
        for pair in "cu121 2.2.0" "cu118 2.2.0"; do
            read -r LVL TV <<< "$pair"
            if install_cuda_torch "$LVL" "$TV"; then break; fi
        done
    elif [ "${DVER:-0}" -ge 515 ] 2>/dev/null; then
        if ! install_cuda_torch "cu118" "2.2.0"; then
            echo "   ❌ Driver ${DRIVER} is too old for CUDA 11.8 wheels."
            exit 1
        fi
    else
        echo "   ❌ Driver ${DRIVER} is too old for any supported CUDA wheel (need >= 515)."
        echo "   → Re-rent an instance with a newer driver."
        exit 1
    fi

    # ── Final verification ──
    if ! cuda_ok; then
        echo "   ❌ CUDA still unavailable after all candidates. Diagnostics:"
        nvidia-smi 2>/dev/null || echo "   (nvidia-smi not found)"
        uv run --no-sync python -c "import torch; print('   torch', torch.__version__, '| cuda_build', torch.version.cuda)"
        echo "   → Re-rent an instance with a newer driver, or set TORCH_CUDA_LEVEL."
        exit 1
    fi
fi

# ============================================================================
#  Phase 2.6: Pin the venv's bundled libnccl (fixes ncclCommResume on old images)
# ============================================================================
# The pytorch/pytorch:2.2.0 base image ships a SYSTEM libnccl < 2.19.3 at
# /usr/local/nccl2/lib (listed first in the container's LD_LIBRARY_PATH). The
# pip torch wheel bundles a NEWER libnccl (with ncclCommResume) inside
# torch/lib, but LD_LIBRARY_PATH is searched before the wheel's RUNPATH, so the
# old system NCCL shadows the bundled one → `import torch` dies with
# "libtorch_cuda.so: undefined symbol: ncclCommResume". Fix: put torch/lib
# first in LD_LIBRARY_PATH and LD_PRELOAD the bundled libnccl so the correct
# (self-consistent) NCCL is always loaded.
TORCH_LIB_DIR="$(uv run --no-sync python - <<'PYEOF' 2>/dev/null
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
if [ -n "$TORCH_LIB_DIR" ] && [ -f "$TORCH_LIB_DIR/libnccl.so.2" ]; then
    echo "   🔧 Pinning bundled NCCL: $TORCH_LIB_DIR/libnccl.so.2"
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_PRELOAD="$TORCH_LIB_DIR/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "   ⚠ torch/lib/libnccl.so.2 not found — skipping NCCL pin (${TORCH_LIB_DIR:-torch not importable})"
fi

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
# RTX 3090/4090/A5000 → vastai      (batch_size=32)
# RTX 3060           → vastai_3060  (batch_size=16)
# RTX A4000 (16GB)   → vastai_a4000 (batch_size=24)
GPU_TARGET="${GPU_TARGET:-RTX_3090}"
case "$GPU_TARGET" in
    RTX_3060)
        echo "   Switching config to 'vastai_3060' profile (batch_size=16)..."
        sed -i 's/mode: local/mode: vastai_3060/' configs/default_config.yaml
        ;;
    RTX_A4000)
        echo "   Switching config to 'vastai_a4000' profile (batch_size=24)..."
        sed -i 's/mode: local/mode: vastai_a4000/' configs/default_config.yaml
        ;;
    *)
        echo "   Switching config to 'vastai' profile (batch_size=32)..."
        sed -i 's/mode: local/mode: vastai/' configs/default_config.yaml
        ;;
esac

# ── Apply encoder fine-tune choice from user selection ──
# FREEZE_ENCODER / UNFREEZE_LAST_N_BLOCKS come from deploy_app.py → deploy.py → env.
# We edit the YAML with Python (encoder-aware) instead of sed, so the right key
# is used per encoder: ECAPA → freeze_encoder/unfreeze_last_n_blocks,
# WavLM/HuBERT → freeze_feature_extractor.
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_BLOCKS="${UNFREEZE_LAST_N_BLOCKS:-0}"
echo "   Applying encoder fine-tune choice (freeze=${FREEZE_ENCODER}, blocks=${UNFREEZE_BLOCKS})..."
uv run --no-sync python - <<'PYEOF' || true
import os, yaml
p = "configs/default_config.yaml"
c = yaml.safe_load(open(p, encoding="utf-8"))
mc = c.setdefault("model", {})
enc_type = mc.get("encoder_type", "ecapa")
enc_cfg = mc.setdefault("encoder_config", {}).setdefault(enc_type, {})
freeze = os.getenv("FREEZE_ENCODER", "true").lower() == "true"
blocks = int(os.getenv("UNFREEZE_LAST_N_BLOCKS", "0") or 0)
if enc_type == "ecapa":
    enc_cfg["freeze_encoder"] = freeze
    enc_cfg["unfreeze_last_n_blocks"] = blocks if (not freeze and blocks > 0) else 0
else:
    enc_cfg["freeze_feature_extractor"] = freeze
yaml.dump(c, open(p, "w", encoding="utf-8"), default_flow_style=False,
          sort_keys=False, allow_unicode=True)
print(f"   Config updated: encoder={enc_type} freeze={freeze} unfreeze_blocks={blocks}")
PYEOF

echo "✅ Environment configured for DagsHub MLflow tracking."
echo "   GPU: $GPU_TARGET | Freeze: $FREEZE_ENCODER | Blocks: $UNFREEZE_BLOCKS | Pipeline: $TARGET_PIPELINE"

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

uv run --no-sync python -m src.pipelines.run_pipeline --run "$TARGET_PIPELINE"

# ============================================================================
#  Phase 7: Complete
# ============================================================================
echo ""
echo "🎉 Pipeline '$TARGET_PIPELINE' completed successfully!"
echo "📊 View results on DagsHub: $DAGSHUB_TRACKING_URI"
echo "🚀 Instance will self-destruct automatically."
echo ""
