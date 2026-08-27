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

# Campaign lifecycle is controlled by the supervisor. Never self-destroy on a
# training/setup error because WAITING_FOR_LEADERBOARD must preserve the worker.
# Manual or budget-guard cleanup is performed through the Vast control plane.
# trap cleanup EXIT ERR

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
#  Phase 2.4: Pre-flight — verify ALL encoder frameworks import
# ============================================================================
# A stale uv.lock can make `uv sync` silently miss undeclared runtime deps
# (modelscope's reduced wheel metadata misses e.g. 'addict'). Fail here with a
# clear message instead of deep inside a ZenML step 5 minutes later. All five
# frameworks are checked so ANY model (not just the active encoder) can start
# training on this instance. eres2net needs only the vendored src.sv_arch
# (torch + torchaudio) — no extra framework.
echo ""
echo "🔎 Pre-flight: importing all encoder frameworks..."
uv run --no-sync python - <<'PYEOF' || { echo "   ❌ Dependency pre-flight failed — see errors above."; exit 1; }
import importlib

# speechbrain registers broken LazyModules (e.g. integrations.k2_fsa → needs the
# optional `k2` package) that break lazy_loader's inspect.stack inside OTHER
# framework imports — so it must be imported LAST.
checks = [
    ("campp", "modelscope.models"),
    ("titanet", "nemo.collections.asr.models"),
    ("wavlm", "transformers"),
    ("ecapa", "speechbrain"),
]
failed = []
for enc, mod in checks:
    try:
        importlib.import_module(mod)
        print(f"   ✅ {enc} → {mod}")
    except Exception as e:
        failed.append(f"{enc} → {mod}: {e}")
        print(f"   ❌ {enc} → {mod}: {e}")
if failed:
    raise SystemExit("dependency pre-flight failed: " + "; ".join(failed))
PYEOF
echo "   ✅ All encoder frameworks importable (pre-flight passed)."

# ============================================================================
#  Phase 2.5: Verify CUDA is available (fail loudly — no auto-install)
# ============================================================================
# uv.lock pins torch 2.13.0 from PyPI (a CUDA 13.x build on Linux that needs a
# recent host driver; CPU-only on Windows). If CUDA cannot initialise, training
# would silently run on CPU. We stop with a clear message instead; if you need
# to, install the wheel that matches the host driver yourself, e.g.:
#   uv pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 \
#       --index-url https://download.pytorch.org/whl/cu121
echo ""
echo "🖥️  Verifying CUDA-enabled PyTorch in the venv..."
if ! uv run --no-sync python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "   ❌ torch.cuda.is_available()=False (torch $(uv run --no-sync python -c "import torch; print(torch.__version__)" 2>/dev/null))."
    echo "   → The installed torch wheel needs a newer host driver than this instance has."
    echo "   → Re-rent with a newer driver, or install a matching wheel yourself (see comment above)."
    exit 1
fi
echo "   ✅ CUDA available: $(uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" 2>/dev/null)"

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

# ── Apply encoder selection + fine-tune choice from user ──
# ENCODER_TYPE / ALLOW_HUB_DOWNLOAD / FREEZE_ENCODER / UNFREEZE_LAST_N_BLOCKS /
# LOCAL_PATH_<ENC> come from deploy_app.py → deploy.py → env. We edit the YAML
# with Python so the right key is used per encoder:
#   ecapa                → freeze_encoder / unfreeze_last_n_blocks
#   wavlm                → freeze_feature_extractor
#   campp/eres2net/titanet → freeze_encoder (only — they ignore the other keys)
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_BLOCKS="${UNFREEZE_LAST_N_BLOCKS:-0}"
echo "   Applying encoder selection (encoder=${ENCODER_TYPE:-<from config>}, freeze=${FREEZE_ENCODER}, blocks=${UNFREEZE_BLOCKS})..."
uv run --no-sync python - <<'PYEOF'
import os, yaml
p = "configs/default_config.yaml"
c = yaml.safe_load(open(p, encoding="utf-8"))
mc = c.setdefault("model", {})

# Encoder selection from deploy.py env (falls back to the committed config).
enc_type = os.getenv("ENCODER_TYPE", mc.get("encoder_type", "ecapa")).lower().strip()
mc["encoder_type"] = enc_type
enc_cfg = mc.setdefault("encoder_config", {}).setdefault(enc_type, {})

# Offline gate — a fresh training machine may pull weights from the hub once.
hub = os.getenv("ALLOW_HUB_DOWNLOAD", "false").lower() == "true"
mc["allow_hub_download"] = hub

# Per-encoder local_path override (defaults to the committed config value).
lp = os.getenv("LOCAL_PATH_{}".format(enc_type.upper()))
if lp:
    enc_cfg["local_path"] = lp

freeze = os.getenv("FREEZE_ENCODER", "true").lower() == "true"
blocks = int(os.getenv("UNFREEZE_LAST_N_BLOCKS", "0") or 0)

if enc_type == "ecapa":
    enc_cfg["freeze_encoder"] = freeze
    enc_cfg["unfreeze_last_n_blocks"] = blocks if (not freeze and blocks > 0) else 0
    enc_cfg.pop("freeze_feature_extractor", None)
elif enc_type == "wavlm":
    enc_cfg["freeze_feature_extractor"] = freeze
    enc_cfg.pop("freeze_encoder", None)
    enc_cfg.pop("unfreeze_last_n_blocks", None)
else:  # campp / eres2net / titanet
    enc_cfg["freeze_encoder"] = freeze
    enc_cfg.pop("freeze_feature_extractor", None)
    enc_cfg.pop("unfreeze_last_n_blocks", None)

yaml.dump(c, open(p, "w", encoding="utf-8"), default_flow_style=False,
          sort_keys=False, allow_unicode=True)
print(f"   Config updated: encoder={enc_type} freeze={freeze} "
      f"unfreeze_blocks={blocks} allow_hub_download={hub}")
PYEOF

# ============================================================================
#  Phase 4.5: Pull encoder weights (idempotent)
# ============================================================================
# weights/ is gitignored and DVC only pulls data/raw, so a fresh instance has
# no encoder weights and training would crash with FileNotFoundError. Download
# them now (idempotent — existing markers are skipped) so training runs
# offline-first regardless of allow_hub_download.
echo ""
echo "⬇️  Ensuring encoder weights are present (idempotent)..."
uv run --no-sync python scripts/download_all_weights.py
echo "   ✅ Encoder weights ready."

# ============================================================================
#  Phase 4.6: Sync the cluster map (closed-set 1000-class experiment)
# ============================================================================
# data/processed/* is gitignored, so a fresh clone has no cluster maps. When
# the config requests cluster mode (model.num_unknown_clusters > 0), the
# committed submission/<map-basename> copy is the durable one (maps are
# k-locked: unknown_clusters_k1000.json etc.) — copy it into data/processed/
# where the data pipeline (and its k-vs-map validation) expects it. Note: this
# covers the base-config path; queue runs (EXPERIMENT_PROFILES) rely on the
# data_pipeline fallback to submission/<basename> per profile instead.
echo ""
echo "🧬  Syncing cluster map for closed-set 1000-class mode (if configured)..."
uv run --no-sync python - <<'PYEOF'
import json, os, shutil, yaml

config_path = "configs/default_config.yaml"
if not os.path.exists(config_path):
    raise SystemExit("configs/default_config.yaml not found — skipping cluster sync")
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
k = int((cfg.get("model", {}) or {}).get("num_unknown_clusters", 0) or 0)
if k <= 0:
    print("   Cluster mode OFF (num_unknown_clusters=0) — nothing to sync.")
    raise SystemExit(0)

map_rel = str((cfg.get("model", {}) or {}).get(
    "unknown_cluster_path", "data/processed/unknown_clusters.json"))
dst = os.path.join("data", "processed", os.path.basename(map_rel))
src = os.path.join("submission", os.path.basename(map_rel))
os.makedirs(os.path.dirname(dst), exist_ok=True)

if os.path.exists(dst):
    with open(dst, encoding="utf-8") as f:
        existing = json.load(f)
    if len({int(v) for v in existing.values()}) == k:
        print(f"   ✓ data/processed cluster map already present at k={k} — skipping.")
        raise SystemExit(0)

if not os.path.exists(src):
    raise SystemExit(
        f"   ❌ Cluster mode requests k={k} but neither {dst} nor the committed "
        f"{src} exists. Rebuild clusters (UI: Config → Cluster Mode → Rebuild, "
        f"or `python -m src.unknown_clustering build --k {k} "
        f"--out {dst}`) and commit {src}."
    )
shutil.copy2(src, dst)
with open(dst, encoding="utf-8") as f:
    n = len({int(v) for v in json.load(f).values()})
print(f"   ✓ Cluster map synced: {src} → {dst} "
      f"(k={n}, requested {k})")
PYEOF

echo "✅ Environment configured for DagsHub MLflow tracking."
echo "   GPU: $GPU_TARGET | Encoder: ${ENCODER_TYPE:-<from config>} | Freeze: $FREEZE_ENCODER | Blocks: $UNFREEZE_BLOCKS | Pipeline: $TARGET_PIPELINE"

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
if [ -n "$EXPERIMENT_PROFILES" ]; then
    # One instance, many runs (Audit §17.2): run the named profiles as a
    # sequential queue. The profiles must be committed + pushed so the clone
    # above sees configs/experiments/*.yaml.
    echo "🔥 Running experiment queue: $EXPERIMENT_PROFILES"
    echo "   Running: python -m src.experiment_queue --profiles $EXPERIMENT_PROFILES --run $TARGET_PIPELINE"
    echo ""
    uv run --no-sync python -m src.experiment_queue --profiles $EXPERIMENT_PROFILES --run "$TARGET_PIPELINE"
elif [ -n "$HPO_STUDY" ]; then
    # Optuna HPO on the instance (Audit §17.4) — tunes the committed recipe and
    # logs every trial to DagsHub MLflow (creds already exported in Phase 4).
    echo "🔥 Running Optuna HPO study: $HPO_STUDY (trials=${HPO_TRIALS:-30}, epochs=${HPO_EPOCHS:-30})"
    echo ""
    if [ -n "$HPO_BASE_PROFILE" ]; then
        uv run --no-sync python -m src.hpo --study "$HPO_STUDY" \
            --trials "${HPO_TRIALS:-30}" --epochs "${HPO_EPOCHS:-30}" \
            --base-profile "$HPO_BASE_PROFILE"
    else
        uv run --no-sync python -m src.hpo --study "$HPO_STUDY" \
            --trials "${HPO_TRIALS:-30}" --epochs "${HPO_EPOCHS:-30}"
    fi
else
    echo "🔥 Starting Pipeline: $TARGET_PIPELINE"
    echo "   Running: python -m src.pipelines.run_pipeline --run $TARGET_PIPELINE"
    echo ""
    uv run --no-sync python -m src.pipelines.run_pipeline --run "$TARGET_PIPELINE"
fi

# ============================================================================
#  Phase 7: Complete
# ============================================================================
echo ""
echo "🎉 Pipeline '$TARGET_PIPELINE' completed successfully!"
echo "📊 View results on DagsHub: $DAGSHUB_TRACKING_URI"
echo "🚀 Instance will self-destruct automatically."
echo ""
