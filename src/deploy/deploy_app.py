"""
Streamlit UI for Speaker-Identification MLOps Center.

Provides a user-friendly interface to:
1. Select GPU type and pipeline stage
2. Choose hardware profile (local or Vast.ai)
3. Toggle feature extractor freeze
4. Launch training on Vast.ai or locally

Usage:
    streamlit run src/deploy/deploy_app.py
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import yaml

# Page config
st.set_page_config(
    page_title="Speaker-ID MLOps Center",
    page_icon="🎤",
    layout="centered",
)

# ─────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
DEPLOY_SCRIPT = PROJECT_ROOT / "src" / "deploy" / "deploy.py"
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipelines" / "run_pipeline.py"


# ─────────────────────────────────────────────────────────
#  Helper: Load Config
# ─────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────

st.title("🎤 Speaker-ID MLOps Center")
st.markdown("Open-Set Speaker Identification — IAAA 2026 Competition")

config = load_config()
mlops_cfg = config.get("mlops", {})
vast_cfg = mlops_cfg.get("vast", {})

# ── Sidebar: Environment Info ──
with st.sidebar:
    st.header("🔧 Environment")
    st.markdown(f"**Mode:** `{config['hardware']['mode']}`")
    st.markdown(f"**Python:** {sys.version.split()[0]}")
    st.markdown(f"**Config:** `{CONFIG_PATH.name}`")

    st.divider()
    st.markdown("**DagsHub MLflow:**")
    tracking_uri = mlops_cfg.get("tracking", {}).get("uri", "Not configured")
    st.markdown(f"[MLflow Dashboard]({tracking_uri})")

    st.divider()
    st.markdown("**Model:**")
    st.markdown(f"- Base: `{config['model']['base_model']}`")
    st.markdown(f"- Classes: 447 (1 unknown + 446 known)")
    st.markdown(f"- Epochs: {config['training']['epochs']}")

# ── Main Panel ──
tab1, tab2 = st.tabs(["🚀 Cloud Deploy (Vast.ai)", "💻 Local Run"])

# ═══════════════════════════════════════════
#  Tab 1: Vast.ai Cloud Deployment
# ═══════════════════════════════════════════
with tab1:
    st.header("☁️ Cloud Training on Vast.ai")
    st.markdown("Rent a GPU, run the pipeline remotely, and auto-destroy when done.")

    col1, col2 = st.columns(2)

    with col1:
        gpu_choice = st.selectbox(
            "🎮 GPU Type",
            options=["RTX_3090", "RTX_3060"],
            index=0,  # Default: RTX 3090
            help="RTX 3090 (24GB) for full fine-tune | RTX 3060 (12GB) for budget training",
        )

        freeze_fe = st.checkbox(
            "🔒 Freeze Feature Extractor",
            value=config["model"].get("freeze_feature_extractor", True),
            help="Freeze WavLM CNN layers to save VRAM. "
                 "Uncheck for full fine-tune (requires 24GB+ GPU).",
        )

    with col2:
        pipeline_choice = st.selectbox(
            "📋 Pipeline Stage",
            options=["all", "data", "train", "eval"],
            format_func=lambda x: {
                "all": "🚀 Full Pipeline (data → train → eval)",
                "data": "📊 Data Preparation Only",
                "train": "🏋️ Training Only",
                "eval": "📈 Evaluation Only",
            }[x],
        )

        st.markdown("### 💰 Estimated Cost")
        gpu_price = {"RTX_3090": 0.35, "RTX_3060": 0.15}
        hours = {"all": 6, "train": 5, "data": 0.5, "eval": 0.5}
        est_hours = hours[pipeline_choice]
        est_cost = gpu_price[gpu_choice] * est_hours
        st.metric(
            label="Estimated Cost",
            value=f"${est_cost:.2f}",
            delta=f"~{est_hours}h on {gpu_choice}",
        )

    # Launch button
    if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True):
        st.info(f"🚀 Connecting to Vast.ai to rent {gpu_choice}...")

        # Inject ALL user choices into environment for deploy.py
        os.environ["GPU_TARGET"] = gpu_choice
        os.environ["TARGET_PIPELINE"] = pipeline_choice
        os.environ["FREEZE_FEATURE_EXTRACTOR"] = str(freeze_fe).lower()

        with st.spinner("Renting GPU instance and starting pipeline..."):
            try:
                result = subprocess.run(
                    ["uv", "run", "python", str(DEPLOY_SCRIPT)],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(PROJECT_ROOT),
                )
                st.success("✅ Server launched successfully!")

                with st.expander("📋 Deployment Log"):
                    st.code(result.stdout)

                st.markdown("### 📊 Monitor Training")
                st.markdown(f"- [DagsHub MLflow]({tracking_uri})")
                st.warning(
                    "⚠️ The server will self-destruct automatically after "
                    "the pipeline completes. No manual cleanup needed."
                )

            except subprocess.TimeoutExpired:
                st.warning(
                    "⏳ The deployment request was sent. It may take a few "
                    "minutes for the server to be ready."
                )
            except subprocess.CalledProcessError as e:
                st.error("❌ Deployment failed!")
                with st.expander("Error Details"):
                    st.code(e.stdout)
                    if e.stderr:
                        st.code(e.stderr)

# ═══════════════════════════════════════════
#  Tab 2: Local Run
# ═══════════════════════════════════════════
with tab2:
    st.header("💻 Local Training")
    st.markdown("Run the pipeline on your local machine.")

    # Current hardware info
    import torch
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        st.success(f"✅ GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        st.warning("⚠️ No CUDA GPU detected. Training will run on CPU (very slow!)")

    local_stage = st.selectbox(
        "Pipeline Stage",
        options=["all", "data", "train", "eval"],
        format_func=lambda x: {
            "all": "🚀 Full Pipeline",
            "data": "📊 Data Preparation Only",
            "train": "🏋️ Training Only",
            "eval": "📈 Evaluation Only",
        }[x],
        key="local_stage",
    )

    local_freeze = st.checkbox(
        "🔒 Freeze Feature Extractor",
        value=True,
        help="Recommended for GPUs with < 8GB VRAM",
        key="local_freeze",
    )

    with_mlflow = st.checkbox(
        "📈 Enable MLflow Tracking",
        value=True,
        help="Log metrics to DagsHub (requires .env config)",
        key="local_mlflow",
    )

    if st.button("▶️ Run Locally", type="primary", use_container_width=True):
        cmd = [
            "uv", "run", "python", str(PIPELINE_SCRIPT),
            "--run", local_stage,
            "--config", str(CONFIG_PATH),
        ]
        if not with_mlflow:
            cmd.append("--no-mlflow")

        # Update config for freeze setting
        if local_freeze != config["model"].get("freeze_feature_extractor", True):
            st.info(f"Setting freeze_feature_extractor = {local_freeze}")
            with open(CONFIG_PATH, "r") as f:
                cfg_data = yaml.safe_load(f)
            cfg_data["model"]["freeze_feature_extractor"] = local_freeze
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg_data, f, default_flow_style=False)
            st.cache_resource.clear()

        with st.spinner("Running pipeline..."):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=7200,
                    cwd=str(PROJECT_ROOT),
                )

                if result.returncode == 0:
                    st.success("✅ Pipeline completed successfully!")
                else:
                    st.error("❌ Pipeline failed!")

                with st.expander("📋 Pipeline Log", expanded=True):
                    st.code(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
                    if result.stderr:
                        st.code(result.stderr[-2000:])

            except subprocess.TimeoutExpired:
                st.error("⏰ Pipeline timed out (2h limit).")

    # Quick actions
    st.divider()
    st.subheader("⚡ Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 View Last Results", use_container_width=True):
            checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
            if checkpoint_dir.exists():
                models = list(checkpoint_dir.glob("*.pt"))
                if models:
                    st.write("**Checkpoints found:**")
                    for m in models:
                        size_mb = m.stat().st_size / 1e6
                        st.write(f"- `{m.name}` ({size_mb:.1f} MB)")
                else:
                    st.info("No checkpoints yet. Train a model first.")
            else:
                st.info("Checkpoint directory not found.")

    with col2:
        if st.button("🧹 Clean Checkpoints", use_container_width=True):
            checkpoint_dir = PROJECT_ROOT / "checkpoints"
            if checkpoint_dir.exists():
                for f in checkpoint_dir.glob("*.pt"):
                    f.unlink()
                st.success("Checkpoints cleaned!")
            else:
                st.info("Nothing to clean.")
