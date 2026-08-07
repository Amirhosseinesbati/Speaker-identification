"""
Streamlit UI — Speaker-ID MLOps Center (Full Control Panel).

All configurable parameters exposed in the UI.
Usage: uv run streamlit run src/deploy/deploy_app.py
"""

import os, subprocess, sys
from pathlib import Path
import streamlit as st
import yaml

st.set_page_config(page_title="Speaker-ID MLOps", page_icon="🎤", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
DEPLOY_SCRIPT = PROJECT_ROOT / "src" / "deploy" / "deploy.py"
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipelines" / "run_pipeline.py"


# ── Load/Save Config ──
@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    st.cache_resource.clear()


config = load_config()
mlops_cfg = config.get("mlops", {})

# ── Helpers ──
def resolve_encoder_config(cfg, key, default=None):
    """Read encoder config with backward compat."""
    mc = cfg.get("model", {})
    enc = mc.get("encoder_type", "wavlm")
    if "encoder_config" in mc and enc in mc["encoder_config"]:
        return mc["encoder_config"][enc].get(key, default)
    return mc.get(key, default)


# ═══════════════════════════════════════════════════════════
#  Sidebar — Live Config View
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📋 Current Config")
    mc = config.get("model", {})
    enc = mc.get("encoder_type", "?")
    pool = mc.get("pooling_type", "?")
    head = mc.get("speaker_head_type", "?")
    dur = config["audio"]["duration_seconds"]
    ep = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    freeze = resolve_encoder_config(config, "freeze_feature_extractor",
                                    resolve_encoder_config(config, "freeze_encoder", True))

    st.markdown(f"""
    | Param | Value |
    |-------|-------|
    | Encoder | `{enc}` |
    | Pooling | `{pool}` |
    | Head | `{head}` |
    | Freeze | `{freeze}` |
    | Duration | `{dur}s` |
    | Epochs | `{ep}` |
    | LR | `{lr}` |
    """)

    st.divider()
    st.caption(f"Config: `{CONFIG_PATH.name}`")
    st.caption(f"Branch: `feature/advanced-speaker-id`")


# ═══════════════════════════════════════════════════════════
#  Main Panel — Tabs
# ═══════════════════════════════════════════════════════════
st.title("🎤 Speaker-ID MLOps Center")
st.markdown("Open-Set Speaker Identification — IAAA 2026")

tab_cfg, tab_cloud, tab_local = st.tabs([
    "⚙️ Configuration", "☁️ Cloud Deploy (Vast.ai)", "💻 Local Run"
])

# ═══════════════════════════════════════════════
#  TAB 1: Configuration Editor
# ═══════════════════════════════════════════════
with tab_cfg:
    st.header("⚙️ Model & Training Configuration")
    st.markdown("Change parameters and click **Save Config** to apply.")

    col1, col2, col3 = st.columns(3)

    # ── Column 1: Encoder ──
    with col1:
        st.subheader("🧠 Encoder")
        encoder_type = st.selectbox(
            "Encoder Type",
            options=["wavlm", "ecapa", "hubert"],
            index=["wavlm", "ecapa", "hubert"].index(mc.get("encoder_type", "wavlm")),
            help="WavLM (best overall) | ECAPA-TDNN (fast, 192d) | HuBERT (large, diverse)",
        )
        freeze_fe = st.checkbox(
            "🔒 Freeze Feature Extractor",
            value=resolve_encoder_config(config, "freeze_feature_extractor",
                                          resolve_encoder_config(config, "freeze_encoder", True)),
            help="Freeze CNN stem to save VRAM. Uncheck for full fine-tune.",
        )

    # ── Column 2: Pooling + Head ──
    with col2:
        st.subheader("📊 Pooling & Head")
        pooling_type = st.selectbox(
            "Pooling Type",
            options=["attentive", "statistical", "identity"],
            index=["attentive", "statistical", "identity"].index(mc.get("pooling_type", "attentive")),
            help="Attentive (ASP) = best | Statistical = simple | Identity = for ECAPA",
        )
        head_type = st.selectbox(
            "Speaker Head",
            options=["arcface", "linear"],
            index=["arcface", "linear"].index(mc.get("speaker_head_type", "linear")),
            help="ArcFace = margin-based (better for OOD) | Linear = simple",
        )
        if head_type == "arcface":
            arc_cfg = mc.get("speaker_head_config", {}).get("arcface", {})
            arc_margin = st.slider("ArcFace margin", 0.1, 0.5, float(arc_cfg.get("margin", 0.3)), 0.05)
            arc_scale = st.slider("ArcFace scale", 5.0, 64.0, float(arc_cfg.get("scale", 15.0)), 1.0)
            arc_emb_dim = st.selectbox("Embedding dim", [128, 192, 256, 512],
                                       index=[128,192,256,512].index(arc_cfg.get("embedding_dim", 192)))

    # ── Column 3: Audio + Loss ──
    with col3:
        st.subheader("🎵 Audio & Loss")
        audio_dur = st.slider("Audio duration (s)", 1.0, 10.0,
                              float(config["audio"]["duration_seconds"]), 0.5,
                              help="Longer = more speaker info, more VRAM")
        min_dur = st.number_input("Min valid duration (s)", 0.0, 5.0,
                                  float(config["audio"].get("min_valid_duration", 1.0)), 0.5,
                                  help="Skip files shorter than this")
        use_focal = st.checkbox("🎯 Use Focal Loss", value=True,
                                help="Down-weight easy samples, focus on hard ones")
        focal_gamma = st.slider("Focal gamma", 0.0, 5.0, 2.0, 0.5,
                                help="0=CE, 2=standard focal")
        ood_hidden = st.number_input("OOD head hidden dim", 0, 1024,
                                     mc.get("ood_head_config", {}).get("hidden_dim", 256), 64)

    st.divider()

    # ── Training Hyperparams ──
    st.subheader("🏋️ Training Hyperparameters")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        epochs = st.number_input("Epochs", 1, 100, config["training"]["epochs"])
    with tc2:
        lr_val = st.number_input("Learning Rate", 1e-6, 1e-2, config["training"]["learning_rate"],
                                 format="%.6f")
    with tc3:
        wd = st.number_input("Weight Decay", 0.0, 1e-2, config["training"]["weight_decay"],
                             format="%.6f")
    with tc4:
        grad_norm = st.number_input("Max Grad Norm", 0.1, 50.0, config["training"]["max_grad_norm"])

    # ── Save Button ──
    st.divider()
    if st.button("💾 Save Config", type="primary", use_container_width=True):
        # Update encoder config
        if encoder_type == "wavlm":
            enc_cfg = {"base_model": "microsoft/wavlm-base-plus", "freeze_feature_extractor": freeze_fe}
        elif encoder_type == "ecapa":
            enc_cfg = {"source": "speechbrain/spkrec-ecapa-voxceleb", "freeze_encoder": freeze_fe}
        else:
            enc_cfg = {"base_model": "facebook/hubert-large-ls960-ft", "freeze_feature_extractor": freeze_fe}

        config["model"]["encoder_type"] = encoder_type
        config["model"]["encoder_config"][encoder_type] = enc_cfg
        config["model"]["pooling_type"] = pooling_type
        config["model"]["speaker_head_type"] = head_type
        if head_type == "arcface":
            config["model"]["speaker_head_config"]["arcface"] = {
                "embedding_dim": arc_emb_dim, "margin": arc_margin, "scale": arc_scale,
            }
        config["model"]["ood_head_config"]["hidden_dim"] = ood_hidden
        config["audio"]["duration_seconds"] = audio_dur
        config["audio"]["min_valid_duration"] = min_dur
        config["training"]["epochs"] = epochs
        config["training"]["learning_rate"] = lr_val
        config["training"]["weight_decay"] = wd
        config["training"]["max_grad_norm"] = grad_norm

        save_config(config)
        st.success("✅ Config saved! All changes applied.")
        st.rerun()


# ═══════════════════════════════════════════════
#  TAB 2: Cloud Deploy (Vast.ai)
# ═══════════════════════════════════════════════
with tab_cloud:
    st.header("☁️ Cloud Training on Vast.ai")
    st.markdown("Rent a GPU, run remotely, auto-destroy when done.")

    c1, c2 = st.columns(2)
    with c1:
        gpu_choice = st.selectbox("🎮 GPU", ["RTX_3090", "RTX_3060"], key="cloud_gpu")
    with c2:
        pipeline_choice = st.selectbox("📋 Stage",
                                       ["all", "data", "train", "eval"],
                                       format_func=lambda x: {
                                           "all": "🚀 Full Pipeline", "data": "📊 Data Only",
                                           "train": "🏋️ Train Only", "eval": "📈 Eval Only",
                                       }[x], key="cloud_stage")

    st.caption(f"💰 ~${ {'RTX_3090':0.35,'RTX_3060':0.15}[gpu_choice] * {'all':6,'train':5,'data':0.5,'eval':0.5}[pipeline_choice]:.1f}h "
               f"≈ ${ {'RTX_3090':0.35,'RTX_3060':0.15}[gpu_choice] * {'all':6,'train':5,'data':0.5,'eval':0.5}[pipeline_choice]:.2f}")

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        st.warning("⚠️ `.env` file not found! Copy `.env.example` → `.env` and fill credentials.")

    if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True, key="launch"):
        if not env_path.exists():
            st.error("❌ `.env` missing!")
            st.stop()

        os.environ["GPU_TARGET"] = gpu_choice
        os.environ["TARGET_PIPELINE"] = pipeline_choice
        os.environ["FREEZE_FEATURE_EXTRACTOR"] = str(freeze_fe).lower()

        with st.spinner("Renting GPU..."):
            try:
                proc_env = os.environ.copy()
                proc_env["PYTHONIOENCODING"] = "utf-8"
                result = subprocess.run(
                    [sys.executable, str(DEPLOY_SCRIPT)],
                    capture_output=True, text=True, encoding="utf-8",
                    env=proc_env, timeout=180, cwd=str(PROJECT_ROOT), check=True,
                )
                st.success("✅ Launched!")
                with st.expander("📋 Log"):
                    st.code(result.stdout)
                st.info(f"📊 Monitor: [{mlops_cfg.get('tracking',{}).get('uri','#')}]({mlops_cfg.get('tracking',{}).get('uri','#')})")
            except subprocess.CalledProcessError as e:
                st.error("❌ Failed!")
                st.code(e.stdout or "")
                if e.stderr:
                    st.code(e.stderr)


# ═══════════════════════════════════════════════
#  TAB 3: Local Run
# ═══════════════════════════════════════════════
with tab_local:
    st.header("💻 Local Training")
    st.caption(f"Config: `{encoder_type}` + `{pooling_type}` + `{head_type}` | "
               f"{audio_dur}s | {epochs} epochs | LR={lr_val}")

    import torch
    if torch.cuda.is_available():
        st.success(f"✅ {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
    else:
        st.warning("⚠️ No GPU — training will be slow on CPU.")

    lc1, lc2 = st.columns(2)
    with lc1:
        local_stage = st.selectbox("Stage", ["all", "data", "train", "eval"],
                                   format_func=lambda x: {"all":"🚀 Full","data":"📊 Data","train":"🏋️ Train","eval":"📈 Eval"}[x],
                                   key="local_stage")
    with lc2:
        with_mlflow = st.checkbox("📈 MLflow Tracking", value=True, key="local_mlflow")

    if st.button("▶️ Run Locally", type="primary", use_container_width=True, key="run_local"):
        cmd = [sys.executable, str(PIPELINE_SCRIPT), "--run", local_stage, "--config", str(CONFIG_PATH)]
        if not with_mlflow:
            cmd.append("--no-mlflow")

        with st.spinner(f"Running `{local_stage}` pipeline..."):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                                        cwd=str(PROJECT_ROOT))
                if result.returncode == 0:
                    st.success("✅ Done!")
                else:
                    st.error("❌ Failed!")
                with st.expander("📋 Log", expanded=True):
                    st.code(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
                    if result.stderr:
                        st.code(result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                st.error("⏰ Timeout (2h)")
