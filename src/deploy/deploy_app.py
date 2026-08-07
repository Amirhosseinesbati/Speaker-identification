"""
Streamlit UI — Speaker-ID MLOps Center.

All meaningful parameters exposed. No clutter.
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


@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    st.cache_resource.clear()


config = load_config()

# ── Helpers ──
def _enc_val(key, default=None):
    mc = config.get("model", {})
    enc = mc.get("encoder_type", "wavlm")
    if "encoder_config" in mc and enc in mc["encoder_config"]:
        return mc["encoder_config"][enc].get(key, default)
    return mc.get(key, default)


# ═══════════════════════════════════════════
#  Sidebar — Live Status
# ═══════════════════════════════════════════
with st.sidebar:
    st.header("📋 Active Config")
    mc = config.get("model", {})
    enc = mc.get("encoder_type", "?")
    pool = mc.get("pooling_type", "?")
    freeze = _enc_val("freeze_feature_extractor", _enc_val("freeze_encoder", True))
    dur = config["audio"]["duration_seconds"]
    arc = mc.get("speaker_head_config", {}).get("arcface", {})

    st.markdown(f"""
    | Param | Value |
    |-------|-------|
    | Encoder | `{enc}` |
    | Pooling | `{pool}` |
    | Head | ArcFace (m={arc.get('margin',0.3)}, s={arc.get('scale',15)}) |
    | Freeze | `{freeze}` |
    | Duration | `{dur}s` |
    | Epochs | `{config['training']['epochs']}` |
    | LR | `{config['training']['learning_rate']}` |
    """)
    st.caption(f"Branch: `feature/advanced-speaker-id`")


# ═══════════════════════════════════════════
#  Main — 3 Tabs
# ═══════════════════════════════════════════
st.title("🎤 Speaker-ID MLOps Center")

tab_cfg, tab_cloud, tab_local = st.tabs([
    "⚙️ Configuration", "☁️ Cloud (Vast.ai)", "💻 Local"
])

# ── TAB 1: Configuration ──
with tab_cfg:
    st.header("⚙️ Model Setup")
    st.caption("Only meaningful options. ArcFace + ASP = best for open-set.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🧠 Encoder")
        encoder_type = st.selectbox(
            "Encoder",
            ["wavlm", "ecapa", "hubert"],
            index=["wavlm", "ecapa", "hubert"].index(mc.get("encoder_type", "wavlm")),
            help="WavLM = best overall | ECAPA = fast, 192d | HuBERT = large, diverse",
        )
        freeze = st.checkbox("🔒 Freeze encoder", value=_enc_val(
            "freeze_feature_extractor", _enc_val("freeze_encoder", True)),
            help="Freeze = less VRAM. Uncheck for full fine-tune (needs 24GB+)")

        # Pooling: attentive by default, identity for ECAPA
        pool_opts = ["attentive", "identity"]
        if encoder_type == "ecapa":
            pool_idx = 1  # identity
            st.info("💡 ECAPA has built-in ASP → pooling = identity")
        else:
            pool_idx = 0  # attentive
        pooling_type = st.selectbox("Pooling", pool_opts, index=pool_idx,
                                    help="Attentive = ASP (best). Identity = encoder has its own.")

        # Head: only ArcFace
        st.subheader("🎯 Speaker Head — ArcFace")
        arc_cfg = mc.get("speaker_head_config", {}).get("arcface", {})
        arc_margin = st.slider("Margin", 0.1, 0.5, float(arc_cfg.get("margin", 0.3)), 0.05,
                               help="Angular margin between speakers. 0.3 = standard.")
        arc_scale = st.slider("Scale", 5.0, 64.0, float(arc_cfg.get("scale", 15.0)), 1.0,
                              help="Feature scale. 15-30 typical.")
        arc_emb = st.selectbox("Embedding dim", [128, 192, 256],
                               index=[128, 192, 256].index(arc_cfg.get("embedding_dim", 192)),
                               help="192 = standard, matches ECAPA/TitaNet.")

    with c2:
        st.subheader("🎵 Audio")
        audio_dur = st.slider("Duration (s)", 2.0, 8.0,
                              float(config["audio"]["duration_seconds"]), 0.5,
                              help="5s = sweet spot. Longer = more info but more VRAM.")
        min_dur = st.number_input("Min valid (s)", 0.0, 5.0,
                                  float(config["audio"].get("min_valid_duration", 1.0)), 0.5,
                                  help="Skip files shorter than this (corrupted detection).")

        st.subheader("🏋️ Training")
        epochs = st.number_input("Epochs", 1, 100, config["training"]["epochs"])
        lr_val = st.number_input("Learning Rate", 1e-6, 1e-2, config["training"]["learning_rate"],
                                 format="%.6f")
        wd = st.number_input("Weight Decay", 0.0, 1e-2, config["training"]["weight_decay"],
                             format="%.6f")
        grad_norm = st.number_input("Max Grad Norm", 0.1, 50.0,
                                    config["training"]["max_grad_norm"])

        st.subheader("🎯 Loss")
        st.caption("Focal Loss always ON (γ=2) — best for imbalanced open-set.")
        ood_hidden = st.number_input("OOD head hidden dim", 0, 1024,
                                     mc.get("ood_head_config", {}).get("hidden_dim", 256), 64,
                                     help="0 = single linear layer.")

    st.divider()
    if st.button("💾 Save Config & Apply", type="primary", use_container_width=True):
        if encoder_type == "wavlm":
            enc_cfg = {"base_model": "microsoft/wavlm-base-plus", "freeze_feature_extractor": freeze}
        elif encoder_type == "ecapa":
            enc_cfg = {"source": "speechbrain/spkrec-ecapa-voxceleb", "freeze_encoder": freeze}
        else:
            enc_cfg = {"base_model": "facebook/hubert-large-ls960-ft", "freeze_feature_extractor": freeze}

        config["model"]["encoder_type"] = encoder_type
        config["model"]["encoder_config"][encoder_type] = enc_cfg
        config["model"]["pooling_type"] = pooling_type
        config["model"]["speaker_head_type"] = "arcface"
        config["model"]["speaker_head_config"]["arcface"] = {
            "embedding_dim": arc_emb, "margin": arc_margin, "scale": arc_scale,
        }
        config["model"]["ood_head_config"]["hidden_dim"] = ood_hidden
        config["audio"]["duration_seconds"] = audio_dur
        config["audio"]["min_valid_duration"] = min_dur
        config["training"]["epochs"] = epochs
        config["training"]["learning_rate"] = lr_val
        config["training"]["weight_decay"] = wd
        config["training"]["max_grad_norm"] = grad_norm

        save_config(config)
        config = load_config()
        st.success("✅ Saved! Refresh sidebar to see changes.")
        st.rerun()


# ── TAB 2: Cloud ──
with tab_cloud:
    st.header("☁️ Vast.ai Cloud Training")
    c1, c2 = st.columns(2)
    with c1:
        gpu = st.selectbox("GPU", ["RTX_3090", "RTX_3060"], key="cgpu")
    with c2:
        stage = st.selectbox("Stage", ["all", "data", "train", "eval"],
                             format_func=lambda x: {"all":"🚀 Full","data":"📊 Data","train":"🏋️ Train","eval":"📈 Eval"}[x],
                             key="cstage")

    st.caption(f"💰 ~${{ {'RTX_3090':0.35,'RTX_3060':0.15}[gpu] * {'all':6,'train':5,'data':0.5,'eval':0.5}[stage]:.1f}h "
               f"≈ ${{ {'RTX_3090':0.35,'RTX_3060':0.15}[gpu] * {'all':6,'train':5,'data':0.5,'eval':0.5}[stage]:.2f}")

    if not (PROJECT_ROOT / ".env").exists():
        st.warning("⚠️ `.env` missing — copy from `.env.example`")

    if st.button("🔥 Launch", type="primary", use_container_width=True, key="launch"):
        if not (PROJECT_ROOT / ".env").exists():
            st.error("❌ .env missing!"); st.stop()
        os.environ["GPU_TARGET"] = gpu
        os.environ["TARGET_PIPELINE"] = stage
        os.environ["FREEZE_FEATURE_EXTRACTOR"] = str(freeze).lower()
        with st.spinner("Renting GPU..."):
            try:
                env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
                r = subprocess.run([sys.executable, str(DEPLOY_SCRIPT)], capture_output=True,
                                   text=True, encoding="utf-8", env=env, timeout=180,
                                   cwd=str(PROJECT_ROOT), check=True)
                st.success("✅ Launched!")
                with st.expander("Log"): st.code(r.stdout)
            except subprocess.CalledProcessError as e:
                st.error("❌ Failed!"); st.code(e.stdout or "")


# ── TAB 3: Local ──
with tab_local:
    st.header("💻 Local Run")
    st.caption(f"`{encoder_type}` + `{pooling_type}` + ArcFace | {audio_dur}s | {epochs}ep | LR={lr_val}")

    import torch
    if torch.cuda.is_available():
        st.success(f"✅ {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
    else:
        st.warning("⚠️ CPU only — will be slow.")

    lc1, lc2 = st.columns(2)
    with lc1:
        ls = st.selectbox("Stage", ["all", "data", "train", "eval"],
                          format_func=lambda x: {"all":"🚀 Full","data":"📊 Data","train":"🏋️ Train","eval":"📈 Eval"}[x],
                          key="lstage")
    with lc2:
        mlflow_on = st.checkbox("📈 MLflow", value=True, key="lmlflow")

    if st.button("▶️ Run", type="primary", use_container_width=True, key="lrun"):
        cmd = [sys.executable, str(PIPELINE_SCRIPT), "--run", ls, "--config", str(CONFIG_PATH)]
        if not mlflow_on: cmd.append("--no-mlflow")
        with st.spinner(f"Running `{ls}`..."):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(PROJECT_ROOT))
                if r.returncode == 0: st.success("✅ Done!")
                else: st.error("❌ Failed!")
                with st.expander("Log", expanded=True):
                    st.code(r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout)
                    if r.stderr: st.code(r.stderr[-2000:])
            except subprocess.TimeoutExpired:
                st.error("⏰ Timeout (2h)")
