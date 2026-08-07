"""
Streamlit UI — Speaker-ID MLOps Center.
Usage: uv run streamlit run src/deploy/deploy_app.py
"""

import os, re, subprocess, sys, threading, time
from pathlib import Path
from queue import Queue

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


def _enc_val(key, default=None):
    mc = config.get("model", {})
    enc = mc.get("encoder_type", "wavlm")
    if "encoder_config" in mc and enc in mc["encoder_config"]:
        return mc["encoder_config"][enc].get(key, default)
    return mc.get(key, default)


# ═══════════════════════════════════════════════════════════
#  Log Streaming Engine
# ═══════════════════════════════════════════════════════════

def _stream_logs(instance_id: str, queue: Queue, stop_event: threading.Event):
    """Background thread: wait for instance to boot, then stream logs via retry loop."""
    max_retries = 24       # ~2 minutes total
    retry_delay = 5        # seconds between retries

    for attempt in range(max_retries):
        if stop_event.is_set():
            queue.put("__STREAM_END__")
            return

        try:
            proc = subprocess.Popen(
                ["vastai", "logs", instance_id],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            # Read first line to check for errors (container not ready yet)
            first_line = proc.stdout.readline()
            is_retryable = any(x in first_line for x in [
                "404", "Invalid instance", "No such container",
            ])
            if is_retryable:
                proc.terminate()
                if attempt == 0:
                    queue.put(f"⏳ Instance booting... waiting for logs (attempt {attempt+1}/{max_retries})")
                elif attempt % 3 == 0:
                    queue.put(f"⏳ Still waiting... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                continue

            # Success — stream the rest
            queue.put(first_line.rstrip())
            for line in iter(proc.stdout.readline, ""):
                if stop_event.is_set():
                    proc.terminate()
                    break
                queue.put(line.rstrip())
            proc.stdout.close()
            proc.wait()
            break  # normal exit

        except Exception as e:
            queue.put(f"[log error] {e}")
            time.sleep(retry_delay)

    else:
        queue.put("⚠️ Could not connect to logs after 2min. Instance may still be provisioning.")
    queue.put("__STREAM_END__")


# ═══════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
st.title("🎤 Speaker-ID MLOps Center")
tab_cfg, tab_cloud, tab_local = st.tabs(["⚙️ Config", "☁️ Cloud (Vast.ai)", "💻 Local"])

# ── TAB: Config ──
with tab_cfg:
    st.header("⚙️ Model Setup")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧠 Encoder")
        encoder_type = st.selectbox(
            "Encoder", ["wavlm", "ecapa", "hubert"],
            index=["wavlm","ecapa","hubert"].index(mc.get("encoder_type","wavlm")))
        freeze = st.checkbox("🔒 Freeze encoder", value=_enc_val(
            "freeze_feature_extractor", _enc_val("freeze_encoder", True)))

        pool_opts = ["attentive", "identity"]
        pool_idx = 1 if encoder_type == "ecapa" else 0
        if encoder_type == "ecapa":
            st.info("💡 ECAPA has built-in ASP → pooling = identity")
        pooling_type = st.selectbox("Pooling", pool_opts, index=pool_idx)

        st.subheader("🎯 ArcFace Head")
        arc_cfg = mc.get("speaker_head_config", {}).get("arcface", {})
        arc_m = st.slider("Margin", 0.1, 0.5, float(arc_cfg.get("margin", 0.3)), 0.05)
        arc_s = st.slider("Scale", 5.0, 64.0, float(arc_cfg.get("scale", 15.0)), 1.0)
        arc_emb = st.selectbox("Embedding dim", [128, 192, 256],
                               index=[128,192,256].index(arc_cfg.get("embedding_dim",192)))
    with c2:
        st.subheader("🎵 Audio")
        audio_dur = st.slider("Duration (s)", 2.0, 8.0, float(config["audio"]["duration_seconds"]), 0.5)
        min_dur = st.number_input("Min valid (s)", 0.0, 5.0,
                                  float(config["audio"].get("min_valid_duration",1.0)), 0.5)
        st.subheader("🏋️ Training")
        epochs = st.number_input("Epochs", 1, 100, config["training"]["epochs"])
        lr_val = st.number_input("LR", 1e-6, 1e-2, config["training"]["learning_rate"], format="%.6f")
        wd = st.number_input("Weight Decay", 0.0, 1e-2, config["training"]["weight_decay"], format="%.6f")
        grad_norm = st.number_input("Max Grad Norm", 0.1, 50.0, config["training"]["max_grad_norm"])
        st.subheader("🎯 Loss")
        st.caption("Focal Loss always ON (γ=2).")
        ood_hidden = st.number_input("OOD head hidden dim", 0, 1024,
                                     mc.get("ood_head_config",{}).get("hidden_dim",256), 64)

    if st.button("💾 Save Config", type="primary", use_container_width=True):
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
            "embedding_dim": arc_emb, "margin": arc_m, "scale": arc_s}
        config["model"]["ood_head_config"]["hidden_dim"] = ood_hidden
        config["audio"]["duration_seconds"] = audio_dur
        config["audio"]["min_valid_duration"] = min_dur
        config["training"]["epochs"] = epochs
        config["training"]["learning_rate"] = lr_val
        config["training"]["weight_decay"] = wd
        config["training"]["max_grad_norm"] = grad_norm
        save_config(config)
        config = load_config()
        st.success("✅ Saved!")
        st.rerun()


# ── TAB: Cloud ──
with tab_cloud:
    st.header("☁️ Vast.ai — Live Logs")
    c1, c2, c3 = st.columns(3)
    with c1:
        gpu = st.selectbox("GPU", ["RTX_3090", "RTX_3060"], key="cgpu")
    with c2:
        stage = st.selectbox("Stage", ["all","data","train","eval"],
                             format_func=lambda x: {"all":"🚀 Full","data":"📊 Data","train":"🏋️ Train","eval":"📈 Eval"}[x],
                             key="cstage")
    with c3:
        disk_gb = st.slider("💾 Disk (GB)", 20, 200,
                            config.get("mlops",{}).get("vast",{}).get("disk_size", 60), 10,
                            help="More disk for bigger models/datasets.")

    if not (PROJECT_ROOT / ".env").exists():
        st.warning("⚠️ `.env` missing — copy from `.env.example`")

    # ── Session state for live logs ──
    if "log_instance_id" not in st.session_state:
        st.session_state.log_instance_id = None
    if "log_running" not in st.session_state:
        st.session_state.log_running = False
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = Queue()
    if "log_stop" not in st.session_state:
        st.session_state.log_stop = threading.Event()

    col_launch, col_destroy = st.columns([3, 1])

    with col_launch:
        if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True,
                     disabled=st.session_state.log_running,
                     key="launch"):
            if not (PROJECT_ROOT / ".env").exists():
                st.error("❌ .env missing!"); st.stop()
            os.environ["GPU_TARGET"] = gpu
            os.environ["TARGET_PIPELINE"] = stage
            os.environ["FREEZE_FEATURE_EXTRACTOR"] = str(freeze).lower()
            os.environ["DISK_SIZE_GB"] = str(disk_gb)
            with st.spinner("Creating instance..."):
                try:
                    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
                    r = subprocess.run([sys.executable, str(DEPLOY_SCRIPT)],
                                       capture_output=True, text=True, encoding="utf-8",
                                       env=env, timeout=180, cwd=str(PROJECT_ROOT), check=True)
                    output = r.stdout
                    # Extract instance ID
                    m = re.search(r"INSTANCE_ID=(\d+)", output)
                    if m:
                        st.session_state.log_instance_id = m.group(1)
                        st.session_state.log_running = True
                        st.session_state.log_stop.clear()
                        st.session_state.log_queue = Queue()
                        # Start background log streamer
                        t = threading.Thread(
                            target=_stream_logs,
                            args=(st.session_state.log_instance_id,
                                  st.session_state.log_queue,
                                  st.session_state.log_stop),
                            daemon=True)
                        t.start()
                        st.success(f"✅ Instance #{st.session_state.log_instance_id} launched!")
                    else:
                        st.warning("⚠️ Launched but could not parse instance ID.")
                    with st.expander("Deploy output"):
                        st.code(output)
                except subprocess.CalledProcessError as e:
                    st.error("❌ Deploy failed!")
                    st.code(e.stdout or "")

    with col_destroy:
        if st.button("🛑 Destroy", type="secondary", use_container_width=True,
                     disabled=not st.session_state.log_running,
                     key="destroy"):
            iid = st.session_state.log_instance_id
            if iid:
                with st.spinner(f"Destroying #{iid}..."):
                    try:
                        subprocess.run(["vastai", "destroy", "instance", iid, "-y"],
                                       capture_output=True, text=True, timeout=30)
                    except Exception:
                        pass
                st.session_state.log_stop.set()
                st.session_state.log_running = False
                st.session_state.log_instance_id = None
                st.warning(f"🛑 Instance #{iid} destroyed.")
                st.rerun()

    # ── Live log display ──
    if st.session_state.log_running and st.session_state.log_instance_id:
        st.divider()
        st.subheader(f"📜 Live Logs — Instance #{st.session_state.log_instance_id}")
        log_placeholder = st.empty()

        # Drain queue and display
        lines = []
        q = st.session_state.log_queue
        while not q.empty():
            line = q.get_nowait()
            if line == "__STREAM_END__":
                st.session_state.log_running = False
                st.warning("🏁 Log stream ended (instance finished or destroyed).")
                break
            lines.append(line)

        if lines:
            # Keep last 200 lines max
            if "log_history" not in st.session_state:
                st.session_state.log_history = []
            st.session_state.log_history.extend(lines)
            st.session_state.log_history = st.session_state.log_history[-200:]
            log_placeholder.code("\n".join(st.session_state.log_history))

        # Auto-refresh every 1 second
        time.sleep(1)
        st.rerun()


# ── TAB: Local ──
with tab_local:
    st.header("💻 Local Run")
    st.caption(f"`{encoder_type}` + `{pooling_type}` + ArcFace | {audio_dur}s | {epochs}ep | LR={lr_val}")
    import torch
    if torch.cuda.is_available():
        st.success(f"✅ {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        st.warning("⚠️ CPU only — slow.")
    lc1, lc2 = st.columns(2)
    with lc1:
        ls = st.selectbox("Stage", ["all","data","train","eval"],
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
