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


def _run_local(cmd: list, timeout: int = 7200) -> str:
    """Run a local subprocess and return the (truncated) combined output."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(PROJECT_ROOT))
    out = r.stdout or ""
    if r.returncode != 0:
        out += "\n[STDERR]\n" + (r.stderr or "")
    return (f"[exit={r.returncode}]\n" + out)[-8000:]


config = load_config()


def _enc_val(key, default=None):
    mc = config.get("model", {})
    enc = mc.get("encoder_type", "wavlm")
    if "encoder_config" in mc and enc in mc["encoder_config"]:
        return mc["encoder_config"][enc].get(key, default)
    return mc.get(key, default)


def _enc_freeze() -> bool:
    """True if the active encoder's freeze flag is set (ECAPA: freeze_encoder)."""
    return bool(_enc_val("freeze_feature_extractor", _enc_val("freeze_encoder", True)))


def _enc_unfreeze_blocks() -> int:
    """unfreeze_last_n_blocks of the active encoder (0 = not partial)."""
    return int(_enc_val("unfreeze_last_n_blocks", 0) or 0)


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
    freeze = _enc_freeze()
    blocks = _enc_unfreeze_blocks()
    dur = config["audio"]["duration_seconds"]
    nwin = config["audio"].get("num_train_windows", "-")
    ehop = config["audio"].get("eval_hop_ratio", "-")
    mwin = config["audio"].get("max_eval_windows", "-")
    oodr = config["audio"].get("ood_batch_ratio", "-")
    enc_lr = config["training"].get("encoder_lr", "-")
    arc = mc.get("speaker_head_config", {}).get("arcface", {})
    ft_label = ("Frozen" if freeze
                else (f"Partial (last {blocks})" if blocks and blocks > 0 else "Full"))
    st.markdown(f"""
    | Param | Value |
    |-------|-------|
    | Encoder | `{enc}` |
    | Pooling | `{pool}` |
    | Head | ArcFace (m={arc.get('margin',0.4)}, s={arc.get('scale',30)}) |
    | Fine-tune | `{ft_label}` |
    | Duration | `{dur}s` |
    | Windows | train `{nwin}` / eval `{mwin}` (hop `{ehop}`) |
    | OOD ratio | `{oodr}` |
    | Epochs | `{config['training']['epochs']}` |
    | LR (head/enc) | `{config['training']['learning_rate']}` / `{enc_lr}` |
    """)
    st.caption(f"Branch: `feature/advanced-speaker-id`")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
st.title("🎤 Speaker-ID MLOps Center")
tab_cfg, tab_cloud, tab_local, tab_analysis = st.tabs(
    ["⚙️ Config", "☁️ Cloud (Vast.ai)", "💻 Local", "🧪 Analysis"])

# ── TAB: Config ──
with tab_cfg:
    st.header("⚙️ Model Setup")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧠 Encoder")
        encoder_type = st.selectbox(
            "Encoder", ["wavlm", "ecapa", "hubert"],
            index=["wavlm","ecapa","hubert"].index(mc.get("encoder_type","wavlm")))

        # Fine-tune mode: Frozen / Partial (last N, ECAPA) / Full
        cur_freeze = _enc_freeze()
        cur_blocks = _enc_unfreeze_blocks()
        ft_options = ["Frozen", "Full"]
        if encoder_type == "ecapa":
            ft_options = ["Frozen", "Partial (last N)", "Full"]
        if cur_freeze:
            ft_idx = 0
        elif encoder_type == "ecapa" and cur_blocks > 0:
            ft_idx = 1
        else:
            ft_idx = len(ft_options) - 1
        ft_mode = st.radio(
            "Fine-tune mode", ft_options, index=ft_idx, horizontal=True,
            help="Frozen: encoder weights fixed. Partial (ECAPA): only the last N "
                 "SE-Res2Blocks are trainable. Full: all encoder parameters trainable.",
        )
        unfreeze_n = 2
        if ft_mode == "Partial (last N)":
            unfreeze_n = st.number_input("Unfreeze last N blocks", 1, 8, int(cur_blocks or 2))

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
        num_win = st.number_input("Train windows/file", 1, 8,
                                  int(config["audio"].get("num_train_windows", 3)),
                                  help="Random crops per file in training (multi-window TTA).")
        hop_ratio = st.slider("Eval hop ratio", 0.25, 0.9,
                              float(config["audio"].get("eval_hop_ratio", 0.5)), 0.05)
        max_win = st.number_input("Max eval windows", 1, 32,
                                  int(config["audio"].get("max_eval_windows", 8)))
        ood_ratio = st.slider("OOD batch ratio", 0.1, 0.9,
                              float(config["audio"].get("ood_batch_ratio", 0.5)), 0.05)
        st.subheader("🏋️ Training")
        epochs = st.number_input("Epochs", 1, 100, config["training"]["epochs"])
        lr_val = st.number_input("LR (heads)", 1e-6, 1e-2, config["training"]["learning_rate"], format="%.6f")
        encoder_lr = st.number_input("LR (encoder)", 1e-7, 1e-2,
                                     float(config["training"].get("encoder_lr", 1e-5)), format="%.6f",
                                     help="LR for unfrozen encoder blocks (fine-tune).")
        wd = st.number_input("Weight Decay", 0.0, 1e-2, config["training"]["weight_decay"], format="%.6f")
        grad_norm = st.number_input("Max Grad Norm", 0.1, 50.0, config["training"]["max_grad_norm"])
        patience = st.number_input("Early stop patience", 1, 50,
                                   int(config["training"].get("early_stopping_patience", 10)),
                                   help="Early stopping / checkpoint selection on val Macro-F1.")
        st.subheader("🎯 Loss")
        st.caption("Focal Loss always ON (γ=2).")
        ood_hidden = st.number_input("OOD head hidden dim", 0, 1024,
                                     mc.get("ood_head_config",{}).get("hidden_dim",256), 64)
        ood_pos_w = st.number_input("OOD pos_weight", 0.1, 10.0,
                                    float(config["training"].get("ood_pos_weight", 1.0)), 0.1)
        ood_w = st.number_input("OOD loss weight", 0.0, 1.0,
                                float(config["training"].get("ood_loss_weight", 0.3)), 0.05)
        spk_w = st.number_input("Speaker loss weight", 0.0, 1.0,
                                float(config["training"].get("speaker_loss_weight", 0.7)), 0.05)
        sm_val = st.number_input("Label smoothing", 0.0, 0.5,
                                 float(config["training"].get("label_smoothing", 0.1)), 0.05)

    if st.button("💾 Save Config", type="primary", use_container_width=True):
        # ── Encoder config: MERGE with existing keys so partial fine-tune
        #    settings (e.g. unfreeze_last_n_blocks) are never silently dropped.
        old_enc = dict(config["model"].get("encoder_config", {}).get(encoder_type, {}))
        if encoder_type == "ecapa":
            new_enc = {
                "source": "speechbrain/spkrec-ecapa-voxceleb",
                "freeze_encoder": ft_mode == "Frozen",
                "unfreeze_last_n_blocks": int(unfreeze_n) if ft_mode == "Partial (last N)" else 0,
            }
            old_enc.pop("freeze_feature_extractor", None)  # stale key for ECAPA
        else:
            new_enc = {
                "base_model": ("microsoft/wavlm-base-plus" if encoder_type == "wavlm"
                               else "facebook/hubert-large-ls960-ft"),
                "freeze_feature_extractor": ft_mode == "Frozen",
            }
            old_enc.pop("freeze_encoder", None)
            old_enc.pop("unfreeze_last_n_blocks", None)
        config["model"]["encoder_type"] = encoder_type
        config["model"].setdefault("encoder_config", {})[encoder_type] = {**old_enc, **new_enc}
        config["model"]["pooling_type"] = pooling_type
        config["model"]["speaker_head_type"] = "arcface"
        config["model"]["speaker_head_config"]["arcface"] = {
            "embedding_dim": arc_emb, "margin": arc_m, "scale": arc_s}
        config["model"]["ood_head_config"]["hidden_dim"] = ood_hidden
        config["audio"]["duration_seconds"] = audio_dur
        config["audio"]["min_valid_duration"] = min_dur
        config["audio"]["num_train_windows"] = int(num_win)
        config["audio"]["eval_hop_ratio"] = float(hop_ratio)
        config["audio"]["max_eval_windows"] = int(max_win)
        config["audio"]["ood_batch_ratio"] = float(ood_ratio)
        config["training"]["epochs"] = epochs
        config["training"]["learning_rate"] = lr_val
        config["training"]["encoder_lr"] = float(encoder_lr)
        config["training"]["weight_decay"] = wd
        config["training"]["max_grad_norm"] = grad_norm
        config["training"]["early_stopping_patience"] = int(patience)
        config["training"]["ood_pos_weight"] = float(ood_pos_w)
        config["training"]["ood_loss_weight"] = float(ood_w)
        config["training"]["speaker_loss_weight"] = float(spk_w)
        config["training"]["label_smoothing"] = float(sm_val)
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
            # Encoder fine-tune choice (ECAPA-aware; setup_vast.sh reads these)
            os.environ["FREEZE_ENCODER"] = str(ft_mode == "Frozen").lower()
            os.environ["UNFREEZE_LAST_N_BLOCKS"] = str(
                unfreeze_n if ft_mode == "Partial (last N)" else 0)
            os.environ["FREEZE_FEATURE_EXTRACTOR"] = str(ft_mode == "Frozen").lower()  # compat
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


# ── TAB: Analysis ──
with tab_analysis:
    st.header("🧪 Analysis & Tools")
    st.caption("Run the new pipeline modules locally: unbiased EDA, centroid baseline, "
               "ensemble calibration, and submission inference.")

    ac1, ac2 = st.columns(2)
    with ac1:
        st.subheader("📊 Unbiased EDA")
        st.caption("Multi-window ECAPA embeddings + LOO centroid + Macro-F1 simulation "
                   "(GPU, several minutes).")
        if st.button("▶️ eda_embeddings", key="a_eda", use_container_width=True):
            with st.spinner("Running eda_embeddings..."):
                try:
                    out = _run_local([sys.executable, "-m", "src.eda_embeddings"], timeout=3600)
                    st.code(out)
                    if out.startswith("[exit=0]"):
                        st.success("Done!")
                    else:
                        st.error("Failed (see log).")
                except subprocess.TimeoutExpired:
                    st.error("⏰ Timeout (1h)")

        st.subheader("🎯 Centroid baseline")
        st.caption("Embedding cache (idempotent) + centroid classifier + fusion report.")
        force_cache = st.checkbox("Rebuild embedding cache", key="a_force")
        if st.button("▶️ centroid_baseline", key="a_centroid", use_container_width=True):
            cmd = [sys.executable, "-m", "src.centroid_baseline"]
            if force_cache:
                cmd.append("--force-cache")
            with st.spinner("Running centroid_baseline..."):
                try:
                    out = _run_local(cmd, timeout=7200)
                    st.code(out)
                    if out.startswith("[exit=0]"):
                        st.success("Done!")
                    else:
                        st.error("Failed (see log).")
                except subprocess.TimeoutExpired:
                    st.error("⏰ Timeout (2h)")

    with ac2:
        st.subheader("🧩 Ensemble + temperature")
        st.caption("Per-model & ensemble Macro-F1 + best softmax temperature "
                   "(needs ≥2 trained checkpoints).")
        ckpts = st.text_input("Checkpoint paths (space-separated)",
                              value="checkpoints/best_seed1.pt checkpoints/best_seed2.pt",
                              key="a_ckpts")
        if st.button("▶️ ensemble_calibrate", key="a_ens", use_container_width=True):
            ckpt_list = [c for c in ckpts.split() if c]
            if len(ckpt_list) < 2:
                st.warning("⚠️ Provide at least 2 checkpoint paths.")
            else:
                cmd = [sys.executable, "-m", "src.ensemble_calibrate",
                       "--checkpoints"] + ckpt_list
                with st.spinner("Running ensemble_calibrate..."):
                    try:
                        out = _run_local(cmd, timeout=7200)
                        st.code(out)
                        if out.startswith("[exit=0]"):
                            st.success("Done!")
                        else:
                            st.error("Failed (see log).")
                    except subprocess.TimeoutExpired:
                        st.error("⏰ Timeout (2h)")

        st.subheader("📤 Submission CSV")
        st.caption("Run submission.inference on a test folder → 447-column CSV.")
        s_dir = st.text_input("Test data dir", value="data/processed/audio_wav", key="a_dir")
        s_out = st.text_input("Output CSV", value="predictions.csv", key="a_out")
        if st.button("▶️ inference → CSV", key="a_inf", use_container_width=True):
            cmd = [sys.executable, "-m", "submission.inference",
                   "--data-dir", s_dir, "--predictions-file-path", s_out]
            with st.spinner("Running inference..."):
                try:
                    out = _run_local(cmd, timeout=7200)
                    st.code(out)
                    if out.startswith("[exit=0]"):
                        st.success("Done!")
                    else:
                        st.error("Failed (see log).")
                except subprocess.TimeoutExpired:
                    st.error("⏰ Timeout (2h)")
