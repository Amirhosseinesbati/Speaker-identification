"""
Streamlit UI — Speaker-ID MLOps Center.
Usage: uv run streamlit run src/deploy/deploy_app.py
"""

import os, re, subprocess, sys, threading, time
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml

st.set_page_config(page_title="Speaker-ID MLOps", page_icon="🎤", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
DEPLOY_SCRIPT = PROJECT_ROOT / "src" / "deploy" / "deploy.py"
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipelines" / "run_pipeline.py"

from src.experiment_config import list_profiles, load_profile, save_profile
from src.cli_utils import pump_pipe

# Vast.ai GPU targets offered in the Cloud tab. Keep in sync with
# configs/default_config.yaml → mlops.vast.gpu_options and setup_vast.sh.
GPU_OPTIONS = ["RTX_3090", "RTX_3060", "RTX_A4000"]


@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    st.cache_resource.clear()


class LocalRunner:
    """
    Long-running local subprocess with live log streaming + stop support.

    The Popen handle lives in st.session_state (via the runner object) so it
    survives Streamlit reruns. A daemon thread drains stdout into `lines`; the
    UI renders them on every rerun and a 🛑 Stop button calls stop().
    """

    def __init__(self, cmd: list, cwd: str):
        self.cmd = cmd
        self.cwd = cwd
        self.lines: list = []
        self.current: str = ""   # latest \r-updated progress line (single live line)
        self.finished = False
        self.returncode: Optional[int] = None
        self._stop = threading.Event()
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        # PYTHONUNBUFFERED=1 + our entry points' setup_utf8_stdio(line_buffering)
        # → every print() line is flushed to the pipe immediately (live logs).
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
               "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=self.cwd, env=env,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            # pump_pipe splits \n lines from \r progress updates, so download /
            # tqdm bars render as ONE live line (self.current) instead of log
            # spam, and never block readline waiting for a newline.
            def on_line(line):
                if self._stop.is_set():
                    return
                self.lines.append(line)
                self.current = ""   # a newline line ends any \r progress bar

            def on_progress(text):
                if self._stop.is_set():
                    return
                self.current = text

            pump_pipe(self.proc.stdout.buffer, on_line, on_progress)
        except Exception:
            pass
        finally:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            try:
                self.proc.wait()
            except Exception:
                pass
            self.returncode = self.proc.returncode
            self.finished = True

    def stop(self):
        """Graceful terminate, then force-kill after a short grace period."""
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def _progress_pct(line: str) -> Optional[float]:
    """First percentage in a progress line (download / extraction / tqdm bar)."""
    m = re.search(r"(\d{1,3})%", line)
    if m:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return pct / 100.0
    return None


def _render_local_runner(key: str, title: str) -> None:
    """Live-log + stop panel for a running/finished LocalRunner in session_state."""
    runner = st.session_state.get(key)
    if runner is None:
        return

    if runner.running and not runner.finished:
        st.warning(f"⏳ `{title}` در حال اجراست — لاگ‌ها به‌صورت زنده به‌روز می‌شوند. "
                   f"برای توقف دکمه‌ی 🛑 را بزنید.")
        if runner.current:
            pct = _progress_pct(runner.current)
            if pct is not None:
                st.progress(pct)
            st.code(runner.current, language=None)
        st.code("\n".join(runner.lines[-500:]), language=None)
        if st.button("🛑 Stop", key=f"stop_{key}", type="secondary"):
            runner.stop()
            st.rerun()
        time.sleep(0.7)
        st.rerun()
    elif runner.finished:
        if runner.returncode == 0:
            st.success(f"✅ `{title}` با موفقیت تمام شد (exit=0).")
        else:
            st.error(f"❌ `{title}` ناموفق بود (exit={runner.returncode}).")
        with st.expander("📜 Full log", expanded=True):
            st.code("\n".join(runner.lines[-800:]), language=None)
        if st.button("🗑 Clear", key=f"clear_{key}"):
            st.session_state[key] = None
            st.rerun()


# Active config source: a loaded experiment profile (session_state) wins over
# the base file. The widgets below read/write this dict; saving branches on the
# experiment name (see the 💾 Save handler at the bottom of the Config tab).
if "edit_config" not in st.session_state:
    st.session_state.edit_config = None
    st.session_state.loaded_profile_name = "(base)"
config = (
    st.session_state.edit_config
    if st.session_state.edit_config is not None
    else load_config()
)


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


def _encoder_save_config(encoder_type: str, old_enc: dict, ft_mode: str,
                         unfreeze_n: int) -> dict:
    """
    Build the encoder_config dict the Save button writes (pure — testable).

    Args:
        encoder_type: active encoder key (ecapa/wavlm/campp/eres2net/titanet)
        old_enc: existing per-encoder config (mutated to drop stale keys)
        ft_mode: "Frozen" | "Partial (last N)" (ECAPA only) | "Full"
        unfreeze_n: number of blocks to unfreeze (partial ECAPA only)

    Returns:
        new_enc dict merged over old_enc by the caller.
    """
    if encoder_type == "ecapa":
        new_enc = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "freeze_encoder": ft_mode == "Frozen",
            "unfreeze_last_n_blocks": int(unfreeze_n) if ft_mode == "Partial (last N)" else 0,
            "local_path": "weights/ecapa",
        }
        old_enc.pop("freeze_feature_extractor", None)  # stale key for ECAPA
    elif encoder_type == "wavlm":
        new_enc = {
            "base_model": "microsoft/wavlm-large",
            "freeze_feature_extractor": ft_mode == "Frozen",
            "local_path": "weights/wavlm_large",
        }
        old_enc.pop("freeze_encoder", None)
        old_enc.pop("unfreeze_last_n_blocks", None)
    elif encoder_type == "campp":
        new_enc = {
            "model_id": "iic/speech_campplus_sv_en_voxceleb_16k",
            "revision": "v1.0.2",
            "freeze_encoder": ft_mode == "Frozen",
            "local_path": "weights/campp",
        }
    elif encoder_type == "eres2net":
        new_enc = {
            "ckpt_name": "eres2netv2.ckpt",
            "freeze_encoder": ft_mode == "Frozen",
            "local_path": "weights/eres2net",
        }
    else:  # titanet
        new_enc = {
            "model_id": "nvidia/speakerverification_en_titanet_large",
            "freeze_encoder": ft_mode == "Frozen",
            "local_path": "weights/titanet/titanet_large.nemo",
        }
    return new_enc


def _head_type(config: dict) -> str:
    """Active speaker-head type (arcface | arcface_subcenter | linear)."""
    return config["model"].get("speaker_head_type", "arcface")


def _head_cfg(config: dict) -> dict:
    """Config block of the active speaker head."""
    ht = _head_type(config)
    return (config["model"].get("speaker_head_config", {}).get(ht, {}) or {})


# ═══════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📋 Active Config")
    mc = config.get("model", {})
    enc = mc.get("encoder_type", "?")
    pool = (mc.get("encoder_config", {}).get(enc, {})
            .get("pooling_type") or mc.get("pooling_type", "?"))
    freeze = _enc_freeze()
    blocks = _enc_unfreeze_blocks()
    dur = config["audio"]["duration_seconds"]
    nwin = config["audio"].get("num_train_windows", "-")
    ehop = config["audio"].get("eval_hop_ratio", "-")
    mwin = config["audio"].get("max_eval_windows", "-")
    oodr = config["audio"].get("ood_batch_ratio", "-")
    enc_lr = config["training"].get("encoder_lr", "-")
    arc = _head_cfg(config)
    head_name = _head_type(config)
    head_label = (
        f"Sub-center ArcFace (k={arc.get('sub_centers', 3)}, "
        f"m={arc.get('margin', 0.3)}, s={arc.get('scale', 32)})"
        if head_name == "arcface_subcenter"
        else f"ArcFace (m={arc.get('margin', 0.4)}, s={arc.get('scale', 30)})"
    )
    ft_label = ("Frozen" if freeze
                else (f"Partial (last {blocks})" if blocks and blocks > 0 else "Full"))
    st.markdown(f"""
    | Param | Value |
    |-------|-------|
    | Encoder | `{enc}` |
    | Pooling | `{pool}` |
    | Head | {head_label} |
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
tab_cfg, tab_cloud, tab_local, tab_matrix, tab_analysis = st.tabs(
    ["⚙️ Config", "☁️ Cloud (Vast.ai)", "💻 Local", "🧬 Experiment Matrix",
     "🧪 Analysis"])

# ── TAB: Config ──
with tab_cfg:
    st.header("🧬 Experiment Profiles")
    st.caption("Profiles inherit from `configs/default_config.yaml` and store only "
               "their overrides (diffable, mergeable). Load one to continue editing; "
               "save the current setup as a named experiment below.")
    profiles = list_profiles()
    prof_c1, prof_c2 = st.columns([3, 1])
    with prof_c1:
        profile_sel = st.selectbox(
            "Load existing", ["(base)"] + profiles, index=0, key="profile_select",
            help="(base) edits configs/default_config.yaml directly.",
        )
    with prof_c2:
        if st.button("📂 Load", key="load_profile_btn", use_container_width=True):
            if profile_sel == "(base)":
                st.session_state.edit_config = None
                st.session_state.loaded_profile_name = "(base)"
            else:
                st.session_state.edit_config = load_profile(profile_sel)
                st.session_state.loaded_profile_name = profile_sel
            st.rerun()
    st.caption(f"Currently editing: **{st.session_state.loaded_profile_name}**")

    st.header("⚙️ Model Setup")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧠 Encoder")
        _ENC_OPTS = ["wavlm", "ecapa", "campp", "eres2net", "titanet"]
        encoder_type = st.selectbox(
            "Encoder", _ENC_OPTS,
            index=_ENC_OPTS.index(mc.get("encoder_type", "wavlm")))

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

        # Pooling is stored per-encoder (model_factory prefers it over the
        # global default). WavLM emits frame-level features → statistical
        # (mean+std); ECAPA/CAM++/ERes2NetV2/TitaNet produce utterance-level
        # vectors already → identity.
        _pool_defaults = {"wavlm": "statistical", "ecapa": "identity",
                          "campp": "identity", "eres2net": "identity",
                          "titanet": "identity"}
        pool_opts = ["identity", "statistical", "attentive"]
        cur_pool = (config["model"].get("encoder_config", {})
                    .get(encoder_type, {}).get("pooling_type"))
        cur_pool = cur_pool or config["model"].get(
            "pooling_type", _pool_defaults.get(encoder_type, "identity"))
        pool_idx = pool_opts.index(cur_pool) if cur_pool in pool_opts else 0
        if encoder_type == "ecapa":
            st.info("💡 ECAPA has built-in ASP → pooling = identity")
        pooling_type = st.selectbox("Pooling", pool_opts, index=pool_idx)

        hub_download = st.checkbox(
            "Allow hub downloads (dev only)",
            value=bool(mc.get("allow_hub_download", False)),
            help="MUST stay OFF at submission time (offline). ON only on the "
                 "dev machine / fresh Vast instance to fetch weights once.",
        )

        st.subheader("🎯 Speaker Head")
        head_type = st.selectbox(
            "Head type", ["arcface", "arcface_subcenter"],
            index=0 if _head_type(config) == "arcface" else 1,
            help="Sub-center ArcFace keeps K centres per speaker (max-cosine score) "
                 "and is more robust for few-shot classes (4-5 files/speaker).",
        )
        arc_cfg = _head_cfg(config)
        arc_m = st.slider("Margin", 0.1, 0.5, float(arc_cfg.get("margin", 0.3)), 0.05)
        arc_s = st.slider("Scale", 5.0, 64.0, float(arc_cfg.get("scale", 32.0)), 1.0)
        arc_emb = st.selectbox("Embedding dim", [128, 192, 256],
                               index=[128,192,256].index(arc_cfg.get("embedding_dim",192)))
        sub_centers = 3
        if head_type == "arcface_subcenter":
            sub_centers = st.number_input("Sub-centers (K)", 1, 8,
                                          int(arc_cfg.get("sub_centers", 3)))
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
        st.subheader("🎲 Split")
        split_cfg = config["data"].get("split", {}) or {}
        split_scheme = st.selectbox(
            "Split scheme", ["single", "kfold"],
            index=0 if str(split_cfg.get("scheme", "single")) == "single" else 1,
            help="kfold = speaker-aware K-fold for out-of-fold (OOF) tuning. "
                 "Each fold runs separately (set the fold index).",
        )
        n_folds = int(split_cfg.get("folds", 3))
        fold_idx = int(split_cfg.get("fold", 0))
        split_seed = int(split_cfg.get("seed", 42))
        if split_scheme == "kfold":
            n_folds = st.number_input("Folds (K)", 2, 10, int(split_cfg.get("folds", 3)))
            fold_idx = st.number_input("Fold index", 0, n_folds - 1, int(split_cfg.get("fold", 0)))
        split_seed = st.number_input("Split seed", 0, 2**31 - 1, split_seed,
                                     help="RNG seed for the val/fold partition (matrix seeds vary this).")
        st.subheader("🎛 Augmentation")
        wf = (config.get("augmentation", {}).get("waveform", {}) or {})
        dom = (config.get("augmentation", {}).get("domain", {}) or {})
        spec = (config.get("augmentation", {}).get("spec", {}) or {})

        def _on(block, key, default=0.0):
            return float((block.get(key) or {}).get("p", default) or 0) > 0

        with st.expander("🌊 Waveform (gentle)", expanded=False):
            wf_gn_on = st.checkbox("Gaussian noise", value=_on(wf, "gaussian_noise", 0.4))
            if wf_gn_on:
                c1, c2 = st.columns(2)
                wf_gn_p = c1.slider("Gaussian p", 0.0, 1.0, float((wf.get("gaussian_noise") or {}).get("p", 0.4)), 0.05)
                amp = (wf.get("gaussian_noise") or {}).get("amp", [0.001, 0.012])
                wf_gn_min = c2.number_input("amp min", 0.0, 0.1, float(amp[0]), 0.001)
                wf_gn_max = c2.number_input("amp max", 0.0, 0.1, float(amp[1]), 0.001)

            wf_gain_on = st.checkbox("Gain", value=_on(wf, "gain", 0.3))
            if wf_gain_on:
                c1, c2 = st.columns(2)
                wf_gain_p = c1.slider("Gain p", 0.0, 1.0, float((wf.get("gain") or {}).get("p", 0.3)), 0.05)
                db = (wf.get("gain") or {}).get("db", [-6, 6])
                wf_gain_min = c2.number_input("dB min", -24.0, 0.0, float(db[0]), 1.0)
                wf_gain_max = c2.number_input("dB max", 0.0, 24.0, float(db[1]), 1.0)

            wf_pol_on = st.checkbox("Polarity inversion", value=_on(wf, "polarity_inversion", 0.5))
            if wf_pol_on:
                wf_pol_p = st.slider("Polarity p", 0.0, 1.0, float((wf.get("polarity_inversion") or {}).get("p", 0.5)), 0.05)

            wf_shift_on = st.checkbox("Shift", value=_on(wf, "shift", 0.3))
            if wf_shift_on:
                c1, c2 = st.columns(2)
                wf_shift_p = c1.slider("Shift p", 0.0, 1.0, float((wf.get("shift") or {}).get("p", 0.3)), 0.05)
                wf_shift_frac = c2.slider("Shift frac", 0.0, 0.5, float((wf.get("shift") or {}).get("frac", 0.1)), 0.01)

            wf_pitch_on = st.checkbox("Pitch shift", value=_on(wf, "pitch_shift", 0.25))
            if wf_pitch_on:
                c1, c2 = st.columns(2)
                wf_pitch_p = c1.slider("Pitch p", 0.0, 1.0, float((wf.get("pitch_shift") or {}).get("p", 0.25)), 0.05)
                semi = (wf.get("pitch_shift") or {}).get("semitones", [-1, 1])
                wf_pitch_min = c2.number_input("Pitch min (st)", -4.0, 0.0, float(semi[0]), 0.5)
                wf_pitch_max = c2.number_input("Pitch max (st)", 0.0, 4.0, float(semi[1]), 0.5)

            wf_str_on = st.checkbox("Time-stretch", value=_on(wf, "time_stretch", 0.2))
            if wf_str_on:
                c1, c2 = st.columns(2)
                wf_str_p = c1.slider("Time-stretch p", 0.0, 1.0, float((wf.get("time_stretch") or {}).get("p", 0.2)), 0.05)
                rate = (wf.get("time_stretch") or {}).get("rate", [0.85, 1.18])
                wf_rate_min = c2.number_input("Rate min", 0.5, 1.0, float(rate[0]), 0.01)
                wf_rate_max = c2.number_input("Rate max", 1.0, 2.0, float(rate[1]), 0.01)

        with st.expander("🏠 Domain (RIR / MUSAN / codec)", expanded=True):
            dom_rir_on = st.checkbox("RIR reverb", value=_on(dom, "rirs_reverb", 0.0))
            if dom_rir_on:
                dom_rir_p = st.slider("RIR p", 0.0, 1.0, float((dom.get("rirs_reverb") or {}).get("p", 0.4)), 0.05)

            dom_musan_noise_on = st.checkbox("MUSAN noise", value=float((dom.get("musan") or {}).get("noise_p", 0.0) or 0) > 0)
            dom_musan_music_on = st.checkbox("MUSAN music", value=float((dom.get("musan") or {}).get("music_p", 0.0) or 0) > 0)
            if dom_musan_noise_on or dom_musan_music_on:
                c1, c2 = st.columns(2)
                if dom_musan_noise_on:
                    dom_musan_noise_p = c1.slider("MUSAN noise p", 0.0, 1.0, float((dom.get("musan") or {}).get("noise_p", 0.4)), 0.05)
                if dom_musan_music_on:
                    dom_musan_music_p = c2.slider("MUSAN music p", 0.0, 1.0, float((dom.get("musan") or {}).get("music_p", 0.2)), 0.05)
                snr = (dom.get("musan") or {}).get("snr_db", [5, 20])
                c1, c2 = st.columns(2)
                dom_snr_min = c1.number_input("MUSAN SNR min", -10.0, 30.0, float(snr[0]), 1.0)
                dom_snr_max = c2.number_input("MUSAN SNR max", -10.0, 30.0, float(snr[1]), 1.0)

            dom_mp3_on = st.checkbox("mp3 codec roundtrip", value=_on(dom, "mp3_codec_roundtrip", 0.0))
            if dom_mp3_on:
                c1, c2 = st.columns(2)
                dom_mp3_p = c1.slider("mp3 p", 0.0, 1.0, float((dom.get("mp3_codec_roundtrip") or {}).get("p", 0.3)), 0.05)
                dom_mp3_min = c2.number_input("mp3 min bitrate", 32, 320, int((dom.get("mp3_codec_roundtrip") or {}).get("min_bitrate", 64)), 8)
                dom_mp3_max = c2.number_input("mp3 max bitrate", 32, 320, int((dom.get("mp3_codec_roundtrip") or {}).get("max_bitrate", 192)), 8)

        with st.expander("🔲 Spec masking", expanded=False):
            spec_tm_on = st.checkbox("Time mask", value=_on(spec, "time_mask", 0.0))
            if spec_tm_on:
                c1, c2 = st.columns(2)
                spec_tm_p = c1.slider("Time mask p", 0.0, 1.0, float((spec.get("time_mask") or {}).get("p", 0.5)), 0.05)
                spec_tm_ratio = c2.slider("Time mask max ratio", 0.0, 0.5, float((spec.get("time_mask") or {}).get("max_mask_ratio", 0.2)), 0.01)

        st.subheader("🏋️ Training")
        epochs = st.number_input("Epochs", 1, None, config["training"]["epochs"],
                                 help="No upper cap — the config default is 150.")
        lr_val = st.number_input("LR (heads)", 1e-6, 1e-2, config["training"]["learning_rate"], format="%.6f")
        encoder_lr = st.number_input("LR (encoder)", 1e-7, 1e-2,
                                     float(config["training"].get("encoder_lr", 1e-5)), format="%.6f",
                                     help="LR for unfrozen encoder blocks (fine-tune).")
        wd = st.number_input("Weight Decay", 0.0, 1e-2, config["training"]["weight_decay"], format="%.6f")
        grad_norm = st.number_input("Max Grad Norm", 0.1, 50.0, config["training"]["max_grad_norm"])
        patience = st.number_input("Early stop patience", 1, 50,
                                   int(config["training"].get("early_stopping_patience", 10)),
                                   help="Early stopping / checkpoint selection on val Macro-F1.")
        st.subheader("📈 Schedule & Precision")
        c1s, c2s = st.columns(2)
        with c1s:
            schedule_type = st.selectbox(
                "Schedule", ["cosine", "cosine_warm_restarts"],
                index=0 if config["training"].get("schedule", "cosine") == "cosine" else 1)
            warmup_ratio = st.slider("Warmup ratio", 0.0, 0.4,
                                     float(config["training"].get("warmup_ratio", 0.1)), 0.01)
        with c2s:
            amp_dtype = st.selectbox(
                "AMP dtype", ["fp16", "bf16"],
                index=0 if config["training"].get("amp_dtype", "fp16") == "fp16" else 1,
                help="bf16 is numerically stable for WavLM-Large (fp16 NaN'd in Phase 6).")
            ema_on = st.checkbox("EMA", value=bool(config["training"].get("ema_enabled", False)),
                                 help="Exponential moving average of weights (saved into best ckpt).")
        ema_decay = 0.999
        if ema_on:
            ema_decay = st.number_input("EMA decay", 0.9, 0.9999,
                                        float(config["training"].get("ema_decay", 0.999)), format="%.4f")
        st.subheader("🎯 Loss")
        loss_spk = (config["training"].get("loss", {}).get("speaker", {}) or {})
        loss_ood = (config["training"].get("loss", {}).get("ood", {}) or {})
        cur_loss_type = str(loss_spk.get("type", "focal"))
        loss_type = st.selectbox(
            "Speaker loss", ["ce", "focal"],
            index=0 if cur_loss_type == "ce" else 1,
            help="CE + label smoothing (metric-aligned, A10) vs Focal (legacy).",
        )
        focal_gamma = 2.0
        if loss_type == "focal":
            focal_gamma = st.number_input("Focal γ", 0.0, 5.0,
                                          float(loss_spk.get("focal_gamma", 2.0)), 0.5)
        ood_hidden = st.number_input("OOD head hidden dim", 0, 1024,
                                     mc.get("ood_head_config",{}).get("hidden_dim",256), 64)
        ood_pos_w = st.number_input("OOD pos_weight", 0.1, 10.0,
                                    float(loss_ood.get("pos_weight", config["training"].get("ood_pos_weight", 1.0))), 0.1)
        ood_w = st.number_input("OOD loss weight", 0.0, 1.0,
                                float(loss_ood.get("weight", config["training"].get("ood_loss_weight", 0.3))), 0.05)
        spk_w = st.number_input("Speaker loss weight", 0.0, 1.0,
                                float(loss_spk.get("weight", config["training"].get("speaker_loss_weight", 0.7))), 0.05)
        sm_val = st.number_input("Label smoothing", 0.0, 0.5,
                                 float(loss_spk.get("label_smoothing", config["training"].get("label_smoothing", 0.1))), 0.05)

    st.subheader("💾 Save")
    exp_name = st.text_input(
        "Experiment name", value="", key="exp_name",
        placeholder="leave empty to write configs/default_config.yaml",
        help="Non-empty → saved to configs/experiments/<name>.yaml as a named profile.",
    )
    if st.button("💾 Save", type="primary", use_container_width=True):
        # ── Encoder config: MERGE with existing keys so partial fine-tune
        #    settings (e.g. unfreeze_last_n_blocks) are never silently dropped.
        old_enc = dict(config["model"].get("encoder_config", {}).get(encoder_type, {}))
        new_enc = _encoder_save_config(encoder_type, old_enc, ft_mode, unfreeze_n)
        config["model"]["encoder_type"] = encoder_type
        config["model"].setdefault("encoder_config", {})[encoder_type] = {**old_enc, **new_enc}
        config["model"]["pooling_type"] = pooling_type
        # Pooling is stored per-encoder so the ensemble can mix statistical
        # (WavLM) with identity-pooled encoders without a global conflict.
        config["model"]["encoder_config"][encoder_type]["pooling_type"] = pooling_type
        config["model"]["allow_hub_download"] = hub_download
        config["model"]["speaker_head_type"] = head_type
        head_block = {"embedding_dim": arc_emb, "margin": arc_m, "scale": arc_s}
        if head_type == "arcface_subcenter":
            head_block["sub_centers"] = int(sub_centers)
        config["model"].setdefault("speaker_head_config", {})[head_type] = head_block
        config["model"]["ood_head_config"]["hidden_dim"] = ood_hidden
        config["audio"]["duration_seconds"] = audio_dur
        config["audio"]["min_valid_duration"] = min_dur
        config["audio"]["num_train_windows"] = int(num_win)
        config["audio"]["eval_hop_ratio"] = float(hop_ratio)
        config["audio"]["max_eval_windows"] = int(max_win)
        config["audio"]["ood_batch_ratio"] = float(ood_ratio)
        # Split (single vs speaker-aware K-fold)
        config["data"].setdefault("split", {})
        config["data"]["split"]["scheme"] = split_scheme
        config["data"]["split"]["folds"] = int(n_folds)
        config["data"]["split"]["fold"] = int(fold_idx)
        config["data"]["split"]["seed"] = int(split_seed)
        # Augmentation — on/off toggles + settings (ranges are preserved from
        # the previous config when an effect is turned off).
        aug = config.setdefault("augmentation", {})
        wf_cfg = aug.setdefault("waveform", {})

        gn = dict(wf_cfg.get("gaussian_noise") or {})
        gn["p"] = wf_gn_p if wf_gn_on else 0.0
        if wf_gn_on:
            gn["amp"] = [wf_gn_min, wf_gn_max]
        wf_cfg["gaussian_noise"] = gn

        gain = dict(wf_cfg.get("gain") or {})
        gain["p"] = wf_gain_p if wf_gain_on else 0.0
        if wf_gain_on:
            gain["db"] = [wf_gain_min, wf_gain_max]
        wf_cfg["gain"] = gain

        pol = dict(wf_cfg.get("polarity_inversion") or {})
        pol["p"] = wf_pol_p if wf_pol_on else 0.0
        wf_cfg["polarity_inversion"] = pol

        shift = dict(wf_cfg.get("shift") or {})
        shift["p"] = wf_shift_p if wf_shift_on else 0.0
        if wf_shift_on:
            shift["frac"] = wf_shift_frac
        wf_cfg["shift"] = shift

        pitch = dict(wf_cfg.get("pitch_shift") or {})
        pitch["p"] = wf_pitch_p if wf_pitch_on else 0.0
        if wf_pitch_on:
            pitch["semitones"] = [wf_pitch_min, wf_pitch_max]
        wf_cfg["pitch_shift"] = pitch

        ts = dict(wf_cfg.get("time_stretch") or {})
        ts["p"] = wf_str_p if wf_str_on else 0.0
        if wf_str_on:
            ts["rate"] = [wf_rate_min, wf_rate_max]
        wf_cfg["time_stretch"] = ts

        dom_cfg = aug.setdefault("domain", {})
        rir = dict(dom_cfg.get("rirs_reverb") or {})
        rir["p"] = dom_rir_p if dom_rir_on else 0.0
        rir.setdefault("path", "data/augmentation/rirs")
        dom_cfg["rirs_reverb"] = rir

        musan = dict(dom_cfg.get("musan") or {})
        musan["noise_p"] = dom_musan_noise_p if dom_musan_noise_on else 0.0
        musan["music_p"] = dom_musan_music_p if dom_musan_music_on else 0.0
        if dom_musan_noise_on or dom_musan_music_on:
            musan["snr_db"] = [dom_snr_min, dom_snr_max]
        musan.setdefault("path", "data/augmentation/musan")
        dom_cfg["musan"] = musan

        mp3 = dict(dom_cfg.get("mp3_codec_roundtrip") or {})
        mp3["p"] = dom_mp3_p if dom_mp3_on else 0.0
        if dom_mp3_on:
            mp3["min_bitrate"] = dom_mp3_min
            mp3["max_bitrate"] = dom_mp3_max
        dom_cfg["mp3_codec_roundtrip"] = mp3

        spec_cfg = aug.setdefault("spec", {})
        tm = dict(spec_cfg.get("time_mask") or {})
        tm["p"] = spec_tm_p if spec_tm_on else 0.0
        if spec_tm_on:
            tm["max_mask_ratio"] = spec_tm_ratio
        spec_cfg["time_mask"] = tm
        config["training"]["epochs"] = epochs
        config["training"]["learning_rate"] = lr_val
        config["training"]["encoder_lr"] = float(encoder_lr)
        config["training"]["weight_decay"] = wd
        config["training"]["max_grad_norm"] = grad_norm
        config["training"]["early_stopping_patience"] = int(patience)
        config["training"]["schedule"] = schedule_type
        config["training"]["warmup_ratio"] = float(warmup_ratio)
        config["training"]["amp_dtype"] = amp_dtype
        config["training"]["ema_enabled"] = bool(ema_on)
        config["training"]["ema_decay"] = float(ema_decay)
        # Loss (nested block + flat keys kept in sync for backward-compat readers)
        loss_block = config["training"].setdefault("loss", {})
        spk_block = loss_block.setdefault("speaker", {})
        ood_block = loss_block.setdefault("ood", {})
        spk_block["type"] = loss_type
        spk_block["focal_gamma"] = float(focal_gamma)
        spk_block["label_smoothing"] = float(sm_val)
        spk_block["weight"] = float(spk_w)
        ood_block["type"] = "bce"
        ood_block["pos_weight"] = float(ood_pos_w)
        ood_block["weight"] = float(ood_w)
        config["training"]["ood_pos_weight"] = float(ood_pos_w)
        config["training"]["ood_loss_weight"] = float(ood_w)
        config["training"]["speaker_loss_weight"] = float(spk_w)
        config["training"]["label_smoothing"] = float(sm_val)
        exp_name = (exp_name or "").strip()
        if exp_name:
            save_profile(exp_name, config)
            st.session_state.edit_config = config
            st.session_state.loaded_profile_name = exp_name
            st.success(f"✅ Saved experiment profile `{exp_name}`!")
        else:
            save_config(config)
            st.session_state.edit_config = None
            st.session_state.loaded_profile_name = "(base)"
            st.success("✅ Saved to base config!")
        config = load_config()
        st.rerun()

    st.divider()
    st.subheader("🧪 Hyperparameter Search (Optuna)")
    st.caption("Coarse phase (Audit §16.2): TPESampler over the key training knobs — "
               "each trial is a named experiment profile + a short training run. The "
               "best trial is saved back as a `«study»-best` profile.")
    with st.expander("🔎 Search space (what Optuna actually tunes)", expanded=False):
        from src.hpo import HPO_SPACE
        space_rows = [{
            "Variable": e["name"],
            "Type": e["kind"],
            "Range": f"{e['low']} … {e['high']}" + ("  (log)" if e["log"] else ""),
            "Meaning": e["description"],
        } for e in HPO_SPACE]
        st.dataframe(space_rows, use_container_width=True)
        st.caption("Sampler: TPESampler (Bayesian) · objective: val Macro-F1 · "
                   "coarse phase = few epochs, one fold per trial.")
    if "hpo_runner" not in st.session_state:
        st.session_state.hpo_runner = None
    hpr = st.session_state.hpo_runner
    hpo_running = hpr is not None and hpr.running and not hpr.finished
    hc1, hc2, hc3, hc4 = st.columns([2, 2, 2, 1])
    with hc1:
        hpo_trials = st.number_input("Trials", 1, 200, 30, key="hpo_trials")
    with hc2:
        hpo_epochs = st.number_input("Epochs/trial", 1, 200, 30, key="hpo_epochs")
    with hc3:
        hpo_study = st.text_input("Study name", value="speaker-hpo", key="hpo_study")
    with hc4:
        st.caption("")
        if st.button("🚀 Launch HPO", type="primary", use_container_width=True,
                     key="hpo_launch", disabled=hpo_running):
            cmd = [sys.executable, "-m", "src.hpo",
                   "--trials", str(int(hpo_trials)),
                   "--epochs", str(int(hpo_epochs)),
                   "--study", (hpo_study or "speaker-hpo").strip()]
            hpr = LocalRunner(cmd, str(PROJECT_ROOT))
            hpr.start()
            st.session_state.hpo_runner = hpr
            st.rerun()
    _render_local_runner("hpo_runner", "Optuna HPO")


# ── TAB: Cloud ──
with tab_cloud:
    st.header("☁️ Vast.ai")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gpu = st.selectbox("GPU", GPU_OPTIONS, key="cgpu")
    with c2:
        stage = st.selectbox("Stage", ["all","data","train","eval"],
                             format_func=lambda x: {"all":"🚀 Full","data":"📊 Data","train":"🏋️ Train","eval":"📈 Eval"}[x],
                             key="cstage")
    with c3:
        disk_gb = st.slider("💾 Disk (GB)", 20, 200,
                            config.get("mlops",{}).get("vast",{}).get("disk_size", 60), 10,
                            help="More disk for bigger models/datasets.")
    with c4:
        min_cuda = st.number_input(
            "Min CUDA", 10.0, 14.0,
            float(config.get("mlops", {}).get("vast", {}).get("min_cuda_version", 13)),
            0.1,
            help="Filter rental offers to hosts whose driver supports at least this "
                 "CUDA version. uv.lock pins a CUDA 13.x torch build, so keep it ≥ 13.",
        )

    run_mode = st.radio(
        "Run mode", ["Single (committed config)", "Experiment queue (profiles)",
                     "HPO study (Optuna)"],
        index=0, horizontal=True, key="crunmode",
        help="Single = one run with the committed config + encoder/freeze overrides. "
             "Queue = run a list of named experiment profiles sequentially on ONE "
             "instance. HPO = Optuna hyperparameter search on a committed recipe.",
    )
    queue_profiles = []
    hpo_study = "speaker-hpo"
    hpo_trials = 30
    hpo_epochs = 30
    hpo_base_profile = ""
    if run_mode.startswith("Experiment"):
        queue_profiles = st.multiselect(
            "Profiles to run", list_profiles(), key="cqueue",
            help="configs/experiments/<name>.yaml — must be committed + pushed so the "
                 "instance's `git clone` sees them.",
        )
    elif run_mode.startswith("HPO"):
        hpo_base_profile = st.selectbox(
            "Recipe to tune", ["(base config)"] + list_profiles(), key="c_hpo_base",
            help="The named recipe (or base config) whose continuous hyperparameters "
                 "Optuna will tune. Must be committed + pushed.",
        )
        hc1, hc2, hc3 = st.columns(3)
        hpo_study = hc1.text_input("Study name", value="speaker-hpo", key="c_hpo_study")
        hpo_trials = hc2.number_input("Trials", 1, 200, 30, key="c_hpo_trials")
        hpo_epochs = hc3.number_input("Epochs/trial", 1, 200, 30, key="c_hpo_epochs")

    if not (PROJECT_ROOT / ".env").exists():
        st.warning("⚠️ `.env` missing — copy from `.env.example`")

    if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True,
                 key="launch"):
        if not (PROJECT_ROOT / ".env").exists():
            st.error("❌ .env missing!"); st.stop()
        os.environ["GPU_TARGET"] = gpu
        os.environ["TARGET_PIPELINE"] = stage
        # Encoder selection + fine-tune choice (setup_vast.sh reads these)
        os.environ["ENCODER_TYPE"] = encoder_type
        os.environ["ALLOW_HUB_DOWNLOAD"] = str(hub_download).lower()
        os.environ["FREEZE_ENCODER"] = str(ft_mode == "Frozen").lower()
        os.environ["UNFREEZE_LAST_N_BLOCKS"] = str(
            unfreeze_n if ft_mode == "Partial (last N)" else 0)
        os.environ["FREEZE_FEATURE_EXTRACTOR"] = str(ft_mode == "Frozen").lower()  # compat
        os.environ["DISK_SIZE_GB"] = str(disk_gb)
        os.environ["MIN_CUDA_VERSION"] = str(min_cuda)
        os.environ["EXPERIMENT_PROFILES"] = (
            " ".join(queue_profiles) if queue_profiles else "")
        if run_mode.startswith("HPO"):
            os.environ["HPO_STUDY"] = hpo_study.strip() or "speaker-hpo"
            os.environ["HPO_TRIALS"] = str(int(hpo_trials))
            os.environ["HPO_EPOCHS"] = str(int(hpo_epochs))
            os.environ["HPO_BASE_PROFILE"] = (
                "" if hpo_base_profile == "(base config)" else hpo_base_profile)
        else:
            os.environ["HPO_STUDY"] = ""
            os.environ["HPO_TRIALS"] = ""
            os.environ["HPO_EPOCHS"] = ""
            os.environ["HPO_BASE_PROFILE"] = ""
        with st.spinner("Creating instance..."):
            try:
                env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
                r = subprocess.run([sys.executable, str(DEPLOY_SCRIPT)],
                                   capture_output=True, text=True, encoding="utf-8",
                                   env=env, timeout=180, cwd=str(PROJECT_ROOT), check=True)
                output = r.stdout
                # Extract instance ID from the deploy output
                m = re.search(r"INSTANCE_ID=(\d+)", output)
                if m:
                    st.success(f"✅ Instance #{m.group(1)} launched!")
                else:
                    st.warning("⚠️ Launched but could not parse instance ID.")
                with st.expander("Deploy output", expanded=True):
                    st.code(output)
            except subprocess.CalledProcessError as e:
                st.error("❌ Deploy failed!")
                st.code(e.stdout or "")


# ── TAB: Local ──
with tab_local:
    st.header("💻 Local Run")
    st.caption(f"`{encoder_type}` + `{pooling_type}` + ArcFace | {audio_dur}s | {epochs}ep | LR={lr_val}")
    import torch
    if torch.cuda.is_available():
        st.success(f"✅ {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        st.warning("⚠️ CPU only — slow.")

    if "local_runner" not in st.session_state:
        st.session_state.local_runner = None
    runner = st.session_state.local_runner
    is_running = runner is not None and runner.running and not runner.finished

    lc1, lc2 = st.columns(2)
    with lc1:
        ls = st.selectbox("Stage", ["all","data","train","eval"],
                          format_func=lambda x: {"all":"🚀 Full","data":"📊 Data","train":"🏋️ Train","eval":"📈 Eval"}[x],
                          key="lstage")
    with lc2:
        mlflow_on = st.checkbox("📈 MLflow", value=True, key="lmlflow")

    if st.button("▶️ Run", type="primary", use_container_width=True, key="lrun",
                 disabled=is_running):
        # Run the LOADED experiment profile when one is active, else the base config.
        if (st.session_state.edit_config is not None
                and st.session_state.loaded_profile_name != "(base)"):
            cmd = [sys.executable, str(PIPELINE_SCRIPT), "--run", ls,
                   "--experiment", st.session_state.loaded_profile_name]
        else:
            cmd = [sys.executable, str(PIPELINE_SCRIPT), "--run", ls,
                   "--config", str(CONFIG_PATH)]
        if not mlflow_on: cmd.append("--no-mlflow")
        runner = LocalRunner(cmd, str(PROJECT_ROOT))
        runner.start()
        st.session_state.local_runner = runner
        st.rerun()

    _render_local_runner("local_runner", "Local pipeline")


# ── TAB: Experiment Matrix ──
with tab_matrix:
    st.header("🧬 Experiment Matrix")
    st.caption("encoders × recipes × seeds × folds → named profiles → sequential queue. "
               "Profiles land in `configs/experiments/`; the queue runs them one by one.")

    from src.experiment_matrix import (
        ENCODERS, RECIPES, DEFAULT_SEEDS, expand_matrix, write_matrix_profiles,
    )
    from src.experiment_queue import clear_state, load_state

    def _parse_ints(text: str):
        if not text.strip():
            return []
        try:
            return [int(x) for x in text.replace(",", " ").split() if x.strip()]
        except ValueError:
            return None

    st.subheader("1️⃣ Generate profiles")
    mc1, mc2 = st.columns(2)
    with mc1:
        enc_sel = st.multiselect("Encoders", ENCODERS, default=["ecapa", "campp"],
                                 key="m_enc")
        recipe_sel = st.multiselect("Recipes", list(RECIPES), default=["full"],
                                    key="m_recipe")
        scheme = st.selectbox("Split scheme", ["single", "kfold"], index=0,
                              key="m_scheme")
    with mc2:
        seeds_text = st.text_input("Seeds (comma-separated)",
                                   value=", ".join(map(str, DEFAULT_SEEDS)),
                                   key="m_seeds")
        folds_text = st.text_input("Folds (comma-separated; empty = no folds)",
                                   value="", key="m_folds",
                                   help="Only used when scheme=kfold, e.g. 0,1,2.")

    seeds = _parse_ints(seeds_text)
    folds = _parse_ints(folds_text)
    parse_ok = seeds is not None and folds is not None
    if not parse_ok:
        st.error("Seeds / folds must be comma-separated integers.")

    if st.button("⚙️ Generate & Save profiles", type="primary",
                 use_container_width=True, key="m_gen", disabled=not parse_ok):
        if not enc_sel or not recipe_sel or not seeds:
            st.warning("Select at least one encoder, one recipe, and one seed.")
        else:
            base = load_config()
            cells = expand_matrix(enc_sel, recipe_sel, seeds,
                                  folds=folds or None, scheme=scheme, base=base)
            names = write_matrix_profiles(cells, base=base)
            st.success(f"Generated {len(names)} profile(s).")
            st.code("\n".join(names))

    st.subheader("2️⃣ Enqueue & run")
    all_profiles = list_profiles()
    q_col1, q_col2, q_col3 = st.columns([3, 1, 1])
    with q_col1:
        queue_sel = st.multiselect("Profiles to run (in order)", all_profiles,
                                   key="m_queue")
    with q_col2:
        q_stage = st.selectbox("Stage", ["all", "data", "train", "eval"],
                               index=0, key="m_qstage")
    with q_col3:
        st.caption("")
        if st.button("🧹 Clear state", key="m_clear", use_container_width=True):
            clear_state()
            st.rerun()

    if "matrix_runner" not in st.session_state:
        st.session_state.matrix_runner = None
    mr = st.session_state.matrix_runner
    mr_running = mr is not None and mr.running and not mr.finished

    if st.button("▶️ Run queue", type="primary", use_container_width=True,
                 key="m_run", disabled=mr_running):
        if not queue_sel:
            st.warning("Select at least one profile to enqueue.")
        else:
            cmd = [sys.executable, "-m", "src.experiment_queue",
                   "--profiles"] + queue_sel + ["--run", q_stage]
            mr = LocalRunner(cmd, str(PROJECT_ROOT))
            mr.start()
            st.session_state.matrix_runner = mr
            st.rerun()

    _render_local_runner("matrix_runner", "Experiment queue")

    # Current queue state
    with st.expander("📋 Queue state", expanded=False):
        st.json(load_state())


# ── TAB: Analysis ──
with tab_analysis:
    st.header("🧪 Analysis & Tools")
    st.caption("Run the new pipeline modules locally: unbiased EDA, centroid baseline, "
               "ensemble calibration, and submission inference.")

    st.divider()
    st.subheader("📊 Run Comparison (MLflow)")
    st.caption("Rank experiments by their logged Macro-F1, then promote the best "
               "into a leaderboard submission.")
    if st.button("🔄 Load MLflow runs", key="a_runs"):
        from src.run_registry import list_runs
        runs = list_runs(max_results=50)
        if not runs:
            st.warning("No runs found (or MLflow is unreachable). "
                       "Check the DagsHub/MLflow tracking URI in `.env` / config.")
        else:
            rows = []
            for r in runs:
                score = f"{r['score']:.4f}" if r["score"] is not None else "—"
                rows.append({
                    "Score": score,
                    "Metric": r["score_metric"] or "—",
                    "Run": r["run_name"],
                    "Encoder": r["encoder"] or "—",
                    "Status": r["status"],
                })
            st.dataframe(rows, use_container_width=True)

    st.divider()
    st.subheader("🚀 Promote to Submission")
    st.caption("Rebuild the submission package + record a row in `reports/lb_log.md`.")
    if "promote_runner" not in st.session_state:
        st.session_state.promote_runner = None
    pr = st.session_state.promote_runner
    pr_running = pr is not None and pr.running and not pr.finished
    p_col1, p_col2 = st.columns([2, 2])
    with p_col1:
        p_label = st.text_input("Config / run label", value="", key="a_label")
    with p_col2:
        p_note = st.text_input("Note", value="", key="a_note")
    p_verify = st.checkbox("Run verify_submission.py after build", value=False,
                           key="a_verify")
    if st.button("📦 Build + record (Promote)", type="primary",
                 use_container_width=True, key="a_promote", disabled=pr_running):
        cmd = [sys.executable, "scripts/promote_run.py",
               "--label", (p_label or "manual").strip()]
        if p_note.strip():
            cmd += ["--note", p_note.strip()]
        if p_verify:
            cmd.append("--verify")
        pr = LocalRunner(cmd, str(PROJECT_ROOT))
        pr.start()
        st.session_state.promote_runner = pr
        st.rerun()
    _render_local_runner("promote_runner", "Promote")

    st.divider()

    if "analysis_runner" not in st.session_state:
        st.session_state.analysis_runner = None
    ar = st.session_state.analysis_runner
    a_running = ar is not None and ar.running and not ar.finished

    ac1, ac2 = st.columns(2)
    with ac1:
        st.subheader("📊 Unbiased EDA")
        st.caption("Multi-window ECAPA embeddings + LOO centroid + Macro-F1 simulation "
                   "(GPU, several minutes).")
        if st.button("▶️ eda_embeddings", key="a_eda", use_container_width=True,
                     disabled=a_running):
            r = LocalRunner([sys.executable, "-m", "src.eda_embeddings"], str(PROJECT_ROOT))
            r.start()
            st.session_state.analysis_runner = r
            st.rerun()

        st.subheader("🎯 Centroid baseline")
        st.caption("Embedding cache (idempotent) + centroid classifier + fusion report.")
        force_cache = st.checkbox("Rebuild embedding cache", key="a_force")
        if st.button("▶️ centroid_baseline", key="a_centroid", use_container_width=True,
                     disabled=a_running):
            cmd = [sys.executable, "-m", "src.centroid_baseline"]
            if force_cache:
                cmd.append("--force-cache")
            r = LocalRunner(cmd, str(PROJECT_ROOT))
            r.start()
            st.session_state.analysis_runner = r
            st.rerun()

    with ac2:
        st.subheader("🧩 Ensemble + temperature")
        st.caption("Per-model & ensemble Macro-F1 + best softmax temperature "
                   "(needs ≥2 trained checkpoints).")
        ckpts = st.text_input("Checkpoint paths (space-separated)",
                              value="checkpoints/best_seed1.pt checkpoints/best_seed2.pt",
                              key="a_ckpts")
        if st.button("▶️ ensemble_calibrate", key="a_ens", use_container_width=True,
                     disabled=a_running):
            ckpt_list = [c for c in ckpts.split() if c]
            if len(ckpt_list) < 2:
                st.warning("⚠️ Provide at least 2 checkpoint paths.")
            else:
                cmd = [sys.executable, "-m", "src.ensemble_calibrate",
                       "--checkpoints"] + ckpt_list
                r = LocalRunner(cmd, str(PROJECT_ROOT))
                r.start()
                st.session_state.analysis_runner = r
                st.rerun()

        st.subheader("📤 Submission CSV")
        st.caption("Run submission.inference on a test folder → 447-column CSV.")
        s_dir = st.text_input("Test data dir", value="data/processed/audio_wav", key="a_dir")
        s_out = st.text_input("Output CSV", value="predictions.csv", key="a_out")
        if st.button("▶️ inference → CSV", key="a_inf", use_container_width=True,
                     disabled=a_running):
            cmd = [sys.executable, "-m", "submission.inference",
                   "--data-dir", s_dir, "--predictions-file-path", s_out]
            r = LocalRunner(cmd, str(PROJECT_ROOT))
            r.start()
            st.session_state.analysis_runner = r
            st.rerun()

    _render_local_runner("analysis_runner", "Analysis tool")
