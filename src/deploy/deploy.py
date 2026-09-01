"""
Vast.ai GPU Deployment Script for Speaker-Identification.

Rents a GPU on Vast.ai, injects the setup_vast_onstart.sh launcher (a tiny
script that clones the repo and runs the committed setup_vast.sh bootstrap —
kept under the API's 16 KB onstart limit), and runs the target pipeline
remotely.

Usage:
    # From project root, after setting up .env:
    python -m src.deploy.deploy

    # With overrides (useful when called from deploy_app.py):
    GPU_TARGET=RTX_3090 TARGET_PIPELINE=all python -m src.deploy.deploy
    # Supported GPU_TARGET values (see configs/default_config.yaml mlops.vast.gpu_options):
    #   RTX_3090  → vastai      profile (batch 32)
    #   RTX_3060  → vastai_3060 profile (batch 16)
    #   RTX_A4000 → vastai_a4000 profile (batch 24)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Fix Unicode encoding for Windows terminals (cp1252 → utf-8)
# Allows emoji and non-ASCII characters in print() statements
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass  # Not all Python versions/terminals support reconfigure

# Ensure the project root is importable even when this file is run directly as
# a script (deploy_app.py invokes it by absolute path, not via `python -m`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy import offer_selector


# ─────────────────────────────────────────────────────────
#  Config Loader
# ─────────────────────────────────────────────────────────

def load_environment() -> dict:
    """Load configuration from .env file and environment variables."""
    load_dotenv()

    config = {
        "VAST_API_KEY": os.getenv("VAST_API_KEY"),
        "DAGSHUB_TOKEN": os.getenv("DAGSHUB_USER_TOKEN"),
        "DAGSHUB_USERNAME": os.getenv("DAGSHUB_REPO_OWNER"),
        "DAGSHUB_REPO_NAME": os.getenv("DAGSHUB_REPO_NAME"),
        "DAGSHUB_TRACKING_URI": os.getenv("DAGSHUB_TRACKING_URI"),
        "GIT_REPO_URL": os.getenv("GIT_REPO_URL"),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", "main"),
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME", ""),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY", ""),
    }

    # Runtime overrides (set by deploy_app.py or CLI)
    config["GPU_TARGET"] = os.getenv("GPU_TARGET", "RTX_3090")
    config["TARGET_PIPELINE"] = os.getenv("TARGET_PIPELINE", "all")
    # Minimum host-driver CUDA version the rental must support. torch 2.13 in
    # uv.lock is a CUDA 13.x build, so a driver < 13.0 makes torch.cuda fail
    # (setup_vast.sh Phase 2.5 aborts). Filter offers by cuda_vers >= this.
    config["MIN_CUDA_VERSION"] = os.getenv("MIN_CUDA_VERSION", "13")
    # Space-separated named experiment profiles to run as a sequential queue on
    # the instance (Audit §17.2 — one instance, many runs). Empty = legacy
    # single run with the committed config + encoder/freeze env overrides.
    config["EXPERIMENT_PROFILES"] = os.getenv("EXPERIMENT_PROFILES", "")
    # Optuna HPO (Audit §17.4) — when HPO_STUDY is set, setup_vast.sh runs a
    # hyperparameter search on the instance instead of a queue/single run.
    config["HPO_STUDY"] = os.getenv("HPO_STUDY", "")
    config["HPO_TRIALS"] = os.getenv("HPO_TRIALS", "30")
    config["HPO_EPOCHS"] = os.getenv("HPO_EPOCHS", "30")
    config["HPO_BASE_PROFILE"] = os.getenv("HPO_BASE_PROFILE", "")
    # Encoder fine-tune choice (ECAPA-aware). FREEZE_FEATURE_EXTRACTOR kept for
    # backward compatibility with the old WavLM-only naming.
    config["FREEZE_ENCODER"] = os.getenv("FREEZE_ENCODER", "true")
    config["UNFREEZE_LAST_N_BLOCKS"] = os.getenv("UNFREEZE_LAST_N_BLOCKS", "0")
    config["FREEZE_FEATURE_EXTRACTOR"] = os.getenv("FREEZE_FEATURE_EXTRACTOR", "true")

    # Validate required variables. Optional keys may legitimately be empty:
    # KAGGLE_* (only needed for Kaggle data pulls) and the run-mode selectors
    # (EXPERIMENT_PROFILES / HPO_STUDY / HPO_BASE_PROFILE are empty when that
    # mode isn't selected — "Single" mode needs none of them).
    _OPTIONAL = frozenset({
        "KAGGLE_USERNAME", "KAGGLE_KEY",
        "EXPERIMENT_PROFILES", "HPO_STUDY", "HPO_BASE_PROFILE",
        "HPO_TRIALS", "HPO_EPOCHS",   # only used when HPO_STUDY is set
    })
    missing = [k for k, v in config.items() if not v and k not in _OPTIONAL]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Please check your .env file.")
        sys.exit(1)

    return config


# ─────────────────────────────────────────────────────────
#  Encoder Selection (config → Vast instance)
# ─────────────────────────────────────────────────────────

def read_model_selection() -> dict:
    """
    Read the active encoder selection from configs/default_config.yaml.

    Returns env-style keys for setup_vast.sh to apply on the instance:
        ENCODER_TYPE, ALLOW_HUB_DOWNLOAD, LOCAL_PATH_<ENC> (per encoder).

    An empty dict is returned if the config cannot be read — deployment then
    simply falls back to the committed config on the instance.
    """
    try:
        import yaml
        with open("configs/default_config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {}

    mc = cfg.get("model", {})
    result = {}
    enc_type = mc.get("encoder_type")
    if enc_type:
        result["ENCODER_TYPE"] = str(enc_type)
    result["ALLOW_HUB_DOWNLOAD"] = str(bool(mc.get("allow_hub_download", False))).lower()

    # Derive the freeze mode of the ACTIVE encoder from config. Without this,
    # a config-level Full-FT choice (Phase 2 default) would be silently reset
    # to Frozen when deploying directly via `python -m src.deploy.deploy`
    # (load_environment defaults FREEZE_ENCODER to "true"). Env overrides from
    # the UI still win over these config-derived values.
    enc_cfg = mc.get("encoder_config", {}).get(str(enc_type), {}) if enc_type else {}
    if enc_type == "ecapa":
        freeze = bool(enc_cfg.get("freeze_encoder", True))
        result["FREEZE_ENCODER"] = str(freeze).lower()
        result["UNFREEZE_LAST_N_BLOCKS"] = str(
            int(enc_cfg.get("unfreeze_last_n_blocks", 0)) if not freeze else 0)
    elif enc_type == "wavlm":
        freeze = bool(enc_cfg.get("freeze_encoder", False))
        result["FREEZE_ENCODER"] = str(freeze).lower()
        result["UNFREEZE_LAST_N_BLOCKS"] = "0"
    else:
        freeze = bool(enc_cfg.get("freeze_encoder", True))
        result["FREEZE_ENCODER"] = str(freeze).lower()
        result["UNFREEZE_LAST_N_BLOCKS"] = "0"

    for enc_name, enc_cfg2 in mc.get("encoder_config", {}).items():
        lp = enc_cfg2.get("local_path")
        if lp:
            result[f"LOCAL_PATH_{str(enc_name).upper()}"] = str(lp)
    return result


# ─────────────────────────────────────────────────────────
#  Shell Command Runner
# ─────────────────────────────────────────────────────────

def run_cmd(command: str, return_output: bool = False, silent_error: bool = False) -> str | None:
    """
    Run a shell command and handle errors.

    Args:
        command: Shell command string.
        return_output: If True, return stdout as string.
        silent_error: If True, don't exit on failure.
    """
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=300,
        )
        output_bytes = result.stdout or b""
        try:
            output = output_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            output = output_bytes.decode("utf-8", errors="replace").strip()

        if result.returncode != 0 and not silent_error:
            print(f"\n🛑 Command failed: {command}")
            print(f"--- Output ---\n{output}\n-------------")
            sys.exit(1)

        return output if return_output else None

    except subprocess.TimeoutExpired:
        print(f"\n🛑 Command timed out: {command}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Subprocess error: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────
#  Server Selection (filters → top-N → retry)
# ─────────────────────────────────────────────────────────

def _parse_bool(raw: str) -> bool:
    """Env-style boolean: true/1/yes/on → True, anything else → False."""
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_selector() -> dict:
    """
    Load the server-selection filters from environment variables.

    The Streamlit Cloud tab (deploy_app.py) sets these before launching; the
    defaults mirror configs/default_config.yaml → mlops.vast.selector. A set
    value always wins; unset/empty means "use the default" (0 / "" = no filter).
    """
    sel = dict(offer_selector.DEFAULTS)
    env_map = {
        "VAST_POOL_SIZE": ("pool_size", int),
        "VAST_RETRY_COUNT": ("retry_count", int),
        "VAST_MIN_RELIABILITY": ("min_reliability", float),
        "VAST_MIN_INET_DOWN_MBS": ("min_inet_down_mbps", float),
        "VAST_MAX_INET_DOWN_MBS": ("max_inet_down_mbps", float),
        "VAST_MIN_INET_UP_MBS": ("min_inet_up_mbps", float),
        "VAST_MAX_INET_UP_MBS": ("max_inet_up_mbps", float),
        "VAST_MIN_CPU_CORES": ("min_cpu_cores", float),
        "VAST_MAX_CPU_CORES": ("max_cpu_cores", float),
        "VAST_MIN_CPU_RAM_GB": ("min_cpu_ram_gb", float),
        "VAST_MAX_CPU_RAM_GB": ("max_cpu_ram_gb", float),
        "VAST_MIN_DISK_GB": ("min_disk_gb", float),
        "VAST_MAX_DISK_GB": ("max_disk_gb", float),
        "VAST_MIN_DURATION_DAYS": ("min_duration_days", float),
        "VAST_MAX_DURATION_DAYS": ("max_duration_days", float),
        "VAST_MIN_PRICE": ("min_price_per_hour", float),
        "VAST_MAX_PRICE": ("max_price_per_hour", float),
        "VAST_MIN_PCIE_BW": ("min_pcie_bw_gbps", float),
        "VAST_MAX_PCIE_BW": ("max_pcie_bw_gbps", float),
        "VAST_MIN_GPU_FRAC": ("min_gpu_frac", float),
        "VAST_BLOCKED_COUNTRIES": ("blocked_countries", str),
        "VAST_PICK_BEST": ("pick_best", _parse_bool),
    }
    for env_key, (key, cast) in env_map.items():
        raw = os.getenv(env_key)
        if raw is None or raw.strip() == "":
            continue
        try:
            sel[key] = cast(raw.strip())
        except ValueError:
            print(f"   ⚠ Ignoring invalid {env_key}={raw!r}")
    return sel


# ─────────────────────────────────────────────────────────
#  Main Deployment Flow
# ─────────────────────────────────────────────────────────

def main():
    # ── Check prerequisites ──
    setup_script = Path("setup_vast_onstart.sh")
    if not setup_script.exists():
        print(f"❌ setup_vast_onstart.sh not found in project root!")
        print("   Make sure you're running from the project root directory.")
        sys.exit(1)

    # ── Load config ──
    config = load_environment()
    print(f"🔍 Searching {config['GPU_TARGET']} offers "
          f"(CUDA ≥ {config['MIN_CUDA_VERSION']}, quality-ranked) "
          f"for pipeline: {config['TARGET_PIPELINE']}...")

    # ── Forward the active encoder selection to the instance ──
    # setup_vast.sh applies ENCODER_TYPE / ALLOW_HUB_DOWNLOAD / LOCAL_PATH_<ENC>
    # on the instance. Env overrides (deploy_app.py / CLI) win over the config
    # file values.
    for k, v in read_model_selection().items():
        config[k] = os.getenv(k, v)
    if config.get("ENCODER_TYPE"):
        print(f"   Encoder: {config['ENCODER_TYPE']} | "
              f"allow_hub_download: {config.get('ALLOW_HUB_DOWNLOAD')}")

    # ── Authenticate Vast.ai ──
    run_cmd(f"vastai set api-key {config['VAST_API_KEY']}", silent_error=True)

    # ── Search & rank offers ──
    # Hard numeric gates go into the query string (reliability, internet, CPU,
    # RAM, disk, duration, price); the CLI also keeps verified=true/rentable=true
    # by default. cuda_vers stays untouched — uv.lock's torch is a CUDA 13.x
    # build, so we still require the host driver to support it (setup_vast.sh
    # Phase 2.5 aborts otherwise). The fetched pool is then re-checked and
    # scored in Python, so an unsupported query field can never slip through.
    selector = load_selector()
    query = offer_selector.build_search_query(
        config["GPU_TARGET"], config["MIN_CUDA_VERSION"], selector)
    pool_size = int(selector["pool_size"])
    search_cmd = (
        f'vastai search offers "{query}" '
        f"-o dph --limit {pool_size} --raw"
    )
    raw_json = run_cmd(search_cmd, return_output=True)

    try:
        offers = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse Vast.ai offers: {e}")
        print(f"   Raw output: {raw_json[:300]}")
        sys.exit(1)

    # The vastai CLI returns exit code 0 AND a dict {"error": true, ...} when
    # the query itself is rejected (e.g. an invalid search key). Distinguish
    # that from a normal list of offers before ranking.
    if isinstance(offers, dict):
        print(f"❌ Vast.ai search error: {offers.get('msg', offers)}")
        print(f"   Query: {query}")
        sys.exit(1)
    if not isinstance(offers, list):
        print(f"❌ Unexpected search response: {str(offers)[:200]}")
        sys.exit(1)

    # pick_best on → quality-ranked (best candidate first); off → cheapest
    # offer that still passes the configured gates.
    mode = "best" if selector["pick_best"] else "cheapest"
    ranked = offer_selector.rank_offers(offers, selector, mode=mode)

    if not ranked:
        print(f"❌ No {config['GPU_TARGET']} instance passes the selection filters!")
        print(f"   Query: {query}")
        print("   → Loosen a filter in the Cloud tab (reliability / internet / "
              "CPU / price) or check availability.")
        sys.exit(1)

    candidates = ranked[:int(selector["retry_count"])]
    how = "quality score" if mode == "best" else "cheapest price"
    print(f"🔍 {len(ranked)} offer(s) pass the filters — "
          f"trying top {len(candidates)} by {how}:")
    for i, off in enumerate(candidates, 1):
        print(f"   {i}. {offer_selector.describe_offer(off)}")

    # ── Read config to get disk size & image (offer-independent) ──
    try:
        import yaml
        with open("configs/default_config.yaml") as f:
            cfg = yaml.safe_load(f)
        disk_size = int(os.getenv("DISK_SIZE_GB",
                       cfg.get("mlops", {}).get("vast", {}).get("disk_size", 60)))
        image = cfg.get("mlops", {}).get("vast", {}).get(
            "image", "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
        )
    except Exception:
        disk_size = int(os.getenv("DISK_SIZE_GB", 60))
        image = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"

    # ── Create the instance with onstart script, trying candidates in order ──
    # Build the env string per candidate (INSTANCE_ID differs per offer).
    def build_env_vars(offer_id: str) -> str:
        return " ".join(
            f"-e {k}={v}"
            for k, v in {
                "VAST_API_KEY": config["VAST_API_KEY"],
                "INSTANCE_ID": offer_id,
                "DAGSHUB_TOKEN": config["DAGSHUB_TOKEN"],
                "DAGSHUB_USERNAME": config["DAGSHUB_USERNAME"],
                "DAGSHUB_REPO_NAME": config["DAGSHUB_REPO_NAME"],
                "DAGSHUB_TRACKING_URI": config["DAGSHUB_TRACKING_URI"],
                "GIT_REPO_URL": config["GIT_REPO_URL"],
                "GIT_BRANCH": config["GIT_BRANCH"],
                "GPU_TARGET": config["GPU_TARGET"],
                "TARGET_PIPELINE": config["TARGET_PIPELINE"],
                "MIN_CUDA_VERSION": config["MIN_CUDA_VERSION"],
                "DISK_SIZE_GB": os.getenv("DISK_SIZE_GB", ""),
                "EXPERIMENT_PROFILES": config.get("EXPERIMENT_PROFILES", ""),
                "HPO_STUDY": config.get("HPO_STUDY", ""),
                "HPO_TRIALS": config.get("HPO_TRIALS", ""),
                "HPO_EPOCHS": config.get("HPO_EPOCHS", ""),
                "HPO_BASE_PROFILE": config.get("HPO_BASE_PROFILE", ""),
                "FREEZE_ENCODER": config["FREEZE_ENCODER"],
                "UNFREEZE_LAST_N_BLOCKS": config["UNFREEZE_LAST_N_BLOCKS"],
                "FREEZE_FEATURE_EXTRACTOR": config["FREEZE_FEATURE_EXTRACTOR"],
                "ENCODER_TYPE": config.get("ENCODER_TYPE", ""),
                "ALLOW_HUB_DOWNLOAD": config.get("ALLOW_HUB_DOWNLOAD", ""),
                "LOCAL_PATH_ECAPA": config.get("LOCAL_PATH_ECAPA", ""),
                "LOCAL_PATH_CAMPP": config.get("LOCAL_PATH_CAMPP", ""),
                "LOCAL_PATH_ERES2NET": config.get("LOCAL_PATH_ERES2NET", ""),
                "LOCAL_PATH_TITANET": config.get("LOCAL_PATH_TITANET", ""),
                "LOCAL_PATH_WAVLM": config.get("LOCAL_PATH_WAVLM", ""),
                "KAGGLE_USERNAME": config["KAGGLE_USERNAME"],
                "KAGGLE_KEY": config["KAGGLE_KEY"],
            }.items()
            if v
        )

    instance_id = None
    for off in candidates:
        print(f"🚀 Creating instance {offer_selector.describe_offer(off)} ...")
        create_cmd = (
            f'vastai create instance {off["id"]} '
            f'--image "{image}" '
            f'--disk {disk_size} '
            f'--env "{build_env_vars(str(off["id"]))}" '
            f'--onstart setup_vast_onstart.sh '
            f'--raw'
        )
        create_output = run_cmd(create_cmd, return_output=True, silent_error=True)
        print(f"  Instance creation response: {create_output[:200]}...")
        try:
            response = json.loads(create_output or "{}")
        except json.JSONDecodeError:
            response = {}
        if isinstance(response, dict) and response.get("error"):
            print(f"  ⚠ {response.get('msg', 'unknown error')} — trying next offer.")
            continue
        instance_id = (str(response.get("new_instance", off["id"]))
                       if isinstance(response, dict) else str(off["id"]))
        break

    if instance_id is None:
        print(f"❌ All {len(candidates)} candidate offers failed to rent.")
        print("   Re-run with looser filters — Vast.ai availability changes fast.")
        sys.exit(1)
    new_instance_id = instance_id
    print(f"\n🎉 Instance #{new_instance_id} is being provisioned!")

    print(f"   GPU:        {config['GPU_TARGET']}")
    print(f"   Pipeline:   {config['TARGET_PIPELINE']}")
    print(f"   Min CUDA:   {config['MIN_CUDA_VERSION']}")
    print(f"   Image:      {image}")
    print(f"   Disk:       {disk_size}GB")
    print()

    # Print instance ID as last line for caller parsing
    print(f"INSTANCE_ID={new_instance_id}")
    print("📊 Monitor progress on DagsHub:")
    print(f"   {config['DAGSHUB_TRACKING_URI']}")
    print()
    print("⚠️  The server will self-destruct automatically after completion.")
    print("   Results will be logged to DagsHub MLflow.")


if __name__ == "__main__":
    main()
