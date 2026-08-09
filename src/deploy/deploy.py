"""
Vast.ai GPU Deployment Script for Speaker-Identification.

Rents the cheapest available GPU on Vast.ai, injects the setup_vast.sh
bootstrap script, and runs the target pipeline remotely.

Usage:
    # From project root, after setting up .env:
    python -m src.deploy.deploy

    # With overrides (useful when called from deploy_app.py):
    GPU_TARGET=RTX_3090 TARGET_PIPELINE=all python -m src.deploy.deploy
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
    # Encoder fine-tune choice (ECAPA-aware). FREEZE_FEATURE_EXTRACTOR kept for
    # backward compatibility with the old WavLM-only naming.
    config["FREEZE_ENCODER"] = os.getenv("FREEZE_ENCODER", "true")
    config["UNFREEZE_LAST_N_BLOCKS"] = os.getenv("UNFREEZE_LAST_N_BLOCKS", "0")
    config["FREEZE_FEATURE_EXTRACTOR"] = os.getenv("FREEZE_FEATURE_EXTRACTOR", "true")

    # Validate required variables
    missing = [k for k, v in config.items()
               if not v and k not in ("KAGGLE_USERNAME", "KAGGLE_KEY")]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Please check your .env file.")
        sys.exit(1)

    return config


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
#  Main Deployment Flow
# ─────────────────────────────────────────────────────────

def main():
    # ── Check prerequisites ──
    setup_script = Path("setup_vast.sh")
    if not setup_script.exists():
        print(f"❌ setup_vast.sh not found in project root!")
        print("   Make sure you're running from the project root directory.")
        sys.exit(1)

    # ── Load config ──
    config = load_environment()
    print(f"🔍 Searching for cheapest {config['GPU_TARGET']} for pipeline: {config['TARGET_PIPELINE']}...")

    # ── Authenticate Vast.ai ──
    run_cmd(f"vastai set api-key {config['VAST_API_KEY']}", silent_error=True)

    # ── Search for cheapest GPU ──
    search_cmd = (
        f'vastai search offers "gpu_name={config["GPU_TARGET"]} num_gpus=1" '
        f"-o dph --raw"
    )
    raw_json = run_cmd(search_cmd, return_output=True)

    try:
        offers = json.loads(raw_json)
        if not offers:
            print(f"❌ No {config['GPU_TARGET']} instances available!")
            print("   Try a different GPU or check Vast.ai availability.")
            sys.exit(1)

        instance_id = str(offers[0]["id"])
        price = offers[0].get("dph_total", 0.0)
        print(f"✅ Found instance #{instance_id} — ${float(price):.3f}/hour")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"❌ Failed to parse Vast.ai offers: {e}")
        sys.exit(1)

    # ── Build environment string for the remote instance ──
    # Include ALL settings so setup_vast.sh knows what the user chose
    env_vars = " ".join(
        f"-e {k}={v}"
        for k, v in {
            "VAST_API_KEY": config["VAST_API_KEY"],
            "INSTANCE_ID": instance_id,
            "DAGSHUB_TOKEN": config["DAGSHUB_TOKEN"],
            "DAGSHUB_USERNAME": config["DAGSHUB_USERNAME"],
            "DAGSHUB_REPO_NAME": config["DAGSHUB_REPO_NAME"],
            "DAGSHUB_TRACKING_URI": config["DAGSHUB_TRACKING_URI"],
            "GIT_REPO_URL": config["GIT_REPO_URL"],
            "GIT_BRANCH": config["GIT_BRANCH"],
            "GPU_TARGET": config["GPU_TARGET"],
            "TARGET_PIPELINE": config["TARGET_PIPELINE"],
            "FREEZE_ENCODER": config["FREEZE_ENCODER"],
            "UNFREEZE_LAST_N_BLOCKS": config["UNFREEZE_LAST_N_BLOCKS"],
            "FREEZE_FEATURE_EXTRACTOR": config["FREEZE_FEATURE_EXTRACTOR"],
            "KAGGLE_USERNAME": config["KAGGLE_USERNAME"],
            "KAGGLE_KEY": config["KAGGLE_KEY"],
        }.items()
        if v
    )

    # ── Read config to get disk size & image ──
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

    # ── Create the instance with onstart script ──
    print(f"🚀 Creating instance with {disk_size}GB disk...")
    create_cmd = (
        f'vastai create instance {instance_id} '
        f'--image "{image}" '
        f'--disk {disk_size} '
        f'--env "{env_vars}" '
        f'--onstart setup_vast.sh '
        f'--raw'
    )

    create_output = run_cmd(create_cmd, return_output=True)
    print(f"  Instance creation response: {create_output[:200]}...")

    # ── Check for errors ──
    try:
        response = json.loads(create_output)
        if response.get("error"):
            print(f"\n❌ Vast.ai Error: {response.get('msg', 'Unknown error')}")
            sys.exit(1)
        new_instance_id = response.get("new_instance", instance_id)
        print(f"\n🎉 Instance #{new_instance_id} is being provisioned!")
    except json.JSONDecodeError:
        print(f"\n🎉 Instance creation initiated!")

    print(f"   GPU:        {config['GPU_TARGET']}")
    print(f"   Pipeline:   {config['TARGET_PIPELINE']}")
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
