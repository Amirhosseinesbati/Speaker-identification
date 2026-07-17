"""
Project Scaffolding Script for IAAA Competition 2026 - Speaker Identification.
Creates directory structure and default configuration file.
"""

import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DIRS = [
    "configs",
    "data/processed",
    "src",
    "checkpoints",
    "logs",
    "submission",
]

DEFAULT_CONFIG = {
    "hardware": {
        "mode": "vastai",  # "local" | "vastai"
        "profiles": {
            "local": {
                "description": "GTX 1660 Ti (6GB VRAM)",
                "device": "cuda" if os.name != "nt" else "cuda",
                "batch_size": 8,
                "num_workers": 2,
                "mixed_precision": True,
            },
            "vastai": {
                "description": "Cloud GPU (24GB+ VRAM)",
                "device": "cuda",
                "batch_size": 32,
                "num_workers": 4,
                "mixed_precision": True,
            },
        },
    },
    "audio": {
        "sample_rate": 16000,
        "duration_seconds": 3.0,
        "n_mels": 80,
        "n_fft": 400,
        "hop_length": 160,
    },
    "model": {
        "base_model": "microsoft/wavlm-base-plus",
        "freeze_feature_extractor": False,  # False for Vast.ai, True for local 1660 Ti
        "embedding_dim": 768,
    },
    "training": {
        "epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "warmup_steps": 500,
        "max_grad_norm": 5.0,
    },
    "data": {
        "labels_path": "data/raw/labels.csv",
        "audio_dir": "data/raw",
        "processed_labels": "data/processed/cleaned_labels.csv",
    },
    "logging": {
        "log_dir": "logs",
        "checkpoint_dir": "checkpoints",
    },
}


def create_directories():
    """Create all required directories."""
    for d in DIRS:
        path = BASE_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {path.relative_to(BASE_DIR)}/")
    print(f"\n  ✅ {len(DIRS)} directories created successfully.")


def write_config():
    """Write default_config.yaml."""
    config_path = BASE_DIR / "configs" / "default_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  ✓ Created: configs/default_config.yaml")
    print(f"  📝 Hardware mode: {DEFAULT_CONFIG['hardware']['mode']}")


def create_init_files():
    """Create __init__.py in src/ and submission/."""
    for pkg in ["src", "submission"]:
        init_path = BASE_DIR / pkg / "__init__.py"
        init_path.touch(exist_ok=True)
        print(f"  ✓ Created: {pkg}/__init__.py")


def write_gitkeep(dir_name):
    """Write .gitkeep in empty dirs that should be tracked."""
    for sub in ["checkpoints", "logs", "data/processed"]:
        gitkeep = BASE_DIR / sub / ".gitkeep"
        gitkeep.touch(exist_ok=True)


def create_gitignore_entry():
    """Ensure logs/ and checkpoints/ content is gitignored (but not the dirs)."""
    gitignore_path = BASE_DIR / ".gitignore"
    patterns_to_add = [
        "",
        "# Speaker Identification project",
        "logs/*",
        "checkpoints/*",
        "!logs/.gitkeep",
        "!checkpoints/.gitkeep",
        "*.pt",
        "*.pth",
        "data/processed/*",
        "!data/processed/.gitkeep",
    ]
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        existing_lines = set(content.splitlines())
        new_patterns = [p for p in patterns_to_add if p and p not in existing_lines]
        if new_patterns:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_patterns) + "\n")
            print(f"  ✓ Updated .gitignore with {len(new_patterns)} new patterns")
    else:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("\n".join(patterns_to_add) + "\n")
        print(f"  ✓ Created .gitignore with project patterns")


def main():
    print("=" * 55)
    print("  IAAA 2026 — Speaker Identification Scaffolding")
    print("=" * 55)
    print()

    print("[1/4] Creating directories...")
    create_directories()

    print("\n[2/4] Writing configuration...")
    write_config()

    print("\n[3/4] Creating package init files...")
    create_init_files()

    print("\n[4/4] Updating .gitignore...")
    create_gitignore_entry()

    print("\n" + "=" * 55)
    print("  ✅ Scaffolding complete! Ready for development.")
    print("=" * 55)


if __name__ == "__main__":
    main()
