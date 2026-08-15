"""
Optuna hyperparameter search (Audit §17.4 + §16.2).

Each trial = a named experiment profile (see ``src/experiment_config.py``)
with a unique ``logging.checkpoint_dir``, run through the SAME training path as
any manual experiment (``run_pipeline.py --experiment <name> --run train``).
The competition Macro-F1 is parsed back from the run's stdout, so the study
optimises the real objective, not a proxy.

This is the **coarse** phase of the two-phase plan in §16.2 (fewer epochs, one
encoder/fold per trial). Fine-grained per-epoch pruning needs in-process
training (a future enhancement); here each trial is a self-contained subprocess,
so TPESampler drives the search while trial failures are penalised to 0.0.

Usage:
    uv run --no-sync python -m src.hpo --trials 30 --epochs 30 --study speaker-hpo
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.experiment_config import load_base, save_profile  # noqa: E402

PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipelines" / "run_pipeline.py"
HPO_DIR = PROJECT_ROOT / "checkpoints" / "hpo"
STUDY_DB_NAME = "study.db"
BEST_JSON = HPO_DIR / "best_params.json"
LOG_DIR = HPO_DIR / "logs"

_METRIC_RE = re.compile(r"Best val Macro-F1:\s*([0-9.]+)")


# Search space (single source of truth — rendered read-only in the Config tab
# HPO section so the operator can see exactly what is being tuned and in what
# range). Each entry: {name, kind, low, high, log, description}.
HPO_SPACE = [
    {"name": "head_lr", "kind": "float", "low": 5e-5, "high": 5e-4, "log": True,
     "description": "LR of the ArcFace/OOD heads"},
    {"name": "encoder_lr", "kind": "float", "low": 1e-6, "high": 1e-4, "log": True,
     "description": "LR of the unfrozen encoder blocks"},
    {"name": "weight_decay", "kind": "float", "low": 1e-5, "high": 1e-3, "log": True,
     "description": "AdamW weight decay"},
    {"name": "arcface_margin", "kind": "float", "low": 0.2, "high": 0.45, "log": False,
     "description": "ArcFace angular margin"},
    {"name": "arcface_scale", "kind": "float", "low": 24.0, "high": 40.0, "log": False,
     "description": "ArcFace scale"},
    {"name": "label_smoothing", "kind": "float", "low": 0.0, "high": 0.15, "log": False,
     "description": "Speaker CE label smoothing"},
    {"name": "ood_loss_weight", "kind": "float", "low": 0.1, "high": 0.3, "log": False,
     "description": "OOD head loss weight"},
    {"name": "ood_batch_ratio", "kind": "float", "low": 0.3, "high": 0.5, "log": False,
     "description": "Fraction of OOD samples per batch"},
]


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample every entry in ``HPO_SPACE`` (used by both the objective and the UI)."""
    return {
        e["name"]: trial.suggest_float(e["name"], e["low"], e["high"], log=e["log"])
        for e in HPO_SPACE
    }


def apply_params_to_config(base: dict, params: dict, epochs: int) -> dict:
    """Map a params dict (from ``suggest_params``) onto a deep copy of ``base``."""
    import copy

    cfg = copy.deepcopy(base)
    tr = cfg.setdefault("training", {})
    tr["learning_rate"] = float(params["head_lr"])
    tr["encoder_lr"] = float(params["encoder_lr"])
    tr["weight_decay"] = float(params["weight_decay"])
    tr["label_smoothing"] = float(params["label_smoothing"])
    tr["epochs"] = int(epochs)
    tr.setdefault("loss", {}).setdefault("ood", {})["weight"] = float(params["ood_loss_weight"])
    tr["ood_loss_weight"] = float(params["ood_loss_weight"])

    cfg.setdefault("audio", {})["ood_batch_ratio"] = float(params["ood_batch_ratio"])

    head_type = cfg.get("model", {}).get("speaker_head_type", "arcface")
    head_cfg = (cfg.setdefault("model", {})
                .setdefault("speaker_head_config", {})
                .setdefault(head_type, {}))
    head_cfg["margin"] = float(params["arcface_margin"])
    head_cfg["scale"] = float(params["arcface_scale"])
    return cfg


def suggest_trial_config(trial: optuna.Trial, base: dict, epochs: int) -> tuple:
    """Build a trial config from ``HPO_SPACE``.

    Returns ``(trial_name, config)``. The config overrides the base's training
    knobs and points ``logging.checkpoint_dir`` at a per-trial directory so
    trials never clobber each other's ``<enc>_best.pt``.
    """
    name = f"hpo-trial-{trial.number:03d}"
    params = suggest_params(trial)
    cfg = apply_params_to_config(base, params, epochs)
    cfg.setdefault("logging", {})["checkpoint_dir"] = str(HPO_DIR / name)
    return name, cfg


def _run_train(profile_name: str, no_mlflow: bool = False,
               trial_env: Optional[dict] = None) -> tuple:
    cmd = [sys.executable, str(PIPELINE_SCRIPT), "--experiment", profile_name,
           "--run", "train"]
    if no_mlflow:
        cmd.append("--no-mlflow")
    # Forward the study context so the trial run's MLflow deployment envelope
    # records which study/trial it belongs to (visible on DagsHub).
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    if trial_env:
        env.update(trial_env)
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    # Persist the FULL trial log so a failure can be inspected after the fact
    # (the objective only prints a short tail).
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{profile_name}.log").write_text(
        stdout + "\n\n===== STDERR =====\n" + stderr, encoding="utf-8",
    )
    return stdout, stderr, proc.returncode


def _parse_macro_f1(stdout: str) -> Optional[float]:
    hits = _METRIC_RE.findall(stdout)
    return float(hits[-1]) if hits else None


def _make_objective(base: dict, epochs: int, no_mlflow: bool = False,
                    study_name: str = "speaker-hpo"):
    def objective(trial: optuna.Trial) -> float:
        name, cfg = suggest_trial_config(trial, base, epochs)
        save_profile(name, cfg, base=base)
        stdout, stderr, code = _run_train(
            name, no_mlflow=no_mlflow,
            trial_env={"HPO_STUDY": study_name, "HPO_TRIAL": str(trial.number)})
        mf1 = _parse_macro_f1(stdout)
        if mf1 is None or code != 0:
            tail = "\n".join((stderr or stdout or "").splitlines()[-15:])
            print(f"\n  ❌ Trial {trial.number} failed (exit {code}) — returning 0.0\n"
                  f"  ── log tail ──\n{tail}\n"
                  f"  (full log: {LOG_DIR / (name + '.log')})")
            return 0.0
        print(f"  ✓ Trial {trial.number} → val Macro-F1 = {mf1:.4f}")
        return mf1
    return objective


def run_study(n_trials: int = 30, epochs: int = 30,
              study_name: str = "speaker-hpo", resume: bool = True,
              base: Optional[dict] = None, base_profile: Optional[str] = None,
              no_mlflow: bool = False) -> dict:
    """Run a coarse Optuna study; persist the best params + best profile.

    ``base_profile`` lets you tune a NAMED recipe (a config under
    configs/experiments/) instead of the default base config.
    """
    if base is None:
        if base_profile:
            from src.experiment_config import load_profile
            base = load_profile(base_profile)
        else:
            base = load_base()
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{HPO_DIR / STUDY_DB_NAME}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=resume,
    )
    study.optimize(_make_objective(base, epochs, no_mlflow=no_mlflow,
                                   study_name=study_name),
                   n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial

    # Rebuild the best trial's config deterministically from its params.
    best_cfg = apply_params_to_config(base, best.params, epochs)

    best_profile = f"{study_name}-best"
    save_profile(best_profile, best_cfg, base=base)

    summary = {
        "study_name": study_name,
        "n_trials": n_trials,
        "epochs": epochs,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_profile": best_profile,
    }
    BEST_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  🏆 HPO finished — best val Macro-F1 = {best.value:.4f}")
    print(f"     Params: {best.params}")
    print(f"     Best profile saved as `{best_profile}`")
    print(f"     Results: {BEST_JSON}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna coarse HPO (Audit §17.4)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of trials (default 30).")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per trial — coarse phase (default 30).")
    parser.add_argument("--study", default="speaker-hpo",
                        help="Study name (default speaker-hpo).")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new study (ignore existing sqlite study).")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking for each trial.")
    parser.add_argument("--base-profile", default=None,
                        help="Named experiment profile to tune (default: "
                             "configs/default_config.yaml).")
    args = parser.parse_args()

    run_study(n_trials=args.trials, epochs=args.epochs,
              study_name=args.study, resume=not args.fresh,
              base_profile=args.base_profile, no_mlflow=args.no_mlflow)
    return 0


if __name__ == "__main__":
    sys.exit(main())
