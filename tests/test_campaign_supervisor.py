from pathlib import Path

import scripts.campaign_supervisor as supervisor
from scripts.campaign_supervisor import (
    ANALYSIS_SPECS,
    _artifact_receipts,
    _dotenv_value,
    build_parser,
)


def test_dotenv_value_strips_matching_quotes() -> None:
    assert _dotenv_value('"https://example.invalid/mlflow"') == (
        "https://example.invalid/mlflow"
    )
    assert _dotenv_value("'secret-value'") == "secret-value"


def test_dotenv_value_preserves_unquoted_and_unmatched_values() -> None:
    assert _dotenv_value("  raw-value  ") == "raw-value"
    assert _dotenv_value('"unmatched') == '"unmatched'


def test_artifact_receipts_include_recovery_files_and_terminal_extras(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    profile = "candidate"
    checkpoint_dir = tmp_path / "checkpoints" / profile
    bundle_dir = checkpoint_dir / "campp_best_bundle"
    bundle_dir.mkdir(parents=True)
    (checkpoint_dir / "campp_best_raw.pt").write_bytes(b"checkpoint")
    (bundle_dir / "resolved_config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (bundle_dir / "MODEL_CARD.md").write_text("model\n", encoding="utf-8")
    (checkpoint_dir / "ignored.tmp").write_text("ignore\n", encoding="utf-8")
    config = tmp_path / "configs" / "experiments" / "candidate.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("profile: candidate\n", encoding="utf-8")
    log = tmp_path / "data" / "experiments" / "run.log"
    log.parent.mkdir(parents=True)
    log.write_text("terminal log\n", encoding="utf-8")

    receipts = _artifact_receipts(profile, extra_paths=(config, log, config))
    paths = {row["path"] for row in receipts}

    assert paths == {
        "checkpoints/candidate/campp_best_raw.pt",
        "checkpoints/candidate/campp_best_bundle/resolved_config.yaml",
        "checkpoints/candidate/campp_best_bundle/MODEL_CARD.md",
        "configs/experiments/candidate.yaml",
        "data/experiments/run.log",
    }
    assert all(len(row["sha256"]) == 64 for row in receipts)


def test_set_max_run_hours_cli_requires_value_and_reason() -> None:
    args = build_parser().parse_args([
        "set-max-run-hours",
        "--hours", "15",
        "--reason", "locked treatment runtime contract",
    ])
    assert args.hours == 15.0
    assert args.reason == "locked treatment runtime contract"
    assert args.func is supervisor.cmd_set_max_run_hours


def test_analyze_cli_accepts_only_allowlisted_preregistration() -> None:
    analysis = "frozen-eres2netv2-lme20-threefold-v1"
    assert analysis in ANALYSIS_SPECS
    args = build_parser().parse_args(["analyze", "--analysis", analysis])
    assert args.analysis == analysis
    assert args.func is supervisor.cmd_analyze
