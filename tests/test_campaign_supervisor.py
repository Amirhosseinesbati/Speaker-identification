from pathlib import Path
from types import SimpleNamespace

import scripts.campaign_supervisor as supervisor
from scripts.campaign_supervisor import (
    ANALYSIS_SPECS,
    _artifact_receipts,
    _dotenv_value,
    _notification_receipt,
    _record_notification,
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


def test_notification_marker_is_atomic_and_deduplicatable(
    tmp_path: Path, monkeypatch
) -> None:
    marker = tmp_path / "campaign_heartbeat_marker.json"
    monkeypatch.setattr(supervisor, "DEFAULT_TELEGRAM_MARKER", marker)

    _record_notification("p13:start:abc", 123, {"profile": "p13"})

    assert _notification_receipt("p13:start:abc") == 123
    assert _notification_receipt("different") is None
    payload = __import__("json").loads(marker.read_text(encoding="utf-8"))
    assert payload["telegram_message_id"] == 123
    assert payload["events"]["p13:start:abc"]["metadata"] == {"profile": "p13"}


def test_notify_works_when_supervisor_is_imported_as_module(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.telegram_notifier as notifier

    marker = tmp_path / "campaign_heartbeat_marker.json"
    monkeypatch.setattr(supervisor, "DEFAULT_TELEGRAM_MARKER", marker)
    monkeypatch.setattr(notifier, "send", lambda message: 456)

    assert supervisor._notify("scientific event", event_key="p13:event") == 456
    assert _notification_receipt("p13:event") == 456


def test_cmd_run_loads_resolved_profile_before_notification(
    tmp_path: Path, monkeypatch
) -> None:
    profile = "p13-test"
    config_path = tmp_path / "configs" / "experiments" / f"{profile}.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    resolved = {
        "experiment": {"purpose": "frozen SSL complement"},
        "model": {"encoder_type": "wavlm"},
        "data": {"split": {"fold": 0, "folds": 3, "seed": 42}},
        "training": {
            "epochs": 40,
            "early_stopping_start_epoch": 10,
            "early_stopping_patience": 8,
        },
    }

    class FakeStore:
        state = {
            "status": "ANALYZING",
            "policy": {"max_run_hours": 12.0, "hourly_rate_usd": 0.17},
            "current_run": None,
        }

        def load(self):
            return self.state

        def assert_budget(self, reserve_hours=0.0):
            return {}

        def start_run(self, run):
            self.state = {**self.state, "status": "RUNNING_EXPERIMENT"}

        def finish_run(self, **kwargs):
            self.state = {**self.state, "status": "ANALYZING"}
            return self.state

        def budget_snapshot(self, state):
            return {"estimated_cost_usd": 1.0}

    store = FakeStore()
    notifications = []
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "load_profile", lambda name: resolved)
    monkeypatch.setattr(supervisor, "_store", lambda args: store)
    monkeypatch.setattr(supervisor, "_assert_reproducible_checkout", lambda _: "abc")
    monkeypatch.setattr(supervisor, "_notify", lambda message, **kwargs: notifications.append((message, kwargs)))
    monkeypatch.setattr(supervisor, "_worker_environment", lambda: {})
    monkeypatch.setattr(supervisor, "_artifact_receipts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    args = SimpleNamespace(
        profile=profile,
        stage="eval",
        timeout_hours=1.0,
        no_mlflow=True,
        allow_dirty=False,
        state="unused",
        events="unused",
    )

    assert supervisor.cmd_run(args) == 0
    assert "Encoder: wavlm" in notifications[0][0]
    assert notifications[0][1]["event_key"].startswith(f"{profile}:started:")
