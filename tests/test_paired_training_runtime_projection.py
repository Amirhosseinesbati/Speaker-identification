import pytest

from scripts.project_paired_training_runtime import (
    parse_complete_epoch_stages,
    probe_files_per_second,
    project_paired_runtime,
)


def _log() -> str:
    return (
        "\r  Train: 100%|##########| 58/58 [01:40<00:00, 1.72s/it]\n"
        "\r  Val: 100%|##########| 34/34 [00:30<00:00, 1.13it/s]\n"
        "\r  Val: 100%|##########| 34/34 [00:31<00:00, 1.13it/s]\n"
        "  Epoch 1/120 — Loss: 1.0\n"
        "\r  Train: 100%|##########| 58/58 [01:42<00:00, 1.75s/it]\n"
        "\r  Val: 100%|##########| 34/34 [00:32<00:00, 1.10it/s]\n"
        "\r  Val: 100%|##########| 34/34 [00:33<00:00, 1.10it/s]\n"
        "  Epoch 2/120 — Loss: 0.9\n"
        # An incomplete next epoch must not enter the projection.
        "\r  Train: 50%|#####| 29/58 [00:50<00:50, 1.72s/it]\n"
    )


def test_parses_only_contiguous_complete_epoch_stage_timings():
    rows = parse_complete_epoch_stages(_log())
    assert rows == [
        {
            "epoch": 1,
            "train_seconds": 100.0,
            "raw_val_seconds": 30.0,
            "ema_val_seconds": 31.0,
        },
        {
            "epoch": 2,
            "train_seconds": 102.0,
            "raw_val_seconds": 32.0,
            "ema_val_seconds": 33.0,
        },
    ]


def test_runtime_gate_uses_probe_ratio_and_observed_nontrain_time():
    result = project_paired_runtime(
        epoch_stages=parse_complete_epoch_stages(_log()),
        control_wall_seconds=360.0,
        control_files_per_second=20.0,
        treatment_files_per_second=10.0,
        treatment_epochs=120,
        timeout_hours=12.0,
        required_headroom_fraction=0.20,
        dph_total=0.1711111111111111,
        max_incremental_cost_usd=2.60,
    )
    # Observed wall is 180s/epoch and median train is 101s, so the
    # conservative non-train component is 79s. Doubling train cost gives 281s.
    assert result["projected_treatment_epoch_seconds"] == pytest.approx(281.0)
    assert result["projected_treatment_hours"] == pytest.approx(9.3666667)
    assert result["projected_headroom_fraction"] == pytest.approx(0.2194444)
    assert result["time_gate_pass"] is True
    assert result["cost_gate_pass"] is True
    assert result["launch_runtime_gate_pass"] is True


def test_runtime_gate_rejects_truncating_timeout():
    result = project_paired_runtime(
        epoch_stages=parse_complete_epoch_stages(_log()),
        control_wall_seconds=360.0,
        control_files_per_second=20.0,
        treatment_files_per_second=10.0,
        treatment_epochs=120,
        timeout_hours=8.0,
        required_headroom_fraction=0.20,
    )
    assert result["time_gate_pass"] is False
    assert result["launch_runtime_gate_pass"] is False


def test_reads_exact_successful_probe_row():
    report = {
        "results": [
            {"batch_size": 48, "status": "ok", "files_per_second": 12.5},
            {"batch_size": 64, "status": "oom"},
        ]
    }
    assert probe_files_per_second(report, 48) == pytest.approx(12.5)
    with pytest.raises(ValueError, match="Expected one"):
        probe_files_per_second(report, 32)
