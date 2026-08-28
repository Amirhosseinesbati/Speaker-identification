import json

import pytest

from scripts.compare_training_histories import (
    compare_histories,
    load_history,
    load_history_context,
)


def _row(epoch, raw, ema, logit, val_loss):
    return {
        "epoch": epoch,
        "val_macro_f1": raw,
        "val_ema_macro_f1": ema,
        "val_logit_avg_macro_f1": logit,
        "val_loss": val_loss,
        "train_loss": val_loss + 0.5,
        "lr": 0.001 / epoch,
    }


def test_compare_uses_common_horizon_and_keeps_full_baseline_best():
    candidate = [_row(1, 0.70, 0.69, 0.68, 1.2), _row(2, 0.80, 0.81, 0.77, 1.0)]
    baseline = [
        _row(1, 0.72, 0.70, 0.69, 1.3),
        _row(2, 0.78, 0.79, 0.76, 1.1),
        _row(3, 0.90, 0.88, 0.86, 1.05),
    ]

    report = compare_histories(candidate, baseline, tail_window=2)

    assert report["latest_common_epoch"] == 2
    assert report["latest"]["val_macro_f1"]["delta"] == pytest.approx(0.02)
    assert report["best"]["val_macro_f1"]["candidate"] == {
        "epoch": 2,
        "value": 0.80,
    }
    assert report["best"]["val_macro_f1"]["baseline_same_horizon"] == {
        "epoch": 2,
        "value": 0.78,
    }
    assert report["best"]["val_macro_f1"]["baseline_full"] == {
        "epoch": 3,
        "value": 0.90,
    }
    assert report["diagnostic_gaps"]["candidate_probability_minus_logit"] == pytest.approx(0.03)
    assert report["diagnostic_gaps"]["candidate_raw_minus_ema"] == pytest.approx(-0.01)
    assert report["diagnostic_gaps"]["candidate_val_loss_above_min"] == 0


def test_load_history_accepts_checkpoint_style_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"training_history": [_row(2, 0.8, 0.8, 0.7, 1.0), _row(1, 0.7, 0.7, 0.6, 1.2)]}),
        encoding="utf-8",
    )

    assert [row["epoch"] for row in load_history(path)] == [1, 2]


def test_compare_rejects_disjoint_epochs():
    with pytest.raises(ValueError, match="common epochs"):
        compare_histories([_row(1, 0.7, 0.7, 0.6, 1.0)], [_row(2, 0.8, 0.8, 0.7, 0.9)])


def test_compare_corrects_primary_loss_weight_scale():
    candidate = [_row(1, 0.7, 0.7, 0.6, 1.9)]
    baseline = [_row(1, 0.7, 0.7, 0.6, 2.0)]

    report = compare_histories(
        candidate,
        baseline,
        candidate_primary_loss_scale=0.95,
        baseline_primary_loss_scale=1.0,
    )

    corrected = report["primary_loss_scale_correction"]["latest"]
    assert corrected["val_loss"]["candidate"] == pytest.approx(2.0)
    assert corrected["val_loss"]["baseline"] == pytest.approx(2.0)
    assert corrected["val_loss"]["delta"] == pytest.approx(0.0)


def test_load_history_context_infers_primary_weight_sum(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "config": {
                    "training": {
                        "loss": {
                            "speaker": {"weight": 0.8075},
                            "ood": {"weight": 0.1425},
                        }
                    }
                },
                "training_history": [_row(1, 0.7, 0.7, 0.6, 1.0)],
            }
        ),
        encoding="utf-8",
    )

    history, scale = load_history_context(path)

    assert len(history) == 1
    assert scale == pytest.approx(0.95)
