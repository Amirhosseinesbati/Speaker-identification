from __future__ import annotations

import math

import pytest

from scripts.evaluate_channelrobust_continuation import evaluate_continuation


def _history(*, final_epoch: int = 50, improving: bool = True) -> list[dict]:
    rows = []
    for epoch in range(1, final_epoch + 1):
        late = max(0, epoch - 40)
        raw = 0.90 + min(epoch, 48) * 0.0007
        ema = 0.89 + (0.0005 * epoch if improving else -0.0001 * epoch)
        train_loss = 3.0 - (0.02 * epoch if improving else 0.001 * epoch)
        rows.append({
            "epoch": epoch,
            "val_macro_f1": raw,
            "val_ema_macro_f1": ema,
            "train_loss": train_loss,
            "val_loss": 1.1 + late * 0.0005,
        })
    return rows


def test_continuation_gate_accepts_locked_positive_terminal_trend() -> None:
    result = evaluate_continuation(
        _history(), selected_raw_epoch=48, termination="timeout",
        provenance_ok=True, safety_ok=True,
    )
    assert result["eligible"] is True
    assert result["trend_tests_passed"] == 3
    assert result["raw_0_93_by_epoch_60"] is True
    assert result["ema_final10_ols_slope"] == pytest.approx(0.0005)


def test_continuation_gate_rejects_nonterminal_flat_run() -> None:
    history = _history(improving=False)
    result = evaluate_continuation(
        history, selected_raw_epoch=48, termination="complete",
        provenance_ok=True, safety_ok=True,
    )
    assert result["eligible"] is False
    assert result["termination_eligible"] is False
    assert result["trend_tests_passed"] < 2


def test_continuation_gate_requires_provenance_and_safety() -> None:
    history = _history()
    result = evaluate_continuation(
        history, selected_raw_epoch=48, termination="patience",
        provenance_ok=False, safety_ok=True,
    )
    assert result["eligible"] is False


def test_continuation_gate_rejects_nonfinite_history() -> None:
    history = _history()
    history[-1]["val_loss"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_continuation(
            history, selected_raw_epoch=48, termination="timeout",
            provenance_ok=True, safety_ok=True,
        )


def test_continuation_gate_rejects_nonbest_selected_epoch() -> None:
    with pytest.raises(ValueError, match="not history best"):
        evaluate_continuation(
            _history(), selected_raw_epoch=47, termination="timeout",
            provenance_ok=True, safety_ok=True,
        )
