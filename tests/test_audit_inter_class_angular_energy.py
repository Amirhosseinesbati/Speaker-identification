from pathlib import Path

import pytest
import torch

from scripts.audit_inter_class_angular_energy import audit_pair


def _checkpoint(path: Path, weights: torch.Tensor, *, epoch: int = 7) -> Path:
    torch.save(
        {
            "epoch": epoch,
            "weight_variant": "raw",
            "model_state_dict": {"head_speaker.weight": weights},
        },
        path,
    )
    return path


def test_pair_audit_measures_and_gates_exact_energy_ratio(tmp_path: Path) -> None:
    control = _checkpoint(
        tmp_path / "control.pt",
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
    )
    treatment = _checkpoint(
        tmp_path / "treatment.pt",
        torch.tensor([[1.0, 0.0], [0.2, 1.0]]),
    )

    report = audit_pair(control, treatment, maximum_energy_ratio=0.95)

    assert report["control"]["exclusive_inter_class_energy"] == pytest.approx(0.5)
    assert report["treatment"]["exclusive_inter_class_energy"] == pytest.approx(
        1.0 / 26.0
    )
    assert report["mechanism_gate"]["observed_energy_ratio"] == pytest.approx(
        1.0 / 13.0
    )
    assert report["mechanism_gate"]["passed"] is True
    assert len(report["control"]["sha256"]) == 64


def test_pair_audit_rejects_shape_mismatch(tmp_path: Path) -> None:
    control = _checkpoint(tmp_path / "control.pt", torch.eye(2))
    treatment = _checkpoint(tmp_path / "treatment.pt", torch.eye(3))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        audit_pair(control, treatment)


def test_pair_audit_rejects_zero_energy_control(tmp_path: Path) -> None:
    control = _checkpoint(tmp_path / "control.pt", torch.eye(2))
    treatment = _checkpoint(tmp_path / "treatment.pt", torch.eye(2))
    with pytest.raises(RuntimeError, match="must be positive"):
        audit_pair(control, treatment)
