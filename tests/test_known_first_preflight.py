from scripts.verify_known_first_experiments import _hardware_checks


def _config(*, batch_size: int = 48) -> dict:
    return {
        "hardware": {
            "mode": "vastai_3090_campp",
            "profiles": {
                "vastai_3090_campp": {
                    "batch_size": batch_size,
                    "num_workers": 8,
                    "mixed_precision": True,
                    "device": "cuda",
                }
            },
        }
    }


def test_measured_3090_hardware_recipe_passes() -> None:
    assert all(_hardware_checks(_config()).values())


def test_hardware_recipe_detects_batch_drift() -> None:
    checks = _hardware_checks(_config(batch_size=64))
    assert checks["batch_size"] is False
    assert all(value for key, value in checks.items() if key != "batch_size")
