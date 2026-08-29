from types import SimpleNamespace

import pytest

from scripts.mlflow_audit_run import audit_run, resolve_run_id


class _Client:
    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id,
                status="FINISHED",
                start_time=10,
                end_time=20,
            ),
            data=SimpleNamespace(
                params={"seed": "42"},
                metrics={"val_macro_f1": 0.9},
                tags={"mlflow.runName": "control"},
            ),
        )

    def get_metric_history(self, run_id, key):
        return [
            SimpleNamespace(step=1, value=0.8),
            SimpleNamespace(step=2, value=0.9),
        ]

    def list_artifacts(self, run_id, path=""):
        if not path:
            return [
                SimpleNamespace(path="model.pt", is_dir=False, file_size=12),
                SimpleNamespace(path="provenance", is_dir=True, file_size=None),
            ]
        if path == "provenance":
            return [
                SimpleNamespace(
                    path="provenance/config.yaml",
                    is_dir=False,
                    file_size=8,
                )
            ]
        return []


def test_audit_run_reports_series_and_artifact_inventory():
    result = audit_run(_Client(), "run-1")

    assert result["status"] == "FINISHED"
    assert result["parameter_keys"] == ["seed"]
    assert result["metric_series"]["val_macro_f1"] == {
        "points": 2,
        "finite_points": 2,
        "min_step": 1,
        "max_step": 2,
        "last_value": 0.9,
    }
    assert result["artifact_count"] == 2
    assert result["artifact_bytes_known"] == 20
    assert result["model_artifacts"] == ["model.pt"]
    assert result["provenance_artifacts"] == ["provenance/config.yaml"]


class _SearchClient:
    def __init__(self):
        self.runs = [
            SimpleNamespace(
                info=SimpleNamespace(
                    run_id="old",
                    status="FINISHED",
                    start_time=100,
                ),
                data=SimpleNamespace(tags={"mlflow.runName": "paired-control"}),
            ),
            SimpleNamespace(
                info=SimpleNamespace(
                    run_id="wanted",
                    status="RUNNING",
                    start_time=300,
                ),
                data=SimpleNamespace(tags={"mlflow.runName": "paired-control"}),
            ),
            SimpleNamespace(
                info=SimpleNamespace(
                    run_id="other",
                    status="RUNNING",
                    start_time=400,
                ),
                data=SimpleNamespace(tags={"mlflow.runName": "treatment"}),
            ),
        ]

    def get_experiment_by_name(self, name):
        assert name == "speaker-identification"
        return SimpleNamespace(experiment_id="experiment-1")

    def search_runs(self, **kwargs):
        assert kwargs["experiment_ids"] == ["experiment-1"]
        assert kwargs["max_results"] == 10_000
        return self.runs


def test_resolve_run_id_uses_exact_name_and_start_time_bound():
    assert resolve_run_id(
        _SearchClient(),
        run_id=None,
        run_name="paired-control",
        experiment_name="speaker-identification",
        started_after_ms=200,
    ) == "wanted"


def test_resolve_run_id_refuses_ambiguous_name():
    with pytest.raises(ValueError, match="expected exactly one"):
        resolve_run_id(
            _SearchClient(),
            run_id=None,
            run_name="paired-control",
            experiment_name="speaker-identification",
            started_after_ms=None,
        )
