from types import SimpleNamespace

from scripts.mlflow_audit_run import audit_run


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
