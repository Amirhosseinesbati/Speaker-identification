"""
MLflow run registry (Audit §17.5) — list + rank experiments for promotion.

Reads runs from the configured MLflow experiment and returns a UI-friendly list
sorted by the most meaningful available score (final eval Macro-F1, then the
decision/ensemble OOF scores, then the last-epoch val Macro-F1). Kept defensive:
if MLflow is unreachable or the experiment is missing, it returns an empty list
instead of raising, so the Analysis tab degrades gracefully.
"""

from __future__ import annotations

from typing import List, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Most → least informative single-number score, in priority order.
SCORE_METRICS = [
    "macro_f1",                 # final eval (argmax, competition metric)
    "decision_val_macro_f1",    # decision bundle (17.3)
    "ensemble_best_macro_f1",   # ensemble selection (17.3)
    "val_macro_f1",             # last epoch of training
    "best_val_macro_f1",
]


def default_experiment_name() -> str:
    from src.experiment_config import load_base
    return str(load_base().get("mlops", {}).get("experiment_name", "speaker-identification"))


def _score_of(metrics: dict) -> tuple:
    """Return (score, metric_name) for a run's metrics dict (latest per key)."""
    for key in SCORE_METRICS:
        if key in metrics:
            return float(metrics[key]), key
    return None, None


def list_runs(experiment_name: Optional[str] = None,
              max_results: int = 100) -> List[dict]:
    """Return ranked run dicts, best score first; [] on any MLflow failure."""
    try:
        client = MlflowClient()
        name = experiment_name or default_experiment_name()
        exp = client.get_experiment_by_name(name)
        if exp is None:
            return []

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=max_results,
        )
    except Exception:
        return []

    rows = []
    for r in runs:
        metrics = dict(r.data.metrics)
        score, metric_name = _score_of(metrics)
        rows.append({
            "run_id": r.info.run_id,
            "run_name": r.info.run_name or r.info.run_id[:8],
            "status": str(r.info.status),
            "score": score,
            "score_metric": metric_name,
            "encoder": r.data.params.get("encoder_type", ""),
            "metrics": metrics,
        })

    # Best first; runs without a score sink to the bottom.
    rows.sort(key=lambda x: (x["score"] is None, -(x["score"] if x["score"] is not None else 0.0)))
    return rows
