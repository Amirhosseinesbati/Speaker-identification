from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_known_hard_sampling_artifact import (  # noqa: E402
    build_artifact_payload,
    classify_known_features,
)


def _rows():
    return [
        {"audio_file": "a.wav", "speaker_id": "s1", "label": 1,
         "duration_seconds": 1.0, "pcm_rms": 0.01, "active_fraction": 0.10},
        {"audio_file": "b.wav", "speaker_id": "s1", "label": 1,
         "duration_seconds": 2.0, "pcm_rms": 0.02, "active_fraction": 0.20},
        {"audio_file": "c.wav", "speaker_id": "s2", "label": 2,
         "duration_seconds": 3.0, "pcm_rms": 0.03, "active_fraction": 0.30},
        {"audio_file": "d.wav", "speaker_id": "s2", "label": 2,
         "duration_seconds": 4.0, "pcm_rms": 0.04, "active_fraction": 0.40},
    ]


def test_quartile_rule_is_fixed_and_two_of_three():
    thresholds, rows = classify_known_features(_rows())
    assert thresholds == pytest.approx({
        "duration_seconds": 1.75,
        "pcm_rms": 0.0175,
        "active_fraction": 0.175,
    })
    assert [row["is_hard"] for row in rows] == [True, False, False, False]
    assert [row["sampling_weight"] for row in rows] == [2.0, 1.0, 1.0, 1.0]


def test_payload_is_train_only_and_split_locked():
    train_df = pd.DataFrame({
        "audio_file": ["ood.wav", "a.wav", "b.wav", "c.wav", "d.wav"],
        "speaker_id": ["unknown", "s1", "s1", "s2", "s2"],
        "label": [0, 1, 1, 2, 2],
    })
    val_df = pd.DataFrame({"audio_file": ["val.wav"], "label": [1]})
    payload = build_artifact_payload(
        train_df, val_df, _rows(), profile="fold0",
        split_metadata={"fold": 0, "folds": 3, "seed": 42},
        competition_known_count=446,
    )
    assert payload["validation_overlap_count"] == 0
    assert payload["hard_file_count"] == 1
    assert payload["weights"]["a.wav"] == 2.0
    assert set(payload["weights"]) == {"a.wav", "b.wav", "c.wav", "d.wav"}
    assert len(payload["training_rows_sha256"]) == 64
    assert len(payload["known_feature_table_sha256"]) == 64


def test_payload_rejects_validation_leakage():
    train_df = pd.DataFrame({
        "audio_file": ["a.wav", "b.wav", "c.wav", "d.wav"],
        "speaker_id": ["s1", "s1", "s2", "s2"],
        "label": [1, 1, 2, 2],
    })
    val_df = pd.DataFrame({"audio_file": ["a.wav"], "label": [1]})
    with pytest.raises(ValueError, match="Validation leakage"):
        build_artifact_payload(
            train_df, val_df, _rows(), profile="fold0",
            split_metadata={"fold": 0}, competition_known_count=446,
        )
