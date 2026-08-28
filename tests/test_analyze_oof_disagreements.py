import json
import wave
from pathlib import Path

import numpy as np

from scripts.analyze_oof_disagreements import analyze_disagreements


def _write_oof(path: Path, files, labels, probabilities, embeddings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        files=np.asarray(files),
        labels=np.asarray(labels, dtype=np.int64),
        competition_probs=np.asarray(probabilities, dtype=np.float32),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        split_scheme=np.asarray(["kfold"]),
        split_fold=np.asarray([0]),
        split_folds=np.asarray([3]),
        split_seed=np.asarray([42]),
    )
    (path.parent / "class_map.json").write_text(
        json.dumps({"unknown": 0, "known": [1, 2, 3]}), encoding="utf-8"
    )


def _write_wav(path: Path, frames: int, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * frames)


def test_disagreement_report_aligns_files_and_tracks_unknown_rescues(tmp_path):
    candidate_path = tmp_path / "candidate" / "oof_predictions.npz"
    baseline_path = tmp_path / "baseline" / "oof_predictions.npz"
    files = ["a.wav", "b.wav", "c.wav", "d.wav"]
    labels = [0, 1, 0, 2]
    candidate_probs = [
        [0.70, 0.10, 0.10, 0.10],
        [0.05, 0.80, 0.10, 0.05],
        [0.20, 0.50, 0.20, 0.10],
        [0.05, 0.80, 0.10, 0.05],
    ]
    baseline_probs = [
        [0.20, 0.60, 0.10, 0.10],
        [0.05, 0.80, 0.10, 0.05],
        [0.75, 0.10, 0.10, 0.05],
        [0.05, 0.10, 0.20, 0.65],
    ]
    embeddings = np.eye(4, dtype=np.float32)
    _write_oof(candidate_path, files, labels, candidate_probs, embeddings)
    order = [2, 0, 3, 1]
    _write_oof(
        baseline_path,
        [files[index] for index in order],
        [labels[index] for index in order],
        [baseline_probs[index] for index in order],
        embeddings[order],
    )
    audio_dir = tmp_path / "audio"
    for index, filename in enumerate(files, start=1):
        _write_wav(audio_dir / filename, frames=index * 16000)
    clusters_path = tmp_path / "clusters.json"
    clusters_path.write_text(json.dumps({"a.wav": 7, "c.wav": 8}), encoding="utf-8")

    report, rows = analyze_disagreements(
        candidate_path,
        baseline_path,
        audio_dir=audio_dir,
        unknown_cluster_path=clusters_path,
        fixed_thresholds=(0.0,),
    )

    by_file = {row["file"]: row for row in rows}
    assert by_file["a.wav"]["outcome"] == "candidate_only_correct"
    assert by_file["c.wav"]["outcome"] == "baseline_only_correct"
    assert by_file["d.wav"]["outcome"] == "both_wrong"
    assert by_file["a.wav"]["duration_seconds"] == 1.0
    assert by_file["a.wav"]["unknown_cluster"] == 7
    assert report["complementarity"]["candidate_only_correct"] == 1
    assert report["complementarity"]["baseline_only_correct"] == 1
    assert report["unknown_cluster_summary"]["7"]["candidate_only_correct"] == 1
    assert report["single_correct_auc_diagnostics"]["ood_probability_delta"][
        "samples"
    ] == 2


def test_fixed_gate_is_descriptive_and_uses_unknown_probability_delta(tmp_path):
    candidate_path = tmp_path / "candidate" / "oof_predictions.npz"
    baseline_path = tmp_path / "baseline" / "oof_predictions.npz"
    files = ["a.wav", "b.wav"]
    labels = [0, 1]
    candidate_probs = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]
    baseline_probs = [[0.3, 0.5, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]
    embeddings = np.eye(2, dtype=np.float32)
    _write_oof(candidate_path, files, labels, candidate_probs, embeddings)
    _write_oof(baseline_path, files, labels, baseline_probs, embeddings)

    report, _ = analyze_disagreements(
        candidate_path, baseline_path, fixed_thresholds=(0.0, 0.5)
    )

    gates = [
        gate
        for gate in report["fixed_gates_descriptive_only"]
        if gate["score"] == "ood_probability_delta"
    ]
    assert gates[0]["candidate_selected"] == 1
    assert gates[0]["overall_acc"] == 1.0
    assert gates[1]["candidate_selected"] == 0
    assert "same-fold" in report["selection_warning"]
