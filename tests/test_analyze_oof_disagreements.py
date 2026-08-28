import json
import subprocess
import sys
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
    assert report["duration_bins"][0]["samples"] == 1
    assert report["duration_bins"][1]["samples"] == 3
    assert report["audio_identity"]["files_scanned"] == 4
    assert report["audio_identity"]["duplicate_decoded_pcm"] == []


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


def test_audio_and_probability_duplicate_groups_are_reported(tmp_path):
    candidate_path = tmp_path / "candidate" / "oof_predictions.npz"
    baseline_path = tmp_path / "baseline" / "oof_predictions.npz"
    files = ["a.wav", "b.wav", "c.wav"]
    labels = [0, 0, 1]
    candidate_probs = [
        [0.7, 0.1, 0.1, 0.1],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
    ]
    baseline_probs = [
        [0.2, 0.6, 0.1, 0.1],
        [0.2, 0.6, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
    ]
    embeddings = np.eye(3, dtype=np.float32)
    _write_oof(candidate_path, files, labels, candidate_probs, embeddings)
    _write_oof(baseline_path, files, labels, baseline_probs, embeddings)
    audio_dir = tmp_path / "audio"
    _write_wav(audio_dir / "a.wav", frames=16000)
    (audio_dir / "b.wav").write_bytes((audio_dir / "a.wav").read_bytes())
    _write_wav(audio_dir / "c.wav", frames=32000)

    report, _ = analyze_disagreements(
        candidate_path, baseline_path, audio_dir=audio_dir
    )

    file_groups = report["audio_identity"]["duplicate_file_bytes"]
    pcm_groups = report["audio_identity"]["duplicate_decoded_pcm"]
    candidate_groups = report["identical_probability_rows"]["candidate"]
    assert file_groups[0]["files"] == ["a.wav", "b.wav"]
    assert pcm_groups[0]["files"] == ["a.wav", "b.wav"]
    assert candidate_groups[0]["files"] == ["a.wav", "b.wav"]


def test_duration_gate_uses_candidate_only_for_short_unknown_switch(tmp_path):
    candidate_path = tmp_path / "candidate" / "oof_predictions.npz"
    baseline_path = tmp_path / "baseline" / "oof_predictions.npz"
    files = ["short.wav", "long.wav"]
    labels = [0, 0]
    candidate_probs = [[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]
    baseline_probs = [[0.2, 0.6, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]]
    embeddings = np.eye(2, dtype=np.float32)
    _write_oof(candidate_path, files, labels, candidate_probs, embeddings)
    _write_oof(baseline_path, files, labels, baseline_probs, embeddings)
    audio_dir = tmp_path / "audio"
    _write_wav(audio_dir / "short.wav", frames=16000)
    _write_wav(audio_dir / "long.wav", frames=12 * 16000)

    report, _ = analyze_disagreements(
        candidate_path, baseline_path, audio_dir=audio_dir
    )

    gates = report["fixed_duration_gates_descriptive_only"]
    assert [gate["candidate_selected"] for gate in gates] == [1, 1, 1]
    assert all(gate["overall_acc"] == 0.5 for gate in gates)


def test_script_help_runs_directly_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/analyze_oof_disagreements.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--case-csv" in result.stdout
