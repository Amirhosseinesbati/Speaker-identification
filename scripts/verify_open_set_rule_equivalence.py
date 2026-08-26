"""Verify packaged top-k inference against the offline decision-evidence cache."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.inference import score_ensemble  # noqa: E402


def load(tag: str) -> dict[str, np.ndarray]:
    with np.load(
        ROOT / "data" / "processed" / "decision_evidence" / f"{tag}.npz",
        allow_pickle=True,
    ) as data:
        return {key: data[key] for key in data.files}


def expected_labels(
    no_proto: dict[str, np.ndarray],
    metric_only: dict[str, np.ndarray],
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    speaker = (
        0.6 * no_proto["speaker_probs"].astype(np.float64)
        + 0.4 * metric_only["speaker_probs"].astype(np.float64)
    )
    known = speaker[:, :446]
    tail = speaker[:, 446:]
    rule = config["open_set_rule"]
    top_k = int(rule["top_k"])
    top_mean = np.partition(tail, tail.shape[1] - top_k, axis=1)[:, -top_k:].mean(axis=1)
    ood = np.clip(no_proto["ood_prob"].astype(np.float64), 1e-9, 1 - 1e-9)
    score = (
        np.log(np.clip(top_mean, 1e-9, 1.0))
        - np.log(np.clip(known.max(axis=1), 1e-9, 1.0))
        + float(rule["ood_weight"]) * np.log(ood / (1.0 - ood))
    )
    labels = known.argmax(axis=1).astype(np.int64) + 1
    labels[score > float(rule["threshold"])] = 0
    return labels, score


def main() -> int:
    no_proto = load("no_proto")
    metric_only = load("metric_only")
    decision = json.loads(
        (ROOT / "submission" / "decision_config.json").read_text(encoding="utf-8")
    )["decision_params"]
    expected, score = expected_labels(no_proto, metric_only, decision)
    threshold = float(decision["open_set_rule"]["threshold"])

    # Exercise both sides of the threshold, especially boundary-adjacent files.
    near = np.argsort(np.abs(score - threshold))[:16]
    far_known = np.argsort(score)[:4]
    far_unknown = np.argsort(score)[-4:]
    selected = np.unique(np.concatenate([near, far_known, far_unknown]))
    audio_root = ROOT / "data" / "processed" / "audio_wav"

    with tempfile.TemporaryDirectory(prefix="open_set_equivalence_") as temporary:
        data_dir = Path(temporary) / "audio"
        data_dir.mkdir()
        for index in selected:
            name = str(no_proto["audio_file"][index])
            shutil.copy2(audio_root / name, data_dir / name)
        result = score_ensemble(
            data_dir=str(data_dir),
            checkpoint_path=[
                str(ROOT / "submission" / "checkpoints" / "campp_no_proto_best.pt"),
                str(ROOT / "submission" / "checkpoints" / "campp_metric_only_best.pt"),
            ],
            fusion_method="weighted_average",
            fusion_weights=[0.6, 0.4],
            centroids=None,
            decision_params=decision,
        )

    expected_by_file = {
        str(no_proto["audio_file"][index]): int(expected[index]) for index in selected
    }
    mismatches = []
    selected_score = {
        str(no_proto["audio_file"][index]): float(score[index]) for index in selected
    }
    for position, (path, actual) in enumerate(zip(result["files"], result["labels"])):
        wanted = expected_by_file[path.name]
        if int(actual) != wanted:
            mismatches.append({
                "audio_file": path.name,
                "expected": wanted,
                "actual": int(actual),
                "offline_score": selected_score[path.name],
                "inference_score": float(result["open_set_score"][position]),
                "score_delta": float(result["open_set_score"][position]) - selected_score[path.name],
            })
    print(json.dumps({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checked": int(len(selected)),
        "threshold": threshold,
        "mismatches": mismatches,
    }, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
