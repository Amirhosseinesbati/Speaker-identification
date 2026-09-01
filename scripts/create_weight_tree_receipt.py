"""Create a deterministic SHA256 receipt for an offline model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RUNTIME_REQUIRED = ("config.json", "preprocessor_config.json")
WEIGHT_ALTERNATIVES = ("model.safetensors", "pytorch_model.bin")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_weight_tree_receipt(
    weights_dir: Path,
    source_model_id: str,
    git_commit: str,
    expected_hidden_size: int | None = None,
    expected_hidden_layers: int | None = None,
) -> dict[str, Any]:
    root = weights_dir.resolve()
    if not root.is_dir():
        raise RuntimeError(f"weight directory does not exist: {root}")
    missing = [name for name in RUNTIME_REQUIRED if not (root / name).is_file()]
    weight_files = [name for name in WEIGHT_ALTERNATIVES if (root / name).is_file()]
    if missing or len(weight_files) != 1:
        raise RuntimeError(
            "incomplete or ambiguous runtime payload: "
            f"missing={missing}, weight_files={weight_files}"
        )

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    hidden_size = int(config.get("hidden_size", -1))
    hidden_layers = int(config.get("num_hidden_layers", -1))
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise RuntimeError(
            f"hidden_size mismatch: {hidden_size} != {expected_hidden_size}"
        )
    if expected_hidden_layers is not None and hidden_layers != expected_hidden_layers:
        raise RuntimeError(
            f"num_hidden_layers mismatch: {hidden_layers} != {expected_hidden_layers}"
        )

    runtime_names = sorted((*RUNTIME_REQUIRED, *weight_files))
    files: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    for relative in runtime_names:
        path = root / relative
        size = path.stat().st_size
        digest = sha256_file(path)
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
        tree_digest.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))

    return {
        "schema_version": 1,
        "source_model_id": source_model_id,
        "git_commit": git_commit,
        "weights_dir": weights_dir.as_posix(),
        "runtime_format": weight_files[0],
        "config": {
            "model_type": config.get("model_type"),
            "hidden_size": hidden_size,
            "num_hidden_layers": hidden_layers,
            "weighted_sum_state_count": hidden_layers + 1,
        },
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "tree_sha256": tree_digest.hexdigest(),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--expected-hidden-size", type=int)
    parser.add_argument("--expected-hidden-layers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = create_weight_tree_receipt(
        args.weights_dir,
        args.source_model_id,
        args.git_commit,
        args.expected_hidden_size,
        args.expected_hidden_layers,
    )
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
