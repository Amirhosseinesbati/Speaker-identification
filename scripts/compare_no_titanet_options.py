"""Compare deployment-safe options without TitaNet and without retraining.

Evaluates three candidates on the common native 891-file validation split:
1. current CAM++ only;
2. historical ``campp_best (5).pt`` only;
3. CAM++ + ECAPA + ERes2Net (TitaNet excluded).

The historical checkpoint is forwarded once and receives fresh *known-speaker*
centroids from the native training partition. No trainable model is updated.
The script never edits production decision/fusion JSON files.
"""

from __future__ import annotations

import json
import sys
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.centroid_baseline import build_checkpoint_centroids  # noqa: E402
from src.cli_utils import setup_utf8_stdio  # noqa: E402
from src.decision_engine import dump_val_checkpoint  # noqa: E402
from src.metrics import macro_f1_score  # noqa: E402
from submission.inference import centroid_probs_matrix, _collapse_centroid_probs  # noqa: E402

setup_utf8_stdio()
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "generated" / "no_titanet_comparison.json"
OLD = ROOT / "checkpoints" / "modelrigestry" / "campp_best (5).pt"


def key(path: Path) -> str:
    return path.name.replace("_best.pt", "")


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_old(device: torch.device) -> str:
    name = key(OLD)
    probs_path, emb_path = DATA / f"val_probs_{name}.npy", DATA / f"val_emb_{name}.npy"
    if not (probs_path.exists() and emb_path.exists()):
        print("[old] forwarding historical checkpoint on native validation split")
        dump_val_checkpoint(str(OLD), device, use_cluster_map=False)
    cpath = DATA / f"centroids_{name}.npz"
    if not cpath.exists():
        print("[old] building known-speaker centroids (inference only)")
        built = build_checkpoint_centroids(str(OLD), device, batch_size=32)
        np.savez(cpath, centroids=built["centroids"], speaker_ids=built["speaker_ids"])
    return name


def load_model_arrays(name: str, add_unknown_clusters: bool) -> dict:
    probs = np.load(DATA / f"val_probs_{name}.npy").astype(np.float64)
    emb = np.load(DATA / f"val_emb_{name}.npy").astype(np.float32)
    z = np.load(DATA / f"centroids_{name}.npz")
    centroids, ids = z["centroids"].astype(np.float32), z["speaker_ids"].astype(np.int64)
    if add_unknown_clusters:
        z_u = np.load(DATA / "centroids_unknown_campp.npz")
        u = z_u["centroids"].astype(np.float32)
        centroids = np.vstack([centroids, u])
        ids = np.concatenate([ids, np.arange(447, 447 + len(u), dtype=np.int64)])
    return {"name": name, "probs": probs, "emb": emb, "centroids": centroids, "ids": ids}


def centroid_for(model: dict, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    cols = int(model["ids"].max()) + 1
    probs, max_cos = centroid_probs_matrix(model["emb"], model["centroids"], model["ids"], cols, kappa)
    if cols > 447:
        probs = _collapse_centroid_probs(probs, 447)
    return probs, max_cos


def score(models: list[dict], weights: np.ndarray, labels: np.ndarray, params: dict,
          cache: dict[float, tuple[np.ndarray, np.ndarray]]) -> float:
    heads = np.tensordot(weights, np.stack([m["probs"] for m in models]), axes=(0, 0))
    cps, maxes = cache[params["kappa"]]
    centroid = np.tensordot(weights, np.stack(cps), axes=(0, 0))
    max_cos = np.tensordot(weights, np.stack(maxes), axes=(0, 0))
    fused = params["alpha"] * heads + (1 - params["alpha"]) * centroid
    fused[:, 0] *= params["lambda_unknown"]
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    pred = fused.argmax(axis=1)
    pred[max_cos < params["tau"]] = 0
    return float(macro_f1_score(labels, pred.astype(np.int64), 447))


def weight_grid(n: int, step: float = 0.1):
    units = int(round(1 / step))
    if n == 1:
        yield np.array([1.0])
        return
    for a in product(range(units + 1), repeat=n - 1):
        rest = units - sum(a)
        if rest >= 0:
            yield np.array([*a, rest], dtype=np.float64) / units


def tune(models: list[dict], labels: np.ndarray) -> dict:
    # Coarse, reproducible alternating optimisation.  A complete Cartesian
    # product would require hundreds of thousands of Macro-F1 evaluations;
    # this examines every 0.1 weight combination and repeatedly retunes every
    # decision knob, making it practical on the local machine.
    grids = {
        "alpha": np.arange(0, 1.01, 0.1),
        "kappa": [4.0, 8.0, 12.0, 16.0, 24.0, 32.0],
        "tau": [0.0, 0.1, 0.2, 0.3, 0.4],
        "lambda_unknown": np.arange(0.5, 1.51, 0.1),
    }
    cache = {k: tuple(np.stack(x, axis=0) for x in zip(*(centroid_for(m, k) for m in models)))
             for k in grids["kappa"]}
    if len(models) == 1:
        # Exact search for a single model (3,630 combinations), rather than
        # relying on coordinate descent in a mildly non-convex objective.
        weights = np.array([1.0])
        best = {"score": -1.0}
        for kappa in grids["kappa"]:
            for alpha in grids["alpha"]:
                for lam in grids["lambda_unknown"]:
                    for tau in grids["tau"]:
                        params = {"alpha": float(alpha), "kappa": float(kappa),
                                  "tau": float(tau), "lambda_unknown": float(lam)}
                        value = score(models, weights, labels, params, cache)
                        if value > best["score"]:
                            best = {"score": value, "weights": [1.0], "params": params}
        return best
    weights = np.ones(len(models), dtype=np.float64) / len(models)
    params = {"alpha": 0.3, "kappa": 16.0, "tau": 0.0, "lambda_unknown": 0.65}
    best_score = score(models, weights, labels, params, cache)
    for _ in range(3):
        for candidate in weight_grid(len(models)):
            value = score(models, candidate, labels, params, cache)
            if value > best_score:
                best_score, weights = value, candidate
        for name, values in grids.items():
            for value in values:
                candidate = dict(params)
                candidate[name] = float(value)
                trial = score(models, weights, labels, candidate, cache)
                if trial > best_score:
                    best_score, params = trial, candidate
    return {"score": best_score, "weights": weights.tolist(), "params": params}


def main(skip_old: bool = False) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    labels = np.load(DATA / "val_labels.npy").astype(np.int64)
    current = load_model_arrays("campp", add_unknown_clusters=True)
    ecapa = load_model_arrays("ecapa", add_unknown_clusters=False)
    eres = load_model_arrays("eres2net", add_unknown_clusters=False)
    if not (len(labels) == len(current["probs"]) == len(ecapa["probs"]) == len(eres["probs"])):
        raise RuntimeError("Validation artifacts are not aligned; refusing comparison.")
    cases = {
        "current_campp_only": [current],
        "campp_ecapa_eres2net_no_titanet": [current, ecapa, eres],
    }
    if not skip_old:
        old_name = load_old(device)
        old = load_model_arrays(old_name, add_unknown_clusters=False)
        if len(old["probs"]) != len(labels):
            raise RuntimeError("Historical validation artifacts are not aligned; refusing comparison.")
        cases["historical_campp_5_only"] = [old]
    results = {}
    for case, models in cases.items():
        print(f"\n[{case}] tuning")
        best = tune(models, labels)
        best["models"] = [m["name"] for m in models]
        results[case] = best
        print(json.dumps(best, ensure_ascii=False))
    output = {
        "scope": "No checkpoint was trained. Historical CAM++ only received inference and train-split centroid extraction.",
        "validation": {"n": int(len(labels)), "known": int((labels > 0).sum()), "unknown": int((labels == 0).sum())},
        "search": "Alternating grid (3 rounds): every model-weight combination at step=0.1; alpha/lambda step=0.1; kappa={4,8,12,16,24,32}; tau={0,.1,.2,.3,.4}.",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-old", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(skip_old=args.skip_old))
