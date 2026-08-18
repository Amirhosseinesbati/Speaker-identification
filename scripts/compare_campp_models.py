"""
Generic side-by-side comparison of N trained checkpoints on ONE common
reference split — head-only + tuned decision-layer Macro-F1.

(Generalises the original two-CAM++-model comparison to any number of
checkpoints, from any encoders.)

Why a common reference split: every checkpoint embeds its own
``data.split`` (scheme/fold/seed) in its config, so the ``val_macro_f1``
stored inside a checkpoint is measured on ITS OWN validation set and is NOT
comparable across checkpoints (e.g. "single" seed-42 vs "kfold" fold-0).
This script evaluates every model with the EXACT submission decision layer
(``predict_proba_and_embed`` + cosine centroids + alpha/kappa/tau/lambda gate)
on one shared split and reports:

  - head-only Macro-F1 (plain argmax of the 447-way head probs)
  - best decision Macro-F1 (alpha/kappa/tau/lambda tuned per model on val)
  - applied decision params Macro-F1 (e.g. the currently-submitted
    decision_config.json, applied unchanged to every model) + unknown recall
  - ``--unseen``: the leak-free subset — reference-val files that ANY compared
    model trained on (name in its own embedded train split) are excluded, so
    a kfold-trained model cannot win by having memorised reference-val files.

Usage:
    uv run --no-sync python scripts/compare_campp_models.py \
        --checkpoints checkpoints/campp_best.pt "checkpoints/campp_best (2).pt" \
        [--split single|kfold --fold 0 --folds 3 --seed 42] \
        [--unseen] \
        [--decision-json data/processed/decision_config.json] \
        [--no-decision-json]

``--decision-json`` defaults to ``data/processed/decision_config.json`` when
that file exists (auto-detect); pass ``--no-decision-json`` to skip applying
it. Results are saved to ``data/processed/model_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.data_pipeline import (  # noqa: E402
    prepare_clean_split, SpeakerDataset, split_args_from_config,
)
from src.model_factory import create_model_from_config  # noqa: E402
from src.metrics import macro_f1_score  # noqa: E402
from submission.inference import centroid_probs_matrix  # noqa: E402

DATA = ROOT / "data" / "processed"
LABELS_PATH = DATA / "audio_wav_labels.csv"
AUDIO_DIR = DATA / "audio_wav"
PROCESSED_LABELS = DATA / "cleaned_labels.csv"
DEFAULT_DECISION_JSON = DATA / "decision_config.json"
OUT_JSON = DATA / "model_comparison.json"

NUM_CLASSES = 447


# ────────────────────────────────────────────────────────────
#  Reference split (common for every model)
# ────────────────────────────────────────────────────────────
def get_reference_split(args):
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=str(LABELS_PATH),
        audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1,
        unknown_val_ratio=0.2,
        random_seed=args.seed,
        split_scheme=args.split,
        fold=args.fold,
        folds=args.folds,
    )
    return train_df, val_df, class_map


# ────────────────────────────────────────────────────────────
#  Model loading (per-checkpoint embedded config)
# ────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    class_map = ck["class_map"]
    num_known = config.get("model", {}).get(
        "competition_num_known", len(class_map) - 1)
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, config, class_map


# ────────────────────────────────────────────────────────────
#  Val probs + embeddings (same forward as submission)
# ────────────────────────────────────────────────────────────
@torch.inference_mode()
def dump_val(model, df, config, device, temperature: float = 1.0) -> tuple:
    audio_cfg = config["audio"]
    ds = SpeakerDataset(
        df, str(AUDIO_DIR), sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    emb_dim = getattr(model.head_speaker, "embedding_dim", 192)
    probs = np.zeros((len(ds), NUM_CLASSES), dtype=np.float32)
    embs = np.zeros((len(ds), emb_dim), dtype=np.float32)
    labels = np.zeros((len(ds),), dtype=np.int64)
    for i in tqdm(range(len(ds)), desc="  val forward", leave=False):
        windows, lab = ds[i]
        p, e = model.predict_proba_and_embed(windows.to(device),
                                             temperature=temperature)
        probs[i] = p.cpu().numpy()
        embs[i] = e.cpu().numpy()
        labels[i] = int(lab.item())
    return probs, embs, labels


# ────────────────────────────────────────────────────────────
#  Centroids on the reference train split (same as submission)
# ────────────────────────────────────────────────────────────
@torch.inference_mode()
def build_centroids(model, train_df, config, device,
                    checkpoint_class_map: dict) -> tuple:
    """Centroids for the reference train split, rows aligned to the
    checkpoint's class_map (global id 1..num_known)."""
    from torch.utils.data import DataLoader

    audio_cfg = config["audio"]
    num_known = config.get("model", {}).get("competition_num_known", 446)

    # Align train labels to the checkpoint's class_map (speaker_id -> global id)
    train_df = train_df.copy()
    train_df["label"] = train_df["speaker_id"].map(checkpoint_class_map).astype(int)
    # Only known speakers contribute to centroids
    train_df = train_df[train_df["label"] > 0].reset_index(drop=True)

    ds = SpeakerDataset(
        train_df, str(AUDIO_DIR), sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    embs, labels = [], []
    for windows, lab in tqdm(dl, desc="  train embeddings", leave=False):
        emb = model.embed(windows.to(device))
        embs.append(emb.cpu().numpy())
        labels.append(lab.numpy())
    embs = np.concatenate(embs, axis=0)
    labels = np.concatenate(labels, axis=0)

    known_mask = labels > 0
    known_embs = embs[known_mask]
    known_ids = labels[known_mask]
    D = embs.shape[1]

    speakers = np.arange(1, num_known + 1)
    centroids = np.zeros((num_known, D), dtype=np.float32)
    for sid in speakers:
        m = known_ids == sid
        if m.sum() == 0:
            raise RuntimeError(f"No train embeddings for speaker {sid}")
        centroids[sid - 1] = known_embs[m].mean(axis=0)
    centroids = centroids / (np.linalg.norm(centroids, axis=1,
                                            keepdims=True) + 1e-12)
    return centroids.astype(np.float32), speakers.astype(np.int64), int(len(known_embs))


# ────────────────────────────────────────────────────────────
#  Decision-layer evaluation (exact math of submission inference)
# ────────────────────────────────────────────────────────────
def evaluate_decision(probs, emb, centroids, speaker_ids, labels, params):
    alpha = float(params["alpha"])
    kappa = float(params["kappa"])
    tau = float(params["tau"])
    lambda_unknown = float(params["lambda_unknown"])

    cp, mc = centroid_probs_matrix(emb, centroids, speaker_ids,
                                   NUM_CLASSES, kappa)
    fused = alpha * probs + (1.0 - alpha) * cp
    fused = fused.copy()
    fused[:, 0] *= lambda_unknown
    fused /= (fused.sum(axis=1, keepdims=True) + 1e-12)
    pred = fused.argmax(axis=1).astype(np.int64)
    pred[mc < tau] = 0
    return macro_f1_score(labels, pred, num_classes=NUM_CLASSES), pred


def tune_decision(probs, emb, centroids, speaker_ids, labels):
    """Coordinate-descent sweep (mirrors src.decision_engine.tune_decision_bundle)."""
    grids = {
        "alpha": np.round(np.arange(0.0, 1.001, 0.05), 2),
        "kappa": np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
        "tau": np.round(np.arange(0.0, 0.60, 0.02), 3),
        "lambda_unknown": np.round(np.arange(0.5, 1.601, 0.05), 2),
    }
    order = ["alpha", "kappa", "tau", "lambda_unknown"]
    best = {"alpha": 1.0, "kappa": 8.0, "tau": 0.0, "lambda_unknown": 1.0}
    best_score, _ = evaluate_decision(probs, emb, centroids, speaker_ids,
                                      labels, best)
    baseline = best_score
    for rnd in range(3):
        for key in order:
            improved = False
            for v in grids[key]:
                cand = dict(best)
                cand[key] = float(v)
                score, _ = evaluate_decision(probs, emb, centroids, speaker_ids,
                                             labels, cand)
                if score > best_score:
                    best_score = score
                    best = cand
                    improved = True
            if improved:
                print(f"    [round {rnd + 1}] {key} -> {best[key]} "
                      f"(Macro-F1 {best_score:.4f})")
    return best, best_score, baseline


def head_only_macro_f1(probs, labels):
    pred = probs.argmax(axis=1).astype(np.int64)
    return macro_f1_score(labels, pred, num_classes=NUM_CLASSES)


def unknown_recall(preds, labels):
    """Recall of the unknown class: TP_unknown / n_unknown (NaN if none)."""
    n_u = int((labels == 0).sum())
    if n_u == 0:
        return float("nan")
    tp = int(((preds == 0) & (labels == 0)).sum())
    return tp / n_u


# ────────────────────────────────────────────────────────────
#  Leak-free subset (--unseen)
# ────────────────────────────────────────────────────────────
def checkpoint_train_files(config: dict) -> set:
    """The set of audio files this checkpoint trained on, from ITS embedded
    split (scheme/fold/folds/seed) — so kfold-trained models are handled too."""
    _, train_df, _ = prepare_clean_split(
        labels_path=str(LABELS_PATH),
        audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1,
        unknown_val_ratio=0.2,
        **split_args_from_config(config),
    )
    return set(train_df["audio_file"])


def load_decision_params(path: Path) -> dict | None:
    """Read a decision-params dict: accepts a flat dict or a JSON with a
    ``decision_params`` key (decision_config.json layout)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("decision_params", data)


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare N trained checkpoints on one common reference "
                    "split (head-only + decision-layer Macro-F1).")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Trained checkpoint paths (2+ recommended).")
    parser.add_argument("--split", default="single", choices=["single", "kfold"],
                        help="Reference split scheme (default: single).")
    parser.add_argument("--fold", type=int, default=0,
                        help="Reference kfold fold index (default: 0).")
    parser.add_argument("--folds", type=int, default=3,
                        help="Reference kfold fold count (default: 3).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reference split RNG seed (default: 42).")
    parser.add_argument("--unseen", action="store_true",
                        help="Also report the leak-free subset (excludes "
                             "reference-val files any compared model trained on).")
    parser.add_argument("--decision-json", default=None, type=Path,
                        help="Decision params to apply unchanged to every model. "
                             "Default: data/processed/decision_config.json when it "
                             "exists (auto-detect).")
    parser.add_argument("--no-decision-json", action="store_true",
                        help="Skip applying any submitted decision params.")
    args = parser.parse_args()

    # ── Fail fast on missing/typo'd checkpoint paths ──
    # cmd.exe / PowerShell / bash all split UNQUOTED arguments on spaces, so a
    # path like "checkpoints/campp_best (3).pt" arrives as two args
    # ("checkpoints/campp_best", "(3).pt") and torch.load fails on the first.
    # List what actually exists so the fix (quoting) is obvious.
    missing = [c for c in args.checkpoints if not Path(c).exists()]
    if missing:
        ckpt_dir = ROOT / "checkpoints"
        # `*_best.pt` misses names like "campp_best (2).pt" — list every
        # top-level .pt so all candidates are visible.
        found = sorted(p.name for p in ckpt_dir.glob("*.pt")) \
            if ckpt_dir.is_dir() else []
        parser.error(
            "checkpoint file(s) not found:\n"
            + "\n".join(f"    {m}" for m in missing)
            + "\n  Tip: in cmd/PowerShell, quote every path that contains "
              "spaces or parentheses, e.g.\n"
            + '      --checkpoints "checkpoints/campp_best (3).pt" '
              "checkpoints/campp_best.pt\n"
            + "  Checkpoints found in checkpoints/:\n"
            + "\n".join(f"    {f}" for f in found or ["<none>"])
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  Model comparison on a common reference split")
    print("=" * 72)
    print(f"  Device: {device}")
    print(f"  Checkpoints ({len(args.checkpoints)}):")
    for c in args.checkpoints:
        print(f"    - {c}")

    # ── Reference split ──
    train_df, val_df, class_map = get_reference_split(args)
    val_labels = val_df["label"].values.astype(np.int64)
    print(f"\n  Reference split: {args.split}"
          f"{f' fold {args.fold}/{args.folds}' if args.split == 'kfold' else ''}"
          f" seed={args.seed}")
    print(f"  Val files: {len(val_df):,} "
          f"(known={int((val_labels > 0).sum())}, "
          f"unknown={int((val_labels == 0).sum())})")
    print(f"  Train files: {len(train_df):,} "
          f"(known={int((train_df['label'] > 0).sum())})")

    # Class-map sanity: every checkpoint's embedded mapping must equal the
    # reference split's (all derive from the same labels file).
    ckpts = [Path(c) for c in args.checkpoints]
    names = [p.name for p in ckpts]
    for name, ckpt_path in zip(names, ckpts):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck["class_map"] == class_map, (
            f"[{name}] embedded class_map differs from the reference split's! "
            f"Check label data drift."
        )
    print("  ✓ Checkpoint class_maps match the reference split.")

    # ── Applied decision params (submitted config) ──
    decision_params = None
    decision_path = None
    if args.no_decision_json:
        print("  ⚠ Skipping applied decision params (--no-decision-json).")
    else:
        candidate = args.decision_json or DEFAULT_DECISION_JSON
        if candidate.exists():
            decision_params = load_decision_params(candidate)
            decision_path = candidate
            print(f"  ⚙  Applying decision params from {candidate.name}: "
                  f"α={decision_params.get('alpha')} κ={decision_params.get('kappa')} "
                  f"τ={decision_params.get('tau')} λ={decision_params.get('lambda_unknown')} "
                  f"T={decision_params.get('temperature')}")
        elif args.decision_json is not None:
            print(f"  ⚠ --decision-json not found: {candidate}")

    # ── Leak-free subset (files no compared model trained on) ──
    unseen_mask = None
    if args.unseen:
        print("\n  Computing each checkpoint's own train set (--unseen)...")
        seen = set()
        for name, ckpt_path in zip(names, ckpts):
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            files = checkpoint_train_files(ck.get("config", {}))
            seen |= files
            print(f"    [{name}] train files in its split: {len(files):,}")
        val_files = val_df["audio_file"].values
        unseen_mask = np.array([f not in seen for f in val_files])
        print(f"  Unseen subset: {int(unseen_mask.sum()):,} files "
              f"(known={int((val_labels[unseen_mask] > 0).sum())}, "
              f"unknown={int((val_labels[unseen_mask] == 0).sum())})")
        print(f"  Seen/duplicated subset: {int((~unseen_mask).sum()):,} files")

    # ── Per-model evaluation ──
    results = {}
    for name, ckpt_path in zip(names, ckpts):
        print(f"\n{'─' * 72}\n  [{name}] {ckpt_path.name}\n{'─' * 72}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, config, checkpoint_class_map = load_model(ckpt_path, device)

        # Val dump at T=1.0 (tuning + head-only use this).
        probs, embs, labels = dump_val(model, val_df, config, device)
        assert (labels == val_labels).all(), \
            f"[{name}] val order/labels mismatch with reference split!"
        head_f1 = head_only_macro_f1(probs, val_labels)

        # Centroids on the reference train split.
        centroids, speaker_ids, n_train = build_centroids(
            model, train_df, config, device, checkpoint_class_map)

        print(f"  Head-only Macro-F1:           {head_f1:.4f}")

        # Per-model tuned decision.
        print("  Tuning decision params...")
        best, best_score, baseline = tune_decision(
            probs, embs, centroids, speaker_ids, val_labels)
        print(f"  Baseline (plain head argmax): {baseline:.4f}")
        print(f"  Best decision Macro-F1:       {best_score:.4f}  params={best}")
        _, best_preds = evaluate_decision(probs, embs, centroids, speaker_ids,
                                          val_labels, best)
        print(f"  Unknown recall (best params):  {unknown_recall(best_preds, val_labels):.3f}")

        entry = {
            "checkpoint": str(ckpt_path),
            "head_only_macro_f1": head_f1,
            "baseline_macro_f1": baseline,
            "best_macro_f1": best_score,
            "best_params": {k: float(v) for k, v in best.items()},
            "unknown_recall_best_params": float(unknown_recall(best_preds, val_labels)),
            "n_train_files": n_train,
        }

        # Applied (submitted) decision params — unchanged for every model.
        if decision_params is not None:
            temp = float(decision_params.get("temperature", 1.0))
            if abs(temp - 1.0) > 1e-6:
                # Re-dump only when the submitted temperature != 1.0.
                probs_t, embs_t, _ = dump_val(model, val_df, config, device,
                                              temperature=temp)
            else:
                probs_t, embs_t = probs, embs
            sub_f1, sub_pred = evaluate_decision(
                probs_t, embs_t, centroids, speaker_ids, val_labels,
                decision_params)
            entry["applied_params_macro_f1"] = sub_f1
            entry["applied_params_unknown_recall"] = float(
                unknown_recall(sub_pred, val_labels))
            print(f"  Applied {decision_path.name} params: {sub_f1:.4f} "
                  f"(unknown recall {unknown_recall(sub_pred, val_labels):.3f})")

        # Leak-free subset numbers (params tuned on FULL val are applied).
        if unseen_mask is not None:
            sub_f1, sub_pred = evaluate_decision(
                probs, embs, centroids, speaker_ids, val_labels, best)
            unseen = {
                "n_files": int(unseen_mask.sum()),
                "head_only_macro_f1": head_only_macro_f1(probs[unseen_mask],
                                                         val_labels[unseen_mask]),
                "best_decision_macro_f1": macro_f1_score(
                    val_labels[unseen_mask], sub_pred[unseen_mask],
                    num_classes=NUM_CLASSES),
                "unknown_recall_best_params": float(unknown_recall(
                    sub_pred[unseen_mask], val_labels[unseen_mask])),
            }
            if decision_params is not None:
                app_f1, app_pred = evaluate_decision(
                    probs, embs, centroids, speaker_ids, val_labels,
                    decision_params)
                unseen["applied_params_macro_f1"] = macro_f1_score(
                    val_labels[unseen_mask], app_pred[unseen_mask],
                    num_classes=NUM_CLASSES)
            entry["unseen"] = unseen
            print(f"  Unseen subset ({unseen['n_files']} files): "
                  f"head {unseen['head_only_macro_f1']:.4f} | "
                  f"best-decision {unseen['best_decision_macro_f1']:.4f}")

        results[name] = entry

    # ── Summary table ──
    print(f"\n{'=' * 72}\n  SUMMARY (reference: {args.split}"
          f"{f' fold {args.fold}/{args.folds}' if args.split == 'kfold' else ''}"
          f" seed={args.seed}, {len(val_df):,} val files)\n{'=' * 72}")
    for name in names:
        r = results[name]
        parts = [f"[{name}] head {r['head_only_macro_f1']:.4f} | "
                 f"best-decision {r['best_macro_f1']:.4f}"]
        if "applied_params_macro_f1" in r:
            parts.append(f"applied {r['applied_params_macro_f1']:.4f}")
        if "unseen" in r:
            parts.append(f"unseen-head {r['unseen']['head_only_macro_f1']:.4f} | "
                         f"unseen-decision {r['unseen']['best_decision_macro_f1']:.4f}")
        print("  " + " | ".join(parts))

    output = {
        "reference_split": {
            "scheme": args.split, "fold": args.fold, "folds": args.folds,
            "seed": args.seed, "n_val": int(len(val_df)),
            "n_train": int(len(train_df)),
        },
        "decision_params_applied": {
            "path": str(decision_path) if decision_path else None,
            "params": decision_params,
        },
        "unseen_subset": (None if unseen_mask is None else {
            "n_files": int(unseen_mask.sum()),
            "n_seen": int((~unseen_mask).sum()),
        }),
        "models": results,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  ✓ Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
