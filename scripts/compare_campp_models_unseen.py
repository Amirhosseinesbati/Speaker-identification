"""
Leak-free sub-evaluation of the two CAM++ checkpoints.

The full reference-val comparison (compare_campp_models.py) is biased toward
the NEW model because 61% of reference-val files are exact duplicates of its
kfold train files.  This script evaluates BOTH models on the subset of
reference-val files that the new model truly never saw in training
(reference-val files NOT in the kfold fold-0 train = 347 files), which is a
symmetric comparison: neither model has seen those exact files, and both have
seen ~3-4 other files of the same speakers.

Also reports the duplicated subset (544 files) to quantify the leak's effect.

Usage:
    uv run --no-sync python scripts/compare_campp_models_unseen.py
"""

from __future__ import annotations

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
    prepare_clean_split, SpeakerDataset,
)
from src.model_factory import create_model_from_config  # noqa: E402
from src.metrics import macro_f1_score  # noqa: E402
from submission.inference import centroid_probs_matrix  # noqa: E402

DATA = ROOT / "data" / "processed"
LABELS_PATH = DATA / "audio_wav_labels.csv"
AUDIO_DIR = DATA / "audio_wav"
PROCESSED_LABELS = DATA / "cleaned_labels.csv"

NUM_CLASSES = 447
OLD_CKPT = ROOT / "checkpoints" / "campp_best.pt"
NEW_CKPT = ROOT / "checkpoints" / "campp_best (2).pt"
SUBMITTED_DECISION = {"alpha": 0.2, "kappa": 24.0, "tau": 0.0,
                      "lambda_unknown": 1.05, "temperature": 1.0}


def reference_split():
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=str(LABELS_PATH), audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1, unknown_val_ratio=0.2, random_seed=42,
    )
    return train_df, val_df, class_map


def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    num_known = config.get("model", {}).get(
        "competition_num_known", len(ck["class_map"]) - 1)
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, config, ck["class_map"]


@torch.inference_mode()
def dump_val(model, df, config, device) -> tuple:
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
        p, e = model.predict_proba_and_embed(windows.to(device), temperature=1.0)
        probs[i] = p.cpu().numpy()
        embs[i] = e.cpu().numpy()
        labels[i] = int(lab.item())
    return probs, embs, labels


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
    return macro_f1_score(labels, pred, num_classes=NUM_CLASSES)


def head_only(probs, labels):
    return macro_f1_score(labels, probs.argmax(axis=1).astype(np.int64),
                          num_classes=NUM_CLASSES)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  Leak-free sub-evaluation (unseen + duplicated subsets)")
    print("=" * 72)
    print(f"  Device: {device}")

    train_df, val_df, class_map = reference_split()

    # kfold fold-0 train files -> the new model's training set
    def _kfold_train_files():
        tr, _, _ = prepare_clean_split(
            labels_path=str(LABELS_PATH), audio_dir=str(AUDIO_DIR),
            processed_labels=str(PROCESSED_LABELS),
            val_per_known=1, unknown_val_ratio=0.2, random_seed=42,
            split_scheme="kfold", fold=0, folds=3,
        )
        return set(tr["audio_file"])

    kft = _kfold_train_files()
    print(f"  kfold fold-0 train files: {len(kft):,}")

    val_files = val_df["audio_file"].values
    val_labels = val_df["label"].values.astype(np.int64)
    unseen_mask = np.array([f not in kft for f in val_files])
    dup_mask = ~unseen_mask
    print(f"  Unseen subset: {int(unseen_mask.sum()):,} files "
          f"(known={int((val_labels[unseen_mask] > 0).sum())}, "
          f"unknown={int((val_labels[unseen_mask] == 0).sum())})")
    print(f"  Duplicated subset: {int(dup_mask.sum()):,} files")

    # Load the full-run tuned decision params if available
    try:
        full = json.loads((DATA / "campp_model_comparison.json").read_text())
        tuned_params = {k: full[k]["best_params"] for k in full}
        print("  Tuned params loaded from campp_model_comparison.json")
    except Exception:
        tuned_params = {"old": SUBMITTED_DECISION, "new": SUBMITTED_DECISION}
        print("  ⚠ No tuned params found — using submitted params.")

    report = {}
    for name, ckpt_path in [("old", OLD_CKPT), ("new", NEW_CKPT)]:
        print(f"\n{'─' * 72}\n  [{name}] {ckpt_path.name}\n{'─' * 72}")
        model, config, ck_class_map = load_model(ckpt_path, device)

        # Align train labels to this checkpoint's class map
        tr = train_df.copy()
        tr["label"] = tr["speaker_id"].map(ck_class_map).astype(int)
        tr = tr[tr["label"] > 0].reset_index(drop=True)

        audio_cfg = config["audio"]
        num_known = config.get("model", {}).get("competition_num_known", 446)

        # Centroids on the reference train split (known speakers only)
        from torch.utils.data import DataLoader
        ds = SpeakerDataset(
            tr, str(AUDIO_DIR), sample_rate=audio_cfg["sample_rate"],
            duration_seconds=audio_cfg["duration_seconds"], augment=False,
            num_train_windows=audio_cfg.get("num_train_windows", 1),
            eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
            max_eval_windows=audio_cfg.get("max_eval_windows", 8),
        )
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        embs_list, lab_list = [], []
        with torch.inference_mode():
            for windows, lab in tqdm(dl, desc="  centroids", leave=False):
                emb = model.embed(windows.to(device))
                embs_list.append(emb.cpu().numpy())
                lab_list.append(lab.numpy())
        embs_all = np.concatenate(embs_list, 0)
        lab_all = np.concatenate(lab_list, 0)
        speakers = np.arange(1, num_known + 1)
        centroids = np.zeros((num_known, embs_all.shape[1]), dtype=np.float32)
        for sid in speakers:
            m = lab_all == sid
            if m.sum() == 0:
                raise RuntimeError(f"No train embeddings for speaker {sid}")
            centroids[sid - 1] = embs_all[m].mean(axis=0)
        centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        centroids = centroids.astype(np.float32)
        speaker_ids = speakers.astype(np.int64)

        # Full-val dump (for subsetting)
        probs, embs, labels = dump_val(model, val_df, config, device)

        def subset_metrics(mask, tag):
            p, e, y = probs[mask], embs[mask], labels[mask]
            head = head_only(p, y)
            sub = evaluate_decision(p, e, centroids, speaker_ids, y,
                                    SUBMITTED_DECISION)
            tn = evaluate_decision(p, e, centroids, speaker_ids, y,
                                   tuned_params[name])
            print(f"  [{tag}] n={len(y):,} "
                  f"head-only={head:.4f} | submitted-params={sub:.4f} | "
                  f"tuned-params={tn:.4f}")
            return {"n": int(len(y)), "known": int((y > 0).sum()),
                    "unknown": int((y == 0).sum()),
                    "head_only": head, "submitted_params": sub,
                    "tuned_params": tn}

        report[name] = {
            "unseen": subset_metrics(unseen_mask, "unseen"),
            "duplicated": subset_metrics(dup_mask, "duplicated"),
            "full": subset_metrics(np.ones(len(val_labels), bool), "full"),
        }

    out = DATA / "campp_model_comparison_unseen.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  ✓ Saved to {out}")

    print("\n" + "=" * 72 + "\n  DECISIVE (unseen subset) SUMMARY\n" + "=" * 72)
    for name in ["old", "new"]:
        u = report[name]["unseen"]
        print(f"  [{name}] unseen: head-only {u['head_only']:.4f} | "
              f"submitted-params {u['submitted_params']:.4f} | "
              f"tuned-params {u['tuned_params']:.4f}")


if __name__ == "__main__":
    main()
