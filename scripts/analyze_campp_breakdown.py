"""
Per-class breakdown (known accuracy / unknown recall) for the two CAM++
checkpoints on the reference val split, split into clean-unseen and
leak-duplicated subsets.

Dumps val probs once per model (full 891 files) and saves them to
data/processed/ so the analysis can be re-run without recompute.
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

from src.data_pipeline import prepare_clean_split, SpeakerDataset  # noqa: E402
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

SUBMITTED = {"alpha": 0.2, "kappa": 24.0, "tau": 0.0, "lambda_unknown": 1.05}


@torch.inference_mode()
def dump_val(ckpt_path, df, config, device, save_tag):
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
    for i in tqdm(range(len(ds)), desc=f"  [{save_tag}] val", leave=False):
        windows, lab = ds[i]
        p, e = model.predict_proba_and_embed(windows.to(device), temperature=1.0)
        probs[i] = p.cpu().numpy()
        embs[i] = e.cpu().numpy()
        labels[i] = int(lab.item())
    np.save(DATA / f"val_probs_{save_tag}.npy", probs)
    np.save(DATA / f"val_emb_{save_tag}.npy", embs)
    np.save(DATA / f"val_labels_{save_tag}.npy", labels)
    return probs, embs, labels


def evaluate(probs, emb, centroids, speaker_ids, labels, params):
    cp, mc = centroid_probs_matrix(emb, centroids, speaker_ids,
                                   NUM_CLASSES, params["kappa"])
    fused = params["alpha"] * probs + (1.0 - params["alpha"]) * cp
    fused = fused.copy()
    fused[:, 0] *= params["lambda_unknown"]
    fused /= (fused.sum(axis=1, keepdims=True) + 1e-12)
    pred = fused.argmax(axis=1).astype(np.int64)
    pred[mc < params["tau"]] = 0
    return pred


def breakdown(pred, y):
    known = y > 0
    unknown = y == 0
    known_acc = float((pred[known] == y[known]).mean()) if known.any() else None
    unknown_rec = float((pred[unknown] == 0).mean()) if unknown.any() else None
    return known_acc, unknown_rec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Reference split + kfold train file set
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=str(LABELS_PATH), audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1, unknown_val_ratio=0.2, random_seed=42)
    tr, _, _ = prepare_clean_split(
        labels_path=str(LABELS_PATH), audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1, unknown_val_ratio=0.2, random_seed=42,
        split_scheme="kfold", fold=0, folds=3)
    kft = set(tr["audio_file"])
    val_files = val_df["audio_file"].values
    val_labels = val_df["label"].values.astype(np.int64)
    unseen = np.array([f not in kft for f in val_files])
    dup = ~unseen
    print(f"unseen={int(unseen.sum())} duplicated={int(dup.sum())}")

    report = {}
    for name, ckpt_path in [("old", OLD_CKPT), ("new", NEW_CKPT)]:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ck["config"]
        num_known = config.get("model", {}).get("competition_num_known", 446)
        global model
        model = create_model_from_config(config, num_known_speakers=num_known)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()

        probs, embs, labels = dump_val(ckpt_path, val_df, config, device, name)
        assert (labels == val_labels).all()

        # centroids on reference train split (known only)
        trc = train_df.copy()
        trc["label"] = trc["speaker_id"].map(ck["class_map"]).astype(int)
        trc = trc[trc["label"] > 0].reset_index(drop=True)
        from torch.utils.data import DataLoader
        audio_cfg = config["audio"]
        ds = SpeakerDataset(
            trc, str(AUDIO_DIR), sample_rate=audio_cfg["sample_rate"],
            duration_seconds=audio_cfg["duration_seconds"], augment=False,
            num_train_windows=audio_cfg.get("num_train_windows", 1),
            eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
            max_eval_windows=audio_cfg.get("max_eval_windows", 8))
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        embs_list, lab_list = [], []
        with torch.inference_mode():
            for w, l in tqdm(dl, desc=f"  [{name}] centroids", leave=False):
                e = model.embed(w.to(device))
                embs_list.append(e.cpu().numpy()); lab_list.append(l.numpy())
        embs_all = np.concatenate(embs_list, 0)
        lab_all = np.concatenate(lab_list, 0)
        spk = np.arange(1, num_known + 1)
        cent = np.zeros((num_known, embs_all.shape[1]), dtype=np.float32)
        for sid in spk:
            m = lab_all == sid
            cent[sid - 1] = embs_all[m].mean(axis=0)
        cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)
        cent = cent.astype(np.float32)

        if name == "old":
            params = SUBMITTED
        else:
            params = {"alpha": 0.4, "kappa": 12.0, "tau": 0.0,
                      "lambda_unknown": 0.9}

        pred = evaluate(probs, embs, cent, spk.astype(np.int64),
                        val_labels, params)
        mf1 = macro_f1_score(val_labels, pred, num_classes=NUM_CLASSES)

        report[name] = {"params": params, "full_macro_f1": mf1}
        for tag, mask in [("unseen", unseen), ("duplicated", dup), ("full",
                                                                    np.ones(len(val_labels), bool))]:
            ka, ur = breakdown(pred[mask], val_labels[mask])
            sub_mf1 = macro_f1_score(val_labels[mask], pred[mask],
                                     num_classes=NUM_CLASSES)
            report[name][tag] = {
                "macro_f1": sub_mf1,
                "known_acc": ka,
                "unknown_recall": ur,
            }
            print(f"  [{name}] {tag}: macro-F1={sub_mf1:.4f} "
                  f"known_acc={ka:.4f} unknown_recall={ur:.4f}")

    out = DATA / "campp_breakdown.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  ✓ Saved to {out}")


if __name__ == "__main__":
    main()
