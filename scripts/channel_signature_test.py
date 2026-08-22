"""Channel-signature experiments (consultant §0/§3).

Tests two hypotheses on the native-split val set with campp embeddings:

  H1 (embedding space): an empty/noisy val file's ArcFace embedding is closest
     to same-speaker TRAIN embeddings (i.e. the embedder already captures the
     recording-channel signature), or at least the same-speaker train file is
     in its top-k.
  H2 (noise-PSD space): 1-NN on the mean log-spectrum of silent frames
     (noise floor fingerprint) retrieves a same-speaker train file.

Outputs:
  data/processed/noise_psd_train.npz   (F,) train file list + (N,F) matrix
  data/processed/noise_psd_val.npz     (F,) val file list + (N,F) matrix
  console report per dirty file + aggregate hit rates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

AUDIO_DIR = ROOT / "data" / "processed" / "audio_wav"
LABELS = ROOT / "data" / "processed" / "audio_wav_labels.csv"
ERR_CSV = ROOT / "error_analysis_report.csv"
PER_FILE = ROOT / "data" / "processed" / "model_compare_legacy_vs_cluster" / "per_file_predictions.csv"
VAL_EMB = ROOT / "data" / "processed" / "val_emb_campp.npy"
TRAIN_EMB = ROOT / "data" / "processed" / "train_emb_campp.npy"
TRAIN_EMB_LAB = ROOT / "data" / "processed" / "train_emb_campp_labels.npy"
TRAIN_FILES = ROOT / "data" / "processed" / "train_emb_campp_files.json"
SPLIT = ROOT / "data" / "processed" / "split_report.json"

SR = 16000
FFT = 512
HOP = 160
FRAME = 512


def noise_psd(x: np.ndarray, n_fft: int = FFT, hop: int = HOP) -> np.ndarray:
    """Mean log-magnitude spectrum over the quietest ~12% of frames.

    Returns an L2-normalised (n_fft//2+1,) vector (the 'noise floor fingerprint').
    """
    n = (len(x) - n_fft) // hop + 1
    if n < 8:
        return np.zeros(n_fft // 2 + 1, dtype=np.float32)
    idx = np.arange(n)[:, None] * hop + np.arange(n_fft)[None, :]
    frames = x[idx]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    n_noise = max(1, int(n * 0.12))
    quiet = np.argsort(rms)[:n_noise]
    spec = np.abs(np.fft.rfft(frames[quiet] * np.hanning(n_fft), axis=1)) + 1e-12
    psd = np.log(spec).mean(axis=0).astype(np.float32)
    psd -= psd.mean()
    denom = np.linalg.norm(psd)
    if denom > 1e-9:
        psd /= denom
    return psd


def load_file_map() -> pd.DataFrame:
    return pd.read_csv(LABELS)


def main() -> int:
    lab = load_file_map()  # speaker_id, audio_file  (4529)
    err = pd.read_csv(ERR_CSV, encoding="utf-8-sig")
    per = pd.read_csv(PER_FILE)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    corrupted = set(split["corrupted_files"]["files"])

    val_files = set(err["audio_file"])
    # sanity: error CSV and per-file predictions cover the same files
    assert set(per["audio_file"]) == val_files, "val set mismatch"

    train_lab = lab[~lab["audio_file"].isin(val_files) &
                    ~lab["audio_file"].isin(corrupted)].reset_index(drop=True)
    print(f"  train pool: {len(train_lab)} files, val: {len(val_files)}")

    # ---------- H1: embedding-space nearest-neighbour ----------
    train_emb = np.load(TRAIN_EMB).astype(np.float32)          # (3568, 192)
    train_lab_ids = np.load(TRAIN_EMB_LAB).astype(np.int64)
    train_files = json.loads(TRAIN_FILES.read_text(encoding="utf-8"))
    assert len(train_files) == len(train_emb) == len(train_lab_ids)
    # speaker uuid per train file (from the label map)
    tf_map = {row["audio_file"]: row["speaker_id"] for _, row in train_lab.iterrows()}
    train_speakers = np.array([tf_map.get(f, "?") for f in train_files])

    val_emb = np.load(VAL_EMB).astype(np.float32)              # (891, 192)
    per_idx = {f: i for i, f in enumerate(per["audio_file"])}
    # align err rows -> val embedding rows
    emb_rows = [per_idx[f] for f in err["audio_file"]]
    val_emb_aligned = val_emb[emb_rows]

    # normalized cosine sim matrix (val x train)
    sims = val_emb_aligned @ train_emb.T                      # (891, 3568)
    nn = sims.argmax(axis=1)

    err2 = err.copy()
    err2["emb_nn_speaker"] = [train_speakers[i] for i in nn]
    err2["emb_nn_sim"] = sims[np.arange(len(err)), nn]
    err2["emb_nn_hit"] = err2["emb_nn_speaker"] == err2["true_speaker"]

    # rank of best same-speaker train file in embedding space
    ranks = []
    for i, f in enumerate(err["audio_file"]):
        spk = err.loc[i, "true_speaker"]
        if pd.isna(spk):
            ranks.append(np.nan)
            continue
        cand = np.where(train_speakers == spk)[0]
        if len(cand) == 0:
            ranks.append(np.nan)
            continue
        order = np.argsort(-sims[i])  # descending
        ranks.append(int(np.argmax(np.isin(order, cand))) + 1)
    err2["emb_true_rank"] = ranks

    print("\n=== H1: embedding 1-NN (whole val) ===")
    known = err2[err2["true_group"] == "known"]
    print(f"  known val files with same-speaker train file: {known['emb_true_rank'].notna().sum()}")
    hit = known["emb_nn_hit"].sum()
    print(f"  emb 1-NN same-speaker hit: {hit}/{len(known)} ({hit/len(known):.3f})")
    print(f"  emb top-1 same-speaker rank<=2: {(known['emb_true_rank']<=2).sum()}, "
          f"rank<=5: {(known['emb_true_rank']<=5).sum()}, "
          f"rank<=10: {(known['emb_true_rank']<=10).sum()}")

    # dirty subset (non_speech + low_snr)
    feat = pd.read_csv(ROOT / "data" / "processed" / "val_acoustic_features.csv",
                       encoding="utf-8-sig")
    feat = feat.loc[:, ~feat.columns.duplicated()]
    if "duration_s_x" in feat.columns:
        feat["duration_s"] = feat["duration_s_x"].fillna(feat["duration_s_y"])
    err3 = err2.merge(feat[["audio_file", "bucket", "vad_ratio"]], on="audio_file", how="left")
    dirty = err3[err3["bucket"] != "clean_speech"]
    print("\n=== H1 on DIRTY (non_speech+low_snr) known val files ===")
    dk = dirty[dirty["true_group"] == "known"]
    print(f"  n={len(dk)}, emb 1-NN hit: {dk['emb_nn_hit'].sum()}, "
          f"rank<=2: {(dk['emb_true_rank']<=2).sum()}, rank<=5: {(dk['emb_true_rank']<=5).sum()}")

    # ---------- H2: noise-PSD 1-NN ----------
    print("\n=== H2: noise-PSD fingerprint 1-NN ===")
    train_psd, train_psd_files = [], []
    for i, row in enumerate(train_lab.itertuples(index=False)):
        wav = AUDIO_DIR / row.audio_file
        if not wav.exists():
            continue
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            n = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        train_psd.append(noise_psd(x))
        train_psd_files.append(row.audio_file)
        if (i + 1) % 800 == 0:
            print(f"    train PSD ...{i+1}/{len(train_lab)}")
    Ptr = np.stack(train_psd).astype(np.float32)
    np.savez(ROOT / "data" / "processed" / "noise_psd_train.npz",
             psd=Ptr, files=np.array(train_psd_files, dtype=object))
    tr_spk = np.array([tf_map.get(f, "?") for f in train_psd_files])

    val_psd, val_psd_files = [], []
    for i, row in enumerate(err.itertuples(index=False)):
        wav = AUDIO_DIR / row.audio_file
        if not wav.exists():
            val_psd.append(np.zeros(FFT // 2 + 1, dtype=np.float32))
            val_psd_files.append(row.audio_file)
            continue
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            n = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        val_psd.append(noise_psd(x))
        val_psd_files.append(row.audio_file)
        if (i + 1) % 300 == 0:
            print(f"    val PSD ...{i+1}/{len(err)}")
    Pval = np.stack(val_psd).astype(np.float32)
    np.savez(ROOT / "data" / "processed" / "noise_psd_val.npz",
             psd=Pval, files=np.array(val_psd_files, dtype=object))

    sims2 = Pval @ Ptr.T
    nn2 = sims2.argmax(axis=1)
    err3["psd_nn_speaker"] = [tr_spk[i] for i in nn2]
    err3["psd_nn_sim"] = sims2[np.arange(len(err3)), nn2]
    err3["psd_nn_hit"] = err3["psd_nn_speaker"] == err3["true_speaker"]

    dk2 = err3[err3["true_group"] == "known"]
    print(f"\n  PSD 1-NN same-speaker hit (all known val): "
          f"{dk2['psd_nn_hit'].sum()}/{len(dk2)} ({dk2['psd_nn_hit'].mean():.3f})")
    print(f"  PSD 1-NN hit on CLEAN known: "
          f"{dk2[dk2['bucket']=='clean_speech']['psd_nn_hit'].sum()}/"
          f"{len(dk2[dk2['bucket']=='clean_speech'])}")
    print(f"  PSD 1-NN hit on DIRTY known: "
          f"{dk2[dk2['bucket']!='clean_speech']['psd_nn_hit'].sum()}/"
          f"{len(dk2[dk2['bucket']!='clean_speech'])}")

    print("\n=== per-file detail for cluster errors ===")
    cols = ["audio_file", "bucket", "vad_ratio", "true_speaker", "cluster_verdict",
            "emb_nn_hit", "emb_true_rank", "psd_nn_hit", "psd_nn_sim"]
    cl = err3[err3["cluster_verdict"] != "OK"]
    print(cl[cols].to_string(index=False))

    err3.to_csv(ROOT / "data" / "processed" / "channel_signature_results.csv",
                index=False, encoding="utf-8-sig")
    print("\n  saved data/processed/channel_signature_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())