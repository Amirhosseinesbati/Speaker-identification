"""Follow-up: who are the error speakers' train files, and is the noise-PSD
match statistically meaningful?

1. For each of the 19 cluster-error speakers: list train files + acoustic
   bucket + noise-PSD cosine to the val file (dirty sibling check).
2. Null distribution of noise-PSD cosine (same-speaker vs different-speaker
   train pairs) to calibrate whether 0.9+ cosines are distinctive.
3. For the 4 'emb_nn_hit=True but decision wrong' files: centroid cosine
   top-5 (explains the tau-gate / wrong-known mechanics).
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
SR = 16000
FFT = 512
HOP = 160
FRAME = 512


def noise_psd(x: np.ndarray, n_fft: int = FFT, hop: int = HOP) -> np.ndarray:
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


def load(x_path: str) -> np.ndarray:
    x, sr = sf.read(str(x_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x


def bucket_of(x: np.ndarray) -> tuple[str, float]:
    n = (len(x) - FRAME) // HOP + 1
    if n < 1:
        return "non_speech", 0.0
    idx = np.arange(n)[:, None] * HOP + np.arange(FRAME)[None, :]
    frames = x[idx]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)
    floor_db = max(np.percentile(rms_db, 10), -55.0)
    speech = rms_db > (floor_db + 12.0)
    vad = float(speech.mean())
    if vad >= 0.30:
        return "clean_speech", vad
    if vad >= 0.05:
        return "low_snr_speech", vad
    return "non_speech", vad


def main() -> int:
    lab = pd.read_csv(ROOT / "data" / "processed" / "audio_wav_labels.csv")
    err = pd.read_csv(ROOT / "error_analysis_report.csv", encoding="utf-8-sig")
    val_files = set(err["audio_file"])
    train_lab = lab[~lab["audio_file"].isin(val_files)].reset_index(drop=True)
    spk_files = {spk: grp["audio_file"].tolist()
                 for spk, grp in train_lab.groupby("speaker_id")}

    results = pd.read_csv(ROOT / "data" / "processed" / "channel_signature_results.csv",
                          encoding="utf-8-sig")
    results = results.loc[:, ~results.columns.duplicated()]

    # ---------- 1. train files of each error speaker ----------
    print("=" * 100)
    print("1) Error speakers: val file vs their train files (acoustic bucket + PSD cosine)")
    print("=" * 100)
    cl = results[results["cluster_verdict"] != "OK"]
    for _, r in cl.iterrows():
        spk = r["true_speaker"]
        if pd.isna(spk):
            continue
        tr = spk_files.get(spk, [])
        if spk == "unknown":
            tr = tr[:8]
        xv = load(AUDIO_DIR / r["audio_file"])
        psd_v = noise_psd(xv)
        vb, vv = bucket_of(xv)
        parts = []
        for f in tr:
            xf = load(AUDIO_DIR / f)
            fb, fv = bucket_of(xf)
            pf = noise_psd(xf)
            c = float(psd_v @ pf)
            parts.append(f"{f[:8]}:{fb}(vad={fv:.2f},cos={c:.2f})")
        print(f"\n  {r['audio_file']}  [{vb} vad={vv:.2f}]  verdict={r['cluster_verdict']} "
              f"emb_rank={r.get('emb_true_rank')}")
        print(f"    train files ({len(parts)}):")
        for p in parts:
            print(f"      {p}")

    # ---------- 2. noise-PSD null distribution ----------
    print("\n" + "=" * 100)
    print("2) noise-PSD cosine null distribution (train pairs)")
    print("=" * 100)
    rng = np.random.default_rng(0)
    n_psd = np.load(ROOT / "data" / "processed" / "noise_psd_train.npz", allow_pickle=True)
    P = n_psd["psd"]
    files = list(n_psd["files"])
    idx = {f: i for i, f in enumerate(files)}
    spk_arr = np.array([lab.loc[lab["audio_file"] == f, "speaker_id"].iloc[0]
                        if (lab["audio_file"] == f).any() else "?" for f in files])

    same, diff = [], []
    for _ in range(6000):
        a, b = rng.integers(0, len(files), 2)
        if a == b:
            continue
        c = float(P[a] @ P[b])
        if spk_arr[a] == spk_arr[b]:
            same.append(c)
        else:
            diff.append(c)
    same, diff = np.array(same), np.array(diff)
    print(f"  same-speaker pairs  (n={len(same)}): mean={same.mean():.3f} "
          f"p50={np.median(same):.3f} p90={np.percentile(same,90):.3f} max={same.max():.3f}")
    print(f"  diff-speaker pairs  (n={len(diff)}): mean={diff.mean():.3f} "
          f"p50={np.median(diff):.3f} p90={np.percentile(diff,90):.3f} max={diff.max():.3f}")

    # ---------- 3. centroid cosine top-5 for the emb-correct-but-decision-wrong files ----------
    print("\n" + "=" * 100)
    print("3) Centroid cosine top-5 for cluster-error files (emb space)")
    print("=" * 100)
    val_emb = np.load(ROOT / "data" / "processed" / "val_emb_campp.npy").astype(np.float32)
    per = pd.read_csv(ROOT / "data" / "processed" / "model_compare_legacy_vs_cluster"
                      "/per_file_predictions.csv")
    per_idx = {f: i for i, f in enumerate(per["audio_file"])}
    cents = np.load(ROOT / "data" / "processed" / "centroids_campp.npz")["centroids"].astype(np.float32)
    # unknown-cluster centroids (k554) merged like inference does
    uc = np.load(ROOT / "data" / "processed" / "centroids_unknown_campp.npz")["centroids"].astype(np.float32)
    allc = np.vstack([np.zeros((1, 192), dtype=np.float32), cents, uc])  # class0=unknown(0), 1..446, clusters
    labmap = pd.read_csv(ROOT / "data" / "processed" / "cleaned_labels.csv")
    lbl_of = {r.audio_file: int(r.label) for r in labmap.itertuples(index=False)}

    for _, r in cl.iterrows():
        f = r["audio_file"]
        emb = val_emb[per_idx[f]]
        cos = allc @ emb
        top = np.argsort(-cos)[:5]
        true_lbl = lbl_of.get(f, 0)
        row = " ".join(f"{t}:{cos[t]:.3f}" for t in top)
        print(f"  {f[:12]} true={true_lbl} verdict={r['cluster_verdict']} | {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())