"""Deep, reproducible data audit for the IAAA 2026 speaker-ID challenge.

This phase is intentionally independent of the project's training strategy.  It
audits the raw corpus, corrects statistical issues in the earlier EDA, and uses
the cached frozen-ECAPA embeddings only as a measurement instrument.

Outputs are written under ``eda/``:

* ``DEEP_DATA_UNDERSTANDING_REPORT.md``
* ``deep_data_summary.json``
* ``deep_audio_inventory.csv``
* ``deep_known_speaker_diagnostics.csv``
* ``deep_nearest_speaker_pairs.csv``
* ``deep_unknown_pseudo_speakers.csv``
* four diagnostic PNG figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EDA_DIR = ROOT / "eda"
LABELS_PATH = RAW_DIR / "labels.csv"
PROCESSED_LABELS_PATH = PROCESSED_DIR / "audio_wav_labels.csv"
EMBEDDINGS_PATH = EDA_DIR / "phase3_embeddings.npy"

INVENTORY_PATH = EDA_DIR / "deep_audio_inventory.csv"
SUMMARY_PATH = EDA_DIR / "deep_data_summary.json"
REPORT_PATH = EDA_DIR / "DEEP_DATA_UNDERSTANDING_REPORT.md"
KNOWN_DIAGNOSTICS_PATH = EDA_DIR / "deep_known_speaker_diagnostics.csv"
NEAREST_PAIRS_PATH = EDA_DIR / "deep_nearest_speaker_pairs.csv"
UNKNOWN_PSEUDO_PATH = EDA_DIR / "deep_unknown_pseudo_speakers.csv"
EXACT_DUPLICATES_PATH = EDA_DIR / "deep_exact_duplicate_groups.csv"

PLOT_QUALITY = EDA_DIR / "deep_quality_distributions.png"
PLOT_ORDER = EDA_DIR / "deep_order_similarity.png"
PLOT_PSEUDO = EDA_DIR / "deep_unknown_pseudo_group_sizes.png"
PLOT_EMBEDDING = EDA_DIR / "deep_embedding_margin_ood.png"

UNKNOWN = "unknown"
TARGET_UNKNOWN_SPEAKERS = 554
MIN_VALID_DURATION = 1.0
RANDOM_SEED = 42


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["audio_file"] = df["audio_file"].astype(str).str.strip()
    df["row_index"] = np.arange(len(df), dtype=int)
    df["is_unknown"] = df["speaker_id"].str.lower().eq(UNKNOWN)
    return df


def label_and_order_audit(df: pd.DataFrame) -> dict:
    files = [p.name for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"]
    labelled = set(df["audio_file"])
    on_disk = set(files)

    run_id = df["speaker_id"].ne(df["speaker_id"].shift()).cumsum()
    runs = (
        df.assign(run_id=run_id)
        .groupby("run_id", sort=False)
        .agg(
            speaker_id=("speaker_id", "first"),
            is_unknown=("is_unknown", "first"),
            n=("audio_file", "size"),
            start=("row_index", "min"),
            end=("row_index", "max"),
        )
        .reset_index(drop=True)
    )
    known_runs = runs[~runs["is_unknown"]]
    unknown_runs = runs[runs["is_unknown"]]
    runs_per_known = known_runs.groupby("speaker_id").size()
    counts = df.loc[~df["is_unknown"], "speaker_id"].value_counts()

    return {
        "rows": int(len(df)),
        "audio_files_on_disk": int(len(files)),
        "missing_labelled_files": int(len(labelled - on_disk)),
        "unlabelled_audio_files": int(len(on_disk - labelled)),
        "duplicate_rows": int(df.duplicated(["speaker_id", "audio_file"]).sum()),
        "duplicate_audio_names": int(df["audio_file"].duplicated().sum()),
        "known_files": int((~df["is_unknown"]).sum()),
        "unknown_files": int(df["is_unknown"].sum()),
        "known_speakers": int(df.loc[~df["is_unknown"], "speaker_id"].nunique()),
        "known_count_distribution": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
        "known_runs": int(len(known_runs)),
        "known_speakers_in_exactly_one_run": int((runs_per_known == 1).sum()),
        "unknown_runs": int(len(unknown_runs)),
        "unknown_run_min": int(unknown_runs["n"].min()),
        "unknown_run_median": float(unknown_runs["n"].median()),
        "unknown_run_max": int(unknown_runs["n"].max()),
        "unknown_runs_multiple_of_five": int((unknown_runs["n"] % 5 == 0).sum()),
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_audio(path: Path, sample_rate: int, frames: int) -> np.ndarray:
    """Read at most three 15-second windows spread over a file."""
    if frames <= 0:
        return np.empty((0, 1), dtype=np.float32)
    window = min(frames, 15 * sample_rate)
    if frames <= window:
        starts = [0]
    else:
        starts = sorted(
            {
                0,
                max(0, (frames - window) // 2),
                max(0, frames - window),
            }
        )
    chunks: list[np.ndarray] = []
    with sf.SoundFile(path) as stream:
        for start in starts:
            stream.seek(int(start))
            x = stream.read(int(window), dtype="float32", always_2d=True)
            if x.size:
                chunks.append(x)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 1), dtype=np.float32)


def _signal_features(x: np.ndarray, sample_rate: int) -> dict:
    if x.size == 0:
        return {
            "rms_dbfs": -120.0,
            "peak": 0.0,
            "clip_ratio": 0.0,
            "dc_abs": 0.0,
            "zcr": 0.0,
            "active_frame_ratio": 0.0,
            "silent_frame_ratio": 1.0,
            "frame_dynamic_range_db": 0.0,
            "channel_rms_delta_db": 0.0,
            "stereo_corr": 1.0,
            "side_to_mid_db": -120.0,
        }
    if x.ndim == 1:
        x = x[:, None]
    eps = 1e-12
    channel_rms = np.sqrt(np.mean(np.square(x), axis=0, dtype=np.float64) + eps)
    if x.shape[1] >= 2:
        left = x[:, 0].astype(np.float64)
        right = x[:, 1].astype(np.float64)
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denom = math.sqrt(float(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
        stereo_corr = float(np.dot(left_centered, right_centered) / denom) if denom > 0 else 1.0
        mid_energy = float(np.mean(np.square(left + right)) + eps)
        side_energy = float(np.mean(np.square(left - right)) + eps)
        side_to_mid_db = float(10 * np.log10(side_energy / mid_energy))
        channel_rms_delta_db = float(abs(20 * np.log10((channel_rms[0] + eps) / (channel_rms[1] + eps))))
    else:
        stereo_corr = 1.0
        side_to_mid_db = -120.0
        channel_rms_delta_db = 0.0
    x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))
    frame_len = max(1, int(round(0.02 * sample_rate)))
    usable = (len(x) // frame_len) * frame_len
    if usable:
        frames = x[:usable].reshape(-1, frame_len)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64) + eps)
        frame_db = 20 * np.log10(frame_rms + eps)
        p90 = float(np.percentile(frame_db, 90))
        active_threshold = max(-50.0, p90 - 30.0)
        active_ratio = float(np.mean(frame_db >= active_threshold))
        silent_ratio = float(np.mean(frame_db < -55.0))
        dynamic = float(np.percentile(frame_db, 90) - np.percentile(frame_db, 10))
    else:
        active_ratio = 0.0
        silent_ratio = 1.0
        dynamic = 0.0
    return {
        "rms_dbfs": float(20 * np.log10(rms + eps)),
        "peak": float(np.max(np.abs(x))),
        "clip_ratio": float(np.mean(np.abs(x) >= 0.999)),
        "dc_abs": float(abs(np.mean(x, dtype=np.float64))),
        "zcr": float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if len(x) > 1 else 0.0,
        "active_frame_ratio": active_ratio,
        "silent_frame_ratio": silent_ratio,
        "frame_dynamic_range_db": dynamic,
        "channel_rms_delta_db": channel_rms_delta_db,
        "stereo_corr": stereo_corr,
        "side_to_mid_db": side_to_mid_db,
    }


def build_audio_inventory(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    if INVENTORY_PATH.exists() and not refresh:
        cached = pd.read_csv(INVENTORY_PATH)
        if len(cached) == len(df) and set(cached["audio_file"]) == set(df["audio_file"]):
            return cached

    size_counts = Counter((RAW_DIR / name).stat().st_size for name in df["audio_file"])
    rows: list[dict] = []
    for n, row in enumerate(df.itertuples(index=False), start=1):
        path = RAW_DIR / row.audio_file
        size = path.stat().st_size
        header = path.read_bytes()[:12]
        is_wave = header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        record = {
            "row_index": int(row.row_index),
            "speaker_id": row.speaker_id,
            "is_unknown": bool(row.is_unknown),
            "audio_file": row.audio_file,
            "bytes": int(size),
            "extension": path.suffix.lower(),
            "container_magic": "RIFF/WAVE" if is_wave else header.hex(),
            "md5": _md5(path) if size_counts[size] > 1 else "",
        }
        try:
            info = sf.info(path)
            record.update(
                {
                    "format": info.format,
                    "subtype": info.subtype,
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "duration_sec": float(info.duration),
                    "read_error": "",
                }
            )
            x = _sample_audio(path, int(info.samplerate), int(info.frames))
            record.update(_signal_features(x, int(info.samplerate)))
        except Exception as exc:  # retain the failure in the audit instead of aborting
            record.update(
                {
                    "format": "ERROR",
                    "subtype": "ERROR",
                    "sample_rate": 0,
                    "channels": 0,
                    "frames": 0,
                    "duration_sec": 0.0,
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )
            record.update(_signal_features(np.empty((0, 1), dtype=np.float32), 16000))
        rows.append(record)
        if n % 250 == 0 or n == len(df):
            print(f"audio inventory: {n:,}/{len(df):,}", flush=True)

    inventory = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    inventory.to_csv(INVENTORY_PATH, index=False)
    return inventory


def audio_audit(inventory: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    inv = inventory.copy()
    inv["is_valid_1s"] = inv["duration_sec"] >= MIN_VALID_DURATION
    inv["is_empty_header"] = (inv["bytes"] == 48) & (inv["frames"] <= 1)
    valid = inv[inv["is_valid_1s"]]
    short = inv[~inv["is_valid_1s"]]

    hashable = inv[inv["md5"].fillna("").astype(str).str.len() > 0]
    duplicate_groups = [g for _, g in hashable.groupby("md5") if len(g) > 1]
    duplicate_files = pd.concat(duplicate_groups, ignore_index=True) if duplicate_groups else inv.iloc[0:0]
    valid_duplicate_groups = [g for g in duplicate_groups if (g["duration_sec"] >= 1.0).any()]
    conflicting_duplicate_groups = [g for g in duplicate_groups if g["speaker_id"].nunique() > 1]
    valid_conflicting_duplicate_groups = [
        g for g in conflicting_duplicate_groups if (g["duration_sec"] >= 1.0).any()
    ]
    inv["exact_duplicate_group"] = ""
    duplicate_exports = []
    for group_number, group in enumerate(duplicate_groups, start=1):
        group_id = f"dup_{group_number:03d}"
        inv.loc[group.index, "exact_duplicate_group"] = group_id
        exported = inv.loc[group.index].copy()
        exported["exact_duplicate_group"] = group_id
        exported["group_has_label_conflict"] = group["speaker_id"].nunique() > 1
        exported["group_has_valid_audio"] = (group["duration_sec"] >= 1.0).any()
        duplicate_exports.append(exported)
    duplicate_export = pd.concat(duplicate_exports, ignore_index=True) if duplicate_exports else inv.iloc[0:0]
    duplicate_export.to_csv(EXACT_DUPLICATES_PATH, index=False)

    valid_known_counts = valid.loc[~valid["is_unknown"], "speaker_id"].value_counts()
    format_counts = (
        inv.groupby(["container_magic", "format", "subtype", "sample_rate", "channels"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )

    features = [
        "rms_dbfs",
        "peak",
        "clip_ratio",
        "dc_abs",
        "zcr",
        "active_frame_ratio",
        "silent_frame_ratio",
        "frame_dynamic_range_db",
        "channel_rms_delta_db",
        "stereo_corr",
        "side_to_mid_db",
    ]
    comparisons = {}
    known = valid[~valid["is_unknown"]]
    unknown = valid[valid["is_unknown"]]
    for feature in features:
        a = known[feature].dropna().to_numpy(float)
        b = unknown[feature].dropna().to_numpy(float)
        pooled = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0)
        d = (float(np.mean(a)) - float(np.mean(b))) / pooled if pooled > 0 else 0.0
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        comparisons[feature] = {
            "known_mean": float(np.mean(a)),
            "unknown_mean": float(np.mean(b)),
            "cohens_d": float(d),
            "mannwhitney_p": float(p),
        }

    summary = {
        "all_files": int(len(inv)),
        "actual_riff_wave_files": int(inv["container_magic"].eq("RIFF/WAVE").sum()),
        "declared_mp3_extension_files": int(inv["extension"].eq(".mp3").sum()),
        "read_errors": int(inv["read_error"].fillna("").astype(str).str.len().gt(0).sum()),
        "under_1s": int((~inv["is_valid_1s"]).sum()),
        "empty_48_byte_headers": int(inv["is_empty_header"].sum()),
        "short_but_nonempty_under_1s": int(((~inv["is_valid_1s"]) & (~inv["is_empty_header"])).sum()),
        "under_1s_known": int(((~inv["is_valid_1s"]) & (~inv["is_unknown"])).sum()),
        "under_1s_unknown": int(((~inv["is_valid_1s"]) & inv["is_unknown"]).sum()),
        "valid_files": int(inv["is_valid_1s"].sum()),
        "valid_known_files": int((inv["is_valid_1s"] & (~inv["is_unknown"])).sum()),
        "valid_unknown_files": int((inv["is_valid_1s"] & inv["is_unknown"]).sum()),
        "valid_known_speakers": int(valid_known_counts.size),
        "valid_known_count_min": int(valid_known_counts.min()),
        "valid_known_count_median": float(valid_known_counts.median()),
        "valid_known_speakers_below_4": int((valid_known_counts < 4).sum()),
        "valid_known_speakers_below_5": int((valid_known_counts < 5).sum()),
        "exact_duplicate_groups": int(len(duplicate_groups)),
        "exact_duplicate_files": int(len(duplicate_files)),
        "valid_exact_duplicate_groups": int(len(valid_duplicate_groups)),
        "conflicting_exact_duplicate_groups": int(len(conflicting_duplicate_groups)),
        "valid_conflicting_exact_duplicate_groups": int(len(valid_conflicting_duplicate_groups)),
        "duration_seconds": {
            "min": float(inv["duration_sec"].min()),
            "p01": float(inv["duration_sec"].quantile(0.01)),
            "p05": float(inv["duration_sec"].quantile(0.05)),
            "median": float(inv["duration_sec"].median()),
            "p95": float(inv["duration_sec"].quantile(0.95)),
            "max": float(inv["duration_sec"].max()),
        },
        "format_counts": {" | ".join(map(str, key)): int(value) for key, value in format_counts.items()},
        "format_outliers": inv.loc[
            ~(
                inv["container_magic"].eq("RIFF/WAVE")
                & inv["format"].eq("WAV")
                & inv["sample_rate"].eq(16000)
                & inv["channels"].eq(2)
            ),
            ["audio_file", "speaker_id", "format", "subtype", "sample_rate", "channels", "duration_sec"],
        ].to_dict(orient="records"),
        "stereo": {
            "files": int((valid["channels"] == 2).sum()),
            "median_channel_rms_delta_db": float(valid.loc[valid["channels"] == 2, "channel_rms_delta_db"].median()),
            "median_channel_correlation": float(valid.loc[valid["channels"] == 2, "stereo_corr"].median()),
            "p05_channel_correlation": float(valid.loc[valid["channels"] == 2, "stereo_corr"].quantile(0.05)),
            "median_side_to_mid_db": float(valid.loc[valid["channels"] == 2, "side_to_mid_db"].median()),
            "files_channel_rms_delta_over_6db": int(
                ((valid["channels"] == 2) & (valid["channel_rms_delta_db"] > 6.0)).sum()
            ),
            "files_channel_correlation_below_0_9": int(
                ((valid["channels"] == 2) & (valid["stereo_corr"] < 0.9)).sum()
            ),
        },
        "known_vs_unknown_quality": comparisons,
    }
    return summary, inv


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _contiguous_groups_from_boundaries(n_items: int, boundary_positions: np.ndarray) -> np.ndarray:
    groups = np.zeros(n_items, dtype=int)
    if n_items <= 1:
        return groups
    boundaries = np.zeros(n_items - 1, dtype=bool)
    boundaries[np.asarray(boundary_positions, dtype=int)] = True
    groups[1:] = np.cumsum(boundaries)
    return groups


def embedding_audit(df: pd.DataFrame, inventory: pd.DataFrame) -> tuple:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Cached frozen-encoder embeddings not found: {EMBEDDINGS_PATH}")

    valid_names = set(inventory.loc[inventory["duration_sec"] >= MIN_VALID_DURATION, "audio_file"])
    clean = df[df["audio_file"].isin(valid_names)].copy().reset_index(drop=True)
    raw_run_id = df["speaker_id"].ne(df["speaker_id"].shift()).cumsum().astype(int)
    raw_run_offset = df.groupby(raw_run_id, sort=False).cumcount().astype(int)
    run_id_lookup = dict(zip(df["row_index"], raw_run_id))
    run_offset_lookup = dict(zip(df["row_index"], raw_run_offset))
    clean["label_run_id"] = clean["row_index"].map(run_id_lookup).astype(int)
    clean["label_run_offset"] = clean["row_index"].map(run_offset_lookup).astype(int)
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    if len(clean) != len(embeddings):
        raise RuntimeError(
            f"Embedding/order mismatch: {len(embeddings)} cached rows vs {len(clean)} valid label rows"
        )
    emb = _l2norm(embeddings)

    known_speaker_ids = clean.loc[~clean["is_unknown"], "speaker_id"].drop_duplicates().tolist()
    sid_to_num = {sid: i + 1 for i, sid in enumerate(known_speaker_ids)}
    numeric = clean["speaker_id"].map(sid_to_num).fillna(0).astype(int).to_numpy()
    known_mask = numeric > 0
    unknown_mask = ~known_mask
    speakers = np.arange(1, len(known_speaker_ids) + 1, dtype=int)

    centroids = np.stack([emb[numeric == sid].mean(axis=0) for sid in speakers])
    centroids = _l2norm(centroids)
    known_emb = emb[known_mask]
    known_ids = numeric[known_mask]
    loo_sims = known_emb @ centroids.T
    for i, sid in enumerate(known_ids):
        members = emb[numeric == sid]
        loo = members.sum(axis=0) - known_emb[i]
        loo = loo / (np.linalg.norm(loo) + 1e-12)
        loo_sims[i, sid - 1] = known_emb[i] @ loo

    known_pred_idx = loo_sims.argmax(axis=1)
    known_pred = known_pred_idx + 1
    known_true_idx = known_ids - 1
    known_top5 = np.argsort(-loo_sims, axis=1)[:, :5]
    true_sim = loo_sims[np.arange(len(known_ids)), known_true_idx]
    imposter_sims = loo_sims.copy()
    imposter_sims[np.arange(len(known_ids)), known_true_idx] = -np.inf
    nearest_impostor_idx = imposter_sims.argmax(axis=1)
    nearest_impostor_sim = imposter_sims[np.arange(len(known_ids)), nearest_impostor_idx]
    margins = true_sim - nearest_impostor_sim

    per_class_f1 = f1_score(known_ids, known_pred, labels=speakers, average=None, zero_division=0)
    known_clean = clean[known_mask].copy().reset_index(drop=True)
    known_clean["numeric_speaker"] = known_ids
    known_clean["pred_numeric"] = known_pred
    known_clean["correct"] = known_pred == known_ids
    known_clean["true_loo_cosine"] = true_sim
    known_clean["nearest_impostor_cosine"] = nearest_impostor_sim
    known_clean["margin"] = margins
    known_clean["nearest_impostor_speaker_id"] = [known_speaker_ids[i] for i in nearest_impostor_idx]

    diagnostics = []
    for sid in speakers:
        g = known_clean[known_clean["numeric_speaker"] == sid]
        diagnostics.append(
            {
                "speaker_id": known_speaker_ids[sid - 1],
                "n_valid_files": int(len(g)),
                "loo_accuracy": float(g["correct"].mean()),
                "closed_set_f1": float(per_class_f1[sid - 1]),
                "mean_true_loo_cosine": float(g["true_loo_cosine"].mean()),
                "mean_margin": float(g["margin"].mean()),
                "min_margin": float(g["margin"].min()),
                "dominant_nearest_impostor": g["nearest_impostor_speaker_id"].mode().iloc[0],
            }
        )
    diagnostics_df = pd.DataFrame(diagnostics).sort_values(
        ["closed_set_f1", "mean_margin", "n_valid_files"], ascending=[True, True, True]
    )

    centroid_similarity = centroids @ centroids.T
    np.fill_diagonal(centroid_similarity, -np.inf)
    upper_i, upper_j = np.triu_indices(len(speakers), k=1)
    pair_sims = centroid_similarity[upper_i, upper_j]
    top_pair_idx = np.argsort(-pair_sims)[:50]
    nearest_pairs_df = pd.DataFrame(
        {
            "speaker_a": [known_speaker_ids[upper_i[i]] for i in top_pair_idx],
            "speaker_b": [known_speaker_ids[upper_j[i]] for i in top_pair_idx],
            "centroid_cosine": pair_sims[top_pair_idx],
        }
    )

    same_scores: list[float] = []
    for sid in speakers:
        x = emb[numeric == sid]
        a, b = np.triu_indices(len(x), k=1)
        same_scores.extend(np.sum(x[a] * x[b], axis=1).tolist())
    same = np.asarray(same_scores, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    known_indices = np.flatnonzero(known_mask)
    cross_scores: list[float] = []
    while len(cross_scores) < 100_000:
        a = rng.choice(known_indices, size=120_000, replace=True)
        b = rng.choice(known_indices, size=120_000, replace=True)
        keep = numeric[a] != numeric[b]
        scores = np.sum(emb[a[keep]] * emb[b[keep]], axis=1)
        cross_scores.extend(scores.tolist())
    cross = np.asarray(cross_scores[:100_000], dtype=float)
    verify_y = np.concatenate([np.ones(len(same), dtype=int), np.zeros(len(cross), dtype=int)])
    verify_score = np.concatenate([same, cross])
    verify_fpr, verify_tpr, verify_thr = roc_curve(verify_y, verify_score)
    verify_fnr = 1.0 - verify_tpr
    eer_i = int(np.argmin(np.abs(verify_fpr - verify_fnr)))
    eer = float((verify_fpr[eer_i] + verify_fnr[eer_i]) / 2.0)
    eer_threshold = float(verify_thr[eer_i])

    ood_scores = np.zeros(len(clean), dtype=float)
    ood_scores[known_mask] = 1.0 - loo_sims.max(axis=1)
    unknown_sims = emb[unknown_mask] @ centroids.T
    ood_scores[unknown_mask] = 1.0 - unknown_sims.max(axis=1)
    ood_y = unknown_mask.astype(int)
    ood_auc = float(roc_auc_score(ood_y, ood_scores))
    ood_fpr, ood_tpr, ood_thr = roc_curve(ood_y, ood_scores)
    tpr95_idx = np.flatnonzero(ood_tpr >= 0.95)
    fpr_at_tpr95 = float(np.min(ood_fpr[tpr95_idx])) if len(tpr95_idx) else 1.0
    youden_i = int(np.argmax(ood_tpr - ood_fpr))

    base_pred = np.zeros(len(clean), dtype=int)
    base_pred[known_mask] = known_pred
    base_pred[unknown_mask] = unknown_sims.argmax(axis=1) + 1
    thresholds = np.linspace(float(np.quantile(ood_scores, 0.001)), float(np.quantile(ood_scores, 0.999)), 401)
    macro_values = []
    all_labels = np.arange(len(speakers) + 1)
    for threshold in thresholds:
        pred = np.where(ood_scores > threshold, 0, base_pred)
        macro_values.append(
            f1_score(numeric, pred, labels=all_labels, average="macro", zero_division=0)
        )
    best_macro_i = int(np.argmax(macro_values))

    # Validate the row-order segmentation signal on the known identities.  We
    # hide their labels, retain only row order, and select exactly S-1 weakest
    # adjacent similarities as speaker boundaries.
    known_order = np.flatnonzero(known_mask)
    known_adj = np.sum(emb[known_order[:-1]] * emb[known_order[1:]], axis=1)
    known_true_boundary = numeric[known_order[:-1]] != numeric[known_order[1:]]
    n_known_boundaries = int(known_true_boundary.sum())
    selected_known_boundaries = np.argsort(known_adj)[:n_known_boundaries]
    selected_boundary_mask = np.zeros(len(known_adj), dtype=bool)
    selected_boundary_mask[selected_known_boundaries] = True
    known_order_groups = _contiguous_groups_from_boundaries(len(known_order), selected_known_boundaries)
    known_boundary_precision = float(known_true_boundary[selected_known_boundaries].mean())
    known_order_ari = float(adjusted_rand_score(numeric[known_order], known_order_groups))

    threshold_grid = np.linspace(float(known_adj.min()), float(known_adj.max()), 1501)
    threshold_f1 = []
    for threshold in threshold_grid:
        pred_boundary = known_adj < threshold
        threshold_f1.append(f1_score(known_true_boundary.astype(int), pred_boundary.astype(int), zero_division=0))
    best_boundary_i = int(np.argmax(threshold_f1))
    calibrated_boundary_threshold = float(threshold_grid[best_boundary_i])
    calibrated_known_boundary = known_adj < calibrated_boundary_threshold
    calibrated_tp = int(np.sum(calibrated_known_boundary & known_true_boundary))
    calibrated_precision = calibrated_tp / max(1, int(calibrated_known_boundary.sum()))
    calibrated_recall = calibrated_tp / max(1, int(known_true_boundary.sum()))

    # Apply the same label-free segmentation to the ordered unknown rows, using
    # the organizer-provided population count (554).  This is an EDA artefact,
    # not a claim that every boundary is certain.
    unknown_order = np.flatnonzero(unknown_mask)
    unknown_adj = np.sum(emb[unknown_order[:-1]] * emb[unknown_order[1:]], axis=1)
    n_unknown_boundaries = min(TARGET_UNKNOWN_SPEAKERS - 1, len(unknown_adj))
    selected_unknown_boundaries = np.argsort(unknown_adj)[:n_unknown_boundaries]
    pseudo_groups = _contiguous_groups_from_boundaries(len(unknown_order), selected_unknown_boundaries)
    pseudo_sizes = pd.Series(pseudo_groups).value_counts().sort_index()
    boundary_mask = np.zeros(len(unknown_adj), dtype=bool)
    boundary_mask[selected_unknown_boundaries] = True

    unknown_order_df = clean.iloc[unknown_order].copy().reset_index(drop=True)
    unknown_run_ids = unknown_order_df["label_run_id"].to_numpy()
    unknown_offsets = unknown_order_df["label_run_offset"].to_numpy()
    forced_run_boundary = unknown_run_ids[:-1] != unknown_run_ids[1:]
    calibrated_unknown_boundary = forced_run_boundary | (unknown_adj < calibrated_boundary_threshold)
    calibrated_groups = _contiguous_groups_from_boundaries(
        len(unknown_order), np.flatnonzero(calibrated_unknown_boundary)
    )
    calibrated_sizes = pd.Series(calibrated_groups).value_counts().sort_index()

    block5_key = np.array(
        [f"{run_id}:{offset // 5}" for run_id, offset in zip(unknown_run_ids, unknown_offsets)],
        dtype=object,
    )
    _, block5_groups = np.unique(block5_key, return_inverse=True)
    # np.unique sorts keys; refactor in encounter order so IDs are contiguous.
    block5_groups = pd.factorize(block5_key, sort=False)[0]
    block5_boundary = block5_groups[:-1] != block5_groups[1:]
    block5_sizes = pd.Series(block5_groups).value_counts().sort_index()

    pseudo_df = unknown_order_df[["row_index", "audio_file", "speaker_id", "label_run_id", "label_run_offset"]].copy()
    pseudo_df["pseudo_forced_554"] = [f"unknown_f554_{g:04d}" for g in pseudo_groups]
    pseudo_df["pseudo_calibrated"] = [f"unknown_cal_{g:04d}" for g in calibrated_groups]
    pseudo_df["pseudo_block5"] = [f"unknown_b5_{g:04d}" for g in block5_groups]
    left_sim = np.r_[np.nan, unknown_adj]
    right_sim = np.r_[unknown_adj, np.nan]
    pseudo_df["left_adjacent_cosine"] = left_sim
    pseudo_df["right_adjacent_cosine"] = right_sim
    pseudo_df["forced_554_group_size"] = pseudo_df.groupby("pseudo_forced_554")["audio_file"].transform("size")
    pseudo_df["calibrated_group_size"] = pseudo_df.groupby("pseudo_calibrated")["audio_file"].transform("size")
    pseudo_df["block5_group_size"] = pseudo_df.groupby("pseudo_block5")["audio_file"].transform("size")

    within_unknown = unknown_adj[~boundary_mask]
    boundary_unknown = unknown_adj[boundary_mask]
    known_within_adj = known_adj[~known_true_boundary]
    known_boundary_adj = known_adj[known_true_boundary]

    summary = {
        "embedding_rows": int(len(clean)),
        "embedding_dim": int(emb.shape[1]),
        "known_files": int(known_mask.sum()),
        "unknown_files": int(unknown_mask.sum()),
        "known_loo_top1_accuracy": float(np.mean(known_pred == known_ids)),
        "known_loo_top5_accuracy": float(np.mean((known_top5 == known_true_idx[:, None]).any(axis=1))),
        "known_closed_set_macro_f1": float(np.mean(per_class_f1)),
        "known_files_negative_margin": int(np.sum(margins < 0)),
        "known_speakers_below_f1_0_8": int(np.sum(per_class_f1 < 0.8)),
        "known_speakers_perfect_f1": int(np.sum(np.isclose(per_class_f1, 1.0))),
        "verification_same_mean": float(same.mean()),
        "verification_cross_mean": float(cross.mean()),
        "verification_eer": eer,
        "verification_eer_threshold": eer_threshold,
        "ood_auc": ood_auc,
        "ood_fpr_at_tpr95": fpr_at_tpr95,
        "ood_youden_threshold": float(ood_thr[youden_i]),
        "ood_youden_tpr": float(ood_tpr[youden_i]),
        "ood_youden_fpr": float(ood_fpr[youden_i]),
        "corrected_best_macro_f1": float(macro_values[best_macro_i]),
        "corrected_best_macro_threshold": float(thresholds[best_macro_i]),
        "known_order_true_boundaries": n_known_boundaries,
        "known_order_boundary_precision_at_known_count": known_boundary_precision,
        "known_order_segmentation_ari": known_order_ari,
        "known_boundary_calibrated_threshold": calibrated_boundary_threshold,
        "known_boundary_calibrated_f1": float(threshold_f1[best_boundary_i]),
        "known_boundary_calibrated_precision": float(calibrated_precision),
        "known_boundary_calibrated_recall": float(calibrated_recall),
        "known_same_speaker_adjacent_cosine_mean": float(known_within_adj.mean()),
        "known_boundary_adjacent_cosine_mean": float(known_boundary_adj.mean()),
        "unknown_pseudo_speakers": int(pseudo_sizes.size),
        "unknown_pseudo_size_min": int(pseudo_sizes.min()),
        "unknown_pseudo_size_median": float(pseudo_sizes.median()),
        "unknown_pseudo_size_max": int(pseudo_sizes.max()),
        "unknown_pseudo_singletons": int((pseudo_sizes == 1).sum()),
        "unknown_pseudo_within_adjacent_cosine_mean": float(within_unknown.mean()),
        "unknown_pseudo_boundary_cosine_mean": float(boundary_unknown.mean()),
        "unknown_calibrated_speakers": int(calibrated_sizes.size),
        "unknown_calibrated_size_median": float(calibrated_sizes.median()),
        "unknown_calibrated_size_max": int(calibrated_sizes.max()),
        "unknown_calibrated_singletons": int((calibrated_sizes == 1).sum()),
        "unknown_calibrated_within_cosine_mean": float(unknown_adj[~calibrated_unknown_boundary].mean()),
        "unknown_calibrated_boundary_cosine_mean": float(unknown_adj[calibrated_unknown_boundary].mean()),
        "unknown_block5_speakers": int(block5_sizes.size),
        "unknown_block5_size_median": float(block5_sizes.median()),
        "unknown_block5_size_max": int(block5_sizes.max()),
        "unknown_block5_singletons": int((block5_sizes == 1).sum()),
        "unknown_block5_within_cosine_mean": float(unknown_adj[~block5_boundary].mean()),
        "unknown_block5_boundary_cosine_mean": float(unknown_adj[block5_boundary].mean()),
        "unknown_max_known_cosine_p95": float(np.quantile(unknown_sims.max(axis=1), 0.95)),
        "unknown_files_max_known_cosine_above_0_8": int(np.sum(unknown_sims.max(axis=1) > 0.8)),
    }

    # Values needed only for figures; kept outside JSON summary.
    plot_data = {
        "margins": margins,
        "known_ood_scores": ood_scores[known_mask],
        "unknown_ood_scores": ood_scores[unknown_mask],
        "known_within_adj": known_within_adj,
        "known_boundary_adj": known_boundary_adj,
        "unknown_within_adj": within_unknown,
        "unknown_boundary_adj": boundary_unknown,
        "pseudo_sizes": pseudo_sizes.to_numpy(),
        "calibrated_sizes": calibrated_sizes.to_numpy(),
        "block5_sizes": block5_sizes.to_numpy(),
    }
    return summary, diagnostics_df, nearest_pairs_df, pseudo_df, plot_data


def make_plots(inventory: pd.DataFrame, plot_data: dict) -> None:
    sns.set_theme(style="whitegrid")
    valid = inventory[inventory["duration_sec"] >= MIN_VALID_DURATION].copy()
    valid["Class"] = np.where(valid["is_unknown"], "Unknown", "Known")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, feature, label in zip(
        axes.flat,
        ["rms_dbfs", "active_frame_ratio", "silent_frame_ratio", "frame_dynamic_range_db"],
        ["RMS (dBFS)", "Active-frame ratio", "Silent-frame ratio", "Frame dynamic range (dB)"],
    ):
        sns.histplot(valid, x=feature, hue="Class", stat="density", common_norm=False, bins=50, element="step", fill=False, ax=ax)
        ax.set_xlabel(label)
    fig.suptitle("Corpus quality features: known vs unknown", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_QUALITY, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for values, label, color in [
        (plot_data["known_within_adj"], "Known: same speaker", "#2ca02c"),
        (plot_data["known_boundary_adj"], "Known: true boundary", "#d62728"),
        (plot_data["unknown_within_adj"], "Unknown: inferred within group", "#1f77b4"),
        (plot_data["unknown_boundary_adj"], "Unknown: inferred boundary", "#9467bd"),
    ]:
        sns.kdeplot(values, ax=ax, label=label, color=color, linewidth=2)
    ax.set_xlabel("Cosine similarity of adjacent rows")
    ax.set_title("Row order contains strong speaker-boundary signal", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ORDER, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    max_size = max(
        float(np.max(plot_data["pseudo_sizes"])),
        float(np.max(plot_data["calibrated_sizes"])),
        float(np.max(plot_data["block5_sizes"])),
    )
    bins = np.arange(0.5, max(12.5, max_size + 1.5), 1)
    ax.hist(plot_data["pseudo_sizes"], bins=bins, histtype="step", linewidth=2, label="Forced 554")
    ax.hist(plot_data["calibrated_sizes"], bins=bins, histtype="step", linewidth=2, label="Known-calibrated threshold")
    ax.hist(plot_data["block5_sizes"], bins=bins, histtype="step", linewidth=2, label="Within-run blocks of 5")
    ax.set_xlabel("Valid training files per inferred unknown identity")
    ax.set_ylabel("Inferred identities")
    ax.set_title("Unknown pseudo-identity hypotheses must remain separate", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PSEUDO, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].hist(plot_data["margins"], bins=60, color="#59a14f", alpha=0.85)
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_xlabel("LOO true-speaker cosine - nearest-impostor cosine")
    axes[0].set_title("Known-speaker decision margins", fontweight="bold")
    axes[1].hist(plot_data["known_ood_scores"], bins=60, alpha=0.65, density=True, label="Known")
    axes[1].hist(plot_data["unknown_ood_scores"], bins=60, alpha=0.65, density=True, label="Unknown")
    axes[1].set_xlabel("OOD score = 1 - max known-centroid cosine")
    axes[1].set_title("Correct LOO OOD-score distributions", fontweight="bold")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(PLOT_EMBEDDING, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(label: dict, audio: dict, embedding: dict) -> str:
    q = audio["known_vs_unknown_quality"]
    largest_effect = max(q.items(), key=lambda item: abs(item[1]["cohens_d"]))
    return f"""# گزارش مرجع درک عمیق داده - IAAA 2026 Speaker Identification

**تاریخ ممیزی:** 2026-08-27  
**منبع:** `data/raw` + EDAهای Phase 0 تا 3 + embedding منجمد ECAPA  
**هدف:** این فایل مرجع canonical مرحله Data Understanding است؛ اعداد مهم نباید دوباره از روی حدس یا گزارش‌های قدیمی استخراج شوند.

## 1. نتیجه اجرایی

1. مسئله واقعاً یک **open-set identification با 446 هویت known و یک خروجی تجمیعی unknown** است. راهنما می‌گوید 554 گوینده unknown نیز بین train/eval تقسیم شده‌اند، پس طبق spec صدای آن جمعیت کاملاً ندیده نیست؛ اما شمارش مشاهده‌شده train با این ادعا سازگار نیست و باید از برگزارکننده تأیید شود.
2. `labels.csv` شامل {label['rows']:,} ردیف است: {label['known_files']:,} known و {label['unknown_files']:,} unknown. هر {label['known_speakers']} گوینده known دقیقاً در یک بلوک پیوسته در ترتیب CSV قرار دارد ({label['known_speakers_in_exactly_one_run']}/{label['known_speakers']}). این ساختار، شاهد قوی حفظ ترتیب هویت‌های اصلی است.
3. هر {audio['declared_mp3_extension_files']:,} فایل پسوند `.mp3` دارند، ولی {audio['actual_riff_wave_files']:,} فایل واقعاً `RIFF/WAVE`، 16kHz و stereo هستند؛ فقط یک فایل MP3 واقعی 48kHz/mono است. دو کانال WAV عملاً یکسان‌اند (median correlation=1.0، side/mid≈{audio['stereo']['median_side_to_mid_db']:.1f}dB)، پس branch دوکاناله ارزش آزمایش GPU ندارد و mono امن است.
4. {audio['under_1s']} فایل کوتاه‌تر از 1 ثانیه‌اند؛ فقط {audio['empty_48_byte_headers']} مورد header خالی 48 بایتی‌اند و {audio['short_but_nonempty_under_1s']} فایل کوتاه ولی non-empty هستند. نامیدن همه آن‌ها به‌عنوان «corrupted» دقیق نیست.
5. بعد از فیلتر 1 ثانیه، {audio['valid_files']:,} فایل باقی می‌ماند. {audio['valid_known_speakers_below_5']} گوینده known کمتر از 5 فایل معتبر و {audio['valid_known_speakers_below_4']} گوینده کمتر از 4 فایل معتبر دارند؛ validation باید این ناهمگنی را صریحاً لحاظ کند.
6. ارزیابی تصحیح‌شده frozen ECAPA: top-1 LOO={embedding['known_loo_top1_accuracy']:.4f}، closed-set macro-F1={embedding['known_closed_set_macro_f1']:.4f}، و {embedding['known_speakers_below_f1_0_8']} هویت F1 زیر 0.8 دارند. مسئله اصلی در یک زیرمجموعه کوچک از speakerهای سخت متمرکز است.
7. EER صحیح verification برابر {embedding['verification_eer']:.4f} در threshold={embedding['verification_eer_threshold']:.4f} است. عدد 0.346 گزارش Phase 3 قدیمی threshold بود، نه EER.
8. segmentation صرفاً با ترتیب ردیف و cosine روی knownها، با تعداد مرز صحیح، boundary precision={embedding['known_order_boundary_precision_at_known_count']:.4f} و ARI={embedding['known_order_segmentation_ari']:.4f} می‌دهد. سیگنال واقعی است، اما برای اعلام ground truth کافی نیست؛ به‌ویژه چون 2275 فایل unknown دقیقاً `455×5` است و با عدد 554 راهنما ناسازگاری دارد.

## 2. صحت برچسب و ساختار فایل

| شاخص | مقدار |
|---|---:|
| ردیف برچسب | {label['rows']:,} |
| فایل صوتی روی دیسک | {label['audio_files_on_disk']:,} |
| فایل label‌شده مفقود | {label['missing_labelled_files']} |
| فایل بدون label | {label['unlabelled_audio_files']} |
| نام فایل تکراری در CSV | {label['duplicate_audio_names']} |
| گوینده known | {label['known_speakers']} |
| runهای unknown در CSV | {label['unknown_runs']} |
| بزرگ‌ترین run پیوسته unknown | {label['unknown_run_max']} |

ترتیب CSV random row order نیست. تمام نمونه‌های هر known speaker کنار هم آمده‌اند و runهای unknown زمانی طولانی می‌شوند که چند هویت unknown پشت سر هم قرار گرفته‌اند. این property باید در split، pseudo-label و تحلیل leakage حفظ و مستند شود.

## 3. کیفیت و فرمت صوت

| شاخص | مقدار |
|---|---:|
| RIFF/WAVE واقعی | {audio['actual_riff_wave_files']:,} |
| خطای decode/header | {audio['read_errors']} |
| کمتر از 1s | {audio['under_1s']} |
| header خالی 48B | {audio['empty_48_byte_headers']} |
| کوتاه ولی non-empty | {audio['short_but_nonempty_under_1s']} |
| exact duplicate group | {audio['exact_duplicate_groups']} |
| duplicate group دارای فایل معتبر | {audio['valid_exact_duplicate_groups']} |
| duplicate group با label متناقض (کل) | {audio['conflicting_exact_duplicate_groups']} |
| duplicate conflict دارای صوت معتبر | {audio['valid_conflicting_exact_duplicate_groups']} |
| median duration | {audio['duration_seconds']['median']:.2f}s |
| p95 duration | {audio['duration_seconds']['p95']:.2f}s |
| median stereo correlation | {audio['stereo']['median_channel_correlation']:.4f} |
| stereo correlation < 0.9 | {audio['stereo']['files_channel_correlation_below_0_9']} |
| channel RMS delta > 6dB | {audio['stereo']['files_channel_rms_delta_over_6db']} |

بزرگ‌ترین اختلاف low-level بین known/unknown مربوط به `{largest_effect[0]}` با Cohen's d={largest_effect[1]['cohens_d']:.3f} است. این اندازه اثر برای تصمیم هویتی قوی نیست؛ featureهای کیفیت باید برای QA/robustness استفاده شوند، نه به‌عنوان OOD shortcut. فهرست کامل exact duplicateها در `deep_exact_duplicate_groups.csv` است؛ مهم‌ترین conflict یک صوت 3.669s است که دقیقاً یکسان، یک‌بار known و یک‌بار unknown برچسب خورده و باید از scoring محلی/آموزش supervised پاک یا quarantine شود.

![quality](deep_quality_distributions.png)

## 4. هندسه embedding و سختی واقعی کلاس‌ها

| شاخص unbiased/corrected | مقدار |
|---|---:|
| known LOO top-1 | {embedding['known_loo_top1_accuracy']:.4f} |
| known LOO top-5 | {embedding['known_loo_top5_accuracy']:.4f} |
| known closed-set macro-F1 | {embedding['known_closed_set_macro_f1']:.4f} |
| speaker با F1 کامل | {embedding['known_speakers_perfect_f1']} / 446 |
| speaker با F1 < 0.8 | {embedding['known_speakers_below_f1_0_8']} |
| verification EER (نرخ، نه threshold) | {embedding['verification_eer']:.4f} |
| OOD AUC | {embedding['ood_auc']:.4f} |
| FPR در TPR>=0.95 | {embedding['ood_fpr_at_tpr95']:.4f} |
| corrected best direct Macro-F1 | {embedding['corrected_best_macro_f1']:.4f} |

فایل `deep_known_speaker_diagnostics.csv` هویت‌های سخت را بر اساس F1، margin و نزدیک‌ترین impostor رتبه‌بندی می‌کند. `deep_nearest_speaker_pairs.csv` نیز نزدیک‌ترین جفت centroidها را ثبت می‌کند. این دو فایل باید مبنای hard-negative mining و ارزیابی per-speaker باشند.

![embedding](deep_embedding_margin_ood.png)

## 5. بازسازی هویت‌های unknown از ترتیب داده

در راهنمای مسابقه صریحاً 554 گوینده unknown ذکر شده، اما train دقیقاً 2275 فایل unknown (`455×5`) دارد و 193 مورد از 204 run آن مضرب پنج‌اند. بنابراین سه hypothesis جدا نگه داشته شده‌اند: (الف) تحمیل 554 گروه مطابق spec، (ب) threshold مرز کالیبره‌شده روی known، و (ج) بلوک‌های پنج‌تایی داخل runهای unknown. قبل از اعمال threshold روی unknown، همان الگوریتم بدون label روی knownها سنجیده شد:

| اعتبارسنجی order-constrained segmentation | مقدار |
|---|---:|
| مرز واقعی known | {embedding['known_order_true_boundaries']} |
| boundary precision با تعداد مرز صحیح | {embedding['known_order_boundary_precision_at_known_count']:.4f} |
| Adjusted Rand Index | {embedding['known_order_segmentation_ari']:.4f} |
| calibrated boundary F1 | {embedding['known_boundary_calibrated_f1']:.4f} |
| calibrated precision / recall | {embedding['known_boundary_calibrated_precision']:.4f} / {embedding['known_boundary_calibrated_recall']:.4f} |
| mean cosine داخل بلوک known | {embedding['known_same_speaker_adjacent_cosine_mean']:.4f} |
| mean cosine روی مرز known | {embedding['known_boundary_adjacent_cosine_mean']:.4f} |

خروجی سه hypothesis:

| hypothesis | گروه | median size | singleton | within cosine | boundary cosine |
|---|---:|---:|---:|---:|---:|
| forced spec=554 | {embedding['unknown_pseudo_speakers']} | {embedding['unknown_pseudo_size_median']:.1f} | {embedding['unknown_pseudo_singletons']} | {embedding['unknown_pseudo_within_adjacent_cosine_mean']:.4f} | {embedding['unknown_pseudo_boundary_cosine_mean']:.4f} |
| known-calibrated threshold | {embedding['unknown_calibrated_speakers']} | {embedding['unknown_calibrated_size_median']:.1f} | {embedding['unknown_calibrated_singletons']} | {embedding['unknown_calibrated_within_cosine_mean']:.4f} | {embedding['unknown_calibrated_boundary_cosine_mean']:.4f} |
| run-local blocks of five | {embedding['unknown_block5_speakers']} | {embedding['unknown_block5_size_median']:.1f} | {embedding['unknown_block5_singletons']} | {embedding['unknown_block5_within_cosine_mean']:.4f} | {embedding['unknown_block5_boundary_cosine_mean']:.4f} |

هر سه ستون در `deep_unknown_pseudo_speakers.csv` ثبت شده‌اند. هیچ‌کدام فعلاً ground truth نیست. استفاده آموزشی باید hypothesis-specific ablation و confidence weighting داشته باشد؛ چیزی که EDA قطعی می‌کند فقط این است که فرض «unknown یک کلاس بدون ساختار داخلی است» نادرست است.

![order](deep_order_similarity.png)

![groups](deep_unknown_pseudo_group_sizes.png)

## 6. اصلاحات لازم نسبت به EDA قبلی

1. **Container mismatch:** گزارش قبلی raw را MP3 می‌نامید؛ 4528 فایل WAVE/PCM stereo با پسوند اشتباه و یک MP3 واقعی داریم.
2. **Short != corrupt:** threshold یک‌ثانیه یک policy است؛ {audio['short_but_nonempty_under_1s']} فایل non-empty کوتاه را نباید بدون سنجش contribution حذف قطعی نامید؛ {audio['empty_48_byte_headers']} فایل 48B عملاً خالی‌اند.
3. **Duplicate wording:** Phase 3 ادعا می‌کرد duplicateها حذف شده‌اند، اما `clean_labels` فقط `corrupted` را در drop-set می‌گذارد. آمار duplicate باید جداگانه گزارش شود.
4. **EER bug:** کد قدیمی مقدار threshold را با نام EER گزارش می‌کرد.
5. **Macro-F1 leakage:** شبیه‌سازی قدیمی OOD score را LOO می‌ساخت ولی speaker prediction را با centroid کامل (شامل خود نمونه) انجام می‌داد. عدد corrected این گزارش از prediction کاملاً LOO برای known استفاده می‌کند.
6. **KMeans=8 بدون مبنا:** specification تعداد 554 هویت unknown را می‌دهد و ترتیب CSV سیگنال مرز قوی دارد؛ 8 خوشه representation مناسبی از ساختار واقعی نیست.
7. **Validation risk:** یک holdout تصادفی از هر speaker برای انتخاب threshold کافی نیست؛ file/session/group-aware folds و گزارش dispersion بین foldها لازم است.

## 7. قرارداد استفاده در مراحل بعد

- برای integrity و کیفیت، `deep_audio_inventory.csv` مرجع است.
- برای hard speakers و hard negatives، دو CSV تشخیصی embedding مرجع‌اند.
- برای unknown identity-aware sampling، فقط `deep_unknown_pseudo_speakers.csv` با confidence/ablation استفاده شود.
- هر ادعای بهبود باید **OOF per-speaker Macro-F1، known recall، unknown precision/recall، و fold variance** را هم‌زمان گزارش کند.
- threshold نهایی نباید روی یک split یا روی train in-sample انتخاب شود.

## 8. محدودیت‌های این گزارش

- embedding ECAPA یک ابزار اندازه‌گیری است و ceiling معماری‌های بهتر نیست.
- pseudo-identityهای unknown ground truth رسمی ندارند؛ اعتبار آن‌ها از کنترل known و جدایی edgeها می‌آید.
- ویژگی active/silent یک energy proxy است، نه neural VAD.
- hidden eval قابل مشاهده نیست؛ هر نتیجه محلی باید در چند split گروه‌محور و سپس روی leaderboard تأیید شود.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-audio", action="store_true", help="Recompute the full audio inventory")
    args = parser.parse_args()

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    label_summary = label_and_order_audit(df)
    inventory = build_audio_inventory(df, refresh=args.refresh_audio)
    audio_summary, inventory = audio_audit(inventory)
    embedding_summary, diagnostics, nearest_pairs, pseudo, plot_data = embedding_audit(df, inventory)

    diagnostics.to_csv(KNOWN_DIAGNOSTICS_PATH, index=False)
    nearest_pairs.to_csv(NEAREST_PAIRS_PATH, index=False)
    pseudo.to_csv(UNKNOWN_PSEUDO_PATH, index=False)
    make_plots(inventory, plot_data)

    summary = {
        "generated_at": "2026-08-27",
        "label_and_order": label_summary,
        "audio": audio_summary,
        "embedding": embedding_summary,
        "known_diagnostics_csv": str(KNOWN_DIAGNOSTICS_PATH.relative_to(ROOT)),
        "nearest_pairs_csv": str(NEAREST_PAIRS_PATH.relative_to(ROOT)),
        "unknown_pseudo_csv": str(UNKNOWN_PSEUDO_PATH.relative_to(ROOT)),
        "exact_duplicates_csv": str(EXACT_DUPLICATES_PATH.relative_to(ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(generate_report(label_summary, audio_summary, embedding_summary), encoding="utf-8")

    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    print(f"wrote: {REPORT_PATH}")


if __name__ == "__main__":
    main()
