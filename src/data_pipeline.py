"""
Phase 1: Robust Data Pipeline for Open-Set Speaker Identification.
Handles stratified 5-shot split, augmentation, and weighted sampling.
"""

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, Sampler

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────
#  Configuration Loader
# ─────────────────────────────────────────────────────────

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_placeholders(value):
    """Resolve ``${VAR}`` placeholders to their env value ("" when unset).

    The config's ``mlops.tracking`` carries ``${DAGSHUB_USER_TOKEN}`` /
    ``${DAGSHUB_REPO_OWNER}`` placeholders that only get values at deploy time.
    ZenML substitutes ``${...}`` in step inputs and RAISES when the env var is
    absent — which breaks any direct ``@step`` call (the HPO/queue subprocess
    path) that doesn't export those vars. Resolve them eagerly here so the
    config dict handed to steps is always self-contained.
    """
    if isinstance(value, str):
        return _ENV_PLACEHOLDER.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value


def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    """Load YAML configuration (with ``${ENV_VAR}`` placeholders resolved)."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _resolve_env_placeholders(config)


def get_active_profile(config: dict) -> dict:
    """Return the active hardware profile (local or vastai)."""
    mode = config["hardware"]["mode"]
    profile = config["hardware"]["profiles"][mode].copy()
    profile.pop("description", None)
    return profile


# ─────────────────────────────────────────────────────────
#  Label Processing & Stratified Split
# ─────────────────────────────────────────────────────────

def create_class_mapping(labels_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create mapping: 'unknown' -> 0, known UUIDs -> 1..446.
    Returns dict mapping speaker_id -> integer class.
    """
    known_ids = sorted(
        labels_df[labels_df["speaker_id"] != "unknown"]["speaker_id"].unique()
    )
    mapping = {"unknown": 0}
    for idx, sid in enumerate(known_ids, start=1):
        mapping[sid] = idx
    return mapping


def apply_unknown_cluster_labels(
    labels_df: pd.DataFrame,
    cluster_map: Optional[Dict[str, int]],
) -> Tuple[pd.DataFrame, dict]:
    """Rewrite 'unknown' rows to pseudo cluster ids for the closed-set 1000-class
    experiment (see ``src/unknown_clustering.py``).

    Files listed in ``cluster_map`` ({audio_file: cluster_id}) become
    pseudo-speakers ``unknown_<n>`` (4-digit zero-padded). Because UUID strings
    sort before "unknown_*" lexicographically, ``create_class_mapping`` assigns
    them ids 447..1000 after the 446 known speakers. Files NOT in the map keep
    the real ``unknown`` label (→ 0), i.e. they stay genuinely-OOD for the head.

    Returns:
        (labels_df copy, stats dict)
    """
    if not cluster_map:
        return labels_df, {"n_rewritten": 0, "n_clusters": 0}

    df = labels_df.copy()
    # Keep the competition target independent from the metric-learning target.
    # ``speaker_id`` remains the backwards-compatible metric identity used by
    # the split/class-map code, while these columns preserve ground truth.
    if "original_speaker_id" not in df:
        df["original_speaker_id"] = df["speaker_id"]
    if "is_ood" not in df:
        df["is_ood"] = df["original_speaker_id"].eq("unknown").astype("int8")
    unk = df["is_ood"].astype(bool)
    mapped = df.loc[unk, "audio_file"].map(cluster_map)
    mask = unk & mapped.notna()
    df.loc[mask, "speaker_id"] = "unknown_" + (
        mapped[mask].astype(int).astype(str).str.zfill(4)
    )
    return df, {
        "n_rewritten": int(mask.sum()),
        "n_clusters": len(set(cluster_map.values())),
    }


def ensure_target_columns(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the explicit dual-target contract used by hybrid training.

    ``metric_label`` is the ArcFace/prototype identity (known or pseudo-OOD),
    while ``is_ood`` always reflects the original competition label.  ``label``
    is retained as an alias for old scripts/checkpoints.
    """
    df = labels_df.copy()
    if "original_speaker_id" not in df:
        df["original_speaker_id"] = df["speaker_id"]
    if "is_ood" not in df:
        df["is_ood"] = df["original_speaker_id"].eq("unknown").astype("int8")
    return df


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_unknown_cluster_map(config: dict) -> Optional[Dict[str, int]]:
    """Load + validate the pseudo-identity cluster map (closed-set 1000-class).

    Called wherever the train/val split is built when
    ``model.num_unknown_clusters > 0``. Returns ``None`` (legacy 447-way) when
    cluster mode is off.

    Validation is the precision guarantee for the UI's k knob: the requested k
    must equal the number of DISTINCT cluster ids in the map. Changing k in the
    config without rebuilding the map would silently misalign the speaker-head
    width with the collapse columns (the 447-way output contract breaks), so it
    is a hard error with rebuild instructions instead.

    When the configured map path is absent, falls back to the committed
    ``submission/<map-basename>`` (a fresh Vast.ai instance clones the repo but
    has no ``data/processed`` — the committed maps are the durable copies).
    Maps are k-locked (``unknown_clusters_k<k>.json``), so each experiment's
    own filename finds its own committed copy.
    """
    model_cfg = config.get("model", {}) or {}
    k = int(model_cfg.get("num_unknown_clusters", 0))
    if k <= 0:
        return None

    import json

    map_path = Path(str(model_cfg.get(
        "unknown_cluster_path", "data/processed/unknown_clusters.json",
    )))
    if not map_path.exists():
        fallback = _PROJECT_ROOT / "submission" / map_path.name
        if fallback.exists():
            print(f"  ⚠ {map_path} missing — using committed {fallback}")
            map_path = fallback
        else:
            raise FileNotFoundError(
                f"model.num_unknown_clusters={k} but cluster map not found: "
                f"{map_path}. Run `python -m src.unknown_clustering build "
                f"--k {k} --out <path>` (or the UI: Config → Cluster Mode → "
                f"Rebuild) first, and commit submission/{map_path.name}."
            )

    with open(map_path, "r", encoding="utf-8") as f:
        cluster_map = {file: int(cid) for file, cid in json.load(f).items()}
    n_clusters = len(set(cluster_map.values()))
    if n_clusters != k:
        raise ValueError(
            f"model.num_unknown_clusters={k} but the cluster map {map_path} "
            f"contains {n_clusters} distinct cluster ids. Rebuild the map at "
            f"the requested k: `python -m src.unknown_clustering build "
            f"--k {k} --checkpoint <ckpt>` (or the UI: Config → Cluster Mode "
            f"→ Rebuild clusters)."
        )
    return cluster_map


def find_duplicate_groups(
    labels_df: pd.DataFrame,
    audio_dir: str,
) -> Dict[str, List[str]]:
    """
    Group audio files with identical byte content (streaming MD5, 1 MB chunks).

    Only MD5 digests shared by more than one file are returned — these are the
    near-certain duplicate recordings that can leak across a random split.

    Args:
        labels_df: DataFrame with an `audio_file` column.
        audio_dir: Directory containing the (converted) audio files.

    Returns:
        dict: md5 hex digest -> sorted list of audio_file names sharing it.
    """
    import hashlib

    audio_dir = Path(audio_dir)
    md5_to_files: Dict[str, List[str]] = {}

    for fname in sorted(labels_df["audio_file"].unique()):
        fpath = audio_dir / fname
        hasher = hashlib.md5()
        try:
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError:
            continue  # missing/unreadable — handled by find_corrupted_files
        md5_to_files.setdefault(hasher.hexdigest(), []).append(fname)

    return {md5: fnames for md5, fnames in md5_to_files.items() if len(fnames) > 1}


def identify_conflicting_duplicates(
    labels_df: pd.DataFrame,
    duplicate_groups: Dict[str, List[str]],
) -> Set[str]:
    """
    Return the set of audio_files whose MD5-duplicate group carries MORE than
    one distinct speaker label (byte-identical audio → contradictory supervision).

    These files are the label-noise hazard described in R7: the same waveform is
    labelled with 2+ different speaker ids (and/or `unknown`), so training on
    them teaches the ArcFace and OOD heads mutually exclusive targets.
    """
    conflicting: Set[str] = set()
    for fnames in duplicate_groups.values():
        lbls = set(labels_df.loc[labels_df["audio_file"].isin(fnames), "speaker_id"])
        if len(lbls) > 1:
            conflicting.update(fnames)
    return conflicting


def clean_conflicting_labels(
    labels_df: pd.DataFrame,
    audio_dir: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove label noise from MD5-duplicate groups (Q5 quick win).

    - Conflicting-label groups (same bytes, 2+ distinct labels) are **dropped
      entirely** — their supervision is irreconcilable.
    - Non-conflicting groups (same bytes, one label) are deduplicated to a
      **single copy** (the lexicographically first file).

    Args:
        labels_df: DataFrame with ``audio_file`` + ``speaker_id`` columns.
        audio_dir: Directory containing the (converted) audio files.

    Returns:
        cleaned_df, stats  (stats: counts of dropped/deduped files)
    """
    dup_groups = find_duplicate_groups(labels_df, audio_dir)
    conflicting = identify_conflicting_duplicates(labels_df, dup_groups)

    drop: Set[str] = set(conflicting)
    dedupe_drop: Set[str] = set()
    for fnames in dup_groups.values():
        lbls = set(labels_df.loc[labels_df["audio_file"].isin(fnames), "speaker_id"])
        if len(lbls) == 1:
            ordered = sorted(fnames)
            dedupe_drop.update(ordered[1:])  # keep first copy, drop the rest

    drop |= dedupe_drop
    cleaned = labels_df[~labels_df["audio_file"].isin(drop)].reset_index(drop=True)

    stats = {
        "n_raw_files": int(len(labels_df)),
        "n_conflicting_files_dropped": int(len(conflicting)),
        "n_nonconflicting_duplicates_dropped": int(len(dedupe_drop)),
        "n_files_after_clean": int(len(cleaned)),
    }
    return cleaned, stats


def scan_durations(labels_df: pd.DataFrame, audio_dir: str) -> Dict[str, float]:
    """
    Header-only duration scan (soundfile.info) for every labelled file.

    Returns {audio_file: duration_seconds}; unreadable/missing files get 0.0
    (they are reported as corrupted by find_corrupted_files).
    """
    import soundfile as sf

    audio_dir = Path(audio_dir)
    durations: Dict[str, float] = {}
    for fname in labels_df["audio_file"].unique():
        fpath = audio_dir / fname
        try:
            durations[fname] = sf.info(str(fpath)).duration
        except Exception:
            durations[fname] = 0.0
    return durations


def find_corrupted_files(
    labels_df: pd.DataFrame,
    audio_dir: str,
    min_valid_duration: float = 1.0,
) -> List[str]:
    """
    Return the list of audio_files that are missing, unreadable, or shorter
    than `min_valid_duration` seconds (header-only soundfile.info check).
    """
    import soundfile as sf

    audio_dir = Path(audio_dir)
    corrupted: List[str] = []
    for fname in labels_df["audio_file"].unique():
        fpath = audio_dir / fname
        if not fpath.exists():
            corrupted.append(fname)
            continue
        try:
            if sf.info(str(fpath)).duration < min_valid_duration:
                corrupted.append(fname)
        except Exception:
            corrupted.append(fname)
    return corrupted


def stratified_split(
    labels_df: pd.DataFrame,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    random_seed: int = 42,
    duplicate_groups: Optional[Dict[str, List[str]]] = None,
    corrupted_files: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-aware stratified split.

    - Corrupted files (via `corrupted_files`) are dropped entirely.
    - Files that belong to an MD5-duplicate group are **never** put in val, so
      byte-identical recordings cannot straddle the train/val boundary.
      A known speaker whose files are all duplicated is excluded from val
      (with a warning) to keep val strictly duplicate-free.
    - Known speakers: exactly `val_per_known` non-duplicate files → val, rest → train.
    - 'unknown' class: `unknown_val_ratio` (of non-duplicate files) → val.
    """
    rng = np.random.default_rng(random_seed)
    df = labels_df.copy()
    if corrupted_files:
        df = df[~df["audio_file"].isin(corrupted_files)].reset_index(drop=True)

    # Files that are byte-identical duplicates must never appear in val
    dup_files: Set[str] = set()
    if duplicate_groups:
        for fnames in duplicate_groups.values():
            dup_files.update(fnames)

    train_rows, val_rows = [], []

    # Splitting must follow the original competition identity, never the
    # pseudo-label. Otherwise applying a cluster map changes the partition and
    # fold-specific maps leak their own validation files back into training.
    split_col = "original_speaker_id" if "original_speaker_id" in df else "speaker_id"
    df_known = df[df[split_col] != "unknown"]
    df_unknown = df[df[split_col] == "unknown"]

    # Known speakers: val from NON-duplicate files only
    for speaker_id, group in df_known.groupby(split_col):
        group = group.reset_index(drop=True)
        n = len(group)
        dup_mask = group["audio_file"].isin(dup_files).values
        non_dup_idx = np.where(~dup_mask)[0]
        n_val = min(val_per_known, n - 1)  # ensure at least 1 train

        if len(non_dup_idx) >= n_val:
            chosen = rng.choice(non_dup_idx, size=n_val, replace=False)
            val_mask = np.zeros(n, dtype=bool)
            val_mask[chosen] = True
        else:
            # Speaker's files are (mostly) duplicated → keep val duplicate-free:
            # the whole speaker goes to train.
            print(f"  ⚠ Speaker {speaker_id[:8]}… has {int(dup_mask.sum())}/{n} "
                  f"duplicated files — excluded from val (val kept duplicate-free)")
            val_mask = np.zeros(n, dtype=bool)

        val_rows.append(group[val_mask])
        train_rows.append(group[~val_mask])

    # Unknown class: ratio split, but duplicate files stay in train
    n_unknown = len(df_unknown)
    cand_idx = np.where(~df_unknown["audio_file"].isin(dup_files).values)[0]
    n_val_unknown = min(int(n_unknown * unknown_val_ratio), len(cand_idx))
    val_idx = (rng.choice(cand_idx, size=n_val_unknown, replace=False)
               if n_val_unknown > 0 else np.array([], dtype=int))
    val_mask = np.zeros(n_unknown, dtype=bool)
    val_mask[val_idx] = True
    val_rows.append(df_unknown[val_mask])
    train_rows.append(df_unknown[~val_mask])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    return train_df, val_df


def speaker_aware_kfold(
    labels_df: pd.DataFrame,
    folds: int = 3,
    random_seed: int = 42,
    duplicate_groups: Optional[Dict[str, List[str]]] = None,
    corrupted_files: Optional[Set[str]] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Speaker-aware K-fold split for out-of-fold (OOF) evaluation (root cause R4/C5).

    The single 1-file-per-speaker val split makes epoch/threshold/fusion
    selection ride on ~891 samples of a single seed. K-fold instead validates
    **every file exactly once** (leave-one-group-out), producing ~all-train
    OOF predictions for stable tuning, while each fold keeps ~(K-1)/K of a
    speaker's files for training (few-shot preserved: with 5 files and K=3,
    each fold trains on ~3-4 files).

    Rules (same leak guards as `stratified_split`):
      - corrupted files are dropped entirely;
      - MD5-duplicate files never appear in a val fold (they always stay in
        train) so byte-identical recordings cannot leak across the boundary.

    Args:
        labels_df: DataFrame with ``audio_file`` + ``speaker_id`` + ``label``.
        folds:     number of folds (K).
        random_seed: RNG seed for the per-speaker/class partition.
        duplicate_groups: MD5 groups (see find_duplicate_groups).
        corrupted_files:  set of audio_files to drop.

    Returns:
        list of ``(train_df, val_df)`` — one pair per fold.
    """
    folds = max(1, int(folds))
    rng = np.random.default_rng(random_seed)
    df = labels_df.copy()
    if corrupted_files:
        df = df[~df["audio_file"].isin(corrupted_files)].reset_index(drop=True)

    dup_files: Set[str] = set()
    if duplicate_groups:
        for fnames in duplicate_groups.values():
            dup_files.update(fnames)

    fold_train_rows: List[List[pd.DataFrame]] = [[] for _ in range(folds)]
    fold_val_rows: List[List[pd.DataFrame]] = [[] for _ in range(folds)]

    def _partition(indices: np.ndarray) -> List[np.ndarray]:
        """Split `indices` into `folds` groups as evenly as possible."""
        idx = rng.permutation(indices)
        groups: List[List[int]] = [[] for _ in range(folds)]
        for i, x in enumerate(idx):
            groups[i % folds].append(int(x))
        return [np.asarray(g, dtype=int) for g in groups]

    # ── Known speakers ──
    split_col = "original_speaker_id" if "original_speaker_id" in df else "speaker_id"
    df_known = df[df[split_col] != "unknown"]
    for _, group in df_known.groupby(split_col):
        group = group.reset_index(drop=True)
        non_dup_idx = np.where(~group["audio_file"].isin(dup_files).values)[0]
        groups = _partition(non_dup_idx)
        for f in range(folds):
            val_idx = groups[f]
            val_files = set(group.iloc[val_idx]["audio_file"])
            train_mask = ~group["audio_file"].isin(val_files).values
            fold_val_rows[f].append(group.iloc[val_idx])
            fold_train_rows[f].append(group[train_mask])

    # ── Unknown class ──
    df_unknown = df[df[split_col] == "unknown"]
    unknown_idx = np.where(~df_unknown["audio_file"].isin(dup_files).values)[0]
    groups = _partition(unknown_idx)
    for f in range(folds):
        val_idx = groups[f]
        val_files = set(df_unknown.iloc[val_idx]["audio_file"])
        train_mask = ~df_unknown["audio_file"].isin(val_files).values
        fold_val_rows[f].append(df_unknown.iloc[val_idx])
        fold_train_rows[f].append(df_unknown[train_mask])

    splits = []
    for f in range(folds):
        train_df = pd.concat(fold_train_rows[f], ignore_index=True)
        val_df = pd.concat(fold_val_rows[f], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=rng).reset_index(drop=True)
        splits.append((train_df, val_df))
    return splits


def _write_split_report(
    labels_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    duplicate_groups: Dict[str, List[str]],
    corrupted: List[str],
    durations: Dict[str, float],
    output_path: str,
) -> None:
    """
    Write data/processed/split_report.json: corrupted files (known/unknown),
    MD5-duplicate groups (incl. conflicting-label groups), and per-known-speaker
    train/val counts + usable seconds.
    """
    import json

    split_col = ("original_speaker_id" if "original_speaker_id" in labels_df
                 else "speaker_id")
    corrupted_known = int(labels_df[
        labels_df["audio_file"].isin(corrupted) & (labels_df[split_col] != "unknown")
    ].shape[0])
    corrupted_unknown = int(labels_df[
        labels_df["audio_file"].isin(corrupted) & (labels_df[split_col] == "unknown")
    ].shape[0])

    groups_info = []
    n_conflicting = 0
    for md5, fnames in duplicate_groups.items():
        lbls = sorted(set(labels_df.loc[labels_df["audio_file"].isin(fnames), "speaker_id"]))
        conflicting = len(lbls) > 1
        if conflicting:
            n_conflicting += 1
        groups_info.append({
            "md5": md5,
            "n_files": len(fnames),
            "files": fnames,
            "labels": lbls,
            "conflicting_labels": conflicting,
        })

    train_files = set(train_df["audio_file"])
    per_speaker = {}
    for sid, group in labels_df[labels_df[split_col] != "unknown"].groupby(split_col):
        good = group[~group["audio_file"].isin(corrupted)]
        per_speaker[sid] = {
            "train_files": int(group[group["audio_file"].isin(train_files)].shape[0]),
            "val_files": int(group[~group["audio_file"].isin(train_files)].shape[0]),
            "usable_seconds": round(float(sum(durations.get(f, 0.0) for f in good["audio_file"])), 2),
        }

    report = {
        "corrupted_files": {
            "total": len(corrupted),
            "known": corrupted_known,
            "unknown": corrupted_unknown,
            "files": corrupted,
        },
        "duplicate_groups": {
            "total_groups": len(duplicate_groups),
            "total_files": sum(len(v) for v in duplicate_groups.values()),
            "conflicting_label_groups": n_conflicting,
            "groups": groups_info,
        },
        "per_known_speaker": per_speaker,
        "split_summary": {
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "train_known": int((train_df.get("is_ood", train_df["label"].eq(0)) == 0).sum()),
            "val_known": int((val_df.get("is_ood", val_df["label"].eq(0)) == 0).sum()),
            "train_unknown": int((train_df.get("is_ood", train_df["label"].eq(0)) == 1).sum()),
            "val_unknown": int((val_df.get("is_ood", val_df["label"].eq(0)) == 1).sum()),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Split report saved to {output_path}")


def _write_kfold_report(
    labels_df: pd.DataFrame,
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    duplicate_groups: Dict[str, List[str]],
    corrupted: List[str],
    output_path: str,
) -> None:
    """Write data/processed/split_report.json for a speaker-aware K-fold split."""
    import json

    fold_summaries = []
    for f, (train_df, val_df) in enumerate(splits):
        fold_summaries.append({
            "fold": f,
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "train_known": int((train_df.get("is_ood", train_df["label"].eq(0)) == 0).sum()),
            "val_known": int((val_df.get("is_ood", val_df["label"].eq(0)) == 0).sum()),
            "train_unknown": int((train_df.get("is_ood", train_df["label"].eq(0)) == 1).sum()),
            "val_unknown": int((val_df.get("is_ood", val_df["label"].eq(0)) == 1).sum()),
        })

    report = {
        "scheme": "speaker_aware_kfold",
        "n_folds": len(splits),
        "corrupted_files": {
            "total": len(corrupted),
            "files": corrupted,
        },
        "duplicate_groups": {
            "total_groups": len(duplicate_groups),
            "total_files": sum(len(v) for v in duplicate_groups.values()),
        },
        "folds": fold_summaries,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ K-fold split report saved to {output_path}")


def split_args_from_config(config: dict) -> dict:
    """Extract split kwargs from ``config['data']['split']`` for `prepare_clean_split`.

    Checkpoints embed their training-time split (scheme/fold/folds/seed), so any
    downstream artifact builder (val-prob dump, centroid build, ensemble collect)
    must reuse the SAME partition — otherwise a kfold-trained checkpoint would be
    validated on the default single seed-42 split (silent OOF mismatch).
    """
    split = (config.get("data", {}) or {}).get("split", {}) or {}
    return {
        "split_scheme": str(split.get("scheme", "single")).lower().strip(),
        "fold": int(split.get("fold", 0)),
        "folds": int(split.get("folds", 3)),
        "random_seed": int(split.get("seed", 42)),
    }


def prepare_clean_split(
    labels_path: str,
    audio_dir: str,
    processed_labels: str,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    min_valid_duration: float = 1.0,
    random_seed: int = 42,
    split_report_path: str = "data/processed/split_report.json",
    split_scheme: str = "single",
    fold: int = 0,
    folds: int = 3,
    unknown_cluster_map: Optional[Dict[str, int]] = None,
    clean_duplicates: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load, clean and leak-free split labels; write data/processed/split_report.json.

    Pipeline:
      1. Load & clean labels (drops exact CSV duplicate rows / NaN).
      2. (Optional) rewrite ``unknown`` rows to pseudo cluster ids
         (``unknown_cluster_map`` — closed-set 1000-class experiment).
      3. Scan durations (header-only) for every labelled file.
      4. Detect corrupted (< min_valid_duration) / missing files.
      5. Detect MD5-duplicate groups (incl. conflicting-label groups).
      6. Leak-free split: ``single`` (default) or ``kfold`` (speaker-aware
         K-fold; returns the ``fold``-th split for OOF training).
      7. Save cleaned labels and split_report.json (single scheme only).

    Returns:
        train_df, val_df, class_map
    """
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["speaker_id", "audio_file"]).reset_index(drop=True)

    # Preserve the original binary target before pseudo-identity relabelling.
    df = ensure_target_columns(df)

    # ── Leak-aware scans ──
    print(f"  Scanning durations ({len(df):,} files)...")
    durations = scan_durations(df, audio_dir)
    print("  Detecting corrupted files...")
    corrupted = find_corrupted_files(df, audio_dir, min_valid_duration)
    if corrupted:
        print(f"  ⚠ {len(corrupted)} corrupted/short files (< {min_valid_duration}s)")
    print("  Detecting MD5-duplicate groups...")
    dup_groups = find_duplicate_groups(df, audio_dir)
    if dup_groups:
        print(f"  ⚠ {len(dup_groups)} duplicate groups "
              f"({sum(len(v) for v in dup_groups.values())} files)")
    if clean_duplicates and dup_groups:
        before = len(df)
        df, duplicate_stats = clean_conflicting_labels(df, audio_dir)
        print(f"  🧹 Duplicate cleaning: {before - len(df)} files removed "
              f"({duplicate_stats['n_conflicting_files_dropped']} conflicting, "
              f"{duplicate_stats['n_nonconflicting_duplicates_dropped']} repeated)")

    # Apply pseudo identities only AFTER duplicate/conflict cleaning. Otherwise
    # byte-identical unknown files assigned to two clusters look like a false
    # label conflict even though their original competition label agrees.
    if unknown_cluster_map:
        df, cluster_stats = apply_unknown_cluster_labels(df, unknown_cluster_map)
        print(f"  🧬 Unknown clusters applied: "
              f"{cluster_stats['n_rewritten']} files → "
              f"{cluster_stats['n_clusters']} pseudo-identities")

    # Create the metric class map after all label rewrites.
    class_map = create_class_mapping(df)
    df["metric_label"] = df["speaker_id"].map(class_map).astype(int)
    df["label"] = df["metric_label"]  # backward-compatible alias

    # ── Leak-free split ──
    scheme = str(split_scheme).lower().strip()
    if scheme == "kfold":
        kfold_splits = speaker_aware_kfold(
            df,
            folds=folds,
            random_seed=random_seed,
            duplicate_groups=dup_groups,
            corrupted_files=set(corrupted),
        )
        fold_idx = max(0, min(int(fold), len(kfold_splits) - 1))
        train_df, val_df = kfold_splits[fold_idx]
        print(f"  ✓ K-fold split (fold {fold_idx}/{folds}, "
              f"train={len(train_df)}, val={len(val_df)})")
    elif scheme == "full":
        # Final retrain: every usable file contributes to optimisation.  Keep a
        # deterministic overlapping diagnostic set so existing monitoring code
        # can detect catastrophic failures; it MUST NOT select the epoch/model.
        usable = df[~df["audio_file"].isin(set(corrupted))].reset_index(drop=True)
        _, val_df = stratified_split(
            usable,
            val_per_known=val_per_known,
            unknown_val_ratio=unknown_val_ratio,
            random_seed=random_seed,
            duplicate_groups=dup_groups,
            corrupted_files=set(),
        )
        train_df = usable.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        print(f"  ⚠ FULL-DATA mode: train={len(train_df)} uses every usable file; "
              f"val={len(val_df)} overlaps train and is diagnostic only. "
              f"Checkpoint selection is forced to the final epoch.")
    else:
        train_df, val_df = stratified_split(
            df,
            val_per_known=val_per_known,
            unknown_val_ratio=unknown_val_ratio,
            random_seed=random_seed,
            duplicate_groups=dup_groups,
            corrupted_files=set(corrupted),
        )

    # ── Save cleaned labels ──
    os.makedirs(os.path.dirname(processed_labels), exist_ok=True)
    df.to_csv(processed_labels, index=False)
    print(f"  ✓ Saved cleaned labels ({len(df)} rows) to {processed_labels}")

    # ── Split report (single-scheme only; kfold writes a per-fold summary) ──
    if scheme == "kfold":
        _write_kfold_report(df, kfold_splits, dup_groups, corrupted, split_report_path)
    else:
        _write_split_report(
            df, train_df, val_df, dup_groups, corrupted, durations, split_report_path,
        )

    print(f"  ✓ Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(
        f"    Train known: {(train_df['is_ood'] == 0).sum()} | "
        f"Train unknown: {(train_df['is_ood'] == 1).sum()}"
    )
    print(
        f"    Val known: {(val_df['is_ood'] == 0).sum()} | "
        f"Val unknown: {(val_df['is_ood'] == 1).sum()}"
    )

    return train_df, val_df, class_map


def prepare_labels(
    labels_path: str,
    output_path: str,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    audio_dir: Optional[str] = None,
    min_valid_duration: float = 1.0,
    split_scheme: str = "single",
    fold: int = 0,
    folds: int = 3,
    unknown_cluster_map: Optional[Dict[str, int]] = None,
    clean_duplicates: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load, clean, split labels and create class mapping.
    Saves cleaned labels to output_path.

    If `audio_dir` is given, the leak-free pipeline (corrupted/duplicate
    detection + split_report.json) is used; otherwise a plain stratified split.
    """
    if audio_dir is not None:
        return prepare_clean_split(
            labels_path=labels_path,
            audio_dir=audio_dir,
            processed_labels=output_path,
            val_per_known=val_per_known,
            unknown_val_ratio=unknown_val_ratio,
            min_valid_duration=min_valid_duration,
            split_scheme=split_scheme,
            fold=fold,
            folds=folds,
            unknown_cluster_map=unknown_cluster_map,
            clean_duplicates=clean_duplicates,
        )

    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["speaker_id", "audio_file"]).reset_index(drop=True)

    df = ensure_target_columns(df)

    # Create class mapping
    class_map = create_class_mapping(df)
    df["metric_label"] = df["speaker_id"].map(class_map).astype(int)
    df["label"] = df["metric_label"]

    # Save cleaned labels
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved cleaned labels ({len(df)} rows) to {output_path}")

    # Stratified split
    train_df, val_df = stratified_split(df, val_per_known, unknown_val_ratio)
    print(f"  ✓ Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(
        f"    Train known: {(train_df['is_ood'] == 0).sum()} | "
        f"Train unknown: {(train_df['is_ood'] == 1).sum()}"
    )
    print(
        f"    Val known: {(val_df['is_ood'] == 0).sum()} | "
        f"Val unknown: {(val_df['is_ood'] == 1).sum()}"
    )

    return train_df, val_df, class_map


# ─────────────────────────────────────────────────────────
#  Audio Augmentation Pipeline
# ─────────────────────────────────────────────────────────

def _mp3_backend_available() -> bool:
    """True if the mp3 codec roundtrip's backend (lameenc) is installed.

    ``audiomentations.Mp3Compression`` is created with ``backend="lameenc"``
    (the only cross-platform wheel — fast_mp3_augment has no Windows wheel and
    pydub needs a system ffmpeg). Detect it up front so a missing backend
    produces one clear warning instead of an ImportError mid-training.
    """
    try:
        import lameenc  # noqa: F401
        return True
    except Exception:
        return False


def _make_background_noise(sounds_path: str, snr, p: float):
    """AddBackgroundNoise with version-compatible SNR kwargs.

    audiomentations renamed ``min_snr_in_db``/``max_snr_in_db`` →
    ``min_snr_db``/``max_snr_db`` in 0.33.0, so the keyword name depends on the
    installed version (the server resolved a newer one than the local venv).
    """
    import inspect

    import audiomentations as AA

    params = inspect.signature(AA.AddBackgroundNoise.__init__).parameters
    if "min_snr_in_db" in params:
        return AA.AddBackgroundNoise(
            sounds_path=sounds_path,
            min_snr_in_db=snr[0], max_snr_in_db=snr[1], p=p,
        )
    return AA.AddBackgroundNoise(
        sounds_path=sounds_path,
        min_snr_db=snr[0], max_snr_db=snr[1], p=p,
    )


class AudioAugmentation:
    """
    Training-time augmentation pipeline (config-driven — root cause R8).

    Reads the ``augmentation`` block of the config:

      - ``waveform``: gentle waveform-level effects (gaussian noise, gain,
        polarity, shift, pitch, time-stretch) — safe for frozen encoders.
      - ``domain``:   RIR reverb, MUSAN noise/music, mp3 codec roundtrip — the
        highest-value additions for generalization to the competition's
        recording conditions (C2). These are SKIPPED with a warning when their
        data dirs / backends are absent, so training never crashes on a
        machine without MUSAN/RIR/ffmpeg.
      - ``spec``:     time-domain masking (a waveform approximation of
        SpecAugment's time masking). Frequency masking needs spectrogram
        access and is left out — the encoders consume raw waveforms.

    Defaults reproduce the previous hardcoded pipeline exactly when the config
    block is absent, so this refactor is backward-compatible.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        augmentation_config: Optional[dict] = None,
    ):
        import audiomentations as AA

        self.sample_rate = sample_rate
        self._warned = False
        cfg = augmentation_config or {}
        wf = cfg.get("waveform", {}) or {}
        domain = cfg.get("domain", {}) or {}
        spec = cfg.get("spec", {}) or {}

        def _p(block: dict, key: str, default: float) -> float:
            sub = block.get(key)
            return float(sub.get("p", default)) if isinstance(sub, dict) else default

        def _range(block: dict, key: str, field: str, default):
            sub = block.get(key)
            if isinstance(sub, dict) and field in sub:
                return sub[field]
            return default

        transforms = []

        def _add(fn):
            """Append a transform, degrading gracefully when a data dir or
            backend (ffmpeg/lameenc/pydub) is unavailable."""
            try:
                transforms.append(fn())
            except Exception as e:  # pragma: no cover — environment-dependent
                print(f"  ⚠ Skipping augmentation: {e}")

        # ── Waveform-level (gentle) ──
        if _p(wf, "gaussian_noise", 0.5) > 0:
            amp = _range(wf, "gaussian_noise", "amp", [0.001, 0.015]) or [0.001, 0.015]
            _add(lambda amp=amp, p=_p(wf, "gaussian_noise", 0.5):
                 AA.AddGaussianNoise(min_amplitude=amp[0], max_amplitude=amp[1], p=p))
        if _p(wf, "gain", 0.3) > 0:
            db = _range(wf, "gain", "db", [-6, 6]) or [-6, 6]
            _add(lambda db=db, p=_p(wf, "gain", 0.3):
                 AA.Gain(min_gain_db=db[0], max_gain_db=db[1], p=p))
        if _p(wf, "polarity_inversion", 0.5) > 0:
            _add(lambda p=_p(wf, "polarity_inversion", 0.5): AA.PolarityInversion(p=p))
        if _p(wf, "shift", 0.3) > 0:
            frac = float((wf.get("shift") or {}).get("frac", 0.1))
            _add(lambda frac=frac, p=_p(wf, "shift", 0.3):
                 AA.Shift(min_shift=-frac, max_shift=frac, shift_unit="fraction",
                          rollover=True, fade_duration=0.005, p=p))
        if _p(wf, "pitch_shift", 0.3) > 0:
            # Gentle pitch shift only — a frozen encoder can't adapt to ±4
            # semitones (it caused the inverted train/val gap in the last run).
            st = _range(wf, "pitch_shift", "semitones", [-1, 1]) or [-1, 1]
            _add(lambda st=st, p=_p(wf, "pitch_shift", 0.3):
                 AA.PitchShift(min_semitones=st[0], max_semitones=st[1], p=p))
        if _p(wf, "time_stretch", 0.2) > 0:
            rate = _range(wf, "time_stretch", "rate", [0.8, 1.25]) or [0.8, 1.25]
            _add(lambda rate=rate, p=_p(wf, "time_stretch", 0.2):
                 AA.TimeStretch(min_rate=rate[0], max_rate=rate[1], p=p))

        # ── Domain (RIR / MUSAN / codec) ──
        rir = domain.get("rirs_reverb", {}) or {}
        if _p(domain, "rirs_reverb", 0.0) > 0:
            path = rir.get("path", "data/augmentation/rirs")
            if path and os.path.isdir(path):
                rir_cls = getattr(AA, "AddImpulseResponse",
                                  getattr(AA, "ApplyImpulseResponse", None))
                if rir_cls is not None:
                    _add(lambda rir_cls=rir_cls, path=path, p=_p(domain, "rirs_reverb", 0.0):
                         rir_cls(ir_path=path, p=p))
            else:
                print(f"  ⚠ RIR reverb skipped: '{path}' not found (download RIRs first)")

        musan = domain.get("musan", {}) or {}
        if _p(domain, "musan", 0.0) > 0 or float(musan.get("noise_p", 0) or 0) > 0 \
                or float(musan.get("music_p", 0) or 0) > 0:
            base = musan.get("path", "data/augmentation/musan")
            snr = musan.get("snr_db", [5, 20]) or [5, 20]
            noise_dir = os.path.join(base, "noise")
            music_dir = os.path.join(base, "music")
            noise_p = float(musan.get("noise_p", 0.4) or 0)
            music_p = float(musan.get("music_p", 0.2) or 0)
            if noise_p > 0 and os.path.isdir(noise_dir):
                _add(lambda noise_dir=noise_dir, snr=snr, noise_p=noise_p:
                     _make_background_noise(noise_dir, snr, noise_p))
            elif noise_p > 0:
                print(f"  ⚠ MUSAN noise skipped: '{noise_dir}' not found")
            if music_p > 0 and os.path.isdir(music_dir):
                _add(lambda music_dir=music_dir, snr=snr, music_p=music_p:
                     _make_background_noise(music_dir, snr, music_p))
            elif music_p > 0:
                print(f"  ⚠ MUSAN music skipped: '{music_dir}' not found")

        if _p(domain, "mp3_codec_roundtrip", 0.0) > 0:
            mp3 = domain.get("mp3_codec_roundtrip", {}) or {}
            if _mp3_backend_available():
                _add(lambda mp3=mp3, p=_p(domain, "mp3_codec_roundtrip", 0.0):
                     AA.Mp3Compression(min_bitrate=int(mp3.get("min_bitrate", 64)),
                                       max_bitrate=int(mp3.get("max_bitrate", 192)),
                                       backend="lameenc", preserve_delay=True,
                                       p=p))
            else:
                print("  ⚠ mp3 codec roundtrip disabled: lameenc not installed "
                      "(`uv sync` / `pip install lameenc`).")

        # ── Spec (time-domain masking approximation of SpecAugment) ──
        if _p(spec, "time_mask", 0.0) > 0:
            tm = spec.get("time_mask", {}) or {}
            _add(lambda tm=tm, p=_p(spec, "time_mask", 0.0):
                 AA.TimeMask(min_band_part=0.0,
                             max_band_part=float(tm.get("max_mask_ratio", 0.2)),
                             fade_duration=0.0, p=p))

        self.pipeline = AA.Compose(transforms) if transforms else None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline to a waveform.

        Args:
            waveform: (1, T) — single-channel audio tensor

        Returns:
            Augmented waveform of same shape (1, T)
        """
        if self.pipeline is None:
            return waveform
        try:
            # audiomentations expects (samples,) numpy array
            audio_np = waveform.squeeze(0).numpy()
            augmented = self.pipeline(samples=audio_np, sample_rate=self.sample_rate)
            return torch.from_numpy(augmented).unsqueeze(0).float()
        except Exception as e:
            # A single bad sample (e.g. ffmpeg missing at call time) must not
            # kill training — degrade to the un-augmented waveform.
            if not self._warned:
                print(f"  ⚠ Augmentation failed for a sample ({e}); "
                      f"returning un-augmented. Further failures are silent.")
                self._warned = True
            return waveform


# ─────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────

class SpeakerDataset(Dataset):
    """
    Dataset for Open-Set Speaker Identification.

    Loads the FULL audio file (MP3/WAV → 16 kHz mono) and returns a stack of
    fixed-length windows so the whole signal is usable:
      - train: `num_train_windows` random crops (each augmented independently)
      - eval/inference: sliding windows with `eval_hop_ratio` overlap over the
        full file, capped at `max_eval_windows` (evenly spread), with the last
        window repeated to a constant count so DataLoader batching stays simple.

    `__getitem__` normally returns (windows, label) with windows shape
    (W, 1, T).  With ``return_clean_aug_pair=True`` the first item is a dict
    containing ``augmented`` and ``clean`` tensors of that same shape.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        sample_rate: int = 16000,
        duration_seconds: float = 8.0,
        augment: bool = False,
        min_valid_duration: float = 1.0,
        mixup_alpha: float = 0.0,
        num_train_windows: int = 1,
        eval_hop_ratio: float = 0.5,
        max_eval_windows: int = 8,
        augmentation: Optional[dict] = None,
        speech_aware_crop_probability: float = 0.0,
        eval_speech_aware: bool = False,
        speech_relative_db: float = 35.0,
        short_audio_mode: str = "pad",
        return_clean_aug_pair: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sr = sample_rate
        self.target_length = int(sample_rate * duration_seconds)
        self.augment = augment
        self.min_valid_duration = min_valid_duration
        self.mixup_alpha = mixup_alpha
        self.num_train_windows = max(1, num_train_windows)
        self.eval_hop_ratio = eval_hop_ratio
        self.max_eval_windows = max(1, max_eval_windows)
        self.augmentation = augmentation
        self.speech_aware_crop_probability = float(speech_aware_crop_probability)
        self.eval_speech_aware = bool(eval_speech_aware)
        self.speech_relative_db = float(speech_relative_db)
        self.short_audio_mode = str(short_audio_mode)
        self.return_clean_aug_pair = bool(return_clean_aug_pair)

        if self.return_clean_aug_pair and not self.augment:
            raise ValueError(
                "return_clean_aug_pair requires augment=True (training only)"
            )
        if self.return_clean_aug_pair and self.mixup_alpha > 0:
            raise ValueError(
                "Paired clean/aug views are incompatible with mixup_alpha > 0: "
                "the clean teacher view must retain one speaker identity"
            )

        if self.augment:
            self.augmentor = AudioAugmentation(sample_rate, self.augmentation)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[object, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = self.audio_dir / row["audio_file"]
        label = torch.tensor(row["label"], dtype=torch.long)

        # Load FULL audio (no crop — windowing happens below)
        waveform = self._load_audio(audio_path)

        # MixUp: mix with another random sample (OOD regularization)
        if self.mixup_alpha > 0 and self.augment and torch.rand(1).item() < 0.5:
            other_idx = torch.randint(0, len(self.df), (1,)).item()
            other_row = self.df.iloc[other_idx]
            other_path = self.audio_dir / other_row["audio_file"]
            other_waveform = self._load_audio(other_path)

            n = max(waveform.size(-1), other_waveform.size(-1))
            if waveform.size(-1) < n:
                waveform = torch.nn.functional.pad(waveform, (0, n - waveform.size(-1)))
            if other_waveform.size(-1) < n:
                other_waveform = torch.nn.functional.pad(other_waveform, (0, n - other_waveform.size(-1)))

            # Mix: λ ~ Beta(α, α); keep original label (ambiguous mixed audio)
            lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
            waveform = lam * waveform + (1 - lam) * other_waveform

        # Windowing: train → random crops; eval → sliding windows
        if self.augment:
            windows = self._train_windows(waveform)
        else:
            windows = self._eval_windows(waveform)

        return windows, label

    def _train_windows(self, waveform: torch.Tensor):
        """Return augmented windows, optionally paired with identical clean crops.

        In paired mode the crop boundary is sampled exactly once.  The clean
        view is cloned before augmentation so every pair has identical speech
        content and differs only by the configured channel/noise transform.
        The default Tensor return remains byte-for-byte compatible with all
        existing experiments.
        """
        T = self.target_length
        augmented_windows = []
        clean_windows = []
        for _ in range(self.num_train_windows):
            w = waveform
            n = w.size(-1)
            if n > T:
                max_start = n - T
                if (self.speech_aware_crop_probability > 0 and
                        torch.rand(1).item() < self.speech_aware_crop_probability):
                    from src.audio_windows import choose_speech_crop_start
                    start = choose_speech_crop_start(
                        w, T, sample_rate=self.target_sr,
                        relative_db=self.speech_relative_db,
                    )
                else:
                    start = torch.randint(0, max_start + 1, (1,)).item()
                w = w[..., start : start + T]
            elif n < T:
                from src.audio_windows import fit_short_audio
                w = fit_short_audio(w, T, mode=self.short_audio_mode)
            clean = w.clone() if self.return_clean_aug_pair else None
            if self.augment:
                augmentation_source = clean.clone() if clean is not None else w
                w = self.augmentor(augmentation_source)
                # TimeStretch / PitchShift can change the window length
                # (newer audiomentations keep the stretched duration instead of
                # resampling back), so re-normalise before stacking — otherwise
                # torch.stack fails on unequal sizes (server R8/C2 crash).
                if w.size(-1) > T:
                    w = w[..., :T]
                elif w.size(-1) < T:
                    w = torch.nn.functional.pad(w, (0, T - w.size(-1)))
            augmented_windows.append(w)
            if self.return_clean_aug_pair:
                assert clean is not None
                clean_windows.append(clean)

        augmented = torch.stack(augmented_windows)
        if not self.return_clean_aug_pair:
            return augmented
        return {
            "augmented": augmented,
            "clean": torch.stack(clean_windows),
        }

    def _eval_windows(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Return (max_eval_windows, 1, T) sliding windows.

        Windows start every `eval_hop_ratio * T` samples over the full file.
        If there are more windows than `max_eval_windows`, they are evenly
        spread across the file; if fewer, the last window is repeated to keep
        a constant count (so DataLoader batching stays simple).
        """
        from src.audio_windows import make_eval_windows
        windows = make_eval_windows(
            waveform,
            target_length=self.target_length,
            hop_ratio=self.eval_hop_ratio,
            max_windows=self.max_eval_windows,
            sample_rate=self.target_sr,
            speech_aware=self.eval_speech_aware,
            speech_relative_db=self.speech_relative_db,
            short_audio_mode=self.short_audio_mode,
        )
        return torch.stack(windows)

    def _load_audio(self, path: Path) -> torch.Tensor:
        """
        Load and resample the FULL audio file (no crop).

        WAV files: uses soundfile backend (fast, no mpg123 dependency)
        Other formats: falls back to librosa
        """
        suffix = path.suffix.lower()
        try:
            if suffix in (".wav",):
                # Use soundfile for WAV — no mpg123, fast, reliable
                import soundfile as sf
                waveform, sr = sf.read(str(path), dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)  # stereo → mono
                if sr != self.target_sr:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
                waveform = torch.from_numpy(waveform).unsqueeze(0).float()  # (1, N)
            else:
                # librosa for MP3 and other formats
                waveform, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
                waveform = torch.from_numpy(waveform).unsqueeze(0).float()  # (1, N)
        except Exception as e:
            # Return silence for corrupted files
            print(f"  ⚠ Warning: Could not load {path.name}: {e}")
            return torch.zeros(1, self.target_length)

        return waveform


# ─────────────────────────────────────────────────────────
#  DataLoader Factory
# ─────────────────────────────────────────────────────────

class BalancedOODBatchSampler(Sampler[List[int]]):
    """Deterministic batch sampler with an exact OOD/known ratio per batch.

    A flat list passed to ``SubsetRandomSampler`` is not sufficient: that
    sampler shuffles the entire list again and destroys the batch boundaries.
    This sampler yields the batches themselves, so the configured ratio is a
    property of what the DataLoader consumes, not merely of an intermediate
    array.
    """

    def __init__(
        self,
        train_labels: np.ndarray,
        batch_size: int,
        ood_ratio: float = 0.5,
        seed: int = 42,
        competition_known_count: Optional[int] = None,
        known_sampling_weights: Optional[np.ndarray] = None,
        pair_known_files: bool = False,
        train_file_ids: Optional[np.ndarray] = None,
    ) -> None:
        labels = np.asarray(train_labels)
        if labels.ndim != 1:
            raise ValueError(f"train_labels must be one-dimensional, got {labels.shape}")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if not 0.0 < float(ood_ratio) < 1.0:
            raise ValueError("ood_ratio must be strictly between 0 and 1")

        if competition_known_count is None:
            is_ood = labels == 0
        else:
            is_ood = ((labels == 0) | (labels > int(competition_known_count)))
        self.ood_indices = np.flatnonzero(is_ood).astype(np.int64)
        self.known_indices = np.flatnonzero(~is_ood).astype(np.int64)
        if not len(self.ood_indices) or not len(self.known_indices):
            raise ValueError(
                "Balanced batch sampling requires non-empty OOD and known pools: "
                f"ood={len(self.ood_indices)}, known={len(self.known_indices)}"
            )

        self.known_probabilities: Optional[np.ndarray] = None
        self.pair_known_files = bool(pair_known_files)
        self.known_indices_by_label: Dict[int, np.ndarray] = {}
        self.known_speaker_labels = np.asarray([], dtype=np.int64)
        if self.pair_known_files and known_sampling_weights is not None:
            raise ValueError(
                "pair_known_files is incompatible with known_sampling_weights: "
                "pairing must select speakers uniformly before selecting files"
            )
        file_ids = None
        if train_file_ids is not None:
            file_ids = np.asarray(train_file_ids).astype(str)
            if file_ids.shape != labels.shape:
                raise ValueError(
                    "train_file_ids must match train_labels shape: "
                    f"{file_ids.shape} != {labels.shape}"
                )
        if self.pair_known_files and file_ids is None:
            raise ValueError(
                "pair_known_files requires train_file_ids so distinct rows "
                "cannot silently refer to the same audio file"
            )
        if known_sampling_weights is not None:
            weights = np.asarray(known_sampling_weights, dtype=np.float64)
            if weights.shape != labels.shape:
                raise ValueError(
                    "known_sampling_weights must match train_labels shape: "
                    f"{weights.shape} != {labels.shape}"
                )
            known_weights = weights[self.known_indices]
            if (not np.isfinite(known_weights).all()
                    or np.any(known_weights <= 0.0)):
                raise ValueError(
                    "known_sampling_weights must be finite and strictly positive "
                    "for every known sample"
                )
            self.known_probabilities = known_weights / known_weights.sum()

        self.batch_size = int(batch_size)
        self.n_ood = max(1, int(round(self.batch_size * float(ood_ratio))))
        self.n_known = self.batch_size - self.n_ood
        if self.n_known <= 0:
            self.n_known = 1
            self.n_ood = self.batch_size - 1
        if self.pair_known_files:
            if self.n_known % 2:
                raise ValueError(
                    "pair_known_files requires an even number of known samples "
                    f"per batch, got {self.n_known}"
                )
            known_labels = labels[self.known_indices]
            for label in sorted(np.unique(known_labels).tolist()):
                indices = self.known_indices[known_labels == label]
                if len(indices) < 2:
                    raise ValueError(
                        "pair_known_files requires at least two distinct files "
                        f"for every known speaker; label {int(label)} has "
                        f"{len(indices)}"
                    )
                assert file_ids is not None
                speaker_files = file_ids[indices]
                if len(np.unique(speaker_files)) != len(speaker_files):
                    raise ValueError(
                        "pair_known_files requires unique audio_file rows within "
                        f"each known speaker; label {int(label)} contains a "
                        "duplicate file id"
                    )
                self.known_indices_by_label[int(label)] = indices
            self.known_speaker_labels = np.asarray(
                sorted(self.known_indices_by_label), dtype=np.int64,
            )
            pairs_per_batch = self.n_known // 2
            if pairs_per_batch > len(self.known_speaker_labels):
                raise ValueError(
                    "pair_known_files requires at least one distinct speaker per "
                    f"known pair: pairs={pairs_per_batch}, speakers="
                    f"{len(self.known_speaker_labels)}"
                )
        self.num_batches = max(1, len(labels) // self.batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle for a training epoch."""
        self.epoch = int(epoch)

    @staticmethod
    def _draw(pool: np.ndarray, count: int, rng: np.random.RandomState) -> np.ndarray:
        """Draw ``count`` items by cycling shuffled full-pool permutations."""
        chunks: List[np.ndarray] = []
        remaining = int(count)
        while remaining > 0:
            shuffled = rng.permutation(pool)
            take = min(remaining, len(shuffled))
            chunks.append(shuffled[:take])
            remaining -= take
        return np.concatenate(chunks).astype(np.int64, copy=False)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        ood_stream = self._draw(
            self.ood_indices, self.num_batches * self.n_ood, rng)
        if self.pair_known_files:
            known_batches: List[np.ndarray] = []
            pairs_per_batch = self.n_known // 2
            # Consume complete shuffled speaker permutations before starting
            # another one.  This preserves full known-speaker exposure within
            # an epoch whenever the number of pair slots is large enough,
            # unlike independent per-batch choices that can omit many speakers.
            speaker_queue: List[int] = []
            for _ in range(self.num_batches):
                speakers: List[int] = []
                deferred: List[int] = []
                while len(speakers) < pairs_per_batch:
                    if not speaker_queue:
                        speaker_queue.extend(
                            int(label)
                            for label in rng.permutation(
                                self.known_speaker_labels
                            ).tolist()
                        )
                    candidate = speaker_queue.pop()
                    if candidate in speakers:
                        deferred.append(candidate)
                    else:
                        speakers.append(candidate)
                # A duplicate deferred at a permutation boundary becomes an
                # early candidate for the next batch rather than being lost.
                speaker_queue.extend(deferred)
                paired_indices = [
                    rng.choice(
                        self.known_indices_by_label[label],
                        size=2,
                        replace=False,
                    )
                    for label in speakers
                ]
                known_batches.append(
                    np.concatenate(paired_indices).astype(np.int64, copy=False)
                )
            known_stream = np.concatenate(known_batches)
        elif self.known_probabilities is None:
            known_stream = self._draw(
                self.known_indices, self.num_batches * self.n_known, rng)
        else:
            known_stream = rng.choice(
                self.known_indices,
                size=self.num_batches * self.n_known,
                replace=True,
                p=self.known_probabilities,
            ).astype(np.int64, copy=False)
        for batch_idx in range(self.num_batches):
            ood_start = batch_idx * self.n_ood
            known_start = batch_idx * self.n_known
            batch = np.concatenate([
                ood_stream[ood_start:ood_start + self.n_ood],
                known_stream[known_start:known_start + self.n_known],
            ])
            rng.shuffle(batch)
            yield batch.tolist()


def make_balanced_batch_sampler(
    train_labels: np.ndarray,
    batch_size: int,
    ood_ratio: float = 0.5,
    seed: int = 42,
    competition_known_count: Optional[int] = None,
    known_sampling_weights: Optional[np.ndarray] = None,
    pair_known_files: bool = False,
    train_file_ids: Optional[np.ndarray] = None,
) -> BalancedOODBatchSampler:
    """
    Build a real batch sampler so every emitted batch contains ~`ood_ratio`
    samples from the unknown class and the rest from known speakers. In known-first
    experiments pseudo labels above ``competition_known_count`` are OOD too.

    Without this, a per-class WeightedRandomSampler gives the unknown class
    (a single 2275-sample super-class) a total probability mass of ~1/447,
    i.e. ~0.2% of every batch — which starves the OOD head and makes it
    collapse to "always known" (the failure from the last run).

    Args:
        train_labels: (N,) global metric ids.
        batch_size:   desired batch size (the last partial batch is dropped).
        ood_ratio:    target fraction of unknown samples per batch (0.5 matches
                      the ~50/50 eval mix).
        seed:         RNG seed for reproducibility.
        pair_known_files: emit the known half as two distinct files for every
                          selected speaker.
        train_file_ids: exact file ids aligned with ``train_labels``; required
                        when ``pair_known_files`` is enabled.

    Returns:
        ``BalancedOODBatchSampler`` for use as DataLoader ``batch_sampler``.
    """
    return BalancedOODBatchSampler(
        train_labels=train_labels,
        batch_size=batch_size,
        ood_ratio=ood_ratio,
        seed=seed,
        competition_known_count=competition_known_count,
        known_sampling_weights=known_sampling_weights,
        pair_known_files=pair_known_files,
        train_file_ids=train_file_ids,
    )


def _sampling_rows_sha256(df: pd.DataFrame) -> str:
    """Stable digest for the exact audio-file/metric-label training rows."""
    rows = sorted(
        f"{audio_file}\t{int(label)}"
        for audio_file, label in zip(df["audio_file"], df["label"])
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def load_known_sampling_weights(
    config: dict,
    train_df: pd.DataFrame,
    competition_known_count: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load a preregistered, train-only known-file weighting artifact.

    The artifact is opt-in through ``data.known_sampling.weights_path``.  Its
    key set must equal the current known training pool exactly, and its digest
    must match the current split rows.  These checks make a Fold-0 artifact
    unusable on another Fold and prevent validation-file weights from entering
    the sampler silently.
    """
    sampling_cfg = (config.get("data", {}).get("known_sampling", {}) or {})
    weights_path = str(sampling_cfg.get("weights_path", "")).strip()
    if not weights_path:
        return None

    path = Path(weights_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Known-sampling artifact not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Known-sampling artifact schema_version must be 1")

    actual_digest = _sampling_rows_sha256(train_df)
    expected_digest = str(payload.get("training_rows_sha256", ""))
    if expected_digest != actual_digest:
        raise ValueError(
            "Known-sampling artifact does not match the current training split: "
            f"{expected_digest or '<missing>'} != {actual_digest}"
        )

    labels = train_df["label"].to_numpy(dtype=np.int64)
    is_known = labels > 0
    if competition_known_count is not None:
        is_known &= labels <= int(competition_known_count)
    known_files = train_df.loc[is_known, "audio_file"].astype(str).tolist()
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("Known-sampling artifact weights must be an object")
    if set(raw_weights) != set(known_files):
        missing = sorted(set(known_files) - set(raw_weights))[:5]
        extra = sorted(set(raw_weights) - set(known_files))[:5]
        raise ValueError(
            "Known-sampling artifact keys must exactly match known training files; "
            f"missing={missing}, extra={extra}"
        )

    weights = np.ones(len(train_df), dtype=np.float64)
    known_weights = np.asarray(
        [raw_weights[file_name] for file_name in known_files], dtype=np.float64,
    )
    if not np.isfinite(known_weights).all() or np.any(known_weights <= 0.0):
        raise ValueError("Known-sampling weights must be finite and strictly positive")
    weights[np.flatnonzero(is_known)] = known_weights
    print(
        "  🎯 Known-file sampling artifact: "
        f"{path} | weighted={(known_weights != 1.0).sum():,}/"
        f"{len(known_weights):,} | range={known_weights.min():.3g}.."
        f"{known_weights.max():.3g}"
    )
    return weights


def get_dataloaders(
    config: Optional[dict] = None,
    config_path: str = "configs/default_config.yaml",
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation DataLoaders with a balanced OOD/known batch
    sampler.

    Returns:
        train_loader, val_loader, class_map
    """
    if config is None:
        config = load_config(config_path)

    audio_cfg = config["audio"]
    data_cfg = config["data"]
    hw_profile = get_active_profile(config)
    consistency_cfg = (
        ((config.get("training", {}).get("loss", {}) or {})
         .get("consistency", {}) or {})
    )
    consistency_enabled = bool(consistency_cfg.get("enabled", False))
    consistency_weight = float(consistency_cfg.get("weight", 0.0))
    consistency_type = str(consistency_cfg.get("type", "cosine")).lower()
    consistency_pairing = str(
        consistency_cfg.get("pairing", "clean_aug")
    ).lower().strip()
    ood_jsd_cfg = (
        (((config.get("training", {}).get("loss", {}) or {})
          .get("ood", {}) or {}).get("clean_aug_jsd", {}) or {})
    )
    ood_jsd_enabled = bool(ood_jsd_cfg.get("enabled", False))
    ood_jsd_weight = float(ood_jsd_cfg.get("weight", 0.0))
    ood_jsd_type = str(
        ood_jsd_cfg.get("type", "target_clean_aug_bernoulli_jsd")
    ).lower().strip()
    if consistency_enabled and consistency_weight <= 0:
        raise ValueError(
            "training.loss.consistency.enabled requires a positive weight"
        )
    if consistency_enabled and consistency_type != "cosine":
        raise ValueError(
            "Only training.loss.consistency.type=cosine is supported"
        )
    if consistency_pairing not in {"clean_aug", "cross_file_batch"}:
        raise ValueError(
            "training.loss.consistency.pairing must be clean_aug or "
            "cross_file_batch"
        )
    if ood_jsd_type != "target_clean_aug_bernoulli_jsd":
        raise ValueError(
            "Only training.loss.ood.clean_aug_jsd.type="
            "target_clean_aug_bernoulli_jsd is supported"
        )
    if not np.isfinite(ood_jsd_weight) or ood_jsd_weight < 0.0:
        raise ValueError(
            "training.loss.ood.clean_aug_jsd.weight must be finite and "
            "non-negative"
        )
    known_sampling_cfg = data_cfg.get("known_sampling", {}) or {}
    pair_known_files = bool(known_sampling_cfg.get("pair_files", False))
    if (consistency_enabled and consistency_pairing == "cross_file_batch"
            and not pair_known_files):
        raise ValueError(
            "cross_file_batch consistency requires "
            "data.known_sampling.pair_files=true"
        )

    batch_size = hw_profile["batch_size"]
    num_workers = hw_profile["num_workers"]

    print("=" * 50)
    print("  Preparing DataLoaders")
    print("=" * 50)
    print(f"  Hardware profile: {config['hardware']['mode']}")
    print(f"  Batch size: {batch_size} | Workers: {num_workers}")
    print(f"  Sample rate: {audio_cfg['sample_rate']} | "
          f"Duration: {audio_cfg['duration_seconds']}s")

    min_valid_duration = audio_cfg.get("min_valid_duration", 0.0)

    # ── Verify audio directory exists ──
    audio_dir = data_cfg["audio_dir"]
    labels_path = data_cfg["labels_path"]

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}\n"
            f"Run: python scripts/convert_mp3_to_wav.py\n"
            f"Or update config to point to existing audio files."
        )
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Count WAV files for info
    wav_count = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    mp3_count = len([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
    print(f"  Audio dir: {audio_dir}")
    print(f"  Files: {wav_count} WAV + {mp3_count} MP3")

    # Prepare labels + leak-free split (corrupted/duplicate filtering inside).
    # `data.split` selects single (legacy) vs speaker_aware_kfold (OOF, C5).
    split_cfg = data_cfg.get("split", {}) or {}
    split_scheme = str(split_cfg.get("scheme", "single")).lower()

    # Closed-set 1000-class experiment: pseudo-identity map for unknown files.
    # load_unknown_cluster_map validates the requested k against the map's
    # distinct cluster ids (a k/rebuild mismatch is a hard error, not a silent
    # mislabelled run) and falls back to the committed submission copy when
    # data/processed is absent (fresh Vast.ai instance).
    unknown_cluster_map = load_unknown_cluster_map(config)
    if unknown_cluster_map is not None:
        print(f"  🧬 Unknown clusters: {len(unknown_cluster_map)} files → "
              f"{len(set(unknown_cluster_map.values()))} pseudo-identities "
              f"({config.get('model', {}).get('unknown_cluster_path', 'data/processed/unknown_clusters.json')})")

    train_df, val_df, class_map = prepare_labels(
        labels_path=labels_path,
        output_path=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        audio_dir=audio_dir,
        min_valid_duration=min_valid_duration,
        split_scheme=split_scheme,
        fold=int(split_cfg.get("fold", 0)),
        folds=int(split_cfg.get("folds", 3)),
        unknown_cluster_map=unknown_cluster_map,
        clean_duplicates=bool(data_cfg.get("clean_duplicates", False)),
    )

    # Create datasets
    train_dataset = SpeakerDataset(
        df=train_df,
        audio_dir=audio_dir,           # ← resolved path
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=True,
        min_valid_duration=min_valid_duration,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
        augmentation=config.get("augmentation"),
        speech_aware_crop_probability=audio_cfg.get("speech_aware_crop_probability", 0.0),
        eval_speech_aware=audio_cfg.get("eval_speech_aware", False),
        speech_relative_db=audio_cfg.get("speech_relative_db", 35.0),
        short_audio_mode=audio_cfg.get("short_audio_mode", "pad"),
        return_clean_aug_pair=(
            consistency_enabled and consistency_pairing == "clean_aug"
        ) or ood_jsd_enabled,
    )

    val_dataset = SpeakerDataset(
        df=val_df,
        audio_dir=audio_dir,           # ← resolved path
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=False,
        min_valid_duration=min_valid_duration,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
        eval_speech_aware=audio_cfg.get("eval_speech_aware", False),
        speech_relative_db=audio_cfg.get("speech_relative_db", 35.0),
        short_audio_mode=audio_cfg.get("short_audio_mode", "pad"),
    )

    # ── Balanced Batch Sampler ──
    # Enforces a target OOD/known ratio in every batch so the OOD head always
    # sees enough positive (unknown) samples (a per-class WeightedRandomSampler
    # would give the unknown super-class ~1/447 of every batch → OOD collapse).
    #
    # Closed-set 1000-class mode: every file has a (pseudo-)class, so the
    # label-0 pool is (nearly) empty and the balanced sampler would over-sample
    # the same few files — fall back to a plain shuffle instead.
    train_labels = train_df["label"].values
    num_unknown_clusters = (
        len(set(unknown_cluster_map.values())) if unknown_cluster_map else 0
    )
    speaker_target_scope = str(
        config.get("model", {}).get("speaker_target_scope", "metric")
    ).lower().strip()
    if num_unknown_clusters > 0 and speaker_target_scope != "known":
        balanced_batch_sampler = None
        print("\n  🧬 1000-class mode: uniform batch sampling "
              "(no OOD/known balance)")
    else:
        ood_ratio = audio_cfg.get("ood_batch_ratio", 0.50)
        competition_known_count = (
            int(config.get("model", {}).get("competition_num_known", 446))
            if speaker_target_scope == "known" else None
        )
        known_sampling_weights = load_known_sampling_weights(
            config, train_df, competition_known_count=competition_known_count,
        )
        balanced_batch_sampler = make_balanced_batch_sampler(
            train_labels, batch_size, ood_ratio=ood_ratio, seed=42,
            competition_known_count=competition_known_count,
            known_sampling_weights=known_sampling_weights,
            pair_known_files=pair_known_files,
            train_file_ids=train_df["audio_file"].astype(str).to_numpy(),
        )
        is_ood = ((train_labels == 0) | (train_labels > competition_known_count)
                  if competition_known_count is not None else train_labels == 0)
        n_ood = max(1, int(round(batch_size * ood_ratio)))
        print(f"\n  ⚖️  Batch balance: {n_ood} OOD + {batch_size - n_ood} known "
              f"({ood_ratio:.0%} / {1 - ood_ratio:.0%})")
        print(f"     OOD pool: {is_ood.sum():,} samples | "
              f"Known pool: {(~is_ood).sum():,} samples "
              f"across {len(class_map) - 1} speakers")
        if pair_known_files:
            print("     Known sampler: two distinct files per selected speaker")
    train_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": hw_profile["device"] == "cuda",
    }
    if balanced_batch_sampler is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **train_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_batch_sampler,
            **train_loader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if hw_profile["device"] == "cuda" else False,
        drop_last=False,
    )

    # Print class distribution info
    known_train = (train_labels != 0).sum()
    unknown_train = (train_labels == 0).sum()
    print(f"\n  📊 Class distribution in training set:")
    print(f"     Known classes: {len(class_map) - 1}")
    print(f"     Known samples: {known_train}")
    print(f"     Unknown samples: {unknown_train}")
    print(f"     Sampler: balanced batch sampler (OOD/known ratio per batch)")

    return train_loader, val_loader, class_map


# ─────────────────────────────────────────────────────────
#  Main Test Block
# ─────────────────────────────────────────────────────────

def main():
    """Test the DataLoader pipeline: load one batch and verify shapes/distribution."""
    print("=" * 55)
    print("  Data Pipeline Test")
    print("=" * 55)
    print()

    train_loader, val_loader, class_map = get_dataloaders()

    print("\n" + "-" * 50)
    print("  Testing Train Loader (1 batch)")
    print("-" * 50)
    waveforms, labels = next(iter(train_loader))
    print(f"  Waveform shape: {waveforms.shape}  (batch, channels, time)")
    print(f"  Labels shape:   {labels.shape}")
    print(f"  Label values:   {labels.tolist()}")
    n_known = (labels != 0).sum().item()
    n_unknown = (labels == 0).sum().item()
    print(f"  Batch composition: {n_known} known + {n_unknown} unknown")
    print(f"  Waveform range: [{waveforms.min().item():.4f}, {waveforms.max().item():.4f}]")

    print("\n" + "-" * 50)
    print("  Testing Val Loader (1 batch)")
    print("-" * 50)
    waveforms_val, labels_val = next(iter(val_loader))
    print(f"  Waveform shape: {waveforms_val.shape}")
    print(f"  Labels shape:   {labels_val.shape}")

    # Verify class mapping sanity
    print(f"\n  ✅ Class mapping: {len(class_map)} total classes "
          f"(0=unknown, 1..{len(class_map)-1}=known speakers)")
    print("  ✅ Pipeline test passed!")


if __name__ == "__main__":
    main()
