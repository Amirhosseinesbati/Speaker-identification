"""Shared, deterministic audio-window selection for training and inference.

The project previously duplicated sliding-window logic in ``SpeakerDataset``
and ``submission/inference.py``.  This module is intentionally dependency-light
(PyTorch + NumPy only) so the exact same speech-aware policy can be shipped in
the leaderboard package.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _frame_rms(waveform: torch.Tensor, frame: int, hop: int) -> torch.Tensor:
    """Return RMS per frame for a mono ``(1, N)`` waveform."""
    x = waveform.squeeze(0).float()
    if x.numel() < frame:
        x = F.pad(x, (0, frame - x.numel()))
    frames = x.unfold(0, frame, hop)
    return frames.square().mean(dim=1).add_(1e-12).sqrt_()


def speech_activity_mask(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    relative_db: float = 35.0,
    absolute_rms: float = 1e-4,
) -> tuple[torch.Tensor, int]:
    """Estimate speech-bearing frames using a robust adaptive energy gate.

    This is not intended to be a phonetic VAD.  It only prevents training crops
    and evaluation windows from being dominated by silence/non-speech.  The
    threshold is relative to the file's high-energy frames and is bounded by an
    absolute floor, making it deterministic and safe for the offline package.
    """
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    rms = _frame_rms(waveform, frame, hop)
    peak_ref = torch.quantile(rms, 0.9) if rms.numel() > 1 else rms.max()
    relative = peak_ref * (10.0 ** (-float(relative_db) / 20.0))
    threshold = max(float(absolute_rms), float(relative))
    return rms >= threshold, hop


def choose_speech_crop_start(
    waveform: torch.Tensor,
    target_length: int,
    sample_rate: int = 16000,
    relative_db: float = 35.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Choose a random crop centred near an active frame; uniform if none."""
    n = int(waveform.size(-1))
    if n <= target_length:
        return 0
    active, hop = speech_activity_mask(
        waveform, sample_rate=sample_rate, relative_db=relative_db,
    )
    idx = torch.nonzero(active, as_tuple=False).flatten()
    if idx.numel() == 0:
        return int(torch.randint(0, n - target_length + 1, (1,), generator=generator).item())
    pick = int(torch.randint(0, idx.numel(), (1,), generator=generator).item())
    center = int(idx[pick].item()) * hop
    return max(0, min(n - target_length, center - target_length // 2))


def fit_short_audio(
    waveform: torch.Tensor,
    target_length: int,
    mode: str = "pad",
) -> torch.Tensor:
    """Fit short audio to a fixed window using zero-pad or explicit tiling."""
    n = int(waveform.size(-1))
    if n >= target_length:
        return waveform[..., :target_length]
    if n <= 0:
        return torch.zeros(*waveform.shape[:-1], target_length, dtype=waveform.dtype)
    if str(mode).lower().strip() in {"tile", "tile_speech", "repeat"}:
        repeats = int(np.ceil(target_length / n))
        return waveform.repeat(*(1 for _ in waveform.shape[:-1]), repeats)[..., :target_length]
    return F.pad(waveform, (0, target_length - n))


def make_eval_windows(
    waveform: torch.Tensor,
    target_length: int,
    hop_ratio: float = 0.5,
    max_windows: int = 8,
    sample_rate: int = 16000,
    speech_aware: bool = False,
    speech_relative_db: float = 35.0,
    short_audio_mode: str = "pad",
) -> List[torch.Tensor]:
    """Create deterministic full-file windows, optionally ranked by speech.

    When the candidate count exceeds ``max_windows``, speech-aware mode keeps
    the windows with the highest active-frame coverage while retaining temporal
    order.  Otherwise the legacy evenly-spread policy is preserved exactly.
    """
    target_length = int(target_length)
    max_windows = max(1, int(max_windows))
    n = int(waveform.size(-1))
    if n <= target_length:
        w = fit_short_audio(waveform, target_length, mode=short_audio_mode)
        return [w] * max_windows

    hop = max(1, int(target_length * float(hop_ratio)))
    starts = list(range(0, n - target_length + 1, hop))
    if starts[-1] != n - target_length:
        starts.append(n - target_length)

    if len(starts) > max_windows:
        if speech_aware:
            active, vad_hop = speech_activity_mask(
                waveform, sample_rate=sample_rate, relative_db=speech_relative_db,
            )
            scores = []
            for start in starts:
                lo = max(0, start // vad_hop)
                hi = min(active.numel(), int(np.ceil((start + target_length) / vad_hop)))
                score = float(active[lo:hi].float().mean()) if hi > lo else 0.0
                scores.append(score)
            keep = sorted(np.argsort(scores)[-max_windows:].tolist())
            starts = [starts[i] for i in keep]
        else:
            starts = np.unique(
                np.linspace(0, n - target_length, max_windows).astype(int)
            ).tolist()

    windows = [waveform[..., s : s + target_length] for s in starts]
    while len(windows) < max_windows:
        windows.append(windows[-1])
    return windows
