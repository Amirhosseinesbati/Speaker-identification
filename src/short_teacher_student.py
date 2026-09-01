"""Long/short teacher-student distillation for duration-robust speaker ID.

The frozen teacher consumes the ordinary long training view while the student
consumes a deterministic short crop of the same recording.  The auxiliary
objective follows Sang, Xia & Hansen (Interspeech 2020): posterior KL plus
cosine embedding alignment, jointly optimized with the ordinary supervised
classification objective.  Competition probabilities include the binary OOD
head, so posterior distillation explicitly preserves the known/unknown border.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Optional

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class ShortTeacherStudent:
    student_duration_seconds: float
    posterior_weight: float
    embedding_weight: float
    sample_rate: int = 16_000
    seed_offset: int = 91_337

    def crop_student_view(
        self,
        waveforms: torch.Tensor,
        source_durations_seconds: torch.Tensor,
        *,
        training_seed: int,
        epoch: int,
        step: int,
        window_index: int,
    ) -> torch.Tensor:
        """Crop each row independently without ever sampling right padding."""
        if waveforms.ndim != 3:
            raise ValueError(
                f"Expected waveforms shaped (B,1,T), got {tuple(waveforms.shape)}"
            )
        if source_durations_seconds.ndim != 1 or int(
            source_durations_seconds.numel()
        ) != int(waveforms.shape[0]):
            raise ValueError(
                "source_durations_seconds must be shaped (B,) and match the batch"
            )
        target = int(round(self.student_duration_seconds * self.sample_rate))
        target = max(1, min(target, int(waveforms.shape[-1])))
        valid_samples = torch.round(
            source_durations_seconds.detach().float().cpu() * self.sample_rate
        ).to(dtype=torch.long)
        valid_samples.clamp_(min=1, max=int(waveforms.shape[-1]))

        payload = (
            f"short-ts:{int(training_seed)}:{int(epoch)}:{int(step)}:"
            f"{int(window_index)}:{self.seed_offset}"
        ).encode("ascii")
        mixed_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        units = torch.rand((int(waveforms.shape[0]),), generator=generator)

        crops = []
        for row, (valid, unit) in enumerate(
            zip(valid_samples.tolist(), units.tolist())
        ):
            available = min(int(valid), int(waveforms.shape[-1]))
            if available >= target:
                max_start = available - target
                start = min(max_start, int(float(unit) * (max_start + 1)))
                crop = waveforms[row, :, start : start + target]
            else:
                crop = waveforms[row, :, :available]
                crop = F.pad(crop, (0, target - available))
            crops.append(crop)
        return torch.stack(crops, dim=0)


def build_short_teacher_student(
    experiment_config: Mapping[str, object],
) -> Optional[ShortTeacherStudent]:
    training = dict(experiment_config.get("training", {}) or {})
    loss = dict(training.get("loss", {}) or {})
    cfg = dict(loss.get("short_teacher_student", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return None

    if not training.get("warm_start_checkpoint"):
        raise ValueError(
            "short_teacher_student requires a warm_start_checkpoint teacher"
        )
    if bool((training.get("adaptive_margin", {}) or {}).get("enabled", False)):
        raise ValueError(
            "short_teacher_student is a distinct hypothesis and cannot be "
            "combined with adaptive_margin"
        )
    if bool((loss.get("proto", {}) or {}).get("enabled", False)):
        raise ValueError("short_teacher_student cannot be combined with proto loss")
    if bool((loss.get("consistency", {}) or {}).get("enabled", False)):
        raise ValueError(
            "short_teacher_student cannot be combined with generic consistency"
        )
    model = dict(experiment_config.get("model", {}) or {})
    if str(model.get("speaker_head_type", "arcface")).lower().strip() != "arcface":
        raise ValueError("short_teacher_student currently requires ArcFace")
    if str(model.get("speaker_target_scope", "metric")).lower().strip() != "known":
        raise ValueError("short_teacher_student requires known-first supervision")
    if not bool(model.get("ood_head", False)):
        raise ValueError("short_teacher_student requires the binary OOD head")

    audio = dict(experiment_config.get("audio", {}) or {})
    student_duration = float(cfg.get("student_duration_seconds", 2.0))
    long_duration = float(audio.get("duration_seconds", 8.0))
    posterior_weight = float(cfg.get("posterior_weight", 1.0))
    embedding_weight = float(cfg.get("embedding_weight", 1.0))
    if (
        not math.isfinite(student_duration)
        or not 0.0 < student_duration < long_duration
    ):
        raise ValueError(
            "student_duration_seconds must be positive and shorter than the "
            "ordinary training window"
        )
    if (
        not math.isfinite(posterior_weight)
        or not math.isfinite(embedding_weight)
        or posterior_weight <= 0
        or embedding_weight <= 0
    ):
        raise ValueError("teacher-student loss weights must be positive")
    if int(audio.get("num_train_windows", 1)) != 1:
        raise ValueError(
            "short_teacher_student preregistration requires num_train_windows=1"
        )
    return ShortTeacherStudent(
        student_duration_seconds=student_duration,
        posterior_weight=posterior_weight,
        embedding_weight=embedding_weight,
        sample_rate=int(audio.get("sample_rate", 16_000)),
        seed_offset=int(cfg.get("seed_offset", 91_337)),
    )


def arcface_logits_from_normalized_embedding(model, embedding: torch.Tensor) -> torch.Tensor:
    """Reconstruct inference logits without a second student encoder forward."""
    head = getattr(model, "head_speaker", None)
    weight = getattr(head, "weight", None)
    scale = getattr(head, "s", None)
    if not isinstance(weight, torch.Tensor) or scale is None:
        raise ValueError(
            "short_teacher_student requires an ArcFace head with weight and scale"
        )
    return F.linear(
        F.normalize(embedding, p=2, dim=1),
        F.normalize(weight, p=2, dim=1),
    ).clamp(-1.0 + 1e-7, 1.0 - 1e-7) * float(scale)


def differentiable_fused_probs(
    ood_logits: torch.Tensor,
    speaker_logits: torch.Tensor,
    *,
    num_unknown_clusters: int = 0,
) -> torch.Tensor:
    """Fuse heads without the evaluation helper's intentional ``no_grad``."""
    p_unknown = torch.sigmoid(ood_logits.float())
    p_known = torch.softmax(speaker_logits.float(), dim=1)
    scaled_known = (1.0 - p_unknown) * p_known
    if num_unknown_clusters > 0:
        num_output_known = int(speaker_logits.shape[1]) - num_unknown_clusters
        if num_output_known <= 0:
            raise ValueError("num_unknown_clusters consumes the whole speaker head")
        cluster_mass = scaled_known[:, num_output_known:].sum(
            dim=1, keepdim=True
        )
        probabilities = torch.cat(
            [p_unknown + cluster_mass, scaled_known[:, :num_output_known]],
            dim=1,
        )
    else:
        probabilities = torch.cat([p_unknown, scaled_known], dim=1)
    probabilities = probabilities.clamp_min(1e-9)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def teacher_student_losses(
    *,
    student_model,
    student_ood_logits: torch.Tensor,
    student_embedding: torch.Tensor,
    teacher_ood_logits: torch.Tensor,
    teacher_speaker_logits: torch.Tensor,
    teacher_embedding: torch.Tensor,
    contract: ShortTeacherStudent,
) -> dict[str, torch.Tensor]:
    """Return differentiable posterior and embedding alignment losses."""
    student_speaker_logits = arcface_logits_from_normalized_embedding(
        student_model, student_embedding
    )
    unknown_clusters = int(getattr(student_model, "num_unknown_clusters", 0))
    student_probs = differentiable_fused_probs(
        student_ood_logits,
        student_speaker_logits,
        num_unknown_clusters=unknown_clusters,
    ).float()
    teacher_probs = differentiable_fused_probs(
        teacher_ood_logits,
        teacher_speaker_logits,
        num_unknown_clusters=unknown_clusters,
    ).float().detach()
    eps = torch.finfo(student_probs.dtype).eps
    posterior_kl = torch.sum(
        teacher_probs
        * (
            torch.log(teacher_probs.clamp_min(eps))
            - torch.log(student_probs.clamp_min(eps))
        ),
        dim=1,
    ).mean()
    embedding_cosine = F.cosine_similarity(
        student_embedding.float(), teacher_embedding.float().detach(), dim=1
    ).mean()
    embedding_loss = 1.0 - embedding_cosine
    weighted_posterior = contract.posterior_weight * posterior_kl
    weighted_embedding = contract.embedding_weight * embedding_loss
    return {
        "posterior_kl": posterior_kl,
        "embedding_loss": embedding_loss,
        "embedding_cosine": embedding_cosine,
        "posterior_weighted": weighted_posterior,
        "embedding_weighted": weighted_embedding,
        "total": weighted_posterior + weighted_embedding,
    }
