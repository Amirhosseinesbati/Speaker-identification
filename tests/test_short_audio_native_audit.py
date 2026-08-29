import numpy as np
import torch
from pathlib import Path

from scripts.audit_short_audio_native import infer_native_short_rows


class _Dataset:
    target_length = 10
    audio_dir = Path("")

    def _load_audio(self, path):
        length = int(str(path))
        return torch.ones(1, length)


class _Model:
    def embed(self, batch):
        length = float(batch.shape[-1])
        value = torch.tensor([[length, 1.0]], device=batch.device)
        return torch.nn.functional.normalize(value, dim=1)

    def predict_proba_and_embed(self, batch, temperature=1.0):
        embedding = self.embed(batch)[0]
        probability = torch.tensor([0.25, 0.75], device=batch.device)
        return probability, embedding


def test_native_inference_replaces_only_short_rows():
    files = np.array(["3", "12", "5"])
    short = np.array([True, False, True])
    base_embeddings = np.array([[9.0, 9.0], [4.0, 5.0], [8.0, 8.0]], np.float32)
    base_probabilities = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.8, 0.2]], np.float32
    )

    embeddings, probabilities, diagnostics = infer_native_short_rows(
        model=_Model(),
        dataset=_Dataset(),
        files=files,
        short_mask=short,
        device=torch.device("cpu"),
        base_embeddings=base_embeddings,
        base_probabilities=base_probabilities,
        description="test",
    )

    assert probabilities is not None
    np.testing.assert_array_equal(probabilities[1], base_probabilities[1])
    np.testing.assert_array_equal(embeddings[1], base_embeddings[1])
    np.testing.assert_allclose(probabilities[[0, 2]], [[0.25, 0.75], [0.25, 0.75]])
    assert diagnostics == {
        "short_rows": 2,
        "minimum_samples": 3,
        "median_samples": 4.0,
        "maximum_samples": 5,
    }
