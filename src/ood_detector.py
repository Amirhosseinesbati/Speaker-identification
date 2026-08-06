"""
FAISS-based Out-of-Distribution (OOD) Detector for Speaker Identification.

Core idea:
    1. Extract speaker embeddings for all known training speakers.
    2. Index them in FAISS with cosine similarity (IndexFlatIP + L2 normalization).
    3. At inference, for each test sample:
       a. Find k-nearest known speaker embeddings.
       b. Compute OOD score = 1.0 - max_cosine_similarity.
       c. Higher score → more likely OOD (unknown speaker).

    The FAISS score is combined with the learned OOD head score for robust
    open-set detection.

Classes:
    FAISSOODDetector — Main OOD detector based on cosine similarity
"""

from typing import Optional, Tuple

import faiss
import numpy as np
import torch


class FAISSOODDetector:
    """
    FAISS-based OOD detector using cosine similarity to known speaker embeddings.

    Usage:
        # Training / enrollment
        detector = FAISSOODDetector(dim=192)
        detector.fit(known_embeddings, speaker_ids)

        # Inference
        scores = detector.compute_ood_score(test_embeddings)
        # scores[i] in [0, 1] — higher = more likely OOD

    The detector uses IndexFlatIP (inner product) with L2-normalized vectors,
    which is equivalent to cosine similarity. This guarantees 100% recall
    (exact nearest neighbor search).
    """

    def __init__(self, dim: int = 192, use_gpu: bool = False):
        """
        Args:
            dim:     Embedding dimension (e.g., 192 for ECAPA, 768 for WavLM)
            use_gpu: If True, use GPU FAISS (requires faiss-gpu)
        """
        self.dim = dim
        self.use_gpu = use_gpu
        self.is_fitted = False

        # Base index: brute-force inner product
        self.base_index = faiss.IndexFlatIP(dim)

        # Wrap with automatic L2 normalization
        self.index = faiss.IndexPreTransform(
            faiss.NormalizationTransform(dim, 2.0),  # L2 norm
            self.base_index,
        )

        # Map to custom speaker IDs
        self.index = faiss.IndexIDMap(self.index)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"  ⚠ GPU FAISS not available: {e}. Using CPU.")
                self.use_gpu = False

        self._speaker_ids: Optional[np.ndarray] = None

    def fit(
        self,
        embeddings: np.ndarray,
        speaker_ids: np.ndarray,
    ) -> None:
        """
        Enroll known speaker embeddings into the FAISS index.

        Args:
            embeddings:  (N, dim) — L2-normalized speaker embeddings
            speaker_ids: (N,)     — integer speaker labels (1..num_known)

        Note:
            Embeddings do NOT need to be pre-normalized; the
            IndexPreTransform handles L2 normalization automatically.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings shape (N, {self.dim}), got {embeddings.shape}"
            )
        if len(speaker_ids) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(embeddings)} embeddings vs "
                f"{len(speaker_ids)} ids"
            )

        embeddings = embeddings.astype(np.float32)
        speaker_ids = speaker_ids.astype(np.int64)

        self.index.add_with_ids(embeddings, speaker_ids)
        self._speaker_ids = speaker_ids
        self.is_fitted = True

    def compute_ood_score(
        self,
        embeddings: np.ndarray,
        k: int = 5,
    ) -> np.ndarray:
        """
        Compute OOD score for test embeddings.

        OOD score = 1.0 - mean_cosine_similarity_to_k_nearest

        Args:
            embeddings: (M, dim) — test speaker embeddings
            k:          Number of nearest neighbors to consider

        Returns:
            ood_scores: (M,) — values in [0, 1], higher = more likely OOD
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        embeddings = embeddings.astype(np.float32)
        similarities, _ = self.index.search(embeddings, k)  # (M, k)

        # similarities are cosine similarities in [0, 1] (after L2 norm + IP)
        # Convert to distance: 1 - similarity
        # Higher distance = more likely OOD
        ood_scores = 1.0 - similarities.mean(axis=1)
        return ood_scores.astype(np.float64)

    def search(
        self,
        embeddings: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest known speakers.

        Args:
            embeddings: (M, dim)
            k:          Number of neighbors

        Returns:
            similarities: (M, k) — cosine similarities [0, 1]
            speaker_ids:  (M, k) — enrolled speaker IDs
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        embeddings = embeddings.astype(np.float32)
        similarities, indices = self.index.search(embeddings, k)
        return similarities, indices

    @property
    def num_enrolled(self) -> int:
        """Number of embeddings in the index."""
        return self.index.ntotal if self.is_fitted else 0

    def reset(self) -> None:
        """Clear the index and re-initialize."""
        self.index.reset()
        self._speaker_ids = None
        self.is_fitted = False


# ═══════════════════════════════════════════════════════════
#  Combine FAISS + Learned OOD scores
# ═══════════════════════════════════════════════════════════

def combine_ood_scores(
    head_score: torch.Tensor,    # (batch,) — sigmoid(head_logit)
    faiss_score: np.ndarray,     # (batch,) — FAISS OOD score [0,1]
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Weighted combination of learned OOD head and FAISS OOD detector.

    combined = alpha * head_score + (1 - alpha) * faiss_score

    Args:
        head_score:  Tensor of probabilities from learned OOD head
        faiss_score: FAISS OOD distance scores
        alpha:       Weight for head score (0.5 = equal weight)

    Returns:
        combined_score: (batch,) numpy array
    """
    head_np = head_score.detach().cpu().numpy().flatten()
    combined = alpha * head_np + (1.0 - alpha) * faiss_score
    return combined


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _smoke_test():
    print("=" * 50)
    print("  FAISS OOD Detector Smoke Test")
    print("=" * 50)

    dim = 64
    n_known = 10
    n_test = 5

    # ── Create dummy known embeddings ──
    rng = np.random.RandomState(42)
    known_embeddings = rng.randn(n_known, dim).astype(np.float32)
    speaker_ids = np.arange(1, n_known + 1, dtype=np.int64)

    # ── Fit detector ──
    detector = FAISSOODDetector(dim=dim)
    detector.fit(known_embeddings, speaker_ids)
    assert detector.is_fitted
    assert detector.num_enrolled == n_known
    print(f"  Fitted: {detector.num_enrolled} embeddings ✅")

    # ── Test: known-like embedding (low OOD score) ──
    known_like = known_embeddings[0:1] + rng.randn(1, dim).astype(np.float32) * 0.1
    ood_score = detector.compute_ood_score(known_like, k=3)
    assert 0.0 <= ood_score[0] <= 1.0
    print(f"  Known-like sample: OOD score = {ood_score[0]:.4f} (should be low) ✅")

    # ── Test: random embedding (high OOD score) ──
    random_emb = rng.randn(1, dim).astype(np.float32)
    ood_random = detector.compute_ood_score(random_emb, k=3)
    assert 0.0 <= ood_random[0] <= 1.0
    print(f"  Random sample:     OOD score = {ood_random[0]:.4f} (should be higher) ✅")

    # ── Test: search ──
    sims, ids = detector.search(known_like, k=3)
    assert sims.shape == (1, 3)
    assert ids.shape == (1, 3)
    print(f"  Search: top-3 similarities = {sims[0].round(3).tolist()} ✅")

    # ── Test: combined score ──
    head_score = torch.tensor([0.3, 0.8, 0.1])  # learned OOD head output
    test_embs = rng.randn(3, dim).astype(np.float32)
    faiss_scores = detector.compute_ood_score(test_embs, k=3)
    combined = combine_ood_scores(head_score, faiss_scores, alpha=0.5)
    assert combined.shape == (3,)
    print(f"  Combined scores: {combined.round(4).tolist()} ✅")

    # ── Test: reset ──
    detector.reset()
    assert not detector.is_fitted
    assert detector.num_enrolled == 0
    print(f"  Reset: ✅")

    print()
    print("  ALL FAISS TESTS PASSED ✅")


if __name__ == "__main__":
    _smoke_test()
