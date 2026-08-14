"""
Ensemble Model for Multi-Encoder Speaker Identification.

Combines multiple TwoHeadedSpeakerModel instances via:
    - Average fusion:     arithmetic mean of all probability vectors
    - Weighted average:   learned/optimised per-model weights
    - Geometric mean:     exp(mean(log(p))) — dampens weak models
    - Rank average:       average per-class ranks across models
    - Max pooling:        element-wise max over model probabilities
    - Learned MLP fusion: MLP on concatenated probability vectors

The ensemble produces a single (batch, 447) probability vector from
multiple base models with potentially different encoder architectures.

The stateless fusion functions (fusion_*) operate on numpy arrays and are
designed for offline analysis (ensemble_calibrate.py) and submission inference.

Usage:
    # Average fusion
    ensemble = EnsembleModel([model_wavlm, model_ecapa, model_campp])

    # Weighted fusion (offline)
    probs = weighted_average_fusion(probs_list, weights=[0.6, 0.25, 0.15])

    # Grid-search optimal weights
    result = grid_search_weights(probs_list, labels, num_classes=447)

    # Learned fusion
    fusion = LearnedFusion(num_models=3, num_classes=447)
    ensemble = EnsembleModel(models, fusion=fusion)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax

from src.model import TwoHeadedSpeakerModel


class LearnedFusion(nn.Module):
    """
    MLP-based learned fusion of ensemble probability vectors.

    Input: concatenated probability vectors from N models: (batch, N * num_classes)
    Output: fused probability vector: (batch, num_classes)

    Architecture:
        Linear(N*C, 512) → ReLU → Dropout(0.3) → Linear(512, C) → Softmax
    """

    def __init__(
        self,
        num_models: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            probs_list: List of (batch, num_classes) probability tensors

        Returns:
            fused_probs: (batch, num_classes) — sum(dim=1) ≈ 1.0
        """
        concat = torch.cat(probs_list, dim=1)  # (batch, N*C)
        logits = self.mlp(concat)
        return F.softmax(logits, dim=1)


# ── Valid fusion method names ──
FUSION_METHODS = [
    "average",
    "weighted_average",
    "geometric_mean",
    "rank_average",
    "max_prob",
    "learned_mlp",
]


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple TwoHeadedSpeakerModel instances.

    Supports six fusion strategies:
        - "average":           Arithmetic mean (equal weights)
        - "weighted_average":  Per-model weights (supplied via `fusion_weights`)
        - "geometric_mean":    exp(mean(log(p + ε))) — dampens weak models
        - "rank_average":      Average per-class ranks → softmax
        - "max_prob":          Element-wise max over model probabilities
        - "learned_mlp":       MLP-based fusion (requires training)

    Usage:
        ensemble = EnsembleModel(models, fusion_method="weighted_average",
                                 fusion_weights=[0.6, 0.25, 0.15])
        probs = ensemble.predict_proba(waveforms)

    For offline (numpy) fusion, use the stateless functions below.
    """

    def __init__(
        self,
        models: List[TwoHeadedSpeakerModel],
        fusion_method: str = "average",
        fusion: Optional[LearnedFusion] = None,
        fusion_weights: Optional[Sequence[float]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            models:         List of trained TwoHeadedSpeakerModel instances
            fusion_method:  one of FUSION_METHODS (default "average")
            fusion:         Pre-built LearnedFusion module (overrides fusion_method)
            fusion_weights: Per-model weights for "weighted_average" (sum should ≈ 1)
            device:         Device to run models on
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = self.models[0].num_known_speakers + 1  # + unknown

        if fusion is not None:
            self.fusion = fusion
            self.fusion_method = "learned_mlp"
        elif fusion_method == "learned_mlp":
            self.fusion = LearnedFusion(self.num_models, self.num_classes)
            self.fusion_method = "learned_mlp"
        else:
            self.fusion = None
            self.fusion_method = fusion_method

        if self.fusion_method not in FUSION_METHODS:
            raise ValueError(
                f"Unknown fusion_method '{self.fusion_method}'. "
                f"Valid: {FUSION_METHODS}"
            )

        self.fusion_weights: Optional[torch.Tensor] = None
        if fusion_weights is not None:
            w = torch.tensor(fusion_weights, dtype=torch.float32)
            self.fusion_weights = w / w.sum()  # normalise to 1
        elif self.fusion_method == "weighted_average":
            # Default: uniform weights
            self.fusion_weights = torch.ones(self.num_models) / self.num_models

        print(f"  🎯 Ensemble: {self.num_models} models, "
              f"fusion={self.fusion_method}, classes={self.num_classes}")

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return list of probability vectors from each model."""
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(waveforms)
            all_probs.append(probs)
        return all_probs

    def predict_proba(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Compute fused probability vector.

        Args:
            waveforms: (batch, 1, T)

        Returns:
            probs: (batch, num_classes) — sum(dim=1) ≈ 1.0
        """
        all_probs = self.forward(waveforms)
        stacked = torch.stack(all_probs, dim=0)  # (N, batch, C)

        if self.fusion_method == "average":
            return stacked.mean(dim=0)

        elif self.fusion_method == "weighted_average":
            w = self.fusion_weights.to(stacked.device).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)

        elif self.fusion_method == "geometric_mean":
            eps = 1e-9
            log_probs = torch.log(stacked.clamp(min=eps))
            geo = torch.exp(log_probs.mean(dim=0))
            return geo / geo.sum(dim=1, keepdim=True)

        elif self.fusion_method == "rank_average":
            # Convert probabilities to ranks (higher prob → higher rank)
            ranks = torch.zeros_like(stacked)
            for i in range(self.num_models):
                order = stacked[i].argsort(dim=1)  # ascending
                ranks[i] = order.float().argsort(dim=1)  # rank 0 = lowest prob
            avg_ranks = ranks.float().mean(dim=0)
            # Convert ranks back to a probability distribution via softmax
            return F.softmax(avg_ranks, dim=1)

        elif self.fusion_method == "max_prob":
            fused = stacked.max(dim=0).values
            return fused / fused.sum(dim=1, keepdim=True).clamp(min=1e-9)

        elif self.fusion_method == "learned_mlp":
            return self.fusion(all_probs)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def get_trainable_params(self) -> int:
        """Count trainable parameters (only fusion MLP if learned)."""
        if self.fusion is not None:
            return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        return 0

    def eval(self):
        """Set all models and fusion to eval mode."""
        super().eval()
        for model in self.models:
            model.eval()

    def train(self, mode: bool = True):
        """Set fusion MLP to train mode (models stay frozen)."""
        super().train(mode)
        # Keep base models in eval mode — only train fusion
        for model in self.models:
            model.eval()
        if self.fusion is not None:
            self.fusion.train(mode)


# ═══════════════════════════════════════════════════════════
#  Stateless fusion functions (numpy) — offline / inference
# ═══════════════════════════════════════════════════════════

def weighted_average_fusion(
    probs_list: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted average of per-model probability vectors.

    Args:
        probs_list: list of (N, C) probability arrays.
        weights:    per-model weights (len = len(probs_list)); uniform if None.

    Returns:
        fused_probs: (N, C) — rows sum to 1.
    """
    stacked = np.stack(probs_list, axis=0)  # (M, N, C)
    if weights is None:
        weights = np.ones(len(probs_list)) / len(probs_list)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    fused = np.tensordot(weights, stacked, axes=(0, 0))  # (N, C)
    row_sums = fused.sum(axis=1, keepdims=True)
    return fused / (row_sums + 1e-12)


def geometric_mean_fusion(
    probs_list: Sequence[np.ndarray],
    eps: float = 1e-9,
) -> np.ndarray:
    """Geometric mean fusion — less affected by a single near-zero model.

    Computes exp(mean(log(p + ε))) then renormalises.
    """
    log_probs = [np.log(np.maximum(p, eps)) for p in probs_list]
    geo = np.exp(np.mean(log_probs, axis=0))  # (N, C)
    row_sums = geo.sum(axis=1, keepdims=True)
    return geo / (row_sums + 1e-12)


def rank_average_fusion(
    probs_list: Sequence[np.ndarray],
    temperature: float = 1.0,
) -> np.ndarray:
    """Rank-based fusion — independent of per-model probability scale.

    1. For each model, compute per-row ranks (higher prob → higher rank).
    2. Average ranks across models.
    3. Convert average ranks back to a probability distribution via softmax.
    """
    all_ranks = []
    for probs in probs_list:
        # argsort ascending (lowest prob gets rank 0)
        # then argsort again to get ranks: rank 0 = lowest prob
        order = np.argsort(probs, axis=1)
        ranks = np.argsort(order, axis=1).astype(np.float64)
        all_ranks.append(ranks)

    avg_ranks = np.mean(all_ranks, axis=0)  # (N, C)
    return softmax(avg_ranks / max(temperature, 1e-6), axis=1)


def max_prob_fusion(
    probs_list: Sequence[np.ndarray],
) -> np.ndarray:
    """Element-wise maximum over model probabilities.

    Take the highest probability for each class across models, then renormalise.
    """
    stacked = np.stack(probs_list, axis=0)  # (M, N, C)
    fused = stacked.max(axis=0)  # (N, C)
    row_sums = fused.sum(axis=1, keepdims=True)
    return fused / (row_sums + 1e-12)


# ── Grid search for optimal per-model weights ──

def grid_search_weights(
    probs_list: Sequence[np.ndarray],
    labels: np.ndarray,
    num_classes: int = 447,
    step: float = 0.05,
) -> Dict:
    """Grid-search per-model weights that maximise Macro-F1 on a held-out set.

    Uses a constrained simplex search: generates all integer partitions of
    1/step across num_models buckets.

    Args:
        probs_list:  list of (N, C) probability arrays (one per model).
        labels:      (N,) ground-truth class ids (0 = unknown, 1..446 = known).
        num_classes: total number of classes (default 447).
        step:        weight resolution (e.g. 0.05 → weights in {0.00, 0.05, …, 1.00}).

    Returns:
        {"best_weights": [w1, w2, ...], "best_macro_f1": float,
         "best_preds": np.ndarray, "all_results": [{weights, macro_f1}, …]}
    """
    from src.metrics import macro_f1_score

    n_models = len(probs_list)
    n_steps = int(1.0 / step)

    # Pre-compute: for integer partitions of n_steps into n_models buckets
    # E.g. step=0.05, n_steps=20, n_models=4 → ~C(23,3) = 1771 combinations
    best_mf1 = -1.0
    best_weights = None
    best_preds = None
    all_results = []

    # Use itertools.combinations_with_replacement on star positions
    # Integer partition: find non-negative ints that sum to n_steps
    # This is equivalent to choosing n_models-1 separators from n_steps+n_models-1 positions
    total_positions = n_steps + n_models - 1
    for combo in itertools.combinations(range(total_positions), n_models - 1):
        # Convert separators to bucket sizes
        counts = []
        prev = -1
        for sep in combo:
            counts.append(sep - prev - 1)
            prev = sep
        counts.append(total_positions - prev - 1)
        weights = np.array(counts, dtype=np.float64) / n_steps

        fused = weighted_average_fusion(probs_list, weights=weights)
        preds = fused.argmax(axis=1)
        mf1 = macro_f1_score(labels, preds, num_classes=num_classes)

        all_results.append({
            "weights": [round(float(w), 4) for w in weights],
            "macro_f1": float(mf1),
        })

        if mf1 > best_mf1:
            best_mf1 = mf1
            best_weights = [round(float(w), 4) for w in weights]
            best_preds = preds

    return {
        "best_weights": best_weights,
        "best_macro_f1": float(best_mf1),
        "best_preds": best_preds,
        "all_results": all_results,
    }


# ── Learned MLP fusion training (offline) ──

def train_learned_fusion(
    probs_list: Sequence[np.ndarray],
    labels: np.ndarray,
    num_classes: int = 447,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 200,
    train_ratio: float = 0.8,
    patience: int = 20,
    device_str: str = "cpu",
) -> Tuple[LearnedFusion, Dict]:
    """Train a LearnedFusion MLP on per-model probability vectors.

    Splits the data into train/val (by `train_ratio`) so Macro-F1 is computed
    on a true held-out subset. Returns the best model (by val Macro-F1).

    Args:
        probs_list:  list of (N, C) probability arrays.
        labels:      (N,) ground-truth class ids.
        num_classes: total classes (447).
        hidden_dim:  MLP hidden dimension.
        dropout:     MLP dropout.
        lr:          learning rate.
        epochs:      max training epochs.
        train_ratio: fraction of data for training (rest for validation).
        patience:    early stopping patience.
        device_str:  "cpu" or "cuda".

    Returns:
        (best_fusion, {"best_val_macro_f1": …, "best_epoch": …, "history": […]})
    """
    from sklearn.model_selection import train_test_split
    from src.metrics import macro_f1_score

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    n_models = len(probs_list)
    n_samples = probs_list[0].shape[0]

    # Build input: concatenated probability vectors (N, M*C)
    X = np.concatenate(probs_list, axis=1)  # (N, M*C)
    y = np.asarray(labels, dtype=np.int64)

    # Train/val split (fall back to unstratified if classes are too sparse)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=train_ratio, random_state=42, stratify=y,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=train_ratio, random_state=42,
        )

    fusion = LearnedFusion(n_models, num_classes, hidden_dim, dropout).to(device)
    opt = torch.optim.Adam(fusion.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_val_mf1 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        fusion.train()
        opt.zero_grad()

        # Forward: split concatenated input back into per-model tensors
        # Input to LearnedFusion.forward expects list of (batch, C)
        probs_chunks = []
        chunk_size = num_classes
        for i in range(n_models):
            chunk = X_train_t[:, i * chunk_size:(i + 1) * chunk_size]
            probs_chunks.append(chunk.to(device))

        output = fusion(probs_chunks)  # (batch, C)
        loss = criterion(torch.log(output.clamp(min=1e-9)), y_train_t.to(device))
        loss.backward()
        opt.step()

        # Validate
        fusion.eval()
        with torch.no_grad():
            probs_chunks_val = []
            for i in range(n_models):
                chunk = X_val_t[:, i * chunk_size:(i + 1) * chunk_size]
                probs_chunks_val.append(chunk.to(device))
            val_output = fusion(probs_chunks_val).cpu().numpy()
            val_preds = val_output.argmax(axis=1)
            val_mf1 = macro_f1_score(y_val, val_preds, num_classes=num_classes)

        train_output = output.detach().cpu().numpy()
        train_preds = train_output.argmax(axis=1)
        train_mf1 = macro_f1_score(y_train, train_preds, num_classes=num_classes)

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_macro_f1": float(train_mf1),
            "val_macro_f1": float(val_mf1),
        })

        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = {k: v.cpu().clone() for k, v in fusion.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best
    if best_state is not None:
        fusion.load_state_dict(best_state)

    return fusion, {
        "best_val_macro_f1": float(best_val_mf1),
        "best_epoch": best_epoch,
        "history": history,
    }


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _smoke_test():
    print("=" * 50)
    print("  Ensemble Smoke Test")
    print("=" * 50)

    # Create dummy models (simple linear for fast testing)
    class DummyModel(nn.Module):
        def __init__(self, num_known=10):
            super().__init__()
            self.num_known_speakers = num_known
            self.linear = nn.Linear(80000, num_known + 1)

        def predict_proba(self, x):
            x = x.squeeze(1)
            logits = self.linear(x)
            return F.softmax(logits, dim=1)

    num_known = 10
    models = [DummyModel(num_known) for _ in range(3)]
    waveforms = torch.randn(4, 1, 80000)

    # ── Average fusion ──
    ensemble_avg = EnsembleModel(models, fusion_method="average")
    probs_avg = ensemble_avg.predict_proba(waveforms)
    assert probs_avg.shape == (4, num_known + 1)
    assert torch.allclose(probs_avg.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Average fusion: {probs_avg.shape} ✅")

    # ── Weighted average fusion ──
    ensemble_w = EnsembleModel(models, fusion_method="weighted_average",
                                fusion_weights=[0.6, 0.3, 0.1])
    probs_w = ensemble_w.predict_proba(waveforms)
    assert probs_w.shape == (4, num_known + 1)
    assert torch.allclose(probs_w.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Weighted avg:   {probs_w.shape} ✅")

    # ── Geometric mean fusion ──
    ensemble_geo = EnsembleModel(models, fusion_method="geometric_mean")
    probs_geo = ensemble_geo.predict_proba(waveforms)
    assert probs_geo.shape == (4, num_known + 1)
    assert torch.allclose(probs_geo.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Geometric mean: {probs_geo.shape} ✅")

    # ── Rank average fusion ──
    ensemble_rank = EnsembleModel(models, fusion_method="rank_average")
    probs_rank = ensemble_rank.predict_proba(waveforms)
    assert probs_rank.shape == (4, num_known + 1)
    assert torch.allclose(probs_rank.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Rank average:   {probs_rank.shape} ✅")

    # ── Max prob fusion ──
    ensemble_max = EnsembleModel(models, fusion_method="max_prob")
    probs_max = ensemble_max.predict_proba(waveforms)
    assert probs_max.shape == (4, num_known + 1)
    assert torch.allclose(probs_max.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Max prob:       {probs_max.shape} ✅")

    # ── Stateless functions (numpy) ──
    probs_np = [torch.rand(8, 11).softmax(dim=1).numpy() for _ in range(3)]
    labels_np = np.random.randint(0, 11, size=8)

    w_avg = weighted_average_fusion(probs_np, weights=[0.5, 0.3, 0.2])
    assert w_avg.shape == (8, 11)
    assert np.allclose(w_avg.sum(axis=1), 1.0)
    print(f"  weighted_average_fusion: {w_avg.shape} ✅")

    geo = geometric_mean_fusion(probs_np)
    assert geo.shape == (8, 11)
    assert np.allclose(geo.sum(axis=1), 1.0)
    print(f"  geometric_mean_fusion:   {geo.shape} ✅")

    rank = rank_average_fusion(probs_np)
    assert rank.shape == (8, 11)
    assert np.allclose(rank.sum(axis=1), 1.0)
    print(f"  rank_average_fusion:     {rank.shape} ✅")

    mx = max_prob_fusion(probs_np)
    assert mx.shape == (8, 11)
    assert np.allclose(mx.sum(axis=1), 1.0, atol=1e-5)
    print(f"  max_prob_fusion:         {mx.shape} ✅")

    # ── Grid search (small test) ──
    gs = grid_search_weights(probs_np, labels_np, num_classes=11, step=0.25)
    assert gs["best_weights"] is not None
    assert len(gs["best_weights"]) == 3
    assert gs["best_macro_f1"] >= 0.0
    print(f"  grid_search_weights: best={gs['best_weights']} "
          f"MF1={gs['best_macro_f1']:.4f} ✅")

    # ── Learned MLP fusion ──
    ensemble_mlp = EnsembleModel(models, fusion_method="learned_mlp")
    probs_mlp = ensemble_mlp.predict_proba(waveforms)
    assert probs_mlp.shape == (4, num_known + 1)
    assert torch.allclose(probs_mlp.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Learned MLP:    {probs_mlp.shape} ✅")

    # ── Train learned fusion offline ──
    # Generate balanced data to avoid stratified split issues
    probs_bal = [torch.rand(20, 11).softmax(dim=1).numpy() for _ in range(3)]
    labels_bal = np.array([i % 11 for i in range(20)], dtype=np.int64)
    fusion_trained, info = train_learned_fusion(
        probs_bal, labels_bal, num_classes=11,
        hidden_dim=32, epochs=20, train_ratio=0.75, patience=5,
    )
    assert info["best_val_macro_f1"] >= 0.0
    print(f"  train_learned_fusion: val_MF1={info['best_val_macro_f1']:.4f} "
          f"epoch={info['best_epoch']} ✅")

    # ── Training mode ──
    ensemble_mlp.train()
    assert ensemble_mlp.fusion.training
    for m in ensemble_mlp.models:
        assert not m.training
    print(f"  Train/eval mode: ✅")

    # ── Gradients flow through fusion ──
    opt = torch.optim.Adam(ensemble_mlp.fusion.parameters(), lr=1e-3)
    ensemble_mlp.train()
    probs = ensemble_mlp.predict_proba(waveforms)
    target = torch.randint(0, num_known + 1, (4,))
    loss = F.cross_entropy(torch.log(probs + 1e-7), target)
    loss.backward()
    opt.step()
    assert loss.item() > 0
    print(f"  Fusion training: loss={loss.item():.4f} ✅")

    # ── Invalid fusion method ──
    try:
        EnsembleModel(models, fusion_method="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        print(f"  Invalid method guard: ✅")

    print()
    print("  ALL ENSEMBLE TESTS PASSED ✅")


if __name__ == "__main__":
    _smoke_test()
