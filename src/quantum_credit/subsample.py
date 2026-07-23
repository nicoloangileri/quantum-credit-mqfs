"""Stratified subsampling — the heart of the fairness rule.

Quantum kernel methods scale as O(n^2) in training size, so quantum models
are trained on small subsamples. To keep the comparison honest, classical
baselines are re-trained on the *exact same* subsample (same indices, same
seed). This module is the single source of those indices.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_subsample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a stratified subsample of size ``n`` (deterministic in seed)."""
    y = np.asarray(y)
    if not 0 < n < len(y):
        raise ValueError(f"n must be in (0, {len(y)}), got {n}")
    idx = np.arange(len(y))
    sub, _ = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    return np.sort(sub)
