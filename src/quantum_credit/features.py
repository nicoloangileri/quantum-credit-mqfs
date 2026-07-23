"""Feature reduction to d = 4-8 dimensions for the quantum models.

Default method: **mutual-information selection** (top-d raw features by
mutual information with the label, estimated on the training subsample
only). Chosen over PCA because (a) it preserves the semantics of the
selected features, so "what the quantum model sees" stays interpretable;
(b) it does not rotate integer-coded categorical columns into dense linear
combinations; and (c) on this dataset the repayment-status variables carry
most of the signal, which MI captures directly. PCA is implemented as an
alternative (``method="pca"``) and selectable from the experiment CLI.

Angle encoding: selected features are min-max scaled to [0, pi] (fitted on
the training subsample only; test values are clipped to the training range,
which is the honest out-of-sample choice and avoids angle wrap-around).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from .data import CATEGORICAL, PAY_COLS


@dataclass
class AngleReducer:
    """Fit on train only; transform maps rows to angles in [0, pi]^d."""

    d: int
    method: str = "mi"  # "mi" | "pca"
    seed: int = 0
    selected: list[str] = field(default_factory=list)
    _lo: np.ndarray | None = None
    _hi: np.ndarray | None = None
    _scaler: StandardScaler | None = None
    _pca: PCA | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> AngleReducer:
        if self.method == "mi":
            discrete = np.array([c in CATEGORICAL or c in PAY_COLS for c in X.columns])
            mi = mutual_info_classif(
                X.to_numpy(), y, discrete_features=discrete, random_state=self.seed
            )
            order = np.argsort(mi)[::-1][: self.d]
            self.selected = [X.columns[i] for i in sorted(order)]
            Z = X[self.selected].to_numpy(dtype=float)
        elif self.method == "pca":
            self._scaler = StandardScaler().fit(X.to_numpy(dtype=float))
            self._pca = PCA(n_components=self.d, random_state=self.seed).fit(
                self._scaler.transform(X.to_numpy(dtype=float))
            )
            self.selected = [f"PC{i + 1}" for i in range(self.d)]
            Z = self._pca.transform(self._scaler.transform(X.to_numpy(dtype=float)))
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self._lo = Z.min(axis=0)
        self._hi = Z.max(axis=0)
        span = self._hi - self._lo
        if np.any(span <= 0):
            raise ValueError("degenerate feature with zero training range")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._lo is None or self._hi is None:
            raise RuntimeError("call fit() first")
        if self.method == "mi":
            Z = X[self.selected].to_numpy(dtype=float)
        else:
            assert self._scaler is not None and self._pca is not None
            Z = self._pca.transform(self._scaler.transform(X.to_numpy(dtype=float)))
        Z = np.clip(Z, self._lo, self._hi)
        return np.pi * (Z - self._lo) / (self._hi - self._lo)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
