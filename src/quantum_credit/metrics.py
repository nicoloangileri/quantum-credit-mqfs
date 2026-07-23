"""Discrimination and calibration metrics with fold aggregation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate_probabilities(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AUC, PR-AUC and Brier score for predicted default probabilities."""
    y = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(np.mean(y)),
    }


def calibration_bins(
    y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10
) -> dict[str, list[float]]:
    """Reliability-diagram coordinates (quantile bins)."""
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean/std/n for every numeric key across repeated runs (CV folds, seeds)."""
    if not runs:
        raise ValueError("no runs to summarize")
    keys = sorted(runs[0])
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[k] = {"mean": float(vals.mean()), "std": std, "n": len(vals)}
    return out
