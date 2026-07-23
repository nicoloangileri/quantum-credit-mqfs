"""Finance-aware, cost-sensitive evaluation.

Two complementary views, both computed on predicted default probabilities:

1. **Unit-exposure expected misclassification cost.** A false negative
   (approving a client who defaults) costs ``c_fn``; a false positive
   (rejecting a good client) costs ``c_fp``. For calibrated probabilities
   the Bayes-optimal decision rule is: predict "default" (reject) iff
   ``p >= c_fp / (c_fp + c_fn)`` (Elkan, 2001). With the classic 5:1
   credit-scoring asymmetry this gives t* = 1/6 ≈ 0.167 — how close each
   model's empirical optimum lands to this value is itself a calibration
   diagnostic.

2. **NT$ profit curves with heterogeneous exposure.** Each approved good
   client yields ``margin * LIMIT_BAL``; each approved defaulter loses
   ``lgd * LIMIT_BAL``. Both parameters are stylised and clearly flagged
   as illustrative; the full curve over thresholds shows sensitivity.

Convention: the positive class (y = 1) is "default"; a client is
**approved** iff the predicted default probability is **below** the
threshold t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class CostMatrix:
    """Unit costs for the two error types (correct decisions cost 0)."""

    fn: float = config.COST_FN  # approved, then defaulted
    fp: float = config.COST_FP  # rejected, would have repaid

    def __post_init__(self) -> None:
        if self.fn <= 0 or self.fp <= 0:
            raise ValueError("costs must be positive")

    @property
    def bayes_threshold(self) -> float:
        """Optimal rejection threshold for perfectly calibrated probabilities."""
        return self.fp / (self.fp + self.fn)


def default_threshold_grid(size: int = config.THRESHOLD_GRID_SIZE) -> np.ndarray:
    return np.linspace(0.0, 1.0, size)


def expected_cost(
    y_true: np.ndarray, y_pred: np.ndarray, cm: CostMatrix = CostMatrix()
) -> float:
    """Average per-applicant cost of hard decisions ``y_pred`` (1 = reject)."""
    y = np.asarray(y_true, dtype=int)
    yhat = np.asarray(y_pred, dtype=int)
    if y.shape != yhat.shape:
        raise ValueError("shape mismatch")
    fn = int(np.sum((yhat == 0) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    return (cm.fn * fn + cm.fp * fp) / len(y)


def cost_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    cm: CostMatrix = CostMatrix(),
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost as a function of the rejection threshold."""
    t = default_threshold_grid() if thresholds is None else np.asarray(thresholds)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    reject = p[None, :] >= t[:, None]  # (T, n)
    fn = np.sum(~reject & (y == 1)[None, :], axis=1)
    fp = np.sum(reject & (y == 0)[None, :], axis=1)
    return t, (cm.fn * fn + cm.fp * fp) / len(y)


def optimal_cost_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    cm: CostMatrix = CostMatrix(),
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """(t*, cost at t*) minimising expected cost on the given sample."""
    t, c = cost_curve(y_true, p, cm, thresholds)
    i = int(np.argmin(c))
    return float(t[i]), float(c[i])


def profit_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    limit_bal: np.ndarray,
    *,
    margin: float = config.PROFIT_MARGIN,
    lgd: float = config.LGD_FRACTION,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Total portfolio profit (NT$) vs approval threshold.

    Approve iff p < t. Approved good client: +margin*limit. Approved
    defaulter: -lgd*limit. Rejected clients contribute 0.
    """
    t = default_threshold_grid() if thresholds is None else np.asarray(thresholds)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    lb = np.asarray(limit_bal, dtype=float)
    if not (y.shape == p.shape == lb.shape):
        raise ValueError("shape mismatch")
    per_client = np.where(y == 0, margin * lb, -lgd * lb)
    approve = p[None, :] < t[:, None]
    return t, approve @ per_client


def optimal_profit_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    limit_bal: np.ndarray,
    *,
    margin: float = config.PROFIT_MARGIN,
    lgd: float = config.LGD_FRACTION,
) -> tuple[float, float]:
    """(t*, profit at t*) maximising NT$ profit on the given sample."""
    t, v = profit_curve(y_true, p, limit_bal, margin=margin, lgd=lgd)
    i = int(np.argmax(v))
    return float(t[i]), float(v[i])
