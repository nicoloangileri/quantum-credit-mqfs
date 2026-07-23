"""Known-answer tests for the cost-matrix math and profit curves."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_credit.costs import (
    CostMatrix,
    cost_curve,
    expected_cost,
    optimal_cost_threshold,
    optimal_profit_threshold,
    profit_curve,
)

CM = CostMatrix(fn=5.0, fp=1.0)


class TestCostMatrix:
    def test_bayes_threshold_is_one_sixth_for_5_to_1(self):
        assert CM.bayes_threshold == pytest.approx(1.0 / 6.0)

    def test_nonpositive_costs_rejected(self):
        with pytest.raises(ValueError):
            CostMatrix(fn=0.0, fp=1.0)


class TestExpectedCost:
    def test_hand_computed_mixed_errors(self):
        # y:    1 approved (FN, cost 5); 0 rejected (FP, cost 1); 2 correct
        y = np.array([1, 0, 1, 0])
        yhat = np.array([0, 1, 1, 0])
        assert expected_cost(y, yhat, CM) == pytest.approx((5.0 + 1.0) / 4.0)

    def test_perfect_decisions_cost_zero(self):
        y = np.array([1, 0, 1])
        assert expected_cost(y, y, CM) == 0.0

    def test_all_approved_costs_fn_times_default_rate(self):
        y = np.array([1, 1, 0, 0])
        yhat = np.zeros(4, dtype=int)
        assert expected_cost(y, yhat, CM) == pytest.approx(5.0 * 2 / 4)


class TestCostCurve:
    def test_endpoints(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.1, 0.2])
        t, c = cost_curve(y, p, CM, thresholds=np.array([0.0, 1.0]))
        # t=0: everyone rejected -> 2 FP -> cost 2/4; t=1: everyone approved -> 2 FN -> 10/4
        assert c[0] == pytest.approx(0.5)
        assert c[1] == pytest.approx(2.5)

    def test_perfect_model_reaches_zero_cost(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.1, 0.2])
        _, best = optimal_cost_threshold(y, p, CM)
        assert best == 0.0

    def test_calibrated_probabilities_recover_bayes_threshold(self):
        """Elkan (2001): with calibrated p, argmin cost ~= c_fp/(c_fp+c_fn)."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0.0, 1.0, 200_000)
        y = rng.binomial(1, p)
        t_star, _ = optimal_cost_threshold(y, p, CM)
        assert abs(t_star - CM.bayes_threshold) < 0.02


class TestProfitCurve:
    def test_hand_computed_profit(self):
        # margin 10%, LGD 60%. Approve iff p < 0.5.
        y = np.array([0, 1, 0])
        p = np.array([0.2, 0.3, 0.9])
        lb = np.array([100_000.0, 50_000.0, 200_000.0])
        t, v = profit_curve(y, p, lb, margin=0.10, lgd=0.60, thresholds=np.array([0.5]))
        # approved: clients 0 (good, +10k) and 1 (default, -30k); client 2 rejected
        assert v[0] == pytest.approx(10_000.0 - 30_000.0)

    def test_threshold_zero_approves_nobody(self):
        y = np.array([0, 1])
        p = np.array([0.1, 0.9])
        lb = np.array([1000.0, 1000.0])
        _, v = profit_curve(y, p, lb, thresholds=np.array([0.0]))
        assert v[0] == 0.0

    def test_perfect_model_approves_only_good_clients(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.05, 0.95, 0.1, 0.9])
        lb = np.full(4, 10_000.0)
        t_star, best = optimal_profit_threshold(y, p, lb, margin=0.1, lgd=0.6)
        assert best == pytest.approx(2 * 1000.0)  # two good clients x 10% margin

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            profit_curve(np.array([0, 1]), np.array([0.1, 0.2]), np.array([1.0]))
