"""Known-answer tests for discrimination/calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_credit.metrics import calibration_bins, evaluate_probabilities, summarize_runs


class TestEvaluateProbabilities:
    def test_perfect_ranking_has_auc_one(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = evaluate_probabilities(y, p)
        assert m["auc"] == pytest.approx(1.0)
        assert m["pr_auc"] == pytest.approx(1.0)

    def test_reversed_ranking_has_auc_zero(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        assert evaluate_probabilities(y, p)["auc"] == pytest.approx(0.0)

    def test_brier_hand_computed(self):
        # errors: 0.1, 0.2, 0.4 -> mean of squares = (0.01+0.04+0.16)/3 = 0.07
        y = np.array([0, 1, 1])
        p = np.array([0.1, 0.8, 0.6])
        assert evaluate_probabilities(y, p)["brier"] == pytest.approx(0.07)

    def test_constant_half_prediction_brier(self):
        y = np.array([0, 1, 0, 1])
        p = np.full(4, 0.5)
        assert evaluate_probabilities(y, p)["brier"] == pytest.approx(0.25)

    def test_base_rate_reported(self):
        y = np.array([0, 0, 0, 1])
        m = evaluate_probabilities(y, np.array([0.1, 0.1, 0.1, 0.9]))
        assert m["base_rate"] == pytest.approx(0.25)

    def test_out_of_range_probabilities_rejected(self):
        with pytest.raises(ValueError):
            evaluate_probabilities(np.array([0, 1]), np.array([-0.1, 0.5]))


class TestCalibrationBins:
    def test_perfectly_calibrated_bins_lie_on_diagonal(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = rng.binomial(1, p)
        bins = calibration_bins(y, p, n_bins=10)
        diff = np.abs(
            np.array(bins["mean_predicted"]) - np.array(bins["fraction_positive"])
        )
        assert diff.max() < 0.05


class TestSummarizeRuns:
    def test_mean_and_std_hand_computed(self):
        runs = [{"auc": 0.70}, {"auc": 0.80}]
        s = summarize_runs(runs)
        assert s["auc"]["mean"] == pytest.approx(0.75)
        assert s["auc"]["std"] == pytest.approx(np.std([0.7, 0.8], ddof=1))
        assert s["auc"]["n"] == 2

    def test_single_run_has_zero_std(self):
        s = summarize_runs([{"auc": 0.7}])
        assert s["auc"]["std"] == 0.0

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            summarize_runs([])
