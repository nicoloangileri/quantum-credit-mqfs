"""VQC tests: probability sanity, determinism, learning on a separable toy."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from quantum_credit.quantum.featuremap import embed_statevectors
from quantum_credit.quantum.vqc import StatevectorVQC, parity_diagonal


class TestParityDiagonal:
    def test_two_qubits_hand_computed(self):
        # |00> +1, |01> -1, |10> -1, |11> +1
        np.testing.assert_array_equal(parity_diagonal(2), [1.0, -1.0, -1.0, 1.0])

    def test_sums_to_zero(self):
        assert parity_diagonal(3).sum() == 0.0


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, np.pi, (80, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    states = embed_statevectors(X, reps=1)
    return states, y


class TestStatevectorVQC:

    def test_probabilities_are_valid(self, toy):
        states, y = toy
        clf = StatevectorVQC(n_qubits=2, maxiter=30, restarts=1, seed=0).fit(states, y)
        proba = clf.predict_proba(states)
        assert proba.shape == (80, 2)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_deterministic_given_seed(self, toy):
        states, y = toy
        a = StatevectorVQC(n_qubits=2, maxiter=25, restarts=1, seed=3).fit(states, y)
        b = StatevectorVQC(n_qubits=2, maxiter=25, restarts=1, seed=3).fit(states, y)
        np.testing.assert_allclose(a.theta_, b.theta_)

    def test_learns_separable_toy_above_chance(self, toy):
        states, y = toy
        clf = StatevectorVQC(n_qubits=2, maxiter=150, restarts=2, seed=0).fit(states, y)
        auc = roc_auc_score(y, clf.predict_proba(states)[:, 1])
        assert auc > 0.75

    def test_training_reduces_loss(self, toy):
        states, y = toy
        clf = StatevectorVQC(n_qubits=2, maxiter=100, restarts=1, seed=1).fit(states, y)
        assert clf.loss_curve_[-1] < clf.loss_curve_[0]

    def test_wrong_state_dimension_rejected(self, toy):
        states, y = toy
        with pytest.raises(ValueError):
            StatevectorVQC(n_qubits=3).fit(states, y)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            StatevectorVQC(n_qubits=2).predict_proba(np.zeros((1, 4), dtype=complex))
