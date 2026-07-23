"""Quantum-kernel tests: symmetry, PSD, diagonal, independent cross-check."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quantum_credit.quantum.featuremap import bind, embed_statevectors, make_feature_map
from quantum_credit.quantum.kernels import fidelity_kernel

RNG = np.random.default_rng(11)
X_SMALL = RNG.uniform(0.0, np.pi, (12, 4))
STATES = embed_statevectors(X_SMALL, reps=2)


class TestFeatureMap:
    def test_statevectors_are_normalised(self):
        norms = np.linalg.norm(STATES, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_dimension_is_2_to_d(self):
        assert STATES.shape == (12, 16)

    def test_too_few_features_rejected(self):
        with pytest.raises(ValueError):
            make_feature_map(1)

    def test_bind_rejects_wrong_length(self):
        fm = make_feature_map(4)
        with pytest.raises(ValueError):
            bind(fm, np.array([0.1, 0.2]))


class TestFidelityKernelProperties:
    def test_diagonal_is_one(self):
        k = fidelity_kernel(STATES)
        np.testing.assert_allclose(np.diag(k), 1.0, atol=1e-10)

    def test_symmetric(self):
        k = fidelity_kernel(STATES)
        np.testing.assert_allclose(k, k.T, atol=1e-12)

    def test_positive_semidefinite(self):
        k = fidelity_kernel(STATES)
        eigs = np.linalg.eigvalsh(k)
        assert eigs.min() > -1e-9

    def test_values_in_unit_interval(self):
        k = fidelity_kernel(STATES)
        assert k.min() >= 0.0 and k.max() <= 1.0

    def test_identical_points_have_unit_kernel(self):
        s = embed_statevectors(np.vstack([X_SMALL[0], X_SMALL[0]]))
        assert fidelity_kernel(s)[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_rectangular_cross_kernel_shape(self):
        k = fidelity_kernel(STATES[:5], STATES[5:])
        assert k.shape == (5, 7)


class TestKnownAnswerCrossCheck:
    """Fast Gram computation must match Qiskit's own inner products exactly."""

    def test_matches_statevector_inner_products(self):
        fm = make_feature_map(4, reps=2)
        k_fast = fidelity_kernel(STATES[:5])
        for i in range(5):
            for j in range(5):
                sv_i = Statevector(bind(fm, X_SMALL[i]))
                sv_j = Statevector(bind(fm, X_SMALL[j]))
                k_ref = abs(sv_i.inner(sv_j)) ** 2
                assert k_fast[i, j] == pytest.approx(k_ref, abs=1e-10)

    def test_zero_angles_give_uniform_superposition_kernel_one(self):
        """x = 0 leaves only the Hadamard layer: |phi(0)> = H^{(x)d}|0>, so
        K(0, 0) = 1 while K(0, x) is generally < 1 for x != 0."""
        z = np.zeros((1, 4))
        s0 = embed_statevectors(z)
        assert fidelity_kernel(s0)[0, 0] == pytest.approx(1.0)
        kx = fidelity_kernel(s0, STATES)
        assert (kx < 1.0 - 1e-6).all()
