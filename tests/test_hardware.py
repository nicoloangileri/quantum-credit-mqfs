"""Hardware-path tests: circuit construction is pure; runtime access is guarded.

The real-QPU call itself is exercised only via `experiments/run_hardware.py
--hardware` with saved user credentials; it is never executed in CI.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quantum_credit.quantum.featuremap import embed_statevectors
from quantum_credit.quantum.hardware import (
    HardwareUnavailableError,
    fidelity_circuit,
    subblock_pairs,
)
from quantum_credit.quantum.kernels import fidelity_kernel

HAS_RUNTIME = importlib.util.find_spec("qiskit_ibm_runtime") is not None


class TestFidelityCircuit:
    def test_compute_uncompute_structure(self):
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0.4, 0.3, 0.2, 0.1])
        qc = fidelity_circuit(4, x, y, reps=2)
        assert qc.num_qubits == 4
        assert qc.num_parameters == 0  # fully bound
        assert qc.num_clbits == 4  # measured
        assert qc.depth() > 0

    def test_identical_points_give_identity_circuit_statistics(self):
        """U(x) followed by U(x)^dagger returns |0...0> exactly."""
        from qiskit.quantum_info import Statevector

        x = np.array([0.7, 1.1, 0.3, 2.0])
        qc = fidelity_circuit(4, x, x, reps=2)
        qc.remove_final_measurements()
        amp0 = Statevector(qc).data[0]
        assert abs(amp0) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_all_zeros_probability_equals_simulated_kernel(self):
        """P(0...0) of the compute-uncompute circuit == statevector kernel."""
        from qiskit.quantum_info import Statevector

        rng = np.random.default_rng(2)
        X = rng.uniform(0, np.pi, (2, 4))
        states = embed_statevectors(X, reps=2)
        k_expected = fidelity_kernel(states)[0, 1]
        qc = fidelity_circuit(4, X[0], X[1], reps=2)
        qc.remove_final_measurements()
        p0 = abs(Statevector(qc).data[0]) ** 2
        assert p0 == pytest.approx(k_expected, abs=1e-10)


class TestSubblockPairs:
    def test_pair_count_is_n_choose_2(self):
        assert len(subblock_pairs(6)) == 15

    def test_budget_guard(self):
        with pytest.raises(ValueError):
            subblock_pairs(9)
        with pytest.raises(ValueError):
            subblock_pairs(1)


@pytest.mark.skipif(HAS_RUNTIME, reason="qiskit-ibm-runtime installed; guard not triggered")
class TestCredentialGuard:
    def test_refuses_without_runtime_package(self):
        from quantum_credit.quantum.hardware import estimate_kernel_subblock

        X = np.random.default_rng(0).uniform(0, np.pi, (3, 4))
        with pytest.raises(HardwareUnavailableError, match="qiskit-ibm-runtime"):
            estimate_kernel_subblock(X)
