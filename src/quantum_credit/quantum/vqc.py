"""A small variational quantum classifier on exact statevectors.

Model (standard VQC, Havlíček et al. 2019 style):

    |psi(x, theta)> = U(theta) |phi(x)>,      m(x, theta) = <psi| P |psi>,

with |phi(x)> the ZZFeatureMap embedding, U(theta) a RealAmplitudes ansatz
and P the qubit-parity observable (diagonal in the computational basis).
The predicted default probability is p(x) = (1 - m) / 2, trained by
minimising class-weighted binary cross-entropy with COBYLA.

Implementation note: because P is diagonal and |phi(x)> is cached once per
dataset, each objective evaluation only applies the (2^d x 2^d) ansatz
unitary to the cached state matrix — exact simulation, no shot noise, and
fast enough to train on thousands of points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

from .. import config


def make_ansatz(d: int, reps: int = config.VQC_ANSATZ_REPS):
    """RealAmplitudes ansatz with a shim over the Qiskit 1.x/2.x API change."""
    try:
        from qiskit.circuit.library import real_amplitudes

        return real_amplitudes(num_qubits=d, reps=reps)
    except ImportError:
        from qiskit.circuit.library import RealAmplitudes

        return RealAmplitudes(num_qubits=d, reps=reps)


def parity_diagonal(n_qubits: int) -> np.ndarray:
    """Diagonal of the parity observable Z^{(x)n}: (-1)^popcount(basis index)."""
    idx = np.arange(2**n_qubits)
    pop = np.array([bin(i).count("1") for i in idx])
    return np.where(pop % 2 == 0, 1.0, -1.0)


@dataclass
class StatevectorVQC:
    """Binary classifier; ``fit``/``predict_proba`` operate on cached statevectors."""

    n_qubits: int
    ansatz_reps: int = config.VQC_ANSATZ_REPS
    maxiter: int = config.VQC_MAXITER
    restarts: int = config.VQC_RESTARTS
    seed: int = 0
    theta_: np.ndarray | None = None
    loss_curve_: list[float] = field(default_factory=list)
    n_params_: int = 0

    def _unitary(self, theta: np.ndarray) -> np.ndarray:
        qc = self._ansatz.assign_parameters(theta)
        return Operator(qc).data

    def _probs(self, states: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """p(default) for each cached statevector row."""
        psi = states @ self._unitary(theta).T  # row form of U|phi>
        m = (np.abs(psi) ** 2) @ self._parity
        return np.clip((1.0 - m) / 2.0, 1e-9, 1.0 - 1e-9)

    def fit(self, states: np.ndarray, y: np.ndarray) -> StatevectorVQC:
        states = np.atleast_2d(states)
        if states.shape[1] != 2**self.n_qubits:
            raise ValueError("statevector dimension does not match n_qubits")
        y = np.asarray(y, dtype=float)
        self._ansatz = make_ansatz(self.n_qubits, self.ansatz_reps)
        self._parity = parity_diagonal(self.n_qubits)
        self.n_params_ = self._ansatz.num_parameters
        # balanced class weights, as for the classical baselines
        w1 = len(y) / (2.0 * max(y.sum(), 1.0))
        w0 = len(y) / (2.0 * max((1.0 - y).sum(), 1.0))
        weights = np.where(y == 1, w1, w0)

        def loss(theta: np.ndarray) -> float:
            p = self._probs(states, theta)
            val = -float(np.mean(weights * (y * np.log(p) + (1 - y) * np.log(1 - p))))
            self.loss_curve_.append(val)
            return val

        rng = np.random.default_rng(self.seed)
        best_val, best_theta = np.inf, None
        for _ in range(self.restarts):
            theta0 = rng.uniform(-np.pi, np.pi, self.n_params_)
            res = minimize(loss, theta0, method="COBYLA", options={"maxiter": self.maxiter})
            if res.fun < best_val:
                best_val, best_theta = float(res.fun), np.asarray(res.x)
        self.theta_ = best_theta
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("call fit() first")
        p1 = self._probs(np.atleast_2d(states), self.theta_)
        return np.column_stack([1.0 - p1, p1])
