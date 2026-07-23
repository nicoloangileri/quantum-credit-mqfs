"""ZZFeatureMap construction and exact statevector embedding.

The data-encoding circuit is the second-order Pauli-Z evolution feature map
of Havlíček et al. (2019): Hadamards followed by single- and two-qubit
phase rotations with angles phi_i(x) = x_i and phi_ij(x) = (pi - x_i)(pi - x_j),
repeated ``reps`` times. Features are expected as angles in [0, pi]
(see ``features.AngleReducer``).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def make_feature_map(d: int, reps: int = 2):
    """ZZFeatureMap with a shim over the Qiskit 1.x/2.x API change."""
    if d < 2:
        raise ValueError("ZZFeatureMap needs at least 2 features")
    if d >= 10:
        raise ValueError("parameter ordering assumes d < 10 (x[i] sorts lexicographically)")
    try:  # Qiskit >= 1.3: function-based constructor
        from qiskit.circuit.library import zz_feature_map

        return zz_feature_map(feature_dimension=d, reps=reps)
    except ImportError:  # older Qiskit: class-based constructor
        from qiskit.circuit.library import ZZFeatureMap

        return ZZFeatureMap(feature_dimension=d, reps=reps)


def bind(feature_map: QuantumCircuit, x: np.ndarray) -> QuantumCircuit:
    """Bind one data point to the feature map's parameters (sorted order x[0..d-1])."""
    params = list(feature_map.parameters)
    if len(params) != len(x):
        raise ValueError(f"expected {len(params)} features, got {len(x)}")
    values = np.asarray(x, dtype=float)
    return feature_map.assign_parameters(dict(zip(params, values, strict=True)))


def embed_statevectors(X: np.ndarray, *, reps: int = 2) -> np.ndarray:
    """Exact statevectors |phi(x_i)> as a complex array of shape (n, 2^d).

    This matrix is the single expensive quantum-simulation step; both the
    fidelity kernel and the VQC reuse it.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fm = make_feature_map(X.shape[1], reps=reps)
    out = np.empty((X.shape[0], 2 ** X.shape[1]), dtype=complex)
    for i, x in enumerate(X):
        out[i] = Statevector(bind(fm, x)).data
    return out
