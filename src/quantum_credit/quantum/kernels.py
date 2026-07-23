"""Fidelity quantum kernel K(x, y) = |<phi(x)|phi(y)>|^2 (exact, statevector).

On a statevector simulator the kernel Gram matrix is computed exactly as
the squared modulus of the inner-product matrix of the embedded states —
mathematically identical to the compute-uncompute fidelity estimated on
hardware (see ``hardware.py``), but with zero shot noise.
"""

from __future__ import annotations

import numpy as np


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    """Kernel matrix between two sets of statevectors (rows).

    Returns K[i, j] = |<a_i|b_j>|^2, clipped into [0, 1] against float
    round-off. With ``states_b=None`` computes the symmetric Gram matrix.
    """
    a = np.atleast_2d(np.asarray(states_a))
    b = a if states_b is None else np.atleast_2d(np.asarray(states_b))
    if a.shape[1] != b.shape[1]:
        raise ValueError("statevector dimensions differ")
    k = np.abs(a @ b.conj().T) ** 2
    if states_b is None:
        k = 0.5 * (k + k.T)  # enforce exact symmetry against round-off
    return np.clip(k, 0.0, 1.0)
