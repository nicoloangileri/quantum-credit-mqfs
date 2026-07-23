"""Optional real-QPU estimation of a tiny kernel Gram sub-block.

Design constraints (deliberate):

- **Credentials never live in this repository.** The IBM Quantum account is
  read exclusively from the user's saved ``QiskitRuntimeService`` account
  (``~/.qiskit/qiskit-ibm.json``). If no saved account exists, this module
  refuses to run with a clear error — it never prompts for or stores tokens.
- **Tiny job budget** (IBM Open plan): a single job estimating the
  ``n*(n-1)/2`` off-diagonal entries of an n x n sub-block, n <= 8, i.e.
  at most 28 circuits.
- Fidelity is estimated with the compute-uncompute circuit
  ``U_phi(y)^dagger U_phi(x) |0>``: the probability of measuring all zeros
  equals |<phi(y)|phi(x)>|^2 (Havlíček et al., 2019).

The pure circuit-construction code has no runtime dependency and is unit
tested; only ``estimate_kernel_subblock`` talks to IBM Quantum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit

from .featuremap import bind, embed_statevectors, make_feature_map
from .kernels import fidelity_kernel

MAX_POINTS = 8


class HardwareUnavailableError(RuntimeError):
    """Raised when the optional hardware path cannot (or must not) run."""


def fidelity_circuit(d: int, x: np.ndarray, y: np.ndarray, *, reps: int = 2) -> QuantumCircuit:
    """Compute-uncompute circuit whose all-zeros probability is K(x, y)."""
    fm = make_feature_map(d, reps=reps)
    qc = QuantumCircuit(d)
    qc.compose(bind(fm, x), inplace=True)
    qc.compose(bind(fm, y).inverse(), inplace=True)
    qc.measure_all()
    return qc


def subblock_pairs(n: int) -> list[tuple[int, int]]:
    """Upper-triangle index pairs of an n x n Gram sub-block."""
    if not 2 <= n <= MAX_POINTS:
        raise ValueError(f"n must be in [2, {MAX_POINTS}] to respect the job budget")
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


@dataclass(frozen=True)
class HardwareKernelResult:
    backend: str
    job_id: str
    shots: int
    n_points: int
    pairs: list[tuple[int, int]]
    k_simulated: list[float]
    k_measured: list[float]

    @property
    def mae(self) -> float:
        a = np.asarray(self.k_simulated)
        b = np.asarray(self.k_measured)
        return float(np.mean(np.abs(a - b)))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["mae"] = self.mae
        return out


def _load_service():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as err:
        raise HardwareUnavailableError(
            "qiskit-ibm-runtime is not installed; install with "
            "`uv pip install 'quantum-credit-mqfs[hardware]'`"
        ) from err
    try:
        return QiskitRuntimeService()  # reads the saved account in ~/.qiskit
    except Exception as err:
        raise HardwareUnavailableError(
            "no saved IBM Quantum account found (~/.qiskit). Save one once with "
            "QiskitRuntimeService.save_account(...); this repo never stores tokens."
        ) from err


def estimate_kernel_subblock(
    X_angles: np.ndarray,
    *,
    shots: int = 4096,
    reps: int = 2,
    backend_name: str | None = None,
) -> HardwareKernelResult:
    """Estimate K over a tiny sub-block on the least-busy real backend."""
    X = np.atleast_2d(np.asarray(X_angles, dtype=float))
    n, d = X.shape
    pairs = subblock_pairs(n)

    states = embed_statevectors(X, reps=reps)
    k_sim = fidelity_kernel(states)

    service = _load_service()
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False, min_num_qubits=d)
    )
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits = [pm.run(fidelity_circuit(d, X[i], X[j], reps=reps)) for i, j in pairs]

    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=shots)
    result = job.result()

    zeros = "0" * d
    k_meas = []
    for pub in result:
        counts = pub.data.meas.get_counts()
        k_meas.append(counts.get(zeros, 0) / shots)

    return HardwareKernelResult(
        backend=backend.name,
        job_id=job.job_id(),
        shots=shots,
        n_points=n,
        pairs=pairs,
        k_simulated=[float(k_sim[i, j]) for i, j in pairs],
        k_measured=k_meas,
    )
