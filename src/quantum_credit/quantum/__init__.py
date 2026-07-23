"""Quantum models: fidelity quantum kernel and variational classifier.

Both run on *exact statevector simulation*: for the circuit sizes studied
here (4-8 qubits) the quantum state can be represented exactly, so kernels
and expectation values are computed to machine precision, with no shot
noise. Wall-clock costs reported by the experiments are therefore the cost
of *simulating* the quantum models — the honest quantity at this scale.
"""
