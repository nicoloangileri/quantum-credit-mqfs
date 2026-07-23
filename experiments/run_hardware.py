"""Optional real-QPU kernel estimation — runs ONLY with the explicit --hardware flag.

Estimates one tiny Gram sub-block (default 6 points -> 15 circuits, one job)
on the least-busy IBM backend and compares measured vs statevector-simulated
kernel entries. Requires a saved QiskitRuntimeService account (~/.qiskit);
this repository never stores or asks for tokens. Designed to fit comfortably
in the IBM Open plan budget.

Usage:
    python experiments/run_hardware.py --hardware [--n 6] [--shots 4096]
"""

from __future__ import annotations

import argparse
import sys

from experiments.common import save_results
from quantum_credit import config
from quantum_credit.data import load_dataset, train_test_frames, xy
from quantum_credit.features import AngleReducer
from quantum_credit.quantum.hardware import (
    MAX_POINTS,
    HardwareUnavailableError,
    estimate_kernel_subblock,
)
from quantum_credit.subsample import stratified_subsample_indices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", action="store_true",
                    help="explicitly enable the real-QPU job (required)")
    ap.add_argument("--n", type=int, default=6, help=f"points in the sub-block (<= {MAX_POINTS})")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", type=str, default=None)
    args = ap.parse_args()

    if not args.hardware:
        print("Refusing to run: pass --hardware to submit a real-QPU job.")
        return 2

    df, _ = load_dataset()
    train, _ = train_test_frames(df)
    X_tr, y_tr = xy(train)
    idx = stratified_subsample_indices(y_tr, 500, seed=config.SUBSAMPLE_SEEDS[0])
    red = AngleReducer(d=config.N_QUBITS, method="mi", seed=config.SUBSAMPLE_SEEDS[0])
    ang = red.fit_transform(X_tr.iloc[idx], y_tr[idx])[: args.n]

    n_circuits = args.n * (args.n - 1) // 2
    print(f"Submitting {n_circuits} fidelity circuits ({args.shots} shots) ...")
    try:
        res = estimate_kernel_subblock(
            ang, shots=args.shots, reps=config.FEATURE_MAP_REPS, backend_name=args.backend
        )
    except HardwareUnavailableError as err:
        print(f"Hardware path unavailable: {err}")
        return 3

    payload = res.to_dict()
    payload["selected_features"] = red.selected
    save_results("hardware_kernel.json", payload)
    print(f"backend={res.backend} job={res.job_id} MAE(sim, measured)={res.mae:.4f}")
    for (i, j), ks, km in zip(res.pairs, res.k_simulated, res.k_measured, strict=True):
        print(f"  K[{i},{j}]  sim={ks:.4f}  measured={km:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
