"""Regenerate every number, figure and table in the repository, end to end.

    python experiments/run_all.py

Designed to complete on a laptop in well under 30 minutes (measured time is
written to results/run_metadata.json). The optional real-QPU step is NOT part
of this pipeline; run experiments/run_hardware.py --hardware separately.
"""

from __future__ import annotations

import datetime as dt
import time

from experiments import make_figures, run_classical_full, run_matched_subsamples
from experiments.common import environment_info, save_results
from experiments.render_tables import sync_readme
from quantum_credit.data import load_dataset


def main() -> None:
    t0 = time.perf_counter()
    stage_times: dict[str, float] = {}

    for stage, fn in [
        ("load_dataset", lambda: load_dataset()),
        ("classical_full", run_classical_full.main),
        ("matched_subsamples", run_matched_subsamples.main),
        ("figures", make_figures.main),
        ("readme_tables", lambda: sync_readme(write=True)),
    ]:
        print(f"\n########## {stage}")
        s = time.perf_counter()
        fn()
        stage_times[stage] = time.perf_counter() - s

    total = time.perf_counter() - t0
    save_results(
        "run_metadata.json",
        {
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "total_wall_clock_s": total,
            "stage_wall_clock_s": stage_times,
            "environment": environment_info(),
        },
    )
    print(f"\nTOTAL {total / 60:.1f} min")


if __name__ == "__main__":
    main()
