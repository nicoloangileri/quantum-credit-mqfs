"""Assemble everything the paper needs from results/*.json (fail-loud).

The paper builder never computes statistics and never invents numbers: this
module only *reads* experiment outputs and derives labels/orderings from them.
"""

from __future__ import annotations

import json
from typing import Any

from experiments.common import load_results
from experiments.render_tables import LABELS
from quantum_credit import config

CLASSICAL_SET = {"logistic_calibrated", "random_forest", "hist_gradient_boosting", "svc_rbf"}
QUANTUM_SET = {"qk_svc_d4", "qk_svc_d8", "vqc_d4"}


def _optional(name: str) -> dict | None:
    try:
        return load_results(name)
    except FileNotFoundError:
        return None


def load_all() -> dict[str, Any]:
    full = load_results("classical_full.json")
    matched = load_results("matched_subsamples.json")
    with open(config.RESULTS_DIR / "data_manifest.json") as fh:
        manifest = json.load(fh)

    best_full = max(full["models"], key=lambda k: full["models"][k]["test"]["auc"])

    biggest = str(max(matched["config"]["sizes"]))
    agg = matched["aggregates"][biggest]
    classical = {k: v["auc"]["mean"] for k, v in agg.items() if k in CLASSICAL_SET}
    quantum = {k: v["auc"]["mean"] for k, v in agg.items() if k in QUANTUM_SET}
    best_c = max(classical, key=classical.get)
    best_q = max(quantum, key=quantum.get)

    vqc_run = next(r for r in matched["runs"] if r["model"] == "vqc_d4")

    thr = []
    for name, e in full["models"].items():
        thr.append(
            f"{LABELS[name].replace('**', '')} "
            f"$\\hat t = {e['thresholds']['cost_optimal_oof']:.3f}$"
        )
    imp = full["models"][best_full]["permutation_importance"]
    runners = ", ".join(
        "\\texttt{" + r["feature"].replace("_", "\\_") + "}" for r in imp[1:3]
    )

    best_dec = full["models"][best_full]["test_decisions"]

    return {
        "full": full,
        "matched": matched,
        "best_profit_m": best_dec["profit_at_oof_threshold_ntd"] / 1e6,
        "best_approval_pct": 100 * best_dec["approval_rate_at_oof_threshold"],
        "manifest": manifest,
        "meta": _optional("run_metadata.json"),
        "hardware": _optional("hardware_kernel.json"),
        "best_full_model": best_full,
        "best_c2000_name": best_c,
        "best_q2000_name": best_q,
        "best_c2000": classical[best_c],
        "best_q2000": quantum[best_q],
        "delta_best_classical_minus_best_quantum_n2000": classical[best_c] - quantum[best_q],
        "vqc_reps": config.VQC_ANSATZ_REPS,
        "vqc_maxiter": config.VQC_MAXITER,
        "vqc_restarts": config.VQC_RESTARTS,
        "vqc_params": vqc_run["detail"]["n_params"],
        "threshold_list_tex": "; ".join(thr),
        "imp_runners_tex": runners,
    }
