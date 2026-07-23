"""Part 2 — the apples-to-apples study at quantum-feasible training sizes.

FAIRNESS RULE (the point of this paper): the fidelity quantum kernel costs
O(n^2) circuit evaluations, so quantum models can only be trained on small
subsamples. Classical baselines are therefore re-trained on the *exact same
stratified subsamples* (same indices, same seeds). Quantum-on-2000 is never
compared against classical-on-22500.

For every (size, seed) draw:
- classical: calibrated LR, RF, HistGB, RBF-SVC (the QK-SVC's classical twin);
- quantum:   QK-SVC (fidelity kernel, ZZFeatureMap, d=4 and d=8) and a VQC
             (d=4), on exact statevector simulation.

All models are evaluated on the same fixed held-out test set (7,500 clients).
Decision costs are compared at the fixed Bayes threshold t* = c_fp/(c_fp+c_fn)
— the same non-tuned rule for every model. Wall-clock timings are recorded
for every phase, honestly, including quantum state embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.common import environment_info, save_results, timed
from quantum_credit import config
from quantum_credit.costs import CostMatrix, expected_cost
from quantum_credit.data import load_dataset, train_test_frames, xy
from quantum_credit.features import AngleReducer
from quantum_credit.metrics import evaluate_probabilities
from quantum_credit.models import SVC_C_GRID, subsample_models
from quantum_credit.quantum.featuremap import embed_statevectors
from quantum_credit.quantum.kernels import fidelity_kernel
from quantum_credit.quantum.vqc import StatevectorVQC
from quantum_credit.subsample import stratified_subsample_indices

CM = CostMatrix()


def eval_probs(y_te: np.ndarray, p_te: np.ndarray) -> dict:
    m = evaluate_probabilities(y_te, p_te)
    m["cost_at_bayes_threshold"] = expected_cost(
        y_te, (p_te >= CM.bayes_threshold).astype(int), CM
    )
    return m


def select_c_precomputed(gram: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Small inner 3-fold CV over the same C grid as the classical SVC."""
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c, best_auc = SVC_C_GRID[0], -np.inf
    for c in SVC_C_GRID:
        aucs = []
        for tr, va in cv.split(gram, y):
            svc = SVC(kernel="precomputed", C=c, class_weight="balanced", random_state=seed)
            svc.fit(gram[np.ix_(tr, tr)], y[tr])
            scores = svc.decision_function(gram[np.ix_(va, tr)])
            aucs.append(roc_auc_score(y[va], scores))
        if np.mean(aucs) > best_auc:
            best_auc, best_c = float(np.mean(aucs)), c
    return best_c


def run_quantum_kernel(
    ang_sub: np.ndarray, y_sub: np.ndarray, ang_te: np.ndarray, y_te: np.ndarray, seed: int
) -> tuple[dict, dict, np.ndarray]:
    timing: dict[str, float] = {}
    with timed(timing, "embed_train_s"):
        s_sub = embed_statevectors(ang_sub, reps=config.FEATURE_MAP_REPS)
    with timed(timing, "gram_s"):
        gram = fidelity_kernel(s_sub)
    with timed(timing, "model_select_fit_s"):
        c = select_c_precomputed(gram, y_sub, seed)
        svc = SVC(
            kernel="precomputed", C=c, class_weight="balanced",
            probability=True, random_state=seed,
        )
        svc.fit(gram, y_sub)
    with timed(timing, "embed_test_s"):
        s_te = embed_statevectors(ang_te, reps=config.FEATURE_MAP_REPS)
    with timed(timing, "predict_s"):
        k_te = fidelity_kernel(s_te, s_sub)
        p_te = svc.predict_proba(k_te)[:, 1]
    timing["total_s"] = sum(timing.values())
    metrics = eval_probs(y_te, p_te)
    return metrics, {"C": c, **timing}, gram


def run_vqc(
    ang_sub: np.ndarray, y_sub: np.ndarray, ang_te: np.ndarray, y_te: np.ndarray, seed: int
) -> tuple[dict, dict]:
    timing: dict[str, float] = {}
    with timed(timing, "embed_train_s"):
        s_sub = embed_statevectors(ang_sub, reps=config.FEATURE_MAP_REPS)
    clf = StatevectorVQC(n_qubits=config.N_QUBITS, seed=seed)
    with timed(timing, "fit_s"):
        clf.fit(s_sub, y_sub)
    with timed(timing, "embed_test_s"):
        s_te = embed_statevectors(ang_te, reps=config.FEATURE_MAP_REPS)
    with timed(timing, "predict_s"):
        p_te = clf.predict_proba(s_te)[:, 1]
    timing["total_s"] = sum(timing.values())
    extra = {
        "n_params": clf.n_params_,
        "n_loss_evals": len(clf.loss_curve_),
        "final_loss": clf.loss_curve_[-1] if clf.loss_curve_ else None,
        **timing,
    }
    return eval_probs(y_te, p_te), extra


def main() -> None:
    df, _ = load_dataset()
    train, test = train_test_frames(df)
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)

    runs: list[dict] = []
    kernel_sample: dict | None = None

    for size in config.SUBSAMPLE_SIZES:
        for seed in config.SUBSAMPLE_SEEDS:
            print(f"=== size={size} seed={seed}")
            idx = stratified_subsample_indices(y_tr, size, seed)
            X_sub, y_sub = X_tr.iloc[idx], y_tr[idx]

            # -------- classical on the exact same subsample --------
            for name, model in subsample_models(seed).items():
                timing: dict[str, float] = {}
                with timed(timing, "fit_s"):
                    model.fit(X_sub, y_sub)
                with timed(timing, "predict_s"):
                    p_te = model.predict_proba(X_te)[:, 1]
                timing["total_s"] = sum(timing.values())
                runs.append(
                    {
                        "size": size, "seed": seed, "model": name, "family": "classical",
                        "metrics": eval_probs(y_te, p_te), "detail": timing,
                    }
                )

            # -------- quantum: shared feature reduction per (size, seed) --------
            qk_variants = (
                (config.N_QUBITS, "qk_svc_d4"),
                (config.N_QUBITS_ROBUSTNESS, "qk_svc_d8"),
            )
            for d, tag in qk_variants:
                red = AngleReducer(d=d, method="mi", seed=seed)
                ang_sub = red.fit_transform(X_sub, y_sub)
                ang_te = red.transform(X_te)
                metrics, detail, gram = run_quantum_kernel(ang_sub, y_sub, ang_te, y_te, seed)
                detail["selected_features"] = red.selected
                runs.append(
                    {
                        "size": size, "seed": seed, "model": tag, "family": "quantum",
                        "metrics": metrics, "detail": detail,
                    }
                )
                if tag == "qk_svc_d4" and size == max(config.SUBSAMPLE_SIZES) and seed == 0:
                    order = np.argsort(y_sub[:100])
                    kernel_sample = {
                        "gram_100": gram[:100, :100][np.ix_(order, order)],
                        "labels_100": y_sub[:100][order],
                        "size": size, "seed": seed, "d": d,
                    }
                if tag == "qk_svc_d4":
                    m_vqc, x_vqc = run_vqc(ang_sub, y_sub, ang_te, y_te, seed)
                    x_vqc["selected_features"] = red.selected
                    runs.append(
                        {
                            "size": size, "seed": seed, "model": "vqc_d4", "family": "quantum",
                            "metrics": m_vqc, "detail": x_vqc,
                        }
                    )

    # -------- aggregate over subsample seeds --------
    aggregates: dict = {}
    model_names = sorted({r["model"] for r in runs})
    for size in config.SUBSAMPLE_SIZES:
        aggregates[str(size)] = {}
        for name in model_names:
            sel = [r for r in runs if r["size"] == size and r["model"] == name]
            if not sel:
                continue
            keys = ["auc", "pr_auc", "brier", "cost_at_bayes_threshold"]
            agg = {}
            for k in keys:
                vals = np.array([r["metrics"][k] for r in sel])
                agg[k] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                    "n": len(vals),
                }
            agg["total_time_s"] = {
                "mean": float(np.mean([r["detail"]["total_s"] for r in sel]))
            }
            aggregates[str(size)][name] = agg

    save_results(
        "matched_subsamples.json",
        {
            "config": {
                "sizes": list(config.SUBSAMPLE_SIZES),
                "seeds": list(config.SUBSAMPLE_SEEDS),
                "n_qubits": config.N_QUBITS,
                "n_qubits_robustness": config.N_QUBITS_ROBUSTNESS,
                "feature_map_reps": config.FEATURE_MAP_REPS,
                "bayes_threshold": CM.bayes_threshold,
                "n_test": len(y_te),
                "reduction": "mutual_information_top_d",
            },
            "environment": environment_info(),
            "runs": runs,
            "aggregates": aggregates,
            "kernel_sample": kernel_sample,
        },
    )


if __name__ == "__main__":
    main()
