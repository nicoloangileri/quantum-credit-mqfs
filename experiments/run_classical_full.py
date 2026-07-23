"""Part 1 — classical baselines on the full training set (22,500 clients).

Protocol (all seeds fixed in quantum_credit.config):
- one fixed stratified 75/25 train/test split;
- 5-fold stratified CV on the training set for fold-level error bars;
- out-of-fold (OOF) probabilities choose the operating thresholds, which
  are then applied unchanged to the test set (no test-set tuning);
- permutation feature importance on the held-out test set.

Every number lands in results/classical_full.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.common import environment_info, save_results, timed
from quantum_credit import config
from quantum_credit.costs import (
    CostMatrix,
    cost_curve,
    expected_cost,
    optimal_cost_threshold,
    optimal_profit_threshold,
    profit_curve,
)
from quantum_credit.data import load_dataset, train_test_frames, xy
from quantum_credit.metrics import calibration_bins, evaluate_probabilities, summarize_runs
from quantum_credit.models import full_data_models


def cv_fold_metrics(model, X, y, cv) -> list[dict[str, float]]:
    """Refit per fold and score the fold's validation part (AUC/PR-AUC/Brier)."""
    from sklearn.base import clone

    rows = []
    for tr, va in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        rows.append(evaluate_probabilities(y[va], p))
    return rows


def main() -> None:
    cm = CostMatrix()
    df, report = load_dataset()
    train, test = train_test_frames(df)
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)
    limit_te = X_te["LIMIT_BAL"].to_numpy(dtype=float)
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    out: dict = {
        "config": {
            "seed": config.SEED,
            "test_fraction": config.TEST_FRACTION,
            "n_folds": config.N_FOLDS,
            "cost_fn": cm.fn,
            "cost_fp": cm.fp,
            "bayes_threshold": cm.bayes_threshold,
            "profit_margin": config.PROFIT_MARGIN,
            "lgd_fraction": config.LGD_FRACTION,
        },
        "data": {
            "n_train": len(train),
            "n_test": len(test),
            "train_default_rate": float(y_tr.mean()),
            "test_default_rate": float(y_te.mean()),
            "education_recoded": report.education_recoded,
            "marriage_recoded": report.marriage_recoded,
        },
        "environment": environment_info(),
        "models": {},
    }

    for name, model in full_data_models(config.SEED).items():
        print(f"=== {name}")
        entry: dict = {"timing_s": {}}

        with timed(entry["timing_s"], "cv_folds"):
            folds = cv_fold_metrics(model, X_tr, y_tr, cv)
        entry["cv"] = summarize_runs(folds)
        entry["cv_folds"] = folds

        with timed(entry["timing_s"], "oof_predict"):
            p_oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        t_cost, oof_cost = optimal_cost_threshold(y_tr, p_oof, cm)
        limit_tr = X_tr["LIMIT_BAL"].to_numpy(dtype=float)
        t_profit, _ = optimal_profit_threshold(y_tr, p_oof, limit_tr)

        with timed(entry["timing_s"], "fit"):
            model.fit(X_tr, y_tr)
        with timed(entry["timing_s"], "predict_test"):
            p_te = model.predict_proba(X_te)[:, 1]

        entry["test"] = evaluate_probabilities(y_te, p_te)
        entry["calibration"] = calibration_bins(y_te, p_te)
        entry["thresholds"] = {
            "cost_optimal_oof": t_cost,
            "cost_at_optimum_oof": oof_cost,
            "profit_optimal_oof": t_profit,
            "bayes": cm.bayes_threshold,
        }
        entry["test_decisions"] = {
            "expected_cost_at_oof_threshold": expected_cost(y_te, (p_te >= t_cost).astype(int), cm),
            "expected_cost_at_bayes_threshold": expected_cost(
                y_te, (p_te >= cm.bayes_threshold).astype(int), cm
            ),
            "profit_at_oof_threshold_ntd": float(
                profit_curve(y_te, p_te, limit_te, thresholds=np.array([t_profit]))[1][0]
            ),
            "approval_rate_at_oof_threshold": float(np.mean(p_te < t_profit)),
        }
        t_grid, c_vals = cost_curve(y_te, p_te, cm)
        _, v_vals = profit_curve(y_te, p_te, limit_te)
        entry["curves"] = {
            "thresholds": t_grid,
            "expected_cost_test": c_vals,
            "profit_ntd_test": v_vals,
        }

        with timed(entry["timing_s"], "permutation_importance"):
            imp = permutation_importance(
                model, X_te, y_te, scoring="roc_auc", n_repeats=10,
                random_state=config.SEED, n_jobs=-1,
            )
        order = np.argsort(imp.importances_mean)[::-1]
        entry["permutation_importance"] = [
            {
                "feature": X_te.columns[i],
                "mean_auc_drop": float(imp.importances_mean[i]),
                "std": float(imp.importances_std[i]),
            }
            for i in order
        ]
        out["models"][name] = entry

    save_results("classical_full.json", out)


if __name__ == "__main__":
    main()
