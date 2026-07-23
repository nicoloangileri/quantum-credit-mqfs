"""Classical model factories with explicit class-imbalance handling.

All models consume the cleaned DataFrame; encoding/scaling is inside each
pipeline (fitted on training folds only — no leakage):

- linear/kernel models: one-hot for SEX/EDUCATION/MARRIAGE + standardised
  numerics;
- tree ensembles: raw integer-coded features (trees are invariant to
  monotone recoding and split categoricals natively well enough here).

Imbalance (~22% defaults) is handled with balanced class weights, and the
logistic model's probabilities are re-calibrated afterwards because class
weighting deliberately distorts them (isotonic on the full training set,
sigmoid/Platt on small subsamples where isotonic would overfit).
"""

from __future__ import annotations

from typing import Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data import CATEGORICAL, FEATURES

NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

SVC_C_GRID = (0.5, 2.0, 8.0)


def linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), list(CATEGORICAL)),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def make_logistic(
    seed: int, *, calibration: Literal["isotonic", "sigmoid"] = "isotonic", cv: int = 5
) -> CalibratedClassifierCV:
    base = Pipeline(
        [
            ("pre", linear_preprocessor()),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000, class_weight="balanced", C=1.0, random_state=seed
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(base, method=calibration, cv=cv)


def make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=seed,
    )


def make_svc_rbf(seed: int, *, cv: int = 3) -> GridSearchCV:
    """RBF-kernel SVC — the direct classical twin of the quantum-kernel SVC.

    Same estimator class, same C grid and inner-CV protocol as the QK-SVC;
    only the kernel differs. Platt scaling (probability=True) supplies
    probabilities for Brier/cost analysis.
    """
    pipe = Pipeline(
        [
            ("pre", linear_preprocessor()),
            (
                "svc",
                SVC(kernel="rbf", gamma="scale", class_weight="balanced",
                    probability=True, random_state=seed),
            ),
        ]
    )
    return GridSearchCV(
        pipe,
        {"svc__C": list(SVC_C_GRID)},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )


def full_data_models(seed: int) -> dict[str, object]:
    """Models for the full-training-set study (Part 1)."""
    return {
        "logistic_calibrated": make_logistic(seed, calibration="isotonic", cv=5),
        "random_forest": make_random_forest(seed),
        "hist_gradient_boosting": make_hgb(seed),
    }


def subsample_models(seed: int) -> dict[str, object]:
    """Classical models re-trained on each quantum-sized subsample (fairness rule)."""
    return {
        "logistic_calibrated": make_logistic(seed, calibration="sigmoid", cv=3),
        "random_forest": make_random_forest(seed),
        "hist_gradient_boosting": make_hgb(seed),
        "svc_rbf": make_svc_rbf(seed),
    }
