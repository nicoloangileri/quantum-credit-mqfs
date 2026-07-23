"""Feature-reduction tests: selection, angle scaling, leakage guards."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_credit.data import clean, xy
from quantum_credit.features import AngleReducer
from tests.conftest import make_raw_frame


@pytest.fixture
def small_xy():
    df, _ = clean(make_raw_frame(n=300, seed=3))
    return xy(df)


class TestMISelection:
    def test_selects_d_real_columns(self, small_xy):
        X, y = small_xy
        red = AngleReducer(d=4, method="mi", seed=0).fit(X, y)
        assert len(red.selected) == 4
        assert set(red.selected) <= set(X.columns)

    def test_deterministic_given_seed(self, small_xy):
        X, y = small_xy
        a = AngleReducer(d=4, method="mi", seed=0).fit(X, y).selected
        b = AngleReducer(d=4, method="mi", seed=0).fit(X, y).selected
        assert a == b

    def test_informative_feature_is_picked_up(self):
        df, _ = clean(make_raw_frame(n=400, seed=5))
        X, y = xy(df)
        # plant a feature that almost determines the label
        X = X.copy()
        X["LIMIT_BAL"] = y * 100_000 + np.arange(len(y))
        red = AngleReducer(d=4, method="mi", seed=0).fit(X, y)
        assert "LIMIT_BAL" in red.selected


class TestAngleScaling:
    def test_train_angles_span_zero_to_pi(self, small_xy):
        X, y = small_xy
        Z = AngleReducer(d=4, method="mi", seed=0).fit_transform(X, y)
        assert Z.shape == (len(X), 4)
        assert Z.min() == pytest.approx(0.0)
        assert Z.max() == pytest.approx(np.pi)

    def test_test_values_clipped_to_training_range(self, small_xy):
        X, y = small_xy
        red = AngleReducer(d=4, method="mi", seed=0).fit(X.iloc[:150], y[:150])
        Z = red.transform(X.iloc[150:])
        assert Z.min() >= 0.0 and Z.max() <= np.pi + 1e-12

    def test_transform_before_fit_raises(self, small_xy):
        X, _ = small_xy
        with pytest.raises(RuntimeError):
            AngleReducer(d=4).transform(X)


class TestPCA:
    def test_pca_shapes_and_range(self, small_xy):
        X, y = small_xy
        red = AngleReducer(d=6, method="pca", seed=0)
        Z = red.fit_transform(X, y)
        assert Z.shape == (len(X), 6)
        assert Z.min() >= 0.0 and Z.max() <= np.pi + 1e-12
        assert red.selected == [f"PC{i}" for i in range(1, 7)]

    def test_unknown_method_rejected(self, small_xy):
        X, y = small_xy
        with pytest.raises(ValueError):
            AngleReducer(d=4, method="banana").fit(X, y)
