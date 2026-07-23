"""Subsampling tests: stratification, determinism, bounds."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_credit.subsample import stratified_subsample_indices


class TestStratifiedSubsample:
    def test_size_and_uniqueness(self):
        y = np.array([0] * 780 + [1] * 220)
        idx = stratified_subsample_indices(y, 100, seed=0)
        assert len(idx) == 100
        assert len(np.unique(idx)) == 100

    def test_class_ratio_preserved(self):
        y = np.array([0] * 780 + [1] * 220)
        idx = stratified_subsample_indices(y, 200, seed=1)
        assert abs(y[idx].mean() - 0.22) < 0.02

    def test_deterministic_in_seed(self):
        y = np.array([0, 1] * 500)
        a = stratified_subsample_indices(y, 100, seed=7)
        b = stratified_subsample_indices(y, 100, seed=7)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        y = np.array([0, 1] * 500)
        a = stratified_subsample_indices(y, 100, seed=0)
        b = stratified_subsample_indices(y, 100, seed=1)
        assert not np.array_equal(a, b)

    def test_out_of_bounds_n_rejected(self):
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError):
            stratified_subsample_indices(y, 4, seed=0)
        with pytest.raises(ValueError):
            stratified_subsample_indices(y, 0, seed=0)
