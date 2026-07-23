"""Shared fixtures: a small synthetic frame mimicking the raw UCI layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantum_credit import config
from quantum_credit.data import BILL_COLS, PAY_AMT_COLS, PAY_COLS, RAW_TARGET


def make_raw_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the exact raw schema (including quirky codes)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"ID": np.arange(1, n + 1)})
    df["LIMIT_BAL"] = rng.integers(10_000, 800_000, n).astype(float)
    df["SEX"] = rng.integers(1, 3, n)
    df["EDUCATION"] = rng.choice([0, 1, 2, 3, 4, 5, 6], n)
    df["MARRIAGE"] = rng.choice([0, 1, 2, 3], n)
    df["AGE"] = rng.integers(21, 80, n)
    for c in PAY_COLS:
        df[c] = rng.integers(-2, 9, n)
    for c in BILL_COLS:
        df[c] = rng.integers(-50_000, 500_000, n)
    for c in PAY_AMT_COLS:
        df[c] = rng.integers(0, 100_000, n)
    df[RAW_TARGET] = rng.binomial(1, 0.22, n)
    return df


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def dataset_available() -> bool:
    return config.RAW_XLS.exists()
