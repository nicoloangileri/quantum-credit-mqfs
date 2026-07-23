"""Parser tests: schema validation, documented recodes, real-data known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantum_credit import config
from quantum_credit.data import (
    RAW_TARGET,
    DataValidationError,
    clean,
    load_dataset,
    read_raw,
    train_test_frames,
    validate_raw,
    xy,
)
from tests.conftest import make_raw_frame


class TestValidateRaw:
    def test_synthetic_frame_with_expected_rows_passes(self):
        df = make_raw_frame(n=config.EXPECTED_ROWS)
        validate_raw(df)  # must not raise

    def test_wrong_row_count_fails(self, raw_frame):
        with pytest.raises(DataValidationError, match="expected 30000 rows"):
            validate_raw(raw_frame)  # only 200 rows

    def test_missing_column_fails(self, raw_frame):
        with pytest.raises(DataValidationError, match="missing columns"):
            validate_raw(raw_frame.drop(columns=["PAY_3"]))

    def test_duplicate_ids_fail(self):
        df = make_raw_frame(n=config.EXPECTED_ROWS)
        df.loc[1, "ID"] = df.loc[0, "ID"]
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_raw(df)

    def test_non_binary_target_fails(self):
        df = make_raw_frame(n=config.EXPECTED_ROWS)
        df.loc[0, RAW_TARGET] = 2
        with pytest.raises(DataValidationError, match="target"):
            validate_raw(df)

    def test_bad_sex_code_fails(self):
        df = make_raw_frame(n=config.EXPECTED_ROWS)
        df.loc[0, "SEX"] = 7
        with pytest.raises(DataValidationError, match="SEX"):
            validate_raw(df)

    def test_negative_pay_amt_fails(self):
        df = make_raw_frame(n=config.EXPECTED_ROWS)
        df.loc[0, "PAY_AMT3"] = -1
        with pytest.raises(DataValidationError, match="PAY_AMT3"):
            validate_raw(df)

    def test_missing_values_fail(self):
        df = make_raw_frame(n=config.EXPECTED_ROWS)
        df.loc[5, "AGE"] = np.nan
        with pytest.raises(DataValidationError):
            validate_raw(df)


class TestClean:
    def test_target_renamed_and_id_dropped(self, raw_frame):
        out, _ = clean(raw_frame)
        assert config.TARGET in out.columns
        assert RAW_TARGET not in out.columns
        assert "ID" not in out.columns

    def test_education_recode_exact_count(self, raw_frame):
        expected = int(raw_frame["EDUCATION"].isin([0, 5, 6]).sum())
        out, report = clean(raw_frame)
        assert report.education_recoded == expected
        assert set(out["EDUCATION"].unique()) <= {1, 2, 3, 4}

    def test_marriage_recode_exact_count(self, raw_frame):
        expected = int((raw_frame["MARRIAGE"] == 0).sum())
        out, report = clean(raw_frame)
        assert report.marriage_recoded == expected
        assert set(out["MARRIAGE"].unique()) <= {1, 2, 3}

    def test_no_rows_dropped(self, raw_frame):
        out, report = clean(raw_frame)
        assert len(out) == len(raw_frame) == report.n_rows

    def test_input_frame_not_mutated(self, raw_frame):
        before = raw_frame.copy()
        clean(raw_frame)
        pd.testing.assert_frame_equal(raw_frame, before)


class TestSplit:
    def test_split_is_stratified_and_deterministic(self, raw_frame):
        df, _ = clean(raw_frame)
        a1, b1 = train_test_frames(df, test_fraction=0.25, seed=7)
        a2, b2 = train_test_frames(df, test_fraction=0.25, seed=7)
        pd.testing.assert_frame_equal(a1, a2)
        pd.testing.assert_frame_equal(b1, b2)
        rate = df[config.TARGET].mean()
        assert abs(b1[config.TARGET].mean() - rate) < 0.05
        assert len(a1) + len(b1) == len(df)

    def test_xy_shapes(self, raw_frame):
        df, _ = clean(raw_frame)
        X, y = xy(df)
        assert config.TARGET not in X.columns
        assert len(X) == len(y)


@pytest.mark.integration
class TestRealDataset:
    """Known-answer checks on the actual UCI file (skipped if not cached)."""

    @pytest.fixture(autouse=True)
    def _need_file(self, dataset_available):
        if not dataset_available:
            pytest.skip("UCI dataset not cached under data/")

    def test_known_aggregates(self):
        df, report = load_dataset(fetch_if_missing=False, write_manifest=False)
        assert report.n_rows == 30_000
        assert report.n_defaults == 6_636  # published default rate 22.12%
        assert abs(report.default_rate - 0.2212) < 1e-4

    def test_raw_passes_validation(self):
        validate_raw(read_raw())

    def test_documented_recode_counts(self):
        _, report = load_dataset(fetch_if_missing=False, write_manifest=False)
        assert report.education_recoded == 345  # 14 zeros + 280 fives + 51 sixes
        assert report.marriage_recoded == 54
