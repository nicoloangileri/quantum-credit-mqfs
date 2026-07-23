"""Cached fetch and defensive parsing of the UCI credit-card default dataset.

Dataset: "Default of Credit Card Clients" (UCI id 350), 30,000 Taiwanese
credit-card clients observed in 2005; Yeh & Lien (2009). The raw file is an
old-style .xls with a quirk: row 0 holds pseudo-headers X1..X23, row 1 holds
the real column names, and the first column is a client ID.

Preprocessing decisions (all deliberate, all documented):

1. Header row 1 is used for names; the ID column is dropped after a
   uniqueness check.
2. The target column ``default payment next month`` is renamed ``default``.
3. EDUCATION has undocumented codes {0, 5, 6} (documented codes are
   1=graduate school, 2=university, 3=high school, 4=others). They are
   collapsed into 4 = "others". No rows are dropped.
4. MARRIAGE has an undocumented code 0 (documented: 1=married, 2=single,
   3=others). It is collapsed into 3 = "others". No rows are dropped.
5. PAY_0, PAY_2..PAY_6 (repayment status; note the dataset skips "PAY_1")
   take values in [-2, 8] and are kept as ordinal integers.
6. The dataset has no missing values; the parser verifies this instead of
   assuming it.
7. No scaling here: scaling/encoding is model-specific and lives inside
   sklearn pipelines fitted on training folds only (no leakage).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RAW_TARGET = "default payment next month"
CATEGORICAL = ("SEX", "EDUCATION", "MARRIAGE")
PAY_COLS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
BILL_COLS = tuple(f"BILL_AMT{i}" for i in range(1, 7))
PAY_AMT_COLS = tuple(f"PAY_AMT{i}" for i in range(1, 7))
FEATURES = ("LIMIT_BAL", *CATEGORICAL, "AGE", *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS)


class DataValidationError(ValueError):
    """Raised when the raw file does not look like the documented dataset."""


@dataclass(frozen=True)
class CleaningReport:
    """Counts of recoded cells, for the data manifest and the paper."""

    education_recoded: int
    marriage_recoded: int
    n_rows: int
    n_defaults: int

    @property
    def default_rate(self) -> float:
        return self.n_defaults / self.n_rows


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    url: str = config.UCI_URL,
    dest: Path = config.RAW_XLS,
    *,
    retries: int = 3,
    timeout: float = 60.0,
    force: bool = False,
) -> Path:
    """Download the raw .xls once and cache it under data/ (gitignored)."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quantum-credit-mqfs/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                blob = resp.read()
            if len(blob) < 1_000_000:  # the real file is ~5.5 MB
                raise DataValidationError(f"downloaded only {len(blob)} bytes from {url}")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(blob)
            tmp.rename(dest)
            return dest
        except Exception as err:  # noqa: BLE001 - report the last failure
            last_err = err
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not fetch dataset after {retries} attempts: {last_err}")


def read_raw(path: Path = config.RAW_XLS) -> pd.DataFrame:
    """Read the raw .xls (row 1 = real header) without any cleaning."""
    return pd.read_excel(path, header=1, engine="xlrd")


def validate_raw(df: pd.DataFrame) -> None:
    """Fail loudly if the frame does not match the documented schema."""
    problems: list[str] = []
    expected_cols = ["ID", *FEATURES, RAW_TARGET]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}; got {list(df.columns)}")
    if len(df) != config.EXPECTED_ROWS:
        problems.append(f"expected {config.EXPECTED_ROWS} rows, got {len(df)}")
    if df[expected_cols].isna().any().any():
        problems.append("unexpected missing values")
    if df["ID"].duplicated().any():
        problems.append("duplicate client IDs")
    target = df[RAW_TARGET]
    if not set(pd.unique(target)) <= {0, 1}:
        problems.append(f"target not binary: {sorted(pd.unique(target))[:10]}")
    checks = {
        "LIMIT_BAL": (df["LIMIT_BAL"] > 0, "LIMIT_BAL must be positive"),
        "SEX": (df["SEX"].isin([1, 2]), "SEX must be in {1, 2}"),
        "EDUCATION": (df["EDUCATION"].between(0, 6), "EDUCATION must be in 0..6"),
        "MARRIAGE": (df["MARRIAGE"].between(0, 3), "MARRIAGE must be in 0..3"),
        "AGE": (df["AGE"].between(18, 100), "AGE must be in 18..100"),
    }
    for col in PAY_COLS:
        checks[col] = (df[col].between(-2, 8), f"{col} must be in -2..8")
    for col in PAY_AMT_COLS:
        checks[col] = (df[col] >= 0, f"{col} must be non-negative")
    for _, (ok, msg) in checks.items():
        if not bool(ok.all()):
            problems.append(f"{msg} ({int((~ok).sum())} offending rows)")
    if problems:
        raise DataValidationError("; ".join(problems))


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the documented recodes; never drops rows."""
    out = df.drop(columns=["ID"]).rename(columns={RAW_TARGET: config.TARGET}).copy()
    edu_mask = out["EDUCATION"].isin([0, 5, 6])
    mar_mask = out["MARRIAGE"] == 0
    out.loc[edu_mask, "EDUCATION"] = 4
    out.loc[mar_mask, "MARRIAGE"] = 3
    out = out.astype({c: np.int64 for c in out.columns if c != "LIMIT_BAL"})
    report = CleaningReport(
        education_recoded=int(edu_mask.sum()),
        marriage_recoded=int(mar_mask.sum()),
        n_rows=len(out),
        n_defaults=int(out[config.TARGET].sum()),
    )
    return out, report


def load_dataset(
    *, fetch_if_missing: bool = True, write_manifest: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Fetch (cached), read, validate and clean the dataset."""
    if not config.RAW_XLS.exists():
        if not fetch_if_missing:
            raise FileNotFoundError(config.RAW_XLS)
        fetch()
    raw = read_raw()
    validate_raw(raw)
    cleaned, report = clean(raw)
    if write_manifest:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source_url": config.UCI_URL,
            "file": str(config.RAW_XLS.relative_to(config.REPO_ROOT)),
            "sha256": sha256_of(config.RAW_XLS),
            "bytes": config.RAW_XLS.stat().st_size,
            "cleaning": asdict(report),
            "default_rate": report.default_rate,
        }
        with open(config.RESULTS_DIR / "data_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    return cleaned, report


def train_test_frames(
    df: pd.DataFrame, *, test_fraction: float = config.TEST_FRACTION, seed: int = config.SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single fixed stratified split used by every experiment."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=test_fraction, stratify=df[config.TARGET], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=[config.TARGET]), df[config.TARGET].to_numpy()


def main() -> None:  # pragma: no cover - thin CLI
    _, report = load_dataset()
    print(f"rows={report.n_rows} defaults={report.n_defaults} rate={report.default_rate:.4f}")
    print(f"recoded EDUCATION={report.education_recoded} MARRIAGE={report.marriage_recoded}")


if __name__ == "__main__":
    main()
