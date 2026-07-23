# Data notes — every preprocessing decision, documented

**Dataset.** "Default of Credit Card Clients" — UCI Machine Learning
Repository id 350 (Yeh & Lien, 2009). 30,000 Taiwanese credit-card clients,
October 2005 snapshot; binary target = default on next month's payment.
Canonical source (cached by `python -m quantum_credit.data` into the
gitignored `data/` directory):

    https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

A SHA-256 of the cached file, its size, and all cleaning counts are recorded
in `results/data_manifest.json` at load time.

## Raw-file quirks handled by the parser (`src/quantum_credit/data.py`)

1. The .xls has two header rows: row 0 = pseudo-headers `X1..X23`,
   row 1 = real names. We read with `header=1`.
2. First column `ID` (1..30000): checked unique, then dropped.
3. The repayment-status columns are `PAY_0, PAY_2..PAY_6` — the source
   skips "PAY_1". We keep the original names to stay traceable to the
   UCI documentation.
4. Target column `default payment next month` is renamed `default`.

## Validation (fail-loud, before any cleaning)

Exactly 30,000 rows; no missing values; target strictly binary;
`LIMIT_BAL > 0`; `SEX ∈ {1,2}`; `EDUCATION ∈ 0..6`; `MARRIAGE ∈ 0..3`;
`AGE ∈ 18..100`; `PAY_* ∈ -2..8`; `PAY_AMT* ≥ 0`. Any violation raises
`DataValidationError` naming the offending columns and row counts.
Known-answer integration tests additionally pin the published aggregates:
6,636 defaults (22.12%), 345 recoded EDUCATION cells, 54 recoded MARRIAGE
cells.

## Cleaning decisions (deliberately minimal; no rows dropped)

| Decision | Rationale |
|---|---|
| `EDUCATION {0,5,6} -> 4` ("others") | Codes 0/5/6 are undocumented in the UCI description; collapsing to "others" (345 cells) avoids inventing categories. |
| `MARRIAGE {0} -> 3` ("others") | Code 0 undocumented (54 cells); same reasoning. |
| `PAY_*` kept as ordinal integers | They encode months of payment delay; order is meaningful. |
| No missing-value imputation | The file has no missing values; the parser *verifies* this rather than assuming it. |
| No outlier removal | Bill/payment amounts are heavy-tailed but legitimate; tree models are robust, linear models standardise. |

## Model-specific encoding (inside sklearn pipelines, fitted on training folds only)

- **Linear / kernel models** (logistic, SVC): one-hot `SEX`, `EDUCATION`,
  `MARRIAGE` (drop-first) + `StandardScaler` on numeric columns.
- **Tree ensembles** (random forest, HistGB): raw integer-coded features.
- **Quantum models**: mutual-information top-d selection (d = 4 / 8),
  then min-max scaling to angles in [0, π], both fitted on each training
  subsample only; test values clipped to the training range. PCA is
  implemented as an alternative (`AngleReducer(method="pca")`).

## Splits, seeds, subsamples

Single stratified 75/25 train/test split (seed 42), 5-fold stratified CV
inside the training set, stratified subsamples of 500/1000/2000 with seeds
0/1/2 for the matched classical-vs-quantum study. All in
`src/quantum_credit/config.py`.
