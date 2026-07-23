"""Central configuration: seeds, paths, protocol and economic parameters.

Every experiment reads from this module, so a single place defines the
random seeds, the evaluation protocol and the stylised economics of the
cost-sensitive analysis. Nothing here is tuned on test data.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# --- dataset (UCI ML Repository id 350, Yeh & Lien 2009) ---------------------
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
RAW_XLS = DATA_DIR / "raw" / "default_of_credit_card_clients.xls"
EXPECTED_ROWS = 30_000
TARGET = "default"  # renamed from "default payment next month"

# --- protocol -----------------------------------------------------------------
SEED = 42  # global seed: train/test split, CV folds, model init
TEST_FRACTION = 0.25
N_FOLDS = 5
SUBSAMPLE_SIZES = (500, 1000, 2000)  # quantum-feasible training sizes
SUBSAMPLE_SEEDS = (0, 1, 2)  # independent stratified subsample draws

# --- quantum ------------------------------------------------------------------
N_QUBITS = 4  # primary reduced feature dimension (= qubits)
N_QUBITS_ROBUSTNESS = 8  # kernel-only robustness check
FEATURE_MAP_REPS = 2  # ZZFeatureMap repetitions (Havlíček et al. 2019 use 2)
VQC_ANSATZ_REPS = 3
VQC_MAXITER = 250
VQC_RESTARTS = 2

# --- stylised credit-card economics (see paper, Sec. "Cost-sensitive eval") ---
# Unit-exposure misclassification cost matrix. Approving a client who then
# defaults (false negative) is taken to be 5x as costly as rejecting a good
# client (false positive) — the classic credit-scoring cost asymmetry.
COST_FN = 5.0
COST_FP = 1.0
# NT$ profit model per approved client, as a fraction of the credit limit
# LIMIT_BAL: a good client yields PROFIT_MARGIN * limit; a defaulter loses
# LGD_FRACTION * limit. Stylised, clearly-flagged illustrative parameters.
PROFIT_MARGIN = 0.10
LGD_FRACTION = 0.60

THRESHOLD_GRID_SIZE = 501  # thresholds in [0, 1] for cost/profit curves
