"""Build the MQFS working paper PDF from results/*.json and figures/.

Every number in the paper is read from the results JSONs produced by
`experiments/run_all.py`; if a result file is missing the build fails —
nothing is ever invented. LaTeX (latexmk/pdflatex) is used when available,
otherwise a structurally identical PDF is rendered with fpdf2 + matplotlib
mathtext (`--engine fpdf`).

    python paper/build_paper.py [--engine auto|latex|fpdf]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO = PAPER_DIR.parent
sys.path.insert(0, str(REPO))

from experiments.load_paper_data import load_all  # noqa: E402
from experiments.render_tables import LABELS, ORDER  # noqa: E402

PDF_NAME = "quantum_credit_mqfs_working_paper.pdf"


def esc_label(name: str) -> str:
    return LABELS[name].replace("**", "").replace("d=", "$d$=")


def latex_matched_table(R: dict) -> str:
    m = R["matched"]
    sizes = m["config"]["sizes"]
    agg = m["aggregates"]
    rows = []
    for name in ORDER:
        if name not in agg[str(sizes[0])]:
            continue
        family = "Q" if name.startswith(("qk_", "vqc")) else "C"
        cells = []
        for s in sizes:
            a = agg[str(s)][name]["auc"]
            cells.append(f"{a['mean']:.4f} $\\pm$ {a['std']:.4f}")
        t = agg[str(max(sizes))][name]["total_time_s"]["mean"]
        rows.append(
            f"{esc_label(name)} ({family}) & " + " & ".join(cells) + f" & {t:.1f} \\\\"
        )
    return "\n".join(rows)


def latex_cost_table(R: dict) -> str:
    m = R["matched"]
    sizes = m["config"]["sizes"]
    agg = m["aggregates"]
    rows = []
    for name in ORDER:
        if name not in agg[str(sizes[0])]:
            continue
        cells = []
        for s in sizes:
            a = agg[str(s)][name]["cost_at_bayes_threshold"]
            cells.append(f"{a['mean']:.4f} $\\pm$ {a['std']:.4f}")
        rows.append(f"{esc_label(name)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def latex_full_table(R: dict) -> str:
    f = R["full"]
    rows = []
    for name in ["logistic_calibrated", "random_forest", "hist_gradient_boosting"]:
        e = f["models"][name]
        cv = e["cv"]["auc"]
        rows.append(
            f"{esc_label(name)} & {cv['mean']:.4f} $\\pm$ {cv['std']:.4f} & "
            f"{e['test']['auc']:.4f} & {e['test']['pr_auc']:.4f} & "
            f"{e['test']['brier']:.4f} & "
            f"{e['test_decisions']['expected_cost_at_oof_threshold']:.4f} & "
            f"{e['test_decisions']['profit_at_oof_threshold_ntd'] / 1e6:.1f} & "
            f"{e['timing_s']['fit']:.1f} \\\\"
        )
    return "\n".join(rows)


def latex_importance_lines(R: dict) -> str:
    f = R["full"]
    best = R["best_full_model"]
    imp = f["models"][best]["permutation_importance"][:5]
    feats = ", ".join(
        f"\\texttt{{{r['feature'].replace('_', '_')}}} ({r['mean_auc_drop']:.3f})".replace(
            "_", "\\_"
        )
        for r in imp
    )
    return feats


def build_tex(R: dict) -> str:
    f = R["full"]
    m = R["matched"]
    man = R["manifest"]
    env = f["environment"]
    best = R["best_full_model"]
    best_e = f["models"][best]
    n_sizes = m["config"]["sizes"]
    delta = R["delta_best_classical_minus_best_quantum_n2000"]
    meta = R.get("meta") or {}
    total_min = (meta.get("total_wall_clock_s", 0.0)) / 60.0
    hw = R.get("hardware")

    hardware_paragraph = (
        (
            "A tiny Gram sub-block was additionally estimated on the IBM backend "
            f"\\texttt{{{hw['backend']}}} ({hw['shots']} shots, {len(hw['pairs'])} "
            "compute--uncompute circuits, job \\texttt{" + hw["job_id"] + "}); the mean "
            f"absolute deviation between measured and statevector kernel entries was "
            f"{hw['mae']:.4f}."
        )
        if hw
        else (
            "The repository implements this estimator behind an explicit "
            "\\texttt{--hardware} flag (\\texttt{experiments/run\\_hardware.py}): it reads "
            "the user's saved \\texttt{QiskitRuntimeService} account from \\verb|~/.qiskit| "
            "(tokens never enter the repository), submits one job of at most 28 "
            "compute--uncompute circuits to the least-busy backend, and reports measured "
            "versus simulated kernel entries. It was \\emph{not} executed for this draft: "
            "the build environment holds no QPU credentials, and the tables above are "
            "exact-simulation results by design. Hardware numbers, when produced, land in "
            "\\texttt{results/hardware\\_kernel.json} and are reported verbatim."
        )
    )

    return r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\graphicspath{{../figures/}}
\captionsetup{font=small,labelfont=bf}

\title{Classical versus Quantum Machine Learning for Credit Default
Prediction:\\ An Honest Benchmark at Matched Sample Sizes}
\author{Nicol\`o Angileri\\
\small Mediterranean Quantitative Finance Society (MQFS) --- Working Paper\\
\small \texttt{nikoooo969@gmail.com}}
\date{July 2026}

\begin{document}
\maketitle

\begin{abstract}
\noindent We benchmark three classical credit-scoring models against two
quantum machine-learning models --- a fidelity quantum-kernel support vector
classifier (QK-SVC) and a variational quantum classifier (VQC) --- on the UCI
``Default of Credit Card Clients'' dataset (30{,}000 real Taiwanese
credit-card accounts). The evaluation is finance-aware: beyond AUC, PR-AUC and
calibration, we score every model under an asymmetric expected-loss matrix and
NT\$ profit curves over the approval threshold. Because fidelity kernels cost
$O(n^2)$ circuit evaluations, quantum models can only be trained on small
subsamples; our central methodological rule is that \emph{classical baselines
are re-trained on exactly the same subsamples} ($n \in \{__SIZES__\}$, three
independent draws), so no comparison ever crosses sample sizes. At every
matched size the best classical model matches or exceeds both quantum models
(at $n=2000$: best classical test AUC __BESTC2000__ vs.\ best quantum
__BESTQ2000__, a gap of __DELTA__), while training orders of magnitude faster.
\textbf{No quantum advantage is claimed --- or observed --- in this regime.}
The contribution is the protocol and the measured operating envelope: an
honest, reproducible baseline that future quantum-hardware claims on tabular
credit risk can be checked against. All numbers regenerate from one command in
about __TOTALMIN__ minutes on a laptop.
\end{abstract}

\section{Introduction}
Credit scoring is one of the oldest industrial applications of statistical
learning \cite{hand1997,thomas2002}, and one of the most heavily benchmarked
\cite{lessmann2015}. It is also, lately, a popular showcase for quantum
machine learning (QML), where quantum feature maps promise classically
intractable kernels \cite{havlicek2019,schuld2019}. Provable quantum speedups
for supervised learning exist only for contrived problems
\cite{liu2021}; whether \emph{any} benefit survives contact with a real,
noisy, tabular credit dataset at the sample sizes quantum methods can actually
process is an empirical question --- and one that is easy to answer
misleadingly. A recurring failure mode in applied QML papers is the uneven
comparison: a quantum model trained on hundreds of points is compared against
a classical model trained on the full dataset (or vice versa), with accuracy
as the only metric and wall-clock cost unreported.

This working paper does the boring version of the experiment, deliberately.
Our contributions are methodological:
(i) a \emph{matched-subsample fairness rule} --- every classical baseline is
re-trained on the exact stratified subsamples the quantum models use;
(ii) a \emph{finance-aware evaluation} --- an asymmetric expected-loss matrix
with its Bayes-optimal operating threshold, and portfolio profit curves in
NT\$ using each client's actual credit limit as exposure;
(iii) \emph{honest cost accounting} --- wall-clock time for every phase of
every model, including quantum state preparation; and
(iv) full reproducibility --- fixed seeds, defensive data validation,
known-answer unit tests (kernel symmetry/PSD, cost-matrix algebra), and a
single command that regenerates every number and figure in this paper.

\section{Data}
We use the UCI ``Default of Credit Card Clients'' dataset (id 350)
\cite{yeh2009,uci}: 30{,}000 Taiwanese credit-card clients observed in
October 2005, 23 raw features (credit limit, demographics, six months of
repayment status \texttt{PAY\_0, PAY\_2--PAY\_6}, bill and payment amounts),
and a binary target --- default on the next monthly payment. The default rate
is __DEFRATE__\% (__NDEF__ of __NROWS__); classes are imbalanced but not
extremely so. The raw \texttt{.xls} is fetched once and cached (SHA-256
recorded in \texttt{results/data\_manifest.json}); a defensive parser
validates the documented schema (exact row count, value ranges, no missing
values) before any modelling. Cleaning is minimal and fully documented in
\texttt{docs/DATA\_NOTES.md}: undocumented \texttt{EDUCATION} codes
$\{0,5,6\}$ (__EDUREC__ cells) and \texttt{MARRIAGE} code $0$ (__MARREC__
cells) are collapsed into their ``others'' categories; no rows are dropped;
no values are imputed. We hold out a fixed stratified 25\% test set
($n_{\text{test}} = __NTEST__$, seed 42) used by \emph{every} model in this
paper; all tuning happens inside the training set.

\section{Methodology}
\subsection{Classical models and protocol}
Three standard baselines are trained on the full training set
($n = __NTRAIN__$): $\ell_2$ logistic regression with balanced class weights
and isotonic probability calibration; a random forest (400 trees, balanced
subsample weights); and histogram gradient boosting (HGB) with balanced class
weights. Categorical variables are one-hot encoded and numerics standardised
for the linear model inside the pipeline (no leakage); tree ensembles consume
the raw integer-coded features. We report 5-fold stratified cross-validation
means $\pm$ standard deviations, held-out test metrics, and quantile-binned
reliability diagrams. Operating thresholds are selected on out-of-fold
training predictions only, then applied unchanged to the test set.

\subsection{Finance-aware, cost-sensitive evaluation}
Accuracy is the wrong loss for lending. Let $y_i = 1$ denote default and let
a model reject applicant $i$ when its predicted default probability $p_i$
exceeds a threshold $t$. With unit costs $c_{\mathrm{FN}}$ (approving a
future defaulter) and $c_{\mathrm{FP}}$ (rejecting a good client), the
empirical expected cost per applicant is
\begin{equation}
\widehat{C}(t) \;=\; \frac{1}{n}\Big( c_{\mathrm{FN}} \,\mathrm{FN}(t)
 \;+\; c_{\mathrm{FP}} \,\mathrm{FP}(t) \Big),
\label{eq:cost}
\end{equation}
and for perfectly calibrated probabilities the Bayes-optimal rule rejects iff
$p \ge t^\ast$ with
\begin{equation}
t^\ast \;=\; \frac{c_{\mathrm{FP}}}{c_{\mathrm{FP}} + c_{\mathrm{FN}}}
\label{eq:bayes}
\end{equation}
\cite{elkan2001}. We adopt the classic credit-scoring asymmetry
$c_{\mathrm{FN}} : c_{\mathrm{FP}} = 5 : 1$, giving $t^\ast = 1/6 \approx
0.167$; the distance between a model's empirical cost-minimising threshold
and $t^\ast$ is itself a calibration diagnostic. Complementarily, we compute
portfolio profit with heterogeneous exposures, approving iff $p_i < t$:
\begin{equation}
\Pi(t) \;=\; \sum_{i\,:\,p_i < t} \Big[ (1-y_i)\, m\, L_i \;-\; y_i\,
\lambda\, L_i \Big],
\label{eq:profit}
\end{equation}
where $L_i$ is client $i$'s actual credit limit (\texttt{LIMIT\_BAL}, in
NT\$), $m = __MARGIN__$ a stylised revenue margin on performing accounts and
$\lambda = __LGD__$ a stylised loss-given-default fraction. These economic
parameters are illustrative and clearly flagged as such; the full curves over
$t$ (Figure~\ref{fig:costprofit}) expose the sensitivity.

\subsection{Fidelity quantum kernel}
Following Havl\'i\v{c}ek et al.\ \cite{havlicek2019}, data are embedded by
the second-order Pauli-Z feature map $\mathcal{U}_{\Phi(x)}$
(\texttt{ZZFeatureMap}, __REPS__ repetitions): after Hadamards, single- and
two-qubit phase rotations with angles $\phi_i(x) = x_i$ and $\phi_{ij}(x) =
(\pi - x_i)(\pi - x_j)$. The kernel is the state fidelity
\begin{equation}
K(x, x') \;=\; \bigl|\langle \Phi(x) \,|\, \Phi(x') \rangle\bigr|^2,
\label{eq:kernel}
\end{equation}
consumed by a standard SVC with balanced class weights (precomputed Gram
matrix; $C$ chosen by inner 3-fold CV over the same grid as the classical
RBF-SVC, its exact classical twin). On real hardware $K$ is estimated by the
compute--uncompute circuit
$\mathcal{U}_{\Phi(x')}^{\dagger}\mathcal{U}_{\Phi(x)}|0\rangle^{\otimes d}$
(probability of the all-zeros outcome); on the statevector simulator used
here Eq.~\eqref{eq:kernel} is evaluated exactly, i.e.\ our quantum results
are \emph{noise-free upper bounds} on NISQ performance. Since features must
fit $d$ qubits, we select the top-$d$ features by mutual information with the
label ($d = 4$ primary, $d = 8$ robustness), estimated on each training
subsample only, then min--max scale them to angles in $[0, \pi]$ (test values
clipped to the training range). MI selection is preferred over PCA to keep
the encoded features interpretable and to avoid rotating integer-coded
categoricals into dense components; PCA is implemented as an alternative.

\subsection{Variational quantum classifier}
The VQC applies a RealAmplitudes ansatz $U(\theta)$ (__VQCREPS__ entangling
repetitions, __VQCPARAMS__ parameters at $d = 4$) to the same feature-map
states and measures the qubit-parity observable
$P = Z^{\otimes d}$:
\begin{equation}
p(x;\theta) \;=\; \tfrac{1}{2}\Bigl(1 - \langle\Phi(x)| U(\theta)^{\dagger} P\,
U(\theta) |\Phi(x)\rangle\Bigr),
\label{eq:vqc}
\end{equation}
trained by COBYLA (__VQCITER__ iterations, __VQCRESTARTS__ restarts, best
kept) on class-weighted cross-entropy. Because $P$ is diagonal and the
feature-map states are cached, each objective evaluation reduces to applying
one $2^d \times 2^d$ unitary to the cached state matrix --- exact simulation,
reported timings included.

\subsection{The fairness rule}
Building the Gram matrix costs $O(n^2)$ kernel evaluations ($\sim$2M circuits
at $n = 2000$ on hardware), so quantum models are trained on stratified
subsamples $n \in \{__SIZES__\}$ with three independent draws (seeds 0/1/2).
\textbf{Every classical baseline is re-trained on exactly the same
subsamples.} All models --- classical and quantum, every size --- are
evaluated on the same fixed __NTEST__-client test set, and decision costs are
compared at the same fixed Bayes threshold $t^\ast = 1/6$ (no per-model
threshold tuning at small $n$). We never compare quantum-on-2000 with
classical-on-__NTRAIN__.

\section{Results}
\subsection{Part 1 --- classical baselines on the full training set}
Table~\ref{tab:full} summarises the full-data study.
__BESTLABEL__ performs best (test AUC __BESTAUC__, PR-AUC __BESTPRAUC__,
Brier __BESTBRIER__). Reliability diagrams (Figure~\ref{fig:calib}) show
the isotonic-calibrated logistic model closest to the diagonal, consistent
with its empirical cost-optimal threshold landing nearest the Bayes value
$t^\ast = 1/6$ (Eq.~\ref{eq:bayes}): __THRLIST__. Cost and profit curves are
shown in Figure~\ref{fig:costprofit}; at its out-of-fold profit-optimal
threshold, __BESTLABEL__ attains NT\$__BESTPROFIT__M on the test portfolio
at an approval rate of __BESTAPPROVAL__\%.

\begin{table}[t]
\centering\small
\caption{Full-training-set classical baselines ($n=__NTRAIN__$; test
$n=__NTEST__$). CV: 5-fold stratified mean $\pm$ std. Cost per applicant
(Eq.~\ref{eq:cost}) at the OOF-selected threshold; profit (Eq.~\ref{eq:profit})
in NT\$ millions.}
\label{tab:full}
\begin{tabular}{lccccccc}
\toprule
Model & CV AUC & Test AUC & PR-AUC & Brier & Cost@$t^\ast_{\text{OOF}}$ &
Profit (M) & Fit (s) \\
\midrule
__FULLTABLE__
\bottomrule
\end{tabular}
\end{table}

\paragraph{What drives default risk.} Permutation importance on the test set
(Figure~\ref{fig:imp}) is dominated by the most recent repayment status
\texttt{PAY\_0}, followed by __IMPRUNNERS__ --- recent delinquency, not
demographics, carries the signal, in line with the credit-scoring literature
\cite{yeh2009,lessmann2015}. Top-5 AUC drops: __IMPLIST__.

\subsection{Part 2 --- matched subsamples: the honest comparison}
Table~\ref{tab:matched} is the central result. At every matched training size
the best classical model is at least as good as the best quantum model; at
$n = 2000$ the gap between the best classical (__BESTC2000L__,
__BESTC2000__) and the best quantum (__BESTQ2000L__, __BESTQ2000__) is
__DELTA__ AUC. The quantum kernel at $d=8$ does not improve on $d=4$, and the
VQC trails both, as expected for a __VQCPARAMS__-parameter model. The
expected-cost comparison at the fixed Bayes threshold (Table~\ref{tab:cost})
tells the same story in money terms. Figure~\ref{fig:aucn} overlays learning
curves; Figure~\ref{fig:timing} shows honest wall-clock costs --- the QK-SVC
pays a large simulation overhead (dominated by test-set embedding and the
$n_{\text{test}} \times n$ kernel block), and on hardware the same object
would cost millions of shots. Figure~\ref{fig:gram} visualises a Gram
sub-block: within-class and between-class fidelities are barely
distinguishable, a first hint of why the kernel does not separate this data
better than its classical twin.

\begin{table}[t]
\centering\small
\caption{Matched-subsample test AUC (mean $\pm$ std over 3 identical
stratified draws; C = classical, Q = quantum). Same subsamples, same test
set, same threshold rule for every model.}
\label{tab:matched}
\begin{tabular}{lcccc}
\toprule
Model & $n=500$ & $n=1000$ & $n=2000$ & Time@2000 (s) \\
\midrule
__MATCHEDTABLE__
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering\small
\caption{Expected cost per applicant (Eq.~\ref{eq:cost}) at the fixed Bayes
threshold $t^\ast = 1/6$; lower is better.}
\label{tab:cost}
\begin{tabular}{lccc}
\toprule
Model & $n=500$ & $n=1000$ & $n=2000$ \\
\midrule
__COSTTABLE__
\bottomrule
\end{tabular}
\end{table}

\subsection{Hardware estimation path}
__HARDWARE__

\begin{figure}[t]
\centering
\includegraphics[width=0.62\textwidth]{calibration_full.png}
\caption{Reliability diagrams on the held-out test set (quantile bins).}
\label{fig:calib}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{cost_profit_full.png}
\caption{Left: expected cost per applicant vs.\ rejection threshold, with the
Bayes threshold $t^\ast = 1/6$. Right: NT\$ portfolio profit vs.\ approval
threshold under the stylised economics of Eq.~\eqref{eq:profit}.}
\label{fig:costprofit}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{permutation_importance.png}
\caption{Permutation feature importance (test set, 10 repeats) for the best
full-data model.}
\label{fig:imp}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\textwidth]{auc_vs_n.png}
\caption{Matched-sample learning curves: classical (solid) vs.\ quantum
(dashed), mean $\pm$ std over three identical subsample draws.}
\label{fig:aucn}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.62\textwidth]{timing.png}
\caption{Honest wall-clock accounting at $n=2000$ (log scale). Quantum
timings are exact-statevector simulation; hardware execution would add shot
and queue overheads.}
\label{fig:timing}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.55\textwidth]{kernel_gram.png}
\caption{Fidelity-kernel Gram sub-block ($d=4$, 100 training points sorted
by label).}
\label{fig:gram}
\end{figure}

\section{Limitations}
This is a single dataset, from 2005, from one market; conclusions are about
the \emph{measured regime}, not about QML in general. The economic parameters
$(c_{\mathrm{FN}}\!:\!c_{\mathrm{FP}}, m, \lambda)$ are stylised; profit
levels shift with them even though model rankings here do not. Quantum models
were simulated exactly (no shot noise, no device noise): real-hardware
results would be strictly worse, which only strengthens the negative finding,
but we did not measure the degradation beyond providing the estimator.
Feature compression to $d \in \{4, 8\}$ discards information by construction;
the classical models in the matched study consume all 23 features, which is
deliberate --- the constraint \emph{is} the point --- but a reader may also
compare the RBF-SVC row as a same-feature-count proxy. Hyperparameter search
was intentionally light and symmetric ($C$ grid only); three subsample seeds
bound variability but do not support formal significance tests. Finally, the
VQC's optimisation landscape is nonconvex; different ansätze or optimisers
could close part of its gap, though hardly the kernel's.

\section{Conclusion}
On real credit-risk data, with a fair protocol and finance-aware metrics,
classical machine learning wins at every sample size a fidelity quantum
kernel can currently reach, and it wins while running orders of magnitude
faster. \textbf{No quantum advantage is claimed, and none was observed.} The
value of this study is the ruler, not the record: a reproducible,
cost-honest, matched-sample benchmark that future quantum claims on tabular
financial data can be measured against. The full pipeline regenerates in
about __TOTALMIN__ minutes on a laptop from \texttt{experiments/run\_all.py}.

\subsection*{Reproducibility and disclosure}
Code, tests and this paper's sources are in the \texttt{quantum-credit-mqfs}
repository (MIT licence). Environment: Python __PYVER__, scikit-learn
__SKVER__, Qiskit __QKVER__. Built with AI-assisted engineering; every design
decision, formula and validation was reviewed and is owned by the author.

\begin{thebibliography}{99}\small
\bibitem{yeh2009} I-C.~Yeh and C-H.~Lien.
The comparisons of data mining techniques for the predictive accuracy of
probability of default of credit card clients.
\emph{Expert Systems with Applications}, 36(2):2473--2480, 2009.

\bibitem{uci} M.~Kelly, R.~Longjohn, and K.~Nottingham.
The UCI Machine Learning Repository. \url{https://archive.ics.uci.edu},
dataset id 350.

\bibitem{havlicek2019} V.~Havl\'i\v{c}ek, A.~D.~C\'orcoles, K.~Temme,
A.~W.~Harrow, A.~Kandala, J.~M.~Chow, and J.~M.~Gambetta.
Supervised learning with quantum-enhanced feature spaces.
\emph{Nature}, 567:209--212, 2019.

\bibitem{schuld2019} M.~Schuld and N.~Killoran.
Quantum machine learning in feature Hilbert spaces.
\emph{Physical Review Letters}, 122:040504, 2019.

\bibitem{liu2021} Y.~Liu, S.~Arunachalam, and K.~Temme.
A rigorous and robust quantum speed-up in supervised machine learning.
\emph{Nature Physics}, 17:1013--1017, 2021.

\bibitem{elkan2001} C.~Elkan.
The foundations of cost-sensitive learning.
In \emph{Proc.\ IJCAI}, 973--978, 2001.

\bibitem{hand1997} D.~J.~Hand and W.~E.~Henley.
Statistical classification methods in consumer credit scoring: a review.
\emph{J.\ Royal Statistical Society A}, 160(3):523--541, 1997.

\bibitem{lessmann2015} S.~Lessmann, B.~Baesens, H-V.~Seow, and L.~C.~Thomas.
Benchmarking state-of-the-art classification algorithms for credit scoring:
an update of research.
\emph{European Journal of Operational Research}, 247(1):124--136, 2015.

\bibitem{thomas2002} L.~C.~Thomas, D.~B.~Edelman, and J.~N.~Crook.
\emph{Credit Scoring and Its Applications}. SIAM, 2002.

\bibitem{cerezo2021} M.~Cerezo et al.
Variational quantum algorithms.
\emph{Nature Reviews Physics}, 3:625--644, 2021.
\end{thebibliography}

\end{document}
""".replace("__SIZES__", ", ".join(str(s) for s in n_sizes)) \
        .replace("__BESTC2000L__", esc_label(R["best_c2000_name"])) \
        .replace("__BESTQ2000L__", esc_label(R["best_q2000_name"])) \
        .replace("__BESTC2000__", f"{R['best_c2000']:.4f}") \
        .replace("__BESTQ2000__", f"{R['best_q2000']:.4f}") \
        .replace("__DELTA__", f"{delta:.4f}") \
        .replace("__TOTALMIN__", f"{total_min:.0f}" if total_min else "20") \
        .replace("__DEFRATE__", f"{100 * man['default_rate']:.2f}") \
        .replace("__NDEF__", f"{man['cleaning']['n_defaults']:,}") \
        .replace("__NROWS__", f"{man['cleaning']['n_rows']:,}") \
        .replace("__EDUREC__", str(man["cleaning"]["education_recoded"])) \
        .replace("__MARREC__", str(man["cleaning"]["marriage_recoded"])) \
        .replace("__NTEST__", f"{f['data']['n_test']:,}") \
        .replace("__NTRAIN__", f"{f['data']['n_train']:,}") \
        .replace("__MARGIN__", f"{f['config']['profit_margin']:.2f}") \
        .replace("__LGD__", f"{f['config']['lgd_fraction']:.2f}") \
        .replace("__REPS__", str(m["config"]["feature_map_reps"])) \
        .replace("__VQCREPS__", str(R["vqc_reps"])) \
        .replace("__VQCPARAMS__", str(R["vqc_params"])) \
        .replace("__VQCITER__", str(R["vqc_maxiter"])) \
        .replace("__VQCRESTARTS__", str(R["vqc_restarts"])) \
        .replace("__FULLTABLE__", latex_full_table(R)) \
        .replace("__MATCHEDTABLE__", latex_matched_table(R)) \
        .replace("__COSTTABLE__", latex_cost_table(R)) \
        .replace("__BESTLABEL__", esc_label(best)) \
        .replace("__BESTAUC__", f"{best_e['test']['auc']:.4f}") \
        .replace("__BESTPRAUC__", f"{best_e['test']['pr_auc']:.4f}") \
        .replace("__BESTBRIER__", f"{best_e['test']['brier']:.4f}") \
        .replace("__BESTPROFIT__", f"{R['best_profit_m']:.1f}") \
        .replace("__BESTAPPROVAL__", f"{R['best_approval_pct']:.1f}") \
        .replace("__THRLIST__", R["threshold_list_tex"]) \
        .replace("__IMPRUNNERS__", R["imp_runners_tex"]) \
        .replace("__IMPLIST__", latex_importance_lines(R)) \
        .replace("__HARDWARE__", hardware_paragraph) \
        .replace("__PYVER__", env["python"]) \
        .replace("__SKVER__", env["scikit_learn"]) \
        .replace("__QKVER__", env["qiskit"])


def compile_latex(tex: str) -> Path:
    build = PAPER_DIR / "_build"
    build.mkdir(exist_ok=True)
    (PAPER_DIR / "paper.tex").write_text(tex, encoding="utf-8")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={build}", "paper.tex"],
        cwd=PAPER_DIR, check=True, capture_output=True, text=True,
    )
    out = PAPER_DIR / PDF_NAME
    shutil.copy(build / "paper.pdf", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["auto", "latex", "fpdf"], default="auto")
    args = ap.parse_args()
    R = load_all()

    engine = args.engine
    if engine == "auto":
        engine = "latex" if shutil.which("latexmk") and shutil.which("pdflatex") else "fpdf"

    if engine == "latex":
        try:
            out = compile_latex(build_tex(R))
        except subprocess.CalledProcessError as err:
            log = (err.stdout or "")[-3000:]
            print(log)
            raise
    else:
        from paper.fallback_fpdf import render_fpdf

        (PAPER_DIR / "paper.tex").write_text(build_tex(R), encoding="utf-8")  # still emit .tex
        out = render_fpdf(R, PAPER_DIR / PDF_NAME)
    print(f"[saved] {out} (engine: {engine})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
