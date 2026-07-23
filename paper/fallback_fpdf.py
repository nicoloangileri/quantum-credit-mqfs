"""fpdf2 fallback renderer: same structure and numbers as the LaTeX paper.

Used when no TeX engine is available. Equations are rendered with matplotlib
mathtext; figures and tables come from the same results/*.json as the LaTeX
build. Unicode is handled with matplotlib's bundled DejaVu fonts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
from experiments.render_tables import LABELS, ORDER
from fpdf import FPDF

from quantum_credit import config

MARGIN = 18


def _dejavu() -> tuple[str, str]:
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return str(base / "DejaVuSans.ttf"), str(base / "DejaVuSans-Bold.ttf")


def _math_png(expr: str, tmpdir: str, name: str) -> str:
    from matplotlib import mathtext

    path = str(Path(tmpdir) / f"{name}.png")
    mathtext.math_to_image(expr, path, dpi=220, format="png")
    return path


class Paper(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        reg, bold = _dejavu()
        self.add_font("DV", "", reg)
        self.add_font("DV", "B", bold)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.set_margins(MARGIN, MARGIN)

    def h1(self, text: str) -> None:
        self.set_font("DV", "B", 13)
        self.ln(4)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def para(self, text: str, size: float = 9.5) -> None:
        self.set_font("DV", "", size)
        self.multi_cell(0, 4.6, text)
        self.ln(1.5)

    def equation(self, expr: str, tmpdir: str, name: str) -> None:
        png = _math_png(expr, tmpdir, name)
        self.ln(1)
        self.image(png, x=self.w / 2 - 35, h=9)
        self.ln(2)

    def table(self, header: list[str], rows: list[list[str]], widths: list[float]) -> None:
        self.set_font("DV", "B", 7.5)
        for h, w in zip(header, widths, strict=True):
            self.cell(w, 5.6, h, border="B")
        self.ln()
        self.set_font("DV", "", 7.5)
        for row in rows:
            for cell, w in zip(row, widths, strict=True):
                self.cell(w, 5.2, cell)
            self.ln()
        self.ln(3)

    def figure(self, name: str, caption: str, width: float = 120) -> None:
        path = config.FIGURES_DIR / name
        if not path.exists():
            return
        if self.get_y() > self.h - 105:
            self.add_page()
        self.image(str(path), x=(self.w - width) / 2, w=width)
        self.set_font("DV", "", 8)
        self.multi_cell(0, 4.2, caption, align="C")
        self.ln(3)


def _label(name: str) -> str:
    return LABELS[name].replace("**", "")


def render_fpdf(R: dict, out: Path) -> Path:
    full, matched, man = R["full"], R["matched"], R["manifest"]
    sizes = matched["config"]["sizes"]
    agg = matched["aggregates"]
    delta = R["delta_best_classical_minus_best_quantum_n2000"]

    pdf = Paper()
    pdf.add_page()
    pdf.set_font("DV", "B", 15)
    pdf.multi_cell(0, 8,
                   "Classical versus Quantum Machine Learning for Credit Default "
                   "Prediction: An Honest Benchmark at Matched Sample Sizes")
    pdf.set_font("DV", "", 10)
    pdf.multi_cell(0, 5.5,
                   "Nicolò Angileri — Mediterranean Quantitative Finance Society (MQFS) "
                   "Working Paper — July 2026")
    pdf.ln(3)

    with tempfile.TemporaryDirectory() as tmp:
        pdf.h1("Abstract")
        pdf.para(
            "We benchmark three classical credit-scoring models against a fidelity "
            "quantum-kernel SVC and a variational quantum classifier on the UCI 'Default of "
            f"Credit Card Clients' dataset ({man['cleaning']['n_rows']:,} real accounts, "
            f"default rate {100 * man['default_rate']:.2f}%). Evaluation is finance-aware: "
            "an asymmetric expected-loss matrix and NT$ profit curves, not accuracy alone. "
            "Fairness rule: classical baselines are re-trained on exactly the same "
            f"stratified subsamples used by the quantum models (n in {list(sizes)}, three "
            "draws). At every matched size the best classical model matches or exceeds both "
            f"quantum models (at n=2000, AUC gap {delta:.4f}) while training orders of "
            "magnitude faster. No quantum advantage is claimed — or observed — in this "
            "regime; the contribution is the protocol and the measured operating envelope.")

        pdf.h1("1  Data")
        pdf.para(
            f"UCI id 350 (Yeh & Lien 2009): {man['cleaning']['n_rows']:,} clients, 23 raw "
            f"features, {man['cleaning']['n_defaults']:,} defaults. SHA-256 of the cached "
            f"raw file: {man['sha256'][:16]}…. Defensive validation (row count, ranges, no "
            f"missing values); documented recodes: {man['cleaning']['education_recoded']} "
            f"EDUCATION cells and {man['cleaning']['marriage_recoded']} MARRIAGE cells to "
            "'others'. Fixed stratified 75/25 split, seed 42.")

        pdf.h1("2  Methodology")
        pdf.para(
            "Classical: calibrated logistic regression, random forest, histogram gradient "
            "boosting (balanced class weights), 5-fold stratified CV; thresholds chosen on "
            "out-of-fold predictions only. Cost-sensitive evaluation with c_FN:c_FP = "
            f"{full['config']['cost_fn']:.0f}:{full['config']['cost_fp']:.0f}:")
        pdf.equation(r"$\hat C(t) = \frac{1}{n}(c_{FN}\,FN(t) + c_{FP}\,FP(t)),\quad "
                     r"t^* = \frac{c_{FP}}{c_{FP}+c_{FN}} = 1/6$", tmp, "cost")
        pdf.para(
            "Profit with heterogeneous exposure (L_i = client credit limit, margin "
            f"m={full['config']['profit_margin']:.2f}, LGD λ="
            f"{full['config']['lgd_fraction']:.2f}, stylised):")
        pdf.equation(r"$\Pi(t) = \sum_{i:\,p_i<t}[(1-y_i)\,m\,L_i - y_i\,\lambda\,L_i]$",
                     tmp, "profit")
        pdf.para(
            "Quantum: ZZFeatureMap embedding (Havlíček et al. 2019), top-d features by "
            "mutual information (d=4 primary, d=8 robustness), angles in [0, π]. Fidelity "
            "kernel with SVC (precomputed Gram; same C grid as the classical RBF-SVC twin):")
        pdf.equation(r"$K(x,x') = |\langle\Phi(x)|\Phi(x')\rangle|^2$", tmp, "kernel")
        pdf.para(
            f"VQC: RealAmplitudes ansatz ({R['vqc_params']} parameters), parity observable, "
            f"COBYLA ({R['vqc_maxiter']} iterations, {R['vqc_restarts']} restarts), "
            "class-weighted cross-entropy:")
        pdf.equation(r"$p(x;\theta) = \tfrac12(1 - \langle\Phi(x)|U^\dagger(\theta)\,"
                     r"Z^{\otimes d}\,U(\theta)|\Phi(x)\rangle)$", tmp, "vqc")
        pdf.para(
            "Both quantum models run on exact statevector simulation (noise-free upper "
            "bounds). All models — classical and quantum — are evaluated on the same fixed "
            f"test set (n={full['data']['n_test']:,}) and compared at the fixed Bayes "
            "threshold; quantum-on-2000 is never compared with classical-on-"
            f"{full['data']['n_train']:,}.")

        pdf.h1("3  Results — full training set (classical)")
        rows = []
        for name in ["logistic_calibrated", "random_forest", "hist_gradient_boosting"]:
            e = full["models"][name]
            rows.append([
                _label(name)[:34],
                f"{e['cv']['auc']['mean']:.4f}±{e['cv']['auc']['std']:.4f}",
                f"{e['test']['auc']:.4f}",
                f"{e['test']['pr_auc']:.4f}",
                f"{e['test']['brier']:.4f}",
                f"{e['test_decisions']['profit_at_oof_threshold_ntd'] / 1e6:.1f}",
            ])
        pdf.table(["Model", "CV AUC", "Test AUC", "PR-AUC", "Brier", "Profit (NT$ M)"],
                  rows, [58, 30, 22, 22, 20, 26])
        pdf.figure("calibration_full.png", "Fig. 1 — Reliability diagrams (test set).")
        pdf.figure("cost_profit_full.png",
                   "Fig. 2 — Expected cost and NT$ profit vs threshold.", width=165)
        pdf.figure("permutation_importance.png",
                   "Fig. 3 — Permutation importance, best full-data model.")

        pdf.h1("4  Results — matched subsamples (the honest table)")
        rows = []
        for name in ORDER:
            if name not in agg[str(sizes[0])]:
                continue
            r = [_label(name)[:34]]
            for s in sizes:
                a = agg[str(s)][name]["auc"]
                r.append(f"{a['mean']:.4f}±{a['std']:.4f}")
            r.append(f"{agg[str(max(sizes))][name]['total_time_s']['mean']:.1f}")
            rows.append(r)
        pdf.table(["Model (C/Q)"] + [f"n={s}" for s in sizes] + ["t@2000 (s)"],
                  rows, [58, 33, 33, 33, 21])
        pdf.figure("auc_vs_n.png",
                   "Fig. 4 — Learning curves: classical (solid) vs quantum (dashed).")
        pdf.figure("timing.png", "Fig. 5 — Honest wall-clock accounting at n=2000.")
        pdf.figure("kernel_gram.png", "Fig. 6 — Fidelity-kernel Gram sub-block (d=4).")

        pdf.h1("5  Limitations and conclusion")
        pdf.para(
            "Single dataset; stylised economics; exact simulation (hardware would be "
            "strictly worse); light symmetric hyperparameter search; three subsample seeds. "
            "Classical ML wins at every matched size while running orders of magnitude "
            "faster: no quantum advantage is claimed, and none was observed. The "
            "contribution is the reproducible, cost-honest, matched-sample protocol. "
            "Built with AI-assisted engineering; every design decision, formula and "
            "validation reviewed and owned by the author.")
        pdf.h1("References")
        pdf.para(
            "Yeh & Lien (2009) Expert Syst. Appl. 36(2). — Kelly, Longjohn & Nottingham, "
            "UCI ML Repository, id 350. — Havlíček et al. (2019) Nature 567. — Schuld & "
            "Killoran (2019) PRL 122. — Liu, Arunachalam & Temme (2021) Nat. Phys. 17. — "
            "Elkan (2001) IJCAI. — Hand & Henley (1997) JRSS-A 160. — Lessmann et al. "
            "(2015) EJOR 247. — Thomas, Edelman & Crook (2002) SIAM. — Cerezo et al. "
            "(2021) Nat. Rev. Phys. 3.", size=8.5)

    pdf.output(str(out))
    return out
