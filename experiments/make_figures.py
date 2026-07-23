"""Regenerate every figure from results/*.json (no recomputation, no invention)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import load_results
from quantum_credit import config

FIG_KW = {"dpi": 150, "bbox_inches": "tight"}
MODEL_LABELS = {
    "logistic_calibrated": "Logistic (calibrated)",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
    "svc_rbf": "SVC (RBF kernel)",
    "qk_svc_d4": "QK-SVC (fidelity, d=4)",
    "qk_svc_d8": "QK-SVC (fidelity, d=8)",
    "vqc_d4": "VQC (d=4)",
}
CLASSICAL = ["logistic_calibrated", "random_forest", "hist_gradient_boosting", "svc_rbf"]
QUANTUM = ["qk_svc_d4", "qk_svc_d8", "vqc_d4"]


def out(name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return str(config.FIGURES_DIR / name)


def fig_calibration(full: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, entry in full["models"].items():
        c = entry["calibration"]
        ax.plot(c["mean_predicted"], c["fraction_positive"], "o-", ms=4,
                label=f"{MODEL_LABELS[name]} (Brier {entry['test']['brier']:.4f})")
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default frequency")
    ax.set_title("Reliability diagram, held-out test set (quantile bins)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out("calibration_full.png"), **FIG_KW)
    plt.close(fig)


def fig_cost_profit(full: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    bayes = full["config"]["bayes_threshold"]
    for name, entry in full["models"].items():
        t = np.array(entry["curves"]["thresholds"])
        axes[0].plot(t, entry["curves"]["expected_cost_test"], label=MODEL_LABELS[name])
        axes[1].plot(t, np.array(entry["curves"]["profit_ntd_test"]) / 1e6,
                     label=MODEL_LABELS[name])
    axes[0].axvline(bayes, color="k", ls=":", lw=1,
                    label=f"Bayes threshold {bayes:.3f}")
    axes[0].set_xlabel("Rejection threshold $t$")
    axes[0].set_ylabel("Expected cost per applicant (units)")
    axes[0].set_title(f"Cost curve ($c_{{FN}}$:{full['config']['cost_fn']:.0f}, "
                      f"$c_{{FP}}$:{full['config']['cost_fp']:.0f})")
    axes[1].set_xlabel("Approval threshold $t$ (approve if $p<t$)")
    axes[1].set_ylabel("Portfolio profit (NT$ millions)")
    axes[1].set_title("Profit curve, stylised economics")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out("cost_profit_full.png"), **FIG_KW)
    plt.close(fig)


def fig_importance(full: dict) -> None:
    best = max(full["models"], key=lambda m: full["models"][m]["test"]["auc"])
    imp = full["models"][best]["permutation_importance"][:10][::-1]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.barh([r["feature"] for r in imp], [r["mean_auc_drop"] for r in imp],
            xerr=[r["std"] for r in imp], color="#4878b0")
    ax.set_xlabel("Mean AUC drop when permuted (test set, 10 repeats)")
    ax.set_title(f"Permutation importance — {MODEL_LABELS[best]}")
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(out("permutation_importance.png"), **FIG_KW)
    plt.close(fig)


def fig_auc_vs_n(matched: dict, full: dict) -> None:
    sizes = matched["config"]["sizes"]
    agg = matched["aggregates"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name in CLASSICAL + QUANTUM:
        if name not in agg[str(sizes[0])]:
            continue
        means = [agg[str(s)][name]["auc"]["mean"] for s in sizes]
        stds = [agg[str(s)][name]["auc"]["std"] for s in sizes]
        style = dict(ls="--", marker="s") if name in QUANTUM else dict(ls="-", marker="o")
        ax.errorbar(sizes, means, yerr=stds, capsize=3, ms=4, label=MODEL_LABELS[name], **style)
    best_full = max(full["models"].values(), key=lambda e: e["test"]["auc"])["test"]["auc"]
    ax.axhline(best_full, color="k", lw=1, ls=":",
               label=f"best classical, full n=22,500 (AUC {best_full:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Training subsample size $n$ (identical subsamples for all models)")
    ax.set_ylabel("Test AUC (mean ± std over 3 subsample seeds)")
    ax.set_title("Matched-sample comparison: classical (solid) vs quantum (dashed)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(out("auc_vs_n.png"), **FIG_KW)
    plt.close(fig)


def fig_kernel(matched: dict) -> None:
    ks = matched.get("kernel_sample")
    if not ks:
        return
    g = np.array(ks["gram_100"])
    labels = np.array(ks["labels_100"])
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(g, cmap="viridis", vmin=0, vmax=1)
    split = int(np.searchsorted(labels, 1))
    ax.axhline(split - 0.5, color="w", lw=0.8)
    ax.axvline(split - 0.5, color="w", lw=0.8)
    ax.set_title(f"Fidelity kernel Gram sub-block (d={ks['d']}, sorted by label)")
    ax.set_xlabel("client index (non-default | default)")
    fig.colorbar(im, ax=ax, label="$K(x_i, x_j)$")
    fig.savefig(out("kernel_gram.png"), **FIG_KW)
    plt.close(fig)


def fig_timing(matched: dict) -> None:
    biggest = str(max(matched["config"]["sizes"]))
    agg = matched["aggregates"][biggest]
    names = [n for n in CLASSICAL + QUANTUM if n in agg]
    times = [agg[n]["total_time_s"]["mean"] for n in names]
    colors = ["#4878b0" if n in CLASSICAL else "#d1615d" for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar([MODEL_LABELS[n] for n in names], times, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock train+test time (s, log scale)")
    ax.set_title(f"Honest cost accounting at n={biggest} (mean over seeds)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for i, t in enumerate(times):
        ax.text(i, t * 1.15, f"{t:.1f}s", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out("timing.png"), **FIG_KW)
    plt.close(fig)


def main() -> None:
    full = load_results("classical_full.json")
    matched = load_results("matched_subsamples.json")
    fig_calibration(full)
    fig_cost_profit(full)
    fig_importance(full)
    fig_auc_vs_n(matched, full)
    fig_kernel(matched)
    fig_timing(matched)
    print(f"[saved] figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
