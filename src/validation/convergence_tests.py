"""
Monte Carlo convergence validation.

Standard model-validation exercise: does the MC estimator converge to the
closed-form Black-Scholes price as n_paths grows, and does its empirical
error shrink at the theoretical O(1/sqrt(n)) rate?

We check both:
    1) bias: MC price - BS price, should shrink towards 0
    2) the MC standard error should track 1/sqrt(n) on a log-log plot
    3) the BS price should lie inside the MC 95% CI at the expected ~95%
       hit rate across repeated trials (a coverage test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..pricing.black_scholes import bs_price
from ..pricing.monte_carlo import mc_european_price


def convergence_table(S0, K, T, r, q, sigma, option_type: str = "call",
                      path_counts=(1_000, 5_000, 25_000, 100_000, 500_000),
                      seed: int = 0) -> pd.DataFrame:
    """MC price/error at increasing path counts vs the BS benchmark."""
    bs_ref = float(bs_price(S0, K, T, r, q, sigma, option_type))
    rows = []
    for n in path_counts:
        res = mc_european_price(S0, K, T, r, q, sigma, option_type,
                                n_paths=n, seed=seed)
        rows.append({
            "n_paths": n,
            "mc_price": res["price"],
            "bs_price": bs_ref,
            "abs_error": abs(res["price"] - bs_ref),
            "mc_stderr": res["stderr"],
            "ci_covers_bs": res["ci_lower"] <= bs_ref <= res["ci_upper"],
        })
    return pd.DataFrame(rows)


def coverage_test(S0, K, T, r, q, sigma, option_type: str = "call",
                  n_paths: int = 20_000, n_trials: int = 200,
                  alpha: float = 0.95) -> dict:
    """Repeat MC pricing n_trials times; check the empirical fraction of
    trials whose CI contains the true BS price against the nominal `alpha`.
    A well-calibrated MC estimator should hit close to `alpha`."""
    bs_ref = float(bs_price(S0, K, T, r, q, sigma, option_type))
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(alpha, 1.96)

    hits = 0
    for trial in range(n_trials):
        res = mc_european_price(S0, K, T, r, q, sigma, option_type,
                                n_paths=n_paths, seed=trial)
        lo = res["price"] - z * res["stderr"]
        hi = res["price"] + z * res["stderr"]
        hits += int(lo <= bs_ref <= hi)

    empirical_coverage = hits / n_trials
    return {
        "nominal_coverage": alpha,
        "empirical_coverage": empirical_coverage,
        "n_trials": n_trials,
        "n_paths_per_trial": n_paths,
    }


def plot_convergence(conv_df: pd.DataFrame, savepath: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(conv_df["n_paths"], conv_df["mc_price"], "o-", label="MC price")
    ax.axhline(conv_df["bs_price"].iloc[0], color="crimson", ls="--", label="BS price")
    ax.fill_between(conv_df["n_paths"],
                    conv_df["mc_price"] - 1.96 * conv_df["mc_stderr"],
                    conv_df["mc_price"] + 1.96 * conv_df["mc_stderr"],
                    alpha=0.2, label="95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("n_paths")
    ax.set_ylabel("price")
    ax.set_title("MC price convergence to BS benchmark")
    ax.legend()

    ax = axes[1]
    ax.loglog(conv_df["n_paths"], conv_df["mc_stderr"], "o-", label="MC stderr")
    ref = conv_df["mc_stderr"].iloc[0] * np.sqrt(conv_df["n_paths"].iloc[0] / conv_df["n_paths"])
    ax.loglog(conv_df["n_paths"], ref, "--", color="grey", label=r"theoretical $O(1/\sqrt{n})$")
    ax.set_xlabel("n_paths")
    ax.set_ylabel("standard error")
    ax.set_title("MC error rate vs theoretical rate")
    ax.legend()

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140)
    return fig
