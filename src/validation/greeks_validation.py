"""
Greeks validation: compare analytical (closed-form) Greeks against
finite-difference Greeks across a grid of strikes and maturities.

A model-validation report typically wants this as a grid, not a single
point -- discrepancies often concentrate near expiry or deep ITM/OTM where
finite-difference approximations get noisy. We report both the raw
discrepancy and a relative (%) discrepancy so reviewers can set a
materiality threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..pricing.greeks import analytical_greeks, finite_diff_greeks


def validate_greeks_grid(
    S0: float, r: float, q: float, sigma: float,
    moneyness_grid=(0.8, 0.9, 1.0, 1.1, 1.2),
    maturity_grid=(1/52, 1/12, 0.25, 0.5, 1.0),
    option_type: str = "call",
    tol: float = 1e-3,
) -> pd.DataFrame:
    """Returns a tidy DataFrame with analytical vs FD Greeks and a pass/fail
    flag per cell using an absolute-difference tolerance `tol`."""
    rows = []
    for m in moneyness_grid:
        K = S0 * m
        for T in maturity_grid:
            a = analytical_greeks(S0, K, T, r, q, sigma, option_type)
            f = finite_diff_greeks(S0, K, T, r, q, sigma, option_type)
            row = {"moneyness": m, "T": T}
            for greek in ["delta", "gamma", "vega", "theta", "rho"]:
                av, fv = float(a[greek]), float(f[greek])
                row[f"{greek}_analytical"] = av
                row[f"{greek}_fd"] = fv
                row[f"{greek}_abs_diff"] = abs(av - fv)
                row[f"{greek}_pass"] = abs(av - fv) < tol
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_validation(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Per-Greek pass rate and max discrepancy across the whole grid."""
    summary = []
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        summary.append({
            "greek": greek,
            "max_abs_diff": grid_df[f"{greek}_abs_diff"].max(),
            "mean_abs_diff": grid_df[f"{greek}_abs_diff"].mean(),
            "pass_rate": grid_df[f"{greek}_pass"].mean(),
        })
    return pd.DataFrame(summary)
