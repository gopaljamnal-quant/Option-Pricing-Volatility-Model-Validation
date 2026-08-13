"""
Volatility smile: construction from an option chain (or a synthetic chain),
a simple parametric fit, and diagnostic plots.

The fit uses a 4-parameter reduced quadratic-in-log-moneyness form (not the
full 5-parameter SVI, to keep the optimizer well-behaved on small synthetic
chains):

    iv(k) = a + b*k + c*k^2                    where k = log(K/S)

This is enough to capture skew (b) and curvature/smile (c), which is what
model validation reports usually care about: is the market's skew being
captured, and how much of the curve is unexplained by the quadratic (fit
residuals)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ..volatility.implied_vol import implied_vol


# ---------------------------------------------------------------------------
# build smile from a chain of (strike, price) at a single maturity
# ---------------------------------------------------------------------------
def build_smile(chain: pd.DataFrame, S: float, T: float, r: float, q: float) -> pd.DataFrame:
    """
    chain columns required: ['strike', 'mid_price', 'option_type']
    Returns a copy with 'iv' and 'log_moneyness' columns added.
    """
    out = chain.copy()
    ivs = [
        implied_vol(row["mid_price"], S, row["strike"], T, r, q, row["option_type"])
        for _, row in out.iterrows()
    ]
    out["iv"] = ivs
    out["log_moneyness"] = np.log(out["strike"] / S)
    return out.dropna(subset=["iv"])


def synthetic_smile(S: float = 100.0, T: float = 0.25,
                    strikes: np.ndarray | None = None,
                    a: float = 0.20, b: float = -0.12, c: float = 0.35,
                    noise_std: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic smile with known (a, b, c) so the fit routine
    can be validated by recovering the true parameters."""
    if strikes is None:
        strikes = np.linspace(0.7 * S, 1.3 * S, 21)
    k = np.log(strikes / S)
    iv = a + b * k + c * k ** 2
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        iv = iv + rng.normal(0, noise_std, size=iv.shape)
    iv = np.maximum(iv, 0.02)
    return pd.DataFrame({"strike": strikes, "log_moneyness": k, "iv": iv, "T": T})


# ---------------------------------------------------------------------------
# parametric fit
# ---------------------------------------------------------------------------
def _quad_form(k, a, b, c):
    return a + b * k + c * k ** 2


def fit_smile(smile_df: pd.DataFrame) -> dict:
    """Least-squares fit of iv(k) = a + b*k + c*k^2. Returns params + fit stats."""
    k = smile_df["log_moneyness"].values
    iv = smile_df["iv"].values
    popt, pcov = curve_fit(_quad_form, k, iv, p0=[0.2, 0.0, 0.1])
    a, b, c = popt
    fitted = _quad_form(k, *popt)
    resid = iv - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((iv - iv.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "a_level": float(a), "b_skew": float(b), "c_curvature": float(c),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r_squared": r_squared,
        "param_std_err": np.sqrt(np.diag(pcov)).tolist(),
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_smile(smile_df: pd.DataFrame, fit_params: dict | None = None,
              title: str = "Volatility Smile", savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(smile_df["log_moneyness"], smile_df["iv"], s=28,
              color="steelblue", label="Observed IV", zorder=3)

    if fit_params is not None:
        k_grid = np.linspace(smile_df["log_moneyness"].min(),
                             smile_df["log_moneyness"].max(), 200)
        fitted = _quad_form(k_grid, fit_params["a_level"],
                            fit_params["b_skew"], fit_params["c_curvature"])
        ax.plot(k_grid, fitted, color="crimson", lw=2,
               label=f"Quadratic fit (R²={fit_params['r_squared']:.3f})")

    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Log-moneyness  ln(K/S)")
    ax.set_ylabel("Implied volatility")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140)
    return fig
