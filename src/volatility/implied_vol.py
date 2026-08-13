"""
Implied volatility solver.

Given an observed (or model-generated) option price, solve for the sigma
that reprices it exactly under Black-Scholes. Two solvers are provided:

    - implied_vol            Newton-Raphson with a bisection fallback
                              (fast; used for single quotes)
    - implied_vol_batch      vectorized bisection over arrays of prices
                              (used when building a smile from a whole chain)

A round-trip test (price at known sigma -> solve -> recover sigma) is the
standard sanity check and is run in `validation/greeks_validation.py`-style
checks inside `main.py`.
"""
from __future__ import annotations

import numpy as np

from ..pricing.black_scholes import bs_price
from ..pricing.greeks import analytical_greeks


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                option_type: str = "call", tol: float = 1e-8,
                max_iter: int = 100) -> float:
    """Newton-Raphson implied vol with a bisection fallback for robustness."""
    if T <= 0 or price <= 0:
        return np.nan

    # arbitrage lower bound uses *discounted forward* intrinsic value, not
    # raw S-K -- with a nonzero dividend yield or long maturity, S*exp(-qT)
    # can sit meaningfully below S, so using undiscounted S-K here would
    # reject perfectly valid low-vol prices as "sub-intrinsic"
    fwd_intrinsic = (
        max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == "call"
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    )
    if price < fwd_intrinsic - 1e-8:
        return np.nan

    sigma = 0.2
    for _ in range(max_iter):
        p = float(bs_price(S, K, T, r, q, sigma, option_type))
        vega = float(analytical_greeks(S, K, T, r, q, sigma, option_type)["vega"]) * 100.0
        diff = p - price
        if abs(diff) < tol:
            return float(sigma)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if sigma <= 0 or sigma > 5.0:
            sigma = 0.2  # reset and let bisection take over
            break

    # bisection fallback -- guaranteed to converge if a solution exists
    lo, hi = 1e-4, 5.0
    p_lo = float(bs_price(S, K, T, r, q, lo, option_type)) - price
    p_hi = float(bs_price(S, K, T, r, q, hi, option_type)) - price
    if p_lo * p_hi > 0:
        return np.nan  # no sign change -> no solution in range

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p_mid = float(bs_price(S, K, T, r, q, mid, option_type)) - price
        if abs(p_mid) < tol:
            return mid
        if p_mid * p_lo > 0:
            lo, p_lo = mid, p_mid
        else:
            hi = mid
    return mid


def implied_vol_batch(prices: np.ndarray, S: float, strikes: np.ndarray,
                      T: np.ndarray, r: float, q: float,
                      option_types: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Vectorized-looking wrapper (loops internally; robust > fast here)."""
    out = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        out[i] = implied_vol(
            price=prices[i], S=S, K=strikes[i], T=T[i],
            r=r, q=q, option_type=option_types[i], tol=tol,
        )
    return out
