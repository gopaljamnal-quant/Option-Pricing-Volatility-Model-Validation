"""
Black-Scholes-Merton closed-form pricer for European options.

This is the reference/benchmark model against which Monte Carlo, implied
vol recovery, and Greeks are all validated elsewhere in this project.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, q, sigma):
    return _d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


def bs_price(S, K, T, r, q, sigma, option_type: str = "call"):
    """European option price under Black-Scholes-Merton. Vectorized."""
    S, K, T, r, q, sigma = map(np.asarray, (S, K, T, r, q, sigma))

    intrinsic = np.where(option_type == "call",
                         np.maximum(S - K, 0.0),
                         np.maximum(K - S, 0.0))
    if np.all(T <= 0):
        return intrinsic

    T_safe = np.where(T <= 0, 1e-12, T)
    sigma_safe = np.where(sigma <= 0, 1e-8, sigma)
    d1 = _d1(S, K, T_safe, r, q, sigma_safe)
    d2 = _d2(S, K, T_safe, r, q, sigma_safe)

    if option_type == "call":
        price = S * np.exp(-q * T_safe) * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * np.exp(-q * T_safe) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return np.where(T <= 0, intrinsic, price)
