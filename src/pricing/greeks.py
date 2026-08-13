"""
Greeks: analytical (closed-form) and finite-difference (numerical) versions.

Kept as two independent implementations on purpose -- the whole point of a
model-validation project is to check that two methodologically different
routes to the same quantity agree. `validation/greeks_validation.py` runs
that comparison and reports the discrepancy.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price, _d1, _d2


# ---------------------------------------------------------------------------
# analytical Greeks
# ---------------------------------------------------------------------------
def analytical_greeks(S, K, T, r, q, sigma, option_type: str = "call") -> dict:
    """Closed-form Greeks. delta/gamma per 1.0 in S, vega per 1 vol pt,
    theta per calendar day, rho per 1% rate move."""
    S, K, T, r, q, sigma = map(np.asarray, (S, K, T, r, q, sigma))
    T_safe = np.where(T <= 0, 1e-12, T)
    d1 = _d1(S, K, T_safe, r, q, sigma)
    d2 = _d2(S, K, T_safe, r, q, sigma)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T_safe) * pdf_d1 / (S * sigma * np.sqrt(T_safe))
    vega = S * np.exp(-q * T_safe) * pdf_d1 * np.sqrt(T_safe) / 100.0

    if option_type == "call":
        delta = np.exp(-q * T_safe) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T_safe) * pdf_d1 * sigma / (2.0 * np.sqrt(T_safe))
            - r * K * np.exp(-r * T_safe) * norm.cdf(d2)
            + q * S * np.exp(-q * T_safe) * norm.cdf(d1)
        ) / 365.0
        rho = K * T_safe * np.exp(-r * T_safe) * norm.cdf(d2) / 100.0
    elif option_type == "put":
        delta = np.exp(-q * T_safe) * (norm.cdf(d1) - 1.0)
        theta = (
            -S * np.exp(-q * T_safe) * pdf_d1 * sigma / (2.0 * np.sqrt(T_safe))
            + r * K * np.exp(-r * T_safe) * norm.cdf(-d2)
            - q * S * np.exp(-q * T_safe) * norm.cdf(-d1)
        ) / 365.0
        rho = -K * T_safe * np.exp(-r * T_safe) * norm.cdf(-d2) / 100.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    mask_expired = (T <= 0)
    return {
        "delta": np.where(mask_expired, 0.0, delta),
        "gamma": np.where(mask_expired, 0.0, gamma),
        "vega":  np.where(mask_expired, 0.0, vega),
        "theta": np.where(mask_expired, 0.0, theta),
        "rho":   np.where(mask_expired, 0.0, rho),
    }


# ---------------------------------------------------------------------------
# finite-difference Greeks (central differences on the BS price itself)
# ---------------------------------------------------------------------------
def finite_diff_greeks(S, K, T, r, q, sigma, option_type: str = "call",
                       h_S: float | None = None, h_sigma: float = 1e-4,
                       h_T: float = 1e-4, h_r: float = 1e-4) -> dict:
    """Bump-and-reprice Greeks using the *same* BS pricer as the analytical
    version, so any discrepancy reflects the finite-difference approximation
    error only -- not a difference in pricing model."""
    S = float(S)
    h_S = h_S or max(S * 1e-4, 1e-4)

    def price(S_=S, T_=T, r_=r, sigma_=sigma):
        return float(bs_price(S_, K, T_, r_, q, sigma_, option_type))

    p0 = price()
    delta = (price(S_=S + h_S) - price(S_=S - h_S)) / (2 * h_S)
    gamma = (price(S_=S + h_S) - 2 * p0 + price(S_=S - h_S)) / (h_S ** 2)
    vega = (price(sigma_=sigma + h_sigma) - price(sigma_=sigma - h_sigma)) / (2 * h_sigma) / 100.0
    # theta: price loses value as T decreases -> dV/dt = -dV/dT
    theta = -(price(T_=T + h_T) - price(T_=T - h_T)) / (2 * h_T) / 365.0
    rho = (price(r_=r + h_r) - price(r_=r - h_r)) / (2 * h_r) / 100.0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
