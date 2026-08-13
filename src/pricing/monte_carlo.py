"""
Monte Carlo pricer under GBM, with antithetic variates.

The pricer itself is deliberately simple (one-step exact GBM sampling for
European payoffs). The interesting validation content -- convergence rate,
confidence intervals, bias vs Black-Scholes -- lives in
`validation/convergence_tests.py`, which calls `mc_european_price` at a
sequence of path counts.
"""
from __future__ import annotations

import numpy as np


def simulate_terminal(S0, r, q, sigma, T, n_paths, antithetic=True, seed=None):
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])
        if z.shape[0] < n_paths:
            z = np.concatenate([z, rng.standard_normal(n_paths - z.shape[0])])
    else:
        z = rng.standard_normal(n_paths)

    S_T = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    return S_T


def mc_european_price(S0, K, T, r, q, sigma, option_type: str = "call",
                      n_paths: int = 100_000, antithetic: bool = True,
                      seed: int | None = None) -> dict:
    """Price + standard error + 95% CI for a European option via MC."""
    S_T = simulate_terminal(S0, r, q, sigma, T, n_paths, antithetic, seed)
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    discounted = np.exp(-r * T) * payoff
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return {
        "price": price,
        "stderr": stderr,
        "ci_lower": price - 1.96 * stderr,
        "ci_upper": price + 1.96 * stderr,
        "n_paths": n_paths,
    }
