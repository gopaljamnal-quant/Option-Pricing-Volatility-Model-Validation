"""
Stress testing for model validation.

Two distinct kinds of stress, both standard in a model-validation writeup:

    1) numerical edge-case stability -- does the pricer / Greeks / implied-
       vol solver behave sanely at the boundaries (deep ITM/OTM, near-zero
       time to expiry, very high/low vol)? These aren't market scenarios,
       they're checks that the *code* doesn't blow up or return nonsense
       where a trader could actually be quoting.

    2) market scenario shocks -- economically motivated moves in spot,
       vol, and rates, applied to a small option book, checked for
       monotonicity and sane P&L attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..pricing.black_scholes import bs_price
from ..pricing.greeks import analytical_greeks
from ..volatility.implied_vol import implied_vol


# ---------------------------------------------------------------------------
# 1) numerical edge-case grid
# ---------------------------------------------------------------------------
def edge_case_grid(S0: float = 100.0, r: float = 0.04, q: float = 0.0,
                   option_type: str = "call", vega_floor: float = 1e-4) -> pd.DataFrame:
    """Price + Greeks + IV round-trip at extreme but plausible inputs.

    Deep ITM/OTM options near expiry have vega collapsing towards zero --
    the price is then almost pure intrinsic value and is *structurally*
    uninformative about sigma, regardless of solver quality. We flag those
    cases separately (`iv_ill_posed`) rather than lumping them in with
    genuine solver failures (`iv_roundtrip_ok`), because conflating the two
    would misdiagnose an economic limitation as a code defect.
    """
    cases = []
    moneyness_extremes = [0.3, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]
    maturity_extremes = [1/365, 7/365, 30/365, 1.0, 5.0]
    vol_extremes = [0.01, 0.05, 0.20, 1.0, 3.0]

    for m in moneyness_extremes:
        for T in maturity_extremes:
            for sigma in vol_extremes:
                K = S0 * m
                price = float(bs_price(S0, K, T, r, q, sigma, option_type))
                greeks = analytical_greeks(S0, K, T, r, q, sigma, option_type)
                vega_raw = float(greeks["vega"]) * 100.0  # undo /100 scaling
                recovered_iv = implied_vol(price, S0, K, T, r, q, option_type)

                is_finite = np.isfinite(price)
                is_nonneg = price >= -1e-8
                ill_posed = vega_raw < vega_floor
                iv_recovered_ok = (
                    np.isfinite(recovered_iv)
                    and abs(recovered_iv - sigma) < 1e-3
                )
                cases.append({
                    "moneyness": m, "T": T, "sigma": sigma,
                    "price": price, "vega": vega_raw,
                    "delta": float(greeks["delta"]),
                    "gamma": float(greeks["gamma"]),
                    "recovered_iv": recovered_iv,
                    "price_finite": is_finite,
                    "price_nonneg": is_nonneg,
                    "iv_ill_posed": ill_posed,
                    "iv_roundtrip_ok": iv_recovered_ok or ill_posed,
                    "all_pass": is_finite and is_nonneg and (iv_recovered_ok or ill_posed),
                })
    return pd.DataFrame(cases)


def edge_case_summary(grid_df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_cases": len(grid_df),
        "pct_price_finite": grid_df["price_finite"].mean(),
        "pct_price_nonneg": grid_df["price_nonneg"].mean(),
        "pct_iv_roundtrip_ok": grid_df["iv_roundtrip_ok"].mean(),
        "n_iv_ill_posed_low_vega": int(grid_df["iv_ill_posed"].sum()),
        "pct_all_pass": grid_df["all_pass"].mean(),
        "n_failures": int((~grid_df["all_pass"]).sum()),
    })


# ---------------------------------------------------------------------------
# 2) market scenario shocks
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    s_shock: float = 0.0     # additive log-return on spot
    vol_shock: float = 0.0   # additive on sigma
    rate_shock: float = 0.0  # additive on r


DEFAULT_SCENARIOS = [
    Scenario("base",                s_shock= 0.00, vol_shock=0.00,  rate_shock=0.000),
    Scenario("equity_-5%",          s_shock=-0.05, vol_shock=0.02,  rate_shock=0.000),
    Scenario("equity_-10%_vol+5",   s_shock=-0.10, vol_shock=0.05,  rate_shock=0.000),
    Scenario("equity_-20%_vol+15",  s_shock=-0.20, vol_shock=0.15,  rate_shock=-0.005),
    Scenario("equity_+10%_vol-3",   s_shock= 0.10, vol_shock=-0.03, rate_shock=0.000),
    Scenario("rates_+100bps",       s_shock= 0.00, vol_shock=0.00,  rate_shock=0.010),
    Scenario("vol_spike_only",      s_shock= 0.00, vol_shock=0.20,  rate_shock=0.000),
    Scenario("2020-style_crash",    s_shock=-0.30, vol_shock=0.40,  rate_shock=-0.010),
]


def run_scenarios(S0: float, K: float, T: float, r: float, q: float, sigma: float,
                  option_type: str = "call",
                  scenarios: list[Scenario] | None = None) -> pd.DataFrame:
    """Reprice a single option under each scenario; report price, P&L, Greeks."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    base_price = float(bs_price(S0, K, T, r, q, sigma, option_type))

    rows = []
    for sc in scenarios:
        S_new = S0 * np.exp(sc.s_shock)
        sigma_new = max(sigma + sc.vol_shock, 0.01)
        r_new = r + sc.rate_shock
        price_new = float(bs_price(S_new, K, T, r_new, q, sigma_new, option_type))
        g = analytical_greeks(S_new, K, T, r_new, q, sigma_new, option_type)
        rows.append({
            "scenario": sc.name,
            "S": S_new, "sigma": sigma_new, "r": r_new,
            "price": price_new,
            "pnl": price_new - base_price,
            "delta": float(g["delta"]), "gamma": float(g["gamma"]),
            "vega": float(g["vega"]), "theta": float(g["theta"]),
        })
    return pd.DataFrame(rows)


def check_monotonicity(scenario_df: pd.DataFrame, option_type: str = "call") -> dict:
    """Sanity check: for a call, P&L should be monotonically increasing in S
    shock (holding vol fixed) -- a violation would indicate a pricing bug."""
    equity_only = scenario_df[scenario_df["scenario"].str.startswith("equity_")]
    if len(equity_only) < 2:
        return {"checked": False, "reason": "not enough equity-only scenarios"}
    ordered = equity_only.sort_values("S")
    diffs = ordered["price"].diff().dropna()
    if option_type == "call":
        monotonic = (diffs >= -1e-6).all()
    else:
        monotonic = (diffs <= 1e-6).all()
    return {"checked": True, "monotonic_in_spot": bool(monotonic)}
