"""
Baseline volatility estimators against which the LSTM is benchmarked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def realized_vol(prices: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    r = log_returns(prices)
    vol = r.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol.dropna()


def ewma_vol(prices: pd.Series, lam: float = 0.94, annualize: bool = True) -> pd.Series:
    r = log_returns(prices)
    var = pd.Series(index=r.index, dtype=float)
    var.iloc[0] = r.iloc[0] ** 2
    for t in range(1, len(r)):
        var.iloc[t] = lam * var.iloc[t - 1] + (1.0 - lam) * r.iloc[t - 1] ** 2
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def garch_vol(prices: pd.Series, annualize: bool = True) -> pd.Series:
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("install the `arch` package to use GARCH") from e

    r = log_returns(prices) * 100.0
    am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100.0
    cond_vol.index = r.index
    if annualize:
        cond_vol = cond_vol * np.sqrt(TRADING_DAYS)
    return cond_vol
