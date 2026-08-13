"""
Benchmark comparison for volatility forecasts.

Compares the LSTM forecaster against EWMA and GARCH(1,1) using the metrics
a model-validation reviewer will actually ask for:

    - MAE, RMSE           on vol level
    - QLIKE                loss used in the vol-forecasting literature,
                            penalizes under-prediction of variance more
                            than MSE does (Patton, 2011)
    - Diebold-Mariano test  formal test of whether one model's forecast
                            errors are significantly smaller than another's
                            (as opposed to just eyeballing which MAE is lower)

All functions take pre-aligned (actual, forecast) series -- alignment
across models with different lookback windows happens in `main.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------
def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def qlike(actual_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss on *variance* (not vol): log(f) + a/f. Standard in the
    realized-vol forecasting literature; robust to noisy vol proxies and
    penalizes under-forecasting more heavily than squared error."""
    ratio = actual_var / forecast_var
    return float(np.mean(np.log(forecast_var) + ratio))


def error_summary(actual: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """actual and each series in `forecasts` must share a comparable index;
    this function aligns them via inner join before scoring."""
    rows = []
    for name, fc in forecasts.items():
        joined = pd.concat([actual.rename("actual"), fc.rename("forecast")], axis=1).dropna()
        a, f = joined["actual"].values, joined["forecast"].values
        rows.append({
            "model": name,
            "n_obs": len(joined),
            "MAE":   mae(a, f),
            "RMSE":  rmse(a, f),
            "QLIKE": qlike(a ** 2, f ** 2),
        })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------
def diebold_mariano(actual: pd.Series, forecast_1: pd.Series, forecast_2: pd.Series,
                    h: int = 1, loss: str = "squared") -> dict:
    """DM test for equal predictive accuracy of forecast_1 vs forecast_2.

    Null hypothesis: the two forecasts have equal expected loss.
    A significantly negative DM statistic favors forecast_1 (lower loss);
    significantly positive favors forecast_2.

    `h` is the forecast horizon used in the long-run variance correction
    (Newey-West-style), matching the horizon the forecasts were made at.
    """
    df = pd.concat([
        actual.rename("actual"),
        forecast_1.rename("f1"),
        forecast_2.rename("f2"),
    ], axis=1).dropna()

    e1 = df["actual"] - df["f1"]
    e2 = df["actual"] - df["f2"]

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = e1.abs() - e2.abs()
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    d_mean = d.mean()

    # Newey-West long-run variance with h-1 lags
    gamma0 = d.var(ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = d.autocorr(lag) * gamma0 if not np.isnan(d.autocorr(lag)) else 0.0
        var_d += 2 * (1 - lag / h) * cov
    var_d = max(var_d, 1e-12)

    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "n_obs": n,
        "conclusion": (
            "forecast_1 significantly more accurate" if dm_stat < -1.96 else
            "forecast_2 significantly more accurate" if dm_stat > 1.96 else
            "no significant difference at 5% level"
        ),
    }
