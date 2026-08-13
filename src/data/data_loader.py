"""
Market-data loader. Tries yfinance first; falls back to a synthetic GBM
series so the project runs offline / in CI.
"""
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd


def load_prices(ticker: str = "SPY", start: str | None = None,
                end: str | None = None, fallback_seed: int = 0) -> pd.Series:
    start = start or (date.today() - timedelta(days=365 * 5)).isoformat()
    end = end or date.today().isoformat()
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError("empty download")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices = df["Close"].dropna()
        prices.name = ticker
        return prices
    except Exception as e:  # noqa: BLE001
        print(f"[data_loader] yfinance failed ({e}); generating synthetic series.")
        return synthetic_prices(seed=fallback_seed)


def synthetic_prices(n_days: int = 252 * 5, S0: float = 100.0, mu: float = 0.08,
                     sigma: float = 0.20, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    z = rng.standard_normal(n_days)
    log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    prices = S0 * np.exp(np.cumsum(log_r))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(),
                         periods=len(prices) + 5)[-len(prices):]
    return pd.Series(prices, index=idx, name="SYNTH")


def load_option_chain(ticker: str = "SPY", n_expiries: int = 1) -> pd.DataFrame:
    """Best-effort real option chain via yfinance. Empty df if it fails."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = tk.options[:n_expiries]
        rows = []
        for exp in exps:
            opt = tk.option_chain(exp)
            for kind, df in [("call", opt.calls), ("put", opt.puts)]:
                for _, row in df.iterrows():
                    if row.get("bid", 0) > 0 and row.get("ask", 0) > 0:
                        mid = 0.5 * (row["bid"] + row["ask"])
                        T = (pd.to_datetime(exp) - pd.Timestamp.today()).days / 365.0
                        if T <= 0:
                            continue
                        rows.append({"strike": row["strike"], "T": T,
                                    "mid_price": mid, "option_type": kind})
        return pd.DataFrame(rows)
    except Exception as e:  # noqa: BLE001
        print(f"[data_loader] option chain failed ({e}); returning empty.")
        return pd.DataFrame(columns=["strike", "T", "mid_price", "option_type"])
