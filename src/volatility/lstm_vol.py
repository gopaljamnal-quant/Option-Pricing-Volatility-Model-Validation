"""
LSTM-based volatility forecaster.

Predicts forward realized volatility over a fixed horizon from a rolling
window of past (log-return, |log-return|) pairs. Trained in log-vol space
so predictions are always positive and the loss surface is well behaved.

This module produces *only* the forecast. All comparison against GARCH /
EWMA / realized-vol baselines -- error metrics, statistical tests, plots --
lives in `validation/benchmark_comparison.py`, because that separation is
the point of a validation project: the model and its evaluation are
independent pieces of code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .historical_vol import log_returns, TRADING_DAYS


def build_features(prices: pd.Series, lookback: int = 21,
                   horizon: int = 5) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """X[i]: last `lookback` (log-return, |log-return|) pairs ending at day i-1.
    y[i]: forward realized vol over the next `horizon` days, annualized, log-space."""
    r = log_returns(prices)
    abs_r = r.abs()

    fwd_var = (r ** 2).rolling(horizon).sum().shift(-horizon) * (TRADING_DAYS / horizon)
    fwd_vol = np.sqrt(fwd_var)
    target_log = np.log(fwd_vol.replace(0, np.nan)).dropna()

    X_list, y_list, date_list = [], [], []
    for t in range(lookback, len(r)):
        date = r.index[t]
        if date not in target_log.index:
            continue
        window = np.column_stack([
            r.iloc[t - lookback: t].values,
            abs_r.iloc[t - lookback: t].values,
        ])
        X_list.append(window)
        y_list.append(target_log.loc[date])
        date_list.append(date)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y, pd.DatetimeIndex(date_list)


@dataclass
class LSTMVolForecaster:
    lookback: int = 21
    horizon: int = 5
    units: int = 32
    epochs: int = 30
    batch_size: int = 64
    val_frac: float = 0.2

    model: object | None = field(default=None, init=False, repr=False)
    history: dict | None = field(default=None, init=False, repr=False)

    def _build(self, n_features: int):
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as e:
            raise ImportError("install tensorflow (>=2.10) to use the LSTM forecaster") from e

        model = keras.Sequential([
            layers.Input(shape=(self.lookback, n_features)),
            layers.LSTM(self.units, return_sequences=False),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation=None),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, prices: pd.Series, verbose: int = 0) -> "LSTMVolForecaster":
        X, y, dates = build_features(prices, self.lookback, self.horizon)
        split = int(len(X) * (1 - self.val_frac))  # chronological split, never shuffle
        X_tr, X_va = X[:split], X[split:]
        y_tr, y_va = y[:split], y[split:]

        self.model = self._build(X.shape[2])
        hist = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va) if len(X_va) else None,
            epochs=self.epochs, batch_size=self.batch_size, verbose=verbose,
        )
        self.history = hist.history
        self._val_dates = dates[split:]
        self._X_va, self._y_va = X_va, y_va
        return self

    def predict(self, prices: pd.Series) -> pd.Series:
        if self.model is None:
            raise RuntimeError("call .fit() first")
        X, _, dates = build_features(prices, self.lookback, self.horizon)
        log_vol = self.model.predict(X, verbose=0).ravel()
        return pd.Series(np.exp(log_vol), index=dates, name="lstm_vol")

    def validation_frame(self) -> pd.DataFrame:
        """Predictions vs actuals on the held-out (chronologically last) slice."""
        if self.model is None:
            raise RuntimeError("call .fit() first")
        pred_log = self.model.predict(self._X_va, verbose=0).ravel()
        return pd.DataFrame({
            "actual": np.exp(self._y_va.ravel()),
            "lstm":   np.exp(pred_log),
        }, index=self._val_dates)
