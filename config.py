"""Project-wide configuration."""
from __future__ import annotations

# market
TICKER = "SPY"
RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.013

# LSTM
LOOKBACK = 21
HORIZON = 5
LSTM_UNITS = 32
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 64
LSTM_VAL_FRAC = 0.2

# MC convergence
MC_PATH_COUNTS = (1_000, 5_000, 25_000, 100_000, 500_000)
MC_COVERAGE_TRIALS = 200

# greeks validation
GREEKS_TOLERANCE = 1e-3

# IO
OUTPUT_DIR = "outputs"
