# Option Pricing & Volatility Model Validation

A model-validation project, not just a pricing library: every model here is
checked against an independent benchmark, and the checks are the deliverable
as much as the models are.

| Model                     | Validated against                                   |
| -------------------------- | ---------------------------------------------------- |
| Monte Carlo pricer          | Black-Scholes closed form (convergence + coverage)   |
| Analytical Greeks           | Finite-difference Greeks (same pricer, independent method) |
| Implied-vol solver          | Round-trip: price at known σ → solve → recover σ     |
| Volatility smile fit        | Parameter recovery on a smile with known ground truth |
| LSTM vol forecaster         | EWMA, GARCH(1,1), and a formal Diebold-Mariano test  |
| All of the above            | Stress-tested at numerical extremes and market scenarios |

## Project structure

```
option_validation_project/
├── main.py                            end-to-end runner; produces the full report
├── config.py
├── requirements.txt
├── src/
│   ├── pricing/
│   │   ├── black_scholes.py           closed-form reference price
│   │   ├── greeks.py                  analytical AND finite-difference Greeks
│   │   └── monte_carlo.py             GBM Monte Carlo pricer
│   ├── volatility/
│   │   ├── implied_vol.py             Newton-Raphson + bisection IV solver
│   │   ├── historical_vol.py          realized / EWMA / GARCH baselines
│   │   ├── lstm_vol.py                Keras LSTM vol forecaster
│   │   └── smile.py                   smile construction + quadratic fit
│   ├── validation/
│   │   ├── convergence_tests.py       MC-vs-BS convergence & coverage test
│   │   ├── greeks_validation.py       analytical-vs-FD Greeks grid
│   │   ├── benchmark_comparison.py    MAE / RMSE / QLIKE + Diebold-Mariano
│   │   └── stress_testing.py          numerical edge cases + market scenarios
│   └── data/
│       └── data_loader.py             yfinance with synthetic fallback
└── outputs/                           created at runtime
```

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

TensorFlow and `arch` are the two heavy/optional deps. If either is missing,
`main.py` prints a note and skips just that part of section 6 — everything
else still runs.

## What each output file is

| File                        | What it shows                                              |
| ---------------------------- | ------------------------------------------------------------ |
| `convergence_table.csv`/`.png` | MC price vs BS as path count grows; error shrinks ~1/√n     |
| `greeks_validation.csv`/`_summary.csv` | Analytical vs finite-diff Greeks across a moneyness×maturity grid |
| `iv_roundtrip.csv`           | Recovered vol vs the true vol used to generate the price    |
| `smile.png`/`smile_fit.csv`  | Observed smile, fitted quadratic curve, R²                  |
| `vol_benchmark.csv`/`.png`   | LSTM/EWMA/GARCH forecast accuracy (MAE, RMSE, QLIKE)         |
| `dm_test.csv`                | Diebold-Mariano statistic + p-value, LSTM vs EWMA/GARCH      |
| `edge_cases.csv`/`_summary.csv` | Pricer/Greeks/IV behavior at extreme moneyness, near-zero T, extreme vol |
| `scenarios.csv`              | Portfolio-level reprice under named market shocks            |

## A real bug this project caught

The implied-vol solver originally rejected some perfectly valid low-vol
option prices as "sub-intrinsic" and returned `NaN`. The lower-arbitrage-
bound check was comparing the price against undiscounted intrinsic value
(`S - K`), but for a long-dated, dividend-paying option the correct bound is
the **discounted forward intrinsic**, `S·e^(-qT) - K·e^(-rT)`. At 5-year
maturity with a ~1.3% dividend yield, the two bounds differ by enough to
flip the comparison. Fixed in `implied_vol.py`; the edge-case grid
(`edge_cases.csv`) is what surfaced it — a 5-year, deep-ITM call was the one
case out of 175 that failed before the fix.

This is left in as documentation of the actual validation process, not
tidied away, because catching this kind of thing is the point of the
project.

## Design notes

**Why analytical and finite-difference Greeks are two separate
implementations.** `finite_diff_greeks` bumps-and-reprices using the *same*
`bs_price` function as `analytical_greeks` derives from analytically. Any
discrepancy you see is purely the finite-difference approximation error —
not a difference in pricing model — which is what makes it a clean
validation check rather than a comparison of two different things.

**Why the edge-case grid separates "solver failed" from "vega too small to
identify vol at all."** Deep ITM/OTM options near expiry have vega
collapsing towards zero; the price becomes almost pure intrinsic value and
genuinely can't tell you what sigma was used to generate it, regardless of
solver quality. Lumping those in with real solver bugs would misdiagnose an
economic limitation as a code defect — a mistake an actual validation
report shouldn't make. `n_iv_ill_posed_low_vega` in the summary counts
these separately from `n_failures`.

**Why QLIKE and a Diebold-Mariano test, not just "LSTM has lower MAE."**
QLIKE (Patton, 2011) penalizes under-forecasting variance more than squared
error does, which matters for anything downstream that uses the forecast to
size a hedge or a margin requirement. The DM test then asks the actual
question a validator needs answered: is the accuracy difference
statistically significant, or within noise? "Lower MAE" alone is not
evidence of that.

**Why the smile fit is validated by parameter recovery.** `synthetic_smile`
generates a smile from known (a, b, c). Fitting it back and checking the
recovered parameters against the true ones (see `main.py` section 5, or run
`fit_smile` on `synthetic_smile` output) is a cleaner check than fitting a
real smile and eyeballing the curve — you know the right answer going in.

## Extending

* Swap the quadratic smile fit for SVI or SABR and re-run parameter recovery
  on synthetic data first, before trusting it on a real chain.
* Add a PIT (probability-integral-transform) histogram for the LSTM
  forecaster to check calibration, not just point-forecast accuracy.
* Extend `edge_case_grid` with American-style early-exercise boundary checks
  if you add an American pricer.
* Backtest the smile fit walk-forward across historical chains instead of
  a single-snapshot fit.
