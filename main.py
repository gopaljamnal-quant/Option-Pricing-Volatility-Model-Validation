"""
Option Pricing & Volatility Model Validation -- end-to-end runner.

Run as:
    python main.py

Produces a full validation report under ./outputs/:
    - convergence.png / convergence_table.csv    MC vs BS convergence + coverage test
    - greeks_validation.csv / greeks_summary.csv  analytical vs finite-difference Greeks
    - iv_roundtrip.csv                            implied-vol solver round-trip test
    - smile.png / smile_fit.csv                   volatility smile + parametric fit
    - vol_benchmark.csv / vol_benchmark.png        LSTM vs EWMA vs GARCH accuracy
    - dm_test.csv                                  Diebold-Mariano significance test
    - edge_cases.csv / edge_case_summary.csv       numerical stability stress grid
    - scenarios.csv                                market scenario stress test
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # Forces Matplotlib to use the standard GUI backend
import matplotlib.pyplot as plt


import config as cfg
from src.data.data_loader import load_prices
from src.pricing.black_scholes import bs_price
from src.pricing.monte_carlo import mc_european_price
from src.volatility.historical_vol import realized_vol, ewma_vol, garch_vol
from src.volatility.implied_vol import implied_vol
from src.volatility.smile import synthetic_smile, fit_smile, plot_smile
from src.volatility.lstm_vol import LSTMVolForecaster
from src.validation.convergence_tests import convergence_table, coverage_test, plot_convergence
from src.validation.greeks_validation import validate_greeks_grid, summarize_validation
from src.validation.benchmark_comparison import error_summary, diebold_mariano
from src.validation.stress_testing import edge_case_grid, edge_case_summary, run_scenarios, check_monotonicity


def section(title: str):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    section("1) load market data")
    # ------------------------------------------------------------------
    prices = load_prices(cfg.TICKER)
    S0 = float(prices.iloc[-1])
    print(f"{cfg.TICKER}: {len(prices)} obs, {prices.index.min().date()} -> "
          f"{prices.index.max().date()}, spot={S0:.2f}")

    rv = realized_vol(prices, window=21)
    sigma = float(rv.iloc[-1])
    r, q = cfg.RISK_FREE_RATE, cfg.DIVIDEND_YIELD
    K, T = round(S0), 0.5
    print(f"realized vol (21d): {sigma:.3%}  |  pricing K={K}, T={T}")

    # ------------------------------------------------------------------
    section("2) Monte Carlo convergence validation (vs Black-Scholes)")
    # ------------------------------------------------------------------
    conv_df = convergence_table(S0, K, T, r, q, sigma, "call",
                                path_counts=cfg.MC_PATH_COUNTS, seed=0)
    print(conv_df.round(5).to_string(index=False))
    conv_df.to_csv(f"{cfg.OUTPUT_DIR}/convergence_table.csv", index=False)

    cov = coverage_test(S0, K, T, r, q, sigma, "call",
                        n_paths=20_000, n_trials=cfg.MC_COVERAGE_TRIALS, alpha=0.95)
    print(f"\ncoverage test: nominal={cov['nominal_coverage']:.0%}  "
          f"empirical={cov['empirical_coverage']:.1%}  "
          f"(over {cov['n_trials']} trials)")

    fig = plot_convergence(conv_df, savepath=f"{cfg.OUTPUT_DIR}/convergence.png")
    plt.close(fig)
    print(f"saved {cfg.OUTPUT_DIR}/convergence.png, convergence_table.csv")

    # ------------------------------------------------------------------
    section("3) Greeks validation (analytical vs finite-difference)")
    # ------------------------------------------------------------------
    greeks_grid = validate_greeks_grid(S0, r, q, sigma, tol=cfg.GREEKS_TOLERANCE)
    greeks_summary = summarize_validation(greeks_grid)
    print(greeks_summary.round(6).to_string(index=False))
    greeks_grid.to_csv(f"{cfg.OUTPUT_DIR}/greeks_validation.csv", index=False)
    greeks_summary.to_csv(f"{cfg.OUTPUT_DIR}/greeks_summary.csv", index=False)
    print(f"saved {cfg.OUTPUT_DIR}/greeks_validation.csv, greeks_summary.csv")

    # ------------------------------------------------------------------
    section("4) implied volatility round-trip test")
    # ------------------------------------------------------------------
    test_vols = np.array([0.05, 0.10, 0.20, 0.35, 0.60, 1.00])
    rows = []
    for sv in test_vols:
        p = float(bs_price(S0, K, T, r, q, sv, "call"))
        recovered = implied_vol(p, S0, K, T, r, q, "call")
        rows.append({"true_vol": sv, "price": p, "recovered_vol": recovered,
                    "abs_error": abs(recovered - sv)})
    iv_df = pd.DataFrame(rows)
    print(iv_df.round(6).to_string(index=False))
    iv_df.to_csv(f"{cfg.OUTPUT_DIR}/iv_roundtrip.csv", index=False)
    print(f"saved {cfg.OUTPUT_DIR}/iv_roundtrip.csv")

    # ------------------------------------------------------------------
    section("5) volatility smile: construction + parametric fit")
    # ------------------------------------------------------------------
    smile_df = synthetic_smile(S=S0, T=0.25, a=sigma, b=-0.12, c=0.35, noise_std=0.004, seed=1)
    fit = fit_smile(smile_df)
    print(f"fit params: level(a)={fit['a_level']:.4f}  skew(b)={fit['b_skew']:.4f}  "
          f"curvature(c)={fit['c_curvature']:.4f}")
    print(f"fit quality: RMSE={fit['rmse']:.5f}  R^2={fit['r_squared']:.4f}")
    pd.DataFrame([fit]).to_csv(f"{cfg.OUTPUT_DIR}/smile_fit.csv", index=False)
    fig = plot_smile(smile_df, fit, title=f"{cfg.TICKER} volatility smile (T=0.25)",
                     savepath=f"{cfg.OUTPUT_DIR}/smile.png")
    plt.close(fig)
    print(f"saved {cfg.OUTPUT_DIR}/smile.png, smile_fit.csv")

    # ------------------------------------------------------------------
    section("6) LSTM volatility forecast vs EWMA / GARCH benchmark")
    # ------------------------------------------------------------------
    ew = ewma_vol(prices, lam=0.94)
    try:
        gv = garch_vol(prices)
        garch_ok = True
    except Exception as e:
        print(f"GARCH unavailable: {e}")
        gv, garch_ok = None, False

    try:
        forecaster = LSTMVolForecaster(
            lookback=cfg.LOOKBACK, horizon=cfg.HORIZON, units=cfg.LSTM_UNITS,
            epochs=cfg.LSTM_EPOCHS, batch_size=cfg.LSTM_BATCH_SIZE,
            val_frac=cfg.LSTM_VAL_FRAC,
        ).fit(prices, verbose=0)
        val_frame = forecaster.validation_frame()

        forecasts = {"EWMA": ew.reindex(val_frame.index)}
        if garch_ok:
            forecasts["GARCH"] = gv.reindex(val_frame.index)
        forecasts["LSTM"] = val_frame["lstm"]

        bench_df = error_summary(val_frame["actual"], forecasts)
        print(bench_df.round(5).to_string(index=False))
        bench_df.to_csv(f"{cfg.OUTPUT_DIR}/vol_benchmark.csv", index=False)

        dm_rows = []
        baseline_name = "EWMA"
        baseline_series = ew.reindex(val_frame.index)
        joined_base = pd.concat([val_frame["actual"], baseline_series.rename("f")], axis=1).dropna()
        for name, series in [("LSTM", val_frame["lstm"])] + ([("GARCH", gv.reindex(val_frame.index))] if garch_ok else []):
            dm = diebold_mariano(val_frame["actual"], series, baseline_series, h=cfg.HORIZON)
            dm_rows.append({"comparison": f"{name}_vs_{baseline_name}", **dm})
        dm_df = pd.DataFrame(dm_rows)
        print("\nDiebold-Mariano test (H0: equal forecast accuracy):")
        print(dm_df[["comparison", "dm_statistic", "p_value", "conclusion"]].to_string(index=False))
        dm_df.to_csv(f"{cfg.OUTPUT_DIR}/dm_test.csv", index=False)

        fig, ax = plt.subplots(figsize=(11, 5))
        val_frame["actual"].plot(ax=ax, label="Realized (target)", lw=1.4, color="black")
        val_frame["lstm"].plot(ax=ax, label="LSTM", lw=1.4, color="crimson")
        ew.reindex(val_frame.index).plot(ax=ax, label="EWMA", lw=1.1, alpha=0.7)
        if garch_ok:
            gv.reindex(val_frame.index).plot(ax=ax, label="GARCH", lw=1.1, alpha=0.7)
        ax.set_title(f"Volatility forecast benchmark on held-out period ({cfg.TICKER})")
        ax.set_ylabel("Annualised vol")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{cfg.OUTPUT_DIR}/vol_benchmark.png", dpi=140)
        plt.close(fig)
        print(f"saved {cfg.OUTPUT_DIR}/vol_benchmark.csv, vol_benchmark.png, dm_test.csv")
    except ImportError as e:
        print(f"skipping LSTM benchmark section: {e}")

    # ------------------------------------------------------------------
    section("7) stress testing: numerical edge cases")
    # ------------------------------------------------------------------
    edge_df = edge_case_grid(S0=S0, r=r, q=q, option_type="call")
    edge_summary = edge_case_summary(edge_df)
    print(edge_summary.to_string())
    edge_df.to_csv(f"{cfg.OUTPUT_DIR}/edge_cases.csv", index=False)
    edge_summary.to_csv(f"{cfg.OUTPUT_DIR}/edge_case_summary.csv")
    if edge_summary["n_failures"] > 0:
        print(f"\n  {int(edge_summary['n_failures'])} edge case(s) failed -- see edge_cases.csv")
    print(f"saved {cfg.OUTPUT_DIR}/edge_cases.csv, edge_case_summary.csv")

    # ------------------------------------------------------------------
    section("8) stress testing: market scenarios")
    # ------------------------------------------------------------------
    scen_df = run_scenarios(S0, K, T, r, q, sigma, "call")
    print(scen_df.round(3).to_string(index=False))
    mono = check_monotonicity(scen_df, "call")
    print(f"\nmonotonicity check (price should rise with spot for a call): {mono}")
    scen_df.to_csv(f"{cfg.OUTPUT_DIR}/scenarios.csv", index=False)
    print(f"saved {cfg.OUTPUT_DIR}/scenarios.csv")

    section("done")
    print(f"all validation outputs written to ./{cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
