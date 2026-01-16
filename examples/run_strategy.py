#!/usr/bin/env python3
"""
Statistical Arbitrage Strategy - Example Usage

This script demonstrates how to use the stat arb strategy framework
to backtest and analyze a market-neutral strategy targeting 2+ Sharpe.

Usage:
    python examples/run_strategy.py

The example uses synthetic data but the same code works with real
market data from any data provider.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from stat_arb.strategy import StatArbStrategy, StrategyConfig, create_strategy
from stat_arb.utils import (
    generate_synthetic_data,
    calculate_performance_metrics,
    format_performance_report,
    bootstrap_sharpe_confidence,
    rolling_sharpe,
)


def main():
    """Run full strategy demonstration."""
    print("=" * 60)
    print("STATISTICAL ARBITRAGE STRATEGY - 2 SHARPE TARGET")
    print("=" * 60)
    print()

    # =========================================================================
    # 1. GENERATE SYNTHETIC DATA
    # =========================================================================
    print("1. Generating synthetic market data...")
    print("   - 200 stocks, 3 years of history")
    print("   - Factor structure with mean-reverting residuals")
    print()

    prices, volumes, sectors, market_caps = generate_synthetic_data(
        n_stocks=200,
        n_days=756,  # 3 years
        n_sectors=11,
        seed=42,
    )

    print(f"   Data generated:")
    print(f"   - Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"   - Universe size: {len(prices.columns)} stocks")
    print(f"   - Sectors: {sectors.nunique()}")
    print()

    # =========================================================================
    # 2. INITIALIZE STRATEGY
    # =========================================================================
    print("2. Initializing stat arb strategy...")
    print()

    config = StrategyConfig(
        target_volatility=0.08,  # 8% target vol
        max_gross_leverage=2.0,
        max_net_exposure=0.05,
        max_position_size=0.02,
        max_sector_exposure=0.10,
        max_drawdown=0.10,
        max_daily_turnover=0.25,
        min_market_cap=1e9,
        min_adv=5e6,
    )

    strategy = StatArbStrategy(config)

    print("   Strategy Configuration:")
    print(f"   - Target Volatility: {config.target_volatility:.1%}")
    print(f"   - Max Gross Leverage: {config.max_gross_leverage:.1f}x")
    print(f"   - Max Net Exposure: {config.max_net_exposure:.1%}")
    print(f"   - Max Position Size: {config.max_position_size:.1%}")
    print(f"   - Max Sector Exposure: {config.max_sector_exposure:.1%}")
    print()

    # =========================================================================
    # 3. RUN BACKTEST
    # =========================================================================
    print("3. Running backtest...")
    print()

    result = strategy.run_backtest(
        prices=prices,
        volumes=volumes,
        sectors=sectors,
        market_caps=market_caps,
    )

    print()
    print("   Backtest completed.")
    print()

    # =========================================================================
    # 4. ANALYZE RESULTS
    # =========================================================================
    print("4. Analyzing results...")
    print()

    # Calculate metrics
    metrics = calculate_performance_metrics(result.returns)

    # Print performance report
    print(format_performance_report(metrics))
    print()

    # =========================================================================
    # 5. SHARPE RATIO ANALYSIS
    # =========================================================================
    print("5. Sharpe Ratio Analysis")
    print("-" * 40)

    sharpe, lower, upper = bootstrap_sharpe_confidence(
        result.returns, n_bootstrap=1000, confidence=0.95
    )

    print(f"   Point Estimate:     {sharpe:.2f}")
    print(f"   95% CI:             [{lower:.2f}, {upper:.2f}]")
    print()

    # Rolling Sharpe
    rolling = rolling_sharpe(result.returns, window=126)  # 6-month rolling
    print(f"   Rolling Sharpe (6M):")
    print(f"   - Mean:             {rolling.mean():.2f}")
    print(f"   - Min:              {rolling.min():.2f}")
    print(f"   - Max:              {rolling.max():.2f}")
    print(f"   - Current:          {rolling.iloc[-1]:.2f}")
    print()

    # =========================================================================
    # 6. EXPOSURE ANALYSIS
    # =========================================================================
    print("6. Exposure Analysis")
    print("-" * 40)

    if len(result.exposure_history) > 0:
        exp = result.exposure_history

        print(f"   Gross Exposure:")
        print(f"   - Mean:             {exp['gross'].mean():.2f}x")
        print(f"   - Max:              {exp['gross'].max():.2f}x")
        print()

        print(f"   Net Exposure:")
        print(f"   - Mean:             {exp['net'].mean():.2%}")
        print(f"   - Std:              {exp['net'].std():.2%}")
        print()

        print(f"   Long/Short Split:")
        print(f"   - Avg Long:         {exp['long'].mean():.2f}x")
        print(f"   - Avg Short:        {exp['short'].mean():.2f}x")
        print()

    # =========================================================================
    # 7. DRAWDOWN ANALYSIS
    # =========================================================================
    print("7. Drawdown Analysis")
    print("-" * 40)

    dd = result.drawdown_series

    print(f"   Maximum Drawdown:   {dd.min():.2%}")

    # Time underwater
    underwater = (dd < 0).sum() / len(dd) * 100
    print(f"   Time Underwater:    {underwater:.1f}%")

    # Recovery analysis
    cum_ret = (1 + result.returns).cumprod()
    at_highs = (cum_ret == cum_ret.expanding().max()).sum() / len(cum_ret) * 100
    print(f"   Time at Highs:      {at_highs:.1f}%")
    print()

    # =========================================================================
    # 8. MONTHLY RETURNS TABLE
    # =========================================================================
    print("8. Monthly Returns Summary")
    print("-" * 40)

    # Resample to monthly
    monthly_returns = result.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    # Create year-month pivot
    monthly_df = pd.DataFrame(
        {
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        }
    )

    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    print("   Monthly Returns (%):")
    print(
        pivot.applymap(lambda x: f"{x*100:.1f}" if pd.notna(x) else "").to_string()
    )
    print()

    # =========================================================================
    # 9. KEY INSIGHTS
    # =========================================================================
    print("9. Key Insights")
    print("-" * 40)
    print()

    sharpe_achieved = metrics.get("sharpe_ratio", 0)
    vol_achieved = metrics.get("annualized_volatility", 0)
    max_dd = metrics.get("max_drawdown", 0)

    if sharpe_achieved >= 2.0:
        print("   [✓] TARGET ACHIEVED: Sharpe ratio >= 2.0")
    else:
        print(f"   [!] Sharpe ratio ({sharpe_achieved:.2f}) below 2.0 target")

    if abs(vol_achieved - 0.08) < 0.02:
        print("   [✓] Volatility within target range (8% ± 2%)")
    else:
        print(f"   [!] Volatility ({vol_achieved:.1%}) outside target range")

    if abs(max_dd) < config.max_drawdown:
        print(f"   [✓] Max drawdown ({abs(max_dd):.1%}) within limit")
    else:
        print(f"   [!] Max drawdown ({abs(max_dd):.1%}) exceeded limit")

    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

    return result, metrics


def run_live_signal_generation():
    """Demonstrate live signal generation (as would be used in production)."""
    print()
    print("=" * 60)
    print("LIVE SIGNAL GENERATION DEMO")
    print("=" * 60)
    print()

    # Generate data
    prices, volumes, sectors, market_caps = generate_synthetic_data(
        n_stocks=100, n_days=300, seed=123
    )

    # Initialize strategy
    strategy = create_strategy(target_sharpe=2.0, target_vol=0.08)

    # Generate positions
    print("Generating target positions...")

    positions, diagnostics = strategy.generate_positions(
        prices=prices,
        volumes=volumes,
        sectors=sectors,
        market_caps=market_caps,
    )

    print()
    print("Position Summary:")
    print(f"  Universe Size: {diagnostics['universe_size']}")
    print(f"  Current Regime: {diagnostics['regime']}")
    print(f"  Portfolio Vol: {diagnostics['portfolio_vol']:.2%}")
    print()

    print("Top 10 Long Positions:")
    top_long = positions.nlargest(10)
    for ticker, weight in top_long.items():
        print(f"  {ticker}: {weight:.2%}")

    print()
    print("Top 10 Short Positions:")
    top_short = positions.nsmallest(10)
    for ticker, weight in top_short.items():
        print(f"  {ticker}: {weight:.2%}")

    print()
    print("Signal Diagnostics:")
    for signal_name, stats in diagnostics["signal_stats"].items():
        print(f"  {signal_name}:")
        print(f"    Sharpe: {stats['sharpe']:.2f}, Weight: {stats['weight']:.2%}")

    return positions, diagnostics


if __name__ == "__main__":
    # Run main backtest
    result, metrics = main()

    # Run live signal generation demo
    positions, diagnostics = run_live_signal_generation()
