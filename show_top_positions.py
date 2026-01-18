#!/usr/bin/env python3
"""
Show Top 10 Stat Arb Positions

Usage:
    python show_top_positions.py                    # Use synthetic data
    python show_top_positions.py --real             # Use real market data (S&P 500)
    python show_top_positions.py --real --stocks 50 # Real data with 50 stocks
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from stat_arb.strategy import StatArbStrategy, StrategyConfig, create_strategy
from stat_arb.utils import generate_synthetic_data
from stat_arb.market_data import fetch_market_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Show top 10 stat arb positions')
parser.add_argument('--real', action='store_true', help='Use real market data from Yahoo Finance')
parser.add_argument('--stocks', type=int, default=100, help='Number of stocks to include (default: 100)')
parser.add_argument('--days', type=int, default=500, help='Days of historical data (default: 500)')
args = parser.parse_args()

print("=" * 70)
print("STATISTICAL ARBITRAGE PORTFOLIO - TOP 10 POSITIONS")
print("=" * 70)
print()

# Get market data
if args.real:
    print("Fetching REAL market data from Yahoo Finance...")
    print(f"  Stocks: {args.stocks}")
    print(f"  History: {args.days} days")
    print()
    try:
        prices, volumes, sectors, market_caps = fetch_market_data(
            n_stocks=args.stocks,
            lookback_days=args.days,
        )
    except Exception as e:
        print(f"ERROR: Failed to fetch real market data: {e}")
        print("Falling back to synthetic data...")
        prices, volumes, sectors, market_caps = generate_synthetic_data(
            n_stocks=200, n_days=300, seed=123
        )
else:
    print("Using SYNTHETIC market data...")
    prices, volumes, sectors, market_caps = generate_synthetic_data(
        n_stocks=200, n_days=300, seed=123
    )
    print(f"  Data range: {prices.index[0]} to {prices.index[-1]}")
    print(f"  Universe: {len(prices.columns)} stocks")

print()

# Initialize strategy
print("Initializing statistical arbitrage strategy...")
strategy = create_strategy(target_sharpe=2.0, target_vol=0.08)
print()

# Generate positions
print("Generating target positions...")
print()

positions, diagnostics = strategy.generate_positions(
    prices=prices,
    volumes=volumes,
    sectors=sectors,
    market_caps=market_caps,
)

# Display portfolio summary
print("PORTFOLIO SUMMARY")
print("-" * 70)
print(f"  Universe Size:        {diagnostics['universe_size']}")
print(f"  Current Regime:       {diagnostics['regime']}")
print(f"  Portfolio Volatility: {diagnostics['portfolio_vol']:.2%}")
print(f"  Gross Exposure:       {diagnostics['optimization']['gross_exposure']:.2f}x")
print(f"  Net Exposure:         {diagnostics['optimization']['net_exposure']:.2%}")
print(f"  Long Positions:       {diagnostics['optimization']['n_long_positions']}")
print(f"  Short Positions:      {diagnostics['optimization']['n_short_positions']}")
print()

# Top 10 Long Positions
print("TOP 10 LONG POSITIONS")
print("-" * 70)
top_long = positions.nlargest(10)
for i, (ticker, weight) in enumerate(top_long.items(), 1):
    print(f"  {i:2d}. {ticker:8s}  {weight:8.2%}")
print()

# Top 10 Short Positions
print("TOP 10 SHORT POSITIONS")
print("-" * 70)
top_short = positions.nsmallest(10)
for i, (ticker, weight) in enumerate(top_short.items(), 1):
    print(f"  {i:2d}. {ticker:8s}  {weight:8.2%}")
print()

# Signal diagnostics
print("SIGNAL DIAGNOSTICS")
print("-" * 70)
for signal_name, stats in diagnostics["signal_stats"].items():
    print(f"  {signal_name:30s}  Sharpe: {stats['sharpe']:5.2f}  Weight: {stats['weight']:6.2%}")
print()

print("=" * 70)
print("COMPLETE")
print("=" * 70)
