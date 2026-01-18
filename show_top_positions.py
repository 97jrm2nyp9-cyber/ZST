#!/usr/bin/env python3
"""
Show Top 10 Stat Arb Positions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from stat_arb.strategy import StatArbStrategy, StrategyConfig, create_strategy
from stat_arb.utils import generate_synthetic_data

print("=" * 70)
print("STATISTICAL ARBITRAGE PORTFOLIO - TOP 10 POSITIONS")
print("=" * 70)
print()

# Generate data
print("Generating market data...")
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
