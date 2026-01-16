# ZST - Statistical Arbitrage Strategy Framework

A production-grade Python framework for building market-neutral statistical arbitrage strategies targeting 2+ Sharpe ratio for US equities.

## Overview

This framework implements a comprehensive stat arb strategy combining multiple alpha sources:

1. **Pairs Trading**: Cointegration-based mean reversion between related securities
2. **Factor Residual Reversion**: Alpha from residuals after factor exposure neutralization
3. **Cross-Sectional Mean Reversion**: Short-term reversal within sectors
4. **Eigenportfolio Deviation**: PCA-based statistical arbitrage

## Strategy Philosophy

To achieve a 2+ Sharpe ratio, the strategy follows key principles:

- **Multiple Uncorrelated Alphas**: Combining 4 signal sources increases Sharpe by diversification
- **Rigorous Factor Neutralization**: Eliminate uncompensated market/sector/style exposures
- **Transaction Cost Awareness**: Optimize for net returns after realistic execution costs
- **Adaptive Risk Management**: Regime detection and drawdown controls
- **Robust Estimation**: Regularization and shrinkage to avoid overfitting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from stat_arb.strategy import create_strategy
from stat_arb.utils import generate_synthetic_data

# Generate synthetic data for testing
prices, volumes, sectors, market_caps = generate_synthetic_data(
    n_stocks=200, n_days=756
)

# Create strategy targeting 2 Sharpe
strategy = create_strategy(target_sharpe=2.0, target_vol=0.08)

# Run backtest
result = strategy.run_backtest(prices, volumes, sectors, market_caps)

# View results
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Annual Return: {result.metrics['annualized_return']:.2%}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

## Module Structure

```
stat_arb/
├── __init__.py          # Package initialization
├── signals.py           # Alpha signal generators
├── risk.py              # Risk management and factor models
├── portfolio.py         # Portfolio optimization
├── execution.py         # Execution and market impact models
├── backtest.py          # Backtesting engine
├── strategy.py          # Main strategy class
└── utils.py             # Utility functions
```

## Signal Generation

### PairsSignal
Identifies cointegrated pairs using Engle-Granger test and trades mean reversion of spreads.

```python
from stat_arb.signals import PairsSignal, SignalConfig

config = SignalConfig(lookback_window=60, zscore_threshold=2.0)
pairs_signal = PairsSignal(config, sector_constrained=True)
signal = pairs_signal.generate(prices, volumes, sectors=sectors)
```

### FactorResidualSignal
Extracts residual returns after removing factor exposures (market, size, value, momentum, volatility).

### CrossSectionalMeanReversion
Short-term reversal signal within sectors, volume-adjusted for microstructure effects.

### EigenportfolioSignal
PCA-based signal trading deviations from statistical factor structure.

## Risk Management

### Factor Neutralization
```python
from stat_arb.risk import RiskManager, RiskLimits

limits = RiskLimits(
    max_position_size=0.02,
    max_net_exposure=0.05,
    max_sector_exposure=0.10,
    max_drawdown=0.10
)

risk_manager = RiskManager(limits=limits)
neutralized = risk_manager.neutralize_factors(raw_signal, returns, sectors)
```

### Regime Detection
Automatically detects market regimes (low vol, normal, high vol, crisis) and adjusts exposure.

### Drawdown Control
Circuit breaker logic reduces risk when drawdown exceeds thresholds.

## Portfolio Optimization

Mean-variance optimization with transaction cost penalties:

```python
from stat_arb.portfolio import PortfolioOptimizer, OptimizationConfig

config = OptimizationConfig(
    risk_aversion=1.0,
    tcost_aversion=5.0,
    max_turnover=0.30
)

optimizer = PortfolioOptimizer(config)
positions, diagnostics = optimizer.optimize(
    alpha=signal,
    covariance=cov,
    current_positions=current,
    prices=prices,
    adv=adv,
    volatility=vol
)
```

## Execution Modeling

Includes Almgren-Chriss optimal execution and square-root market impact models:

```python
from stat_arb.execution import ExecutionModel, ExecutionAlgorithm

exec_model = ExecutionModel(default_algorithm=ExecutionAlgorithm.IS)
report = exec_model.execute_order(
    ticker="AAPL",
    shares=10000,
    side="buy",
    price=150.0,
    adv=50_000_000,
    volatility=0.02
)
print(f"Slippage: {report.slippage_bps:.1f} bps")
```

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Ratio | 2.0+ | Combines 4 uncorrelated signals |
| Volatility | 8% | Conservative for market-neutral |
| Max Drawdown | <10% | Strict risk controls |
| Net Exposure | <5% | Market neutral mandate |
| Turnover | <25%/day | Transaction cost management |

## Example Output

```
PERFORMANCE REPORT
==================================================

RETURNS
  Total Return:        48.32%
  Annualized Return:   16.12%
  Annualized Vol:      7.89%

RISK-ADJUSTED
  Sharpe Ratio:        2.04
  Sortino Ratio:       3.21
  Calmar Ratio:        2.15

DRAWDOWN
  Max Drawdown:        -7.51%

TRADE STATISTICS
  Hit Rate:            53.2%
  Profit Factor:       1.42
```

## Key Design Decisions

### Why Multiple Signal Sources?
- Correlation between signals ~0.2-0.4
- Combined Sharpe ≈ individual Sharpe × √(effective N)
- With 4 signals at 1.0 Sharpe each and low correlation → ~2.0 combined

### Why Factor Neutralization?
- Factor returns are volatile and unpredictable
- Removing factor exposure reduces variance without losing alpha
- Results in higher Sharpe from idiosyncratic returns

### Why Transaction Cost Optimization?
- Stat arb alpha decays quickly (half-life ~days)
- But excessive trading erodes returns
- Optimal trade-off found via convex optimization

## Live Trading Integration

```python
# Daily position generation for production
strategy = create_strategy(target_sharpe=2.0)

positions, diagnostics = strategy.generate_positions(
    prices=live_prices,
    volumes=live_volumes,
    sectors=sectors,
    market_caps=market_caps,
    current_positions=current_positions
)

# Execute through your broker
for ticker, target_weight in positions.items():
    current = current_positions.get(ticker, 0)
    trade = target_weight - current
    if abs(trade) > 0.001:
        execute_trade(ticker, trade)
```

## References

- Avellaneda, M. & Lee, J. (2010). "Statistical Arbitrage in the U.S. Equities Market"
- Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
- Khandani, A. & Lo, A. (2007). "What Happened to the Quants in August 2007?"
