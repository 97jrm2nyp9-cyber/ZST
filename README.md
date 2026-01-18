# ZST - Statistical Arbitrage Trading Strategy

A production-ready statistical arbitrage trading strategy targeting 2+ Sharpe ratio.

## Features

- **Multiple Alpha Sources**: Combines 4 uncorrelated signal generators
  - Pairs trading (cointegration-based)
  - Factor residual mean reversion
  - Cross-sectional mean reversion
  - Eigenportfolio deviation trading

- **Risk Management**: Comprehensive risk controls
  - Factor neutralization
  - Regime detection and adaptation
  - Position limits and exposure constraints
  - Drawdown monitoring

- **Real Market Data**: Fetch live data from Yahoo Finance
  - Supports custom ticker lists
  - Default large-cap US stock universe
  - Automatic data validation and cleaning

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Show Top 10 Positions (Synthetic Data)

```bash
python show_top_positions.py
```

### Show Top 10 Positions (Real Market Data)

```bash
# Use default large-cap stocks (100 stocks, 500 days)
python show_top_positions.py --real

# Customize number of stocks and history
python show_top_positions.py --real --stocks 50 --days 252
```

## Using Real Market Data

### Command Line

```bash
# 20 stocks with 1 year of data
python show_top_positions.py --real --stocks 20 --days 252

# 100 stocks with 2 years of data
python show_top_positions.py --real --stocks 100 --days 504
```

### Programmatic Usage

```python
from stat_arb.market_data import fetch_market_data
from stat_arb.strategy import create_strategy

# Fetch real market data
prices, volumes, sectors, market_caps = fetch_market_data(
    n_stocks=50,
    lookback_days=252,
)

# Create and run strategy
strategy = create_strategy(target_sharpe=2.0, target_vol=0.08)
positions, diagnostics = strategy.generate_positions(
    prices=prices,
    volumes=volumes,
    sectors=sectors,
    market_caps=market_caps,
)

# Display results
print(f"Top Long: {positions.nlargest(10)}")
print(f"Top Short: {positions.nsmallest(10)}")
```

### Custom Ticker Lists

```python
from stat_arb.market_data import fetch_market_data

# Define your own universe
my_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']

prices, volumes, sectors, market_caps = fetch_market_data(
    tickers=my_tickers,
    lookback_days=252,
)
```

## Network Requirements

When using real market data (`--real` flag), the system requires:
- Internet connection to fetch data from Yahoo Finance
- Access to `finance.yahoo.com` (port 443)

If network access is restricted, the system will automatically fall back to synthetic data.

## Examples

```bash
# Run with synthetic data
python show_top_positions.py

# Run with 30 real stocks
python show_top_positions.py --real --stocks 30 --days 252
```

## Performance Target

- **Sharpe Ratio**: 2.0+
- **Volatility**: 8% annualized
- **Returns**: 16%+ annualized
- **Max Drawdown**: <10%
- **Market Neutrality**: Net exposure <5%
