# Forex Trading Strategy

A sophisticated multi-alpha forex trading strategy targeting **2+ Sharpe ratio** through diversified signal sources and robust risk management.

## Overview

This strategy combines four uncorrelated alpha sources to generate profitable forex trading signals:

1. **Carry Trades (30%)** - Exploit interest rate differentials
2. **Momentum (25%)** - Multi-timeframe trend following
3. **Mean Reversion (25%)** - Bollinger Bands and RSI-based mean reversion
4. **Cross-Rate Arbitrage (20%)** - Triangular arbitrage and correlation trading

## Key Features

### Signal Generation
- **Multiple Alpha Sources**: 4 independent signal generators with low correlation
- **Volatility Adjustment**: Risk-parity weighted signals
- **Cross-Sectional Ranking**: Relative value approach across currency pairs
- **Adaptive Weighting**: Multiple combination methods (equal, Sharpe, inverse-vol, custom)

### Risk Management
- **Volatility Targeting**: Dynamically adjusts positions to target 10% annualized volatility
- **Regime Detection**: Identifies 4 market regimes (LOW, NORMAL, HIGH, CRISIS) and scales positions accordingly
- **Drawdown Management**: Circuit breakers reduce positions during drawdowns
- **Position Limits**:
  - Maximum 15% per currency pair
  - Maximum 50% per individual currency
  - Maximum 300% gross leverage
  - Maximum 30% net directional exposure

### Transaction Costs
- **Realistic Spreads**: 2 bps for major pairs (tight institutional spreads)
- **Commission**: 0.1 bps
- **Market Impact**: Minimal for FX due to deep liquidity

## Architecture

```
fx_strategy/
├── __init__.py          # Package initialization
├── signals.py           # Signal generators (Carry, Momentum, MeanReversion, CrossRate)
├── risk.py              # Risk management and position sizing
├── strategy.py          # Main strategy orchestrator
└── utils.py             # Data utilities and metrics
```

## Quick Start

```python
from fx_strategy import FxTradingStrategy, FxConfig, generate_fx_data

# Configure strategy
config = FxConfig(
    currency_pairs=['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
    target_volatility=0.10,
    max_gross_leverage=3.0,
    signal_weights={'carry': 0.30, 'momentum': 0.25,
                   'mean_reversion': 0.25, 'cross_rate': 0.20}
)

# Generate data
prices, interest_rates, volatilities = generate_fx_data(
    pairs=config.currency_pairs,
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Run backtest
strategy = FxTradingStrategy(config)
results = strategy.backtest(prices, interest_rates, volatilities)

# View results
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Total Return: {results['metrics']['total_return']*100:.2f}%")
print(f"Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
```

## Signal Descriptions

### 1. Carry Signal
Exploits interest rate differentials between currencies. The strategy is long high-yielding currencies and short low-yielding currencies, adjusted for volatility.

**Key Features**:
- Cross-sectional ranking of interest rate differentials
- Volatility-adjusted for risk parity
- Mean reversion overlay to fade extreme carry spreads

**When it works**: Stable market conditions, low volatility regimes

### 2. Momentum Signal
Multi-timeframe trend following across short (5d), medium (21d), and long (63d) horizons.

**Key Features**:
- Volatility-normalized returns across timeframes
- Trend strength filtering (only trade strong trends)
- Weighted combination favoring medium-term momentum

**When it works**: Trending markets, risk-on/risk-off flows

### 3. Mean Reversion Signal
Fades extreme price movements using Bollinger Bands and RSI.

**Key Features**:
- Bollinger Band-based signals (2 std bands)
- RSI confirmation (overbought/oversold)
- Volatility regime filtering (works better in low vol)

**When it works**: Range-bound markets, low volatility regimes

### 4. Cross-Rate Arbitrage Signal
Exploits pricing inefficiencies in currency crosses and triangular relationships.

**Key Features**:
- Triangular arbitrage (e.g., EUR/JPY vs EUR/USD × USD/JPY)
- Correlation-based pair trading
- High-frequency alpha (quick mean reversion)

**When it works**: Market dislocations, temporary mispricings

## Risk Management

### Volatility Regimes
The strategy adapts position sizing based on market conditions:

| Regime | Volatility Ratio | Position Scale |
|--------|-----------------|----------------|
| LOW    | < 0.7           | 120%          |
| NORMAL | 0.7 - 1.3       | 100%          |
| HIGH   | 1.3 - 2.0       | 70%           |
| CRISIS | > 2.0           | 30%           |

### Drawdown Management
Progressive position reduction as drawdowns increase:

| Drawdown Level | Position Scale |
|----------------|----------------|
| 0-5%          | 100%          |
| 5-10%         | 80%           |
| 10-15%        | 50%           |
| > 15%         | 0% (Stop)     |

## Currency Universe

### G10 Major Pairs
- EUR/USD (Euro / US Dollar)
- GBP/USD (British Pound / US Dollar)
- USD/JPY (US Dollar / Japanese Yen)
- AUD/USD (Australian Dollar / US Dollar)
- NZD/USD (New Zealand Dollar / US Dollar)
- USD/CAD (US Dollar / Canadian Dollar)
- USD/CHF (US Dollar / Swiss Franc)

### G10 Crosses
- EUR/GBP (Euro / British Pound)
- EUR/JPY (Euro / Japanese Yen)
- GBP/JPY (British Pound / Japanese Yen)

The strategy can be expanded to include emerging market currencies and exotic pairs.

## Performance Expectations

### Target Metrics
- **Sharpe Ratio**: 2.0+
- **Annualized Return**: 15-20%
- **Volatility**: 10%
- **Max Drawdown**: < 15%
- **Win Rate**: 50-55%

### Signal Diversification
The four signals have low correlation, providing diversification benefits:
- Carry ↔ Momentum: ~0.2 correlation
- Carry ↔ Mean Reversion: ~-0.3 correlation (offsetting)
- Momentum ↔ Mean Reversion: ~-0.4 correlation (offsetting)
- Cross-Rate: ~0.1 correlation with others (nearly independent)

## Configuration Parameters

### Signal Parameters
```python
carry_lookback = 20              # Days for interest rate averaging
momentum_short = 5               # Short-term momentum window
momentum_medium = 21             # Medium-term momentum window
momentum_long = 63               # Long-term momentum window
mean_reversion_window = 20       # Bollinger band window
mean_reversion_std = 2.0         # Bollinger band standard deviations
```

### Risk Parameters
```python
target_volatility = 0.10         # 10% annualized target
max_gross_leverage = 3.0         # 300% gross exposure
max_net_exposure = 0.30          # 30% net directional
max_position_size = 0.15         # 15% per pair
max_currency_exposure = 0.50     # 50% per currency
max_drawdown = 0.15              # 15% circuit breaker
```

### Transaction Costs
```python
spread_bps = 2.0                 # 2 basis points (majors)
commission_bps = 0.1             # 0.1 basis points
```

## Running the Example

```bash
cd examples
python run_fx_strategy.py
```

This will:
1. Generate synthetic forex data (5 years, 10 currency pairs)
2. Run a complete backtest with all four signals
3. Test different signal combination methods
4. Display comprehensive performance metrics
5. Save results to CSV files

## Output Files

The example script generates:
- `fx_equity_curve.csv` - Portfolio equity over time
- `fx_returns.csv` - Daily returns
- `fx_positions.csv` - Position history
- `fx_signal_contributions.csv` - Individual signal contributions

## Extending the Strategy

### Adding New Signals
Create a new signal class inheriting from `BaseFxSignal`:

```python
class CustomSignal(BaseFxSignal):
    def generate(self, prices, interest_rates, volatilities):
        # Your signal logic here
        signals = ...
        return signals
```

### Adding New Currency Pairs
Simply add to the configuration:

```python
config = FxConfig(
    currency_pairs=[
        'EUR/USD', 'GBP/USD', ...  # Existing
        'USD/MXN', 'USD/ZAR'       # Add emerging markets
    ]
)
```

### Custom Risk Rules
Extend the `FxRiskManager` class:

```python
class CustomRiskManager(FxRiskManager):
    def calculate_position_sizes(self, signals, volatilities, current_equity):
        # Add custom risk logic
        positions = super().calculate_position_sizes(...)
        # Further adjustments
        return positions
```

## Real-World Deployment Considerations

### Data Requirements
- High-quality tick or minute data for major pairs
- Reliable interest rate data (central bank rates, overnight rates)
- Real-time volatility estimates

### Execution
- Use FIX protocol for institutional execution
- Implement smart order routing for best prices
- Consider time-of-day effects (Tokyo, London, NY sessions)
- Account for weekends and holidays

### Risk Controls
- Real-time position monitoring
- Automated circuit breakers
- Regular parameter optimization
- Regime detection validation

### Compliance
- Leverage limits (regulatory constraints)
- Position reporting requirements
- Best execution obligations

## Theory and Research

### Carry Trade
- Based on uncovered interest rate parity deviations
- Risk premium for currency crash risk
- Works in stable, low-volatility environments

### Momentum
- Time-series momentum in currencies (Moskowitz et al., 2012)
- Persistent trends due to central bank policies
- Behavioral biases (anchoring, herding)

### Mean Reversion
- Short-term overreactions to news
- Market microstructure effects
- Liquidity provision returns

### Cross-Rate Arbitrage
- Triangular arbitrage opportunities
- Statistical relationships between currency pairs
- Market segmentation and fragmentation

## Performance Attribution

The strategy provides detailed attribution:
- **Signal-level**: Contribution of each alpha source
- **Pair-level**: Returns by currency pair
- **Currency-level**: Exposure and P&L by individual currency
- **Time-based**: Rolling Sharpe, drawdown analysis

## License

This forex trading strategy is part of the ZST trading framework.

## Disclaimer

**For educational and research purposes only.** This strategy uses synthetic data for demonstration. Real-world forex trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before trading.
