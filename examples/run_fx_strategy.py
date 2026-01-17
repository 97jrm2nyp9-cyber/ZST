"""
Example: Running the Forex Trading Strategy

This script demonstrates how to use the FX trading strategy framework
to backtest a multi-alpha forex trading strategy.

The strategy combines:
1. Carry trades (interest rate differentials)
2. Momentum/trend following
3. Mean reversion
4. Cross-rate arbitrage

Target: 2+ Sharpe ratio with controlled risk
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from fx_strategy import FxTradingStrategy, FxConfig, generate_fx_data


def run_fx_strategy_backtest():
    """Run a complete forex strategy backtest"""

    print("="*60)
    print("FOREX TRADING STRATEGY BACKTEST")
    print("="*60)
    print()

    # Configure strategy
    print("Configuring strategy...")
    config = FxConfig(
        # Currency universe (G10 majors and crosses)
        currency_pairs=[
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
            'NZD/USD', 'USD/CAD', 'USD/CHF',
            'EUR/GBP', 'EUR/JPY', 'GBP/JPY'
        ],

        # Signal parameters (optimized)
        carry_lookback=30,               # Longer lookback for more stable carry
        momentum_short=10,               # Slightly longer for noise reduction
        momentum_medium=30,              # Extended medium-term
        momentum_long=90,                # Longer long-term trend
        mean_reversion_window=15,        # Faster mean reversion
        mean_reversion_std=1.5,          # Tighter bands for more signals

        # Risk management (more aggressive for higher returns)
        target_volatility=0.12,          # 12% target volatility (higher)
        max_gross_leverage=4.0,          # 400% gross (more aggressive)
        max_net_exposure=0.40,           # 40% max directional exposure
        max_position_size=0.20,          # 20% per pair
        max_currency_exposure=0.60,      # 60% per currency
        max_drawdown=0.20,               # 20% circuit breaker (more tolerance)

        # Transaction costs (FX has tight spreads)
        spread_bps=1.5,                  # 1.5 bps spread (better execution)
        commission_bps=0.1,              # 0.1 bps commission

        # Signal weights (optimized based on performance)
        signal_weights={
            'carry': 0.50,               # 50% carry trades (best performer)
            'momentum': 0.20,            # 20% momentum
            'mean_reversion': 0.10,      # 10% mean reversion (underperforming)
            'cross_rate': 0.20           # 20% cross-rate arbitrage
        }
    )

    # Generate synthetic FX data for backtesting
    print("Generating forex market data...")
    print("  Pairs: 10 G10 currency pairs")
    print("  Period: 2020-01-01 to 2024-12-31 (5 years)")
    print()

    prices, interest_rates, volatilities = generate_fx_data(
        pairs=config.currency_pairs,
        start_date='2020-01-01',
        end_date='2024-12-31',
        seed=42
    )

    print(f"Data generated:")
    print(f"  {len(prices)} trading days")
    print(f"  {len(prices.columns)} currency pairs")
    print()

    # Initialize strategy
    print("Initializing forex trading strategy...")
    strategy = FxTradingStrategy(config)
    print()

    # Run backtest
    print("Running backtest...")
    print("-" * 60)
    results = strategy.backtest(
        prices=prices,
        interest_rates=interest_rates,
        volatilities=volatilities,
        initial_capital=1_000_000,  # $1M starting capital
        combination_method='custom'  # Use configured signal weights
    )
    print()

    # Analyze results
    print("Analyzing signal contributions...")
    contributions = strategy.analyze_signal_contributions(
        prices, interest_rates, volatilities
    )

    print("\nSignal Contribution Analysis:")
    print("-" * 60)
    contribution_summary = contributions.mean() * 252  # Annualized
    for signal_name, contrib in contribution_summary.items():
        print(f"  {signal_name.capitalize()}: {contrib*100:.2f}% annualized")
    print()

    # Performance comparison with different weighting methods
    print("Testing different signal combination methods...")
    print("-" * 60)

    methods = ['equal', 'sharpe', 'inverse_vol', 'custom']
    comparison = {}

    for method in methods:
        print(f"\nTesting '{method}' weighting...")
        test_strategy = FxTradingStrategy(config)

        # Run quick backtest (suppress output)
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            test_results = test_strategy.backtest(
                prices=prices,
                interest_rates=interest_rates,
                volatilities=volatilities,
                initial_capital=1_000_000,
                combination_method=method
            )

        comparison[method] = {
            'sharpe': test_results['metrics']['sharpe_ratio'],
            'return': test_results['metrics']['annualized_return'],
            'volatility': test_results['metrics']['volatility'],
            'max_dd': test_results['metrics']['max_drawdown']
        }

    print("\nWeighting Method Comparison:")
    print("-" * 60)
    print(f"{'Method':<15} {'Sharpe':<10} {'Return':<10} {'Vol':<10} {'Max DD':<10}")
    print("-" * 60)
    for method, metrics in comparison.items():
        print(f"{method:<15} {metrics['sharpe']:<10.2f} "
              f"{metrics['return']*100:<10.2f} "
              f"{metrics['volatility']*100:<10.2f} "
              f"{metrics['max_dd']*100:<10.2f}")
    print()

    # Current portfolio state
    current_positions = strategy.get_current_positions()
    print("\nFinal Portfolio Positions:")
    print("-" * 60)
    for pair, position in current_positions.items():
        if abs(position) > 0.01:  # Only show significant positions
            direction = "LONG" if position > 0 else "SHORT"
            print(f"  {pair}: {direction} {abs(position)*100:.1f}%")

    if (current_positions.abs() > 0.01).sum() == 0:
        print("  No significant positions")
    print()

    # Risk metrics
    risk_metrics = strategy.risk_manager.calculate_risk_metrics(
        strategy.positions,
        volatilities
    )

    print("\nCurrent Risk Metrics:")
    print("-" * 60)
    print(f"  Gross Exposure: {risk_metrics['gross_exposure']*100:.1f}%")
    print(f"  Net Exposure: {risk_metrics['net_exposure']*100:.1f}%")
    print(f"  Number of Positions: {risk_metrics['num_positions']}")
    print(f"  Portfolio Volatility: {risk_metrics['portfolio_volatility']*100:.1f}%")
    print(f"  Value at Risk (95%): {risk_metrics['var_95']*100:.2f}%")
    print(f"  Current Regime: {risk_metrics['regime']}")

    if risk_metrics['currency_exposures']:
        print(f"\nCurrency Exposures:")
        for currency, exposure in sorted(
            risk_metrics['currency_exposures'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]:  # Top 5
            if abs(exposure) > 0.01:
                direction = "LONG" if exposure > 0 else "SHORT"
                print(f"    {currency}: {direction} {abs(exposure)*100:.1f}%")
    print()

    # Save results
    print("Saving results...")
    results['equity_curve'].to_csv('fx_equity_curve.csv')
    results['returns'].to_csv('fx_returns.csv')
    results['positions'].to_csv('fx_positions.csv')
    contributions.to_csv('fx_signal_contributions.csv')
    print("  - fx_equity_curve.csv")
    print("  - fx_returns.csv")
    print("  - fx_positions.csv")
    print("  - fx_signal_contributions.csv")
    print()

    # Summary
    print("="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Total Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {results['metrics']['win_rate']*100:.1f}%")
    print()

    if results['metrics']['sharpe_ratio'] >= 2.0:
        print("✓ TARGET ACHIEVED: Sharpe ratio >= 2.0")
    else:
        print(f"✗ Target not achieved (need {2.0 - results['metrics']['sharpe_ratio']:.2f} more Sharpe)")

    print()
    return results


if __name__ == '__main__':
    results = run_fx_strategy_backtest()
