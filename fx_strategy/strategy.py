"""
Forex Trading Strategy

Main strategy orchestrator combining signals, risk management, and execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .signals import FxSignalCombiner
from .risk import FxRiskManager, calculate_transaction_costs
from .utils import FxConfig, calculate_fx_metrics


class FxTradingStrategy:
    """
    Multi-Alpha Forex Trading Strategy

    Combines multiple alpha signals with robust risk management
    to achieve profitable forex trading.

    Target: 2+ Sharpe ratio with controlled drawdowns

    Features:
    - 4 alpha sources (carry, momentum, mean reversion, cross-rate)
    - Volatility targeting
    - Regime-aware position sizing
    - Currency exposure limits
    - Transaction cost modeling
    """

    def __init__(self, config: Optional[FxConfig] = None):
        """
        Initialize forex trading strategy.

        Parameters
        ----------
        config : FxConfig, optional
            Strategy configuration. If None, uses defaults.
        """
        self.config = config or FxConfig()
        self.signal_combiner = FxSignalCombiner(self.config)
        self.risk_manager = FxRiskManager(self.config)

        # Strategy state
        self.positions = None
        self.equity_curve = None
        self.returns = None
        self.metrics = None

    def generate_signals(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame,
        combination_method: str = 'custom'
    ) -> pd.DataFrame:
        """
        Generate combined trading signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Exchange rates (dates × pairs)
        interest_rates : pd.DataFrame
            Interest rate differentials (dates × pairs)
        volatilities : pd.DataFrame
            Realized volatilities (dates × pairs)
        combination_method : str
            Signal combination method

        Returns
        -------
        signals : pd.DataFrame
            Combined trading signals
        """
        return self.signal_combiner.combine(
            prices,
            interest_rates,
            volatilities,
            method=combination_method
        )

    def backtest(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame,
        initial_capital: float = 1000000,
        combination_method: str = 'custom'
    ) -> Dict:
        """
        Run backtest of the forex trading strategy.

        Parameters
        ----------
        prices : pd.DataFrame
            Exchange rates (dates × pairs)
        interest_rates : pd.DataFrame
            Interest rate differentials (dates × pairs)
        volatilities : pd.DataFrame
            Realized volatilities (dates × pairs)
        initial_capital : float
            Starting capital in base currency
        combination_method : str
            Signal combination method

        Returns
        -------
        results : dict
            Backtest results including metrics and time series
        """
        print("Starting forex strategy backtest...")

        # Generate signals
        print("Generating trading signals...")
        signals = self.generate_signals(
            prices,
            interest_rates,
            volatilities,
            combination_method
        )

        # Get individual signals for analysis
        individual_signals = self.signal_combiner.get_individual_signals(
            prices,
            interest_rates,
            volatilities
        )

        # Calculate returns
        returns = prices.pct_change()

        # Initialize portfolio tracking
        equity = initial_capital
        equity_curve = pd.Series(index=prices.index, dtype=float)
        equity_curve.iloc[0] = equity

        positions_list = []
        portfolio_returns = []

        # Previous positions (start with no positions)
        prev_positions = pd.Series(0.0, index=prices.columns)

        print(f"Backtesting {len(prices)} days...")

        # Simulation loop
        for i in range(1, len(prices)):
            date = prices.index[i]

            # Get signal for previous day (trade on yesterday's signal)
            signal = signals.iloc[i-1]

            # Calculate position sizes with risk management
            current_volatility = volatilities.iloc[:i]

            # Create DataFrame for single-row signal
            signal_df = pd.DataFrame([signal], columns=signal.index)

            # Calculate positions
            positions = self.risk_manager.calculate_position_sizes(
                signal_df,
                current_volatility,
                current_equity=equity / initial_capital
            )

            # Get positions as Series
            if len(positions) > 0:
                current_positions = positions.iloc[0]
            else:
                current_positions = pd.Series(0.0, index=prices.columns)

            # Calculate transaction costs
            turnover = (current_positions - prev_positions).abs().sum()
            tcost = turnover * (self.config.spread_bps + self.config.commission_bps) / 10000

            # Calculate portfolio return
            # Position × return for each pair
            pair_returns = current_positions * returns.iloc[i]
            portfolio_return = pair_returns.sum()

            # Subtract transaction costs
            portfolio_return -= tcost

            # Add carry (interest rate differential)
            # Carry benefit/cost: position × (interest_rate / 252) for daily
            if i < len(interest_rates):
                carry_benefit = (current_positions * interest_rates.iloc[i] / 100 / 252).sum()
                portfolio_return += carry_benefit

            # Update equity
            equity *= (1 + portfolio_return)
            equity_curve.iloc[i] = equity

            # Store
            positions_list.append(current_positions)
            portfolio_returns.append(portfolio_return)
            prev_positions = current_positions

        # Create DataFrames from results
        self.positions = pd.DataFrame(positions_list, index=prices.index[1:])
        self.returns = pd.Series(portfolio_returns, index=prices.index[1:])
        self.equity_curve = equity_curve

        # Calculate metrics
        print("Calculating performance metrics...")
        self.metrics = calculate_fx_metrics(
            self.returns,
            self.positions,
            prices,
            self.config
        )

        # Calculate signal-specific metrics
        signal_metrics = {}
        for signal_name, signal_values in individual_signals.items():
            # Calculate returns for each signal
            signal_returns_list = []
            for i in range(1, len(prices)):
                sig = signal_values.iloc[i-1]
                ret = returns.iloc[i]
                sig_return = (sig * ret).mean()  # Average across pairs
                signal_returns_list.append(sig_return)

            signal_returns_series = pd.Series(signal_returns_list, index=prices.index[1:])

            # Calculate Sharpe for this signal
            if signal_returns_series.std() > 0:
                signal_sharpe = (
                    signal_returns_series.mean() / signal_returns_series.std() * np.sqrt(252)
                )
            else:
                signal_sharpe = 0

            signal_metrics[signal_name] = {
                'sharpe': signal_sharpe,
                'mean_return': signal_returns_series.mean() * 252,
                'volatility': signal_returns_series.std() * np.sqrt(252)
            }

        # Print results
        print("\n" + "="*60)
        print("FOREX TRADING STRATEGY BACKTEST RESULTS")
        print("="*60)
        print(f"\nStrategy Configuration:")
        print(f"  Currency Pairs: {len(self.config.currency_pairs)}")
        print(f"  Target Volatility: {self.config.target_volatility*100:.1f}%")
        print(f"  Max Gross Leverage: {self.config.max_gross_leverage:.1f}x")
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {self.metrics['total_return']*100:.2f}%")
        print(f"  Annualized Return: {self.metrics['annualized_return']*100:.2f}%")
        print(f"  Volatility: {self.metrics['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")
        print(f"  Max Drawdown: {self.metrics['max_drawdown']*100:.2f}%")
        print(f"  Win Rate: {self.metrics['win_rate']*100:.1f}%")
        print(f"\nTrading Activity:")
        print(f"  Avg Daily Turnover: {self.metrics['avg_daily_turnover']*100:.1f}%")
        print(f"  Avg Gross Exposure: {self.metrics['avg_gross_exposure']*100:.1f}%")
        print(f"  Avg Net Exposure: {self.metrics['avg_net_exposure']*100:.1f}%")
        print(f"\nIndividual Signal Performance:")
        for signal_name, metrics in signal_metrics.items():
            print(f"  {signal_name.capitalize()}:")
            print(f"    Sharpe: {metrics['sharpe']:.2f}")
            print(f"    Ann. Return: {metrics['mean_return']*100:.2f}%")
            print(f"    Volatility: {metrics['volatility']*100:.2f}%")
        print("="*60)

        # Return comprehensive results
        return {
            'metrics': self.metrics,
            'signal_metrics': signal_metrics,
            'equity_curve': self.equity_curve,
            'returns': self.returns,
            'positions': self.positions,
            'final_equity': equity,
            'config': self.config
        }

    def get_current_positions(self) -> pd.Series:
        """Get current portfolio positions"""
        if self.positions is not None and len(self.positions) > 0:
            return self.positions.iloc[-1]
        return pd.Series(dtype=float)

    def get_performance_summary(self) -> Dict:
        """Get summary of strategy performance"""
        if self.metrics is None:
            return {}

        return {
            'sharpe_ratio': self.metrics['sharpe_ratio'],
            'total_return': self.metrics['total_return'],
            'max_drawdown': self.metrics['max_drawdown'],
            'volatility': self.metrics['volatility'],
            'win_rate': self.metrics['win_rate']
        }

    def analyze_signal_contributions(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze contribution of each signal to portfolio returns.

        Parameters
        ----------
        prices : pd.DataFrame
            Exchange rates
        interest_rates : pd.DataFrame
            Interest rate differentials
        volatilities : pd.DataFrame
            Realized volatilities

        Returns
        -------
        contributions : pd.DataFrame
            Signal contributions over time
        """
        individual_signals = self.signal_combiner.get_individual_signals(
            prices,
            interest_rates,
            volatilities
        )

        returns = prices.pct_change()
        contributions = {}

        for signal_name, signal_values in individual_signals.items():
            weight = self.config.signal_weights.get(signal_name, 0)

            # Calculate contribution: weight × signal × return
            signal_contribution = []
            for i in range(1, len(prices)):
                sig = signal_values.iloc[i-1] * weight
                ret = returns.iloc[i]
                contrib = (sig * ret).mean()  # Average across pairs
                signal_contribution.append(contrib)

            contributions[signal_name] = signal_contribution

        return pd.DataFrame(contributions, index=prices.index[1:])
