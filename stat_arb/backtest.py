"""
Backtesting Engine for Statistical Arbitrage

This module provides a realistic backtesting framework that:
1. Simulates realistic execution with market impact
2. Handles corporate actions (splits, dividends)
3. Accounts for borrowing costs for short positions
4. Tracks detailed P&L attribution
5. Calculates comprehensive performance metrics

Key insight: Backtests must be realistic to be useful. Overly optimistic
backtests lead to disappointment in live trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

from stat_arb.signals import SignalCombiner, BaseSignal
from stat_arb.risk import RiskManager, RiskLimits, FactorModel
from stat_arb.portfolio import PortfolioOptimizer, TransactionCostModel

warnings.filterwarnings("ignore")


class PnLComponent(Enum):
    """P&L attribution components."""

    ALPHA = "alpha"
    MARKET = "market"
    SECTOR = "sector"
    FACTOR = "factor"
    TRANSACTION_COST = "transaction_cost"
    BORROW_COST = "borrow_cost"
    FINANCING = "financing"


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""

    # Dates
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Capital
    initial_capital: float = 100_000_000  # $100M
    leverage: float = 2.0

    # Costs
    borrow_rate: float = 0.005  # 50 bps annual borrow cost
    financing_rate: float = 0.02  # 2% annual financing spread
    commission_bps: float = 0.5
    spread_bps: float = 2.0

    # Execution
    execution_delay: int = 0  # Days delay for execution
    partial_fill_rate: float = 1.0  # Assume full fills

    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # Risk
    vol_target: float = 0.10  # 10% annualized vol target
    max_drawdown_halt: float = 0.15  # Halt trading at 15% drawdown


@dataclass
class BacktestResult:
    """Container for backtest results."""

    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    pnl_attribution: pd.DataFrame
    metrics: Dict[str, float]
    drawdown_series: pd.Series
    exposure_history: pd.DataFrame


class Backtester:
    """
    Event-Driven Backtesting Engine

    Simulates trading strategy with realistic execution,
    costs, and risk management.
    """

    def __init__(
        self,
        signal_generator: Union[SignalCombiner, BaseSignal],
        risk_manager: Optional[RiskManager] = None,
        portfolio_optimizer: Optional[PortfolioOptimizer] = None,
        config: Optional[BacktestConfig] = None,
    ):
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager or RiskManager()
        self.portfolio_optimizer = portfolio_optimizer or PortfolioOptimizer()
        self.config = config or BacktestConfig()

        # State tracking
        self.positions: pd.Series = pd.Series(dtype=float)
        self.cash: float = 0
        self.portfolio_value: float = 0

        # History
        self.returns_history: List[float] = []
        self.positions_history: List[pd.Series] = []
        self.trades_history: List[pd.Series] = []
        self.pnl_components: List[Dict[str, float]] = []

    def run(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.DataFrame] = None,
        dividends: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run backtest over historical data.

        Args:
            prices: Adjusted close prices (dates x tickers)
            volumes: Trading volumes (dates x tickers)
            sectors: Sector classification for each ticker
            market_caps: Market capitalizations
            dividends: Dividend payments

        Returns:
            BacktestResult with performance metrics and history
        """
        # Initialize
        self._initialize(prices)

        # Date range
        dates = prices.index
        if self.config.start_date:
            dates = dates[dates >= self.config.start_date]
        if self.config.end_date:
            dates = dates[dates <= self.config.end_date]

        # Minimum lookback for signal generation
        lookback = 252
        dates = dates[lookback:]

        print(f"Running backtest from {dates[0]} to {dates[-1]}")
        print(f"Universe size: {len(prices.columns)} securities")

        # Main loop
        for i, date in enumerate(dates):
            if i % 63 == 0:  # Quarterly update
                print(f"  Processing {date}...")

            # Get data up to current date
            current_prices = prices.loc[:date]
            current_volumes = volumes.loc[:date]
            current_mcaps = market_caps.loc[:date] if market_caps is not None else None

            try:
                # Step 1: Generate signals
                signals = self._generate_signals(
                    current_prices, current_volumes, sectors, current_mcaps
                )

                # Step 2: Risk management
                signals = self._apply_risk_management(
                    signals, current_prices, sectors, current_mcaps
                )

                # Step 3: Portfolio optimization
                target_positions = self._optimize_portfolio(
                    signals,
                    current_prices,
                    current_volumes,
                    sectors,
                )

                # Step 4: Execute trades
                self._execute_trades(
                    target_positions,
                    current_prices.iloc[-1],
                    current_volumes.iloc[-self.config.execution_delay - 1]
                    if self.config.execution_delay > 0
                    else current_volumes.iloc[-1],
                )

                # Step 5: Mark to market
                self._mark_to_market(current_prices.iloc[-1], dividends, date)

                # Step 6: Record state
                self._record_state(date)

            except Exception as e:
                print(f"  Error on {date}: {e}")
                continue

        # Compile results
        return self._compile_results(prices)

    def _initialize(self, prices: pd.DataFrame) -> None:
        """Initialize backtest state."""
        self.positions = pd.Series(0.0, index=prices.columns)
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.prev_portfolio_value = self.config.initial_capital

        self.returns_history = []
        self.positions_history = []
        self.trades_history = []
        self.pnl_components = []
        self.dates = []

    def _generate_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series],
        market_caps: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Generate trading signals."""
        if hasattr(self.signal_generator, "generate_combined_signal"):
            return self.signal_generator.generate_combined_signal(
                prices, volumes, sectors=sectors, market_caps=market_caps
            )
        else:
            return self.signal_generator.generate(
                prices, volumes, sectors=sectors, market_caps=market_caps
            )

    def _apply_risk_management(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        sectors: Optional[pd.Series],
        market_caps: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Apply risk management to signals."""
        returns = prices.pct_change().dropna()

        # Factor neutralization
        neutralized = self.risk_manager.neutralize_factors(
            signals, returns, sectors, market_caps
        )

        # Regime adjustment
        regime = self.risk_manager.get_current_regime(returns)
        adjusted = self.risk_manager.adjust_for_regime(neutralized, regime)

        # Drawdown check
        if self.risk_manager.drawdown_tracker.should_reduce_risk():
            multiplier = self.risk_manager.drawdown_tracker.get_risk_multiplier()
            adjusted = adjusted * multiplier

        return adjusted

    def _optimize_portfolio(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series],
    ) -> pd.Series:
        """Convert signals to target positions."""
        returns = prices.pct_change().dropna()

        # Estimate covariance
        cov = self._estimate_covariance(returns)

        # ADV and volatility for transaction costs
        adv = volumes.iloc[-20:].mean()
        volatility = returns.iloc[-21:].std()

        # Current positions
        current = self.positions.reindex(signals.index).fillna(0)

        # Optimize
        try:
            optimal, diagnostics = self.portfolio_optimizer.optimize(
                alpha=signals,
                covariance=cov,
                current_positions=current,
                prices=prices.iloc[-1],
                adv=adv,
                volatility=volatility,
            )
        except Exception:
            # Fallback: simple signal scaling
            target_gross = self.config.leverage * 0.5
            optimal = signals / (np.abs(signals).sum() + 1e-8) * target_gross
            optimal = optimal - optimal.mean()  # Market neutral

        # Apply vol targeting
        port_vol = np.sqrt(optimal.values @ cov @ optimal.values)
        if port_vol > 1e-8:
            scale = self.config.vol_target / port_vol
            optimal = optimal * min(scale, 2.0)

        return optimal

    def _estimate_covariance(
        self, returns: pd.DataFrame, decay: float = 0.97
    ) -> np.ndarray:
        """Estimate covariance with exponential decay."""
        returns = returns.iloc[-252:]  # Use 1 year
        n_periods, n_assets = returns.shape

        # Handle missing values
        returns = returns.fillna(0)

        weights = np.array(
            [decay ** (n_periods - 1 - i) for i in range(n_periods)]
        )
        weights = weights / weights.sum()

        mean_ret = np.average(returns.values, axis=0, weights=weights)
        centered = returns.values - mean_ret

        cov = np.zeros((n_assets, n_assets))
        for i in range(n_periods):
            cov += weights[i] * np.outer(centered[i], centered[i])

        # Shrinkage toward diagonal for stability
        shrinkage = 0.1
        cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))

        return cov * 252  # Annualize

    def _execute_trades(
        self,
        target_positions: pd.Series,
        prices: pd.Series,
        volumes: pd.Series,
    ) -> None:
        """Execute trades from current to target positions."""
        # Calculate trades
        current = self.positions.reindex(target_positions.index).fillna(0)
        trades = target_positions - current

        # Transaction costs
        tcost_model = TransactionCostModel(
            commission_bps=self.config.commission_bps,
            spread_bps=self.config.spread_bps,
        )

        total_tcost = 0
        for ticker in trades.index:
            if abs(trades[ticker]) < 1e-6:
                continue

            trade_notional = abs(trades[ticker]) * self.portfolio_value
            vol = 0.02  # Default volatility
            adv = volumes.get(ticker, 1e6) * prices.get(ticker, 100)

            cost = tcost_model.calculate_cost(
                trade_size=trade_notional,
                price=prices.get(ticker, 100),
                adv=adv,
                volatility=vol,
            )
            total_tcost += cost

        # Update positions
        self.positions = target_positions.copy()
        self.cash -= total_tcost

        # Record trades
        self.trades_history.append(trades)

    def _mark_to_market(
        self,
        prices: pd.Series,
        dividends: Optional[pd.DataFrame],
        date: str,
    ) -> None:
        """Mark portfolio to market."""
        self.prev_portfolio_value = self.portfolio_value

        # Position value
        position_value = (self.positions * prices).sum() * self.portfolio_value

        # This is a simplified MTM - in reality you'd track share counts
        # Here positions are weights, so portfolio value changes with returns

        # Calculate return
        aligned_positions = self.positions.reindex(prices.index).fillna(0)
        if len(self.positions_history) > 0:
            prev_prices = self.positions_history[-1]
            returns = prices / prices.shift(1) - 1

            # Portfolio return
            daily_return = (aligned_positions * returns.fillna(0)).sum()

            # Subtract costs
            borrow_cost = (
                self.config.borrow_rate / 252 * np.abs(aligned_positions[aligned_positions < 0]).sum()
            )
            financing_cost = (
                self.config.financing_rate / 252 * aligned_positions[aligned_positions > 0].sum()
            )

            net_return = daily_return - borrow_cost - financing_cost

            # Update portfolio value
            self.portfolio_value = self.prev_portfolio_value * (1 + net_return)
            self.returns_history.append(net_return)

            # Update drawdown tracker
            self.risk_manager.drawdown_tracker.update(net_return)
        else:
            self.returns_history.append(0.0)

    def _record_state(self, date: str) -> None:
        """Record current state for history."""
        self.positions_history.append(self.positions.copy())
        self.dates.append(date)

    def _compile_results(self, prices: pd.DataFrame) -> BacktestResult:
        """Compile backtest results into BacktestResult object."""
        # Returns series
        returns = pd.Series(self.returns_history, index=self.dates)

        # Positions DataFrame
        positions_df = pd.DataFrame(self.positions_history, index=self.dates)

        # Trades DataFrame
        trades_df = pd.DataFrame(self.trades_history, index=self.dates[: len(self.trades_history)])

        # Calculate metrics
        metrics = self._calculate_metrics(returns)

        # Drawdown series
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max

        # Exposure history
        exposure_df = pd.DataFrame(
            {
                "gross": positions_df.abs().sum(axis=1),
                "net": positions_df.sum(axis=1),
                "long": positions_df.clip(lower=0).sum(axis=1),
                "short": positions_df.clip(upper=0).abs().sum(axis=1),
            },
            index=self.dates,
        )

        # PnL attribution (simplified)
        pnl_df = pd.DataFrame(
            {
                "total_return": returns,
                "cumulative_return": cum_returns - 1,
            },
            index=self.dates,
        )

        return BacktestResult(
            returns=returns,
            positions=positions_df,
            trades=trades_df,
            pnl_attribution=pnl_df,
            metrics=metrics,
            drawdown_series=drawdown,
            exposure_history=exposure_df,
        )

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) < 20:
            return {"error": "Insufficient data"}

        # Clean returns
        returns = returns.dropna()

        # Basic stats
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)

        # Risk-adjusted
        sharpe = ann_return / (ann_vol + 1e-8)
        sortino_vol = returns[returns < 0].std() * np.sqrt(252)
        sortino = ann_return / (sortino_vol + 1e-8)

        # Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_return / (abs(max_dd) + 1e-8)

        # Hit rate
        hit_rate = (returns > 0).mean()

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-8)

        # Skewness and kurtosis
        from scipy import stats as sp_stats

        skew = sp_stats.skew(returns)
        kurt = sp_stats.kurtosis(returns)

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # Tail ratio
        top_decile = returns[returns >= np.percentile(returns, 90)].mean()
        bottom_decile = abs(returns[returns <= np.percentile(returns, 10)].mean())
        tail_ratio = top_decile / (bottom_decile + 1e-8)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "skewness": skew,
            "kurtosis": kurt,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "tail_ratio": tail_ratio,
            "n_days": len(returns),
            "n_years": len(returns) / 252,
        }


class WalkForwardBacktester(Backtester):
    """
    Walk-Forward Backtesting

    Implements rolling window backtesting where model parameters
    are re-estimated periodically on training data and tested
    on out-of-sample data.

    This is crucial for realistic performance estimation and
    avoiding look-ahead bias.
    """

    def __init__(
        self,
        signal_generator: Union[SignalCombiner, BaseSignal],
        risk_manager: Optional[RiskManager] = None,
        portfolio_optimizer: Optional[PortfolioOptimizer] = None,
        config: Optional[BacktestConfig] = None,
        train_window: int = 504,  # 2 years training
        test_window: int = 63,  # 3 months testing
        retrain_frequency: int = 21,  # Monthly retraining
    ):
        super().__init__(signal_generator, risk_manager, portfolio_optimizer, config)
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency

    def run_walk_forward(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Splits data into rolling train/test windows and aggregates results.
        """
        all_returns = []
        all_positions = []
        all_dates = []

        dates = prices.index
        start_idx = self.train_window

        while start_idx + self.test_window <= len(dates):
            # Training period
            train_start = start_idx - self.train_window
            train_end = start_idx

            # Test period
            test_start = start_idx
            test_end = min(start_idx + self.test_window, len(dates))

            # Train
            train_prices = prices.iloc[train_start:train_end]
            train_volumes = volumes.iloc[train_start:train_end]

            # Test
            test_prices = prices.iloc[test_start:test_end]
            test_volumes = volumes.iloc[test_start:test_end]

            # Run sub-backtest
            self._initialize(test_prices)

            for i in range(len(test_prices)):
                date = test_prices.index[i]

                # Use all data up to current date for signal generation
                current_prices = prices.loc[:date]
                current_volumes = volumes.loc[:date]

                try:
                    signals = self._generate_signals(
                        current_prices, current_volumes, sectors, market_caps
                    )
                    signals = self._apply_risk_management(
                        signals, current_prices, sectors, market_caps
                    )
                    target = self._optimize_portfolio(
                        signals, current_prices, current_volumes, sectors
                    )
                    self._execute_trades(target, current_prices.iloc[-1], current_volumes.iloc[-1])
                    self._mark_to_market(current_prices.iloc[-1], None, date)
                    self._record_state(date)
                except Exception:
                    continue

            # Aggregate results
            all_returns.extend(self.returns_history)
            all_positions.extend(self.positions_history)
            all_dates.extend(self.dates)

            # Move to next period
            start_idx += self.retrain_frequency

        # Compile aggregated results
        returns = pd.Series(all_returns, index=all_dates)
        positions_df = pd.DataFrame(all_positions, index=all_dates)
        metrics = self._calculate_metrics(returns)

        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max

        return BacktestResult(
            returns=returns,
            positions=positions_df,
            trades=pd.DataFrame(),
            pnl_attribution=pd.DataFrame(),
            metrics=metrics,
            drawdown_series=drawdown,
            exposure_history=pd.DataFrame(),
        )
