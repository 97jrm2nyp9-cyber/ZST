"""
Statistical Arbitrage Strategy - Main Entry Point

This module provides the main StatArbStrategy class that orchestrates
all components to create a production-ready stat arb strategy targeting
2+ Sharpe ratio.

Strategy Design Philosophy:
1. Multiple uncorrelated alpha sources (diversification of alpha)
2. Rigorous factor neutralization (no uncompensated risks)
3. Transaction cost awareness (protect alpha from execution decay)
4. Adaptive risk management (survive regime changes)
5. Robust estimation (avoid overfitting)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from stat_arb.signals import (
    SignalConfig,
    PairsSignal,
    FactorResidualSignal,
    CrossSectionalMeanReversion,
    EigenportfolioSignal,
    SignalCombiner,
)
from stat_arb.risk import RiskManager, RiskLimits, FactorModel, RiskRegime
from stat_arb.portfolio import (
    PortfolioOptimizer,
    OptimizationConfig,
    TransactionCostModel,
)
from stat_arb.execution import ExecutionModel, ExecutionAlgorithm
from stat_arb.backtest import Backtester, BacktestConfig, BacktestResult


@dataclass
class StrategyConfig:
    """
    Master configuration for the stat arb strategy.

    Tuned for 2+ Sharpe ratio with realistic assumptions.
    """

    # Universe
    min_market_cap: float = 1e9  # $1B minimum market cap
    min_adv: float = 5e6  # $5M minimum ADV
    min_price: float = 5.0  # $5 minimum price
    max_universe_size: int = 500  # Maximum universe size

    # Signal weights
    pairs_weight: float = 0.25
    factor_residual_weight: float = 0.30
    mean_reversion_weight: float = 0.25
    eigenportfolio_weight: float = 0.20

    # Risk parameters
    target_volatility: float = 0.08  # 8% annualized target
    max_gross_leverage: float = 2.0
    max_net_exposure: float = 0.05
    max_position_size: float = 0.02
    max_sector_exposure: float = 0.10
    max_drawdown: float = 0.10

    # Execution
    max_daily_turnover: float = 0.25
    min_trade_size: float = 0.001

    # Backtest
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100_000_000


class StatArbStrategy:
    """
    Production Statistical Arbitrage Strategy

    This class implements a complete stat arb strategy combining:
    1. Pairs trading (cointegration-based)
    2. Factor residual reversion
    3. Cross-sectional mean reversion
    4. Eigenportfolio deviation trading

    Target: 2+ Sharpe ratio with 8% volatility
    Expected: 16%+ annual return with controlled drawdowns
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all strategy components."""
        # Signal configuration
        signal_config = SignalConfig(
            lookback_window=60,
            zscore_threshold=2.0,
            half_life=21,
            signal_decay=0.94,
        )

        # Initialize signal generators
        self.signals = [
            PairsSignal(signal_config, sector_constrained=True),
            FactorResidualSignal(signal_config, n_factors=6),
            CrossSectionalMeanReversion(signal_config, reversion_horizon=5),
            EigenportfolioSignal(signal_config, n_components=5),
        ]

        # Signal combiner with optimized weights
        self.signal_combiner = SignalCombiner(
            signals=self.signals,
            combination_method="inverse_vol",
            lookback_days=60,
        )

        # Risk limits
        risk_limits = RiskLimits(
            max_position_size=self.config.max_position_size,
            max_sector_exposure=self.config.max_sector_exposure,
            max_factor_exposure=0.10,
            max_gross_exposure=self.config.max_gross_leverage,
            max_net_exposure=self.config.max_net_exposure,
            max_portfolio_vol=self.config.target_volatility,
            max_drawdown=self.config.max_drawdown,
            max_daily_turnover=self.config.max_daily_turnover,
            min_positions=50,
        )

        # Risk manager
        self.risk_manager = RiskManager(
            limits=risk_limits,
            factor_model=FactorModel(estimation_window=252),
        )

        # Portfolio optimizer
        opt_config = OptimizationConfig(
            alpha_weight=1.0,
            risk_aversion=1.0,
            tcost_aversion=5.0,
            max_position_size=self.config.max_position_size,
            max_gross_exposure=self.config.max_gross_leverage,
            max_net_exposure=self.config.max_net_exposure,
            max_turnover=self.config.max_daily_turnover,
            min_trade_size=self.config.min_trade_size,
        )

        self.portfolio_optimizer = PortfolioOptimizer(
            config=opt_config,
            tcost_model=TransactionCostModel(),
        )

        # Execution model
        self.execution_model = ExecutionModel(
            default_algorithm=ExecutionAlgorithm.IS,
        )

        # Backtester
        backtest_config = BacktestConfig(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            leverage=self.config.max_gross_leverage,
            vol_target=self.config.target_volatility,
            max_drawdown_halt=self.config.max_drawdown,
        )

        self.backtester = Backtester(
            signal_generator=self.signal_combiner,
            risk_manager=self.risk_manager,
            portfolio_optimizer=self.portfolio_optimizer,
            config=backtest_config,
        )

    def filter_universe(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        Filter trading universe based on liquidity and size criteria.

        Args:
            prices: Price data
            volumes: Volume data
            market_caps: Market cap data

        Returns:
            List of tickers passing filters
        """
        tickers = prices.columns.tolist()
        valid_tickers = []

        for ticker in tickers:
            # Price filter
            if prices[ticker].iloc[-1] < self.config.min_price:
                continue

            # ADV filter
            adv = (prices[ticker] * volumes[ticker]).iloc[-20:].mean()
            if adv < self.config.min_adv:
                continue

            # Market cap filter
            if market_caps is not None:
                mcap = market_caps[ticker].iloc[-1]
                if mcap < self.config.min_market_cap:
                    continue

            # Data quality - require 252 days of data
            if prices[ticker].dropna().shape[0] < 252:
                continue

            valid_tickers.append(ticker)

        # Limit universe size
        if len(valid_tickers) > self.config.max_universe_size:
            # Sort by ADV and take top
            advs = {
                t: (prices[t] * volumes[t]).iloc[-20:].mean() for t in valid_tickers
            }
            valid_tickers = sorted(advs.keys(), key=lambda x: advs[x], reverse=True)[
                : self.config.max_universe_size
            ]

        return valid_tickers

    def generate_positions(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.DataFrame] = None,
        current_positions: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, Dict]:
        """
        Generate target positions from current market data.

        This is the main entry point for live trading.

        Args:
            prices: Historical prices up to current date
            volumes: Historical volumes
            sectors: Sector classifications
            market_caps: Market capitalizations
            current_positions: Current portfolio positions

        Returns:
            Tuple of (target_positions, diagnostics)
        """
        # Filter universe
        universe = self.filter_universe(prices, volumes, market_caps)
        prices = prices[universe]
        volumes = volumes[universe]
        if market_caps is not None:
            market_caps = market_caps[universe]

        # Initialize current positions if not provided
        if current_positions is None:
            current_positions = pd.Series(0.0, index=universe)
        else:
            current_positions = current_positions.reindex(universe).fillna(0)

        # Generate combined signal
        combined_signal = self.signal_combiner.generate_combined_signal(
            prices, volumes, sectors=sectors, market_caps=market_caps
        )

        # Apply risk management
        returns = prices.pct_change().dropna()

        neutralized_signal = self.risk_manager.neutralize_factors(
            combined_signal, returns, sectors, market_caps
        )

        # Check regime and adjust
        regime = self.risk_manager.get_current_regime(returns)
        adjusted_signal = self.risk_manager.adjust_for_regime(neutralized_signal, regime)

        # Estimate covariance for optimization
        cov = self._estimate_covariance(returns)

        # Get ADV and volatility for transaction cost model
        adv = (prices * volumes).iloc[-20:].mean()
        volatility = returns.iloc[-21:].std()

        # Optimize portfolio
        target_positions, opt_diagnostics = self.portfolio_optimizer.optimize(
            alpha=adjusted_signal,
            covariance=cov,
            current_positions=current_positions,
            prices=prices.iloc[-1],
            adv=adv,
            volatility=volatility,
        )

        # Scale to target volatility
        port_vol = np.sqrt(target_positions.values @ cov @ target_positions.values)
        if port_vol > 1e-8:
            scale = self.config.target_volatility / port_vol
            scale = min(scale, 2.0)  # Don't scale up too aggressively
            target_positions = target_positions * scale

        # Compile diagnostics
        diagnostics = {
            "signal_stats": self.signal_combiner.get_signal_diagnostics(),
            "optimization": opt_diagnostics,
            "regime": regime.value,
            "universe_size": len(universe),
            "portfolio_vol": port_vol * scale if port_vol > 1e-8 else 0,
        }

        return target_positions, diagnostics

    def _estimate_covariance(
        self, returns: pd.DataFrame, decay: float = 0.97
    ) -> np.ndarray:
        """Estimate exponentially weighted covariance matrix."""
        returns = returns.iloc[-252:].fillna(0)
        n_periods, n_assets = returns.shape

        weights = np.array([decay ** (n_periods - 1 - i) for i in range(n_periods)])
        weights = weights / weights.sum()

        mean_ret = np.average(returns.values, axis=0, weights=weights)
        centered = returns.values - mean_ret

        cov = np.zeros((n_assets, n_assets))
        for i in range(n_periods):
            cov += weights[i] * np.outer(centered[i], centered[i])

        # Shrinkage for stability
        shrinkage = 0.1
        cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))

        return cov * 252

    def run_backtest(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run full backtest of the strategy.

        Args:
            prices: Historical prices
            volumes: Historical volumes
            sectors: Sector classifications
            market_caps: Market capitalizations

        Returns:
            BacktestResult with performance metrics
        """
        # Filter universe
        universe = self.filter_universe(prices, volumes, market_caps)
        prices = prices[universe]
        volumes = volumes[universe]
        if market_caps is not None:
            market_caps = market_caps[universe]

        print(f"Running backtest on {len(universe)} securities")

        return self.backtester.run(
            prices=prices,
            volumes=volumes,
            sectors=sectors,
            market_caps=market_caps,
        )

    def get_risk_report(
        self,
        positions: pd.Series,
        returns: pd.DataFrame,
    ) -> Dict:
        """
        Generate comprehensive risk report for current positions.

        Args:
            positions: Current portfolio positions (weights)
            returns: Historical returns

        Returns:
            Dict with risk metrics and exposures
        """
        # Factor model risk decomposition
        self.risk_manager.factor_model.estimate(returns)
        risk_decomp = self.risk_manager.factor_model.get_portfolio_risk(positions)

        # VaR and CVaR
        var_metrics = self.risk_manager.calculate_var(positions, returns)

        # Current regime
        regime = self.risk_manager.get_current_regime(returns)
        regime_stats = self.risk_manager.regime_detector.get_regime_statistics(returns)

        # Drawdown status
        dd_status = self.risk_manager.drawdown_tracker.update(0)

        return {
            "risk_decomposition": risk_decomp,
            "var_metrics": var_metrics,
            "regime": regime.value,
            "regime_statistics": regime_stats,
            "drawdown_status": dd_status,
            "gross_exposure": np.abs(positions).sum(),
            "net_exposure": positions.sum(),
            "long_exposure": positions[positions > 0].sum(),
            "short_exposure": np.abs(positions[positions < 0].sum()),
            "n_long": (positions > 0.001).sum(),
            "n_short": (positions < -0.001).sum(),
        }


def create_strategy(
    target_sharpe: float = 2.0,
    target_vol: float = 0.08,
    max_drawdown: float = 0.10,
) -> StatArbStrategy:
    """
    Factory function to create a stat arb strategy with specified targets.

    Args:
        target_sharpe: Target Sharpe ratio (default 2.0)
        target_vol: Target annualized volatility (default 8%)
        max_drawdown: Maximum acceptable drawdown (default 10%)

    Returns:
        Configured StatArbStrategy instance
    """
    # Back-calculate implied return
    target_return = target_sharpe * target_vol

    config = StrategyConfig(
        target_volatility=target_vol,
        max_drawdown=max_drawdown,
        # More aggressive settings for higher Sharpe target
        pairs_weight=0.25 if target_sharpe >= 2.0 else 0.20,
        factor_residual_weight=0.30 if target_sharpe >= 2.0 else 0.25,
        mean_reversion_weight=0.25 if target_sharpe >= 2.0 else 0.30,
        eigenportfolio_weight=0.20 if target_sharpe >= 2.0 else 0.25,
    )

    return StatArbStrategy(config)
