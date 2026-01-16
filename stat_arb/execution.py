"""
Execution and Market Impact Module for Statistical Arbitrage

This module provides realistic execution modeling including:
1. Market impact models (Almgren-Chriss, square-root)
2. Optimal execution algorithms (VWAP, TWAP, IS)
3. Slippage estimation
4. Order scheduling

Key insight: For stat arb strategies, execution quality can make the
difference between profit and loss. A 2 Sharpe strategy can become
unprofitable with poor execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    VWAP = "vwap"  # Volume Weighted Average Price
    TWAP = "twap"  # Time Weighted Average Price
    IS = "implementation_shortfall"  # Minimize implementation shortfall
    ADAPTIVE = "adaptive"  # Adaptive to market conditions
    AGGRESSIVE = "aggressive"  # Front-load execution
    PASSIVE = "passive"  # Back-load execution


@dataclass
class MarketConditions:
    """Current market microstructure conditions."""

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    volatility: float
    spread_bps: float


@dataclass
class ExecutionReport:
    """Execution quality report."""

    avg_fill_price: float
    total_shares: float
    total_notional: float
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    spread_cost: float
    slippage_bps: float
    participation_rate: float
    execution_time_minutes: float


class AlmgrenChrissModel:
    """
    Almgren-Chriss Optimal Execution Model

    Implements the classic Almgren-Chriss framework for optimal
    execution that balances market impact vs. timing risk.

    Key equation:
    Total cost = permanent impact + temporary impact + volatility risk

    Optimal trajectory minimizes expected cost + risk aversion * variance
    """

    def __init__(
        self,
        eta: float = 0.1,  # Temporary impact coefficient
        gamma: float = 0.1,  # Permanent impact coefficient
        sigma: float = 0.02,  # Daily volatility
        lambda_risk: float = 1e-6,  # Risk aversion
    ):
        self.eta = eta  # Temporary impact
        self.gamma = gamma  # Permanent impact
        self.sigma = sigma  # Volatility
        self.lambda_risk = lambda_risk  # Risk aversion

    def optimal_trajectory(
        self,
        total_shares: float,
        total_time: int,
        adv: float,
    ) -> np.ndarray:
        """
        Calculate optimal trading trajectory.

        Args:
            total_shares: Total shares to trade
            total_time: Number of time periods
            adv: Average daily volume

        Returns:
            Array of shares to trade in each period
        """
        T = total_time
        X = total_shares

        # Normalize by ADV
        x_normalized = X / adv

        # Kappa parameter (balance of impact vs risk)
        kappa = np.sqrt(self.lambda_risk * self.sigma**2 / self.eta)

        # Optimal trajectory (Almgren-Chriss solution)
        trajectory = np.zeros(T)

        for t in range(T):
            # Remaining time
            tau = T - t

            # Optimal trade rate
            sinh_kappa_tau = np.sinh(kappa * tau)
            sinh_kappa_T = np.sinh(kappa * T)

            if sinh_kappa_T > 1e-8:
                x_t = X * sinh_kappa_tau / sinh_kappa_T
            else:
                x_t = X * tau / T  # Linear fallback

            # Trade in this period
            if t == 0:
                trajectory[t] = X - x_t
            else:
                prev_remaining = X * np.sinh(kappa * (T - t + 1)) / sinh_kappa_T
                trajectory[t] = prev_remaining - x_t

        # Ensure we trade exactly total shares
        trajectory[-1] += X - trajectory.sum()

        return trajectory * adv  # Denormalize

    def expected_cost(
        self,
        trajectory: np.ndarray,
        total_shares: float,
        price: float,
        adv: float,
    ) -> Dict[str, float]:
        """Calculate expected execution cost for a trajectory."""
        T = len(trajectory)
        notional = abs(total_shares * price)

        # Permanent impact
        permanent_impact = self.gamma * abs(total_shares / adv) * notional

        # Temporary impact (sum of per-period impacts)
        temporary_impact = 0
        for t in range(T):
            trade_rate = trajectory[t] / adv
            temporary_impact += self.eta * abs(trade_rate) * abs(trajectory[t] * price)

        # Timing risk (variance cost)
        remaining = total_shares - np.cumsum(trajectory)
        timing_variance = self.sigma**2 * np.sum(remaining**2 * price**2)
        timing_cost = self.lambda_risk * timing_variance

        return {
            "permanent_impact": permanent_impact,
            "temporary_impact": temporary_impact,
            "timing_cost": timing_cost,
            "total_cost": permanent_impact + temporary_impact + timing_cost,
            "total_cost_bps": (permanent_impact + temporary_impact) / notional * 10000,
        }


class SquareRootImpactModel:
    """
    Square Root Market Impact Model

    Implements the empirically-validated square root law:
    Impact = sigma * sqrt(Q / V)

    Where Q is trade size and V is average daily volume.
    This model is widely used in practice due to its empirical validity.
    """

    def __init__(
        self,
        impact_coefficient: float = 0.1,
        decay_factor: float = 0.5,  # Impact decay rate
    ):
        self.impact_coefficient = impact_coefficient
        self.decay_factor = decay_factor

    def calculate_impact(
        self,
        trade_size: float,
        price: float,
        adv: float,
        volatility: float,
    ) -> Dict[str, float]:
        """
        Calculate market impact for a single trade.

        Returns permanent and temporary impact components.
        """
        participation_rate = abs(trade_size) / (adv + 1e-8)
        notional = abs(trade_size * price)

        # Square root model
        impact_bps = (
            self.impact_coefficient * volatility * np.sqrt(participation_rate) * 10000
        )

        # Decompose into permanent (persists) and temporary (decays)
        permanent_impact = impact_bps * self.decay_factor
        temporary_impact = impact_bps * (1 - self.decay_factor)

        return {
            "total_impact_bps": impact_bps,
            "permanent_impact_bps": permanent_impact,
            "temporary_impact_bps": temporary_impact,
            "participation_rate": participation_rate,
            "impact_cost": notional * impact_bps / 10000,
        }

    def optimal_participation_rate(
        self,
        urgency: float,  # 0 = patient, 1 = urgent
        volatility: float,
        spread_bps: float,
    ) -> float:
        """
        Calculate optimal participation rate based on urgency.

        Higher urgency = trade faster despite higher impact.
        Lower urgency = trade slower to minimize impact.
        """
        # Base participation rate
        base_rate = 0.05  # 5% of ADV

        # Urgency adjustment
        urgency_multiplier = 1 + urgency * 4  # 1x to 5x

        # Volatility adjustment (trade slower in high vol)
        vol_adjustment = 0.02 / (volatility + 0.01)

        # Spread adjustment (trade faster if spread is tight)
        spread_adjustment = 5 / (spread_bps + 1)

        optimal_rate = base_rate * urgency_multiplier * vol_adjustment * spread_adjustment

        # Clip to reasonable range
        return np.clip(optimal_rate, 0.01, 0.25)


class ExecutionModel:
    """
    Comprehensive Execution Modeling System

    Combines market impact models with execution algorithms
    to provide realistic execution simulation and optimization.
    """

    def __init__(
        self,
        impact_model: Optional[SquareRootImpactModel] = None,
        almgren_chriss: Optional[AlmgrenChrissModel] = None,
        default_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IS,
    ):
        self.impact_model = impact_model or SquareRootImpactModel()
        self.almgren_chriss = almgren_chriss or AlmgrenChrissModel()
        self.default_algorithm = default_algorithm
        self.execution_history: List[ExecutionReport] = []

    def execute_order(
        self,
        ticker: str,
        shares: float,
        side: str,  # "buy" or "sell"
        price: float,
        adv: float,
        volatility: float,
        spread_bps: float = 2.0,
        algorithm: Optional[ExecutionAlgorithm] = None,
        urgency: float = 0.5,
        time_horizon_minutes: int = 60,
    ) -> ExecutionReport:
        """
        Simulate order execution with realistic market impact.

        Args:
            ticker: Security identifier
            shares: Number of shares to trade
            side: "buy" or "sell"
            price: Current market price
            adv: Average daily volume
            volatility: Daily return volatility
            spread_bps: Bid-ask spread in basis points
            algorithm: Execution algorithm to use
            urgency: Trading urgency (0-1)
            time_horizon_minutes: Maximum execution time

        Returns:
            ExecutionReport with fill details and costs
        """
        algorithm = algorithm or self.default_algorithm
        direction = 1 if side == "buy" else -1

        # Calculate optimal trajectory
        n_periods = max(1, time_horizon_minutes // 5)  # 5-minute buckets

        if algorithm == ExecutionAlgorithm.IS:
            trajectory = self.almgren_chriss.optimal_trajectory(
                total_shares=abs(shares),
                total_time=n_periods,
                adv=adv,
            )
        elif algorithm == ExecutionAlgorithm.TWAP:
            trajectory = np.ones(n_periods) * abs(shares) / n_periods
        elif algorithm == ExecutionAlgorithm.VWAP:
            # Assume U-shaped volume profile
            volume_profile = self._generate_volume_profile(n_periods)
            trajectory = volume_profile * abs(shares)
        elif algorithm == ExecutionAlgorithm.AGGRESSIVE:
            # Front-load: 70% in first quarter
            trajectory = self._aggressive_trajectory(abs(shares), n_periods)
        elif algorithm == ExecutionAlgorithm.PASSIVE:
            # Back-load: 70% in last quarter
            trajectory = self._passive_trajectory(abs(shares), n_periods)
        else:
            trajectory = np.ones(n_periods) * abs(shares) / n_periods

        # Simulate execution with impact
        total_impact = 0
        total_spread_cost = 0
        executed_shares = 0
        execution_prices = []

        for i, trade in enumerate(trajectory):
            if trade < 1:  # Skip tiny trades
                continue

            # Calculate impact for this slice
            impact = self.impact_model.calculate_impact(
                trade_size=trade,
                price=price,
                adv=adv,
                volatility=volatility,
            )

            # Execution price includes spread and impact
            spread_cost = spread_bps / 2 / 10000 * price
            impact_cost = impact["total_impact_bps"] / 10000 * price

            if side == "buy":
                fill_price = price + spread_cost + impact_cost
            else:
                fill_price = price - spread_cost - impact_cost

            execution_prices.append((trade, fill_price))
            total_impact += impact["impact_cost"]
            total_spread_cost += trade * spread_cost
            executed_shares += trade

        # Calculate average fill price
        if executed_shares > 0:
            avg_fill = sum(t * p for t, p in execution_prices) / executed_shares
        else:
            avg_fill = price

        # Implementation shortfall
        arrival_price = price
        is_cost = abs(avg_fill - arrival_price) * abs(shares)

        # Slippage in bps
        slippage_bps = abs(avg_fill - arrival_price) / arrival_price * 10000

        report = ExecutionReport(
            avg_fill_price=avg_fill,
            total_shares=abs(shares),
            total_notional=abs(shares * avg_fill),
            implementation_shortfall=is_cost,
            market_impact=total_impact,
            timing_cost=0,  # Would need price path simulation
            spread_cost=total_spread_cost,
            slippage_bps=slippage_bps,
            participation_rate=abs(shares) / adv,
            execution_time_minutes=time_horizon_minutes,
        )

        self.execution_history.append(report)
        return report

    def estimate_execution_cost(
        self,
        trades: pd.Series,
        prices: pd.Series,
        adv: pd.Series,
        volatility: pd.Series,
        spread_bps: float = 2.0,
    ) -> pd.DataFrame:
        """
        Estimate execution costs for a set of trades.

        Args:
            trades: Series of trade sizes (+ for buy, - for sell)
            prices: Current prices
            adv: Average daily volumes
            volatility: Volatilities

        Returns:
            DataFrame with cost breakdown for each trade
        """
        results = []

        for ticker in trades.index:
            trade = trades[ticker]
            if abs(trade) < 1e-8:
                continue

            price = prices.get(ticker, 100)
            daily_adv = adv.get(ticker, 1e6)
            vol = volatility.get(ticker, 0.02)

            # Notional trade value
            trade_notional = abs(trade * price)

            # Spread cost
            spread_cost = trade_notional * spread_bps / 2 / 10000

            # Impact cost
            impact = self.impact_model.calculate_impact(
                trade_size=abs(trade),
                price=price,
                adv=daily_adv,
                volatility=vol,
            )

            results.append(
                {
                    "ticker": ticker,
                    "trade_shares": trade,
                    "trade_notional": trade_notional,
                    "spread_cost": spread_cost,
                    "impact_cost": impact["impact_cost"],
                    "total_cost": spread_cost + impact["impact_cost"],
                    "cost_bps": (spread_cost + impact["impact_cost"])
                    / trade_notional
                    * 10000,
                    "participation_rate": impact["participation_rate"],
                }
            )

        return pd.DataFrame(results)

    def _generate_volume_profile(self, n_periods: int) -> np.ndarray:
        """Generate typical U-shaped intraday volume profile."""
        # U-shape: higher at open and close, lower midday
        x = np.linspace(0, 1, n_periods)
        profile = 1 - 0.5 * np.sin(np.pi * x)  # U-shape
        return profile / profile.sum()

    def _aggressive_trajectory(self, shares: float, n_periods: int) -> np.ndarray:
        """Generate front-loaded trajectory."""
        trajectory = np.zeros(n_periods)
        first_quarter = max(1, n_periods // 4)

        # 70% in first quarter
        trajectory[:first_quarter] = 0.7 * shares / first_quarter

        # 30% in remaining time
        remaining_periods = n_periods - first_quarter
        if remaining_periods > 0:
            trajectory[first_quarter:] = 0.3 * shares / remaining_periods

        return trajectory

    def _passive_trajectory(self, shares: float, n_periods: int) -> np.ndarray:
        """Generate back-loaded trajectory."""
        trajectory = np.zeros(n_periods)
        last_quarter = max(1, n_periods // 4)

        # 30% in first three quarters
        first_periods = n_periods - last_quarter
        if first_periods > 0:
            trajectory[:first_periods] = 0.3 * shares / first_periods

        # 70% in last quarter
        trajectory[first_periods:] = 0.7 * shares / last_quarter

        return trajectory


class SlippageEstimator:
    """
    Historical Slippage Analysis

    Analyzes historical execution data to estimate realistic
    slippage for backtesting.
    """

    def __init__(self):
        self.historical_slippage: List[Dict] = []

    def add_execution(
        self,
        intended_price: float,
        actual_price: float,
        shares: float,
        adv: float,
        volatility: float,
    ) -> None:
        """Record an execution for slippage analysis."""
        slippage_bps = (actual_price - intended_price) / intended_price * 10000
        participation = abs(shares) / adv

        self.historical_slippage.append(
            {
                "slippage_bps": slippage_bps,
                "participation_rate": participation,
                "volatility": volatility,
                "shares": shares,
            }
        )

    def estimate_slippage(
        self,
        shares: float,
        adv: float,
        volatility: float,
    ) -> Dict[str, float]:
        """
        Estimate expected slippage based on historical data.

        Uses regression on historical executions to predict slippage.
        """
        if len(self.historical_slippage) < 10:
            # Default model
            participation = abs(shares) / adv
            expected_slippage = 5 * np.sqrt(participation) * volatility / 0.02
            return {
                "expected_slippage_bps": expected_slippage,
                "slippage_std_bps": expected_slippage * 0.5,
                "model": "default",
            }

        # Fit regression model
        df = pd.DataFrame(self.historical_slippage)

        # Features
        X = np.column_stack(
            [
                np.sqrt(df["participation_rate"]),
                df["volatility"],
            ]
        )
        y = np.abs(df["slippage_bps"])

        # Simple OLS
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.array([2.0, 5.0, 50.0])

        # Predict
        participation = abs(shares) / adv
        features = np.array([1, np.sqrt(participation), volatility])
        expected_slippage = features @ beta

        # Estimate uncertainty
        residuals = y - X_with_const @ beta
        slippage_std = np.std(residuals)

        return {
            "expected_slippage_bps": max(0, expected_slippage),
            "slippage_std_bps": slippage_std,
            "model": "regression",
        }


class TradingScheduler:
    """
    Trading Schedule Optimizer

    Determines optimal times to trade based on:
    1. Market microstructure patterns
    2. Signal urgency
    3. Risk constraints
    """

    def __init__(
        self,
        market_open: str = "09:30",
        market_close: str = "16:00",
        avoid_open_minutes: int = 15,
        avoid_close_minutes: int = 15,
    ):
        self.market_open = market_open
        self.market_close = market_close
        self.avoid_open_minutes = avoid_open_minutes
        self.avoid_close_minutes = avoid_close_minutes

    def create_schedule(
        self,
        total_shares: float,
        adv: float,
        urgency: float,
        algorithm: ExecutionAlgorithm,
    ) -> List[Tuple[str, float]]:
        """
        Create intraday trading schedule.

        Returns list of (time, shares) tuples.
        """
        # Effective trading minutes
        total_minutes = 390 - self.avoid_open_minutes - self.avoid_close_minutes

        # Participation rate determines how long to trade
        target_participation = 0.05 + urgency * 0.15  # 5% to 20% of ADV

        # Minutes needed at target participation
        minutes_needed = total_shares / (adv / 390 * target_participation)
        minutes_needed = min(minutes_needed, total_minutes)

        # Number of 5-minute intervals
        n_intervals = max(1, int(minutes_needed / 5))

        # Generate volume profile
        if algorithm == ExecutionAlgorithm.VWAP:
            profile = self._vwap_profile(n_intervals)
        elif algorithm == ExecutionAlgorithm.AGGRESSIVE:
            profile = self._aggressive_profile(n_intervals)
        elif algorithm == ExecutionAlgorithm.PASSIVE:
            profile = self._passive_profile(n_intervals)
        else:
            profile = np.ones(n_intervals) / n_intervals

        # Create schedule
        schedule = []
        start_time = self.avoid_open_minutes

        for i in range(n_intervals):
            time_offset = start_time + i * 5
            hours = 9 + (30 + time_offset) // 60
            minutes = (30 + time_offset) % 60
            time_str = f"{hours:02d}:{minutes:02d}"

            shares = total_shares * profile[i]
            schedule.append((time_str, shares))

        return schedule

    def _vwap_profile(self, n: int) -> np.ndarray:
        """U-shaped volume profile."""
        x = np.linspace(0, 1, n)
        profile = 1 + 0.5 * np.cos(2 * np.pi * x)
        return profile / profile.sum()

    def _aggressive_profile(self, n: int) -> np.ndarray:
        """Front-loaded profile."""
        profile = np.exp(-np.arange(n) / (n / 3))
        return profile / profile.sum()

    def _passive_profile(self, n: int) -> np.ndarray:
        """Back-loaded profile."""
        profile = np.exp(np.arange(n) / (n / 3))
        return profile / profile.sum()
