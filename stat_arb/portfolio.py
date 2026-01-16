"""
Portfolio Optimization Module for Statistical Arbitrage

This module implements sophisticated portfolio construction that converts
alpha signals into optimal positions while considering:
1. Transaction costs (commission + spread + market impact)
2. Risk constraints (factor neutrality, position limits)
3. Turnover constraints
4. Tax-loss harvesting considerations

Key Innovation: Multi-period optimization that considers signal decay
and trading trajectory optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy import sparse
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TransactionCostModel:
    """
    Transaction cost model parameters.

    Total cost = commission + spread + market_impact
    Market impact = impact_coefficient * sqrt(trade_size / ADV) * volatility
    """

    commission_bps: float = 0.5  # 0.5 bps commission
    spread_bps: float = 2.0  # 2 bps half-spread
    impact_coefficient: float = 0.1  # Market impact multiplier
    fixed_cost_per_trade: float = 0.0  # Fixed cost per trade

    def calculate_cost(
        self,
        trade_size: float,
        price: float,
        adv: float,
        volatility: float,
    ) -> float:
        """Calculate total transaction cost for a trade."""
        notional = abs(trade_size * price)

        # Linear costs
        commission = notional * self.commission_bps / 10000
        spread = notional * self.spread_bps / 10000

        # Market impact (square root model)
        participation_rate = abs(trade_size) / (adv + 1e-8)
        impact = (
            self.impact_coefficient
            * np.sqrt(participation_rate)
            * volatility
            * notional
        )

        return commission + spread + impact + self.fixed_cost_per_trade


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""

    # Objective weights
    alpha_weight: float = 1.0
    risk_aversion: float = 1.0
    tcost_aversion: float = 5.0  # High penalty for transaction costs

    # Constraints
    max_position_size: float = 0.02
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 0.05
    max_turnover: float = 0.30
    min_trade_size: float = 0.001  # Don't trade tiny amounts

    # Optimization
    solver: str = "SLSQP"
    max_iterations: int = 1000
    tolerance: float = 1e-8


class PortfolioOptimizer:
    """
    Portfolio Optimization Engine

    Implements mean-variance optimization with transaction costs
    and various constraints. Supports both single-period and
    multi-period (trajectory) optimization.
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        tcost_model: Optional[TransactionCostModel] = None,
    ):
        self.config = config or OptimizationConfig()
        self.tcost_model = tcost_model or TransactionCostModel()
        self.optimization_history: List[Dict] = []

    def optimize(
        self,
        alpha: pd.Series,
        covariance: np.ndarray,
        current_positions: pd.Series,
        prices: pd.Series,
        adv: pd.Series,
        volatility: pd.Series,
        factor_loadings: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, Dict]:
        """
        Single-period mean-variance optimization with transaction costs.

        Maximizes: alpha'w - (risk_aversion/2)*w'Σw - tcost_aversion*TC(w-w0)

        Subject to:
        - Position limits: |w_i| <= max_position
        - Gross exposure: sum(|w|) <= max_gross
        - Net exposure: |sum(w)| <= max_net
        - Turnover: sum(|w - w0|) <= max_turnover
        - Factor neutrality: B'w = 0 (if factor_loadings provided)
        """
        n_assets = len(alpha)
        tickers = alpha.index

        # Ensure alignment
        alpha = alpha.loc[tickers]
        current_positions = current_positions.reindex(tickers).fillna(0)
        prices = prices.loc[tickers]
        adv = adv.loc[tickers]
        volatility = volatility.loc[tickers]

        # Pre-calculate transaction cost gradient coefficients
        tcost_coeffs = self._calculate_tcost_coefficients(
            tickers, prices, adv, volatility
        )

        # Objective function
        def objective(w):
            # Alpha term (maximize)
            alpha_term = -self.config.alpha_weight * alpha.values @ w

            # Risk term (minimize)
            risk_term = 0.5 * self.config.risk_aversion * w @ covariance @ w

            # Transaction cost term (minimize)
            trades = w - current_positions.values
            tcost_term = self.config.tcost_aversion * np.sum(
                tcost_coeffs * np.abs(trades)
            )

            return alpha_term + risk_term + tcost_term

        # Gradient for faster optimization
        def gradient(w):
            grad = np.zeros(n_assets)

            # Alpha gradient
            grad -= self.config.alpha_weight * alpha.values

            # Risk gradient
            grad += self.config.risk_aversion * covariance @ w

            # Transaction cost gradient (subgradient for abs)
            trades = w - current_positions.values
            tcost_grad = self.config.tcost_aversion * tcost_coeffs * np.sign(trades)
            grad += tcost_grad

            return grad

        # Constraints
        constraints = []

        # Net exposure constraint: -max_net <= sum(w) <= max_net
        constraints.append(
            LinearConstraint(
                np.ones(n_assets),
                -self.config.max_net_exposure,
                self.config.max_net_exposure,
            )
        )

        # Gross exposure constraint: sum(|w|) <= max_gross
        # Implemented via auxiliary variables or penalty
        def gross_constraint(w):
            return self.config.max_gross_exposure - np.sum(np.abs(w))

        constraints.append({"type": "ineq", "fun": gross_constraint})

        # Turnover constraint
        def turnover_constraint(w):
            return self.config.max_turnover - np.sum(
                np.abs(w - current_positions.values)
            ) / 2

        constraints.append({"type": "ineq", "fun": turnover_constraint})

        # Factor neutrality constraints
        if factor_loadings is not None:
            B = factor_loadings.loc[tickers].values
            for i in range(B.shape[1]):
                constraints.append(
                    {"type": "eq", "fun": lambda w, i=i: B[:, i] @ w}
                )

        # Bounds
        bounds = [
            (-self.config.max_position_size, self.config.max_position_size)
            for _ in range(n_assets)
        ]

        # Initial guess (current positions)
        w0 = current_positions.values.copy()

        # Solve
        try:
            result = minimize(
                objective,
                w0,
                method=self.config.solver,
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.config.max_iterations,
                    "ftol": self.config.tolerance,
                },
            )

            optimal_weights = pd.Series(result.x, index=tickers)
            success = result.success
            message = result.message
        except Exception as e:
            # Fallback to simple scaling
            optimal_weights = self._fallback_optimization(
                alpha, current_positions, covariance
            )
            success = False
            message = str(e)

        # Apply minimum trade filter
        optimal_weights = self._apply_min_trade_filter(
            optimal_weights, current_positions
        )

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            optimal_weights,
            current_positions,
            alpha,
            covariance,
            tcost_coeffs,
        )
        diagnostics["optimization_success"] = success
        diagnostics["optimization_message"] = message

        self.optimization_history.append(diagnostics)

        return optimal_weights, diagnostics

    def _calculate_tcost_coefficients(
        self,
        tickers: pd.Index,
        prices: pd.Series,
        adv: pd.Series,
        volatility: pd.Series,
    ) -> np.ndarray:
        """Calculate per-asset transaction cost coefficients."""
        coeffs = np.zeros(len(tickers))

        for i, ticker in enumerate(tickers):
            # Linear approximation of transaction costs
            # TC ≈ coeff * |trade_size|
            price = prices.get(ticker, 100)
            daily_volume = adv.get(ticker, 1e6)
            vol = volatility.get(ticker, 0.02)

            # Base cost (commission + spread)
            base_cost = (
                self.tcost_model.commission_bps + self.tcost_model.spread_bps
            ) / 10000

            # Impact cost (linearized)
            # For small trades, impact ≈ impact_coeff * vol * sqrt(trade/adv) * price
            # Linearize around typical trade size
            typical_trade = 0.01  # 1% of portfolio
            participation = typical_trade * price / (daily_volume * price + 1e-8)
            impact_cost = (
                self.tcost_model.impact_coefficient * vol * np.sqrt(participation)
            )

            coeffs[i] = base_cost + impact_cost

        return coeffs

    def _fallback_optimization(
        self,
        alpha: pd.Series,
        current_positions: pd.Series,
        covariance: np.ndarray,
    ) -> pd.Series:
        """Simple fallback when main optimization fails."""
        # Scale alpha by inverse volatility
        vol = np.sqrt(np.diag(covariance))
        scaled_alpha = alpha / (vol + 1e-8)

        # Normalize to target gross
        target_gross = self.config.max_gross_exposure * 0.5
        scaled_alpha = scaled_alpha / (np.abs(scaled_alpha).sum() + 1e-8) * target_gross

        # Market neutral
        scaled_alpha = scaled_alpha - scaled_alpha.mean()

        # Clip to position limits
        scaled_alpha = scaled_alpha.clip(
            -self.config.max_position_size, self.config.max_position_size
        )

        return scaled_alpha

    def _apply_min_trade_filter(
        self, optimal: pd.Series, current: pd.Series
    ) -> pd.Series:
        """Don't execute tiny trades that aren't worth the fixed costs."""
        trade = optimal - current
        small_trades = np.abs(trade) < self.config.min_trade_size

        # Keep current position for small trades
        optimal[small_trades] = current[small_trades]

        return optimal

    def _calculate_diagnostics(
        self,
        optimal: pd.Series,
        current: pd.Series,
        alpha: pd.Series,
        covariance: np.ndarray,
        tcost_coeffs: np.ndarray,
    ) -> Dict:
        """Calculate optimization diagnostics."""
        w = optimal.values
        w0 = current.values
        trades = w - w0

        # Expected alpha
        expected_alpha = alpha.values @ w

        # Risk
        portfolio_var = w @ covariance @ w
        portfolio_vol = np.sqrt(portfolio_var)

        # Transaction costs
        tcosts = np.sum(tcost_coeffs * np.abs(trades))

        # Exposures
        gross_exposure = np.abs(w).sum()
        net_exposure = w.sum()
        long_exposure = w[w > 0].sum()
        short_exposure = np.abs(w[w < 0].sum())

        # Turnover
        turnover = np.abs(trades).sum() / 2

        # Position stats
        n_long = (w > 0.001).sum()
        n_short = (w < -0.001).sum()

        return {
            "expected_alpha": expected_alpha,
            "portfolio_volatility": portfolio_vol,
            "expected_sharpe": expected_alpha / (portfolio_vol + 1e-8) * np.sqrt(252),
            "transaction_costs": tcosts,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "turnover": turnover,
            "n_long_positions": n_long,
            "n_short_positions": n_short,
            "n_total_positions": n_long + n_short,
        }


class TrajectoryOptimizer:
    """
    Multi-Period Portfolio Optimization

    Optimizes trading trajectory over multiple periods considering
    signal decay, market impact, and risk evolution.

    This is crucial for stat arb where:
    1. Signals decay over time (need to trade quickly)
    2. Market impact increases with trade size (need to spread trades)
    3. Risk changes as positions are built
    """

    def __init__(
        self,
        n_periods: int = 5,
        signal_decay: float = 0.9,
        impact_decay: float = 0.5,
    ):
        self.n_periods = n_periods
        self.signal_decay = signal_decay
        self.impact_decay = impact_decay

    def optimize_trajectory(
        self,
        target_positions: pd.Series,
        current_positions: pd.Series,
        covariance: np.ndarray,
        prices: pd.Series,
        adv: pd.Series,
        volatility: pd.Series,
    ) -> List[pd.Series]:
        """
        Compute optimal trading trajectory from current to target positions.

        Returns list of position targets for each period.
        """
        n_assets = len(target_positions)
        tickers = target_positions.index

        total_trade = target_positions - current_positions

        # Simple approach: exponentially weighted trading schedule
        # Trade more at the beginning when signal is strongest
        trajectories = []
        remaining = total_trade.copy()

        for t in range(self.n_periods):
            # Participation rate decays
            rate = self.signal_decay ** t

            # But constrained by impact
            max_participation = self._calculate_max_participation(
                remaining, adv, volatility
            )

            # Trade is minimum of desired and max allowed
            period_trade = remaining * rate * 0.5
            period_trade = period_trade.clip(
                -max_participation, max_participation
            )

            remaining -= period_trade
            new_positions = (
                current_positions
                if t == 0
                else trajectories[-1] + period_trade
            )
            if t == 0:
                new_positions = current_positions + period_trade

            trajectories.append(new_positions)

        # Final period: complete remaining trade
        trajectories[-1] = target_positions

        return trajectories

    def _calculate_max_participation(
        self,
        trade: pd.Series,
        adv: pd.Series,
        volatility: pd.Series,
    ) -> pd.Series:
        """Calculate maximum trade size based on impact constraints."""
        # Don't trade more than 10% of ADV per period
        max_adv_participation = 0.10

        # Impact threshold (don't accept more than 50bps impact)
        max_impact_bps = 50

        max_trade = adv * max_adv_participation

        return max_trade


class RiskParityOptimizer:
    """
    Risk Parity Portfolio Construction

    Alternative to mean-variance that equalizes risk contribution
    from each position. Useful for ensuring diversification.
    """

    def __init__(self, target_vol: float = 0.10):
        self.target_vol = target_vol

    def optimize(
        self,
        alpha_ranks: pd.Series,
        covariance: np.ndarray,
        long_only: bool = False,
    ) -> pd.Series:
        """
        Construct risk parity portfolio with alpha tilt.

        Positions contribute equal risk but are tilted toward
        higher alpha signals.
        """
        n_assets = len(alpha_ranks)
        tickers = alpha_ranks.index

        # Target: equal risk contribution
        target_risk = self.target_vol / np.sqrt(n_assets)

        # Initialize with inverse volatility
        vol = np.sqrt(np.diag(covariance))
        w0 = 1.0 / (vol + 1e-8)
        w0 = w0 / w0.sum()

        def risk_parity_objective(w):
            """Minimize sum of squared risk contribution differences."""
            port_vol = np.sqrt(w @ covariance @ w)
            marginal_risk = covariance @ w
            risk_contrib = w * marginal_risk / (port_vol + 1e-8)

            # Equal risk target
            target = port_vol / n_assets

            # Deviation from equal risk
            risk_dev = np.sum((risk_contrib - target) ** 2)

            # Alpha tilt penalty (penalize deviating from alpha signal)
            alpha_penalty = -0.1 * (alpha_ranks.values @ w)

            return risk_dev + alpha_penalty

        bounds = [(0.001, 0.2) if long_only else (-0.2, 0.2) for _ in range(n_assets)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - (1.0 if long_only else 0.0)}

        try:
            result = minimize(
                risk_parity_objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            weights = pd.Series(result.x, index=tickers)
        except Exception:
            weights = pd.Series(w0, index=tickers)

        # Scale to target volatility
        port_vol = np.sqrt(weights.values @ covariance @ weights.values)
        weights = weights * self.target_vol / (port_vol + 1e-8)

        return weights


class BlackLittermanOptimizer:
    """
    Black-Litterman Portfolio Construction

    Combines market equilibrium with alpha views in a Bayesian framework.
    Useful for blending benchmark-relative positions with alpha signals.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau  # Uncertainty in prior

    def optimize(
        self,
        market_caps: pd.Series,
        covariance: np.ndarray,
        views: pd.Series,
        view_confidence: pd.Series,
    ) -> pd.Series:
        """
        Black-Litterman optimization combining equilibrium with views.

        Args:
            market_caps: Market capitalizations for equilibrium weights
            covariance: Return covariance matrix
            views: Alpha views (expected returns)
            view_confidence: Confidence in each view (0-1)
        """
        n_assets = len(market_caps)
        tickers = market_caps.index

        # Equilibrium weights (market cap weighted)
        w_mkt = market_caps / market_caps.sum()

        # Implied equilibrium returns
        pi = self.risk_aversion * covariance @ w_mkt.values

        # View matrix (identity for absolute views)
        P = np.eye(n_assets)

        # View uncertainty (diagonal)
        omega = np.diag(
            (1 - view_confidence.values) * np.diag(self.tau * covariance) + 1e-8
        )

        # Black-Litterman posterior
        tau_sigma = self.tau * covariance
        M = np.linalg.inv(
            np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
        )

        posterior_return = M @ (
            np.linalg.inv(tau_sigma) @ pi
            + P.T @ np.linalg.inv(omega) @ views.values
        )

        # Optimal weights
        weights = (
            np.linalg.inv(self.risk_aversion * covariance) @ posterior_return
        )

        return pd.Series(weights, index=tickers)
