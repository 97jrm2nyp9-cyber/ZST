"""
Forex Risk Management

Risk management, position sizing, and portfolio construction for FX strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from enum import Enum
from .utils import FxConfig, parse_currency_pair


class VolatilityRegime(Enum):
    """Market volatility regimes"""
    LOW = "low_volatility"
    NORMAL = "normal_volatility"
    HIGH = "high_volatility"
    CRISIS = "crisis"


class FxRiskManager:
    """
    Comprehensive risk management for forex trading.

    Features:
    - Correlation-aware volatility targeting
    - Position limits (per pair, per currency)
    - Leverage constraints
    - Smooth drawdown management
    - Volatility regime detection
    - Currency exposure tracking
    - Position smoothing to reduce turnover
    """

    def __init__(self, config: FxConfig):
        self.config = config
        self.current_regime = VolatilityRegime.NORMAL
        self.peak_equity = 1.0
        self.current_drawdown = 0.0
        # Position smoothing parameter (higher = smoother, less turnover)
        self.position_smoothing = getattr(config, 'position_smoothing', 0.3)
        # Previous positions for smoothing
        self.prev_positions = None
        # Rolling correlation matrix
        self.correlation_matrix = None
        self.correlation_lookback = getattr(config, 'correlation_lookback', 60)

    def detect_regime(
        self,
        volatilities: pd.DataFrame,
        lookback: int = 60
    ) -> VolatilityRegime:
        """
        Detect current volatility regime.

        Parameters
        ----------
        volatilities : pd.DataFrame
            Realized volatilities (annualized %)
        lookback : int
            Lookback period for regime detection

        Returns
        -------
        regime : VolatilityRegime
            Current volatility regime
        """
        # Get recent volatility
        recent_vol = volatilities.iloc[-lookback:].mean().mean()
        long_term_vol = volatilities.mean().mean()

        # Calculate volatility ratio
        vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1.0

        # Regime classification
        if vol_ratio < 0.7:
            regime = VolatilityRegime.LOW
        elif vol_ratio < 1.3:
            regime = VolatilityRegime.NORMAL
        elif vol_ratio < 2.0:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.CRISIS

        self.current_regime = regime
        return regime

    def scale_for_regime(self, regime: VolatilityRegime) -> float:
        """
        Get position scaling factor for volatility regime.

        Parameters
        ----------
        regime : VolatilityRegime
            Current volatility regime

        Returns
        -------
        scale : float
            Position scaling factor
        """
        # More aggressive scaling to exploit low vol and protect in high vol
        scaling = {
            VolatilityRegime.LOW: 1.4,       # Increase positions more in low vol
            VolatilityRegime.NORMAL: 1.1,    # Slight boost in normal
            VolatilityRegime.HIGH: 0.6,      # More defensive in high vol
            VolatilityRegime.CRISIS: 0.2     # Drastically reduce in crisis
        }
        return scaling.get(regime, 1.0)

    def update_drawdown(self, current_equity: float):
        """
        Update drawdown tracking.

        Parameters
        ----------
        current_equity : float
            Current portfolio equity value
        """
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown
        self.current_drawdown = (current_equity - self.peak_equity) / self.peak_equity

    def get_drawdown_scale(self) -> float:
        """
        Get position scaling based on current drawdown.

        Uses smooth scaling function instead of step-wise to avoid
        abrupt position changes at thresholds, which improves Sharpe
        by reducing turnover and path-dependent behavior.

        Returns
        -------
        scale : float
            Position scaling factor (0 to 1)
        """
        dd = abs(self.current_drawdown)
        max_dd = self.config.max_drawdown

        if dd >= max_dd:
            # Exceeded max drawdown - stop trading
            return 0.0

        # Smooth scaling using exponential decay
        # At dd=0: scale=1.0, at dd=max_dd: scale approaches 0
        # The decay rate is tuned to start reducing earlier for better risk control
        # This replaces the step-wise function with a smooth curve

        # Exponential decay: scale = exp(-k * dd / max_dd)
        # k controls steepness; k=3 means at 50% of max_dd, scale is ~22%
        k = 2.5  # Tuned for gradual reduction
        scale = np.exp(-k * (dd / max_dd) ** 1.5)

        # Floor at 0.1 to allow some trading even in drawdown
        # (complete stop only at max_dd)
        scale = max(scale, 0.1)

        return scale

    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        current_equity: float = 1.0,
        prices: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate position sizes from signals with risk management.

        Parameters
        ----------
        signals : pd.DataFrame
            Raw trading signals in [-1, 1] range
        volatilities : pd.DataFrame
            Realized volatilities (annualized %)
        current_equity : float
            Current portfolio value
        prices : pd.DataFrame, optional
            Price data for correlation calculation

        Returns
        -------
        positions : pd.DataFrame
            Position sizes (fraction of equity per pair)
        """
        # Update drawdown
        self.update_drawdown(current_equity)

        # Detect regime
        regime = self.detect_regime(volatilities)

        # Get scaling factors
        regime_scale = self.scale_for_regime(regime)
        dd_scale = self.get_drawdown_scale()
        total_scale = regime_scale * dd_scale

        # Volatility targeting: scale positions to achieve target portfolio volatility
        target_vol = self.config.target_volatility

        # Start with base positions from signals
        positions = signals.copy()

        # Get recent volatility (last row of volatilities DataFrame)
        if len(volatilities) > 0:
            recent_vol = volatilities.iloc[-1] / 100  # Convert % to decimal
        else:
            recent_vol = pd.Series(0.10, index=signals.columns)  # Default 10%

        # Calculate expected portfolio volatility with correlation awareness
        # This is more accurate than assuming independence
        if prices is not None and len(prices) >= self.correlation_lookback:
            # Update correlation matrix from recent returns
            returns = prices.pct_change().dropna()
            if len(returns) >= self.correlation_lookback:
                recent_returns = returns.iloc[-self.correlation_lookback:]
                self.correlation_matrix = recent_returns.corr()

        if self.correlation_matrix is not None and len(positions) > 0:
            # Use correlation-aware portfolio volatility calculation
            expected_portfolio_vol = self._calculate_portfolio_vol_with_correlation(
                positions, recent_vol
            )
        else:
            # Fallback: simplified calculation (assume independence)
            position_vol_contributions = (positions.abs() * recent_vol) ** 2
            expected_portfolio_vol = np.sqrt(position_vol_contributions.sum(axis=1))

        # Scale positions to achieve target volatility
        # Avoid division by zero
        vol_scale = pd.Series(1.0, index=positions.index)
        nonzero = expected_portfolio_vol > 0.001
        vol_scale[nonzero] = target_vol / expected_portfolio_vol[nonzero]

        # Cap the vol_scale to prevent excessive leverage
        vol_scale = vol_scale.clip(upper=2.0)

        # Broadcast vol_scale across all columns
        for col in positions.columns:
            positions[col] = positions[col] * vol_scale

        # Apply regime and drawdown scaling
        positions = positions * total_scale

        # Apply position limits
        positions = self._apply_position_limits(positions)

        # Apply currency exposure limits
        positions = self._apply_currency_limits(positions)

        # Apply leverage constraints
        positions = self._apply_leverage_constraints(positions)

        # Apply position smoothing to reduce turnover
        positions = self._apply_position_smoothing(positions)

        return positions

    def _calculate_portfolio_vol_with_correlation(
        self,
        positions: pd.DataFrame,
        volatilities: pd.Series
    ) -> pd.Series:
        """
        Calculate portfolio volatility accounting for correlations.

        Uses the formula: σ_p = sqrt(w' Σ w)
        where w is the position vector and Σ is the covariance matrix.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes
        volatilities : pd.Series
            Asset volatilities

        Returns
        -------
        portfolio_vol : pd.Series
            Portfolio volatility for each time step
        """
        portfolio_vols = []

        # Ensure correlation matrix columns match position columns
        common_cols = [c for c in positions.columns if c in self.correlation_matrix.columns]

        if len(common_cols) == 0:
            # Fallback to independence assumption
            position_vol_contributions = (positions.abs() * volatilities) ** 2
            return np.sqrt(position_vol_contributions.sum(axis=1))

        corr_matrix = self.correlation_matrix.loc[common_cols, common_cols]

        for idx in positions.index:
            pos = positions.loc[idx, common_cols].values
            vols = volatilities.loc[common_cols].values if hasattr(volatilities, 'loc') else \
                   np.array([volatilities.get(c, 0.1) for c in common_cols])

            # Covariance matrix: Σ = diag(σ) @ ρ @ diag(σ)
            cov_matrix = np.outer(vols, vols) * corr_matrix.values

            # Portfolio variance: w' Σ w
            port_var = pos @ cov_matrix @ pos

            portfolio_vols.append(np.sqrt(max(port_var, 0)))

        return pd.Series(portfolio_vols, index=positions.index)

    def _apply_position_smoothing(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply exponential smoothing to positions to reduce turnover.

        This reduces transaction costs and whipsaw losses by gradually
        transitioning between positions rather than jumping immediately.

        Parameters
        ----------
        positions : pd.DataFrame
            New target positions

        Returns
        -------
        smoothed_positions : pd.DataFrame
            Smoothed positions
        """
        if self.position_smoothing <= 0 or self.prev_positions is None:
            # No smoothing or first call - save positions and return
            if len(positions) > 0:
                self.prev_positions = positions.iloc[-1].copy()
            return positions

        alpha = 1 - self.position_smoothing  # Higher smoothing = lower alpha

        smoothed = positions.copy()

        for idx in positions.index:
            target = positions.loc[idx]
            # Exponential smoothing: new = alpha * target + (1-alpha) * prev
            smoothed.loc[idx] = alpha * target + (1 - alpha) * self.prev_positions

            # Update previous for next iteration
            self.prev_positions = smoothed.loc[idx].copy()

        return smoothed

    def _apply_position_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply per-pair position size limits"""
        max_pos = self.config.max_position_size

        # Clip each position
        positions = positions.clip(-max_pos, max_pos)

        return positions

    def _apply_currency_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply per-currency exposure limits.

        Each individual currency (e.g., EUR, USD, JPY) should not exceed
        max_currency_exposure across all pairs.
        """
        max_exposure = self.config.max_currency_exposure

        # Calculate exposure for each currency
        currency_exposure = self._calculate_currency_exposure(positions)

        # Scale down positions if any currency exceeds limit
        for currency, exposure in currency_exposure.items():
            if abs(exposure) > max_exposure:
                scale = max_exposure / abs(exposure)

                # Scale down all positions involving this currency
                for pair in positions.columns:
                    base, quote = parse_currency_pair(pair)

                    if currency in [base, quote]:
                        positions[pair] *= scale

        return positions

    def _calculate_currency_exposure(self, positions: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate net exposure for each individual currency.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes

        Returns
        -------
        exposures : dict
            Currency -> net exposure mapping
        """
        # Get latest positions (last row)
        if len(positions) == 0:
            return {}

        latest_positions = positions.iloc[-1]

        exposures = {}

        for pair, position in latest_positions.items():
            base, quote = parse_currency_pair(pair)

            # Long the pair = long base, short quote
            # Short the pair = short base, long quote

            # Base currency exposure
            if base not in exposures:
                exposures[base] = 0
            exposures[base] += position

            # Quote currency exposure
            if quote not in exposures:
                exposures[quote] = 0
            exposures[quote] -= position

        return exposures

    def _apply_leverage_constraints(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply gross and net leverage constraints.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes

        Returns
        -------
        positions : pd.DataFrame
            Constrained positions
        """
        # Calculate gross and net exposure
        gross = positions.abs().sum(axis=1)
        net = positions.sum(axis=1).abs()

        # Check if constraints are violated
        max_gross = self.config.max_gross_leverage
        max_net = self.config.max_net_exposure

        # Scale down if gross leverage exceeded
        gross_scale = np.where(gross > max_gross, max_gross / gross, 1.0)

        # Scale down if net exposure exceeded
        net_scale = np.where(net > max_net, max_net / net, 1.0)

        # Apply most restrictive constraint
        scale = np.minimum(gross_scale, net_scale)

        # Broadcast scale to all columns
        scale_df = pd.DataFrame(
            np.tile(scale.reshape(-1, 1), (1, positions.shape[1])),
            index=positions.index,
            columns=positions.columns
        )

        positions = positions * scale_df

        return positions

    def calculate_risk_metrics(
        self,
        positions: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> Dict:
        """
        Calculate current risk metrics.

        Parameters
        ----------
        positions : pd.DataFrame
            Current positions
        volatilities : pd.DataFrame
            Realized volatilities

        Returns
        -------
        metrics : dict
            Risk metrics
        """
        if len(positions) == 0:
            return {
                'gross_exposure': 0,
                'net_exposure': 0,
                'num_positions': 0,
                'portfolio_volatility': 0,
                'var_95': 0,
                'currency_exposures': {}
            }

        # Latest positions
        latest = positions.iloc[-1]

        # Exposure metrics
        gross = latest.abs().sum()
        net = latest.sum()
        num_positions = (latest.abs() > 0.001).sum()

        # Portfolio volatility (simplified - assumes independence)
        latest_vol = volatilities.iloc[-1] / 100  # Convert to decimal
        position_vol_contrib = (latest.abs() * latest_vol) ** 2
        portfolio_vol = np.sqrt(position_vol_contrib.sum())

        # Value at Risk (95% confidence, 1-day)
        var_95 = 1.645 * portfolio_vol  # 95% quantile of normal distribution

        # Currency exposures
        currency_exposures = self._calculate_currency_exposure(positions)

        return {
            'gross_exposure': gross,
            'net_exposure': abs(net),
            'num_positions': num_positions,
            'portfolio_volatility': portfolio_vol * np.sqrt(252),  # Annualized
            'var_95': var_95,
            'currency_exposures': currency_exposures,
            'regime': self.current_regime.value,
            'drawdown': self.current_drawdown
        }


def calculate_transaction_costs(
    positions: pd.DataFrame,
    prev_positions: pd.DataFrame,
    config: FxConfig
) -> float:
    """
    Calculate transaction costs for position changes.

    Parameters
    ----------
    positions : pd.DataFrame
        Current positions
    prev_positions : pd.DataFrame
        Previous positions
    config : FxConfig
        Strategy configuration

    Returns
    -------
    cost : float
        Transaction cost as fraction of equity
    """
    # Calculate position turnover
    turnover = (positions - prev_positions).abs().sum(axis=1)

    # Total cost = spread + commission
    total_cost_bps = config.spread_bps + config.commission_bps

    # Convert to fraction
    cost = turnover * (total_cost_bps / 10000)

    return cost
