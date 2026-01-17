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
    - Volatility targeting
    - Position limits (per pair, per currency)
    - Leverage constraints
    - Drawdown management
    - Volatility regime detection
    - Currency exposure tracking
    """

    def __init__(self, config: FxConfig):
        self.config = config
        self.current_regime = VolatilityRegime.NORMAL
        self.peak_equity = 1.0
        self.current_drawdown = 0.0

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
        scaling = {
            VolatilityRegime.LOW: 1.2,      # Increase positions in low vol
            VolatilityRegime.NORMAL: 1.0,    # Normal positions
            VolatilityRegime.HIGH: 0.7,      # Reduce positions in high vol
            VolatilityRegime.CRISIS: 0.3     # Drastically reduce in crisis
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

        Returns
        -------
        scale : float
            Position scaling factor (0 to 1)
        """
        dd = abs(self.current_drawdown)

        if dd < 0.05:  # Less than 5% drawdown
            return 1.0
        elif dd < 0.10:  # 5-10% drawdown
            return 0.8
        elif dd < self.config.max_drawdown:  # 10-15% drawdown
            return 0.5
        else:  # Exceeded max drawdown
            return 0.0  # Stop trading

    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        current_equity: float = 1.0
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

        # Calculate expected portfolio volatility (simplified: assume independence)
        # Portfolio vol ≈ sqrt(sum of (position × vol)^2)
        position_vol_contributions = (positions.abs() * recent_vol) ** 2
        expected_portfolio_vol = np.sqrt(position_vol_contributions.sum(axis=1))

        # Scale positions to achieve target volatility
        # Avoid division by zero
        vol_scale = pd.Series(1.0, index=positions.index)
        nonzero = expected_portfolio_vol > 0.001
        vol_scale[nonzero] = target_vol / expected_portfolio_vol[nonzero]

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

        return positions

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
