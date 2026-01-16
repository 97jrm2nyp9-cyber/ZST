"""
Risk Management Module for Statistical Arbitrage

This module implements comprehensive risk management including:
1. Factor exposure neutralization (market, sector, style factors)
2. Position and concentration limits
3. Drawdown controls and circuit breakers
4. VaR and Expected Shortfall monitoring
5. Correlation regime detection

Key insight: Risk management is NOT about avoiding all risk - it's about
taking compensated risks while neutralizing uncompensated factor exposures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class RiskRegime(Enum):
    """Market risk regime classification."""

    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    # Position limits
    max_position_size: float = 0.02  # 2% of portfolio per position
    max_sector_exposure: float = 0.15  # 15% net sector exposure
    max_factor_exposure: float = 0.10  # 10% factor beta
    max_gross_exposure: float = 2.0  # 200% gross
    max_net_exposure: float = 0.10  # 10% net market exposure

    # Risk limits
    max_portfolio_vol: float = 0.10  # 10% annualized vol target
    max_daily_var_99: float = 0.02  # 2% daily VaR
    max_drawdown: float = 0.10  # 10% max drawdown trigger

    # Concentration
    min_positions: int = 50  # Minimum positions for diversification
    max_single_name_contribution: float = 0.05  # 5% risk contribution

    # Turnover
    max_daily_turnover: float = 0.30  # 30% one-way daily turnover


@dataclass
class FactorExposures:
    """Current factor exposure state."""

    market_beta: float = 0.0
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    size_exposure: float = 0.0
    value_exposure: float = 0.0
    momentum_exposure: float = 0.0
    volatility_exposure: float = 0.0
    quality_exposure: float = 0.0


class FactorModel:
    """
    Multi-Factor Risk Model

    Estimates factor exposures and covariance structure for risk
    decomposition and neutralization. Based on Barra-style factor models.
    """

    def __init__(
        self,
        estimation_window: int = 252,
        decay_factor: float = 0.97,
    ):
        self.estimation_window = estimation_window
        self.decay_factor = decay_factor
        self.factor_returns: Optional[pd.DataFrame] = None
        self.factor_covariance: Optional[np.ndarray] = None
        self.specific_variance: Optional[pd.Series] = None
        self.loadings: Optional[pd.DataFrame] = None

    def estimate(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
        sectors: Optional[pd.Series] = None,
        book_values: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Estimate factor model parameters.

        Uses cross-sectional regression to estimate factor returns
        and time-series analysis for factor covariance.
        """
        n_periods, n_assets = returns.shape

        # Build factor exposure matrix
        exposures = self._build_exposure_matrix(
            returns, market_caps, sectors, book_values
        )

        # Cross-sectional regression for factor returns
        self.factor_returns = self._estimate_factor_returns(returns, exposures)

        # Factor covariance with exponential decay
        self.factor_covariance = self._estimate_factor_covariance()

        # Specific (idiosyncratic) variance
        self.specific_variance = self._estimate_specific_variance(returns, exposures)

        # Store loadings
        self.loadings = exposures

    def _build_exposure_matrix(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[pd.DataFrame],
        sectors: Optional[pd.Series],
        book_values: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build factor exposure matrix for each asset."""
        tickers = returns.columns
        exposures = pd.DataFrame(index=tickers)

        # Market factor (all 1s)
        exposures["market"] = 1.0

        # Sector dummies
        if sectors is not None:
            for sector in sectors.unique():
                exposures[f"sector_{sector}"] = (sectors == sector).astype(float)

        # Size factor (log market cap, standardized)
        if market_caps is not None:
            log_mcap = np.log(market_caps.iloc[-1] + 1)
            exposures["size"] = (log_mcap - log_mcap.mean()) / (log_mcap.std() + 1e-8)
        else:
            exposures["size"] = 0.0

        # Value factor (B/M ratio)
        if book_values is not None and market_caps is not None:
            bm = book_values.iloc[-1] / (market_caps.iloc[-1] + 1e-8)
            exposures["value"] = (bm - bm.mean()) / (bm.std() + 1e-8)
        else:
            exposures["value"] = 0.0

        # Momentum (12-1 month return)
        if len(returns) > 252:
            mom = returns.iloc[-252:-21].sum()
            exposures["momentum"] = (mom - mom.mean()) / (mom.std() + 1e-8)
        else:
            exposures["momentum"] = 0.0

        # Short-term reversal
        if len(returns) > 5:
            reversal = -returns.iloc[-5:].sum()
            exposures["reversal"] = (reversal - reversal.mean()) / (
                reversal.std() + 1e-8
            )
        else:
            exposures["reversal"] = 0.0

        # Volatility (realized vol)
        if len(returns) > 21:
            vol = returns.iloc[-21:].std()
            exposures["volatility"] = (vol - vol.mean()) / (vol.std() + 1e-8)
        else:
            exposures["volatility"] = 0.0

        return exposures.fillna(0)

    def _estimate_factor_returns(
        self, returns: pd.DataFrame, exposures: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimate factor returns using cross-sectional regression."""
        factor_returns = pd.DataFrame(
            index=returns.index, columns=exposures.columns
        )

        X = exposures.values
        XtX_inv = np.linalg.pinv(X.T @ X)

        for date in returns.index:
            y = returns.loc[date].values
            valid_mask = ~np.isnan(y)

            if valid_mask.sum() < exposures.shape[1]:
                factor_returns.loc[date] = 0.0
                continue

            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            # WLS with market cap weights would go here
            try:
                beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                factor_returns.loc[date] = beta
            except np.linalg.LinAlgError:
                factor_returns.loc[date] = 0.0

        return factor_returns.astype(float)

    def _estimate_factor_covariance(self) -> np.ndarray:
        """Estimate factor covariance with exponential decay."""
        if self.factor_returns is None:
            raise ValueError("Must estimate factor returns first")

        returns = self.factor_returns.dropna().values
        n_periods, n_factors = returns.shape

        # Exponential decay weights
        weights = np.array(
            [self.decay_factor ** (n_periods - 1 - i) for i in range(n_periods)]
        )
        weights = weights / weights.sum()

        # Weighted covariance
        mean_returns = np.average(returns, axis=0, weights=weights)
        centered = returns - mean_returns

        cov = np.zeros((n_factors, n_factors))
        for i in range(n_periods):
            cov += weights[i] * np.outer(centered[i], centered[i])

        # Annualize
        return cov * 252

    def _estimate_specific_variance(
        self, returns: pd.DataFrame, exposures: pd.DataFrame
    ) -> pd.Series:
        """Estimate idiosyncratic variance for each asset."""
        if self.factor_returns is None:
            raise ValueError("Must estimate factor returns first")

        # Calculate residuals
        predicted = self.factor_returns.values @ exposures.T.values
        residuals = returns.values - predicted

        # Specific variance with decay
        n_periods = len(returns)
        weights = np.array(
            [self.decay_factor ** (n_periods - 1 - i) for i in range(n_periods)]
        )
        weights = weights / weights.sum()

        specific_var = np.zeros(len(returns.columns))
        for i in range(n_periods):
            specific_var += weights[i] * residuals[i] ** 2

        # Annualize
        return pd.Series(specific_var * 252, index=returns.columns)

    def get_portfolio_risk(
        self, weights: pd.Series
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk into factor and specific components.

        Returns total risk, factor risk, specific risk, and contribution
        of each factor.
        """
        if self.loadings is None or self.factor_covariance is None:
            raise ValueError("Must estimate factor model first")

        w = weights.values
        B = self.loadings.loc[weights.index].values
        F = self.factor_covariance
        D = np.diag(self.specific_variance.loc[weights.index].values)

        # Factor risk
        portfolio_factor_exposure = B.T @ w
        factor_variance = portfolio_factor_exposure @ F @ portfolio_factor_exposure

        # Specific risk
        specific_variance = w @ D @ w

        # Total variance
        total_variance = factor_variance + specific_variance
        total_vol = np.sqrt(total_variance)

        # Factor contributions
        factor_contrib = {}
        for i, factor in enumerate(self.loadings.columns):
            factor_exposure = (B[:, i] * w).sum()
            factor_contrib[factor] = factor_exposure * np.sqrt(F[i, i])

        return {
            "total_vol": total_vol,
            "factor_vol": np.sqrt(factor_variance),
            "specific_vol": np.sqrt(specific_variance),
            "factor_contributions": factor_contrib,
        }


class RiskManager:
    """
    Comprehensive Risk Management System

    Handles:
    1. Factor exposure neutralization
    2. Position sizing based on risk budget
    3. Drawdown monitoring and circuit breakers
    4. VaR/ES risk metrics
    5. Regime detection and adaptation
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        factor_model: Optional[FactorModel] = None,
    ):
        self.limits = limits or RiskLimits()
        self.factor_model = factor_model or FactorModel()
        self.current_exposures = FactorExposures()
        self.drawdown_tracker = DrawdownTracker()
        self.regime_detector = RegimeDetector()
        self.position_history: List[pd.Series] = []

    def check_limits(
        self,
        proposed_positions: pd.Series,
        current_positions: pd.Series,
        prices: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if proposed positions violate any risk limits.

        Returns (is_valid, violations_dict)
        """
        violations = {}

        # Position size limits
        max_pos = np.abs(proposed_positions).max()
        if max_pos > self.limits.max_position_size:
            violations["position_size"] = (
                f"Max position {max_pos:.2%} exceeds limit {self.limits.max_position_size:.2%}"
            )

        # Gross exposure
        gross = np.abs(proposed_positions).sum()
        if gross > self.limits.max_gross_exposure:
            violations["gross_exposure"] = (
                f"Gross exposure {gross:.2%} exceeds limit {self.limits.max_gross_exposure:.2%}"
            )

        # Net exposure (market beta proxy)
        net = proposed_positions.sum()
        if abs(net) > self.limits.max_net_exposure:
            violations["net_exposure"] = (
                f"Net exposure {net:.2%} exceeds limit {self.limits.max_net_exposure:.2%}"
            )

        # Sector exposure
        if sectors is not None:
            sector_exp = self._calculate_sector_exposure(proposed_positions, sectors)
            for sector, exp in sector_exp.items():
                if abs(exp) > self.limits.max_sector_exposure:
                    violations[f"sector_{sector}"] = (
                        f"Sector {sector} exposure {exp:.2%} exceeds limit"
                    )

        # Number of positions
        n_positions = (np.abs(proposed_positions) > 0.001).sum()
        if n_positions < self.limits.min_positions:
            violations["diversification"] = (
                f"Only {n_positions} positions, minimum is {self.limits.min_positions}"
            )

        # Turnover
        if current_positions is not None:
            turnover = np.abs(proposed_positions - current_positions).sum() / 2
            if turnover > self.limits.max_daily_turnover:
                violations["turnover"] = (
                    f"Turnover {turnover:.2%} exceeds limit {self.limits.max_daily_turnover:.2%}"
                )

        return len(violations) == 0, violations

    def neutralize_factors(
        self,
        raw_positions: pd.Series,
        returns: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Neutralize factor exposures while preserving alpha signal.

        Uses constrained optimization to find positions closest to
        raw signal while satisfying factor neutrality constraints.
        """
        # Estimate factor model if needed
        self.factor_model.estimate(returns, market_caps, sectors)

        tickers = raw_positions.index
        n_assets = len(tickers)

        # Get factor loadings
        B = self.factor_model.loadings.loc[tickers].values

        # Optimization: minimize ||w - w_raw||^2 subject to factor neutrality
        def objective(w):
            return np.sum((w - raw_positions.values) ** 2)

        def gradient(w):
            return 2 * (w - raw_positions.values)

        # Constraints: B'w = 0 for all factors, sum(w) = 0
        constraints = []

        # Market neutral
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w)})

        # Factor neutral (except market which is already handled)
        for i in range(1, B.shape[1]):
            constraints.append(
                {"type": "eq", "fun": lambda w, i=i: B[:, i] @ w}
            )

        # Bounds
        bounds = [
            (-self.limits.max_position_size, self.limits.max_position_size)
            for _ in range(n_assets)
        ]

        # Solve
        try:
            result = minimize(
                objective,
                raw_positions.values,
                method="SLSQP",
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )
            neutralized = pd.Series(result.x, index=tickers)
        except Exception:
            # Fallback: simple demean
            neutralized = raw_positions - raw_positions.mean()

        return neutralized

    def apply_risk_scaling(
        self,
        positions: pd.Series,
        returns: pd.DataFrame,
        target_vol: Optional[float] = None,
    ) -> pd.Series:
        """
        Scale positions to target portfolio volatility.

        Uses exponentially-weighted covariance for volatility estimation.
        """
        target_vol = target_vol or self.limits.max_portfolio_vol

        # Estimate covariance
        cov = self._estimate_covariance(returns)

        # Current portfolio volatility
        w = positions.values
        port_var = w @ cov @ w
        port_vol = np.sqrt(port_var)

        if port_vol < 1e-8:
            return positions

        # Scale factor
        scale = target_vol / port_vol

        # Don't scale up too much
        scale = min(scale, 2.0)

        return positions * scale

    def _estimate_covariance(
        self, returns: pd.DataFrame, decay: float = 0.97
    ) -> np.ndarray:
        """Estimate covariance with exponential decay."""
        returns = returns.dropna()
        n_periods, n_assets = returns.shape

        # Decay weights
        weights = np.array(
            [decay ** (n_periods - 1 - i) for i in range(n_periods)]
        )
        weights = weights / weights.sum()

        # Weighted covariance
        mean_ret = np.average(returns.values, axis=0, weights=weights)
        centered = returns.values - mean_ret

        cov = np.zeros((n_assets, n_assets))
        for i in range(n_periods):
            cov += weights[i] * np.outer(centered[i], centered[i])

        # Annualize
        return cov * 252

    def _calculate_sector_exposure(
        self, positions: pd.Series, sectors: pd.Series
    ) -> Dict[str, float]:
        """Calculate net exposure to each sector."""
        sector_exposure = {}

        for sector in sectors.unique():
            sector_stocks = sectors[sectors == sector].index
            sector_stocks = [s for s in sector_stocks if s in positions.index]
            sector_exposure[sector] = positions[sector_stocks].sum()

        return sector_exposure

    def calculate_var(
        self,
        positions: pd.Series,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk and Expected Shortfall.

        Uses both parametric (Gaussian) and historical simulation.
        """
        # Portfolio returns
        port_returns = (returns[positions.index] * positions).sum(axis=1)

        # Parametric VaR (Gaussian)
        port_mean = port_returns.mean() * horizon
        port_std = port_returns.std() * np.sqrt(horizon)
        z_score = stats.norm.ppf(1 - confidence)
        parametric_var = -(port_mean + z_score * port_std)

        # Historical VaR
        historical_var = -port_returns.quantile(1 - confidence) * np.sqrt(horizon)

        # Expected Shortfall (CVaR)
        var_threshold = port_returns.quantile(1 - confidence)
        es = -port_returns[port_returns <= var_threshold].mean() * np.sqrt(horizon)

        return {
            "parametric_var": parametric_var,
            "historical_var": historical_var,
            "expected_shortfall": es,
            "confidence": confidence,
            "horizon_days": horizon,
        }

    def get_current_regime(self, returns: pd.DataFrame) -> RiskRegime:
        """Detect current market regime based on volatility and correlation."""
        return self.regime_detector.detect(returns)

    def adjust_for_regime(
        self, positions: pd.Series, regime: RiskRegime
    ) -> pd.Series:
        """
        Adjust positions based on detected regime.

        In high vol regimes, reduce gross exposure.
        In crisis, dramatically reduce risk.
        """
        if regime == RiskRegime.LOW_VOL:
            scale = 1.2  # Slight increase
        elif regime == RiskRegime.NORMAL:
            scale = 1.0
        elif regime == RiskRegime.HIGH_VOL:
            scale = 0.7  # Reduce exposure
        elif regime == RiskRegime.CRISIS:
            scale = 0.3  # Significant reduction

        return positions * scale


class DrawdownTracker:
    """
    Track and manage drawdowns for circuit breaker logic.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        recovery_threshold: float = 0.05,
    ):
        self.max_drawdown = max_drawdown
        self.recovery_threshold = recovery_threshold
        self.peak_value = 1.0
        self.current_value = 1.0
        self.in_drawdown_mode = False
        self.drawdown_history: List[float] = []

    def update(self, daily_return: float) -> Dict[str, Union[float, bool]]:
        """Update tracker with daily return."""
        self.current_value *= 1 + daily_return
        self.peak_value = max(self.peak_value, self.current_value)

        current_dd = (self.peak_value - self.current_value) / self.peak_value
        self.drawdown_history.append(current_dd)

        # Check circuit breaker
        if current_dd > self.max_drawdown:
            self.in_drawdown_mode = True

        # Check recovery
        if self.in_drawdown_mode and current_dd < self.recovery_threshold:
            self.in_drawdown_mode = False

        return {
            "current_drawdown": current_dd,
            "peak_value": self.peak_value,
            "in_drawdown_mode": self.in_drawdown_mode,
            "max_historical_dd": max(self.drawdown_history),
        }

    def should_reduce_risk(self) -> bool:
        """Check if risk should be reduced due to drawdown."""
        return self.in_drawdown_mode

    def get_risk_multiplier(self) -> float:
        """Get risk multiplier based on drawdown state."""
        if not self.in_drawdown_mode:
            return 1.0

        current_dd = self.drawdown_history[-1] if self.drawdown_history else 0
        # Linear reduction as drawdown increases
        return max(0.2, 1.0 - current_dd / self.max_drawdown)


class RegimeDetector:
    """
    Market Regime Detection

    Uses volatility level, volatility of volatility, and correlation
    structure to classify market regimes.
    """

    def __init__(
        self,
        vol_lookback: int = 21,
        long_vol_lookback: int = 252,
    ):
        self.vol_lookback = vol_lookback
        self.long_vol_lookback = long_vol_lookback

    def detect(self, returns: pd.DataFrame) -> RiskRegime:
        """Detect current market regime."""
        # Calculate realized volatility
        recent_vol = returns.iloc[-self.vol_lookback :].std().mean() * np.sqrt(252)
        long_vol = returns.iloc[-self.long_vol_lookback :].std().mean() * np.sqrt(252)

        # Vol of vol (proxy for regime instability)
        rolling_vol = returns.rolling(self.vol_lookback).std().mean(axis=1)
        vol_of_vol = rolling_vol.iloc[-self.vol_lookback :].std()

        # Average correlation
        recent_corr = returns.iloc[-self.vol_lookback :].corr()
        avg_corr = recent_corr.values[np.triu_indices_from(recent_corr.values, k=1)].mean()

        # Classification logic
        vol_ratio = recent_vol / (long_vol + 1e-8)

        if vol_ratio < 0.7 and avg_corr < 0.3:
            return RiskRegime.LOW_VOL
        elif vol_ratio > 2.0 or avg_corr > 0.7:
            return RiskRegime.CRISIS
        elif vol_ratio > 1.3 or avg_corr > 0.5:
            return RiskRegime.HIGH_VOL
        else:
            return RiskRegime.NORMAL

    def get_regime_statistics(
        self, returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Get detailed statistics about current regime."""
        recent_vol = returns.iloc[-self.vol_lookback :].std().mean() * np.sqrt(252)
        long_vol = returns.iloc[-self.long_vol_lookback :].std().mean() * np.sqrt(252)

        recent_corr = returns.iloc[-self.vol_lookback :].corr()
        avg_corr = recent_corr.values[np.triu_indices_from(recent_corr.values, k=1)].mean()

        return {
            "recent_volatility": recent_vol,
            "long_term_volatility": long_vol,
            "vol_ratio": recent_vol / (long_vol + 1e-8),
            "average_correlation": avg_corr,
            "regime": self.detect(returns).value,
        }
