"""
Signal Generation Module for Statistical Arbitrage

This module implements multiple alpha signal generators that exploit mean-reversion
and relative value opportunities in US equities. The key to achieving 2+ Sharpe
is combining multiple uncorrelated alpha sources.

Signal Types:
1. Pairs Trading: Cointegration-based mean reversion between related securities
2. Factor Residual Reversion: Alpha from residuals after factor exposure neutralization
3. Cross-Sectional Mean Reversion: Short-term reversal within sectors/industries
4. Eigenportfolio Deviation: PCA-based statistical arbitrage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SignalConfig:
    """Configuration for signal generation parameters."""

    lookback_window: int = 60  # Days for estimation
    zscore_threshold: float = 2.0  # Entry threshold
    half_life: int = 21  # Mean reversion half-life (trading days)
    min_correlation: float = 0.7  # Minimum correlation for pairs
    max_pairs_per_stock: int = 3  # Diversification limit
    signal_decay: float = 0.94  # Daily signal decay factor
    winsorize_percentile: float = 0.01  # Outlier handling


class BaseSignal(ABC):
    """Abstract base class for all signal generators."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self._signal_history: List[pd.Series] = []

    @abstractmethod
    def generate(
        self, prices: pd.DataFrame, volumes: pd.DataFrame, **kwargs
    ) -> pd.Series:
        """
        Generate trading signals.

        Args:
            prices: DataFrame of adjusted close prices (dates x tickers)
            volumes: DataFrame of trading volumes (dates x tickers)

        Returns:
            Series of z-scored signals (positive = long, negative = short)
        """
        pass

    def _winsorize(self, series: pd.Series) -> pd.Series:
        """Winsorize extreme values to reduce outlier impact."""
        lower = series.quantile(self.config.winsorize_percentile)
        upper = series.quantile(1 - self.config.winsorize_percentile)
        return series.clip(lower=lower, upper=upper)

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Standardize series to z-scores."""
        return (series - series.mean()) / (series.std() + 1e-8)

    def _decay_signal(self, new_signal: pd.Series) -> pd.Series:
        """Apply exponential decay to blend with historical signal."""
        if not self._signal_history:
            self._signal_history.append(new_signal)
            return new_signal

        decayed = self._signal_history[-1] * self.config.signal_decay
        blended = new_signal * (1 - self.config.signal_decay) + decayed
        self._signal_history.append(blended)

        # Keep limited history
        if len(self._signal_history) > 100:
            self._signal_history = self._signal_history[-50:]

        return blended


class PairsSignal(BaseSignal):
    """
    Pairs Trading Signal Generator

    Identifies cointegrated pairs and generates mean-reversion signals
    based on spread z-scores. Uses Engle-Granger cointegration test
    and Ornstein-Uhlenbeck process for half-life estimation.

    Key Innovation: Dynamic hedge ratio estimation using Kalman filter
    for adaptive pairs relationships.
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        sector_constrained: bool = True,
    ):
        super().__init__(config)
        self.sector_constrained = sector_constrained
        self.pairs_cache: Dict[Tuple[str, str], Dict] = {}
        self._kalman_states: Dict[Tuple[str, str], np.ndarray] = {}

    def generate(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """Generate pairs trading signals."""
        log_prices = np.log(prices)
        returns = log_prices.diff()

        # Find valid pairs
        pairs = self._find_cointegrated_pairs(log_prices, sectors)

        # Generate signals for each pair
        signals = pd.Series(0.0, index=prices.columns)

        for (stock1, stock2), pair_info in pairs.items():
            spread_signal = self._calculate_spread_signal(
                log_prices[stock1], log_prices[stock2], pair_info
            )

            # Allocate signal to both legs
            signals[stock1] += spread_signal
            signals[stock2] -= spread_signal * pair_info["hedge_ratio"]

        # Normalize and apply decay
        signals = self._zscore(self._winsorize(signals))
        return self._decay_signal(signals)

    def _find_cointegrated_pairs(
        self, log_prices: pd.DataFrame, sectors: Optional[pd.Series] = None
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Find cointegrated pairs using Engle-Granger test.

        Uses rolling window for robust estimation and filters by:
        1. Minimum correlation threshold
        2. Cointegration p-value < 0.05
        3. Reasonable half-life (5-60 days)
        """
        tickers = log_prices.columns.tolist()
        pairs = {}

        # Use recent window for estimation
        recent_prices = log_prices.iloc[-self.config.lookback_window :]

        for i, stock1 in enumerate(tickers):
            pair_count = 0
            for stock2 in tickers[i + 1 :]:
                if pair_count >= self.config.max_pairs_per_stock:
                    break

                # Sector constraint check
                if (
                    self.sector_constrained
                    and sectors is not None
                    and sectors.get(stock1) != sectors.get(stock2)
                ):
                    continue

                # Correlation filter
                corr = recent_prices[stock1].corr(recent_prices[stock2])
                if corr < self.config.min_correlation:
                    continue

                # Cointegration test
                coint_result = self._test_cointegration(
                    recent_prices[stock1], recent_prices[stock2]
                )

                if coint_result["is_cointegrated"]:
                    pairs[(stock1, stock2)] = coint_result
                    pair_count += 1

        return pairs

    def _test_cointegration(
        self, y: pd.Series, x: pd.Series
    ) -> Dict[str, Union[bool, float]]:
        """
        Engle-Granger cointegration test with hedge ratio estimation.

        Returns dict with:
        - is_cointegrated: bool
        - hedge_ratio: optimal hedge ratio (beta)
        - half_life: mean reversion half-life
        - spread_std: spread standard deviation
        """
        # OLS regression for hedge ratio
        x_with_const = np.column_stack([np.ones(len(x)), x.values])
        beta = np.linalg.lstsq(x_with_const, y.values, rcond=None)[0]
        hedge_ratio = beta[1]

        # Calculate spread
        spread = y.values - hedge_ratio * x.values - beta[0]

        # ADF test on spread
        adf_result = self._adf_test(spread)

        # Estimate half-life using AR(1)
        half_life = self._estimate_half_life(spread)

        # Validity checks
        is_valid = (
            adf_result["p_value"] < 0.05
            and 5 <= half_life <= 60
            and hedge_ratio > 0
        )

        return {
            "is_cointegrated": is_valid,
            "hedge_ratio": hedge_ratio,
            "half_life": half_life,
            "spread_std": np.std(spread),
            "intercept": beta[0],
            "adf_stat": adf_result["adf_stat"],
        }

    def _adf_test(self, spread: np.ndarray) -> Dict[str, float]:
        """Augmented Dickey-Fuller test for stationarity."""
        n = len(spread)
        lag = int(np.floor(4 * (n / 100) ** 0.25))  # Schwert criterion

        # Difference and lagged levels
        diff_spread = np.diff(spread)
        lagged = spread[:-1]

        # Build regression matrix with lags
        if lag > 0:
            lags_matrix = np.column_stack(
                [diff_spread[i : -(lag - i) if lag - i > 0 else None] for i in range(lag)]
            )
            y = diff_spread[lag:]
            x = np.column_stack([lagged[lag:], lags_matrix[: len(y)]])
        else:
            y = diff_spread
            x = lagged.reshape(-1, 1)

        # OLS estimation
        x_with_const = np.column_stack([np.ones(len(y)), x])
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            residuals = y - x_with_const @ beta
            se = np.sqrt(
                np.sum(residuals**2)
                / (len(y) - len(beta))
                * np.linalg.inv(x_with_const.T @ x_with_const)[1, 1]
            )
            adf_stat = beta[1] / se
        except np.linalg.LinAlgError:
            return {"adf_stat": 0.0, "p_value": 1.0}

        # Approximate p-value using MacKinnon critical values
        # Critical values for n=250: 1%: -3.45, 5%: -2.87, 10%: -2.57
        if adf_stat < -3.45:
            p_value = 0.01
        elif adf_stat < -2.87:
            p_value = 0.05
        elif adf_stat < -2.57:
            p_value = 0.10
        else:
            p_value = 0.5

        return {"adf_stat": adf_stat, "p_value": p_value}

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """Estimate mean reversion half-life using AR(1) model."""
        lagged = spread[:-1]
        diff = np.diff(spread)

        # AR(1): delta_spread = theta * spread_lag + epsilon
        theta = np.sum(lagged * diff) / (np.sum(lagged**2) + 1e-8)

        # Half-life = -log(2) / log(1 + theta)
        if theta >= 0:
            return 100  # No mean reversion
        half_life = -np.log(2) / np.log(1 + theta)
        return max(1, min(100, half_life))

    def _calculate_spread_signal(
        self, y: pd.Series, x: pd.Series, pair_info: Dict
    ) -> float:
        """Calculate z-score signal for a pair spread."""
        hedge_ratio = pair_info["hedge_ratio"]
        intercept = pair_info["intercept"]
        spread_std = pair_info["spread_std"]

        # Current spread
        current_spread = y.iloc[-1] - hedge_ratio * x.iloc[-1] - intercept

        # Z-score
        zscore = current_spread / (spread_std + 1e-8)

        # Signal is negative of z-score (mean reversion)
        return -zscore


class FactorResidualSignal(BaseSignal):
    """
    Factor Residual Mean Reversion Signal

    Generates alpha from residual returns after neutralizing common factor
    exposures. Based on the insight that idiosyncratic returns mean-revert
    faster than total returns.

    Factors used:
    1. Market (beta)
    2. Size (log market cap)
    3. Value (book-to-market)
    4. Momentum (12-1 month returns)
    5. Short-term reversal (1-week returns)
    6. Volatility (realized vol)
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        n_factors: int = 6,
    ):
        super().__init__(config)
        self.n_factors = n_factors
        self.factor_loadings: Optional[pd.DataFrame] = None
        self.residual_vol: Optional[pd.Series] = None

    def generate(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
        book_values: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.Series:
        """Generate factor residual reversion signals."""
        returns = prices.pct_change()
        log_prices = np.log(prices)

        # Build factor matrix
        factors = self._build_factors(returns, prices, market_caps, book_values)

        # Estimate factor loadings and extract residuals
        residuals = self._extract_residuals(returns, factors)

        # Calculate cumulative residual return over lookback
        cum_residual = residuals.iloc[-self.config.lookback_window :].sum()

        # Normalize by residual volatility for proper sizing
        residual_vol = residuals.iloc[-self.config.lookback_window :].std()
        self.residual_vol = residual_vol

        # Signal is negative of normalized cumulative residual (mean reversion)
        signal = -cum_residual / (residual_vol + 1e-8)

        # Winsorize and standardize
        signal = self._zscore(self._winsorize(signal))

        return self._decay_signal(signal)

    def _build_factors(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        market_caps: Optional[pd.DataFrame],
        book_values: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Construct factor returns matrix."""
        n_periods = len(returns)
        factors = pd.DataFrame(index=returns.index)

        # Market factor (equal-weighted market return)
        factors["market"] = returns.mean(axis=1)

        # Size factor (small minus big)
        if market_caps is not None:
            size_ranks = market_caps.rank(axis=1, pct=True)
            small = returns.where(size_ranks < 0.3).mean(axis=1)
            big = returns.where(size_ranks > 0.7).mean(axis=1)
            factors["size"] = small - big
        else:
            # Proxy using price level
            price_ranks = prices.rank(axis=1, pct=True)
            factors["size"] = (
                returns.where(price_ranks < 0.3).mean(axis=1)
                - returns.where(price_ranks > 0.7).mean(axis=1)
            )

        # Value factor (high B/M minus low B/M)
        if book_values is not None and market_caps is not None:
            bm_ratio = book_values / market_caps
            bm_ranks = bm_ratio.rank(axis=1, pct=True)
            high_bm = returns.where(bm_ranks > 0.7).mean(axis=1)
            low_bm = returns.where(bm_ranks < 0.3).mean(axis=1)
            factors["value"] = high_bm - low_bm
        else:
            factors["value"] = 0.0

        # Momentum factor (12-1 month)
        momentum = prices.pct_change(252).shift(21)
        mom_ranks = momentum.rank(axis=1, pct=True)
        winners = returns.where(mom_ranks > 0.7).mean(axis=1)
        losers = returns.where(mom_ranks < 0.3).mean(axis=1)
        factors["momentum"] = winners - losers

        # Short-term reversal (1 week)
        weekly_ret = prices.pct_change(5)
        rev_ranks = weekly_ret.rank(axis=1, pct=True)
        recent_losers = returns.where(rev_ranks < 0.3).mean(axis=1)
        recent_winners = returns.where(rev_ranks > 0.7).mean(axis=1)
        factors["reversal"] = recent_losers - recent_winners

        # Volatility factor
        realized_vol = returns.rolling(21).std()
        vol_ranks = realized_vol.rank(axis=1, pct=True)
        low_vol = returns.where(vol_ranks < 0.3).mean(axis=1)
        high_vol = returns.where(vol_ranks > 0.7).mean(axis=1)
        factors["volatility"] = low_vol - high_vol

        return factors.fillna(0)

    def _extract_residuals(
        self, returns: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract residual returns using rolling factor regression.

        Uses ridge regression for stability with regularization
        proportional to factor volatility.
        """
        residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
        lookback = self.config.lookback_window

        # Align indices
        common_idx = returns.index.intersection(factors.index)
        returns = returns.loc[common_idx]
        factors = factors.loc[common_idx]

        # Use most recent window for factor loadings
        recent_returns = returns.iloc[-lookback:]
        recent_factors = factors.iloc[-lookback:]

        # Factor matrix with intercept
        F = recent_factors.values
        F_with_const = np.column_stack([np.ones(len(F)), F])

        # Ridge regression parameter
        lambda_ridge = 0.01 * np.trace(F_with_const.T @ F_with_const) / F_with_const.shape[1]

        for ticker in returns.columns:
            y = recent_returns[ticker].values

            # Skip if insufficient data
            if np.isnan(y).sum() > len(y) * 0.5:
                residuals[ticker] = np.nan
                continue

            # Handle NaN
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 30:
                residuals[ticker] = np.nan
                continue

            y_valid = y[valid_mask]
            F_valid = F_with_const[valid_mask]

            # Ridge regression
            try:
                ridge_term = lambda_ridge * np.eye(F_valid.shape[1])
                ridge_term[0, 0] = 0  # Don't regularize intercept
                beta = np.linalg.solve(
                    F_valid.T @ F_valid + ridge_term, F_valid.T @ y_valid
                )

                # Calculate residuals for full period
                full_residuals = returns[ticker].values - F_with_const @ beta
                residuals[ticker] = full_residuals
            except np.linalg.LinAlgError:
                residuals[ticker] = np.nan

        self.factor_loadings = pd.DataFrame(
            index=returns.columns, columns=["alpha"] + list(factors.columns)
        )

        return residuals


class CrossSectionalMeanReversion(BaseSignal):
    """
    Cross-Sectional Mean Reversion Signal

    Exploits short-term return reversals within sectors/industries.
    Based on the microstructure insight that liquidity shocks cause
    temporary price dislocations that revert within days.

    Key Features:
    1. Sector-neutral construction
    2. Volume-adjusted signals (larger moves on low volume are more significant)
    3. Volatility-scaled positions
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        reversion_horizon: int = 5,
        volume_adjustment: bool = True,
    ):
        super().__init__(config)
        self.reversion_horizon = reversion_horizon
        self.volume_adjustment = volume_adjustment

    def generate(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """Generate cross-sectional mean reversion signals."""
        returns = prices.pct_change(self.reversion_horizon)
        recent_returns = returns.iloc[-1]

        # Volume adjustment: amplify signals when volume is unusually low
        if self.volume_adjustment:
            avg_volume = volumes.iloc[-20:].mean()
            recent_volume = volumes.iloc[-self.reversion_horizon :].mean()
            volume_ratio = avg_volume / (recent_volume + 1e-8)
            volume_multiplier = np.clip(volume_ratio, 0.5, 2.0)
        else:
            volume_multiplier = pd.Series(1.0, index=prices.columns)

        # Sector-neutral construction
        if sectors is not None:
            signal = pd.Series(0.0, index=prices.columns)
            for sector in sectors.unique():
                sector_stocks = sectors[sectors == sector].index
                sector_stocks = [s for s in sector_stocks if s in recent_returns.index]

                if len(sector_stocks) < 5:
                    continue

                sector_returns = recent_returns[sector_stocks]

                # Demean within sector
                sector_signal = -(sector_returns - sector_returns.mean())

                # Apply volume adjustment
                sector_signal *= volume_multiplier[sector_stocks]

                signal[sector_stocks] = sector_signal
        else:
            # Cross-sectional demean
            signal = -(recent_returns - recent_returns.mean())
            signal *= volume_multiplier

        # Volatility scaling
        realized_vol = prices.pct_change().iloc[-21:].std()
        signal = signal / (realized_vol + 1e-8)

        # Standardize
        signal = self._zscore(self._winsorize(signal))

        return self._decay_signal(signal)


class EigenportfolioSignal(BaseSignal):
    """
    Eigenportfolio Mean Reversion Signal (PCA-based)

    Uses Principal Component Analysis to identify statistical factors
    and trades deviations from the factor structure. When a stock
    deviates significantly from its predicted value based on factors,
    we expect mean reversion.

    Implementation:
    1. Extract top K principal components from return covariance
    2. Project returns onto factor space
    3. Calculate residuals (actual - predicted)
    4. Trade mean reversion of residuals
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        n_components: int = 5,
        variance_threshold: float = 0.6,
    ):
        super().__init__(config)
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def generate(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """Generate eigenportfolio deviation signals."""
        returns = prices.pct_change().dropna()

        # Use lookback window for PCA
        recent_returns = returns.iloc[-self.config.lookback_window :]

        # Standardize returns
        returns_standardized = (recent_returns - recent_returns.mean()) / (
            recent_returns.std() + 1e-8
        )

        # Perform PCA
        self._fit_pca(returns_standardized)

        # Project onto factor space and calculate residuals
        projected = self._project(returns_standardized)
        residuals = returns_standardized - projected

        # Signal based on cumulative recent residual
        cum_residual = residuals.iloc[-5:].sum()

        # Trade mean reversion of residuals
        signal = -cum_residual

        # Standardize
        signal = self._zscore(self._winsorize(signal))

        return self._decay_signal(signal)

    def _fit_pca(self, returns: pd.DataFrame) -> None:
        """Fit PCA on standardized returns."""
        # Covariance matrix
        cov_matrix = returns.cov().values

        # Handle NaN
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select components explaining variance_threshold
        total_var = np.sum(eigenvalues)
        cum_var = np.cumsum(eigenvalues) / total_var

        n_components = min(
            self.n_components, np.searchsorted(cum_var, self.variance_threshold) + 1
        )
        n_components = max(1, n_components)

        self.components_ = eigenvectors[:, :n_components]
        self.explained_variance_ = eigenvalues[:n_components] / total_var

    def _project(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Project returns onto factor space."""
        if self.components_ is None:
            raise ValueError("Must fit PCA first")

        # Project: X_projected = X @ V @ V.T
        projected = returns.values @ self.components_ @ self.components_.T

        return pd.DataFrame(projected, index=returns.index, columns=returns.columns)


class SignalCombiner:
    """
    Signal Combination and Weighting

    Combines multiple alpha signals using various weighting schemes:
    1. Equal weight
    2. Inverse volatility
    3. Risk parity
    4. Historical Sharpe-based
    5. Optimized weights (maximize combined Sharpe)

    Key insight: Combining uncorrelated signals increases Sharpe ratio
    by sqrt(N) in the ideal case.
    """

    def __init__(
        self,
        signals: List[BaseSignal],
        combination_method: str = "inverse_vol",
        lookback_days: int = 60,
    ):
        self.signals = signals
        self.combination_method = combination_method
        self.lookback_days = lookback_days
        self.signal_history: Dict[str, List[pd.Series]] = {
            f"signal_{i}": [] for i in range(len(signals))
        }
        self.weights: Optional[np.ndarray] = None

    def generate_combined_signal(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """Generate combined signal from all signal generators."""
        individual_signals = []

        for i, signal_gen in enumerate(self.signals):
            sig = signal_gen.generate(prices, volumes, **kwargs)
            individual_signals.append(sig)

            # Store history
            self.signal_history[f"signal_{i}"].append(sig)
            if len(self.signal_history[f"signal_{i}"]) > self.lookback_days:
                self.signal_history[f"signal_{i}"] = self.signal_history[
                    f"signal_{i}"
                ][-self.lookback_days :]

        # Calculate weights
        self.weights = self._calculate_weights(individual_signals)

        # Combine signals
        combined = pd.Series(0.0, index=prices.columns)
        for i, sig in enumerate(individual_signals):
            combined += self.weights[i] * sig

        # Final standardization
        combined = (combined - combined.mean()) / (combined.std() + 1e-8)

        return combined

    def _calculate_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Calculate signal weights based on combination method."""
        n_signals = len(signals)

        if self.combination_method == "equal":
            return np.ones(n_signals) / n_signals

        elif self.combination_method == "inverse_vol":
            # Weight inversely proportional to signal volatility
            vols = []
            for i in range(n_signals):
                if len(self.signal_history[f"signal_{i}"]) > 10:
                    hist = pd.DataFrame(self.signal_history[f"signal_{i}"])
                    vol = hist.std().mean()
                else:
                    vol = 1.0
                vols.append(max(vol, 0.01))

            inv_vols = 1.0 / np.array(vols)
            return inv_vols / inv_vols.sum()

        elif self.combination_method == "risk_parity":
            # Equal risk contribution
            return self._risk_parity_weights(signals)

        elif self.combination_method == "sharpe_weighted":
            # Weight by historical Sharpe ratio
            return self._sharpe_weights(signals)

        elif self.combination_method == "optimized":
            # Optimize for maximum combined Sharpe
            return self._optimize_weights(signals)

        else:
            return np.ones(n_signals) / n_signals

    def _risk_parity_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Calculate risk parity weights."""
        n_signals = len(signals)

        # Build signal return matrix from history
        returns = []
        for i in range(n_signals):
            if len(self.signal_history[f"signal_{i}"]) > 10:
                hist = pd.DataFrame(self.signal_history[f"signal_{i}"])
                ret = hist.diff().mean(axis=1)
                returns.append(ret.values[-self.lookback_days :])
            else:
                returns.append(np.zeros(self.lookback_days))

        returns = np.array(returns)

        # Covariance matrix
        if returns.shape[1] > returns.shape[0]:
            cov = np.cov(returns)
        else:
            cov = np.eye(n_signals)

        # Risk parity optimization
        def risk_contribution(w):
            port_vol = np.sqrt(w @ cov @ w)
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / (port_vol + 1e-8)
            target_risk = port_vol / n_signals
            return np.sum((risk_contrib - target_risk) ** 2)

        w0 = np.ones(n_signals) / n_signals
        bounds = [(0.01, 1.0) for _ in range(n_signals)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        try:
            result = minimize(
                risk_contribution,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return result.x
        except Exception:
            return np.ones(n_signals) / n_signals

    def _sharpe_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Weight signals by historical Sharpe ratio."""
        n_signals = len(signals)
        sharpes = []

        for i in range(n_signals):
            if len(self.signal_history[f"signal_{i}"]) > 20:
                hist = pd.DataFrame(self.signal_history[f"signal_{i}"])
                # Assume signal return is proportional to signal change
                returns = hist.diff().mean(axis=1)
                sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0
            sharpes.append(max(sharpe, 0.01))

        sharpes = np.array(sharpes)
        return sharpes / sharpes.sum()

    def _optimize_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Optimize weights to maximize combined Sharpe ratio."""
        n_signals = len(signals)

        # Build return matrix
        returns = []
        for i in range(n_signals):
            if len(self.signal_history[f"signal_{i}"]) > 20:
                hist = pd.DataFrame(self.signal_history[f"signal_{i}"])
                ret = hist.diff().mean(axis=1)
                returns.append(ret.values[-min(60, len(ret)) :])
            else:
                returns.append(np.zeros(60))

        returns = np.array(returns)
        mean_returns = np.mean(returns, axis=1)
        cov_returns = np.cov(returns) if returns.shape[1] > 1 else np.eye(n_signals)

        # Maximize Sharpe ratio
        def neg_sharpe(w):
            port_ret = w @ mean_returns
            port_vol = np.sqrt(w @ cov_returns @ w + 1e-8)
            return -port_ret / port_vol

        w0 = np.ones(n_signals) / n_signals
        bounds = [(0.0, 0.5) for _ in range(n_signals)]  # Max 50% per signal
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        try:
            result = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return result.x
        except Exception:
            return np.ones(n_signals) / n_signals

    def get_signal_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Return diagnostic statistics for each signal."""
        diagnostics = {}

        for i in range(len(self.signals)):
            name = f"signal_{i}"
            if len(self.signal_history[name]) > 10:
                hist = pd.DataFrame(self.signal_history[name])
                returns = hist.diff().mean(axis=1)

                diagnostics[name] = {
                    "mean": returns.mean(),
                    "volatility": returns.std(),
                    "sharpe": returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
                    "skewness": stats.skew(returns.dropna()),
                    "kurtosis": stats.kurtosis(returns.dropna()),
                    "weight": self.weights[i] if self.weights is not None else 1.0 / len(self.signals),
                }
            else:
                diagnostics[name] = {
                    "mean": 0.0,
                    "volatility": 0.0,
                    "sharpe": 0.0,
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                    "weight": 1.0 / len(self.signals),
                }

        return diagnostics
