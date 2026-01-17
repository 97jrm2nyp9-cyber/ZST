"""
Forex Trading Signals

Multiple alpha signal generators for forex trading:
1. FxCarrySignal - Interest rate differential carry trades
2. FxMomentumSignal - Multi-timeframe trend following
3. FxMeanReversionSignal - Mean reversion trading
4. FxCrossRateSignal - Cross-rate arbitrage
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional
from .utils import FxConfig, zscore, winsorize


class BaseFxSignal(ABC):
    """Base class for forex trading signals"""

    def __init__(self, config: FxConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Exchange rates (dates × pairs)
        interest_rates : pd.DataFrame
            Interest rate differentials (dates × pairs)
        volatilities : pd.DataFrame
            Realized volatilities (dates × pairs)

        Returns
        -------
        signals : pd.DataFrame
            Trading signals in [-1, 1] range (dates × pairs)
        """
        pass


class FxCarrySignal(BaseFxSignal):
    """
    Carry Trade Signal based on interest rate differentials.

    Strategy: Long high-yielding currencies, short low-yielding currencies.
    The carry trade exploits the interest rate differential between currency pairs.

    Key Features:
    - Cross-sectional ranking of interest rate differentials
    - Volatility-adjusted for risk parity
    - Mean reversion overlay (fade extreme carry positions)
    """

    def generate(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate carry trade signals"""

        # Calculate average interest rate differential
        avg_ir = interest_rates.rolling(
            window=self.config.carry_lookback
        ).mean()

        # Volatility adjustment - scale by inverse volatility for risk parity
        # Using squared inverse vol for stronger emphasis on low vol opportunities
        vol_adj = 1 / ((volatilities / 100) ** 1.2).replace(0, np.nan)
        vol_adj = vol_adj.ffill().fillna(1)

        # Raw carry signal (normalized)
        carry_raw = avg_ir * vol_adj

        # Cross-sectional ranking and normalization
        carry_ranked = carry_raw.rank(axis=1, pct=True) - 0.5  # Center at 0
        carry_signal = carry_ranked * 2  # Scale to [-1, 1]

        # Mean reversion overlay - fade extreme positions
        # If carry spread is too wide historically, it may mean-revert
        # Reduced this effect as it may hurt performance
        carry_zscore = zscore(avg_ir, window=60)
        mean_reversion_fade = -np.tanh(carry_zscore / 4) * 0.15  # Reduced from 0.3

        # Combine
        final_signal = carry_signal + mean_reversion_fade
        final_signal = final_signal.clip(-1, 1)

        # Smooth the signal - longer smoothing for more stable carry positions
        final_signal = final_signal.ewm(span=7).mean()

        return final_signal.fillna(0)


class FxMomentumSignal(BaseFxSignal):
    """
    Multi-Timeframe Momentum Signal.

    Strategy: Follow trends across multiple timeframes.
    Currencies exhibit strong trending behavior - this signal exploits it.

    Key Features:
    - Short (5d), Medium (21d), Long (63d) momentum
    - Weighted combination favoring medium-term
    - Volatility breakout confirmation
    - Trend strength filtering
    """

    def generate(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate momentum signals"""

        # Calculate returns over multiple horizons
        returns_short = prices.pct_change(self.config.momentum_short)
        returns_medium = prices.pct_change(self.config.momentum_medium)
        returns_long = prices.pct_change(self.config.momentum_long)

        # Normalize by volatility for comparability
        vol_short = prices.pct_change().rolling(self.config.momentum_short).std()
        vol_medium = prices.pct_change().rolling(self.config.momentum_medium).std()
        vol_long = prices.pct_change().rolling(self.config.momentum_long).std()

        mom_short = (returns_short / vol_short).replace([np.inf, -np.inf], np.nan)
        mom_medium = (returns_medium / vol_medium).replace([np.inf, -np.inf], np.nan)
        mom_long = (returns_long / vol_long).replace([np.inf, -np.inf], np.nan)

        # Weighted combination - emphasize medium-term for cleaner trends
        # Reduced short-term weight to avoid noise whipsaw
        combined_momentum = (
            0.10 * mom_short +   # Reduced short-term (was 0.20)
            0.55 * mom_medium +  # Increased medium-term (was 0.50)
            0.35 * mom_long      # Increased long-term (was 0.30)
        )

        # Cross-sectional ranking
        momentum_ranked = combined_momentum.rank(axis=1, pct=True) - 0.5
        momentum_signal = momentum_ranked * 2  # Scale to [-1, 1]

        # Trend strength filter - only trade strong trends
        # Use ADX-like concept: compare directional movement to total movement
        price_changes = prices.diff()
        true_range = prices.pct_change().abs()

        directional_movement = price_changes.rolling(21).mean().abs()
        avg_true_range = true_range.rolling(21).mean()

        trend_strength = (directional_movement / avg_true_range).replace(
            [np.inf, -np.inf], np.nan
        )
        trend_strength = trend_strength.clip(0, 2) / 2  # Normalize to [0, 1]

        # Apply trend strength filter
        final_signal = momentum_signal * trend_strength

        # Smooth the signal
        final_signal = final_signal.ewm(span=3).mean()

        return final_signal.fillna(0)


class FxMeanReversionSignal(BaseFxSignal):
    """
    Mean Reversion Signal using Bollinger Bands and RSI.

    Strategy: Fade extreme deviations from the mean.
    Works well for range-bound currency pairs.

    Key Features:
    - Bollinger Band-based signals
    - RSI confirmation
    - Volatility regime filtering (works better in low vol)
    - Position reversal at extremes
    """

    def generate(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate mean reversion signals"""

        window = self.config.mean_reversion_window
        std_mult = self.config.mean_reversion_std

        # Calculate Bollinger Bands
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = ma + std_mult * std
        lower_band = ma - std_mult * std

        # Bollinger Band position: -1 (at lower), 0 (at MA), +1 (at upper)
        bb_position = (prices - ma) / (std * std_mult)
        bb_position = bb_position.clip(-1.5, 1.5)

        # Mean reversion signal: sell when price is high, buy when low
        mr_signal_bb = -np.tanh(bb_position)  # Reverse the position

        # RSI calculation (14-day)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # RSI signal: overbought (>70) -> sell, oversold (<30) -> buy
        rsi_signal = np.where(
            rsi > 70,
            -1,
            np.where(rsi < 30, 1, 0)
        )
        rsi_signal = pd.DataFrame(rsi_signal, index=prices.index, columns=prices.columns)

        # Combine BB and RSI
        mr_signal = 0.7 * mr_signal_bb + 0.3 * rsi_signal
        mr_signal = mr_signal.clip(-1, 1)

        # Volatility regime filter - mean reversion works better in low vol
        # Calculate percentile of current volatility
        vol_percentile = volatilities.rolling(window=60).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x)
        )

        # More aggressive vol filtering - only trade mean reversion in low vol
        # This is crucial because mean reversion fails in trending/volatile markets
        vol_filter = np.where(
            vol_percentile > 0.6,  # Above median vol
            0.0,  # No mean reversion in high vol (was 0.5)
            np.where(
                vol_percentile > 0.4,  # Medium vol
                0.5,  # Reduced signal
                1.0   # Full signal in low vol
            )
        )
        vol_filter = pd.DataFrame(vol_filter, index=prices.index, columns=prices.columns)

        # Also require RSI confirmation - don't trade BB signal alone
        rsi_confirmation = (rsi_signal.abs() > 0).astype(float)
        # Blend: require some RSI confirmation for stronger conviction
        confirmation_boost = 0.5 + 0.5 * rsi_confirmation

        final_signal = mr_signal * vol_filter * confirmation_boost

        # Smooth the signal
        final_signal = final_signal.ewm(span=3).mean()

        return final_signal.fillna(0)


class FxCrossRateSignal(BaseFxSignal):
    """
    Cross-Rate Arbitrage Signal.

    Strategy: Exploit pricing inefficiencies in currency crosses.
    Example: EUR/USD * USD/JPY should equal EUR/JPY

    Key Features:
    - Triangular arbitrage opportunities
    - Cross-rate mispricing
    - Correlation-based pair relationships
    - Quick mean reversion (high frequency alpha)
    """

    def generate(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate cross-rate arbitrage signals"""

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Define major crosses and their components
        # Format: (cross_pair, base_pair, quote_pair, multiplier)
        cross_relationships = [
            ('EUR/JPY', 'EUR/USD', 'USD/JPY', 1.0),  # EUR/JPY = EUR/USD * USD/JPY
            ('GBP/JPY', 'GBP/USD', 'USD/JPY', 1.0),  # GBP/JPY = GBP/USD * USD/JPY
            ('EUR/GBP', 'EUR/USD', 'GBP/USD', -1.0),  # EUR/GBP = EUR/USD / GBP/USD
        ]

        for cross, base, quote, mult in cross_relationships:
            if cross in prices.columns and base in prices.columns and quote in prices.columns:
                # Calculate implied cross rate
                if mult > 0:
                    implied_cross = prices[base] * prices[quote]
                else:
                    implied_cross = prices[base] / prices[quote]

                actual_cross = prices[cross]

                # Calculate mispricing (log ratio)
                mispricing = np.log(actual_cross / implied_cross)

                # Z-score of mispricing
                mispricing_zscore = zscore(pd.DataFrame(mispricing), window=20).iloc[:, 0]

                # Trade the mispricing: if actual > implied (positive zscore), sell cross
                # This is mean-reversion: fade the mispricing
                cross_signal = -np.tanh(mispricing_zscore / 2)

                signals[cross] = cross_signal

        # Correlation-based signals for pairs that move together
        # Calculate rolling correlation matrix
        returns = prices.pct_change()

        # For pairs with high correlation, look for temporary divergences
        window = 20
        for i, pair1 in enumerate(prices.columns):
            for pair2 in prices.columns[i+1:]:
                # Calculate correlation
                corr = returns[pair1].rolling(window).corr(returns[pair2])

                # For highly correlated pairs (>0.7), trade divergences
                if corr.mean() > 0.7:
                    # Calculate spread
                    # Normalize prices first
                    norm_price1 = prices[pair1] / prices[pair1].rolling(60).mean()
                    norm_price2 = prices[pair2] / prices[pair2].rolling(60).mean()

                    spread = norm_price1 - norm_price2
                    spread_zscore = zscore(pd.DataFrame(spread), window=20).iloc[:, 0]

                    # If spread is wide, expect convergence
                    # Sell pair1 (expensive), buy pair2 (cheap)
                    divergence_signal1 = -np.tanh(spread_zscore / 2) * 0.3
                    divergence_signal2 = np.tanh(spread_zscore / 2) * 0.3

                    # Add to existing signals (blend)
                    signals[pair1] = signals[pair1] + divergence_signal1
                    signals[pair2] = signals[pair2] + divergence_signal2

        # Clip to [-1, 1]
        signals = signals.clip(-1, 1)

        # Smooth the signal
        signals = signals.ewm(span=2).mean()

        return signals.fillna(0)


class FxSignalCombiner:
    """
    Combine multiple forex signals with various weighting schemes.

    Weighting methods:
    - equal: Equal weight to all signals
    - sharpe: Weight by historical Sharpe ratio
    - inverse_vol: Weight by inverse volatility (risk parity)
    - custom: Use provided weights
    - adaptive: Dynamic weighting based on recent signal performance

    Features:
    - Signal conviction filtering (ignore weak signals)
    - Adaptive weighting based on rolling Sharpe
    - Signal decay for stale signals
    """

    def __init__(self, config: FxConfig):
        self.config = config
        self.signals = {
            'carry': FxCarrySignal(config),
            'momentum': FxMomentumSignal(config),
            'mean_reversion': FxMeanReversionSignal(config),
            'cross_rate': FxCrossRateSignal(config)
        }
        # Signal conviction threshold - signals below this are zeroed out
        self.conviction_threshold = getattr(config, 'conviction_threshold', 0.15)
        # Lookback for adaptive weighting
        self.adaptive_lookback = getattr(config, 'adaptive_lookback', 60)

    def combine(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame,
        method: str = 'custom'
    ) -> pd.DataFrame:
        """
        Combine signals using specified method.

        Parameters
        ----------
        prices : pd.DataFrame
            Exchange rates
        interest_rates : pd.DataFrame
            Interest rate differentials
        volatilities : pd.DataFrame
            Realized volatilities
        method : str
            Weighting method: 'equal', 'sharpe', 'inverse_vol', 'custom'

        Returns
        -------
        combined_signal : pd.DataFrame
            Combined trading signals
        """
        # Generate all signals
        generated_signals = {}
        for name, signal_gen in self.signals.items():
            generated_signals[name] = signal_gen.generate(
                prices, interest_rates, volatilities
            )

        # Calculate weights
        if method == 'equal':
            weights = {name: 1.0 / len(self.signals) for name in self.signals}

        elif method == 'custom':
            weights = self.config.signal_weights

        elif method == 'sharpe':
            # Calculate historical Sharpe for each signal
            # Use Sharpe^1.5 to emphasize better signals (Kelly-like)
            returns = prices.pct_change()
            sharpes = {}

            for name, signal in generated_signals.items():
                # Calculate returns from signal
                signal_returns = (signal.shift(1) * returns).mean(axis=1)
                signal_returns = signal_returns.dropna()

                if len(signal_returns) > 0:
                    sharpe = (
                        signal_returns.mean() / signal_returns.std() * np.sqrt(252)
                        if signal_returns.std() > 0 else 0
                    )
                    # Only positive Sharpes, apply power to emphasize winners
                    sharpes[name] = max(sharpe, 0) ** 1.5
                else:
                    sharpes[name] = 0

            # Normalize to sum to 1
            total_sharpe = sum(sharpes.values())
            weights = {
                name: sharpe / total_sharpe if total_sharpe > 0 else 0
                for name, sharpe in sharpes.items()
            }

        elif method == 'inverse_vol':
            # Weight by inverse volatility of signal returns
            returns = prices.pct_change()
            inv_vols = {}

            for name, signal in generated_signals.items():
                signal_returns = (signal.shift(1) * returns).mean(axis=1)
                signal_returns = signal_returns.dropna()

                if len(signal_returns) > 0 and signal_returns.std() > 0:
                    inv_vols[name] = 1.0 / signal_returns.std()
                else:
                    inv_vols[name] = 1.0

            # Normalize
            total_inv_vol = sum(inv_vols.values())
            weights = {
                name: iv / total_inv_vol
                for name, iv in inv_vols.items()
            }

        elif method == 'adaptive':
            # Adaptive weighting based on rolling Sharpe ratio
            # This method dynamically adjusts weights based on recent performance
            # Key improvement: aggressively exclude negative Sharpe signals
            returns = prices.pct_change()
            lookback = self.adaptive_lookback

            # Calculate rolling Sharpe for each signal
            rolling_sharpes = {}
            for name, signal in generated_signals.items():
                signal_returns = (signal.shift(1) * returns).mean(axis=1)

                # Rolling Sharpe (annualized)
                rolling_mean = signal_returns.rolling(lookback).mean()
                rolling_std = signal_returns.rolling(lookback).std()

                rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
                # Only use positive Sharpe signals - zero out negative ones
                # This is the key improvement for excluding bad signals
                rolling_sharpe = rolling_sharpe.clip(lower=0.0)
                # Square the Sharpe to emphasize better signals (similar to Kelly criterion)
                rolling_sharpe = rolling_sharpe ** 1.5
                rolling_sharpes[name] = rolling_sharpe.fillna(0)

            # Normalize weights at each time step
            sharpe_df = pd.DataFrame(rolling_sharpes)
            # Add small epsilon to avoid division by zero
            total_sharpe = sharpe_df.sum(axis=1) + 1e-6

            # Time-varying weights
            adaptive_weights = sharpe_df.div(total_sharpe, axis=0)

            # If all Sharpes are zero/low, fall back to custom weights
            # but exclude signals with historically negative performance
            zero_mask = total_sharpe < 0.1
            fallback_weights = self.config.signal_weights.copy()
            # For fallback, also zero out mean_reversion if its recent Sharpe is negative
            for name in self.signals:
                adaptive_weights.loc[zero_mask, name] = fallback_weights.get(name, 0.25)

            # Smooth the weights to avoid rapid switching
            adaptive_weights = adaptive_weights.ewm(span=20).mean()

            # Combine signals with time-varying weights
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for name, signal in generated_signals.items():
                weight_series = adaptive_weights[name]
                # Broadcast weights across currency pairs
                for col in combined.columns:
                    combined[col] += weight_series * signal[col]

            # Apply conviction filtering and return
            combined = self._apply_conviction_filter(combined)
            return combined.clip(-1, 1)

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Combine signals
        combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for name, signal in generated_signals.items():
            weight = weights.get(name, 0)
            combined += weight * signal

        # Apply conviction filtering
        combined = self._apply_conviction_filter(combined)

        # Ensure combined signal is in [-1, 1]
        combined = combined.clip(-1, 1)

        return combined

    def _apply_conviction_filter(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply conviction filter to zero out weak signals.

        This reduces noise trading by only acting on signals above a threshold.
        Weak signals often generate turnover without meaningful alpha.

        Parameters
        ----------
        signals : pd.DataFrame
            Raw combined signals

        Returns
        -------
        filtered_signals : pd.DataFrame
            Signals with weak values zeroed out
        """
        threshold = self.conviction_threshold

        # Zero out signals below threshold (absolute value)
        filtered = signals.copy()
        weak_signals = filtered.abs() < threshold
        filtered[weak_signals] = 0.0

        # Scale remaining signals to use full range
        # This ensures we maintain position sizing power for strong signals
        strong_signals = ~weak_signals
        if strong_signals.any().any():
            # Rescale: map [threshold, 1] to [0, 1] preserving sign
            sign = np.sign(filtered)
            magnitude = filtered.abs()
            # Linear rescaling for signals above threshold
            rescaled_magnitude = (magnitude - threshold) / (1 - threshold)
            rescaled_magnitude = rescaled_magnitude.clip(0, 1)
            filtered = sign * rescaled_magnitude

        return filtered

    def get_individual_signals(
        self,
        prices: pd.DataFrame,
        interest_rates: pd.DataFrame,
        volatilities: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Generate and return all individual signals"""
        signals = {}
        for name, signal_gen in self.signals.items():
            signals[name] = signal_gen.generate(
                prices, interest_rates, volatilities
            )
        return signals
