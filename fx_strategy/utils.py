"""
Forex Trading Utilities

Data generation, metrics calculation, and helper functions for FX strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FxConfig:
    """Configuration for forex trading strategy"""
    # Universe
    currency_pairs: list = None

    # Signal parameters
    carry_lookback: int = 20  # Days for interest rate estimation
    momentum_short: int = 5   # Short-term momentum window
    momentum_medium: int = 21  # Medium-term momentum window
    momentum_long: int = 63   # Long-term momentum window
    mean_reversion_window: int = 20  # Bollinger band window
    mean_reversion_std: float = 2.0  # Bollinger band std

    # Risk management
    target_volatility: float = 0.10  # 10% annualized
    max_gross_leverage: float = 3.0  # 300% gross (FX allows higher leverage)
    max_net_exposure: float = 0.30  # 30% net directional exposure
    max_position_size: float = 0.15  # 15% per currency pair
    max_currency_exposure: float = 0.50  # 50% per individual currency
    max_drawdown: float = 0.15  # 15% circuit breaker

    # Transaction costs
    spread_bps: float = 2.0  # 2 bps spread (tight for majors)
    commission_bps: float = 0.1  # 0.1 bps commission

    # Signal weights
    signal_weights: Dict[str, float] = None

    # Sharpe enhancement parameters
    conviction_threshold: float = 0.15  # Min signal strength to trade
    adaptive_lookback: int = 60  # Days for adaptive signal weighting
    position_smoothing: float = 0.3  # Position smoothing factor (0-1)
    correlation_lookback: int = 60  # Days for correlation estimation

    def __post_init__(self):
        if self.currency_pairs is None:
            # G10 major pairs
            self.currency_pairs = [
                'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
                'NZD/USD', 'USD/CAD', 'USD/CHF', 'EUR/GBP',
                'EUR/JPY', 'GBP/JPY'
            ]

        if self.signal_weights is None:
            self.signal_weights = {
                'carry': 0.30,
                'momentum': 0.25,
                'mean_reversion': 0.25,
                'cross_rate': 0.20
            }


def generate_fx_data(
    pairs: list,
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic forex data for backtesting.

    Parameters
    ----------
    pairs : list
        List of currency pairs (e.g., ['EUR/USD', 'GBP/USD'])
    start_date : str
        Start date for data generation
    end_date : str
        End date for data generation
    seed : int
        Random seed for reproducibility

    Returns
    -------
    prices : pd.DataFrame
        Exchange rates (dates × pairs)
    interest_rates : pd.DataFrame
        Interest rate differentials in % (dates × pairs)
    volatilities : pd.DataFrame
        Realized volatilities (dates × pairs)
    """
    np.random.seed(seed)

    # Generate date range (business days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    n_pairs = len(pairs)

    # Initialize prices and rates
    prices = pd.DataFrame(index=dates, columns=pairs)
    interest_rates = pd.DataFrame(index=dates, columns=pairs)
    volatilities = pd.DataFrame(index=dates, columns=pairs)

    # Base parameters for each pair
    base_rates = {
        'EUR/USD': 1.1000, 'GBP/USD': 1.3000, 'USD/JPY': 110.00,
        'AUD/USD': 0.7000, 'NZD/USD': 0.6500, 'USD/CAD': 1.2500,
        'USD/CHF': 0.9200, 'EUR/GBP': 0.8500, 'EUR/JPY': 121.00,
        'GBP/JPY': 143.00
    }

    # Interest rate differentials (annualized %)
    base_interest_rates = {
        'EUR/USD': -1.5, 'GBP/USD': -0.5, 'USD/JPY': 2.0,
        'AUD/USD': 1.0, 'NZD/USD': 1.5, 'USD/CAD': -0.3,
        'USD/CHF': 1.8, 'EUR/GBP': -1.0, 'EUR/JPY': 0.5,
        'GBP/JPY': 1.5
    }

    # Volatility levels (annualized %)
    base_vols = {
        'EUR/USD': 0.08, 'GBP/USD': 0.10, 'USD/JPY': 0.09,
        'AUD/USD': 0.12, 'NZD/USD': 0.13, 'USD/CAD': 0.08,
        'USD/CHF': 0.09, 'EUR/GBP': 0.08, 'EUR/JPY': 0.10,
        'GBP/JPY': 0.12
    }

    for pair in pairs:
        # Get base values
        price = base_rates.get(pair, 1.0)
        ir_diff = base_interest_rates.get(pair, 0.0)
        vol = base_vols.get(pair, 0.10)

        # Generate price path with drift (carry) and diffusion
        # dS/S = (r_domestic - r_foreign)dt + σdW
        dt = 1/252  # Daily
        drift = ir_diff / 100  # Convert % to decimal

        # Add momentum regime shifts
        regime_changes = np.random.randint(60, 120, size=n_days//90 + 1)
        momentum = np.zeros(n_days)
        for i, duration in enumerate(regime_changes):
            start_idx = sum(regime_changes[:i])
            end_idx = min(start_idx + duration, n_days)
            if end_idx <= n_days:
                momentum[start_idx:end_idx] = np.random.choice(
                    [-0.02, -0.01, 0, 0.01, 0.02],
                    p=[0.15, 0.20, 0.30, 0.20, 0.15]
                )

        # Add mean reversion component
        log_prices = np.zeros(n_days)
        log_prices[0] = np.log(price)

        mean_reversion_speed = 0.05
        long_term_mean = np.log(price)

        # Stochastic volatility (simple GARCH-like)
        vols = np.zeros(n_days)
        vols[0] = vol
        vol_of_vol = 0.3
        vol_mean_reversion = 0.1

        for t in range(1, n_days):
            # Update volatility
            vol_shock = np.random.normal(0, vol_of_vol * np.sqrt(dt))
            vols[t] = vols[t-1] + vol_mean_reversion * (vol - vols[t-1]) * dt + vol_shock
            vols[t] = max(vols[t], 0.03)  # Floor volatility

            # Update price with carry, momentum, and mean reversion
            total_drift = (
                drift +  # Carry
                momentum[t] +  # Momentum regime
                mean_reversion_speed * (long_term_mean - log_prices[t-1])  # Mean reversion
            )

            diffusion = vols[t] * np.random.normal(0, np.sqrt(dt))
            log_prices[t] = log_prices[t-1] + total_drift * dt + diffusion

        # Convert back to levels
        prices[pair] = np.exp(log_prices)

        # Interest rates with slow variation
        ir_noise = np.random.normal(0, 0.2, n_days)
        ir_series = ir_diff + np.cumsum(ir_noise * 0.01)
        ir_series = np.clip(ir_series, -5, 5)  # Reasonable bounds
        interest_rates[pair] = ir_series

        # Realized volatility
        volatilities[pair] = vols * 100  # Convert to percentage

    return prices, interest_rates, volatilities


def calculate_fx_metrics(
    returns: pd.Series,
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    config: FxConfig
) -> Dict:
    """
    Calculate comprehensive performance metrics for FX strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    positions : pd.DataFrame
        Daily positions (dates × pairs)
    prices : pd.DataFrame
        Exchange rates
    config : FxConfig
        Strategy configuration

    Returns
    -------
    metrics : dict
        Performance statistics
    """
    # Remove any NaN values
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }

    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

    # Risk metrics
    volatility = returns.std() * np.sqrt(252)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-10

    # Risk-adjusted returns
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Turnover
    position_changes = positions.diff().abs().sum(axis=1)
    avg_turnover = position_changes.mean()

    # Exposure metrics
    gross_exposure = positions.abs().sum(axis=1).mean()
    net_exposure = positions.sum(axis=1).abs().mean()

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_daily_turnover': avg_turnover,
        'avg_gross_exposure': gross_exposure,
        'avg_net_exposure': net_exposure,
        'num_trades': len(returns),
        'best_day': returns.max(),
        'worst_day': returns.min()
    }


def parse_currency_pair(pair: str) -> Tuple[str, str]:
    """
    Parse currency pair into base and quote currencies.

    Parameters
    ----------
    pair : str
        Currency pair like 'EUR/USD'

    Returns
    -------
    base : str
        Base currency
    quote : str
        Quote currency
    """
    parts = pair.split('/')
    return parts[0], parts[1]


def get_inverse_pair(pair: str) -> str:
    """Get inverse of a currency pair (EUR/USD -> USD/EUR)"""
    base, quote = parse_currency_pair(pair)
    return f"{quote}/{base}"


def calculate_cross_rate(pair1_price: float, pair2_price: float, target_pair: str) -> float:
    """
    Calculate implied cross rate from two currency pairs.

    Example: EUR/USD * USD/JPY = EUR/JPY
    """
    # This is a simplified version - real implementation would need to handle
    # all possible combinations and directions
    return pair1_price * pair2_price


def winsorize(data: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize data at specified percentiles"""
    return data.clip(
        lower=data.quantile(lower),
        upper=data.quantile(upper),
        axis=1
    )


def zscore(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate rolling z-score"""
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return (data - mean) / std.replace(0, np.nan)
