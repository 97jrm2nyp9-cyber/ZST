"""
Utility Functions for Statistical Arbitrage

Helper functions for data processing, performance analysis,
and strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


def generate_synthetic_data(
    n_stocks: int = 100,
    n_days: int = 756,  # 3 years
    n_sectors: int = 11,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Generate synthetic market data for backtesting.

    Creates realistic price, volume, sector, and market cap data
    with embedded factor structure and mean-reverting relationships.

    Args:
        n_stocks: Number of stocks in universe
        n_days: Number of trading days
        n_sectors: Number of sectors
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices, volumes, sectors, market_caps)
    """
    np.random.seed(seed)

    # Generate tickers
    tickers = [f"STOCK_{i:03d}" for i in range(n_stocks)]

    # Assign sectors
    sectors = pd.Series(
        [f"SECTOR_{i % n_sectors}" for i in range(n_stocks)], index=tickers
    )

    # Generate dates
    dates = pd.date_range(end="2024-12-31", periods=n_days, freq="B")

    # Factor structure
    n_factors = 5
    factor_loadings = np.random.randn(n_stocks, n_factors) * 0.5
    factor_loadings[:, 0] = 1.0  # Market factor

    # Sector factor loadings
    for i, ticker in enumerate(tickers):
        sector_idx = i % n_sectors
        sector_loading = np.zeros(n_sectors)
        sector_loading[sector_idx] = 0.5
        factor_loadings[i, 1] = sector_loading.sum()

    # Generate factor returns
    factor_returns = np.random.randn(n_days, n_factors) * 0.01
    factor_returns[:, 0] *= 1.5  # Market has higher vol

    # Idiosyncratic returns with mean reversion
    idio_returns = np.zeros((n_days, n_stocks))
    idio_vol = np.random.uniform(0.01, 0.03, n_stocks)

    for t in range(1, n_days):
        # Mean-reverting component
        mean_reversion = -0.1 * idio_returns[t - 1]
        # New shock
        shock = np.random.randn(n_stocks) * idio_vol
        idio_returns[t] = mean_reversion + shock

    # Total returns
    total_returns = factor_loadings @ factor_returns.T + idio_returns.T
    total_returns = total_returns.T

    # Convert to prices
    log_prices = np.cumsum(total_returns, axis=0)
    # Scale to realistic price levels
    initial_prices = np.random.uniform(20, 200, n_stocks)
    log_prices = log_prices + np.log(initial_prices)
    prices = np.exp(log_prices)

    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

    # Generate volumes (correlated with volatility and price)
    base_volume = np.random.uniform(1e5, 1e7, n_stocks)
    vol_factor = 1 + 2 * np.abs(total_returns)  # Higher vol = higher volume
    volumes = base_volume * vol_factor
    volumes_df = pd.DataFrame(volumes, index=dates, columns=tickers)

    # Generate market caps
    base_mcap = np.random.uniform(1e9, 1e12, n_stocks)
    market_caps = prices * base_mcap / initial_prices
    market_caps_df = pd.DataFrame(market_caps, index=dates, columns=tickers)

    return prices_df, volumes_df, sectors, market_caps_df


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Series of daily returns

    Returns:
        Dict of performance metrics
    """
    returns = returns.dropna()

    if len(returns) < 20:
        return {"error": "Insufficient data"}

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)

    # Risk-adjusted
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

    # Win/loss stats
    hit_rate = (returns > 0).mean()
    win_avg = returns[returns > 0].mean() if (returns > 0).any() else 0
    loss_avg = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
    profit_factor = win_avg / loss_avg if loss_avg > 0 else np.inf

    # Higher moments
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)

    # VaR/CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

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
        "skewness": skewness,
        "kurtosis": kurtosis,
        "var_95_daily": var_95,
        "cvar_95_daily": cvar_95,
        "n_observations": len(returns),
        "n_years": n_years,
    }


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)


def calculate_information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, window: int = 252
) -> float:
    """Calculate Information Ratio vs benchmark."""
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    active_return = active_returns.mean() * 252
    return active_return / tracking_error if tracking_error > 0 else 0


def bootstrap_sharpe_confidence(
    returns: pd.Series, n_bootstrap: int = 1000, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for Sharpe ratio using bootstrap.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    returns = returns.dropna().values
    n = len(returns)

    sharpes = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        sharpe = sample.mean() / sample.std() * np.sqrt(252)
        sharpes.append(sharpe)

    point_estimate = returns.mean() / returns.std() * np.sqrt(252)
    lower = np.percentile(sharpes, (1 - confidence) / 2 * 100)
    upper = np.percentile(sharpes, (1 + confidence) / 2 * 100)

    return point_estimate, lower, upper


def calculate_turnover(positions: pd.DataFrame) -> pd.Series:
    """Calculate daily turnover from positions DataFrame."""
    changes = positions.diff().abs()
    turnover = changes.sum(axis=1) / 2  # One-way turnover
    return turnover


def calculate_factor_attribution(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> Dict[str, float]:
    """
    Attribute returns to factors using regression.

    Args:
        returns: Strategy returns
        factor_returns: DataFrame of factor returns

    Returns:
        Dict with alpha and factor exposures
    """
    # Align data
    common_idx = returns.index.intersection(factor_returns.index)
    y = returns.loc[common_idx].values
    X = factor_returns.loc[common_idx].values

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS regression
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Regression failed"}

    # R-squared
    residuals = y - X_with_const @ beta
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    result = {
        "alpha_daily": beta[0],
        "alpha_annualized": beta[0] * 252,
        "r_squared": r_squared,
    }

    for i, factor in enumerate(factor_returns.columns):
        result[f"beta_{factor}"] = beta[i + 1]

    return result


def format_performance_report(metrics: Dict[str, float]) -> str:
    """Format performance metrics as a readable report."""
    lines = [
        "=" * 50,
        "PERFORMANCE REPORT",
        "=" * 50,
        "",
        "RETURNS",
        f"  Total Return:        {metrics.get('total_return', 0):.2%}",
        f"  Annualized Return:   {metrics.get('annualized_return', 0):.2%}",
        f"  Annualized Vol:      {metrics.get('annualized_volatility', 0):.2%}",
        "",
        "RISK-ADJUSTED",
        f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}",
        f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}",
        f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}",
        "",
        "DRAWDOWN",
        f"  Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}",
        "",
        "TRADE STATISTICS",
        f"  Hit Rate:            {metrics.get('hit_rate', 0):.2%}",
        f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}",
        "",
        "TAIL RISK",
        f"  Skewness:            {metrics.get('skewness', 0):.2f}",
        f"  Kurtosis:            {metrics.get('kurtosis', 0):.2f}",
        f"  Daily VaR (95%):     {metrics.get('var_95_daily', 0):.2%}",
        f"  Daily CVaR (95%):    {metrics.get('cvar_95_daily', 0):.2%}",
        "",
        "DATA",
        f"  Observations:        {metrics.get('n_observations', 0):.0f}",
        f"  Years:               {metrics.get('n_years', 0):.2f}",
        "=" * 50,
    ]

    return "\n".join(lines)
