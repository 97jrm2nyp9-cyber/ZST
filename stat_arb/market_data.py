"""
Market Data Fetcher

This module provides functionality to fetch real market data from Yahoo Finance
and other data sources for use in the statistical arbitrage strategy.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import warnings

try:
    import pandas_datareader as pdr
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False
    warnings.warn("pandas_datareader not available, using alternative method")

import requests
from io import StringIO


class MarketDataFetcher:
    """
    Fetches market data from Yahoo Finance and other sources.

    Supports:
    - Historical price data
    - Volume data
    - Market capitalization
    - Sector classifications
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def fetch_yahoo_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ticker from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if HAS_DATAREADER:
            try:
                df = pdr.DataReader(ticker, 'yahoo', start_date, end_date)
                return df
            except Exception as e:
                print(f"Failed to fetch {ticker} via datareader: {e}")
                return None
        else:
            # Fallback to direct API call
            return self._fetch_yahoo_direct(ticker, start_date, end_date)

    def _fetch_yahoo_direct(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data directly from Yahoo Finance API.

        This is a fallback method when pandas_datareader is not available.
        """
        try:
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts = int(pd.Timestamp(end_date).timestamp())

            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'events': 'history',
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
            return df

        except Exception as e:
            print(f"Failed to fetch {ticker} directly: {e}")
            return None

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        price_col: str = 'Adj Close',
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        """
        Fetch data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            price_col: Which price column to use (default: 'Adj Close')

        Returns:
            Tuple of (prices, volumes, failed_tickers_dict)
        """
        prices_dict = {}
        volumes_dict = {}
        failed = {}

        print(f"Fetching data for {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(tickers)}")

            df = self.fetch_yahoo_data(ticker, start_date, end_date)

            if df is not None and len(df) > 0:
                prices_dict[ticker] = df[price_col]
                volumes_dict[ticker] = df['Volume']
            else:
                failed[ticker] = "No data returned"

        print(f"Successfully fetched {len(prices_dict)}/{len(tickers)} tickers")

        if failed:
            print(f"Failed tickers: {list(failed.keys())}")

        prices = pd.DataFrame(prices_dict)
        volumes = pd.DataFrame(volumes_dict)

        # Forward fill missing data (weekends, holidays)
        prices = prices.fillna(method='ffill')
        volumes = volumes.fillna(0)

        return prices, volumes, failed

    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 tickers from Wikipedia.

        Returns:
            List of ticker symbols
        """
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            # Clean tickers (remove dots for Yahoo Finance compatibility)
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e:
            print(f"Failed to fetch S&P 500 tickers: {e}")
            return []

    def get_sector_mapping(self, tickers: List[str]) -> pd.Series:
        """
        Get sector classifications for tickers.

        Note: This is a simplified version. For production use,
        you would want to use a more reliable data source.

        Args:
            tickers: List of ticker symbols

        Returns:
            Series mapping ticker to sector
        """
        # For now, return a simple mapping
        # In production, fetch from a proper data source
        sectors = {}
        sector_list = [
            'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
            'Communication Services', 'Industrials', 'Consumer Staples',
            'Energy', 'Utilities', 'Real Estate', 'Materials'
        ]

        for ticker in tickers:
            # Simple hash-based assignment for demo
            sectors[ticker] = sector_list[hash(ticker) % len(sector_list)]

        return pd.Series(sectors)

    def estimate_market_caps(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate market capitalizations.

        This is a rough approximation using price * volume as a proxy.
        For production use, fetch actual market cap data.

        Args:
            prices: Price data
            volumes: Volume data

        Returns:
            DataFrame of estimated market caps
        """
        # Use average dollar volume as a proxy for market cap
        # Scale it to realistic market cap range
        adv = (prices * volumes).rolling(20).mean()

        # Scale to billions (rough approximation)
        # Typical ADV is 0.1-1% of market cap
        market_caps = adv * 200

        return market_caps


def get_default_tickers(n_stocks: int = 100) -> List[str]:
    """
    Get a default list of large-cap US stocks.

    Returns a curated list of liquid, large-cap stocks suitable for
    statistical arbitrage.
    """
    # Large, liquid stocks across sectors
    default_tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
        'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'IBM', 'TXN', 'AMAT', 'MU', 'NOW',
        # Healthcare
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'MDT', 'ISRG', 'VRTX', 'REGN', 'HUM', 'ZTS',
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI',
        'CB', 'PGR', 'MMC', 'AON', 'USB', 'TFC', 'PNC', 'BK', 'COF', 'AIG',
        # Consumer Discretionary
        'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'MAR', 'ORLY',
        # Consumer Staples
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'KMB',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB',
        # Industrials
        'BA', 'CAT', 'GE', 'HON', 'UNP', 'RTX', 'LMT', 'UPS', 'DE', 'MMM',
        # Communication Services
        'DIS', 'CMCSA', 'NFLX', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'ATVI', 'TTWO',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED',
        # Real Estate
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'WELL', 'SPG', 'O', 'DLR', 'VICI',
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'DOW', 'PPG',
        # Additional large caps
        'V', 'MA', 'PYPL', 'ADSK', 'INTU', 'SHOP', 'SQ', 'ABNB', 'UBER', 'LYFT',
    ]

    return default_tickers[:n_stocks]


def fetch_market_data(
    tickers: Optional[List[str]] = None,
    n_stocks: int = 100,
    lookback_days: int = 500,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Convenience function to fetch market data for stat arb strategy.

    Args:
        tickers: List of tickers (if None, uses default large-cap stocks)
        n_stocks: Number of stocks to include (if using default)
        lookback_days: Number of days of history
        end_date: End date (defaults to today)

    Returns:
        Tuple of (prices, volumes, sectors, market_caps)
    """
    fetcher = MarketDataFetcher()

    # Get tickers
    if tickers is None:
        print(f"Using default large-cap stock universe...")
        tickers = get_default_tickers(n_stocks)

    # Calculate date range
    if end_date is None:
        end = datetime.now()
    else:
        end = pd.Timestamp(end_date)

    start = end - timedelta(days=lookback_days)

    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')

    print(f"Fetching data from {start_date} to {end_date}")

    # Fetch data
    prices, volumes, failed = fetcher.fetch_multiple_tickers(
        tickers, start_date, end_date
    )

    if len(prices.columns) == 0:
        raise ValueError("No data fetched successfully")

    # Get sectors
    print("Assigning sector classifications...")
    sectors = fetcher.get_sector_mapping(prices.columns.tolist())

    # Estimate market caps
    print("Estimating market capitalizations...")
    market_caps = fetcher.estimate_market_caps(prices, volumes)

    print(f"\nData Summary:")
    print(f"  Tickers: {len(prices.columns)}")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Sectors: {sectors.nunique()}")

    return prices, volumes, sectors, market_caps
