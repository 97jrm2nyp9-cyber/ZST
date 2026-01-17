"""
Forex Trading Strategy Framework

A multi-alpha forex trading strategy targeting 2+ Sharpe ratio through:
1. Carry trades (interest rate differentials)
2. Momentum/trend following
3. Mean reversion
4. Cross-rate arbitrage

Author: ZST Framework
"""

from .signals import (
    FxCarrySignal,
    FxMomentumSignal,
    FxMeanReversionSignal,
    FxCrossRateSignal,
    FxSignalCombiner
)
from .strategy import FxTradingStrategy
from .utils import FxConfig, generate_fx_data, calculate_fx_metrics
from .risk import FxRiskManager

__all__ = [
    'FxCarrySignal',
    'FxMomentumSignal',
    'FxMeanReversionSignal',
    'FxCrossRateSignal',
    'FxSignalCombiner',
    'FxTradingStrategy',
    'FxRiskManager',
    'FxConfig',
    'generate_fx_data',
    'calculate_fx_metrics'
]

__version__ = '1.0.0'
