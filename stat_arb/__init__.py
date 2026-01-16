"""
Statistical Arbitrage Strategy Framework

A comprehensive framework for building market-neutral statistical arbitrage
strategies targeting 2+ Sharpe ratio for US equities.

Core Components:
- Signal Generation: Pairs trading, factor residuals, cross-sectional mean reversion
- Risk Management: Factor neutralization, position limits, drawdown control
- Portfolio Construction: Optimization with transaction cost penalties
- Execution: Slippage modeling and trade scheduling

Usage:
    from stat_arb import StatArbStrategy, create_strategy
    from stat_arb.utils import generate_synthetic_data

    # Quick start with synthetic data
    prices, volumes, sectors, market_caps = generate_synthetic_data()
    strategy = create_strategy(target_sharpe=2.0)
    result = strategy.run_backtest(prices, volumes, sectors, market_caps)
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "PairsSignal":
        from stat_arb.signals import PairsSignal
        return PairsSignal
    elif name == "FactorResidualSignal":
        from stat_arb.signals import FactorResidualSignal
        return FactorResidualSignal
    elif name == "CrossSectionalMeanReversion":
        from stat_arb.signals import CrossSectionalMeanReversion
        return CrossSectionalMeanReversion
    elif name == "EigenportfolioSignal":
        from stat_arb.signals import EigenportfolioSignal
        return EigenportfolioSignal
    elif name == "SignalCombiner":
        from stat_arb.signals import SignalCombiner
        return SignalCombiner
    elif name == "RiskManager":
        from stat_arb.risk import RiskManager
        return RiskManager
    elif name == "FactorModel":
        from stat_arb.risk import FactorModel
        return FactorModel
    elif name == "RiskLimits":
        from stat_arb.risk import RiskLimits
        return RiskLimits
    elif name == "PortfolioOptimizer":
        from stat_arb.portfolio import PortfolioOptimizer
        return PortfolioOptimizer
    elif name == "Backtester":
        from stat_arb.backtest import Backtester
        return Backtester
    elif name == "ExecutionModel":
        from stat_arb.execution import ExecutionModel
        return ExecutionModel
    elif name == "StatArbStrategy":
        from stat_arb.strategy import StatArbStrategy
        return StatArbStrategy
    elif name == "create_strategy":
        from stat_arb.strategy import create_strategy
        return create_strategy
    raise AttributeError(f"module 'stat_arb' has no attribute '{name}'")

__all__ = [
    "PairsSignal",
    "FactorResidualSignal",
    "CrossSectionalMeanReversion",
    "EigenportfolioSignal",
    "SignalCombiner",
    "RiskManager",
    "FactorModel",
    "RiskLimits",
    "PortfolioOptimizer",
    "Backtester",
    "ExecutionModel",
    "StatArbStrategy",
    "create_strategy",
]
