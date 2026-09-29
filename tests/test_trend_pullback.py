from datetime import UTC, datetime

from medium_trading.domain import Candle, StrategyContext
from medium_trading.strategy import TrendPullbackStrategy


def _candle() -> Candle:
    return Candle(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
    )


def test_strategy_does_not_signal_without_required_history() -> None:
    candle = _candle()
    context = StrategyContext(
        symbol="EUR/USD",
        candles_4h=(candle,),
        candles_1h=(candle,),
        candles_30m=(candle,),
    )

    assert TrendPullbackStrategy().evaluate(context) is None
