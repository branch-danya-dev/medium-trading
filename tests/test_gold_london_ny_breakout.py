from datetime import UTC, datetime

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import GoldLondonNewYorkBreakoutStrategy


def _candle(
    hour: int,
    minute: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=datetime(2025, 7, 7, hour, minute, tzinfo=UTC),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def _context(candles: tuple[Candle, ...]) -> StrategyContext:
    return StrategyContext(
        symbol="XAU/USD",
        candles_4h=(),
        candles_1h=(),
        candles_30m=candles,
    )


def _london_reference() -> tuple[Candle, ...]:
    return (
        _candle(7, 0, open_=3300, high=3304, low=3298, close=3302),
        _candle(7, 30, open_=3302, high=3306, low=3300, close=3304),
        _candle(8, 0, open_=3304, high=3307, low=3301, close=3303),
        _candle(8, 30, open_=3303, high=3305, low=3299, close=3301),
        _candle(9, 0, open_=3301, high=3304, low=3297, close=3300),
    )


def test_long_breakout_on_first_new_york_close_above_london_range() -> None:
    candles = _london_reference() + (
        _candle(12, 0, open_=3305, high=3309, low=3304, close=3308),
    )

    signal = GoldLondonNewYorkBreakoutStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 3297
    assert signal.target is None


def test_short_breakout_on_first_new_york_close_below_london_range() -> None:
    candles = _london_reference() + (
        _candle(12, 0, open_=3299, high=3300, low=3295, close=3296),
    )

    signal = GoldLondonNewYorkBreakoutStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.stop == 3307


def test_inside_range_does_not_signal() -> None:
    candles = _london_reference() + (
        _candle(12, 0, open_=3302, high=3304, low=3300, close=3303),
    )

    assert GoldLondonNewYorkBreakoutStrategy().evaluate(_context(candles)) is None


def test_missing_london_reference_bar_rejects_session() -> None:
    candles = _london_reference()[:-1] + (
        _candle(12, 0, open_=3305, high=3309, low=3304, close=3308),
    )

    assert GoldLondonNewYorkBreakoutStrategy().evaluate(_context(candles)) is None


def test_second_breakout_same_session_is_not_retried() -> None:
    candles = _london_reference() + (
        _candle(12, 0, open_=3305, high=3309, low=3304, close=3308),
        _candle(12, 30, open_=3308, high=3311, low=3307, close=3310),
    )

    assert GoldLondonNewYorkBreakoutStrategy().evaluate(_context(candles)) is None
