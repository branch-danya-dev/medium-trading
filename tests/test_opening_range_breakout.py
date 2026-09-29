from datetime import UTC, datetime

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import (
    OpeningRangeBreakoutQualityStrategy,
    OpeningRangeBreakoutStrategy,
)


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
        symbol="USA500.IDX/USD",
        candles_4h=(),
        candles_1h=(),
        candles_30m=candles,
    )


def test_long_signal_on_first_close_above_opening_range() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6215, low=6200, close=6212),
    )

    signal = OpeningRangeBreakoutStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 6190
    assert signal.target is None


def test_short_signal_on_first_close_below_opening_range() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6206, low=6185, close=6188),
    )

    signal = OpeningRangeBreakoutStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.stop == 6210


def test_later_breakout_does_not_create_second_trade_same_session() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6215, low=6200, close=6212),
        _candle(14, 30, open_=6212, high=6214, low=6180, close=6185),
    )

    signal = OpeningRangeBreakoutStrategy().evaluate(_context(candles))

    assert signal is None


def test_no_signal_when_close_stays_inside_opening_range() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6209, low=6195, close=6207),
    )

    assert OpeningRangeBreakoutStrategy().evaluate(_context(candles)) is None


def test_quality_variant_requires_ten_percent_extension() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6213, low=6200, close=6211),
    )

    assert OpeningRangeBreakoutQualityStrategy().evaluate(_context(candles)) is None


def test_quality_variant_accepts_first_breakout_at_ten_percent_extension() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6214, low=6200, close=6212),
    )

    signal = OpeningRangeBreakoutQualityStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 6190


def test_quality_variant_does_not_allow_second_chance_after_weak_breakout() -> None:
    candles = (
        _candle(13, 30, open_=6200, high=6210, low=6190, close=6205),
        _candle(14, 0, open_=6205, high=6212, low=6200, close=6211),
        _candle(14, 30, open_=6211, high=6218, low=6208, close=6216),
    )

    assert OpeningRangeBreakoutQualityStrategy().evaluate(_context(candles)) is None
