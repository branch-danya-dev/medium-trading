from datetime import UTC, datetime, timedelta

import pytest

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import MeanReversionStrategy


def _range_candles(last_close: float) -> tuple[Candle, ...]:
    candles: list[Candle] = []
    for index in range(30):
        close = 1.0990 if index % 2 == 0 else 1.1010
        candles.append(
            Candle(
                timestamp=datetime(2026, 1, 1, tzinfo=UTC)
                + timedelta(hours=4 * index),
                open=1.1000,
                high=max(1.1015, close + 0.0005),
                low=min(1.0985, close - 0.0005),
                close=close,
            )
        )

    candles.append(
        Candle(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=4 * 30),
            open=1.1000,
            high=max(1.1010, last_close + 0.0010),
            low=min(1.0990, last_close - 0.0010),
            close=last_close,
        )
    )
    return tuple(candles)


def _context(last_close: float, *, on_close: bool = True) -> StrategyContext:
    candles_4h = _range_candles(last_close)
    last_4h_end = candles_4h[-1].timestamp + timedelta(hours=4)
    last_30m_timestamp = (
        last_4h_end - timedelta(minutes=30)
        if on_close
        else last_4h_end
    )
    trigger = Candle(
        timestamp=last_30m_timestamp,
        open=last_close,
        high=last_close + 0.0005,
        low=last_close - 0.0005,
        close=last_close,
    )
    return StrategyContext(
        symbol="EUR/USD",
        candles_4h=candles_4h,
        candles_1h=(),
        candles_30m=(trigger,),
    )


def test_long_signal_on_large_negative_deviation_in_range() -> None:
    signal = MeanReversionStrategy().evaluate(_context(1.0960))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.entry == pytest.approx(1.0960)
    assert signal.target == pytest.approx(1.1000)
    assert signal.stop < signal.entry


def test_short_signal_on_large_positive_deviation_in_range() -> None:
    signal = MeanReversionStrategy().evaluate(_context(1.1040))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.target == pytest.approx(1.1000)
    assert signal.stop > signal.entry


def test_small_deviation_does_not_signal() -> None:
    assert MeanReversionStrategy().evaluate(_context(1.1015)) is None


def test_strategy_only_fires_when_new_4h_candle_has_closed() -> None:
    assert MeanReversionStrategy().evaluate(_context(1.0960, on_close=False)) is None


def test_trending_regime_is_rejected() -> None:
    candles_4h = tuple(
        Candle(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC)
            + timedelta(hours=4 * index),
            open=1.1000 + index * 0.0010,
            high=1.1010 + index * 0.0010,
            low=1.0990 + index * 0.0010,
            close=1.1000 + index * 0.0010,
        )
        for index in range(31)
    )
    last_4h_end = candles_4h[-1].timestamp + timedelta(hours=4)
    trigger = Candle(
        timestamp=last_4h_end - timedelta(minutes=30),
        open=candles_4h[-1].close,
        high=candles_4h[-1].close + 0.0005,
        low=candles_4h[-1].close - 0.0005,
        close=candles_4h[-1].close,
    )

    signal = MeanReversionStrategy().evaluate(
        StrategyContext(
            symbol="EUR/USD",
            candles_4h=candles_4h,
            candles_1h=(),
            candles_30m=(trigger,),
        )
    )

    assert signal is None
