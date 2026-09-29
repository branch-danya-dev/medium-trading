from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import TimeSeriesMomentumStrategy


def _four_hour_candles(direction: int) -> tuple[Candle, ...]:
    candles: list[Candle] = []
    for index in range(31):
        close = 1.1000 + direction * index * 0.0010
        candles.append(
            Candle(
                timestamp=datetime(2026, 1, 1, tzinfo=UTC)
                + timedelta(hours=4 * index),
                open=close - direction * 0.0002,
                high=close + 0.0010,
                low=close - 0.0010,
                close=close,
            )
        )
    return tuple(candles)


def _context(direction: int, *, on_close: bool = True) -> StrategyContext:
    candles_4h = _four_hour_candles(direction)
    last_4h_end = candles_4h[-1].timestamp + timedelta(hours=4)
    last_30m_timestamp = (
        last_4h_end - timedelta(minutes=30)
        if on_close
        else last_4h_end
    )
    trigger = Candle(
        timestamp=last_30m_timestamp,
        open=candles_4h[-1].close,
        high=candles_4h[-1].close + 0.0005,
        low=candles_4h[-1].close - 0.0005,
        close=candles_4h[-1].close,
    )
    return StrategyContext(
        symbol="EUR/USD",
        candles_4h=candles_4h,
        candles_1h=(),
        candles_30m=(trigger,),
    )


def test_long_signal_on_positive_4h_momentum() -> None:
    signal = TimeSeriesMomentumStrategy().evaluate(_context(1))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop < signal.entry
    assert signal.strategy == "time_series_momentum"


def test_short_signal_on_negative_4h_momentum() -> None:
    signal = TimeSeriesMomentumStrategy().evaluate(_context(-1))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.stop > signal.entry


def test_signal_only_fires_when_new_4h_candle_has_just_closed() -> None:
    signal = TimeSeriesMomentumStrategy().evaluate(_context(1, on_close=False))

    assert signal is None


def test_strategy_requires_full_momentum_lookback() -> None:
    context = _context(1)
    shortened = StrategyContext(
        symbol=context.symbol,
        candles_4h=context.candles_4h[:-1],
        candles_1h=context.candles_1h,
        candles_30m=context.candles_30m,
    )

    assert TimeSeriesMomentumStrategy().evaluate(shortened) is None
