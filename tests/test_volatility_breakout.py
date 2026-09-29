from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import VolatilityBreakoutStrategy


def _candle(
    index: int,
    *,
    minutes: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index * minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def _long_context() -> StrategyContext:
    candles_4h = tuple(
        _candle(
            index,
            minutes=240,
            open_=1.10,
            high=1.20,
            low=1.00,
            close=1.15,
        )
        for index in range(20)
    )
    candles_1h = tuple(
        _candle(
            index,
            minutes=60,
            open_=1.1000,
            high=1.1050,
            low=1.0950,
            close=1.1000,
        )
        for index in range(60, 80)
    )
    candles_30m = [
        _candle(
            index,
            minutes=30,
            open_=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1000,
        )
        for index in range(145, 160)
    ]
    candles_30m.append(
        _candle(
            160,
            minutes=30,
            open_=1.1000,
            high=1.1080,
            low=1.0995,
            close=1.1070,
        )
    )
    return StrategyContext(
        symbol="EUR/USD",
        candles_4h=candles_4h,
        candles_1h=candles_1h,
        candles_30m=tuple(candles_30m),
    )


def test_strategy_does_not_signal_without_required_history() -> None:
    candle = _candle(
        0,
        minutes=30,
        open_=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
    )
    context = StrategyContext(
        symbol="EUR/USD",
        candles_4h=(candle,),
        candles_1h=(candle,),
        candles_30m=(candle,),
    )

    assert VolatilityBreakoutStrategy().evaluate(context) is None


def test_long_breakout_requires_channel_break_and_range_expansion() -> None:
    signal = VolatilityBreakoutStrategy().evaluate(_long_context())

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.entry == 1.1070
    assert signal.stop < signal.entry
    assert signal.strategy == "volatility_breakout"


def test_no_breakout_means_no_signal() -> None:
    context = _long_context()
    candles_30m = list(context.candles_30m)
    candles_30m[-1] = _candle(
        160,
        minutes=30,
        open_=1.1000,
        high=1.1040,
        low=1.0995,
        close=1.1040,
    )

    signal = VolatilityBreakoutStrategy().evaluate(
        StrategyContext(
            symbol=context.symbol,
            candles_4h=context.candles_4h,
            candles_1h=context.candles_1h,
            candles_30m=tuple(candles_30m),
        )
    )

    assert signal is None


def test_h1_candle_containing_trigger_is_not_used_in_channel() -> None:
    context = _long_context()
    contaminated_h1 = _candle(
        80,
        minutes=60,
        open_=1.1000,
        high=1.2000,
        low=1.0950,
        close=1.1500,
    )

    signal = VolatilityBreakoutStrategy().evaluate(
        StrategyContext(
            symbol=context.symbol,
            candles_4h=context.candles_4h,
            candles_1h=context.candles_1h + (contaminated_h1,),
            candles_30m=context.candles_30m,
        )
    )

    assert signal is not None
    assert signal.side is Side.LONG
