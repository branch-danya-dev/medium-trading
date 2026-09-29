from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import GoldNewYorkMomentumContinuationStrategy


def _bar(
    timestamp: datetime,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=timestamp,
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


def _history() -> tuple[Candle, ...]:
    start = datetime(2025, 7, 7, 2, 0, tzinfo=UTC)
    candles = [
        _bar(
            start + timedelta(minutes=30 * index),
            open_=3300.0,
            high=3300.5,
            low=3299.5,
            close=3300.0,
        )
        for index in range(14)
    ]
    impulse = (
        (9, 0, 3300.0, 3302.0, 3302.5, 3299.5),
        (9, 30, 3302.0, 3304.0, 3304.5, 3301.5),
        (10, 0, 3304.0, 3306.0, 3306.5, 3303.5),
        (10, 30, 3306.0, 3307.5, 3308.0, 3305.5),
        (11, 0, 3307.5, 3309.0, 3309.5, 3307.0),
        (11, 30, 3309.0, 3308.5, 3309.5, 3308.0),
        (12, 0, 3308.5, 3310.5, 3311.0, 3308.0),
    )
    for hour, minute, open_, close, high, low in impulse:
        candles.append(
            _bar(
                datetime(2025, 7, 7, hour, minute, tzinfo=UTC),
                open_=open_,
                high=high,
                low=low,
                close=close,
            )
        )
    return tuple(candles)


def test_long_continuation_after_shallow_pullback() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3310.5,
            high=3311.0,
            low=3307.5,
            close=3308.5,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
            open_=3308.5,
            high=3312.0,
            low=3308.0,
            close=3311.5,
        ),
    )

    signal = GoldNewYorkMomentumContinuationStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 3307.5
    assert signal.target is None


def test_deep_pullback_invalidates_session() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3310.5,
            high=3311.0,
            low=3304.0,
            close=3306.0,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
            open_=3306.0,
            high=3312.0,
            low=3305.5,
            close=3311.5,
        ),
    )

    assert (
        GoldNewYorkMomentumContinuationStrategy().evaluate(_context(candles))
        is None
    )


def test_no_trade_without_counter_direction_pullback() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3310.5,
            high=3312.0,
            low=3310.0,
            close=3311.5,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
            open_=3311.5,
            high=3313.0,
            low=3311.0,
            close=3312.5,
        ),
    )

    assert (
        GoldNewYorkMomentumContinuationStrategy().evaluate(_context(candles))
        is None
    )


def test_later_continuation_is_not_second_attempt() -> None:
    first_continuation = _bar(
        datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
        open_=3308.5,
        high=3312.0,
        low=3308.0,
        close=3311.5,
    )
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3310.5,
            high=3311.0,
            low=3307.5,
            close=3308.5,
        ),
        first_continuation,
        _bar(
            datetime(2025, 7, 7, 13, 30, tzinfo=UTC),
            open_=3311.5,
            high=3314.0,
            low=3311.0,
            close=3313.5,
        ),
    )

    assert (
        GoldNewYorkMomentumContinuationStrategy().evaluate(_context(candles))
        is None
    )
