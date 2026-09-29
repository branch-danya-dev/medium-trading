from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import GoldNewYorkExhaustionReversalStrategy


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


def _warmup() -> tuple[Candle, ...]:
    start = datetime(2025, 7, 7, 5, 0, tzinfo=UTC)
    return tuple(
        _bar(
            start + timedelta(minutes=30 * index),
            open_=3300.0,
            high=3300.5,
            low=3299.5,
            close=3300.0,
        )
        for index in range(14)
    )


def _up_reference() -> tuple[Candle, ...]:
    return (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3300.0,
            high=3300.6,
            low=3299.9,
            close=3300.4,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
            open_=3300.4,
            high=3301.0,
            low=3300.3,
            close=3300.8,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 30, tzinfo=UTC),
            open_=3300.8,
            high=3301.2,
            low=3300.7,
            close=3301.0,
        ),
    )


def _down_reference() -> tuple[Candle, ...]:
    return (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3300.0,
            high=3300.1,
            low=3299.4,
            close=3299.6,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
            open_=3299.6,
            high=3299.7,
            low=3299.0,
            close=3299.2,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 30, tzinfo=UTC),
            open_=3299.2,
            high=3299.3,
            low=3298.8,
            close=3299.0,
        ),
    )


def test_short_after_upward_opening_exhaustion() -> None:
    candles = _warmup() + _up_reference() + (
        _bar(
            datetime(2025, 7, 7, 14, 0, tzinfo=UTC),
            open_=3301.05,
            high=3301.1,
            low=3300.6,
            close=3300.7,
        ),
    )

    signal = GoldNewYorkExhaustionReversalStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.stop == 3301.2
    assert signal.target is None


def test_long_after_downward_opening_exhaustion() -> None:
    candles = _warmup() + _down_reference() + (
        _bar(
            datetime(2025, 7, 7, 14, 0, tzinfo=UTC),
            open_=3298.95,
            high=3299.4,
            low=3298.9,
            close=3299.3,
        ),
    )

    signal = GoldNewYorkExhaustionReversalStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 3298.8


def test_reference_move_below_point_seven_five_atr_is_rejected() -> None:
    weak_reference = (
        _bar(
            datetime(2025, 7, 7, 12, 30, tzinfo=UTC),
            open_=3300.0,
            high=3300.3,
            low=3299.9,
            close=3300.2,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
            open_=3300.2,
            high=3300.4,
            low=3300.1,
            close=3300.3,
        ),
        _bar(
            datetime(2025, 7, 7, 13, 30, tzinfo=UTC),
            open_=3300.3,
            high=3300.6,
            low=3300.2,
            close=3300.5,
        ),
    )
    candles = _warmup() + weak_reference + (
        _bar(
            datetime(2025, 7, 7, 14, 0, tzinfo=UTC),
            open_=3300.55,
            high=3300.6,
            low=3300.2,
            close=3300.3,
        ),
    )

    assert GoldNewYorkExhaustionReversalStrategy().evaluate(_context(candles)) is None


def test_first_counter_candle_outside_reversal_band_invalidates_day() -> None:
    candles = _warmup() + _up_reference() + (
        _bar(
            datetime(2025, 7, 7, 14, 0, tzinfo=UTC),
            open_=3301.05,
            high=3301.1,
            low=3300.2,
            close=3300.3,
        ),
        _bar(
            datetime(2025, 7, 7, 14, 30, tzinfo=UTC),
            open_=3300.8,
            high=3300.9,
            low=3300.6,
            close=3300.7,
        ),
    )

    assert GoldNewYorkExhaustionReversalStrategy().evaluate(_context(candles)) is None


def test_no_new_decision_after_noon_new_york() -> None:
    candles = _warmup() + _up_reference() + (
        _bar(
            datetime(2025, 7, 7, 16, 0, tzinfo=UTC),
            open_=3301.05,
            high=3301.1,
            low=3300.6,
            close=3300.7,
        ),
    )

    assert GoldNewYorkExhaustionReversalStrategy().evaluate(_context(candles)) is None
