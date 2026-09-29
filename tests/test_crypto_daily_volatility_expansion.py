from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import CryptoDailyVolatilityExpansionStrategy


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
        symbol="BTC/USD",
        candles_4h=(),
        candles_1h=(),
        candles_30m=candles,
    )


def _history() -> tuple[Candle, ...]:
    warmup_start = datetime(2025, 7, 6, 17, 0, tzinfo=UTC)
    warmup = tuple(
        _bar(
            warmup_start + timedelta(minutes=30 * index),
            open_=100_000,
            high=100_050,
            low=99_950,
            close=100_000,
        )
        for index in range(14)
    )
    reference = tuple(
        _bar(
            datetime(2025, 7, 7, 0, 0, tzinfo=UTC)
            + timedelta(minutes=30 * index),
            open_=100_000,
            high=100_500 if index == 2 else 100_300,
            low=99_500 if index == 5 else 99_700,
            close=100_000,
        )
        for index in range(8)
    )
    return warmup + reference


def test_long_first_close_beyond_reference_and_atr_extension() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 4, 0, tzinfo=UTC),
            open_=100_300,
            high=100_700,
            low=100_250,
            close=100_560,
        ),
    )

    signal = CryptoDailyVolatilityExpansionStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 100_000
    assert signal.target is None


def test_short_first_close_beyond_reference_and_atr_extension() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 4, 0, tzinfo=UTC),
            open_=99_700,
            high=99_750,
            low=99_300,
            close=99_440,
        ),
    )

    signal = CryptoDailyVolatilityExpansionStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.stop == 100_000


def test_close_inside_extension_does_not_signal() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 4, 0, tzinfo=UTC),
            open_=100_300,
            high=100_550,
            low=100_250,
            close=100_530,
        ),
    )

    assert CryptoDailyVolatilityExpansionStrategy().evaluate(_context(candles)) is None


def test_later_breakout_is_not_second_attempt() -> None:
    candles = _history() + (
        _bar(
            datetime(2025, 7, 7, 4, 0, tzinfo=UTC),
            open_=100_300,
            high=100_700,
            low=100_250,
            close=100_560,
        ),
        _bar(
            datetime(2025, 7, 7, 4, 30, tzinfo=UTC),
            open_=100_520,
            high=100_900,
            low=100_500,
            close=100_800,
        ),
    )

    assert CryptoDailyVolatilityExpansionStrategy().evaluate(_context(candles)) is None
