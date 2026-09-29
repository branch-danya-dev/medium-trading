from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import CryptoIntradayMomentumContinuationStrategy


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


def _warmup() -> tuple[Candle, ...]:
    start = datetime(2025, 7, 7, 0, 0, tzinfo=UTC)
    return tuple(
        _bar(
            start + timedelta(minutes=30 * index),
            open_=100_000,
            high=100_050,
            low=99_950,
            close=100_000,
        )
        for index in range(14)
    )


def _uptrend() -> tuple[Candle, ...]:
    start = datetime(2025, 7, 7, 7, 0, tzinfo=UTC)
    bars = []
    price = 100_000.0
    for index in range(8):
        close = price + 100
        bars.append(
            _bar(
                start + timedelta(minutes=30 * index),
                open_=price,
                high=close + 20,
                low=price - 20,
                close=close,
            )
        )
        price = close
    return tuple(bars)


def _downtrend() -> tuple[Candle, ...]:
    start = datetime(2025, 7, 7, 7, 0, tzinfo=UTC)
    bars = []
    price = 100_800.0
    for index in range(8):
        close = price - 100
        bars.append(
            _bar(
                start + timedelta(minutes=30 * index),
                open_=price,
                high=price + 20,
                low=close - 20,
                close=close,
            )
        )
        price = close
    return tuple(bars)


def test_long_after_shallow_pullback_and_resume() -> None:
    candles = _warmup() + _uptrend() + (
        _bar(
            datetime(2025, 7, 7, 11, 0, tzinfo=UTC),
            open_=100_800,
            high=100_820,
            low=100_600,
            close=100_650,
        ),
        _bar(
            datetime(2025, 7, 7, 11, 30, tzinfo=UTC),
            open_=100_650,
            high=100_880,
            low=100_640,
            close=100_850,
        ),
    )

    signal = CryptoIntradayMomentumContinuationStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 100_600
    assert signal.target is None


def test_short_after_shallow_pullback_and_resume() -> None:
    candles = _warmup() + _downtrend() + (
        _bar(
            datetime(2025, 7, 7, 11, 0, tzinfo=UTC),
            open_=100_000,
            high=100_200,
            low=99_980,
            close=100_150,
        ),
        _bar(
            datetime(2025, 7, 7, 11, 30, tzinfo=UTC),
            open_=100_150,
            high=100_160,
            low=99_920,
            close=99_950,
        ),
    )

    signal = CryptoIntradayMomentumContinuationStrategy().evaluate(_context(candles))

    assert signal is not None
    assert signal.side is Side.SHORT
    assert signal.stop == 100_200


def test_deep_pullback_is_rejected() -> None:
    candles = _warmup() + _uptrend() + (
        _bar(
            datetime(2025, 7, 7, 11, 0, tzinfo=UTC),
            open_=100_800,
            high=100_820,
            low=100_300,
            close=100_350,
        ),
        _bar(
            datetime(2025, 7, 7, 11, 30, tzinfo=UTC),
            open_=100_350,
            high=100_900,
            low=100_340,
            close=100_850,
        ),
    )

    assert (
        CryptoIntradayMomentumContinuationStrategy().evaluate(_context(candles))
        is None
    )


def test_gap_in_recent_m30_history_rejects_setup() -> None:
    candles = list(_warmup() + _uptrend())
    candles[-1] = _bar(
        candles[-1].timestamp + timedelta(minutes=30),
        open_=100_700,
        high=100_820,
        low=100_680,
        close=100_800,
    )
    candles.extend(
        (
            _bar(
                datetime(2025, 7, 7, 11, 30, tzinfo=UTC),
                open_=100_800,
                high=100_820,
                low=100_600,
                close=100_650,
            ),
            _bar(
                datetime(2025, 7, 7, 12, 0, tzinfo=UTC),
                open_=100_650,
                high=100_880,
                low=100_640,
                close=100_850,
            ),
        )
    )

    assert (
        CryptoIntradayMomentumContinuationStrategy().evaluate(_context(tuple(candles)))
        is None
    )
