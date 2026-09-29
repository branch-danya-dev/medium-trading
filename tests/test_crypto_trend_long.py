from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import CryptoTrendLongStrategy


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


def _context(
    *,
    rising_4h: bool = True,
    bearish_pullback: bool = True,
    confirmation_breaks_high: bool = True,
) -> StrategyContext:
    start_4h = datetime(2025, 1, 1, tzinfo=UTC)
    candles_4h = []
    price = 90_000.0
    for index in range(24):
        if rising_4h:
            close = price + 200
        else:
            close = price - 200
        candles_4h.append(
            _bar(
                start_4h + timedelta(hours=4 * index),
                open_=price,
                high=max(price, close) + 50,
                low=min(price, close) - 50,
                close=close,
            )
        )
        price = close

    pullback = _bar(
        datetime(2025, 1, 5, 12, 0, tzinfo=UTC),
        open_=100_000,
        high=100_100,
        low=99_700,
        close=99_800 if bearish_pullback else 100_050,
    )
    confirmation_close = 100_200 if confirmation_breaks_high else 100_050
    confirmation = _bar(
        datetime(2025, 1, 5, 12, 30, tzinfo=UTC),
        open_=99_800,
        high=max(confirmation_close, 100_120),
        low=99_780,
        close=confirmation_close,
    )

    return StrategyContext(
        symbol="BTC/USD",
        candles_4h=tuple(candles_4h),
        candles_1h=(),
        candles_30m=(pullback, confirmation),
    )


def test_emits_long_after_pullback_inside_rising_4h_trend() -> None:
    signal = CryptoTrendLongStrategy().evaluate(_context())

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 99_700
    assert signal.target is None


def test_rejects_falling_4h_regime() -> None:
    assert (
        CryptoTrendLongStrategy().evaluate(_context(rising_4h=False))
        is None
    )


def test_requires_bearish_pullback() -> None:
    assert (
        CryptoTrendLongStrategy().evaluate(_context(bearish_pullback=False))
        is None
    )


def test_requires_confirmation_close_above_pullback_high() -> None:
    assert (
        CryptoTrendLongStrategy().evaluate(
            _context(confirmation_breaks_high=False)
        )
        is None
    )
