from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, StrategyContext
from medium_trading.strategy import (
    CryptoTrendLongStrategy,
    CryptoTrendLongV11Strategy,
    CryptoTrendLongV2Strategy,
)


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



def _with_m30_warmup(context: StrategyContext) -> StrategyContext:
    extra_start = context.candles_30m[0].timestamp - timedelta(minutes=30 * 13)
    warmup = tuple(
        _bar(
            extra_start + timedelta(minutes=30 * index),
            open_=99_900,
            high=100_000,
            low=99_800,
            close=99_900,
        )
        for index in range(13)
    )
    return StrategyContext(
        symbol=context.symbol,
        candles_4h=context.candles_4h,
        candles_1h=(),
        candles_30m=warmup + context.candles_30m,
    )


def test_v1_1_adds_m30_atr_stop_floor_without_changing_long_setup() -> None:
    context = _with_m30_warmup(_context())

    signal = CryptoTrendLongV11Strategy().evaluate(context)

    assert signal is not None
    assert signal.side is Side.LONG
    assert signal.stop == 99_700
    assert signal.minimum_stop_distance is not None
    assert signal.minimum_stop_distance > 0



def test_v2_accepts_persistent_h4_trend() -> None:
    context = _with_m30_warmup(_context())

    signal = CryptoTrendLongV2Strategy().evaluate(context)

    assert signal is not None
    assert signal.strategy == "crypto_trend_long_v2"
    assert any("persistent H4 regime" in reason for reason in signal.reasons)


def test_v2_rejects_temporary_h4_recovery_that_v1_1_accepts() -> None:
    context = _with_m30_warmup(_context())
    candles_4h = list(context.candles_4h)
    disrupted = candles_4h[-2]
    candles_4h[-2] = _bar(
        disrupted.timestamp,
        open_=94_400,
        high=94_500,
        low=91_900,
        close=92_000,
    )
    context = StrategyContext(
        symbol=context.symbol,
        candles_4h=tuple(candles_4h),
        candles_1h=context.candles_1h,
        candles_30m=context.candles_30m,
    )

    assert CryptoTrendLongV11Strategy().evaluate(context) is not None
    assert CryptoTrendLongV2Strategy().evaluate(context) is None
