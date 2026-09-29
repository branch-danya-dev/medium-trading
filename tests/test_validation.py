from datetime import UTC, datetime, timedelta

from medium_trading.backtest.validation import chronological_splits
from medium_trading.domain import Candle


def _candles(count: int) -> tuple[Candle, ...]:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return tuple(
        Candle(
            timestamp=start + timedelta(minutes=30 * index),
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.0,
        )
        for index in range(count)
    )


def test_chronological_splits_keep_warmup_outside_trade_period() -> None:
    candles = _candles(1_000)

    train, validation, out_of_sample = chronological_splits(
        candles,
        warmup_bars=100,
    )

    assert len(train.candles) == 600
    assert validation.candles[0] == candles[500]
    assert validation.trade_start == candles[600].timestamp
    assert out_of_sample.candles[0] == candles[700]
    assert out_of_sample.trade_start == candles[800].timestamp
