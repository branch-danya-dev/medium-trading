from datetime import UTC, datetime, timedelta

import pytest

from medium_trading.backtest.model import BacktestConfig
from medium_trading.backtest.validation import (
    chronological_splits,
    evaluate_forward_symbol,
)
from medium_trading.domain import Candle, Signal, StrategyContext


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


class NoSignalStrategy:
    name = "no_signal"

    def evaluate(self, context: StrategyContext) -> Signal | None:
        return None


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


def test_forward_evaluation_uses_warmup_and_fixed_trade_window() -> None:
    candles = _candles(1_000)
    evaluation = evaluate_forward_symbol(
        symbol="EUR/USD",
        candles=candles,
        strategy=NoSignalStrategy(),
        config=BacktestConfig(round_trip_cost_pips=1.0),
        trade_start=candles[100].timestamp,
        trade_end=candles[800].timestamp,
    )

    assert evaluation.symbol == "EUR/USD"
    assert evaluation.forward.signal_count == 0
    assert evaluation.forward_2x_costs.signal_count == 0


def test_forward_evaluation_requires_pre_start_warmup() -> None:
    candles = _candles(100)

    with pytest.raises(ValueError, match="warmup"):
        evaluate_forward_symbol(
            symbol="EUR/USD",
            candles=candles,
            strategy=NoSignalStrategy(),
            config=BacktestConfig(round_trip_cost_pips=1.0),
            trade_start=candles[0].timestamp,
            trade_end=candles[50].timestamp,
        )
