from datetime import UTC, datetime, timedelta

import pytest

from medium_trading.backtest.engine import aggregate_candles, run_backtest
from medium_trading.backtest.model import BacktestConfig
from medium_trading.domain import Candle, Side, Signal, StrategyContext


def _candle(
    index: int,
    *,
    open_: float = 1.1000,
    high: float = 1.1002,
    low: float = 1.0998,
    close: float = 1.1000,
) -> Candle:
    return Candle(
        timestamp=datetime(2026, 1, 5, tzinfo=UTC) + timedelta(minutes=30 * index),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def test_aggregate_candles_builds_complete_one_hour_bars() -> None:
    candles = (
        _candle(0, open_=1.0, high=1.2, low=0.9, close=1.1),
        _candle(1, open_=1.1, high=1.3, low=1.0, close=1.2),
        _candle(2, open_=1.2, high=1.4, low=1.1, close=1.3),
        _candle(3, open_=1.3, high=1.5, low=1.2, close=1.4),
    )

    hourly = aggregate_candles(candles, 60)

    assert len(hourly) == 2
    assert hourly[0].open == 1.0
    assert hourly[0].high == 1.3
    assert hourly[0].low == 0.9
    assert hourly[0].close == 1.2


class OneShotStrategy:
    name = "one_shot"

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if len(context.candles_30m) != 16:
            return None
        return Signal(
            symbol=context.symbol,
            side=Side.LONG,
            entry=context.candles_30m[-1].close,
            stop=1.0990,
            confidence=1.0,
            strategy=self.name,
            reasons=("test",),
        )


def test_backtest_enters_next_bar_and_deducts_costs() -> None:
    candles = [_candle(index) for index in range(20)]
    candles[16] = _candle(
        16,
        open_=1.1000,
        high=1.1021,
        low=1.0995,
        close=1.1020,
    )

    report = run_backtest(
        symbol="EUR/USD",
        candles_30m=tuple(candles),
        strategy=OneShotStrategy(),
        config=BacktestConfig(
            round_trip_cost_pips=1.0,
            minimum_cost_multiple=8.0,
        ),
    )

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.entry == pytest.approx(1.1000)
    assert trade.gross_r == pytest.approx(2.0)
    assert trade.cost_r == pytest.approx(0.1)
    assert trade.net_r == pytest.approx(1.9)
    assert report.final_equity == pytest.approx(1009.5)


def test_trade_start_prevents_warmup_signals_from_becoming_trades() -> None:
    candles = tuple(_candle(index) for index in range(20))

    report = run_backtest(
        symbol="EUR/USD",
        candles_30m=candles,
        strategy=OneShotStrategy(),
        config=BacktestConfig(round_trip_cost_pips=1.0),
        trade_start=candles[17].timestamp,
    )

    assert report.signal_count == 0
    assert not report.trades


class ContextSizeStrategy:
    name = "context_size"

    def __init__(self) -> None:
        self.maximum_30m = 0
        self.maximum_1h = 0
        self.maximum_4h = 0

    def evaluate(self, context: StrategyContext) -> Signal | None:
        self.maximum_30m = max(self.maximum_30m, len(context.candles_30m))
        self.maximum_1h = max(self.maximum_1h, len(context.candles_1h))
        self.maximum_4h = max(self.maximum_4h, len(context.candles_4h))
        return None


def test_backtest_bounds_strategy_history() -> None:
    strategy = ContextSizeStrategy()
    candles = tuple(_candle(index) for index in range(2_200))

    run_backtest(
        symbol="EUR/USD",
        candles_30m=candles,
        strategy=strategy,
        config=BacktestConfig(round_trip_cost_pips=1.0),
    )

    assert strategy.maximum_30m <= 256
    assert strategy.maximum_1h <= 256
    assert strategy.maximum_4h <= 256
