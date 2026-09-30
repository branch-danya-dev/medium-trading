from datetime import UTC, datetime, timedelta

import pytest

from medium_trading.backtest.model import BacktestTrade
from medium_trading.domain import Candle, Side
from medium_trading.market_structure_events import (
    BODY_BREAK_UNCONFIRMED,
    BREAK_ACCEPTED,
    BREAK_RECLAIMED,
    RETEST_HELD,
    SWEEP_RECLAIM,
    _latest_structure,
    _simulate_trade_pair,
)


def _bar(
    index: int,
    *,
    open_: float = 100.0,
    high: float = 102.0,
    low: float = 98.0,
    close: float = 100.0,
) -> Candle:
    return Candle(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10.0,
    )


def _history(extra: tuple[Candle, ...]) -> tuple[Candle, ...]:
    candles = [_bar(index) for index in range(20)]
    candles[16] = _bar(16, open_=100, high=102, low=95, close=100)
    candles.extend(extra)
    return tuple(candles)


def _trade(candles: tuple[Candle, ...]) -> BacktestTrade:
    return BacktestTrade(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_time=candles[20].timestamp,
        exit_time=candles[-1].timestamp + timedelta(minutes=5),
        entry=100.0,
        stop=90.0,
        exit=90.0,
        gross_r=-1.0,
        net_r=-1.15,
        cost_r=0.15,
        exit_reason="stop",
    )


def test_swing_becomes_usable_only_after_two_right_bars_close() -> None:
    candles = _history(())

    assert _latest_structure(candles, 17) is None
    level, atr5 = _latest_structure(candles, 18)

    assert level == 95
    assert atr5 > 0


def test_accepted_break_exits_at_next_m5_open() -> None:
    candles = _history(
        (
            _bar(20, open_=100, high=100, low=93, close=94),
            _bar(21, open_=94, high=94.2, low=92, close=93),
            _bar(22, open_=93, high=94.2, low=91, close=92),
            _bar(23, open_=92, high=93, low=91, close=92),
            _bar(24, open_=92, high=93, low=89, close=90),
        )
    )

    comparison = _simulate_trade_pair(
        candles_5m=candles,
        trade=_trade(candles),
        fee_bps_per_side=5.5,
        slippage_bps_per_side=2.0,
    )

    assert [event.kind for event in comparison.events] == [
        BODY_BREAK_UNCONFIRMED,
        BREAK_ACCEPTED,
    ]
    assert comparison.structural_exit_trigger == BREAK_ACCEPTED
    assert comparison.managed.exit_reason == "structure_break_accepted"
    assert comparison.managed.exit_time == candles[23].timestamp
    assert comparison.managed.exit_price == 92
    assert comparison.original.exit_reason == "stop"
    assert comparison.delta_gross_r == pytest.approx(0.2)


def test_held_retest_exits_before_acceptance() -> None:
    candles = _history(
        (
            _bar(20, open_=100, high=100, low=93, close=94),
            _bar(21, open_=94, high=95, low=92, close=93),
            _bar(22, open_=93, high=94, low=91, close=92),
            _bar(23, open_=92, high=93, low=89, close=90),
        )
    )

    comparison = _simulate_trade_pair(
        candles_5m=candles,
        trade=_trade(candles),
        fee_bps_per_side=5.5,
        slippage_bps_per_side=2.0,
    )

    assert [event.kind for event in comparison.events] == [
        BODY_BREAK_UNCONFIRMED,
        RETEST_HELD,
    ]
    assert comparison.structural_exit_trigger == RETEST_HELD
    assert comparison.managed.exit_reason == "structure_retest_held"
    assert comparison.managed.exit_time == candles[22].timestamp
    assert comparison.break_outcomes == (RETEST_HELD,)


def test_fast_break_reclaim_cancels_wait_and_keeps_trade_alive() -> None:
    candles = _history(
        (
            _bar(20, open_=100, high=100, low=93, close=94),
            _bar(21, open_=94, high=97, low=93, close=96),
            _bar(22, open_=96, high=121, low=95, close=120),
        )
    )

    comparison = _simulate_trade_pair(
        candles_5m=candles,
        trade=_trade(candles),
        fee_bps_per_side=5.5,
        slippage_bps_per_side=2.0,
    )

    assert [event.kind for event in comparison.events] == [
        BODY_BREAK_UNCONFIRMED,
        BREAK_RECLAIMED,
    ]
    assert comparison.break_outcomes == (BREAK_RECLAIMED,)
    assert comparison.structural_exit_trigger is None
    assert comparison.original.exit_reason == "target"
    assert comparison.managed.exit_reason == "target"
    assert comparison.delta_net_r == pytest.approx(0.0)


def test_sweep_reclaim_holds_and_reports_target_followthrough() -> None:
    candles = _history(
        (
            _bar(20, open_=100, high=101, low=94.8, close=96),
            _bar(21, open_=96, high=121, low=95, close=120),
        )
    )

    comparison = _simulate_trade_pair(
        candles_5m=candles,
        trade=_trade(candles),
        fee_bps_per_side=5.5,
        slippage_bps_per_side=2.0,
    )

    assert [event.kind for event in comparison.events] == [SWEEP_RECLAIM]
    assert comparison.structural_exit_trigger is None
    assert comparison.sweep_followthrough == ((True, True, True),)
    assert comparison.managed.exit_reason == "target"
