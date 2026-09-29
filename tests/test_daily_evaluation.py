from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from medium_trading.backtest.model import BacktestTrade
from medium_trading.daily_evaluation import (
    MAX_BEST_DAY_PROFIT_SHARE,
    MAX_LOSING_DAY_STREAK,
    MIN_ACTIVE_DAY_RATE,
    MIN_NET_PROFIT_FACTOR,
    MIN_PROFITABLE_ACTIVE_DAY_RATE,
    MIN_TRADES,
    _metrics_from_trades,
    _session_dates,
)
from medium_trading.domain import Candle, Side


def _trade(day: int, *, net_r: float, gross_r: float) -> BacktestTrade:
    entry = datetime(2025, 1, day, 15, 0, tzinfo=UTC)
    return BacktestTrade(
        symbol="USA500.IDX/USD",
        side=Side.LONG,
        entry_time=entry,
        exit_time=entry + timedelta(hours=1),
        entry=6000.0,
        stop=5990.0,
        exit=6010.0,
        gross_r=gross_r,
        net_r=net_r,
        cost_r=gross_r - net_r,
        exit_reason="target" if net_r > 0 else "stop",
    )


def test_daily_metrics_include_no_trade_sessions() -> None:
    sessions = (
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 7),
    )
    trades = (
        _trade(2, net_r=1.4, gross_r=1.5),
        _trade(3, net_r=-1.1, gross_r=-1.0),
    )

    metrics = _metrics_from_trades(
        trades,
        session_dates=sessions,
        timezone=ZoneInfo("America/New_York"),
    )

    assert metrics.session_days == 4
    assert metrics.active_days == 2
    assert metrics.active_day_rate == pytest.approx(0.5)
    assert metrics.profitable_active_day_rate == pytest.approx(0.5)
    assert metrics.average_net_r_per_session == pytest.approx(0.075)
    assert metrics.median_net_r_per_session == pytest.approx(0.0)
    assert metrics.best_day_r == pytest.approx(1.4)
    assert metrics.worst_day_r == pytest.approx(-1.1)
    assert metrics.longest_losing_day_streak == 1


def test_daily_income_gate_is_pre_registered_and_nontrivial() -> None:
    assert MIN_TRADES == 300
    assert MIN_ACTIVE_DAY_RATE == pytest.approx(0.70)
    assert MIN_PROFITABLE_ACTIVE_DAY_RATE == pytest.approx(0.45)
    assert MIN_NET_PROFIT_FACTOR == pytest.approx(1.15)
    assert MAX_BEST_DAY_PROFIT_SHARE == pytest.approx(0.15)
    assert MAX_LOSING_DAY_STREAK == 8


def test_crypto_session_dates_exclude_fragmented_days() -> None:
    candles = []
    for day, count in ((2, 40), (3, 39), (4, 48)):
        start = datetime(2025, 1, day, tzinfo=UTC)
        for index in range(count):
            candles.append(
                Candle(
                    timestamp=start + timedelta(minutes=30 * index),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                )
            )

    sessions = _session_dates(
        tuple(candles),
        year=2025,
        timezone=ZoneInfo("UTC"),
        required_session_time=None,
        minimum_session_bars=40,
    )

    assert sessions == (
        date(2025, 1, 2),
        date(2025, 1, 4),
    )
