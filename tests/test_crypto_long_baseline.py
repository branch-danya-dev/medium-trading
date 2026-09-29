from datetime import UTC, datetime, timedelta

import pytest

from medium_trading.backtest.model import BacktestTrade
from medium_trading.crypto_long_baseline import _trade_diagnostic
from medium_trading.domain import Candle, Side


def _bar(
    index: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=datetime(2025, 1, 2, tzinfo=UTC)
        + timedelta(minutes=30 * index),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def test_trade_diagnostic_reports_24h_mfe_mae_and_post_exit_move() -> None:
    candles = (
        _bar(0, open_=100, high=103, low=98, close=102),
        _bar(1, open_=102, high=110, low=99, close=108),
        _bar(2, open_=108, high=120, low=95, close=115),
        _bar(3, open_=115, high=118, low=85, close=90),
    )
    trade = BacktestTrade(
        symbol="BTC/USD",
        side=Side.LONG,
        entry_time=candles[0].timestamp,
        exit_time=candles[2].timestamp,
        entry=100,
        stop=90,
        exit=110,
        gross_r=1.0,
        net_r=0.5,
        cost_r=0.5,
        exit_reason="timeout",
    )

    diagnostic = _trade_diagnostic(candles, trade)

    assert diagnostic["mfe_r_24h"] == pytest.approx(2.0)
    assert diagnostic["mae_r_24h"] == pytest.approx(1.5)
    assert diagnostic["post_exit_mfe_r_24h"] == pytest.approx(1.8)
    assert diagnostic["reached_1r_24h"] is True
    assert diagnostic["reached_2r_24h"] is True
    assert diagnostic["reached_3r_24h"] is False



def test_stop_diagnostic_reports_recovery_time_after_stop() -> None:
    candles = (
        _bar(0, open_=100, high=101, low=99, close=100),
        _bar(1, open_=100, high=100.5, low=89, close=90),
        _bar(2, open_=90, high=111, low=90, close=108),
        _bar(3, open_=108, high=121, low=107, close=120),
    )
    trade = BacktestTrade(
        symbol="BTC/USD",
        side=Side.LONG,
        entry_time=candles[0].timestamp,
        exit_time=candles[1].timestamp,
        entry=100,
        stop=90,
        exit=90,
        gross_r=-1.0,
        net_r=-1.2,
        cost_r=0.2,
        exit_reason="stop",
    )

    diagnostic = _trade_diagnostic(candles, trade)

    assert diagnostic["mfe_r_before_exit"] == pytest.approx(0.1)
    assert diagnostic["mae_r_before_exit"] == pytest.approx(1.1)
    assert diagnostic["mfe_r_after_stop_24h"] == pytest.approx(2.1)
    assert diagnostic["time_from_stop_to_1r_hours"] == pytest.approx(1.0)
    assert diagnostic["time_from_stop_to_2r_hours"] == pytest.approx(1.5)
