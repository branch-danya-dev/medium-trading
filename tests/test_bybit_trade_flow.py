import gzip
import io
from datetime import UTC, date, datetime, timedelta

import pytest

from medium_trading.data import bybit_trade_flow as flow


def _archive_bytes(rows: list[str]) -> io.BytesIO:
    text = (
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional\n"
        + "\n".join(rows)
        + "\n"
    )
    return io.BytesIO(gzip.compress(text.encode("utf-8")))


def test_aggregate_archive_uses_taker_side_and_m5_buckets() -> None:
    stream = _archive_bytes(
        [
            (
                "1672531201.0,BTCUSDT,Buy,1.0,100.0,PlusTick,a,"
                "0,0,0"
            ),
            (
                "1672531210.0,BTCUSDT,Sell,0.25,104.0,MinusTick,b,"
                "0,0,0"
            ),
            (
                "1672531501.0,BTCUSDT,Buy,0.5,110.0,PlusTick,c,"
                "0,0,0"
            ),
        ]
    )

    points = flow._aggregate_archive(
        stream,
        symbol="BTCUSDT",
        day=date(2023, 1, 1),
    )

    assert len(points) == 2
    first, second = points
    assert first.timestamp == datetime(2023, 1, 1, tzinfo=UTC)
    assert first.buy_qty == pytest.approx(1.0)
    assert first.sell_qty == pytest.approx(0.25)
    assert first.buy_notional == pytest.approx(100.0)
    assert first.sell_notional == pytest.approx(26.0)
    assert first.buy_count == 1
    assert first.sell_count == 1

    assert second.timestamp == datetime(
        2023,
        1,
        1,
        0,
        5,
        tzinfo=UTC,
    )
    assert second.buy_qty == pytest.approx(0.5)
    assert second.sell_qty == 0
    assert second.buy_count == 1
    assert second.sell_count == 0


def test_archive_timestamp_accepts_milliseconds() -> None:
    stream = _archive_bytes(
        [
            (
                "1672531201000,BTCUSDT,Buy,1.0,100.0,PlusTick,a,"
                "0,0,0"
            ),
        ]
    )

    points = flow._aggregate_archive(
        stream,
        symbol="BTCUSDT",
        day=date(2023, 1, 1),
    )

    assert points[0].timestamp == datetime(2023, 1, 1, tzinfo=UTC)


def test_archive_rejects_unknown_side() -> None:
    stream = _archive_bytes(
        [
            (
                "1672531201.0,BTCUSDT,Mystery,1.0,100.0,PlusTick,a,"
                "0,0,0"
            ),
        ]
    )

    with pytest.raises(flow.BybitTradeFlowError, match="unsupported trade side"):
        flow._aggregate_archive(
            stream,
            symbol="BTCUSDT",
            day=date(2023, 1, 1),
        )


def test_trade_flow_round_trip(tmp_path) -> None:
    points = tuple(
        flow.TradeFlowPoint(
            timestamp=datetime(2023, 1, 1, tzinfo=UTC)
            + timedelta(minutes=5 * index),
            buy_qty=1.0 + index,
            sell_qty=0.5,
            buy_notional=100.0 + index,
            sell_notional=50.0,
            buy_count=10 + index,
            sell_count=5,
        )
        for index in range(3)
    )
    path = tmp_path / "flow.csv"

    flow.save_trade_flow(path, points)
    loaded = flow.load_trade_flow(path)

    assert loaded == points
