from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest

import medium_trading.data.bybit_download as bybit
from medium_trading.data.bybit_download import (
    BybitDownloadError,
    build_kline_url,
    decode_kline_payload,
)


def _row(timestamp: datetime, price: float) -> list[str]:
    milliseconds = int(timestamp.timestamp() * 1000)
    return [
        str(milliseconds),
        str(price),
        str(price + 10),
        str(price - 10),
        str(price + 5),
        "1.25",
        str((price + 5) * 1.25),
    ]


def test_build_kline_url_uses_linear_btcusdt_m30() -> None:
    url = build_kline_url(
        symbol="btcusdt",
        category="linear",
        start_ms=1_700_000_000_000,
        end_ms=1_700_001_799_999,
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "api.bybit.com"
    assert parsed.path == "/v5/market/kline"
    assert query["category"] == ["linear"]
    assert query["symbol"] == ["BTCUSDT"]
    assert query["interval"] == ["30"]
    assert query["limit"] == ["1000"]


def test_decode_kline_payload_sorts_reverse_bybit_rows() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "symbol": "BTCUSDT",
            "category": "linear",
            "list": [
                _row(start + timedelta(minutes=30), 101_000),
                _row(start, 100_000),
            ],
        },
    }

    candles = decode_kline_payload(
        payload,
        expected_symbol="BTCUSDT",
        expected_category="linear",
    )

    assert tuple(candle.timestamp for candle in candles) == (
        start,
        start + timedelta(minutes=30),
    )
    assert candles[0].open == 100_000
    assert candles[1].volume == pytest.approx(1.25)


def test_download_m30_requires_complete_exchange_native_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_start = date(2025, 1, 1)
    requested_end = date(2025, 1, 1)
    start_dt = datetime(2025, 1, 1, tzinfo=UTC)

    def fake_fetch(
        *,
        symbol: str,
        category: str,
        start_ms: int,
        end_ms: int,
        base_url: str,
        interval_minutes: int,
    ) -> dict[str, object]:
        assert symbol == "BTCUSDT"
        assert category == "linear"
        assert base_url == "https://api.bybit.com/v5/market/kline"
        assert interval_minutes == 30
        rows = [
            _row(start_dt + timedelta(minutes=30 * index), 100_000 + index)
            for index in reversed(range(48))
        ]
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "symbol": symbol,
                "category": category,
                "list": rows,
            },
        }

    monkeypatch.setattr(bybit, "_fetch_kline_page", fake_fetch)

    result = bybit.download_m30(
        symbol="BTCUSDT",
        category="linear",
        start=requested_start,
        end=requested_end,
    )

    assert len(result.candles) == 48
    assert result.expected_candles == 48
    assert result.request_count == 1
    assert result.candles[0].timestamp == start_dt
    assert result.candles[-1].timestamp == start_dt + timedelta(minutes=30 * 47)


def test_download_m30_rejects_missing_exchange_candle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_dt = datetime(2025, 1, 1, tzinfo=UTC)

    def fake_fetch(**_kwargs: object) -> dict[str, object]:
        rows = [
            _row(start_dt + timedelta(minutes=30 * index), 100_000 + index)
            for index in reversed(range(48))
            if index != 17
        ]
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "symbol": "BTCUSDT",
                "category": "linear",
                "list": rows,
            },
        }

    monkeypatch.setattr(bybit, "_fetch_kline_page", fake_fetch)

    with pytest.raises(BybitDownloadError, match="incomplete"):
        bybit.download_m30(
            symbol="BTCUSDT",
            category="linear",
            start=date(2025, 1, 1),
            end=date(2025, 1, 1),
        )



def test_download_m5_requests_five_minute_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_dt = datetime(2025, 1, 1, tzinfo=UTC)

    def fake_fetch(
        *,
        symbol: str,
        category: str,
        start_ms: int,
        end_ms: int,
        base_url: str,
        interval_minutes: int,
    ) -> dict[str, object]:
        assert interval_minutes == 5
        rows = [
            _row(start_dt + timedelta(minutes=5 * index), 100_000 + index)
            for index in reversed(range(288))
        ]
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "symbol": symbol,
                "category": category,
                "list": rows,
            },
        }

    monkeypatch.setattr(bybit, "_fetch_kline_page", fake_fetch)

    result = bybit.download_m5(
        symbol="BTCUSDT",
        category="linear",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
    )

    assert len(result.candles) == 288
    assert result.expected_candles == 288
