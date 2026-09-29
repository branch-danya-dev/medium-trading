import json
from datetime import UTC, date, datetime

import pytest

import medium_trading.data.dukascopy_download as dukascopy
from medium_trading.data.dukascopy_download import (
    aggregate_m30,
    build_endpoint_url,
    decode_minute_payload,
    endpoint_symbol,
)


def _payload(count: int = 60) -> dict[str, object]:
    timestamp = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1000)
    deltas = [0] + [1] * (count - 1)
    zeros = [0] * count
    return {
        "timestamp": timestamp,
        "multiplier": 0.00001,
        "shift": 60000,
        "times": deltas,
        "open": 1.10000,
        "high": 1.10020,
        "low": 1.09980,
        "close": 1.10010,
        "opens": zeros,
        "highs": zeros,
        "lows": zeros,
        "closes": zeros,
        "volumes": [0.001] * count,
    }


def test_endpoint_symbol_accepts_common_fx_notation() -> None:
    assert endpoint_symbol("EUR/USD") == "EUR-USD"
    assert endpoint_symbol("eur_usd") == "EUR-USD"
    assert endpoint_symbol("EUR-USD") == "EUR-USD"


def test_build_endpoint_url_uses_calendar_date() -> None:
    assert build_endpoint_url("EUR/USD", "BID", date(2024, 1, 2)) == (
        "https://jetta.dukascopy.com/v1/candles/minute/EUR-USD/BID/2024/1/2"
    )


def test_decode_and_aggregate_minute_payload() -> None:
    minutes = decode_minute_payload(_payload())

    assert len(minutes) == 60
    assert minutes[0].timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert minutes[1].timestamp == datetime(2024, 1, 2, 0, 1, tzinfo=UTC)
    assert minutes[0].volume == pytest.approx(1000.0)

    m30 = aggregate_m30(minutes)

    assert len(m30) == 2
    assert m30[0].timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert m30[1].timestamp == datetime(2024, 1, 2, 0, 30, tzinfo=UTC)
    assert m30[0].open == pytest.approx(1.1)
    assert m30[0].high == pytest.approx(1.1002)
    assert m30[0].low == pytest.approx(1.0998)
    assert m30[0].close == pytest.approx(1.1001)
    assert m30[0].volume == pytest.approx(30_000.0)


def test_incomplete_m30_bucket_is_dropped() -> None:
    minutes = decode_minute_payload(_payload(29))

    assert aggregate_m30(minutes) == ()


def test_negative_time_delta_is_rejected() -> None:
    payload = _payload(3)
    payload["times"] = [0, 1, -1]

    with pytest.raises(ValueError):
        decode_minute_payload(payload)


def test_connection_reset_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    calls = 0

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    def fake_urlopen(*_args: object, **_kwargs: object) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError(10054, "connection reset by peer")
        return Response()

    monkeypatch.setattr(dukascopy.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dukascopy.time, "sleep", lambda _seconds: None)

    result = dukascopy._fetch_day("EUR/JPY", "BID", date(2021, 1, 14))

    assert result is not None
    assert result["timestamp"] == payload["timestamp"]
    assert calls == 2
