from datetime import UTC, date, datetime, timedelta

import pytest

import medium_trading.data.bybit_state_history as state


def _ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


def test_open_interest_uses_cursor_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    calls = []

    def fake_request(endpoint: str, query: dict[str, object]):
        calls.append((endpoint, dict(query)))
        cursor = query.get("cursor")
        if cursor is None:
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "timestamp": _ms(start + timedelta(minutes=30)),
                            "openInterest": "101.5",
                        }
                    ],
                    "nextPageCursor": "next",
                },
            }
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "timestamp": _ms(start),
                        "openInterest": "100.0",
                    }
                ],
                "nextPageCursor": "",
            },
        }

    monkeypatch.setattr(state, "_request_json", fake_request)

    points = state.download_open_interest(
        symbol="BTCUSDT",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
    )

    assert [point.open_interest for point in points] == [100.0, 101.5]
    assert calls[0][1]["intervalTime"] == "30min"
    assert calls[1][1]["cursor"] == "next"


def test_account_ratio_decodes_buy_and_sell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)

    def fake_request(_endpoint: str, _query: dict[str, object]):
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "timestamp": _ms(timestamp),
                        "buyRatio": "0.52",
                        "sellRatio": "0.48",
                    }
                ],
                "nextPageCursor": "",
            },
        }

    monkeypatch.setattr(state, "_request_json", fake_request)

    points = state.download_account_ratio(
        symbol="BTCUSDT",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
    )

    assert len(points) == 1
    assert points[0].buy_ratio == pytest.approx(0.52)
    assert points[0].sell_ratio == pytest.approx(0.48)


def test_funding_history_pages_backwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    calls = 0

    def fake_request(_endpoint: str, query: dict[str, object]):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "fundingRateTimestamp": _ms(
                                start + timedelta(hours=16)
                            ),
                            "fundingRate": "0.0001",
                        },
                        {
                            "fundingRateTimestamp": _ms(
                                start + timedelta(hours=8)
                            ),
                            "fundingRate": "0.0002",
                        },
                    ]
                },
            }
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "fundingRateTimestamp": _ms(start),
                        "fundingRate": "-0.0001",
                    }
                ]
            },
        }

    monkeypatch.setattr(state, "_request_json", fake_request)

    points = state.download_funding_history(
        symbol="BTCUSDT",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
    )

    assert calls == 2
    assert [point.funding_rate for point in points] == [
        -0.0001,
        0.0002,
        0.0001,
    ]


def test_state_csv_roundtrip(tmp_path) -> None:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    oi_path = tmp_path / "oi.csv"
    ratio_path = tmp_path / "ratio.csv"
    funding_path = tmp_path / "funding.csv"

    state.save_open_interest(
        oi_path,
        (state.OpenInterestPoint(timestamp, 123.0),),
    )
    state.save_account_ratio(
        ratio_path,
        (state.AccountRatioPoint(timestamp, 0.51, 0.49),),
    )
    state.save_funding(
        funding_path,
        (state.FundingPoint(timestamp, 0.0001),),
    )

    assert state.load_open_interest(oi_path)[0].open_interest == 123.0
    assert state.load_account_ratio(ratio_path)[0].buy_ratio == 0.51
    assert state.load_funding(funding_path)[0].funding_rate == 0.0001
