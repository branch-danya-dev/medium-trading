import json
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from medium_trading.domain import Candle

_PRACTICE_URL = "https://api-fxpractice.oanda.com"
_LIVE_URL = "https://api-fxtrade.oanda.com"


class OandaHistoryClient:
    def __init__(
        self,
        *,
        account_id: str,
        token: str,
        environment: str = "practice",
        timeout_seconds: float = 30.0,
    ) -> None:
        if environment not in {"practice", "live"}:
            raise ValueError("environment must be 'practice' or 'live'")
        self._account_id = account_id
        self._token = token
        self._base_url = _PRACTICE_URL if environment == "practice" else _LIVE_URL
        self._timeout_seconds = timeout_seconds

    def fetch_m30(
        self,
        *,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> tuple[Candle, ...]:
        start = _as_utc(start)
        end = _as_utc(end)
        if start >= end:
            raise ValueError("start must be before end")

        candles: dict[datetime, Candle] = {}
        cursor = start

        # 90 days of M30 candles stays below OANDA's 5,000-candle response limit.
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=90), end)
            for candle in self._fetch_chunk(
                instrument=instrument,
                start=cursor,
                end=chunk_end,
            ):
                candles[candle.timestamp] = candle
            cursor = chunk_end

        return tuple(candles[timestamp] for timestamp in sorted(candles))

    def _fetch_chunk(
        self,
        *,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> tuple[Candle, ...]:
        account_id = quote(self._account_id, safe="-_.")
        instrument_name = quote(instrument, safe="-_.")
        query = urlencode(
            {
                "price": "M",
                "granularity": "M30",
                "from": _format_timestamp(start),
                "to": _format_timestamp(end),
                "smooth": "false",
            }
        )
        url = (
            f"{self._base_url}/v3/accounts/{account_id}/instruments/"
            f"{instrument_name}/candles?{query}"
        )
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept-Datetime-Format": "RFC3339",
            },
        )

        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = json.load(response)

        result: list[Candle] = []
        for item in payload.get("candles", []):
            if not item.get("complete", False):
                continue
            mid = item["mid"]
            result.append(
                Candle(
                    timestamp=_parse_timestamp(item["time"]),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(item.get("volume", 0.0)),
                )
            )
        return tuple(result)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
