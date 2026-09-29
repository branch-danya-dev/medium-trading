import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Callable, Mapping

from medium_trading.domain import Candle

_BASE_URL = "https://jetta.dukascopy.com/v1/candles/minute"
_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)
_MILLION = Decimal("1000000")
_CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD"}


class DukascopyDownloadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DukascopyDownloadResult:
    symbol: str
    start: date
    end: date
    candles: tuple[Candle, ...]
    requested_days: int
    data_days: int
    empty_days: int


def endpoint_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("_", "/").replace("-", "/")
    parts = normalized.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("symbol must look like EUR/USD, EUR_USD or EUR-USD")
    return f"{parts[0]}-{parts[1]}"


def build_endpoint_url(symbol: str, side: str, requested_date: date) -> str:
    normalized_side = side.strip().upper()
    if normalized_side not in {"BID", "ASK"}:
        raise ValueError("side must be BID or ASK")
    instrument = endpoint_symbol(symbol)
    return (
        f"{_BASE_URL}/{instrument}/{normalized_side}/"
        f"{requested_date.year}/{requested_date.month}/{requested_date.day}"
    )


def decode_minute_payload(payload: Mapping[str, object]) -> tuple[Candle, ...]:
    timestamp_ms = _int_field(payload, "timestamp")
    multiplier = Decimal(str(_required(payload, "multiplier")))
    shift_ms = _int_field(payload, "shift")

    if timestamp_ms <= 0:
        raise ValueError("Dukascopy timestamp must be positive")
    if multiplier <= 0:
        raise ValueError("Dukascopy multiplier must be positive")
    if shift_ms <= 0:
        raise ValueError("Dukascopy shift must be positive")

    times = _list_field(payload, "times")
    opens = _list_field(payload, "opens")
    highs = _list_field(payload, "highs")
    lows = _list_field(payload, "lows")
    closes = _list_field(payload, "closes")
    volumes = _list_field(payload, "volumes")

    length = len(times)
    if any(len(series) != length for series in (opens, highs, lows, closes, volumes)):
        raise ValueError("Dukascopy compressed arrays must have equal lengths")
    if length == 0:
        return ()

    expanded = {
        "open": _expand_prices(payload, "open", opens, multiplier),
        "high": _expand_prices(payload, "high", highs, multiplier),
        "low": _expand_prices(payload, "low", lows, multiplier),
        "close": _expand_prices(payload, "close", closes, multiplier),
    }

    candles: list[Candle] = []
    elapsed_units = 0
    previous_timestamp: datetime | None = None

    for index, raw_delta in enumerate(times):
        delta = int(raw_delta)
        if delta < 0:
            raise ValueError("Dukascopy time deltas cannot be negative")
        elapsed_units += delta

        candle_time = datetime.fromtimestamp(
            (timestamp_ms + elapsed_units * shift_ms) / 1000,
            tz=UTC,
        )
        if previous_timestamp is not None and candle_time <= previous_timestamp:
            raise ValueError("decoded Dukascopy timestamps must be strictly increasing")
        previous_timestamp = candle_time

        volume = Decimal(str(volumes[index])) * _MILLION
        if volume < 0:
            raise ValueError("Dukascopy volume cannot be negative")

        candles.append(
            Candle(
                timestamp=candle_time,
                open=expanded["open"][index],
                high=expanded["high"][index],
                low=expanded["low"][index],
                close=expanded["close"][index],
                volume=float(volume),
            )
        )

    return tuple(candles)


def aggregate_m30(candles_m1: tuple[Candle, ...]) -> tuple[Candle, ...]:
    buckets: dict[datetime, list[Candle]] = {}
    for candle in candles_m1:
        timestamp = candle.timestamp.astimezone(UTC)
        bucket = timestamp.replace(
            minute=(timestamp.minute // 30) * 30,
            second=0,
            microsecond=0,
        )
        buckets.setdefault(bucket, []).append(candle)

    result: list[Candle] = []
    for bucket in sorted(buckets):
        group = sorted(buckets[bucket], key=lambda candle: candle.timestamp)
        if len(group) != 30:
            continue

        result.append(
            Candle(
                timestamp=bucket,
                open=group[0].open,
                high=max(candle.high for candle in group),
                low=min(candle.low for candle in group),
                close=group[-1].close,
                volume=sum(candle.volume for candle in group),
            )
        )
    return tuple(result)


def download_m30(
    *,
    symbol: str,
    start: date,
    end: date,
    side: str = "BID",
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> DukascopyDownloadResult:
    if end < start:
        raise ValueError("end date cannot be before start date")
    if workers <= 0:
        raise ValueError("workers must be positive")

    dates = _requested_dates(symbol=symbol, start=start, end=end)

    candles: list[Candle] = []
    data_days = 0
    empty_days = 0
    failed: list[tuple[date, str]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_day, symbol, side, requested_date): requested_date
            for requested_date in dates
        }

        for future in as_completed(futures):
            requested_date = futures[future]
            try:
                payload = future.result()
                if payload is None:
                    empty_days += 1
                else:
                    minute_candles = decode_minute_payload(payload)
                    if minute_candles:
                        candles.extend(aggregate_m30(minute_candles))
                        data_days += 1
                    else:
                        empty_days += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((requested_date, str(exc)))

            completed += 1
            if progress is not None:
                progress(completed, len(dates))

    if failed:
        examples = ", ".join(
            f"{requested_date.isoformat()}: {message}"
            for requested_date, message in sorted(failed)[:5]
        )
        raise DukascopyDownloadError(
            f"{len(failed)} day(s) failed after retries. First failures: {examples}"
        )

    candles.sort(key=lambda candle: candle.timestamp)
    _ensure_unique(candles)

    return DukascopyDownloadResult(
        symbol=symbol,
        start=start,
        end=end,
        candles=tuple(candles),
        requested_days=len(dates),
        data_days=data_days,
        empty_days=empty_days,
    )


def _fetch_day(symbol: str, side: str, requested_date: date) -> Mapping[str, object] | None:
    url = build_endpoint_url(symbol, side, requested_date)
    attempts = len(_RETRY_DELAYS) + 1

    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "medium-trading/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                if response.status == 204:
                    return None
                raw = response.read()
                if not raw:
                    return None
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Dukascopy response must be a JSON object")
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in _RETRYABLE_HTTP_CODES:
                raise DukascopyDownloadError(
                    f"HTTP {exc.code} for {requested_date.isoformat()}"
                ) from exc
            error: Exception = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = exc

        if attempt < len(_RETRY_DELAYS):
            time.sleep(_RETRY_DELAYS[attempt])
            continue
        raise DukascopyDownloadError(
            f"request failed for {requested_date.isoformat()}: {error}"
        ) from error

    raise AssertionError("unreachable")


def _required(payload: Mapping[str, object], field: str) -> object:
    if field not in payload:
        raise ValueError(f"Dukascopy response is missing {field!r}")
    return payload[field]


def _int_field(payload: Mapping[str, object], field: str) -> int:
    value = _required(payload, field)
    if isinstance(value, bool):
        raise ValueError(f"Dukascopy field {field!r} must be an integer")
    return int(value)


def _list_field(payload: Mapping[str, object], field: str) -> list[object]:
    value = _required(payload, field)
    if not isinstance(value, list):
        raise ValueError(f"Dukascopy field {field!r} must be an array")
    return value


def _expand_prices(
    payload: Mapping[str, object],
    base_field: str,
    deltas: list[object],
    multiplier: Decimal,
) -> list[float]:
    if not deltas:
        return []

    first_delta = int(deltas[0])
    if first_delta != 0:
        raise ValueError(f"Dukascopy {base_field} delta series must start with zero")

    current = Decimal(str(_required(payload, base_field)))
    values: list[float] = []
    for index, raw_delta in enumerate(deltas):
        delta = int(raw_delta)
        if index > 0:
            current += Decimal(delta) * multiplier
        values.append(float(current))
    return values


def _ensure_unique(candles: list[Candle]) -> None:
    for previous, current in zip(candles, candles[1:], strict=False):
        if current.timestamp <= previous.timestamp:
            raise ValueError("downloaded M30 timestamps are not unique and increasing")


def _requested_dates(*, symbol: str, start: date, end: date) -> tuple[date, ...]:
    normalized = symbol.strip().upper().replace("_", "/").replace("-", "/")
    include_saturdays = normalized in _CRYPTO_SYMBOLS
    return tuple(
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
        if include_saturdays or (start + timedelta(days=offset)).weekday() != 5
    )
