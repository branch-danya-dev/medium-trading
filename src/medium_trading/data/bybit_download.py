import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Callable, Mapping, Sequence

from medium_trading.domain import Candle

_BASE_URL = "https://api.bybit.com/v5/market/kline"
_INTERVAL = "30"
_INTERVAL_DELTA = timedelta(minutes=30)
_INTERVAL_MS = 30 * 60 * 1000
_PAGE_LIMIT = 1000
_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_RET_CODES = {10006}
_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)


class BybitDownloadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BybitDownloadResult:
    symbol: str
    category: str
    start: date
    end: date
    candles: tuple[Candle, ...]
    request_count: int
    expected_candles: int


def build_kline_url(
    *,
    symbol: str,
    category: str,
    start_ms: int,
    end_ms: int,
    limit: int = _PAGE_LIMIT,
    base_url: str = _BASE_URL,
) -> str:
    normalized_symbol = symbol.strip().upper()
    normalized_category = category.strip().lower()
    if not normalized_symbol:
        raise ValueError("Bybit symbol cannot be empty")
    if normalized_category not in {"linear", "inverse", "spot"}:
        raise ValueError("Bybit category must be linear, inverse or spot")
    if start_ms < 0 or end_ms < start_ms:
        raise ValueError("invalid Bybit kline timestamp range")
    if not 1 <= limit <= _PAGE_LIMIT:
        raise ValueError("Bybit kline limit must be between 1 and 1000")

    query = urllib.parse.urlencode(
        {
            "category": normalized_category,
            "symbol": normalized_symbol,
            "interval": _INTERVAL,
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
        }
    )
    return f"{base_url.rstrip('/')}?{query}"


def decode_kline_payload(
    payload: Mapping[str, object],
    *,
    expected_symbol: str,
    expected_category: str,
) -> tuple[Candle, ...]:
    ret_code = _integer(payload.get("retCode"), field="retCode")
    if ret_code != 0:
        ret_msg = str(payload.get("retMsg") or "unknown Bybit error")
        raise BybitDownloadError(f"Bybit retCode={ret_code}: {ret_msg}")

    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise BybitDownloadError("Bybit response is missing result object")

    symbol = str(result.get("symbol") or "").upper()
    category = str(result.get("category") or "").lower()
    if symbol != expected_symbol.strip().upper():
        raise BybitDownloadError(
            f"Bybit returned symbol {symbol!r}, expected {expected_symbol!r}"
        )
    if category != expected_category.strip().lower():
        raise BybitDownloadError(
            f"Bybit returned category {category!r}, expected {expected_category!r}"
        )

    rows = result.get("list")
    if not isinstance(rows, list):
        raise BybitDownloadError("Bybit result.list must be an array")

    candles: list[Candle] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise BybitDownloadError("Bybit kline row must be an array")
        if len(row) < 6:
            raise BybitDownloadError("Bybit kline row must contain at least 6 fields")

        timestamp_ms = _integer(row[0], field="startTime")
        candles.append(
            Candle(
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        )

    candles.sort(key=lambda candle: candle.timestamp)
    _ensure_unique(candles)
    return tuple(candles)


def download_m30(
    *,
    symbol: str,
    category: str,
    start: date,
    end: date,
    progress: Callable[[int, int], None] | None = None,
    base_url: str = _BASE_URL,
) -> BybitDownloadResult:
    if end < start:
        raise ValueError("end date cannot be before start date")

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    end_exclusive = datetime.combine(
        end + timedelta(days=1),
        datetime.min.time(),
        tzinfo=UTC,
    )
    start_ms = int(start_dt.timestamp() * 1000)
    end_exclusive_ms = int(end_exclusive.timestamp() * 1000)

    expected_candles = int((end_exclusive - start_dt) / _INTERVAL_DELTA)
    total_requests = (expected_candles + _PAGE_LIMIT - 1) // _PAGE_LIMIT
    request_count = 0
    candles: list[Candle] = []
    cursor_ms = start_ms

    while cursor_ms < end_exclusive_ms:
        chunk_end_exclusive = min(
            cursor_ms + _PAGE_LIMIT * _INTERVAL_MS,
            end_exclusive_ms,
        )
        payload = _fetch_kline_page(
            symbol=symbol,
            category=category,
            start_ms=cursor_ms,
            end_ms=chunk_end_exclusive - 1,
            base_url=base_url,
        )
        page = decode_kline_payload(
            payload,
            expected_symbol=symbol,
            expected_category=category,
        )
        candles.extend(
            candle
            for candle in page
            if start_dt <= candle.timestamp < end_exclusive
        )

        request_count += 1
        if progress is not None:
            progress(request_count, total_requests)
        cursor_ms = chunk_end_exclusive

    candles.sort(key=lambda candle: candle.timestamp)
    _ensure_unique(candles)
    _validate_complete_range(
        candles=tuple(candles),
        start=start_dt,
        end_exclusive=end_exclusive,
        expected_candles=expected_candles,
    )

    return BybitDownloadResult(
        symbol=symbol.strip().upper(),
        category=category.strip().lower(),
        start=start,
        end=end,
        candles=tuple(candles),
        request_count=request_count,
        expected_candles=expected_candles,
    )


def _fetch_kline_page(
    *,
    symbol: str,
    category: str,
    start_ms: int,
    end_ms: int,
    base_url: str = _BASE_URL,
) -> Mapping[str, object]:
    url = build_kline_url(
        symbol=symbol,
        category=category,
        start_ms=start_ms,
        end_ms=end_ms,
        base_url=base_url,
    )
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
                raw = response.read()
            if not raw:
                raise BybitDownloadError("Bybit returned an empty response")
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise BybitDownloadError("Bybit response must be a JSON object")

            ret_code = _integer(payload.get("retCode"), field="retCode")
            if ret_code == 0:
                return payload
            if ret_code not in _RETRYABLE_RET_CODES:
                ret_msg = str(payload.get("retMsg") or "unknown Bybit error")
                raise BybitDownloadError(
                    f"Bybit retCode={ret_code}: {ret_msg}"
                )
            error: Exception = BybitDownloadError(
                f"retryable Bybit retCode={ret_code}"
            )
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_CODES:
                raise BybitDownloadError(f"Bybit HTTP {exc.code}") from exc
            error = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            error = exc

        if attempt < len(_RETRY_DELAYS):
            time.sleep(_RETRY_DELAYS[attempt])
            continue
        raise BybitDownloadError(f"Bybit request failed after retries: {error}") from error

    raise AssertionError("unreachable")


def _validate_complete_range(
    *,
    candles: tuple[Candle, ...],
    start: datetime,
    end_exclusive: datetime,
    expected_candles: int,
) -> None:
    if len(candles) != expected_candles:
        raise BybitDownloadError(
            "Bybit history is incomplete: "
            f"expected {expected_candles} M30 candles, received {len(candles)}"
        )
    if not candles:
        raise BybitDownloadError("Bybit returned no M30 candles")
    if candles[0].timestamp != start:
        raise BybitDownloadError(
            f"Bybit history starts at {candles[0].timestamp.isoformat()}, "
            f"expected {start.isoformat()}"
        )
    expected_last = end_exclusive - _INTERVAL_DELTA
    if candles[-1].timestamp != expected_last:
        raise BybitDownloadError(
            f"Bybit history ends at {candles[-1].timestamp.isoformat()}, "
            f"expected {expected_last.isoformat()}"
        )

    for previous, current in pairwise(candles):
        if current.timestamp - previous.timestamp != _INTERVAL_DELTA:
            raise BybitDownloadError(
                "Bybit M30 history contains a gap between "
                f"{previous.timestamp.isoformat()} and {current.timestamp.isoformat()}"
            )


def _ensure_unique(candles: Sequence[Candle]) -> None:
    for previous, current in zip(candles, candles[1:], strict=False):
        if current.timestamp <= previous.timestamp:
            raise BybitDownloadError(
                "Bybit candle timestamps must be unique and strictly increasing"
            )


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise BybitDownloadError(f"Bybit field {field!r} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BybitDownloadError(
            f"Bybit field {field!r} must be an integer"
        ) from exc
