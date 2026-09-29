import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable, Mapping

_BASE_URL = "https://api.bybit.com"
_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_RET_CODES = {10006}
_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)


class BybitStateHistoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenInterestPoint:
    timestamp: datetime
    open_interest: float


@dataclass(frozen=True, slots=True)
class AccountRatioPoint:
    timestamp: datetime
    buy_ratio: float
    sell_ratio: float


@dataclass(frozen=True, slots=True)
class FundingPoint:
    timestamp: datetime
    funding_rate: float


def download_open_interest(
    *,
    symbol: str,
    start: date,
    end: date,
    interval: str = "30min",
    base_url: str = _BASE_URL,
    progress: Callable[[int], None] | None = None,
) -> tuple[OpenInterestPoint, ...]:
    if end < start:
        raise ValueError("end date cannot be before start date")
    if interval not in {"5min", "15min", "30min", "1h", "4h", "1d"}:
        raise ValueError("unsupported Bybit open-interest interval")

    start_ms, end_ms = _date_range_ms(start, end)
    cursor: str | None = None
    points: dict[int, OpenInterestPoint] = {}
    request_count = 0

    while True:
        query: dict[str, object] = {
            "category": "linear",
            "symbol": symbol.strip().upper(),
            "intervalTime": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200,
        }
        if cursor:
            query["cursor"] = cursor
        payload = _request_json(
            f"{base_url.rstrip('/')}/v5/market/open-interest",
            query,
        )
        result = _result(payload)
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitStateHistoryError("open-interest result.list must be an array")

        for row in rows:
            if not isinstance(row, Mapping):
                raise BybitStateHistoryError("open-interest row must be an object")
            timestamp_ms = _integer(row.get("timestamp"), field="timestamp")
            if start_ms <= timestamp_ms <= end_ms:
                points[timestamp_ms] = OpenInterestPoint(
                    timestamp=_timestamp(timestamp_ms),
                    open_interest=_number(
                        row.get("openInterest"),
                        field="openInterest",
                    ),
                )

        request_count += 1
        if progress is not None:
            progress(request_count)

        next_cursor = str(result.get("nextPageCursor") or "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return tuple(points[key] for key in sorted(points))


def download_account_ratio(
    *,
    symbol: str,
    start: date,
    end: date,
    period: str = "30min",
    base_url: str = _BASE_URL,
    progress: Callable[[int], None] | None = None,
) -> tuple[AccountRatioPoint, ...]:
    if end < start:
        raise ValueError("end date cannot be before start date")
    if period not in {"5min", "15min", "30min", "1h", "4h", "1d"}:
        raise ValueError("unsupported Bybit account-ratio period")

    start_ms, end_ms = _date_range_ms(start, end)
    cursor: str | None = None
    points: dict[int, AccountRatioPoint] = {}
    request_count = 0

    while True:
        query: dict[str, object] = {
            "category": "linear",
            "symbol": symbol.strip().upper(),
            "period": period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 500,
        }
        if cursor:
            query["cursor"] = cursor
        payload = _request_json(
            f"{base_url.rstrip('/')}/v5/market/account-ratio",
            query,
        )
        result = _result(payload)
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitStateHistoryError("account-ratio result.list must be an array")

        for row in rows:
            if not isinstance(row, Mapping):
                raise BybitStateHistoryError("account-ratio row must be an object")
            timestamp_ms = _integer(row.get("timestamp"), field="timestamp")
            if start_ms <= timestamp_ms <= end_ms:
                points[timestamp_ms] = AccountRatioPoint(
                    timestamp=_timestamp(timestamp_ms),
                    buy_ratio=_number(row.get("buyRatio"), field="buyRatio"),
                    sell_ratio=_number(row.get("sellRatio"), field="sellRatio"),
                )

        request_count += 1
        if progress is not None:
            progress(request_count)

        next_cursor = str(result.get("nextPageCursor") or "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return tuple(points[key] for key in sorted(points))


def download_funding_history(
    *,
    symbol: str,
    start: date,
    end: date,
    base_url: str = _BASE_URL,
    progress: Callable[[int], None] | None = None,
) -> tuple[FundingPoint, ...]:
    if end < start:
        raise ValueError("end date cannot be before start date")

    start_ms, end_ms = _date_range_ms(start, end)
    cursor_end = end_ms
    points: dict[int, FundingPoint] = {}
    request_count = 0

    while cursor_end >= start_ms:
        payload = _request_json(
            f"{base_url.rstrip('/')}/v5/market/funding/history",
            {
                "category": "linear",
                "symbol": symbol.strip().upper(),
                "endTime": cursor_end,
                "limit": 200,
            },
        )
        result = _result(payload)
        rows = result.get("list")
        if not isinstance(rows, list):
            raise BybitStateHistoryError("funding result.list must be an array")
        if not rows:
            break

        oldest: int | None = None
        for row in rows:
            if not isinstance(row, Mapping):
                raise BybitStateHistoryError("funding row must be an object")
            timestamp_ms = _integer(
                row.get("fundingRateTimestamp"),
                field="fundingRateTimestamp",
            )
            oldest = timestamp_ms if oldest is None else min(oldest, timestamp_ms)
            if start_ms <= timestamp_ms <= end_ms:
                points[timestamp_ms] = FundingPoint(
                    timestamp=_timestamp(timestamp_ms),
                    funding_rate=_number(
                        row.get("fundingRate"),
                        field="fundingRate",
                    ),
                )

        request_count += 1
        if progress is not None:
            progress(request_count)

        if oldest is None or oldest <= start_ms:
            break
        cursor_end = oldest - 1

    return tuple(points[key] for key in sorted(points))


def save_open_interest(
    path: Path,
    points: tuple[OpenInterestPoint, ...],
) -> None:
    _write_rows(
        path,
        ("timestamp", "open_interest"),
        (
            (point.timestamp.isoformat(), point.open_interest)
            for point in points
        ),
    )


def save_account_ratio(
    path: Path,
    points: tuple[AccountRatioPoint, ...],
) -> None:
    _write_rows(
        path,
        ("timestamp", "buy_ratio", "sell_ratio"),
        (
            (
                point.timestamp.isoformat(),
                point.buy_ratio,
                point.sell_ratio,
            )
            for point in points
        ),
    )


def save_funding(
    path: Path,
    points: tuple[FundingPoint, ...],
) -> None:
    _write_rows(
        path,
        ("timestamp", "funding_rate"),
        (
            (point.timestamp.isoformat(), point.funding_rate)
            for point in points
        ),
    )


def load_open_interest(path: Path) -> tuple[OpenInterestPoint, ...]:
    return tuple(
        OpenInterestPoint(
            timestamp=datetime.fromisoformat(row["timestamp"]).astimezone(UTC),
            open_interest=float(row["open_interest"]),
        )
        for row in _read_rows(path)
    )


def load_account_ratio(path: Path) -> tuple[AccountRatioPoint, ...]:
    return tuple(
        AccountRatioPoint(
            timestamp=datetime.fromisoformat(row["timestamp"]).astimezone(UTC),
            buy_ratio=float(row["buy_ratio"]),
            sell_ratio=float(row["sell_ratio"]),
        )
        for row in _read_rows(path)
    )


def load_funding(path: Path) -> tuple[FundingPoint, ...]:
    return tuple(
        FundingPoint(
            timestamp=datetime.fromisoformat(row["timestamp"]).astimezone(UTC),
            funding_rate=float(row["funding_rate"]),
        )
        for row in _read_rows(path)
    )


def _request_json(
    endpoint: str,
    query: Mapping[str, object],
) -> Mapping[str, object]:
    url = f"{endpoint}?{urllib.parse.urlencode(query)}"
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
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise BybitStateHistoryError("Bybit response must be a JSON object")
            ret_code = _integer(payload.get("retCode"), field="retCode")
            if ret_code == 0:
                return payload
            if ret_code not in _RETRYABLE_RET_CODES:
                ret_msg = str(payload.get("retMsg") or "unknown Bybit error")
                raise BybitStateHistoryError(
                    f"Bybit retCode={ret_code}: {ret_msg}"
                )
            error: Exception = BybitStateHistoryError(
                f"retryable Bybit retCode={ret_code}"
            )
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_CODES:
                raise BybitStateHistoryError(f"Bybit HTTP {exc.code}") from exc
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
        raise BybitStateHistoryError(
            f"Bybit request failed after retries: {error}"
        ) from error

    raise AssertionError("unreachable")


def _result(payload: Mapping[str, object]) -> Mapping[str, object]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise BybitStateHistoryError("Bybit response is missing result object")
    return result


def _date_range_ms(start: date, end: date) -> tuple[int, int]:
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


def _timestamp(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise BybitStateHistoryError(f"Bybit field {field!r} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BybitStateHistoryError(
            f"Bybit field {field!r} must be an integer"
        ) from exc


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise BybitStateHistoryError(f"Bybit field {field!r} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BybitStateHistoryError(
            f"Bybit field {field!r} must be numeric"
        ) from exc


def _write_rows(
    path: Path,
    header: tuple[str, ...],
    rows,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
