import csv
import gzip
import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Callable

_DEFAULT_ARCHIVE_BASE = "https://public.bybit.com/trading"
_RETRY_DELAYS = (1.0, 2.0, 4.0)
_REQUIRED_COLUMNS = {"timestamp", "symbol", "side", "size", "price"}
_BUCKET_SECONDS = 300
_BUCKETS_PER_DAY = 24 * 60 // 5


class BybitTradeFlowError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TradeFlowPoint:
    timestamp: datetime
    buy_qty: float
    sell_qty: float
    buy_notional: float
    sell_notional: float
    buy_count: int
    sell_count: int

    @property
    def total_qty(self) -> float:
        return self.buy_qty + self.sell_qty

    @property
    def total_notional(self) -> float:
        return self.buy_notional + self.sell_notional

    @property
    def total_count(self) -> int:
        return self.buy_count + self.sell_count


def download_trade_flow_range(
    *,
    symbol: str,
    start: date,
    end: date,
    output_dir: Path,
    base_url: str = _DEFAULT_ARCHIVE_BASE,
    workers: int = 2,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[TradeFlowPoint, ...]:
    """
    Stream Bybit's daily public trade archives and persist only aggregated M5 flow.

    Raw tick files are never written to disk. Each completed day is cached as an
    aggregated M5 CSV so interrupted multi-year downloads can resume without
    re-downloading finished days.
    """
    if end < start:
        raise ValueError("end date cannot be before start date")
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")

    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol cannot be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    days = _date_range(start, end)
    results: dict[date, tuple[TradeFlowPoint, ...]] = {}
    pending: list[date] = []

    for day in days:
        cache = _daily_path(daily_dir, normalized_symbol, day)
        if cache.exists():
            points = load_trade_flow(cache)
            _validate_day(points, day)
            results[day] = points
        else:
            pending.append(day)

    completed = len(results)
    if progress is not None and completed:
        progress(completed, len(days))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_trade_flow_day,
                    symbol=normalized_symbol,
                    day=day,
                    base_url=base_url,
                ): day
                for day in pending
            }
            for future in as_completed(futures):
                day = futures[future]
                points = future.result()
                save_trade_flow(
                    _daily_path(daily_dir, normalized_symbol, day),
                    points,
                )
                results[day] = points
                completed += 1
                if progress is not None:
                    progress(completed, len(days))

    combined = tuple(
        point
        for day in days
        for point in results[day]
    )
    _validate_range(combined, start, end)
    combined_path = output_dir / f"{normalized_symbol}_TRADE_FLOW_M5.csv"
    save_trade_flow(combined_path, combined)
    return combined


def download_trade_flow_day(
    *,
    symbol: str,
    day: date,
    base_url: str = _DEFAULT_ARCHIVE_BASE,
) -> tuple[TradeFlowPoint, ...]:
    normalized_symbol = symbol.strip().upper()
    filename = f"{normalized_symbol}{day.isoformat()}.csv.gz"
    url = f"{base_url.rstrip('/')}/{normalized_symbol}/{filename}"

    attempts = len(_RETRY_DELAYS) + 1
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/csv,application/gzip,*/*",
                "Accept-Encoding": "identity",
                "User-Agent": "medium-trading/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60.0) as response:
                points = _aggregate_archive(
                    response,
                    symbol=normalized_symbol,
                    day=day,
                )
            _validate_day(points, day)
            return points
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise BybitTradeFlowError(
                    f"missing Bybit trade archive for {normalized_symbol} {day}"
                ) from exc
            error: Exception = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            csv.Error,
        ) as exc:
            error = exc

        if attempt < len(_RETRY_DELAYS):
            time.sleep(_RETRY_DELAYS[attempt])
            continue
        raise BybitTradeFlowError(
            f"failed to download {url} after retries: {error}"
        ) from error

    raise AssertionError("unreachable")


def save_trade_flow(
    path: Path,
    points: tuple[TradeFlowPoint, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "buy_qty",
                "sell_qty",
                "buy_notional",
                "sell_notional",
                "buy_count",
                "sell_count",
            )
        )
        writer.writerows(
            (
                point.timestamp.isoformat(),
                point.buy_qty,
                point.sell_qty,
                point.buy_notional,
                point.sell_notional,
                point.buy_count,
                point.sell_count,
            )
            for point in points
        )


def load_trade_flow(path: Path) -> tuple[TradeFlowPoint, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    points = tuple(
        TradeFlowPoint(
            timestamp=datetime.fromisoformat(row["timestamp"]).astimezone(UTC),
            buy_qty=float(row["buy_qty"]),
            sell_qty=float(row["sell_qty"]),
            buy_notional=float(row["buy_notional"]),
            sell_notional=float(row["sell_notional"]),
            buy_count=int(row["buy_count"]),
            sell_count=int(row["sell_count"]),
        )
        for row in rows
    )
    _validate_contiguous(points)
    return points


def _aggregate_archive(
    binary_stream,
    *,
    symbol: str,
    day: date,
) -> tuple[TradeFlowPoint, ...]:
    buckets: dict[int, list[float | int]] = {}

    with gzip.GzipFile(fileobj=binary_stream, mode="rb") as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            columns = set(reader.fieldnames or ())
            missing = _REQUIRED_COLUMNS - columns
            if missing:
                raise BybitTradeFlowError(
                    "Bybit trade archive is missing required columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:
                if str(row["symbol"]).strip().upper() != symbol:
                    continue
                timestamp = _timestamp_seconds(row["timestamp"])
                trade_time = datetime.fromtimestamp(timestamp, tz=UTC)
                if trade_time.date() != day:
                    continue
                side = str(row["side"]).strip().upper()
                if side not in {"BUY", "SELL"}:
                    raise BybitTradeFlowError(
                        f"unsupported trade side {row['side']!r}"
                    )
                try:
                    size = float(row["size"])
                    price = float(row["price"])
                except (TypeError, ValueError) as exc:
                    raise BybitTradeFlowError(
                        "trade size and price must be numeric"
                    ) from exc
                if size < 0 or price <= 0:
                    raise BybitTradeFlowError(
                        "trade size must be non-negative and price positive"
                    )

                bucket_epoch = (
                    int(timestamp) // _BUCKET_SECONDS * _BUCKET_SECONDS
                )
                values = buckets.setdefault(
                    bucket_epoch,
                    [0.0, 0.0, 0.0, 0.0, 0, 0],
                )
                notional = size * price
                if side == "BUY":
                    values[0] = float(values[0]) + size
                    values[2] = float(values[2]) + notional
                    values[4] = int(values[4]) + 1
                else:
                    values[1] = float(values[1]) + size
                    values[3] = float(values[3]) + notional
                    values[5] = int(values[5]) + 1

    return tuple(
        TradeFlowPoint(
            timestamp=datetime.fromtimestamp(epoch, tz=UTC),
            buy_qty=float(values[0]),
            sell_qty=float(values[1]),
            buy_notional=float(values[2]),
            sell_notional=float(values[3]),
            buy_count=int(values[4]),
            sell_count=int(values[5]),
        )
        for epoch, values in sorted(buckets.items())
    )


def _timestamp_seconds(value: object) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise BybitTradeFlowError(
            f"invalid archive timestamp {value!r}"
        ) from exc
    if timestamp > 10_000_000_000:
        timestamp /= 1000.0
    return timestamp


def _validate_day(
    points: tuple[TradeFlowPoint, ...],
    day: date,
) -> None:
    if len(points) != _BUCKETS_PER_DAY:
        raise BybitTradeFlowError(
            f"{day} trade-flow coverage has {len(points)} M5 buckets; "
            f"expected {_BUCKETS_PER_DAY}"
        )
    expected_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    if points[0].timestamp != expected_start:
        raise BybitTradeFlowError(
            f"{day} trade-flow starts at {points[0].timestamp.isoformat()} "
            f"instead of {expected_start.isoformat()}"
        )
    _validate_contiguous(points)


def _validate_range(
    points: tuple[TradeFlowPoint, ...],
    start: date,
    end: date,
) -> None:
    expected_days = (end - start).days + 1
    expected_points = expected_days * _BUCKETS_PER_DAY
    if len(points) != expected_points:
        raise BybitTradeFlowError(
            f"trade-flow range has {len(points)} points; "
            f"expected {expected_points}"
        )
    _validate_contiguous(points)


def _validate_contiguous(points: tuple[TradeFlowPoint, ...]) -> None:
    for previous, current in zip(points, points[1:]):
        if current.timestamp - previous.timestamp != timedelta(minutes=5):
            raise BybitTradeFlowError(
                "trade-flow gap between "
                f"{previous.timestamp.isoformat()} and "
                f"{current.timestamp.isoformat()}"
            )


def _daily_path(
    directory: Path,
    symbol: str,
    day: date,
) -> Path:
    return directory / f"{symbol}_{day.isoformat()}_FLOW_M5.csv"


def _date_range(start: date, end: date) -> tuple[date, ...]:
    count = (end - start).days + 1
    return tuple(start + timedelta(days=offset) for offset in range(count))
