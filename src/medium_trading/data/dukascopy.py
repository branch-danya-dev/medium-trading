import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from medium_trading.domain import Candle


@dataclass(frozen=True, slots=True)
class DukascopyImport:
    candles: tuple[Candle, ...]
    source_kind: str
    mean_spread: float | None


_TIMESTAMP_ALIASES = (
    "timestamp",
    "datetime",
    "gmttime",
    "time",
    "date",
)
_OPEN_ALIASES = ("open", "openprice")
_HIGH_ALIASES = ("high", "highprice")
_LOW_ALIASES = ("low", "lowprice")
_CLOSE_ALIASES = ("close", "closeprice")
_VOLUME_ALIASES = ("volume", "vol", "tickvolume")
_BID_ALIASES = ("bid", "bidprice")
_ASK_ALIASES = ("ask", "askprice")
_BID_VOLUME_ALIASES = ("bidvolume", "bidvol")
_ASK_VOLUME_ALIASES = ("askvolume", "askvol")


def import_dukascopy(path: str | Path) -> DukascopyImport:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = _dict_reader(handle)
        if reader.fieldnames is None:
            raise ValueError("Dukascopy CSV has no header")

        normalized = {_normalize_header(name): name for name in reader.fieldnames}
        timestamp_key = _find_key(normalized, _TIMESTAMP_ALIASES)
        if timestamp_key is None:
            raise ValueError("could not find a timestamp column in Dukascopy CSV")

        candle_keys = _candle_keys(normalized)
        if candle_keys is not None:
            candles = _read_candles(reader, timestamp_key, candle_keys)
            return DukascopyImport(
                candles=candles,
                source_kind="candles",
                mean_spread=None,
            )

        bid_key = _find_key(normalized, _BID_ALIASES)
        ask_key = _find_key(normalized, _ASK_ALIASES)
        if bid_key is not None and ask_key is not None:
            return _read_ticks(
                reader,
                timestamp_key=timestamp_key,
                bid_key=bid_key,
                ask_key=ask_key,
                bid_volume_key=_find_key(normalized, _BID_VOLUME_ALIASES),
                ask_volume_key=_find_key(normalized, _ASK_VOLUME_ALIASES),
            )

    raise ValueError(
        "unsupported Dukascopy CSV: expected OHLC candles or bid/ask tick columns"
    )


def _dict_reader(handle: TextIO) -> csv.DictReader:
    sample = handle.read(8192)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return csv.DictReader(handle, dialect=dialect)


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_key(
    normalized: dict[str, str],
    aliases: tuple[str, ...],
) -> str | None:
    for alias in aliases:
        key = normalized.get(alias)
        if key is not None:
            return key
    return None


def _candle_keys(
    normalized: dict[str, str],
) -> tuple[str, str, str, str, str | None] | None:
    open_key = _find_key(normalized, _OPEN_ALIASES)
    high_key = _find_key(normalized, _HIGH_ALIASES)
    low_key = _find_key(normalized, _LOW_ALIASES)
    close_key = _find_key(normalized, _CLOSE_ALIASES)
    if None in {open_key, high_key, low_key, close_key}:
        return None
    return (
        str(open_key),
        str(high_key),
        str(low_key),
        str(close_key),
        _find_key(normalized, _VOLUME_ALIASES),
    )


def _read_candles(
    reader: csv.DictReader,
    timestamp_key: str,
    keys: tuple[str, str, str, str, str | None],
) -> tuple[Candle, ...]:
    open_key, high_key, low_key, close_key, volume_key = keys
    candles: list[Candle] = []

    for row in reader:
        if not row.get(timestamp_key):
            continue
        candle = Candle(
            timestamp=_parse_timestamp(row[timestamp_key]),
            open=_float(row[open_key]),
            high=_float(row[high_key]),
            low=_float(row[low_key]),
            close=_float(row[close_key]),
            volume=_float(row.get(volume_key) if volume_key else None),
        )
        _validate_m30_timestamp(candle.timestamp)
        candles.append(candle)

    return _sort_unique(candles)


def _read_ticks(
    reader: csv.DictReader,
    *,
    timestamp_key: str,
    bid_key: str,
    ask_key: str,
    bid_volume_key: str | None,
    ask_volume_key: str | None,
) -> DukascopyImport:
    candles: list[Candle] = []
    current_bucket: datetime | None = None
    open_price = high = low = close = 0.0
    volume = 0.0
    spread_total = 0.0
    spread_count = 0
    total_spread = 0.0
    total_ticks = 0
    previous_timestamp: datetime | None = None

    def flush() -> None:
        nonlocal current_bucket
        if current_bucket is None:
            return
        candles.append(
            Candle(
                timestamp=current_bucket,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                spread=spread_total / spread_count if spread_count else 0.0,
            )
        )

    for row in reader:
        if not row.get(timestamp_key):
            continue
        timestamp = _parse_timestamp(row[timestamp_key])
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("Dukascopy tick CSV must be ordered chronologically")
        previous_timestamp = timestamp

        bid = _float(row[bid_key])
        ask = _float(row[ask_key])
        if ask < bid:
            raise ValueError("Dukascopy tick has ask below bid")

        midpoint = (bid + ask) / 2.0
        tick_spread = ask - bid
        bucket = timestamp.replace(
            minute=(timestamp.minute // 30) * 30,
            second=0,
            microsecond=0,
        )
        tick_volume = _float(row.get(bid_volume_key) if bid_volume_key else None)
        tick_volume += _float(row.get(ask_volume_key) if ask_volume_key else None)

        if current_bucket != bucket:
            flush()
            current_bucket = bucket
            open_price = high = low = close = midpoint
            volume = 0.0
            spread_total = 0.0
            spread_count = 0

        high = max(high, midpoint)
        low = min(low, midpoint)
        close = midpoint
        volume += tick_volume
        spread_total += tick_spread
        spread_count += 1
        total_spread += tick_spread
        total_ticks += 1

    flush()

    return DukascopyImport(
        candles=tuple(candles),
        source_kind="ticks",
        mean_spread=total_spread / total_ticks if total_ticks else None,
    )


def _sort_unique(candles: list[Candle]) -> tuple[Candle, ...]:
    candles.sort(key=lambda candle: candle.timestamp)
    for previous, current in zip(candles, candles[1:], strict=False):
        if current.timestamp <= previous.timestamp:
            raise ValueError("Dukascopy candle timestamps must be unique")
    return tuple(candles)


def _parse_timestamp(value: str) -> datetime:
    clean = value.strip()
    iso_candidate = clean.replace("Z", "+00:00")

    try:
        timestamp = datetime.fromisoformat(iso_candidate)
    except ValueError:
        formats = (
            "%d.%m.%Y %H:%M:%S.%f",
            "%d.%m.%Y %H:%M:%S",
            "%Y.%m.%d %H:%M:%S.%f",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        )
        for format_string in formats:
            try:
                timestamp = datetime.strptime(clean, format_string)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"unsupported Dukascopy timestamp: {value!r}") from None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _validate_m30_timestamp(timestamp: datetime) -> None:
    if timestamp.minute not in {0, 30} or timestamp.second != 0:
        raise ValueError(
            "candle CSV must contain M30 candles; export 30-minute candles from Dukascopy"
        )


def _float(value: str | None) -> float:
    if value is None or not value.strip():
        return 0.0
    return float(value.strip().replace(",", "."))
