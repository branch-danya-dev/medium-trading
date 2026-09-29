from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Regime(str, Enum):
    TREND = "trend"
    RANGE = "range"
    COMPRESSION = "compression"
    CHAOTIC = "chaotic"


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketState:
    symbol: str
    regime: Regime
    side: Side | None
    trend_strength: float
    volatility: float


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: Side
    entry: float
    stop: float
    confidence: float
    strategy: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrategyContext:
    symbol: str
    candles_4h: tuple[Candle, ...]
    candles_1h: tuple[Candle, ...]
    candles_30m: tuple[Candle, ...]
