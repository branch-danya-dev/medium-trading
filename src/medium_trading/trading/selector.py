from collections.abc import Iterable

from medium_trading.domain import Signal


def select_best(signals: Iterable[Signal], limit: int = 1) -> tuple[Signal, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = sorted(signals, key=lambda signal: signal.confidence, reverse=True)
    return tuple(ranked[:limit])
