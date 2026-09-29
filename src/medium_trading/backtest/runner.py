from collections.abc import Iterable

from medium_trading.domain import Signal, StrategyContext
from medium_trading.strategy.base import Strategy


def generate_signals(
    strategy: Strategy,
    contexts: Iterable[StrategyContext],
) -> tuple[Signal, ...]:
    signals: list[Signal] = []
    for context in contexts:
        signal = strategy.evaluate(context)
        if signal is not None:
            signals.append(signal)
    return tuple(signals)
