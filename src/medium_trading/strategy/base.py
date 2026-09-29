from typing import Protocol

from medium_trading.domain import Signal, StrategyContext


class Strategy(Protocol):
    name: str

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Return one actionable signal or None."""
        ...
