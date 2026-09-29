from collections.abc import Sequence
from typing import Protocol

from medium_trading.domain import Candle, Signal


class Broker(Protocol):
    def get_closed_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Sequence[Candle]:
        """Return normalized closed candles only."""
        ...

    def account_equity(self) -> float:
        """Return account equity in the account currency."""
        ...

    def place_order(self, signal: Signal, quantity: float) -> str:
        """Place one order and return the broker order identifier."""
        ...
