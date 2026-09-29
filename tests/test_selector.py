from medium_trading.domain import Side, Signal
from medium_trading.trading.selector import select_best


def _signal(symbol: str, confidence: float) -> Signal:
    return Signal(
        symbol=symbol,
        side=Side.LONG,
        entry=1.0,
        stop=0.9,
        confidence=confidence,
        strategy="test",
        reasons=("test",),
    )


def test_select_best_returns_highest_confidence_first() -> None:
    selected = select_best(
        [_signal("EUR/USD", 0.7), _signal("GBP/USD", 0.9)],
        limit=1,
    )

    assert selected[0].symbol == "GBP/USD"
