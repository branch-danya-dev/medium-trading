from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    symbols: tuple[str, ...] = ("EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD")
    timeframes: tuple[str, ...] = ("4h", "1h", "30m")
    starting_equity_usd: float = 1_000.0
    risk_per_trade: float = 0.005
    max_open_risk: float = 0.01
