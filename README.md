# medium-trading

Compact medium-frequency systematic trading bot.

## MVP thesis

The first version is intentionally small:

- market: spot FX;
- symbols: EUR/USD, GBP/USD, USD/JPY, AUD/USD;
- decision timeframes: 4h regime, 1h setup, 30m trigger;
- first strategy: trend pullback;
- starting equity model: USD 1,000;
- default risk: 0.5% per trade, 1.0% maximum combined open risk;
- decisions are made from closed candles, not tick/order-book noise.

The project is not an HFT/scalping system. Latency-sensitive microstructure, large replay infrastructure,
ML, multi-service architecture and exchange-specific complexity are explicitly out of MVP scope.

## Design rule

Live trading and backtesting must call the same strategy and risk code. Only the data/execution adapters differ.

```text
Market adapter
     |
Normalized candles
     |
Market/strategy core
     |
Opportunity selector
     |
Risk + cost gate
     |
Execution adapter
```

The project must prove positive expectancy after realistic costs before it grows.

## Bootstrap

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev]"
pytest
ruff check .
python -m medium_trading.main
```

## Status

Repository bootstrap only. The included trend-pullback implementation is a deterministic baseline for
development and testing, not a claim of validated trading edge.

See [agent.md](agent.md) before making changes.
