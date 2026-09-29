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

Live trading and backtesting must call the same strategy and risk code. Only the data/execution adapters
differ.

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
```

## Historical data

The first data adapter uses OANDA v20 only as a source of normalized M30 historical candles. Credentials
are read from environment variables and are never stored in the repository.

```bash
medium-trading download-oanda \
  --instrument EUR_USD \
  --from 2020-01-01 \
  --to 2026-01-01 \
  --output data/EUR_USD_M30.csv
```

The downloader fetches M30 history in bounded chunks. The backtest derives 1h and 4h bars from the same
M30 source, so higher-timeframe context cannot see unfinished future candles.

## Backtest

```bash
medium-trading backtest \
  --symbol EUR/USD \
  --data data/EUR_USD_M30.csv \
  --round-trip-cost-pips 1.2
```

Current simulation assumptions are deliberately conservative:

- strategy sees closed candles only;
- entry happens at the next M30 open;
- target is 2R by default;
- stop is the structural stop emitted by the strategy;
- maximum holding time is 48 M30 bars;
- if stop and target are touched inside the same candle, stop is assumed first;
- round-trip trading costs are deducted from every trade;
- a setup is rejected if expected target movement is less than 8x modeled costs.

## Multi-pair chronological evaluation

The evaluation command is the default strategy gate. It runs each pair through a chronological
60% train / 20% validation / 20% out-of-sample split and repeats the out-of-sample run at 2x costs.

```bash
medium-trading evaluate \
  --dataset EUR/USD=data/EUR_USD_M30.csv \
  --dataset GBP/USD=data/GBP_USD_M30.csv \
  --dataset USD/JPY=data/USD_JPY_M30.csv \
  --dataset AUD/USD=data/AUD_USD_M30.csv \
  --cost EUR/USD=1.0 \
  --cost GBP/USD=1.2 \
  --cost USD/JPY=1.0 \
  --cost AUD/USD=1.2 \
  --json artifacts/trend_pullback_eval.json
```

Validation and out-of-sample segments receive historical candles before their start only as indicator
warmup. No trade may start in that warmup. The out-of-sample segment must not be used for parameter
tuning.

The multi-pair report is a strategy evaluation, not yet a capital-constrained portfolio simulation.

## Status

Repository bootstrap plus historical multi-pair evaluation path. Live/paper execution is intentionally
not implemented until the strategy survives realistic historical costs and out-of-sample validation.

See [agent.md](agent.md) before making changes.
