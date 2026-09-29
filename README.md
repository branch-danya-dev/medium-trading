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

The project must prove positive expectancy after realistic costs before it grows.

## Bootstrap

    python -m venv .venv

Windows:

    .venv\Scripts\activate

Linux/macOS:

    source .venv/bin/activate

Then:

    python -m pip install -U pip
    pip install -e ".[dev]"
    pytest
    ruff check .

## Automatic Dukascopy history download

The primary historical-data path requires no broker account, API token or .env file.

To download the full MVP universe for an inclusive UTC range:

    medium-trading download-dukascopy ^
      --from 2020-01-01 ^
      --to 2026-01-01

In PowerShell use backticks instead of carets for line continuation, or run it on one line.

With no --symbol arguments the command downloads:

- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD

and writes:

    data/EUR_USD_M30.csv
    data/GBP_USD_M30.csv
    data/USD_JPY_M30.csv
    data/AUD_USD_M30.csv

To download explicit pairs:

    medium-trading download-dukascopy --symbol EUR/USD --symbol GBP/USD --from 2024-01-01 --to 2026-01-01 --workers 4 --output-dir data

The downloader:

- requests Dukascopy minute BID candles by UTC calendar date;
- downloads up to four days concurrently by default;
- retries transient HTTP errors;
- skips Saturdays but keeps Sundays because FX can reopen late Sunday UTC;
- converts one-minute source candles into complete M30 bars;
- refuses to silently save a dataset when a requested day still fails after retries;
- prints progress every 100 completed dates.

BID is the default because current backtests model spread/slippage separately. ASK can be requested with
--side ASK.

## Manual Dukascopy import fallback

Browser exports remain supported:

    medium-trading import-dukascopy --symbol EUR/USD --input downloads/EURUSD.csv --output data/EUR_USD_M30.csv

OANDA v20 remains optional and is not required for the MVP.

## Backtest

    medium-trading backtest --symbol EUR/USD --data data/EUR_USD_M30.csv --round-trip-cost-pips 1.2

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

    medium-trading evaluate ^
      --dataset EUR/USD=data/EUR_USD_M30.csv ^
      --dataset GBP/USD=data/GBP_USD_M30.csv ^
      --dataset USD/JPY=data/USD_JPY_M30.csv ^
      --dataset AUD/USD=data/AUD_USD_M30.csv ^
      --cost EUR/USD=1.0 ^
      --cost GBP/USD=1.2 ^
      --cost USD/JPY=1.0 ^
      --cost AUD/USD=1.2 ^
      --json artifacts/trend_pullback_eval.json

Validation and out-of-sample segments receive historical candles before their start only as indicator
warmup. No trade may start in that warmup. The out-of-sample segment must not be used for parameter
tuning.

## Status

Automatic Dukascopy date-range download, manual CSV import, and historical multi-pair evaluation are
implemented. Live/paper execution is intentionally not implemented until the strategy survives realistic
historical costs and out-of-sample validation.

See agent.md before making changes.
