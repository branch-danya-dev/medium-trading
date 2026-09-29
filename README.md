# medium-trading

Compact medium-frequency systematic trading bot.

## MVP thesis

The first version is intentionally small:

- market: spot FX;
- symbols: EUR/USD, GBP/USD, USD/JPY, AUD/USD;
- core data timeframes: 4h, 1h, 30m;
- current research candidate: frozen ML filter over 4h mean-reversion entries;
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

## Strategy research

Trend Pullback has been rejected by the historical gate: it was negative before costs on out-of-sample
data across all four MVP FX pairs. It remains available only for reproducibility.

Volatility Breakout has also been rejected by the historical gate: gross out-of-sample results were
negative across all four MVP FX pairs.

4H Time-Series Momentum has also been rejected by the historical gate: gross out-of-sample results were
negative across all four MVP FX pairs.

4H Mean Reversion / Range Trading has also been rejected as a universal FX strategy after a mixed
cross-instrument holdout and a negative frozen 2026 temporal forward test across eight pairs. The rule
set remains available as a reproducible setup generator for the next research step.

The current research candidate is a frozen ML filter over the existing Mean Reversion trade stream.
It predicts realized net R for an already-valid Mean Reversion trade and accepts it only when predicted
net R is above 0.0R. Symbol identity is not a feature.

The first ML baseline is fixed before evaluation:

- model: HistGradientBoostingRegressor;
- learning rate 0.05, 100 iterations, 7 max leaf nodes, minimum 20 samples per leaf;
- L2 regularization 1.0, early stopping disabled, random state 42;
- features: side, absolute z-score, efficiency ratio, ATR/price, target distance in ATR, reward/risk,
  modeled cost in R, 1/3/6-bar returns in ATR, 10-bar range in ATR, and current 4H body in ATR;
- expanding-window walk-forward test years: 2023, 2024 and 2025;
- training labels must be fully known before each test year begins;
- ordinary costs and a same-trade 2x-cost stress result are reported.

These historical folds are research evidence, not a pristine final holdout. The already-inspected 2026
period cannot be reused as final proof for the ML model.

## Backtest

    medium-trading backtest --strategy mean-reversion --symbol EUR/USD --data data/EUR_USD_M30.csv --round-trip-cost-pips 1.2

Current simulation assumptions are deliberately conservative:

- strategy sees closed candles only;
- entry happens at the next M30 open;
- target and maximum holding time use strategy-specific defaults unless explicitly overridden;
- strategies may emit an explicit structural target; otherwise the configured R-multiple target is used;
- current mean-reversion baseline targets the prior 20-bar mean and uses 192 M30 bars (4 days);
- stop is the structural stop emitted by the strategy;
- if stop and target are touched inside the same candle, stop is assumed first;
- round-trip trading costs are deducted from every trade;
- a setup is rejected if expected target movement is less than 8x modeled costs.

## Multi-pair chronological evaluation

The evaluation command is the default strategy gate. It runs each pair through a chronological
60% train / 20% validation / 20% out-of-sample split and repeats the out-of-sample run at 2x costs.

    medium-trading evaluate ^
      --strategy mean-reversion ^
      --dataset EUR/USD=data/EUR_USD_M30.csv ^
      --dataset GBP/USD=data/GBP_USD_M30.csv ^
      --dataset USD/JPY=data/USD_JPY_M30.csv ^
      --dataset AUD/USD=data/AUD_USD_M30.csv ^
      --cost EUR/USD=1.0 ^
      --cost GBP/USD=1.2 ^
      --cost USD/JPY=1.0 ^
      --cost AUD/USD=1.2 ^
      --json artifacts/mean_reversion_eval.json

Validation and out-of-sample segments receive historical candles before their start only as indicator
warmup. No trade may start in that warmup. The out-of-sample segment must not be used for parameter
tuning.

## Temporal forward holdout

After the initial and cross-instrument Mean Reversion evaluations, the strategy parameters remain frozen.
The next gate uses fresh 2026 data across all eight researched FX pairs.

Download a dedicated dataset with December 2025 warmup and enough post-window data for the four-day
maximum holding horizon:

    medium-trading download-dukascopy ^
      --symbol EUR/USD --symbol GBP/USD --symbol USD/JPY --symbol AUD/USD ^
      --symbol USD/CAD --symbol NZD/USD --symbol EUR/GBP --symbol EUR/JPY ^
      --from 2025-12-01 ^
      --to 2026-09-28 ^
      --workers 4 ^
      --output-dir data/forward_2026

Only entries from 2026-01-02 through 2026-09-18 are counted. Candles before the start are warmup only;
candles after the end exist only so open trades can finish normally.

    medium-trading forward-evaluate ^
      --strategy mean-reversion ^
      --trade-start 2026-01-02 ^
      --trade-end 2026-09-18 ^
      --dataset EUR/USD=data/forward_2026/EUR_USD_M30.csv ^
      --dataset GBP/USD=data/forward_2026/GBP_USD_M30.csv ^
      --dataset USD/JPY=data/forward_2026/USD_JPY_M30.csv ^
      --dataset AUD/USD=data/forward_2026/AUD_USD_M30.csv ^
      --dataset USD/CAD=data/forward_2026/USD_CAD_M30.csv ^
      --dataset NZD/USD=data/forward_2026/NZD_USD_M30.csv ^
      --dataset EUR/GBP=data/forward_2026/EUR_GBP_M30.csv ^
      --dataset EUR/JPY=data/forward_2026/EUR_JPY_M30.csv ^
      --cost EUR/USD=1.0 --cost GBP/USD=1.2 ^
      --cost USD/JPY=1.0 --cost AUD/USD=1.2 ^
      --cost USD/CAD=1.5 --cost NZD/USD=1.5 ^
      --cost EUR/GBP=1.5 --cost EUR/JPY=1.5 ^
      --json artifacts/mean_reversion_forward_2026.json

The command reports the frozen forward window at ordinary modeled costs and at 2x costs without a new
train/validation/OOS split.

## ML filter research

Install the optional ML research dependency:

    pip install -e ".[ml]"

The first frozen ML test pools the eight historical FX datasets but does not include symbol identity as
a feature. It trains only on trades whose outcomes were already known before each calendar-year test
fold and compares the filtered subset against the unfiltered Mean Reversion baseline.

    medium-trading ml-evaluate ^
      --dataset EUR/USD=data/EUR_USD_M30.csv ^
      --dataset GBP/USD=data/GBP_USD_M30.csv ^
      --dataset USD/JPY=data/USD_JPY_M30.csv ^
      --dataset AUD/USD=data/AUD_USD_M30.csv ^
      --dataset USD/CAD=data/USD_CAD_M30.csv ^
      --dataset NZD/USD=data/NZD_USD_M30.csv ^
      --dataset EUR/GBP=data/EUR_GBP_M30.csv ^
      --dataset EUR/JPY=data/EUR_JPY_M30.csv ^
      --cost EUR/USD=1.0 --cost GBP/USD=1.2 ^
      --cost USD/JPY=1.0 --cost AUD/USD=1.2 ^
      --cost USD/CAD=1.5 --cost NZD/USD=1.5 ^
      --cost EUR/GBP=1.5 --cost EUR/JPY=1.5 ^
      --json artifacts/mean_reversion_ml_walk_forward.json

Do not tune the feature list, model parameters, threshold, or pair selection after reading this first
result. A positive historical walk-forward would justify a later paper-forward test, not a profitability
claim.

## Status

Automatic Dukascopy date-range download, manual CSV import, and historical multi-pair evaluation are
implemented. Live/paper execution is intentionally not implemented until the strategy survives realistic
historical costs and out-of-sample validation.

See agent.md before making changes.
