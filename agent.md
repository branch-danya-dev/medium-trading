# Agent instructions

These instructions are mandatory for every agent working in this repository.

## Mission

Build a small, understandable medium-frequency systematic trading bot. The first goal is to prove or
disprove a net trading edge after realistic costs. Engineering complexity is secondary.

## Fixed MVP scope

- Market: spot FX.
- Symbols: EUR/USD, GBP/USD, USD/JPY, AUD/USD.
- Core data timeframes: 4h, 1h, 30m.
- Current strategy candidate: 4H Time-Series Momentum.
- Starting equity model: USD 1,000.
- Default risk: 0.5% of equity per trade.
- Maximum combined open risk: 1.0% of equity.
- Strategy decisions use closed candles.
- Automatic Dukascopy minute-candle download is the primary historical-data path.
- Manual Dukascopy CSV import is a fallback.
- OANDA v20 remains optional and must not become an MVP dependency.
- One broker integration only when live execution is implemented.

Trend Pullback has been evaluated and is REJECTED: it was negative before costs on out-of-sample data across all four MVP pairs. Do not tune or revive it without an explicit new research task.

Volatility Breakout has been evaluated and is REJECTED: it was negative before costs on out-of-sample data across all four MVP pairs. Do not tune or revive it without an explicit new research task.

4H Time-Series Momentum is the current fixed baseline candidate. Parameters are locked before its first evaluation: 30-bar 4H momentum lookback, signal only when a new 4H candle closes, ATR14 volatility estimate, 2 ATR stop, 3R target, and 240 M30 bars (5 days) maximum holding. It does not use 1H or 30m entry triggers. Do not tune these parameters on out-of-sample results.

## Architecture rules

1. Keep one shared strategy core for backtest, paper and live modes.
2. Broker/data adapters may translate external formats, but must not contain strategy logic.
3. Strategy code must not place orders directly.
4. Risk and transaction-cost checks happen before execution.
5. Position management must be deterministic and testable.
6. Prefer plain Python and small modules over frameworks and services.
7. Add abstraction only when a second real implementation needs it.

## Explicit non-goals

Do not add these without a concrete, measured need and a task that explicitly requires them:

- HFT or scalping logic;
- order-book L2/L3 pipelines;
- latency optimization measured in milliseconds/microseconds;
- Kafka, Redis, Celery or distributed services;
- large recorder/replay platforms;
- a separate research runtime with duplicated strategy logic;
- ML/LLM trading decisions;
- microservices;
- a plugin framework;
- multiple databases;
- dozens of indicators or parameter grids;
- AWS/S3 infrastructure only to obtain historical data.

SQLite is acceptable later if persistence is needed. Flat CSV/Parquet is acceptable for historical data.

## Historical-data correctness

- Historical downloader failures must be visible; never silently continue with failed requested days.
- Do not synthesize missing minute candles.
- Aggregate M30 only from complete 30-minute source buckets.
- Use UTC throughout historical-data processing.
- Keep generated data files out of Git.
- Do not increase Dukascopy downloader concurrency aggressively without measuring rate limits.
- BID is the default research price stream while spread/slippage remains modeled separately.

## Trading correctness

Every strategy change must answer:

- What market regime is it intended for?
- What invalidates the trade?
- What is the expected holding horizon?
- What transaction costs are assumed?
- Does the result survive realistic spread, commission and slippage?
- Does it survive a 2x cost stress test before being considered robust?

Validation rules are mandatory:

- keep training, validation and out-of-sample periods chronological;
- never tune strategy parameters on the out-of-sample segment;
- preserve enough pre-period candles only as indicator warmup;
- report ordinary and 2x-cost out-of-sample results;
- do not call a strategy profitable from one pair or one in-sample period.

Never claim profitability from an in-sample backtest.

## Development workflow

Before changing code:

1. Read README.md and this file.
2. Inspect existing tests and the relevant implementation.
3. Keep the change as small as possible.

Before finishing:

1. Run ruff check .
2. Run pytest.
3. Add or update tests for changed behavior.
4. Document any changed trading assumption.
5. Report what is implemented, what was tested and what remains unproven.

Do not silently relax risk limits or cost assumptions to make a backtest look better.

## Project boundaries

The project should remain understandable from the repository tree. If a feature requires many new
subsystems, first look for a simpler design. The default answer to speculative infrastructure is "no".

Agents may refactor when it removes duplication or clarifies ownership, but should not redesign the
project without evidence that the current architecture blocks a concrete requirement.
