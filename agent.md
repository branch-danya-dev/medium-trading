# Agent instructions

These instructions are mandatory for every agent working in this repository.

## Mission

Build a small, understandable medium-frequency systematic trading bot. The first goal is to prove or
disprove a net trading edge after realistic costs. Engineering complexity is secondary.

## Fixed MVP scope

- Market: spot FX.
- Symbols: EUR/USD, GBP/USD, USD/JPY, AUD/USD.
- Timeframes: 4h regime, 1h setup, 30m trigger.
- First strategy: Trend Pullback.
- Starting equity model: USD 1,000.
- Default risk: 0.5% of equity per trade.
- Maximum combined open risk: 1.0% of equity.
- Strategy decisions use closed candles.
- One broker integration only when live execution is implemented.

Trend Pullback is a baseline candidate, not a proven edge. Breakout is the next strategy candidate only
after Trend Pullback is evaluated.

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
- dozens of indicators or parameter grids.

SQLite is acceptable later if persistence is needed. Flat CSV/Parquet is acceptable for historical data.

## Trading correctness

Every strategy change must answer:

- What market regime is it intended for?
- What invalidates the trade?
- What is the expected holding horizon?
- What transaction costs are assumed?
- Does the result survive realistic spread, commission and slippage?
- Does it survive a 2x cost stress test before being considered robust?

Never claim profitability from an in-sample backtest.

## Development workflow

Before changing code:

1. Read README.md and this file.
2. Inspect existing tests and the relevant implementation.
3. Keep the change as small as possible.

Before finishing:

1. Run `ruff check .`.
2. Run `pytest`.
3. Add or update tests for changed behavior.
4. Document any changed trading assumption.
5. Report what is implemented, what was tested and what remains unproven.

Do not silently relax risk limits or cost assumptions to make a backtest look better.

## Project boundaries

The project should remain understandable from the repository tree. If a feature requires many new
subsystems, first look for a simpler design. The default answer to speculative infrastructure is "no".

Agents may refactor when it removes duplication or clarifies ownership, but should not redesign the
project without evidence that the current architecture blocks a concrete requirement.
