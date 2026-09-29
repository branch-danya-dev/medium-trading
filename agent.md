# Agent instructions

These instructions are mandatory for every agent working in this repository.

## Mission

Build a small, understandable medium-frequency systematic trading bot. The first goal is to prove or
disprove a net trading edge after realistic costs. Engineering complexity is secondary.

## Fixed MVP scope

- Research scope: multiple liquid markets, evaluated one strategy/market hypothesis at a time.
- Legacy FX universe: EUR/USD, GBP/USD, USD/JPY, AUD/USD plus four external FX pairs used in validation.
- Current market: Crypto, starting with BTC/USD.
- Core data timeframe for the current candidate: 30m.
- Current research candidate: frozen BTC/USD Trend LONG v1.1 stop-width diagnostic.
- Default project starting equity model: USD 1,000. The BTC Trend LONG v1 diagnostic baseline uses USD 10,000 so R-to-USD interpretation is explicit.
- Default risk: 0.5% of equity per trade.
- Maximum combined open risk: 1.0% of equity.
- Strategy decisions use closed candles.
- Automatic Dukascopy minute-candle download is the primary historical-data path.
- Manual Dukascopy CSV import is a fallback.
- OANDA v20 remains optional and must not become an MVP dependency.
- One broker integration only when live execution is implemented.

Trend Pullback has been evaluated and is REJECTED: it was negative before costs on out-of-sample data across all four MVP pairs. Do not tune or revive it without an explicit new research task.

Volatility Breakout has been evaluated and is REJECTED: it was negative before costs on out-of-sample data across all four MVP pairs. Do not tune or revive it without an explicit new research task.

4H Time-Series Momentum has been evaluated and is REJECTED: it was negative before costs on out-of-sample data across all four MVP pairs. Do not tune or revive it without an explicit new research task.

4H Mean Reversion / Range Trading has been evaluated and is REJECTED as a universal FX strategy. Its first four-pair OOS was positive overall, but the external four-pair holdout was negative overall and the frozen 2026 temporal forward test was negative overall across eight pairs. Its parameters remain locked for reproducibility.

The first frozen ML filter over Mean Reversion entries was useful diagnostically but did not make the 2026 forward period profitable: it reduced the loss materially while remaining negative. Do not tune that filter further on the inspected periods.

A prior research task was a frozen direct 4H ML opportunity model. It evaluates every completed 4H state rather than requiring a Mean Reversion setup. Two independent HistGradientBoostingRegressor models predict realized net R for a fixed LONG candidate and a fixed SHORT candidate. The higher prediction is traded only when it is above 0.0R; otherwise the decision is NO TRADE.

Direct-ML execution is fixed before the first historical evaluation: entry at the next M30 open, stop at 1.5 ATR14, target at 2.0 ATR14, maximum holding of 48 M30 bars (24 hours), the existing 8x cost gate, conservative same-bar stop priority, and the same modeled FX transaction costs.

Direct-ML model parameters are fixed: learning_rate=0.05, max_iter=120, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=42. The fixed features are 1/3/6/12/30-bar 4H returns in ATR, 20/50-bar z-scores, 10/30-bar efficiency ratios, ATR/price, ATR14/ATR50, 10/30-bar ranges in ATR, current 4H candle body and range in ATR, and position inside the 20-bar range. Symbol identity is deliberately excluded.

Historical direct-ML evaluation is expanding-window walk-forward with calendar-year test folds 2023, 2024 and 2025. A training opportunity is usable only if both hypothetical LONG and SHORT outcomes were fully known before the test year begins.

The first Direct ML walk-forward failed, and its final pre-registered best-opportunity kill-gate was also rejected. Direct ML is REJECTED. Do not tune or revive it on the inspected FX datasets without an explicit new research task.

The project now evaluates market-specific strategies with daily-income consistency as a first-class gate. Do not optimize for a requested percentage return per day. Measure R/day first; translate to percentage returns only after a strategy survives costs and stability gates.

The first frozen US index Opening Range Breakout baseline was evaluated on USA500.IDX/USD and USATECH.IDX/USD over 2023-2025. USA500 is REJECTED and receives no second attempt. USATECH was positive overall and positive in 2 of 3 years, but failed the Daily Income Gate because combined net PF, 2x-cost PF and positive-month rate were below threshold.

Exactly one final USATECH modification is allowed. It keeps the same opening range, entry, stop, target, holding period, risk, cost model and session window, and changes only breakout quality:
- opening range: the 09:30-10:00 America/New_York M30 candle;
- identify the first later M30 close outside that range, with breakout decisions ending at 13:00 New York time;
- that first breakout close must extend at least 10% of the opening-range width beyond the relevant boundary;
- if the first breakout close is weaker than 10%, skip the entire session; do not give a second chance later that day;
- entry: next M30 open;
- stop: opposite side of the opening range;
- target: 1.5R;
- maximum holding: 6 M30 bars (3 hours);
- maximum one breakout attempt per session;
- default risk remains 0.5% per trade;
- modeled round-trip research cost remains frozen at 3.0 index price points for USATECH.IDX/USD;
- fixed evaluation years remain 2023, 2024 and 2025.

The final USATECH quality-filter variant failed the Daily Income Gate. US-index ORB is REJECTED and CLOSED. Do not tune or revive its threshold, target, stop, holding period or costs on the inspected datasets.

The first frozen Gold / XAU/USD London-reference to New York breakout baseline failed the Daily Income Gate and is REJECTED. It is not being tuned further.

The frozen Gold / XAU/USD New York momentum-continuation baseline failed the Daily Income Gate and is REJECTED. It produced only 54 trades across 773 eligible sessions, so the setup was far too rare for the daily-income objective; the 2x-cost result was also negative. Do not loosen its filters after observing this result.

The frozen Gold / XAU/USD New York opening exhaustion / reversal baseline also failed the Daily Income Gate. It produced only 104 trades across 773 eligible sessions, negative net R, net PF below 1, negative 2x-cost results, and zero positive yearly folds. Gold is CLOSED for the current research cycle; do not tune the three inspected gold hypotheses on 2023-2025 data.

The first frozen BTC/USD UTC Daily Volatility Expansion baseline failed the Daily Income Gate and is REJECTED. It produced 665 trades but negative net R and negative 2x-cost results; only one yearly fold was positive. Do not tune its fixed 00:00-04:00 UTC range after observing this result.

The uploaded BTC/USD history also exposed material historical-data fragmentation, especially in 2023. For crypto daily-income evaluation, an eligible UTC day must now contain at least 40 of the possible 48 M30 bars. Days below that coverage threshold are excluded from the session denominator; missing bars are never synthesized. Strategy signals additionally require contiguous recent M30 history when their logic depends on rolling momentum.

The frozen BTC/USD rolling intraday momentum-continuation baseline also failed. It produced 333 trades across 867 quality-filtered UTC sessions, negative gross/net performance overall, and only 2024 was positive. It is REJECTED. Do not tune that mixed LONG/SHORT hypothesis further on the inspected 2023-2025 data.

BTC/USD Trend LONG v1 was evaluated as a diagnostic baseline:
- 1,092 LONG trades;
- gross +47.13R with gross PF 1.067, so the raw direction/setup stream was weakly positive before costs;
- modeled costs consumed 284.56R, producing net -237.42R and a roughly 71% drawdown;
- 695 trades exited by stop/stop-gap;
- 425 stopped trades subsequently reached at least +1R from the original entry inside the 24-hour diagnostic horizon;
- many trades had very small structural stop distances, making the fixed USD 50 modeled cost extremely large in R terms.

The evidence does not justify ML yet. The immediate failure mode is execution/stop width: many LONG candidates later move strongly upward after being stopped, while narrow pullback stops inflate transaction-cost drag measured in R.

The current frozen modification is BTC/USD Trend LONG v1.1. It changes only stop width; trend regime, entry trigger, target, holding period, risk, costs and ML/noise-filter status remain unchanged:
- instrument: BTC/USD only;
- side: LONG only;
- ML: OFF;
- noise filter: OFF;
- 4H trend regime: latest completed 4H close above EMA20, and EMA20 above its value three completed 4H bars earlier;
- M30 setup: one bearish pullback candle followed immediately by a bullish M30 candle closing above the pullback high;
- entry: next M30 open;
- structural stop anchor: pullback low;
- minimum stop distance: 1.0 x M30 ATR14 measured from the actual next-open entry;
- resolved stop for LONG = lower of pullback low or actual entry minus ATR14;
- target: 2.0R from the resolved stop distance;
- maximum holding: 48 M30 bars / nominally 24 hours, unchanged from v1 for comparability;
- maximum one open BTC position at a time via the shared backtest engine;
- starting equity: USD 10,000;
- risk: 0.5% per trade;
- modeled BTC/USD round-trip cost: USD 50 price points;
- old 8x expected-move cost gate remains disabled at minimum_cost_multiple=0.01 so micro/noise candidates remain visible and costs are still deducted from every trade;
- fixed historical trade window remains 2023-01-01 through 2025-12-31.

BTC Trend LONG v1.1 diagnostics must include MFE and MAE up to exit, 24-hour MFE/MAE, post-stop favorable movement, time from stop to later +1R/+2R when applicable, ATR14 at entry, stop distance in ATR and cost in R.

Do not use the legacy Daily Income Gate for this v1/v1.1 diagnostic sequence. Compare v1.1 directly with v1 to answer one question: whether widening only the stop removes a material share of false stop-outs and reduces cost drag in R without destroying the weak positive gross edge. Do not change EMA period, EMA slope lookback, entry rule, target, holding period, cost, risk, or add ML before reading v1.1.

Daily Income Gate is frozen before the first result. A strategy passes only if all conditions hold across the combined fixed test years:
- at least 250 eligible session days;
- at least 300 trades;
- active on at least 70% of eligible sessions;
- profitable on at least 45% of active days;
- average net R per eligible session > 0;
- net profit factor >= 1.15;
- 2x-cost profit factor > 1.00;
- positive net R in at least 2 yearly folds;
- at least 60% of calendar months positive;
- worst day >= -2.0R;
- longest losing-day streak <= 8 sessions;
- best day contributes less than 15% of total positive daily R.

For each new market strategy: one frozen baseline plus at most one materially justified modification. If it still fails the applicable gate, mark it REJECTED and move to the next market hypothesis.

## Architecture rules

1. Keep one shared strategy core for backtest, paper and live modes.
2. Broker/data adapters may translate external formats, but must not contain strategy logic.
3. Strategy code must not place orders directly.
4. Risk and transaction-cost checks happen before execution.
5. Position management must be deterministic and testable.
6. A strategy may emit an explicit target when the hypothesis requires a structural exit; otherwise the backtest may use the configured R-multiple target.
7. Prefer plain Python and small modules over frameworks and services.
8. Add abstraction only when a second real implementation needs it.

## Explicit non-goals

Do not add these without a concrete, measured need and a task that explicitly requires them:

- HFT or scalping logic;
- order-book L2/L3 pipelines;
- latency optimization measured in milliseconds/microseconds;
- Kafka, Redis, Celery or distributed services;
- large recorder/replay platforms;
- a separate research runtime with duplicated strategy logic;
- ML/LLM trading decisions outside the explicitly frozen ML research task;
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
- for daily-income strategies, report eligible session days, active-day rate, R/day, profitable-day rate, losing-day streaks and monthly consistency;
- never tune strategy parameters on the out-of-sample segment;
- preserve enough pre-period candles only as indicator warmup;
- report ordinary and 2x-cost out-of-sample results;
- for temporal forward tests, use explicit trade start/end dates and keep warmup/exit-horizon data outside the entry window;
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
