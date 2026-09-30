# medium-trading

Compact medium-frequency systematic trading bot.

## MVP thesis

The first version is intentionally small:

- research markets are tested separately rather than forcing one universal strategy;
- legacy FX research remains reproducible;
- current market: Crypto, starting with BTC/USD;
- current candidate: Bybit BTCUSDT Market Structure Event Model v0.3;
- daily-income consistency is now a primary evaluation target;
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
- skips Saturdays for non-crypto symbols but keeps all seven UTC days for BTC/USD and ETH/USD;
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

The already-inspected 2026 Mean Reversion period can be used only as a frozen diagnostic check of this
exact ML filter. Train on the historical files, then apply the unchanged model to the separate 2026
forward files:

    medium-trading ml-forward-evaluate ^
      --train-dataset EUR/USD=data/EUR_USD_M30.csv ^
      --train-dataset GBP/USD=data/GBP_USD_M30.csv ^
      --train-dataset USD/JPY=data/USD_JPY_M30.csv ^
      --train-dataset AUD/USD=data/AUD_USD_M30.csv ^
      --train-dataset USD/CAD=data/USD_CAD_M30.csv ^
      --train-dataset NZD/USD=data/NZD_USD_M30.csv ^
      --train-dataset EUR/GBP=data/EUR_GBP_M30.csv ^
      --train-dataset EUR/JPY=data/EUR_JPY_M30.csv ^
      --forward-dataset EUR/USD=data/forward_2026/EUR_USD_M30.csv ^
      --forward-dataset GBP/USD=data/forward_2026/GBP_USD_M30.csv ^
      --forward-dataset USD/JPY=data/forward_2026/USD_JPY_M30.csv ^
      --forward-dataset AUD/USD=data/forward_2026/AUD_USD_M30.csv ^
      --forward-dataset USD/CAD=data/forward_2026/USD_CAD_M30.csv ^
      --forward-dataset NZD/USD=data/forward_2026/NZD_USD_M30.csv ^
      --forward-dataset EUR/GBP=data/forward_2026/EUR_GBP_M30.csv ^
      --forward-dataset EUR/JPY=data/forward_2026/EUR_JPY_M30.csv ^
      --trade-start 2026-01-02 ^
      --trade-end 2026-09-18 ^
      --cost EUR/USD=1.0 --cost GBP/USD=1.2 ^
      --cost USD/JPY=1.0 --cost AUD/USD=1.2 ^
      --cost USD/CAD=1.5 --cost NZD/USD=1.5 ^
      --cost EUR/GBP=1.5 --cost EUR/JPY=1.5 ^
      --json artifacts/mean_reversion_ml_forward_2026.json

This diagnostic trains only on labels known before the 2026 window and reports the same selected trade
set at ordinary and 2x modeled costs. It is not a pristine final holdout because the underlying 2026
Mean Reversion results have already been inspected.

## Direct ML opportunity research

The next frozen baseline removes Mean Reversion as a mandatory setup generator. Every completed 4H
state becomes a candidate. Two separate models estimate the net-R outcome of a fixed LONG candidate and
a fixed SHORT candidate; the higher prediction is used only when it is above 0.0R, otherwise the model
returns NO TRADE.

The execution template is fixed before the first run:

- entry at the next M30 open;
- stop at 1.5 x ATR14;
- target at 2.0 x ATR14;
- maximum holding time 48 M30 bars (24 hours);
- existing 8x transaction-cost gate;
- conservative same-bar stop priority.

The direct model uses no symbol identity. Fixed features are 1/3/6/12/30-bar returns in ATR, 20/50-bar
z-scores, 10/30-bar efficiency ratios, ATR/price, ATR14/ATR50, 10/30-bar ranges in ATR, current 4H body
and range in ATR, and position inside the 20-bar range.

Run the first frozen walk-forward study on the eight 2020-2025 datasets:

    medium-trading direct-ml-evaluate ^
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
      --json artifacts/direct_ml_walk_forward.json

The test folds remain 2023, 2024 and 2025. Training uses only candidate outcomes that were fully known
before each test year.

The broad Direct ML baseline failed. One final, pre-registered modification is allowed: at every 4H
decision timestamp, rank all eight pairs and both directions with the unchanged models, open only the
single highest predicted opportunity above 0.0R, and allow at most one open position globally. No model,
feature, threshold, execution, cost or pair-universe parameter changes are allowed.

Run the final gate:

    medium-trading direct-ml-final-evaluate ^
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
      --json artifacts/direct_ml_final_gate.json

The final kill-gate is strict and conjunctive: at least 500 selected trades, combined gross PF > 1.00,
combined net PF >= 1.10, combined 2x-cost PF > 1.00, and positive net R in at least 2 of the 3 test
years. If any condition fails, Direct ML is rejected and is not tuned again on these datasets.

## Daily-income market research

Direct ML failed its final pre-registered kill-gate and is rejected. The project now tests separate
strategies for separate markets, with daily consistency measured directly instead of trying to force one
universal model.

The first candidate is a 30-minute Opening Range Breakout for the US 500 and US 100 Tech index CFDs:

- opening range: 09:30-10:00 America/New_York;
- first later M30 close outside the range, only through 13:00 New York time;
- entry at the next M30 open;
- stop at the opposite side of the opening range;
- target at 1.5R;
- maximum holding 6 M30 bars (3 hours);
- maximum one breakout attempt per session;
- 0.5% risk per trade.

Initial research costs are frozen at 1.0 index price point round trip for USA500.IDX/USD and 3.0 index
price points for USATECH.IDX/USD. Index costs are modeled in whole index price points, not FX pips.

Download a fresh index dataset before evaluating the strategy:

    medium-trading download-dukascopy ^
      --symbol USA500.IDX/USD ^
      --symbol USATECH.IDX/USD ^
      --from 2020-01-01 ^
      --to 2025-12-31 ^
      --workers 4 ^
      --output-dir data/index

Then run the fixed 2023-2025 daily-income gate:

    medium-trading daily-evaluate ^
      --strategy opening-range-breakout ^
      --dataset USA500.IDX/USD=data/index/USA500.IDX_USD_M30.csv ^
      --dataset USATECH.IDX/USD=data/index/USATECH.IDX_USD_M30.csv ^
      --cost USA500.IDX/USD=1.0 ^
      --cost USATECH.IDX/USD=3.0 ^
      --json artifacts/index_orb_daily_eval.json

The baseline rejected USA500. USATECH was positive overall and positive in two yearly folds, but missed
the fixed net-PF, 2x-cost-PF and positive-month-rate thresholds. One final modification is therefore
allowed for USATECH only.

The final variant keeps the same stop, target, holding period, cost and session window, but requires the
first M30 breakout close to extend at least 10% of the opening-range width beyond the relevant boundary.
If that first breakout is weaker, the entire session is skipped; there is no second chance later that
day.

Run the final USATECH ORB test:

    medium-trading daily-evaluate ^
      --strategy opening-range-breakout-quality ^
      --dataset USATECH.IDX/USD=data/index/USATECH.IDX_USD_M30.csv ^
      --cost USATECH.IDX/USD=3.0 ^
      --json artifacts/usatech_orb_quality_daily_eval.json

The final quality-filter variant also failed the gate, so US-index ORB is rejected and closed.

The next market is Gold / XAU/USD. The first frozen gold candidate uses the 08:00-10:30 Europe/London
reference range and trades only the first M30 close outside that range during 08:30-11:00
America/New_York. Entry is the next M30 open, stop is the opposite side of the London range, target is
1.5R, maximum holding is 6 M30 bars, and there is at most one attempt per session. If the complete
five-bar London reference is missing, the day is skipped.

XAU/USD costs use a 0.01 USD pip in the research engine. The ordinary frozen round-trip cost is 80 pips
(0.80 USD price movement); the built-in 2x stress therefore uses 160 pips (1.60 USD).

Download gold history:

    medium-trading download-dukascopy ^
      --symbol XAU/USD ^
      --from 2020-01-01 ^
      --to 2025-12-31 ^
      --workers 4 ^
      --output-dir data/gold

Run the fixed 2023-2025 daily-income baseline:

    medium-trading daily-evaluate ^
      --strategy gold-london-ny-breakout ^
      --dataset XAU/USD=data/gold/XAU_USD_M30.csv ^
      --cost XAU/USD=80 ^
      --json artifacts/gold_london_ny_breakout_daily_eval.json

That London-range breakout baseline failed the gate and is rejected.

The New York momentum-continuation baseline was then tested and rejected. It produced only 54 trades over
773 eligible sessions, making it unsuitable for the daily-income objective, and its 2x-cost result was
negative.

The next gold hypothesis is New York opening exhaustion / reversal. The reference is the three M30 bars
starting at 08:30, 09:00 and 09:30 New York time. The move from the 08:30 open to the 10:00 close must
be at least 0.75 ATR14. After 10:00, only the first counter-direction M30 candle is considered. For an
upward opening impulse it must be bearish and close inside the band from the 50% midpoint to the 10:00
close; for a downward impulse it must be bullish and close inside the mirrored band. If that first
counter-direction candle fails the band condition, the day is skipped. Entry is the next M30 open, stop
is the opening-reference extreme, target is 1.25R, maximum holding is 4 M30 bars (2 hours), and no new
decision is allowed after 12:00 New York time.

Run the frozen 2023-2025 exhaustion-reversal baseline with the existing gold dataset:

    medium-trading daily-evaluate ^
      --strategy gold-ny-exhaustion-reversal ^
      --dataset XAU/USD=data/gold/XAU_USD_M30.csv ^
      --cost XAU/USD=80 ^
      --json artifacts/gold_ny_exhaustion_reversal_daily_eval.json

That exhaustion-reversal baseline also failed. Across the three inspected gold hypotheses, none delivered
the required activity, profitability and cost robustness, so gold is closed for this research cycle.

The next market is crypto, starting with BTC/USD. Daily accounting is by UTC calendar day. The first
frozen hypothesis builds a 00:00-04:00 UTC reference range from eight M30 bars and then takes only the
first qualifying close through 18:00 UTC. A LONG close must extend at least 0.10 ATR14 above the reference
high; a SHORT close must extend at least 0.10 ATR14 below the reference low. Entry is the next M30 open,
stop is the reference midpoint, target is 1.5R, maximum holding is 8 M30 bars (4 hours), and there is at
most one attempt per UTC day.

BTC/USD and ETH/USD use a 1 USD research pip. The frozen BTC/USD ordinary round-trip cost is 50 USD price
points and the 2x stress is 100 USD. The 50 USD value is a conservative research assumption rather than
a broker spread quote.

Download BTC/USD history. Crypto downloads include Saturdays:

    medium-trading download-dukascopy ^
      --symbol BTC/USD ^
      --from 2020-01-01 ^
      --to 2025-12-31 ^
      --workers 4 ^
      --output-dir data/crypto

Run the frozen 2023-2025 BTC daily-income baseline:

    medium-trading daily-evaluate ^
      --strategy crypto-daily-volatility-expansion ^
      --dataset BTC/USD=data/crypto/BTC_USD_M30.csv ^
      --cost BTC/USD=50 ^
      --json artifacts/btc_daily_volatility_expansion_eval.json

That fixed-range volatility-expansion baseline failed and is rejected.

The BTC/USD history is not uniformly complete across the inspected years. Crypto daily evaluation
therefore treats a UTC day as eligible only when it contains at least 40 of the possible 48 M30 bars.
No missing candles are synthesized. Rolling-momentum setups also require exact 30-minute continuity in
their recent input window.

The next frozen BTC hypothesis is rolling intraday momentum continuation, with no fixed early-day
reference range. It looks for an eight-bar (4-hour) directional move of at least 0.75 ATR14, requires at
least five of those eight candle bodies to agree with the direction, then waits for one shallow
counter-direction M30 pullback that does not cross the 50% trend midpoint. The immediately following
M30 candle must resume through the pullback boundary. Decisions are allowed from 02:00 through 22:00 UTC.
Entry is the next M30 open, stop is the pullback extreme, target is 1.5R, maximum holding is 8 M30 bars
(4 hours), and there is at most one attempt per UTC day.

Run the frozen 2023-2025 momentum-continuation baseline:

    medium-trading daily-evaluate ^
      --strategy crypto-intraday-momentum-continuation ^
      --dataset BTC/USD=data/crypto/BTC_USD_M30.csv ^
      --cost BTC/USD=50 ^
      --json artifacts/btc_intraday_momentum_continuation_eval.json

That mixed-direction momentum-continuation baseline failed and is rejected.

Research then narrowed to BTC/USD LONG only. Trend LONG v1 used completed 4H bars for regime context
and M30 for execution: latest 4H close above EMA20, EMA20 rising versus three completed 4H bars earlier,
one bearish M30 pullback, then a bullish M30 close above the pullback high. Entry was the next M30 open,
stop the pullback low, target 2.0R, maximum holding 48 M30 bars, no ML and no noise filter.

The first diagnostic produced a weakly positive raw stream before costs: 1,092 trades, +47.13R gross and
gross PF 1.067. But the pullback-low stop was often very tight: modeled USD 50 costs consumed 284.56R,
net result was -237.42R, and 425 stopped trades later recovered to at least +1R from the original entry
inside the diagnostic horizon. This points to stop/execution width before ML filtering.

BTC Trend LONG v1.1 changes only the stop. The resolved LONG stop is the lower of the pullback low or the
actual next-open entry minus 1.0 x M30 ATR14. Trend regime, pullback/confirmation entry, 2.0R target,
48-bar holding limit, USD 50 modeled cost, 0.5% risk, USD 10,000 starting equity and disabled ML/noise
filter remain unchanged. The old 8x expected-move cost gate remains disabled for diagnostic visibility.

Run v1.1:

    medium-trading btc-long-v1-1-evaluate ^
      --data data/crypto/BTC_USD_M30.csv ^
      --cost-usd 50 ^
      --starting-equity 10000 ^
      --risk 0.005 ^
      --json artifacts/btc_trend_long_v1_1.json

The v1.1 JSON adds MFE/MAE up to exit, post-stop MFE, time from a stop to later +1R/+2R, ATR14 at entry,
stop distance in ATR and cost in R. Compare v1.1 directly with v1 before changing any other parameter or
adding ML.

The v1.1 stop-width run improved the old baseline but still finished deeply negative under the fixed USD 50
BTC cost assumption. It also exposed that fixed price-point costs and bar-count holding are unsuitable for
this dataset. The strategy itself is therefore frozen while execution research is corrected.

The corrected rerun uses a notional bps model. Defaults are 5.5 bps fee per side (a conservative non-VIP
taker/taker perpetual assumption) plus a separate 2.0 bps slippage assumption per side. These inputs are
configurable; they are not fitted to the historical result. Funding is intentionally not modeled yet because
the Dukascopy candle file does not include a venue funding series.

The corrected run also enforces a true 24-hour wall-clock holding limit and rejects gap-contaminated data:
192 recent M30 candles must be consecutive before a decision, and the signal/entry plus the next 48 M30
candles must also be consecutive. Gap rejections are reported separately.

Run the corrected v1.1 research pass:

    medium-trading btc-long-v1-1-corrected-evaluate ^
      --data data/crypto/BTC_USD_M30.csv ^
      --fee-bps-per-side 5.5 ^
      --slippage-bps-per-side 2.0 ^
      --starting-equity 10000 ^
      --risk 0.005 ^
      --json artifacts/btc_trend_long_v1_1_corrected.json

On PowerShell, replace carets with backticks. The output separates fee R, slippage R, total cost R,
holding hours and gap diagnostics. Do not add ML or alter the trading rules until this corrected baseline
has been inspected.

The Dukascopy BTC history proved too fragmented for the next conclusion, so the same frozen strategy is now
rerun on exchange-native Bybit BTCUSDT USDT-perpetual M30 candles. Bybit V5 public kline history does not
require an API key. The downloader is strict: it expects every requested 30-minute slot and aborts instead
of synthesizing or silently accepting a missing candle.

Download the fixed 2023-2025 Bybit history:

    medium-trading download-bybit ^
      --symbol BTCUSDT ^
      --category linear ^
      --from 2023-01-01 ^
      --to 2025-12-31 ^
      --output-dir data/bybit

PowerShell:

    medium-trading download-bybit `
      --symbol BTCUSDT `
      --category linear `
      --from 2023-01-01 `
      --to 2025-12-31 `
      --output-dir data/bybit

The default REST endpoint is https://api.bybit.com/v5/market/kline. If Bybit requires a regional endpoint,
pass the full kline endpoint with --base-url. The saved file is data/bybit/BTCUSDT_M30.csv.

Then rerun the unchanged corrected LONG v1.1:

    medium-trading btc-long-v1-1-corrected-evaluate `
      --data "data/bybit/BTCUSDT_M30.csv" `
      --symbol BTCUSDT `
      --fee-bps-per-side 5.5 `
      --slippage-bps-per-side 2.0 `
      --starting-equity 10000 `
      --risk 0.005 `
      --json artifacts/bybit_btcusdt_trend_long_v1_1.json

This is a data-source validation pass, not a strategy modification. EMA regime, entry, ATR stop floor, 2R
target, risk and ML/noise-filter status remain frozen.

The Bybit-native rerun produced 1,130 trades and only a near-flat gross edge before costs (+9.20R,
gross PF 1.013). That is enough for the next diagnostic question: can a small causal ML layer remove
entries that do not begin clean continuation, without asking ML to invent direction or economics?

ML noise-filter v0.1 keeps the LONG strategy frozen. It first restores the deterministic 8x
target-distance/execution-cost gate (equivalent to expected cost <=0.25R for the fixed 2R target), then
classifies only the remaining candidates. Costs are not model features.

The v0.1 label is:
- CLEAN: +1R before stop and within 8 hours;
- WHIPSAW: stop first, then +1R within the 24-hour label horizon;
- NOISE: all remaining cases.

The binary model learns CLEAN vs not-CLEAN, while WHIPSAW and NOISE remain visible in diagnostics.
HistGradientBoostingClassifier uses a fixed 0.50 probability threshold and fixed parameters; there is no
threshold/grid search in v0.1. 2023 is training-only, 2024 and 2025 are expanding-window walk-forward test
folds, and 2026 remains untouched.

Run:

    medium-trading btc-long-noise-ml-v0-1-evaluate `
      --data "data/bybit/BTCUSDT_M30.csv" `
      --symbol BTCUSDT `
      --fee-bps-per-side 5.5 `
      --slippage-bps-per-side 2.0 `
      --starting-equity 10000 `
      --risk 0.005 `
      --json artifacts/bybit_btcusdt_noise_ml_v0_1.json

The report compares raw baseline, economic-gate only and economic-gate + ML on the same fixed candidate
stream, with ordinary/2x-cost PnL metrics plus CLEAN precision/recall and class counts. v0.1 only answers
whether the noise-filter concept has useful out-of-sample discrimination; do not tune it from the first
result.

v0.1 failed the feasibility test. After the economic gate, CLEAN prevalence was about 47.3%, while the
model's selected set had only 41.6% CLEAN precision and 31.8% recall. Economic-gate only was +27.16R
gross / -21.45R net; adding v0.1 ML made the selected set -15.57R gross / -34.38R net. The failure
direction appeared in both 2024 and 2025, so v0.1 is not threshold-tuned.

v0.2 changes only the label question. The model, features, 0.50 threshold, economic gate and trading
strategy remain frozen. REAL_MOVE means the market reaches +2R from entry within 24 hours regardless of
whether the original stop is touched first. Diagnostics split positives into DIRECT_MOVE (+2R before
stop) and POST_STOP_MOVE (stop first, then +2R). NO_MOVE means +2R is not reached within 24 hours.

2024 and 2025 are now development walk-forward folds rather than pristine OOS because their v0.1 results
have already been inspected. 2026 remains untouched.

Run v0.2:

    medium-trading btc-long-move-ml-v0-2-evaluate `
      --data "data/bybit/BTCUSDT_M30.csv" `
      --symbol BTCUSDT `
      --fee-bps-per-side 5.5 `
      --slippage-bps-per-side 2.0 `
      --starting-equity 10000 `
      --risk 0.005 `
      --json artifacts/bybit_btcusdt_move_ml_v0_2.json

The primary v0.2 question is classification lift: does selected REAL_MOVE precision exceed the
economic-gate REAL_MOVE base rate with useful recall, consistently across both development folds?
Trade PnL remains reported because POST_STOP_MOVE can still lose under the current execution despite
being a true REAL_MOVE label.

v0.2 also failed: combined REAL_MOVE precision was below the economic-gate base rate, so the project
does not continue tuning the M30 entry classifier. The next experiment moves ML into the position-management
problem directly.

## Market State Model v0.1

Market State Model v0.1 asks whether a richer state can distinguish recoverable market noise from genuine
reversal risk while an economic-pass LONG position is alive. It is diagnostic only: it does not yet replace
the hard stop or issue live HOLD/EXIT decisions.

Additional Bybit inputs:
- native M5 klines;
- 30-minute open interest;
- 30-minute long/short account ratio;
- historical funding rates.

The public Bybit V5 endpoints used are market kline, open interest, long/short ratio and funding history.
Higher timeframes are built only from completed M5 buckets so a snapshot never sees a partially formed
future candle.

Download the 2023-2025 state bundle:

    medium-trading download-bybit-state `
      --symbol BTCUSDT `
      --from 2023-01-01 `
      --to 2025-12-31 `
      --output-dir data/bybit/state

This creates:
- data/bybit/state/BTCUSDT_M5.csv
- data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv
- data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv
- data/bybit/state/BTCUSDT_FUNDING.csv

The model samples every 15 minutes only while the original position is still alive, and only at management
stress points: retrace from prior MFE >=0.15R or current PnL <=-0.05R. Each snapshot sees M5/M15/M30/1H/4H
price/volume state, current trade path, OI, account positioning and latest funding.

The 8-hour diagnostic labels are:
- TREND_VALID: +0.5R before -0.5R;
- NOISE_PULLBACK: -0.5R first, then recovery to +0.5R within 8h;
- REVERSAL: -0.5R without recovery to +0.5R within 8h;
- STALL: neither threshold reached.

The classifier predicts REVERSAL vs all other states. Two regressors separately predict future 2-hour MFE
and MAE. 2024 and 2025 are expanding-window development folds; 2026 remains untouched.

Run:

    medium-trading btc-market-state-v0-1-evaluate `
      --m30 "data/bybit/BTCUSDT_M30.csv" `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --symbol BTCUSDT `
      --fee-bps-per-side 5.5 `
      --slippage-bps-per-side 2.0 `
      --starting-equity 10000 `
      --risk 0.005 `
      --json artifacts/bybit_btcusdt_market_state_v0_1.json

The first result should be judged as an information test: reversal precision/recall/lift and ROC AUC,
plus whether the MFE/MAE regressors beat a naive training-mean predictor. Do not tune the threshold from
the first result.

Market State Model v0.1 failed this information test. Combined 2024-2025 reversal precision was 25.8%
against a 27.5% base rate, recall was 7.7%, and ROC AUC was 0.512. The 2-hour MFE/MAE regressors also
failed to beat naive training-mean predictors overall. The v0.1 threshold and boosting parameters are
therefore not tuned further.

## Market Structure Model v0.2

v0.2 changes the representation rather than the strategy. The same economic-pass LONG trades and the same
stress snapshots are reused, but a deterministic causal structure layer now identifies confirmed M5 swing
highs/lows, structural breaks, sweeps, reclaims, acceptance and retests before the classifier sees the state.

Frozen structure rules:
- confirmed M5 swing = 2 bars left + 2 bars right;
- primary LONG invalidation level = most recent confirmed swing low;
- structural close break = close at least 0.10 ATR5 below that swing low;
- acceptance = at least 3 of the next 5 M5 closes remain below the level;
- retest tolerance = 0.15 ATR5 around the broken level;
- fast reclaim window = 3 M5 bars;
- 8-hour label horizon and +/-0.5R consequence scale remain unchanged.

The labels are deliberately more explicit:
- NOISE: sweep/break attempt is quickly reclaimed without structural acceptance;
- CORRECTION: key swing low remains structurally intact and +0.5R recovery occurs before -0.5R continuation;
- REVERSAL_CANDIDATE: body closes through the swing low but confirmation is incomplete;
- CONFIRMED_REVERSAL: structural break plus acceptance or held retest, followed by adverse continuation;
- AMBIGUOUS: unclear outcomes.

The classifier is trained/scored only on clear NOISE, CORRECTION and CONFIRMED_REVERSAL samples.
REVERSAL_CANDIDATE and AMBIGUOUS are reported but excluded from classifier scoring so uncertain cases are
not forced into a binary label. 2024 and 2025 remain development folds; 2026 remains untouched.

Run with the already-downloaded state bundle:

    medium-trading btc-market-structure-v0-2-evaluate `
      --m30 "data/bybit/BTCUSDT_M30.csv" `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --symbol BTCUSDT `
      --fee-bps-per-side 5.5 `
      --slippage-bps-per-side 2.0 `
      --starting-equity 10000 `
      --risk 0.005 `
      --json artifacts/bybit_btcusdt_market_structure_v0_2.json

The first v0.2 result is still a feasibility test. Judge it by class distribution, clear/excluded sample
counts, reversal precision/recall/lift, ROC AUC and stability across both development folds. Do not tune
the 0.50 threshold from the first result.

Market Structure Model v0.2 failed as an ML discriminator. Combined 2024-2025 precision was 53.39%
against a 50.44% reversal base rate, precision lift was 1.058x, recall was 49.45%, ROC AUC was 0.524
and Brier score was 0.289. The weak lift appeared in both development folds and deteriorated from
1.076x in 2024 to 1.035x in 2025. The structural extractor itself remained useful enough to test
directly, but the periodic-snapshot classifier is not promoted to trade management.

## Market Structure Event Model v0.3

v0.3 removes ML entirely and tests whether the already-frozen causal structure rules improve management
of the same economic-pass Trend LONG v1.1 candidate stream. Samples are no longer created every 15 minutes.
The system reacts only after an observable M5 structural event has completed.

Frozen event policy:
- SWEEP_RECLAIM: M5 trades below the current confirmed swing low and closes back above it inside the
  frozen 3-bar reclaim window -> HOLD;
- BODY_BREAK_UNCONFIRMED: M5 closes at least 0.10 ATR5 below the swing low -> WAIT;
- BREAK_ACCEPTED: the break reaches at least 3 closes below the level inside the 5-bar acceptance window
  -> EXIT;
- RETEST_HELD: after a break, price retests within 0.15 ATR5 of the level and closes below it -> EXIT;
- BREAK_RECLAIMED: a body break closes back above the level inside 3 M5 bars -> HOLD / cancel WAIT;
- structural EXIT is executed at the next M5 open, never at the confirming candle close;
- the original hard stop, 2R target and 24-hour maximum holding remain active and unchanged;
- the candidate stream is fixed. An earlier structural exit never introduces a replacement trade;
- fee/slippage assumptions remain 5.5 bps + 2.0 bps per side, and ordinary plus 2x-cost results are reported;
- 2026 remains untouched.

The report deliberately includes three layers: the source M30 trade result, a control re-simulation of the
same trade on native M5, and the managed M5 result. This separates any benefit from finer execution
resolution from the actual event-management effect.

Run:

    medium-trading btc-market-structure-events-v0-3-evaluate `
      --m30 "data/bybit/BTCUSDT_M30.csv" `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --symbol BTCUSDT `
      --fee-bps-per-side 5.5 `
      --slippage-bps-per-side 2.0 `
      --starting-equity 10000 `
      --risk 0.005 `
      --json artifacts/bybit_btcusdt_market_structure_events_v0_3.json

Judge v0.3 by managed-vs-original M5 net R, profit factor, drawdown, 2x-cost robustness, yearly stability,
the number of structural exits that improve versus worsen the original outcome, delta-R by BREAK_ACCEPTED
and RETEST_HELD, body-break resolution counts and SWEEP_RECLAIM follow-through. Do not add ML or tune the
frozen structure thresholds before reading this deterministic result.

v0.3 did not improve trade management. The native-M5 control matched source-M30 economics, while the
structural EXIT policy moved the 407-candidate stream from +25.42R gross / -45.22R net / PF 0.843 to
+7.93R gross / -62.70R net / PF 0.658. Structural exits improved 191/292 cases but worsened 101/292,
and those mistakes were large enough to produce -17.48R delta. The structure layer therefore remains
diagnostic; BREAK_ACCEPTED and RETEST_HELD are not promoted to hard trading actions.

## Market Observer v0.4

v0.4 changes the task completely. ML no longer evaluates trades or position management. It observes
BTCUSDT market state continuously and classifies whether a causal structural disturbance is noise,
a recoverable correction or a real trend reversal.

There are no trade inputs in this experiment: no entry, stop, target, current R, MFE/MAE, PnL, risk,
fees or slippage. The observer uses only the existing 2023-2025 Bybit state bundle:
- native M5 price/volume;
- completed M15/M30/H1/H4 context derived causally from M5;
- open interest;
- long/short account ratio;
- settled funding.

Sampling is market-driven. A completed-H4 trend must first exist: BULL means close above rising EMA20,
BEAR means close below falling EMA20. Confirmed M5 swings remain 2 bars left + 2 bars right. In a BULL
trend the observer watches the latest confirmed swing low; in a BEAR trend it watches the latest confirmed
swing high. The first causal disturbance of each defended swing becomes one sample:
- SWEEP_RECLAIM: the level is pierced but the M5 close returns to the trend side;
- BODY_BREAK: the M5 close finishes at least 0.10 ATR5 through the defended level.

Features are normalized and market-only: multi-timeframe ATR-scaled returns, ATR14/ATR200,
realized-volatility ratio, Kaufman efficiency, bar overlap, candle body/wicks, relative and z-scored
volume, EMA distance/slope, RSI14 and RSI z-score, swing geometry and structural penetration,
OI changes, price/OI interaction, account positioning and funding. Directional features are signed so
positive always means with the prevailing H4 trend. CatBoost additionally receives trend side, event type,
UTC session bucket and weekday as categorical features.

The 8-hour labels are market outcomes rather than trade results:
- NOISE: +0.75 ATR with-trend occurs before -0.75 ATR adverse;
- CORRECTION: -0.75 ATR adverse occurs first, but +0.75 ATR with-trend later recovers inside 8 hours;
- REAL_REVERSAL: structurally accepted damage extends at least -1.50 ATR against trend and the +0.75 ATR
  recovery barrier is not reached inside 8 hours;
- AMBIGUOUS: unresolved outcomes, reported but excluded from model scoring.

The primary model is CatBoostClassifier (300 trees/iterations, depth 6, learning rate 0.05, L2 5).
A fixed RandomForestClassifier (300 trees, depth 8, min leaf 20) is the benchmark. 2023 is training-only;
2024 and 2025 are expanding walk-forward development folds. Every training label must finish before the
test fold and an additional 8-hour embargo is applied. 2026 is untouched.

Run:

    medium-trading btc-market-observer-v0-4-evaluate `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --json artifacts/bybit_btcusdt_market_observer_v0_4.json

Judge the first result only as an observer-information test. Report per-class precision/recall/F1/lift,
macro F1, REAL_REVERSAL PR-AUC/ROC-AUC/Brier/calibration, probability bands, class/event/trend balance,
feature importance stability and CatBoost versus RandomForest consistency. v0.4 does not emit BUY/SELL/
HOLD/EXIT decisions and must not be judged by trading PnL. Do not tune it from the first result.

v0.4 showed weak but repeatable market-only reversal information rather than a deployable observer.
REAL_REVERSAL was only 8.26% of clear 2024-2025 samples. Multiclass CatBoost never chose REAL_REVERSAL
as its argmax class, but reversal ranking reached ROC-AUC 0.592 and PR-AUC 0.115 versus an 8.26% base
rate. RandomForest with balanced_subsample independently reached ROC-AUC 0.595, PR-AUC 0.115 and 1.46x
reversal precision lift. The signal appeared in both development years, so the next experiment isolates
the rare-class learning architecture without changing market sampling, labels or features.

## Market Observer v0.5

v0.5 is hierarchical and market-only:
- Stage 1: REAL_REVERSAL vs NOT_REVERSAL (NOISE + CORRECTION);
- primary Stage 1 CatBoost uses Logloss plus auto_class_weights=Balanced;
- an otherwise identical unweighted binary CatBoost is retained as a direct class-imbalance control;
- Stage 2: CORRECTION vs NOISE, trained only on NOT_REVERSAL samples;
- no Focal Loss, SMOTE, undersampling, new indicators or label changes.

Each yearly fold now has four chronological regions:
1. model fit history;
2. a dedicated 60-day probability-calibration window;
3. a later dedicated 60-day threshold-validation window;
4. the untouched calendar-year test fold.

The original 8-hour label embargo remains. Platt-style logistic calibration is fit only on the calibration
window. A diagnostic reversal threshold is selected by maximum F1 only on the later pre-test validation
window. Test-year outcomes never select a threshold. Live consumers should still use the calibrated
probabilities rather than the hard diagnostic class.

The final state probabilities are composed as:

    P(REAL_REVERSAL) = Stage1
    P(CORRECTION) = (1 - P(REAL_REVERSAL)) * P(CORRECTION | NOT_REVERSAL)
    P(NOISE) = (1 - P(REAL_REVERSAL)) * (1 - P(CORRECTION | NOT_REVERSAL))

Run:

    medium-trading btc-market-observer-v0-5-evaluate `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --json artifacts/bybit_btcusdt_market_observer_v0_5.json

Judge v0.5 primarily by weighted-vs-unweighted reversal ROC-AUC/PR-AUC, PR-AUC lift, calibrated Brier/ECE,
top 1/5/10/20% reversal-rate lift, past-only operating-point precision/recall/F1 in each fold, Stage-2
noise/correction quality and hierarchical probability quality. 2026 remains untouched. Do not tune v0.5
from its first result.

v0.5 showed that class imbalance was only part of the problem. The balanced binary model improved combined
ranking modestly, but not consistently in both years, and the separate NOISE-vs-CORRECTION stage remained
close to random. The next experiment therefore keeps the same binary reversal observer and adds only one
new information source: genuine taker trade flow.

## Market Observer v0.6

Bybit publishes daily gzip archives of public BTCUSDT trades. v0.6 streams those archives and aggregates
them into completed UTC M5 flow buckets without persisting raw ticks.

Download and aggregate the frozen 2023-2025 flow history:

    medium-trading download-bybit-trade-flow `
      --symbol BTCUSDT `
      --from 2023-01-01 `
      --to 2025-12-31 `
      --workers 2 `
      --output-dir data/bybit/flow

Completed days are cached under data/bybit/flow/daily, so an interrupted multi-year collection can resume.
The final combined file is:

    data/bybit/flow/BTCUSDT_TRADE_FLOW_M5.csv

Every UTC day must contain exactly 288 contiguous M5 flow buckets; missing buckets are rejected rather than
synthesized. Each aggregate stores taker-buy/taker-sell quantity, notional and trade count.

v0.6 adds only causal flow features to the frozen v0.5 weighted binary REAL_REVERSAL observer:
- taker quantity delta over 5m / 15m / 30m / 2h;
- taker notional delta over 5m / 30m;
- buy/sell trade-count imbalance over 5m / 30m;
- trend-aligned delta over 5m / 30m / 2h;
- 5m-vs-30m delta acceleration;
- current trade-count activity relative to the trailing 2h average;
- average buy-vs-sell trade-size imbalance;
- 30m persistence of aggressive flow with/against the prevailing H4 trend.

The exact v0.5 weighted binary model without flow is rerun inside the same experiment as a control. Sampling,
labels, base market features, CatBoost parameters, chronological calibration/threshold-validation and the
2024/2025 test folds remain unchanged.

Run:

    medium-trading btc-market-observer-v0-6-evaluate `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --trade-flow "data/bybit/flow/BTCUSDT_TRADE_FLOW_M5.csv" `
      --json artifacts/bybit_btcusdt_market_observer_v0_6.json

Judge v0.6 only by incremental information: flow-enhanced versus the exact v0.5 control on ROC-AUC, PR-AUC,
PR-AUC lift, calibration and top-probability reversal concentration, separately in 2024 and 2025 as well as
combined. Flow must improve ranking directionally in both yearly folds to justify keeping it. Do not tune
v0.6 from the first result, and do not use 2026.

v0.6 showed genuine but regime-dependent incremental information. Combined ROC-AUC improved from 0.5652
to 0.5711 and PR-AUC from 0.1069 to 0.1122; top-5% and top-10% reversal concentration also improved.
However, 2024 ROC-AUC deteriorated while 2025 improved, so taker flow is retained as context but the
first-disturbance prediction moment remains too unstable.

## Market Observer v0.7

v0.7 tests one hypothesis only: the observer may be looking at the market too early.

The original disturbance remains the episode anchor, but prediction moves exactly 15 minutes later, after
three fully completed M5 bars. The original v0.6 market + taker-flow feature vector remains unchanged.
The confirmed model adds only information that is genuinely known by T+15m:

- adverse displacement from the disturbance close;
- maximum adverse extension and mean close distance beyond the defended level;
- counts of closes/touches beyond the level;
- final reclaim and reclaim speed;
- retest happened / retest held using the frozen 0.15 ATR tolerance;
- post-event price-path efficiency, range and adverse-body fraction;
- 15-minute post/pre candle-volume ratio;
- post-event taker delta, notional delta and trade-count imbalance;
- trend-aligned post-event delta;
- shift in taker delta versus the pre-event 15 minutes;
- aggressive-flow persistence;
- trade-count activity versus the pre-event 2-hour average;
- average buy-vs-sell trade-size imbalance.

The control is intentionally evaluated on the same delayed samples and the same chronological folds, but
receives only the original v0.6 features. This isolates the information value of waiting for confirmation.

The binary target is unchanged:

    REAL_REVERSAL vs TREND_SURVIVES

The original disturbance labels are preserved. No candle or flow bucket ending after T+15m enters a
feature. CatBoost parameters, class weighting, calibration, threshold-validation and 2024/2025 folds are
unchanged; 2026 remains untouched.

Run:

    medium-trading btc-market-observer-v0-7-evaluate `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --trade-flow "data/bybit/flow/BTCUSDT_TRADE_FLOW_M5.csv" `
      --json artifacts/bybit_btcusdt_market_observer_v0_7.json

Judge v0.7 by delayed-control versus confirmed-event ROC-AUC, PR-AUC, PR-AUC lift, calibration and
top-probability reversal concentration separately in 2024 and 2025, then combined. Confirmation must
improve ranking directionally in both development folds to survive. Do not test alternate 10/20-minute
windows after seeing this result.

v0.7 passed that full-sample feasibility test strongly: combined ROC-AUC reached 0.8463 and PR-AUC 0.2740,
with directionally similar improvements in both 2024 and 2025. Before promoting the observer, run one
diagnostic audit for label overlap at the T+15 prediction time.

## v0.7 label-overlap audit

This audit does not change or retrain v0.7. It asks whether some frozen outcome labels were already fully
known by the time the confirmed observer made its T+15 prediction.

Resolution rules match the original label definition:
- NOISE is already resolved only if +0.75 ATR with-trend was reached before -0.75 ATR adverse inside the
  first three completed post-disturbance M5 bars;
- CORRECTION is already resolved only if -0.75 ATR adverse occurred first and +0.75 ATR with-trend recovery
  also occurred by T+15;
- a same-bar trend/adverse touch is resolved as AMBIGUOUS;
- REAL_REVERSAL is never considered resolved at T+15 because its label requires no recovery over the full
  8-hour horizon.

The frozen v0.7 models are reproduced exactly. Metrics are then recalculated only on the clear test samples
whose labels were still unresolved at prediction time. There is no retraining, threshold retuning or
recalibration on that clean subset.

Run:

    medium-trading btc-market-observer-v0-7-overlap-audit `
      --m5 "data/bybit/state/BTCUSDT_M5.csv" `
      --open-interest "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv" `
      --account-ratio "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv" `
      --funding "data/bybit/state/BTCUSDT_FUNDING.csv" `
      --trade-flow "data/bybit/flow/BTCUSDT_TRADE_FLOW_M5.csv" `
      --json artifacts/bybit_btcusdt_market_observer_v0_7_overlap_audit.json

Read the audit by comparing full-v0.7 versus unresolved-only ROC-AUC, PR-AUC and top-tail lift, separately
for 2024 and 2025 and combined. If the unresolved-only confirmed observer remains materially above the
delayed control in both years, the 15-minute confirmation result is not merely a target-overlap artifact.

The audit passed. Roughly half of clear 2024-2025 cases were already resolved at T+15, so the original
full-sample ROC-AUC 0.846 was partly inflated. After removing all such cases without retraining or
recalibration, confirmed v0.7 still achieved ROC-AUC 0.7065 and PR-AUC 0.2875 versus the delayed control
at 0.5567 / 0.2031, with the advantage remaining positive in both 2024 and 2025.

## Frozen v0.7 forward validation — 2026

2026 is now opened once as untouched forward data. Do not tune v0.7 from this result.

The frozen forward window is:

    2026-01-01T00:00:00Z <= event_time < 2026-09-30T00:00:00Z

September 29, 2026 is the last included full UTC day. Download a separate forward bundle with warmup from
November 1, 2025 so the existing 2023-2025 development files are not overwritten:

    medium-trading download-bybit-state `
      --symbol BTCUSDT `
      --from 2025-11-01 `
      --to 2026-09-29 `
      --output-dir data/bybit/forward2026/state

    medium-trading download-bybit-trade-flow `
      --symbol BTCUSDT `
      --from 2025-11-01 `
      --to 2026-09-29 `
      --workers 2 `
      --output-dir data/bybit/forward2026/flow

Then run the one-shot forward evaluator:

    medium-trading btc-market-observer-v0-7-forward-evaluate `
      --json artifacts/bybit_btcusdt_market_observer_v0_7_forward_2026.json

The command defaults to the frozen development bundle under data/bybit/state and data/bybit/flow and the
separate forward bundle under data/bybit/forward2026. It trains, calibrates and selects the diagnostic
threshold only from pre-2026 samples, then scores 2026 once.

The primary forward evidence is the unresolved-at-T+15 subset. The pass gate was frozen before inspecting
2026:
- at least 100 unresolved clear samples;
- unresolved confirmed ROC-AUC >= 0.65;
- unresolved confirmed PR-AUC lift >= 1.40x versus the 2026 unresolved base rate;
- unresolved confirmed ROC-AUC and PR-AUC must both exceed the delayed control;
- unresolved top-10% reversal-rate lift >= 1.50x.

Full-sample metrics and monthly unresolved metrics are reported as diagnostics. Brier/ECE are reported but
are not forward pass/fail gates because no 2026 recalibration is allowed. If this one-shot window fails,
do not modify v0.7 and rerun the same 2026 period as untouched forward validation.

The one-shot 2026 observer gate passed. On the unresolved-at-T+15 subset, frozen v0.7 produced
ROC-AUC 0.7278, PR-AUC 0.3015 and 2.135x PR lift versus the delayed control at ROC-AUC 0.5627 /
PR-AUC 0.1718. Market Observer v0.7 is therefore frozen as forward-validated observer evidence.
2026 is observed evidence from this point onward.

## Observer -> LONG Trading Policy v1

The observer remains independent from the trading bot. Communication is an in-process, transport-neutral
`ObserverSnapshot` message:

    ObserverSnapshot(
        timestamp=...,
        model_version="market-observer-v0.7",
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=...,
        trend_survives_probability=...,
        reversal_threshold=...,
        market_state=...
    )

The snapshot contains no position, entry, stop, target, risk or PnL state. A separate
`LongObserverPolicy` translates the market message into position management:

    BULL + REAL_REVERSAL_RISK -> EXIT_LONG
    BULL + TREND_SURVIVES     -> HOLD_LONG
    BEAR snapshot             -> NO_ACTION

v1 changes only management of an already-open LONG. Entry selection is unchanged. The candidate stream is
the fixed economic-pass Trend LONG v1.1 executed stream. An observer exit happens at the next native-M5
open after the snapshot; a stop-gap or target-gap at that same open has priority. The fold-specific
observer threshold remains the past-only v0.7 threshold and is never optimized against trading PnL.

The historical runtime generates out-of-time snapshots for 2024 and 2025. Model fit, Platt calibration and
threshold selection use only earlier clear labels, but inference is performed for every structural event in
the test year, including events whose eventual research label is AMBIGUOUS. This matches how the observer
would communicate live.

The first policy experiment deliberately does not block entries and does not add replacement trades after
an early observer exit. It isolates one question: does the observer improve management of the same open
LONG positions?

Run:

    medium-trading btc-observer-long-policy-v1-evaluate `
      --json artifacts/bybit_btcusdt_observer_long_policy_v1.json

The command defaults to:
- `data/bybit/BTCUSDT_M30.csv`;
- `data/bybit/state/BTCUSDT_M5.csv`;
- the existing 2023-2025 OI/account-ratio/funding bundle;
- `data/bybit/flow/BTCUSDT_TRADE_FLOW_M5.csv`;
- fee 5.5 bps/side and slippage 2.0 bps/side;
- USD 1,000 starting equity and 0.5% risk.

The policy-value gate is frozen before reading the first PnL result:
- at least 100 managed trades;
- managed net R > baseline net R;
- managed PF > baseline PF;
- managed max drawdown <= baseline max drawdown;
- positive delta net R in both 2024 and 2025;
- observer-triggered exits add positive total delta net R;
- managed 2x-cost net R > baseline 2x-cost net R.

This gate measures incremental management value only. A PASS does not by itself prove that the underlying
Trend LONG strategy is profitable. Do not tune observer v0.7 or the v1 policy after reading this result.

v1 failed the frozen gate. Across 279 fixed trades, baseline net R was -21.45R and managed net R was
-19.01R, so observer exits added +2.44R in aggregate, but PF fell from 0.890 to 0.864 and max drawdown
rose slightly from 13.39% to 13.59%. The effect was unstable by year: +6.17R in 2024 and -3.74R in 2025.
Of 146 observer exits, 98 improved the trade and 48 worsened it; 91 avoided a later baseline stop, while
32 prematurely exited trades that later reached the 2R target. This shows that the observer signal carries
economic information, but treating every classification-level reversal signal as an immediate full exit is
too aggressive.

## Observer -> LONG Trading Policy v2

v2 keeps Market Observer v0.7, its probabilities and its fold thresholds unchanged. Only the policy layer
changes from a stateless immediate-exit rule to a stateful confirmation rule:

    CLEAR
      + BULL REAL_REVERSAL_RISK
      -> WARNING_LONG

    WARNING
      + later BULL REAL_REVERSAL_RISK
      -> EXIT_LONG

    WARNING
      + BULL TREND_SURVIVES
      -> HOLD_LONG and clear WARNING

    any BEAR snapshot
      -> NO_ACTION and clear stale BULL WARNING

The second reversal-risk snapshot is independent when its timestamp is strictly later than the snapshot
that armed WARNING. Event type does not need to differ. Every new trade starts in CLEAR state, so warning
state never leaks between positions.

Everything else is frozen from v1:
- same 2024-2025 walk-forward v0.7 snapshots;
- same fold-specific observer thresholds;
- same economic-pass Trend LONG v1.1 candidate stream;
- same entries, stops, 2R targets and 24h holding horizon;
- same fees/slippage and 2x-cost stress;
- no entry blocking;
- no replacement trades;
- observer still receives no position/PnL state;
- 2026 trading outcomes remain unused.

Run:

    medium-trading btc-observer-long-policy-v2-evaluate `
      --json artifacts/bybit_btcusdt_observer_long_policy_v2.json

The frozen v2 gate requires:
- at least 100 managed trades;
- positive combined delta net R versus baseline;
- v2 managed net R > v1 managed net R;
- v2 PF > baseline PF;
- v2 PF > v1 PF;
- v2 max drawdown <= baseline max drawdown;
- positive delta net R in both 2024 and 2025;
- confirmed v2 exits add positive total delta net R;
- v2 2x-cost net R > baseline 2x-cost net R;
- v2 2x-cost net R > v1 2x-cost net R;
- fewer premature exits before baseline targets than v1.

This is still an incremental position-management test, not proof of a profitable entry strategy. Do not
change warning count, reset semantics or probability thresholds after seeing the first v2 result.

The legacy daily gate remains strict and conjunctive for the older daily-income studies: at least 250 eligible sessions, at least 300 trades, >=70%
active-session rate, >=45% profitable active days, positive average net R/session, net PF >=1.15,
2x-cost PF >1.00, at least 2 positive yearly folds, >=60% positive months, worst day no worse than -2R,
losing-day streak no longer than 8 sessions, and no single best day contributing 15% or more of total
positive daily R.

The project evaluates daily returns in R first. A requested daily percentage return is not used to tune
the strategy; percentage expectations are considered only after a strategy survives the fixed research
gate.

## Status

Automatic Dukascopy date-range download, manual CSV import, and historical multi-pair evaluation are
implemented. Live/paper execution is intentionally not implemented until the strategy survives realistic
historical costs and out-of-sample validation.

See agent.md before making changes.
