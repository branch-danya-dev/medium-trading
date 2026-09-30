# Agent instructions

These instructions are mandatory for every agent working in this repository.

## Mission

Build a small, understandable medium-frequency systematic trading bot. The first goal is to prove or
disprove a net trading edge after realistic costs. Engineering complexity is secondary.

## Fixed MVP scope

- Research scope: multiple liquid markets, evaluated one strategy/market hypothesis at a time.
- Legacy FX universe: EUR/USD, GBP/USD, USD/JPY, AUD/USD plus four external FX pairs used in validation.
- Current market: Crypto, starting with BTC/USD.
- Core data timeframe for the current candidate: native M5 with completed M15/M30/H1/H4 context.
- Current research candidate: BTC Trend LONG Entry Strategy v2 — persistent H4 regime, tested only on 2023-2025 development data against a matched-random entry control.
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

BTC/USD Trend LONG v1.1 was then evaluated:
- 1,019 LONG trades;
- gross +35.57R with gross PF 1.055;
- net -173.98R under the old fixed USD 50 price-point cost model;
- total modeled cost 209.55R;
- 649 stop/stop-gap exits;
- 351 stopped trades later reached +1R and 251 later reached +2R inside the 24-hour diagnostic horizon;
- widening the stop improved the result materially versus v1 but did not remove the core execution/timing problem.

That v1.1 result also exposed two research-engine problems that must be corrected before any ML/noise filter is added:
1. a fixed USD 50 BTC cost is not a stable transaction-cost model across BTC price regimes;
2. 48 encountered M30 bars are not necessarily 24 real hours when the source history contains gaps.

The current rerun keeps the Trend LONG v1.1 strategy completely unchanged and corrects only research mechanics:
- fee model: notional percentage/bps, default 5.5 bps per side. This mirrors a conservative non-VIP taker/taker perpetual execution assumption; it is configurable and is not tuned to the observed result;
- slippage model: separate configurable research assumption, default 2.0 bps per side;
- fee and slippage are reported separately in R and summed into total execution cost;
- funding is not modeled in this rerun because the Dukascopy candle dataset does not contain historical funding-rate series. Any later perpetual-live conclusion must add venue-specific funding;
- maximum holding is a true 1,440 wall-clock minutes from entry;
- the strategy is evaluated only after 192 consecutive M30 bars (96 hours) with exact 30-minute spacing so the ATR and completed 4H EMA context are not built across gaps;
- after a signal, the entry plus 48-bar forward execution window must also be contiguous. Otherwise the setup is rejected as a data-quality gap instead of being simulated across missing history;
- report gap_context_rejections separately from gap_signal_rejections;
- ML remains OFF and the entry, EMA regime, ATR stop floor, 2R target, 0.5% risk and USD 10,000 starting equity remain unchanged.

This corrected rerun is not a new trading strategy and must not be interpreted as parameter tuning. Its purpose is to establish a trustworthy deterministic LONG baseline before deciding whether an ML noise filter has a valid role.

The next data-source step is exchange-native Bybit history without changing the strategy:
- venue/product: Bybit USDT perpetual, category=linear, symbol=BTCUSDT;
- source endpoint: public V5 GET /v5/market/kline;
- interval: native 30-minute klines, not locally aggregated;
- fixed research window: 2023-01-01 through 2025-12-31 UTC;
- downloader must require exactly continuous M30 coverage for the requested historical range and fail visibly on a missing interval; never synthesize a Bybit candle;
- downloader uses no API key for public market history;
- default mainnet endpoint is https://api.bybit.com/v5/market/kline, with CLI override for Bybit region-specific REST hosts when required;
- rerun the exact same v1.1 strategy and corrected execution assumptions on BTCUSDT. Only the venue/data source changes;
- keep 5.5 bps per-side taker fee and 2.0 bps per-side slippage for the initial Bybit rerun so the result is directly interpretable as conservative taker execution;
- funding remains out of this specific rerun until the deterministic gross edge is established. If the strategy survives, add actual Bybit historical funding before any paper/live conclusion.

Do not compare the new Bybit result to Dukascopy as though sample counts must match. The purpose is to remove the severe data-fragmentation bias and determine whether the deterministic LONG setup has gross edge on continuous exchange-native BTCUSDT history.

The exchange-native Bybit BTCUSDT rerun produced 1,130 trades over 2023-2025. The deterministic LONG v1.1 stream was approximately flat before costs: gross +9.20R with gross PF 1.013, but realistic taker-fee plus slippage assumptions consumed 393.28R and made the net result deeply negative. About 77.8% of candidates reached +1R at some point within 24 hours, while many stopped trades later recovered. This establishes a useful diagnostic basis for a narrow ML noise-filter feasibility experiment; it does not establish profitability.

The current frozen ML task is BTCUSDT LONG noise-filter v0.1. It must not revive the previously rejected Mean Reversion ML filter or Direct ML opportunity model. Its purpose is only to test whether a small causal model can distinguish immediate clean continuation from whipsaw/noise inside the already-frozen BTC LONG candidate stream.

Noise-filter v0.1 is frozen as follows:
- data: Bybit BTCUSDT linear-perpetual native M30, development years 2023-2025 only;
- 2026 is untouched and must not be loaded, trained on, threshold-tuned on or evaluated during v0.1;
- underlying trading strategy is unchanged: Trend LONG v1.1, same 4H EMA20 regime, M30 pullback/confirmation entry, ATR14 stop floor, 2R target, 24h wall-clock holding, 0.5% risk;
- transaction-cost assumptions remain 5.5 bps fee per side plus 2.0 bps slippage per side;
- candidate stream for v0.1 is the fixed executed deterministic v1.1 baseline stream. Rejecting a candidate does not introduce replacement candidates during this feasibility test; this isolates filter value and is not yet the final live execution policy;
- deterministic economic gate is restored before ML using the project's old 8x expected-move/cost rule: 2R target distance / expected round-trip execution cost must be >=8, equivalent to expected cost <=0.25R. Economics are not an ML feature;
- ML label is not trade WIN/LOSS and not net R;
- CLEAN = +1R is reached before the stop and within 8 hours of entry;
- WHIPSAW = stop is touched before +1R, but +1R is later reached inside the full 24-hour label horizon;
- NOISE = all remaining cases, including no +1R within 24 hours or continuation too slow to qualify as CLEAN;
- binary training target is CLEAN versus not-CLEAN; WHIPSAW/NOISE are retained separately for diagnostics;
- conservative same-bar labeling gives adverse/stop touch priority when both thresholds are present in one M30 candle;
- features are causal and known by entry: 1/2/4/8-bar returns in ATR, ATR/price, ATR14/ATR50, 8-bar efficiency and bullish ratio, confirmation and pullback body/range in ATR, relative volume, stop distance in ATR, and 4H EMA20 distance/slope in 4H ATR;
- model type: HistGradientBoostingClassifier;
- model params: learning_rate=0.05, max_iter=120, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=42;
- probability threshold is fixed at 0.50. There is no threshold search or tuning in v0.1;
- walk-forward tests are calendar 2024 and 2025. 2023 is training-only because the current Bybit file starts in 2023. For each fold, a training label is usable only when its full 24-hour label horizon is known before the test year begins;
- compare three layers on the same test candidate stream: raw baseline, economic-gate only, economic-gate + ML;
- report ordinary and 2x-cost trade metrics, clean precision/recall, class distribution and each yearly fold.

v0.1 is a feasibility test, not an optimization pass. Do not tune features, probability threshold, label horizon, CLEAN definition, economic gate or model hyperparameters after seeing the first v0.1 result. First determine whether the fixed model has useful out-of-sample discrimination at all.

The v0.1 result failed its feasibility purpose. Across the 2024-2025 development walk-forward test stream after the economic gate, CLEAN prevalence was about 47.3%, but the model selected 101/279 candidates with only 41.6% CLEAN precision and 31.8% recall. The economic gate alone improved the stream to +27.16R gross and -21.45R net, while economic-gate + ML deteriorated it to -15.57R gross and -34.38R net. The same failure direction appeared in both 2024 and 2025. Do not tune v0.1 thresholds or hyperparameters on these inspected years.

The next frozen feasibility task is real-move ML filter v0.2. It deliberately simplifies the target instead of tuning the v0.1 model:
- the deterministic Trend LONG v1.1 strategy remains unchanged;
- the same economic gate remains unchanged at 8x target-distance / expected-cost;
- the same 16 causal features remain unchanged;
- the same HistGradientBoostingClassifier and all model hyperparameters remain unchanged;
- probability threshold remains fixed at 0.50;
- 2023-2025 remain development data and are no longer described as pristine out-of-sample because v0.1 results on 2024-2025 have already been inspected;
- 2026 remains untouched and must not be used during v0.2 design or evaluation;
- v0.2 binary label is REAL_MOVE versus NO_MOVE;
- REAL_MOVE = price reaches +2.0R from the original entry at any time within 24 hours, regardless of whether the original stop was touched first;
- NO_MOVE = +2.0R is not reached within 24 hours;
- diagnostics split REAL_MOVE into DIRECT_MOVE (+2R before any stop touch) and POST_STOP_MOVE (stop touched first, then +2R within 24h);
- if stop and +2R occur in the same M30 candle, classify timing as POST_STOP_MOVE conservatively, while the binary REAL_MOVE label remains true;
- use the same fixed executed v1.1 candidate stream and do not introduce replacement candidates when the filter rejects one;
- evaluate raw, economic-gate only, and economic-gate + ML trade metrics, but v0.2's primary feasibility question is classification: does model precision exceed the economic-gate REAL_MOVE base rate with useful recall in both development folds?
- report precision, recall, base REAL_MOVE rate and precision lift for 2024 and 2025 separately and combined, plus DIRECT_MOVE / POST_STOP_MOVE / NO_MOVE distributions.

v0.2 is not a profitability claim and is not allowed to tune features, target horizon, +2R label, economic gate, threshold or model hyperparameters after the result. If v0.2 cannot show meaningful discrimination over the base REAL_MOVE rate, the next discussion must focus on information/features or abandon this ML-filter approach rather than threshold hunting.

The real-move v0.2 result also failed. Combined 2024-2025 development precision was 47.0% against a 49.5% REAL_MOVE base rate (0.95x precision lift), with 39.9% recall. 2024 showed only a small positive lift and 2025 deteriorated materially. Economic-gate only remained much better than economic-gate + ML. Therefore the project does not continue threshold or hyperparameter tuning on the M30 snapshot classifier.

The current task is Market State Model v0.1. This is a different research question: while an economic-pass LONG position is alive, can a richer multi-timeframe state distinguish a temporary adverse fluctuation from a genuine reversal risk?

Market State Model v0.1 is frozen as follows:
- data years: 2023-2025 development only; 2026 remains untouched;
- strategy and economic gate remain unchanged;
- native Bybit M5 candles are added and aggregated causally into completed M15, M30, 1H and 4H state features;
- Bybit 30-minute open interest, 30-minute long/short account ratio and historical funding are added as public market-state inputs;
- no order-book, L2/L3, tick/HFT infrastructure or historical taker-flow dependency is added in v0.1;
- only economic-pass Trend LONG v1.1 trades are used;
- snapshots are sampled every 15 minutes while the position is still alive under the original hard stop/target/24h rules;
- snapshots are considered only when management matters: retrace from prior MFE >=0.15R or current PnL <=-0.05R;
- features must be known at snapshot time. Higher-timeframe aggregates must be fully completed; never use a partially formed M15/M30/1H/4H bucket;
- feature groups: M5 returns/acceleration/efficiency/candle shape/volume/compression, M15/M30/1H/4H trend context, current trade path (current R, MFE/MAE so far, retrace, time in trade, distance to stop/target), OI changes, long-account ratio changes and latest settled funding;
- label horizon is 8 hours and local decision scale is +/-0.5R from the snapshot price;
- TREND_VALID = +0.5R is reached before -0.5R;
- NOISE_PULLBACK = -0.5R is reached first but +0.5R is later recovered inside 8 hours;
- REVERSAL = -0.5R is reached and +0.5R is not recovered inside 8 hours;
- STALL = neither side reaches 0.5R inside 8 hours;
- the first classifier is REVERSAL versus all other states, threshold fixed at 0.50;
- two additional regressors predict future MFE and future MAE over the next 2 hours;
- models are simple HistGradientBoosting models; v0.1 is an information/feasibility test, not a deep-learning or sequence-model attempt;
- expanding-window development folds are 2024 and 2025, with labels required to be fully known before each fold;
- primary diagnostics: reversal base rate, precision, recall, precision lift, ROC AUC, Brier score, state-class distribution, and model-vs-naive MAE for 2h MFE/MAE;
- Market State Model v0.1 is diagnostic only. It does not yet replace the deterministic stop or issue live HOLD/EXIT commands.

Do not tune the v0.1 threshold or model after the first result. First determine whether richer M5/multi-timeframe/positioning state contains measurable information about reversal versus recoverable noise.

Market State Model v0.1 FAILED as a discriminator. On the combined 2024-2025 development folds, reversal base rate was 27.55%, precision 25.79%, recall 7.70%, precision lift 0.94x and ROC AUC 0.512. The 2h MFE and MAE regressors also failed to beat naive training-mean baselines overall. Do not tune its threshold, label distances or boosting hyperparameters on these inspected years.

The next frozen task is Market Structure Model v0.2. It changes the representation, not the trading strategy or economic gate. The purpose is to give ML explicit structural events instead of expecting boosting to reconstruct market structure from generic returns.

Market Structure Model v0.2 is frozen as follows:
- reuse the existing 2023-2025 Bybit M30/M5/open-interest/account-ratio/funding bundle; do not touch 2026;
- use the same economic-pass Trend LONG v1.1 candidate stream and the same 15-minute stress snapshots while the original position is alive;
- identify causal confirmed M5 swing lows/highs with 2 bars on the left and 2 bars on the right; a swing is usable only after its right-side confirmation bars have closed;
- for LONG management, the most recent confirmed swing low is the primary structural invalidation level;
- add explicit structural features: distance to last swing low/high, swing range and age, low/close break flags, close displacement through the level, wick/sweep depth, body fraction below the level, count/consecutive closes below, fast reclaim, break-volume z-score, post-break volume ratio, retest occurrence/hold, and lower-low/lower-high state;
- keep slower M15/M30/1H/4H context, current trade path, OI, account positioning and funding as secondary context;
- a structural break requires an M5 body close at least 0.10 ATR5 below the confirmed swing low;
- acceptance window is the next 5 M5 bars; acceptance requires at least 3 closes below the level;
- retest confirmation uses a 0.15 ATR5 tolerance around the broken level and requires a close to remain below it;
- fast reclaim window is 3 M5 bars;
- label horizon remains 8 hours and local adverse/favorable consequence scale remains +/-0.5R;
- NOISE = a sweep/break attempt is quickly reclaimed without structural acceptance;
- CORRECTION = the key swing low remains structurally intact and +0.5R recovery occurs before -0.5R adverse continuation;
- REVERSAL_CANDIDATE = body closes through the swing low but confirmation remains incomplete;
- CONFIRMED_REVERSAL = structural body break plus acceptance or held retest, followed by adverse continuation before +0.5R recovery;
- AMBIGUOUS = all other unclear outcomes;
- to avoid forcing uncertain labels, train/evaluate the binary classifier only on clear NOISE, CORRECTION and CONFIRMED_REVERSAL samples; exclude REVERSAL_CANDIDATE and AMBIGUOUS from classifier scoring, but report their counts;
- binary target: CONFIRMED_REVERSAL versus NOISE/CORRECTION;
- classifier remains a simple HistGradientBoostingClassifier with fixed threshold 0.50; no threshold search;
- primary diagnostics: clear/excluded sample counts, class distribution, reversal base rate, precision, recall, precision lift, ROC AUC and Brier score for 2024 and 2025 separately and combined;
- v0.2 remains diagnostic only and does not yet replace the hard stop or issue HOLD/WAIT/EXIT decisions.

Do not tune v0.2 after the first result. First determine whether explicit swing-break/reclaim/retest/volume structure provides stable discrimination in both development folds.

Market Structure Model v0.2 FAILED as an ML discriminator. Combined 2024-2025 precision was 53.39%
against a 50.44% reversal base rate, precision lift was 1.058x, recall was 49.45%, ROC AUC was 0.524
and Brier score was 0.289. The weak lift was present in both development folds and deteriorated from
1.076x in 2024 to 1.035x in 2025. Do not tune the v0.2 threshold, boosting parameters or inspected
feature set. The periodic stress-snapshot forecasting formulation is REJECTED.

The current frozen task is Market Structure Event Model v0.3. It tests the structural rules directly
before any further ML attempt:
- ML is OFF;
- use the same economic-pass Trend LONG v1.1 candidate stream;
- reuse native Bybit M30 and M5 history from 2023-2025 only; 2026 remains untouched;
- confirmed M5 swing remains 2 bars left + 2 bars right and is usable only after both right bars close;
- the most recent causally confirmed swing low is the active LONG structural level;
- structural body break remains an M5 close at least 0.10 ATR5 below that level;
- acceptance remains at least 3 closes below the level inside the 5-bar window, counting the break bar;
- retest tolerance remains 0.15 ATR5 and a held retest requires the retest candle to close below the level;
- fast reclaim remains 3 M5 bars;
- SWEEP_RECLAIM -> HOLD;
- BODY_BREAK_UNCONFIRMED -> WAIT;
- BREAK_RECLAIMED -> HOLD and cancel WAIT;
- BREAK_ACCEPTED -> EXIT;
- RETEST_HELD -> EXIT;
- when BREAK_ACCEPTED and RETEST_HELD become true on the same closed M5 candle, RETEST_HELD is the
  attributed exit trigger while both observable events may be reported;
- structural EXIT executes at the next M5 open. Never use the confirming close as the fill;
- hard stop and 2R target remain active while waiting and take conservative precedence if touched before
  the structural exit fill;
- maximum holding remains 24 real hours;
- the candidate stream is fixed. Earlier exits do not create replacement entries;
- ordinary execution costs remain 5.5 bps taker fee + 2.0 bps slippage per side;
- report ordinary and 2x-cost results;
- use native M5 for both a control original-path re-simulation and the managed path. Also retain the source
  M30 result so any M5-resolution effect is visible separately from event-management delta;
- primary comparison is managed M5 versus original M5, not managed M5 versus source M30;
- report gross/net R, PF, average net R/trade, max drawdown, yearly 2023/2024/2025 results, event counts,
  structural-exit improved/worsened counts, average delta-R, delta-R by BREAK_ACCEPTED and RETEST_HELD,
  BODY_BREAK_UNCONFIRMED outcomes, and SWEEP_RECLAIM follow-through to +0.5R/+1R/2R;
- do not add ML, tune structure thresholds or change the underlying entry strategy before reading v0.3.

v0.3 FAILED as a deterministic EXIT policy but produced a useful market-structure diagnostic.
The native-M5 control matched the source M30 economics exactly, so the result was not an execution-resolution
artifact. On 407 fixed candidates, original M5 was +25.42R gross / -45.22R net with net PF 0.843 and
25.4% max drawdown; deterministic structural management fell to +7.93R gross / -62.70R net with net PF
0.658 and 29.2% max drawdown. Structural exits improved 191/292 cases and worsened 101/292, but the
mistakes were more expensive, producing -17.48R delta overall. BREAK_ACCEPTED was -2.99R delta and
RETEST_HELD was -14.49R delta. SWEEP_RECLAIM remained diagnostically useful: about 64.6% later reached
+0.5R, 51.2% reached +1R and 29.9% reached the 2R target. Do not promote BREAK_ACCEPTED or RETEST_HELD
to hard EXIT rules.

The current frozen task is Market Observer v0.4. It is an ML observer of the market, not a trading-decision
model. It must remain completely independent of open positions and strategy outcomes:
- input data: existing 2023-2025 Bybit BTCUSDT native M5, open interest, long/short account ratio and funding;
- do not load or use 2026;
- do not use trade entry, stop, target, current R, MFE, MAE, PnL, risk, fee or slippage features;
- build completed M15/M30/H1/H4 context causally from M5;
- prevailing market trend is defined only from completed H4: BULL when close > EMA20 and EMA20 is above
  its value three completed H4 bars earlier; BEAR is the symmetric inverse; otherwise do not sample;
- confirmed M5 swing remains 2 bars left + 2 bars right and becomes usable only after both right bars close;
- BULL disturbances attack the latest confirmed swing low; BEAR disturbances attack the latest confirmed
  swing high;
- create at most one sample for the first causal disturbance of each defended swing while that trend side
  is active;
- event types are SWEEP_RECLAIM and BODY_BREAK; BODY_BREAK requires a close at least 0.10 ATR5 through
  the defended swing;
- feature groups are normalized multi-timeframe returns, ATR14/ATR200, realized-volatility ratio,
  Kaufman-style efficiency, candle overlap, body/wick geometry, relative/z-scored volume, EMA distance
  and slope, RSI14 plus rolling RSI z-score, swing distances/age/quality, structural penetration/counts,
  open-interest changes, price/OI interaction, long-account ratio and funding;
- directional price features are signed relative to the prevailing H4 trend so positive means with-trend;
- categorical CatBoost features are trend side, event type, UTC session bucket and weekday;
- label horizon is fixed at 8 hours and labels are market outcomes, never trade WIN/LOSS:
  * NOISE = +0.75 ATR with-trend is reached before -0.75 ATR adverse;
  * CORRECTION = -0.75 ATR adverse is reached first, then +0.75 ATR with-trend recovers inside 8h;
  * REAL_REVERSAL = accepted structural damage reaches at least -1.50 ATR against trend and the +0.75 ATR
    trend-recovery barrier is not reached inside 8h;
  * AMBIGUOUS = unresolved outcomes;
- acceptance remains at least 3 closes on the adverse side inside the 5-bar window; BODY_BREAK counts the
  event bar itself;
- train and score only clear NOISE/CORRECTION/REAL_REVERSAL samples; always report AMBIGUOUS counts;
- primary model: CatBoostClassifier with fixed MultiClass settings: iterations=300, depth=6,
  learning_rate=0.05, l2_leaf_reg=5, random_seed=42, no threshold search;
- fixed benchmark: RandomForestClassifier with 300 trees, max_depth=8, min_samples_leaf=20,
  max_features=sqrt, balanced_subsample classes, random_state=42;
- expanding walk-forward test folds are 2024 and 2025; training labels must be fully known before the
  fold and an additional 8-hour embargo is required after label completion;
- 2023 is training-only; 2024/2025 are development folds; 2026 remains untouched;
- report class distribution, event/trend distribution, per-class precision/recall/F1/lift, macro F1,
  multiclass ROC-AUC, log loss, REAL_REVERSAL PR-AUC/ROC-AUC/Brier/calibration error, fixed probability
  bands >=0.50/0.60/0.70/0.80, feature importance and top-10 feature-rank overlap across folds;
- v0.4 emits probabilities/state estimates only. It must not issue BUY/SELL/HOLD/EXIT decisions and must
  not be evaluated by trading PnL.

Market Observer v0.4 did not pass as a deployable observer, but it produced the first
repeatable market-only reversal signal in this research sequence. Combined 2024-2025 clear samples were
12,369 with REAL_REVERSAL prevalence 8.26%. Multiclass CatBoost never emitted REAL_REVERSAL as argmax,
yet its reversal probability ranking reached ROC-AUC 0.592 and PR-AUC 0.115 versus an 8.26% base rate.
The direction repeated across development folds: 2024 ROC-AUC 0.617 / PR-AUC 0.112 at 7.77% base, and
2025 ROC-AUC 0.570 / PR-AUC 0.125 at 8.77% base. RandomForest with balanced_subsample independently
produced ROC-AUC 0.595, PR-AUC 0.115 and 1.46x reversal precision lift, supporting the diagnosis that
market information exists but the rare class is suppressed by the current multiclass objective. The
reported v0.4 multiclass ROC-AUC=0.5 is invalid because the metric path fell back to 0.5 after a label-order
ValueError; do not use that field.

The current frozen task is Market Observer v0.5. It changes the learning architecture only; sampling,
features and market labels remain exactly v0.4:
- Stage 1 target is binary REAL_REVERSAL vs NOT_REVERSAL (NOISE + CORRECTION);
- primary Stage 1 model is CatBoost Logloss with the same 300 iterations, depth=6, learning_rate=0.05,
  l2_leaf_reg=5 and random_seed=42, plus auto_class_weights=Balanced;
- a second unweighted binary CatBoost with identical parameters is evaluated as a control so the effect
  of class balancing is measured directly;
- Stage 2 is a separate unweighted CatBoost Logloss model trained only on NOT_REVERSAL samples to estimate
  CORRECTION vs NOISE;
- no Focal Loss, SMOTE, random undersampling or new features are allowed in v0.5;
- weighted-model raw scores are calibrated with a Platt-style LogisticRegression on a dedicated historical
  calibration window that is never used to fit the CatBoost model;
- each test fold reserves the 60 days immediately before a later 60-day threshold-validation window for
  calibration; both windows are strictly before the test year and labels must be fully known;
- the 8-hour label embargo remains in force between historical validation and each test year;
- the diagnostic REAL_REVERSAL hard threshold is chosen by maximum F1 only on the dedicated historical
  threshold-validation window; no threshold is selected from 2024/2025 test outcomes;
- the observer's primary output remains calibrated probabilities, not a hard action;
- final probabilities are composed hierarchically:
  P(REAL_REVERSAL)=Stage1,
  P(CORRECTION)=(1-P(REAL_REVERSAL))*P(CORRECTION|NOT_REVERSAL),
  P(NOISE)=(1-P(REAL_REVERSAL))*(1-P(CORRECTION|NOT_REVERSAL));
- report weighted and unweighted reversal ROC-AUC, PR-AUC, PR-AUC lift, Brier, log loss, calibration error,
  probability quantiles, top 1/5/10/20% reversal-rate lift, past-only operating-point precision/recall/F1,
  Stage 2 metrics, hierarchical per-class metrics and feature-importance stability;
- v0.5 remains market-only: no trade entry, stop, target, PnL, R, MFE/MAE, risk, fee or slippage inputs;
- 2026 remains untouched;
- v0.5 must not emit BUY/SELL/HOLD/EXIT decisions and must not be evaluated by trading PnL.

Market Observer v0.5 showed that class imbalance is not the primary bottleneck. Combined weighted
binary CatBoost improved only modestly over the identical unweighted control: ROC-AUC 0.5652 vs 0.5636
and PR-AUC 0.1069 vs 0.0992. The advantage was not stable across both years: weighted was better in 2024
(ROC-AUC 0.5709 / PR-AUC 0.0914 vs 0.5574 / 0.0863), while unweighted was slightly better in 2025
(0.5750 / 0.1268 vs weighted 0.5718 / 0.1239). The separate NOISE-vs-CORRECTION stage was approximately
random in both years at ROC-AUC about 0.514, so do not continue that second-stage classifier. The useful
remaining signal is ranking: the weighted model concentrated REAL_REVERSAL meaningfully in the highest
probability tail, but not strongly enough for a reliable observer.

The current frozen task is Market Observer v0.6. It tests whether genuine Bybit taker trade flow adds
incremental information to the same binary REAL_REVERSAL-vs-TREND_SURVIVES observer:
- sampling, H4 trend regime, structural event generation and v0.4 labels are unchanged;
- the base v0.4 feature set is unchanged;
- the model remains the v0.5 weighted binary CatBoost with the same parameters, chronological fit,
  60-day calibration window, later 60-day threshold-validation window, 8-hour embargo and 2024/2025
  development folds;
- the v0.5 weighted binary observer without trade flow is rerun inside v0.6 as the exact control;
- the only experimental change is adding genuine taker-flow features from Bybit public historical trades;
- source path is https://public.bybit.com/trading/BTCUSDT/ with daily gzip CSV archives;
- use 2023-01-01 through 2025-12-31 only; never download or use 2026 for v0.6;
- downloader streams raw gzip archives and persists only completed UTC M5 aggregates; raw ticks are not
  stored by this project;
- each completed day is cached as aggregated M5 data so interrupted multi-year downloads can resume;
- require exactly 288 contiguous M5 flow buckets per UTC day; never synthesize missing trade-flow buckets;
- archive fields required are timestamp, symbol, side, size and price;
- Buy/Sell is interpreted as taker/aggressor side, consistent with Bybit public trade semantics;
- M5 aggregates store taker buy/sell quantity, taker buy/sell notional and buy/sell trade counts;
- v0.6 adds only these 15 causal features:
  * taker delta ratio over 5m / 15m / 30m / 2h;
  * taker notional delta ratio over 5m / 30m;
  * taker trade-count imbalance over 5m / 30m;
  * trend-aligned taker delta ratio over 5m / 30m / 2h;
  * 5m-vs-30m delta acceleration;
  * current 5m trade-count activity versus the 2h average;
  * average buy-vs-sell trade-size imbalance on the completed event M5;
  * 30m persistence of aggressive flow with or against the prevailing H4 trend;
- an event at time T may use only trade-flow buckets whose M5 interval is fully completed by T;
- do not add order-book, liquidation, new candle indicators, new labels or model tuning in v0.6;
- compare baseline-v0.5 and flow-enhanced models on identical samples and splits;
- primary evidence is fold-by-fold and combined delta in ROC-AUC, PR-AUC, PR-AUC lift, Brier/ECE and
  top-probability reversal concentration;
- flow is useful only if ranking improvement is directionally consistent in both 2024 and 2025, not merely
  positive in the combined aggregate;
- report all flow-feature importances so we can see whether CatBoost actually uses the new information;
- v0.6 remains an observer research task only and emits no trading action or PnL conclusion.

Market Observer v0.6 showed that genuine taker flow adds some information, but not with
the yearly consistency required by the frozen robustness rule. Combined flow-enhanced ROC-AUC improved
from 0.5652 to 0.5711 and PR-AUC from 0.1069 to 0.1122; PR-AUC lift improved from 1.294x to 1.358x.
The top 5% and top 10% tails also improved materially. However, 2024 ROC-AUC fell by about 0.0104 while
2025 improved by about 0.0141. The flow features are therefore retained as useful market context, but
v0.6 does not establish that first-disturbance prediction is robust enough.

The current frozen task is Market Observer v0.7. It changes only the prediction moment:
- keep the original v0.4/v0.6 structural disturbance as the episode anchor;
- do not predict at the disturbance close;
- wait exactly 3 fully completed M5 bars / 15 wall-clock minutes after the disturbance;
- the v0.6 market + taker-flow feature vector is preserved unchanged and remains visible to the model;
- add only causal confirmation-state features from those 3 completed M5 bars and matching completed
  taker-flow buckets;
- the control model is evaluated on the exact same delayed samples and chronological splits but receives
  only the original v0.6 features; this isolates the information value of waiting for confirmation;
- the experimental model receives the same v0.6 features plus confirmation features;
- labels remain the original disturbance outcome labels. REAL_REVERSAL still means the original structural
  episode eventually satisfies the frozen v0.4 reversal definition; NOISE/CORRECTION together remain
  TREND_SURVIVES for the binary target;
- because prediction is intentionally delayed, market information observed inside the first 15 minutes is
  valid input, not leakage. No candle or trade-flow bucket ending after T+15m may enter any feature;
- CatBoost parameters, Balanced class weighting, Platt calibration, 60-day calibration window, later 60-day
  threshold-validation window, 8-hour embargo and 2024/2025 development folds remain unchanged;
- 2026 remains untouched;
- v0.7 adds exactly these confirmation features:
  * adverse displacement from disturbance close to T+15m in event ATR;
  * maximum adverse extension beyond the defended level;
  * mean close distance beyond the defended level;
  * count of closes and touches beyond the defended level;
  * final reclaim flag and reclaim speed;
  * retest-happened and retest-held flags using the frozen 0.15 ATR tolerance;
  * 15-minute post-event price-path efficiency and range in ATR;
  * fraction of adverse-direction candle bodies;
  * post/pre 15-minute candle-volume ratio;
  * post-event taker delta, notional delta and trade-count imbalance over 15m;
  * trend-aligned post-event taker delta;
  * change in 15-minute taker delta versus the pre-event 15 minutes;
  * post-event aggressive-flow persistence;
  * post-event trade-count activity versus the pre-event 2-hour average;
  * post-event average buy-vs-sell trade-size imbalance;
- do not add order-book, liquidation data, alternate confirmation windows, regime interactions, YetiRank,
  PairLogit, new labels or new CatBoost tuning in v0.7;
- primary comparison is delayed control vs confirmed-event observer on identical samples;
- report fold-by-fold and combined ROC-AUC, PR-AUC, PR-AUC lift, Brier/ECE, top-tail concentration,
  past-only operating-point metrics, confirmation-feature importance and feature-rank stability;
- v0.7 passes its feasibility purpose only if confirmation improves reversal ranking directionally in both
  2024 and 2025, not merely in the combined aggregate;
- v0.7 remains an observer research task and emits no BUY/SELL/HOLD/EXIT action and no PnL conclusion.

Market Observer v0.7 passed its frozen full-sample feasibility gate by a wide margin.
Combined confirmed-event ROC-AUC was 0.8463 versus delayed-control 0.5711; PR-AUC was 0.2740 versus
0.1122; PR-AUC lift was 3.316x versus 1.358x. The improvement repeated in both development folds:
2024 confirmed ROC-AUC 0.8539 / PR-AUC 0.2676 and 2025 confirmed ROC-AUC 0.8393 / PR-AUC 0.2848.
The dominant feature was 15-minute adverse displacement after the disturbance, followed by closes and
mean close distance beyond the defended level and post-event path efficiency. This strongly supports the
hypothesis that the earlier observer was evaluating the market too early.

However, v0.7 is not promoted yet. The current frozen task is the v0.7 label-overlap audit:
- do not alter or retrain the v0.7 model, feature set, label definitions, 15-minute window, calibration,
  thresholds or chronological folds;
- reproduce the exact frozen v0.7 delayed-control and confirmed models;
- for every original disturbance, determine whether its frozen label was already fully knowable at T+15m;
- NOISE is resolved at T+15 only when the +0.75 ATR with-trend barrier was reached before the -0.75 ATR
  adverse barrier inside the first three completed post-disturbance M5 bars;
- CORRECTION is resolved at T+15 only when the -0.75 ATR adverse barrier occurred first and the +0.75 ATR
  with-trend recovery barrier also occurred inside those same three completed M5 bars;
- same-bar trend/adverse barrier touch is resolved as AMBIGUOUS, matching the original label semantics;
- REAL_REVERSAL is never considered resolved at T+15 because the frozen label requires no +0.75 ATR
  trend recovery across the full 8-hour horizon;
- mark every clear test sample as resolved-before-prediction or unresolved-at-prediction;
- report counts and class distributions for both groups in 2024, 2025 and combined;
- keep the trained v0.7 models unchanged and recalculate metrics only on the unresolved test subset;
- report delayed-control and confirmed ROC-AUC, PR-AUC, PR-AUC lift, Brier/ECE, top-tail concentration
  and frozen-threshold operating-point metrics on the unresolved subset;
- no recalibration is allowed on the unresolved subset; its shifted class prevalence must remain visible;
- 2026 remains untouched;
- this audit is diagnostic only and emits no trading action or PnL conclusion.

The v0.7 label-overlap audit PASSED. Of 12,369 clear 2024-2025 samples, 6,078 (49.1%) were
already resolved at T+15 and 6,291 remained unresolved. On the unresolved-only subset, without retraining,
recalibration or threshold changes, confirmed v0.7 retained ROC-AUC 0.7065 and PR-AUC 0.2875 versus a
16.25% reversal base rate (1.77x PR lift); the delayed control was ROC-AUC 0.5567 / PR-AUC 0.2031.
The effect remained positive in both years: unresolved 2024 confirmed ROC-AUC 0.7193 / PR-AUC 0.2812
versus control 0.5459 / 0.1740, and unresolved 2025 confirmed ROC-AUC 0.6948 / PR-AUC 0.2981 versus
control 0.5723 / 0.2260. The full-sample 0.846 result was therefore partly inflated by easy already-resolved
NOISE/CORRECTION cases, but the confirmation signal survives their complete removal.

The one-shot untouched 2026 forward validation PASSED. On 2,379 unresolved-at-T+15 samples,
frozen v0.7 achieved ROC-AUC 0.7278 and PR-AUC 0.3015 versus a 14.12% base rate (2.135x PR lift),
with top-10% reversal-rate lift 2.439x. The delayed control was ROC-AUC 0.5627 / PR-AUC 0.1718.
All six pre-frozen forward conditions passed. Full 2026 clear-sample ROC-AUC was 0.8577. v0.7 is now
frozen as forward-validated observer evidence. 2026 is observed evidence from this point onward and must
not be described as untouched again.

Observer -> LONG Trading Policy v1 FAILED its frozen policy-value gate, while still showing that observer
messages contain some economic value. On 279 fixed trades, baseline net R was -21.45R and v1 managed
net R was -19.01R, a +2.44R improvement. 146 observer exits produced +2.44R total delta net R; 98 exits
improved their trade, 48 worsened it, 91 avoided a later baseline stop and 32 prematurely exited trades
that later reached the 2R target. The failure was structural rather than a failure of Market Observer v0.7:
managed PF fell from 0.890 to 0.864, max drawdown rose slightly from 13.39% to 13.59%, 2024 added
+6.17R but 2025 lost -3.74R. Classification threshold therefore must not be equated with an immediate
full-position exit threshold.

Observer -> LONG Trading Policy v2 PASSED its frozen 2024-2025 development gate. On the same 279
economic-pass Trend LONG v1.1 trades, baseline was -21.45R / PF 0.890 / max drawdown 13.39%, frozen v1
was -19.01R / PF 0.864 / drawdown 13.59%, and v2 improved to -17.08R / PF 0.905 / drawdown 11.88%.
v2 added +4.37R versus baseline, reduced observer exits from 146 to 34 and premature exits before later
2R targets from 32 to 6. Confirmed exits added +4.37R total. The effect was positive in both development
years: +2.63R in 2024 and +1.74R in 2025. 2x-cost net R also improved versus both baseline and v1.
Market Observer v0.7 and LongObserverPolicy v2 are now frozen.

The current frozen task is the first and only 2026 trading-outcome validation of the complete stack:
Trend LONG v1.1 entry bot + forward-validated Market Observer v0.7 + frozen LongObserverPolicy v2.
2026 observer labels have already been observed during observer validation, but 2026 trading outcomes have
not been used to design or tune the entry strategy, observer threshold or v2 policy:
- do not change Trend LONG v1.1 entry logic, economic gate, stop rule, 2R target or 24-hour holding horizon;
- do not change Market Observer v0.7, its features, calibration, model parameters or pre-2026 threshold rule;
- do not change LongObserverPolicy v2 warning count, reset semantics or exit execution;
- fixed trading entry window is 2026-01-01T00:00:00Z inclusive through 2026-09-28T00:00:00Z exclusive;
- September 28-29 data is horizon-only data so every allowed entry has a complete 24-hour trade path and
  enough future market data for the historical observer event extractor; do not admit later entries;
- forward M30 data must be downloaded separately with warmup from 2025-11-01 through 2026-09-29;
- reuse the existing separate forward2026 M5/OI/account-ratio/funding/trade-flow bundle;
- observer snapshots for 2026 must be produced by the same chronological v0.7 runtime with all model fit,
  Platt calibration and threshold selection restricted to pre-2026 development samples;
- inference must still be emitted for every 2026 structural event, including events whose research label is
  AMBIGUOUS; never filter live-like messages by future outcome labels;
- candidate trades are exactly the economic-pass Trend LONG v1.1 executed candidates whose entry times fall
  inside the frozen trading entry window;
- execution remains native M5; confirmed observer exits occur at the next M5 open with stop-gap/target-gap
  priority;
- no entry blocking and no replacement trades are introduced in this validation;
- starting equity remains USD 1,000 and risk remains 0.5% of current equity per trade;
- fees remain 5.5 bps/side and slippage 2.0 bps/side; also report the frozen 2x-cost stress;
- report baseline bot versus bot+observer v2 on identical entries: trades, gross/net R, PF, win rate,
  drawdown, final compounded equity, 2x-cost metrics, warning/confirmed-exit counts, avoided stops,
  premature target exits, monthly results and incremental exit value;
- the one-shot full-stack trading gate is frozen before reading 2026 trading PnL:
  * at least 30 managed trades;
  * bot+v2 net R > 0 after modeled costs;
  * bot+v2 PF > 1.0;
  * bot+v2 net R must exceed the same baseline bot;
  * bot+v2 PF must exceed the same baseline bot;
  * bot+v2 max drawdown must not exceed baseline max drawdown;
  * confirmed v2 exits must add positive total delta net R;
  * bot+v2 2x-cost net R > 0;
  * bot+v2 2x-cost PF > 1.0;
- PASS means this frozen bot+observer stack produced positive net edge after modeled costs on the external
  2026 trading window, improved the same entries, and survived 2x modeled costs;
- FAIL must be reported as FAIL. Do not tune any component on this 2026 PnL window and rerun it as if it
  were still external validation;
- after the first result, 2026 trading outcomes are observed evidence and cannot be described as untouched
  trading validation again.

The frozen 2026 full-stack trading test FAILED. On 89 economic-pass entries, the baseline Trend LONG v1.1
stream was already negative before costs: gross -8.22R, gross PF 0.859. After modeled 5.5 bps/side fees
plus 2.0 bps/side slippage it produced -24.91R, PF 0.643 and 12.10% max drawdown. Frozen Observer v0.7
+ Policy v2 did not rescue the entry stream: managed gross -9.00R, net -25.69R, PF 0.631, 12.44% drawdown,
and final USD 1,000 equity 877.76. At 2x costs managed net was -42.37R and final equity 807.37.
Only 2 confirmed observer exits occurred; their total incremental value was -0.78R. The forward trading
gate failed every condition except minimum trade count. 2026 entry/trading outcomes are now observed and
must never be used as a clean forward window again.

The current frozen task is BTC Trend LONG Entry Strategy v2. The goal is to prove or reject information
in the entry itself before any further Observer/Policy work:
- Market Observer v0.7 and LongObserverPolicy v2 remain frozen and are not used in this experiment;
- 2026 must not be used for parameter selection, filtering, threshold selection or v2 evaluation;
- development data is frozen to 2023-01-01 through 2025-12-31;
- baseline is the exact economic-pass Trend LONG v1.1 executed stream;
- v2 is allowed exactly one structural modification: persistent H4 trend confirmation;
- the M30 setup is unchanged: bearish M30 pullback followed by bullish confirmation close above pullback high;
- the ATR14 minimum stop distance, 2R target, 24-hour holding horizon, fees and slippage are unchanged;
- persistent H4 confirmation reuses the existing 3-H4-bar lookback and adds no fitted numeric threshold:
  * EMA20 must rise on each of the last 3 completed H4 steps;
  * each of the last 3 completed H4 closes must remain above its contemporaneous EMA20;
  * current completed H4 close must exceed the close 3 H4 bars ago;
- evaluate only economic-pass trades with target/expected-cost ratio >= 8;
- before judging PnL, compare actual entries with a deterministic matched-random control:
  * same calendar year, month and UTC hour where possible;
  * same causal H4 regime definition;
  * preserve each real trade's stop-distance fraction;
  * same 2R target and 24-hour horizon;
  * exclude real strategy entry timestamps from random candidates;
  * require >=90% matched-trade coverage and >=5 random matches per trade on average;
- report gross mean/R/PF, net R/PF, 2x-cost results, MFE/MAE, +1R/+2R-before-stop rates, stop-distance bins,
  yearly results, matched-random gross edge and bootstrap 95% confidence intervals;
- diagnostic stop-distance bins are fixed before the first v2 result: <0.8%, 0.8-1.0%, 1.0-1.5%, >=1.5%;
  they are descriptive only and must not become v2 filters after reading the result;
- the frozen Entry v2 information gate requires:
  * at least 150 economic-pass trades;
  * positive combined gross R;
  * gross PF >=1.10;
  * positive gross R in at least 2 of 3 development years;
  * matched-random coverage >=90%;
  * >=5 random matches per trade on average;
  * positive mean gross edge versus matched random;
  * bootstrap 95% lower bound of matched-random gross edge >0;
  * +1R-before-stop rate above matched random;
- the frozen Entry v2 economic gate additionally requires:
  * net R >0 and PF >1.0 after normal execution costs;
  * net R >0 and PF >1.0 at 2x execution costs;
  * v2 net R > baseline v1.1 net R;
  * v2 gross PF > baseline v1.1 gross PF;
- funding is intentionally not used to decide whether the entry contains information. If v2 first passes both
  information and execution-cost gates, add funding before any deployment claim;
- if v2 fails, do not mine 2026 for a stop-width/day-of-week/filter patch. Mark this entry hypothesis rejected
  and move to a materially different entry hypothesis or timeframe;
- if v2 passes, freeze it and start a new paper-forward validation from October 2026. The paper-forward sample
  size/gate must be fixed before observing results.

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
