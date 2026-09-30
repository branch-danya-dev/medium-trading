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
- Current research candidate: Bybit BTCUSDT Market Observer v0.5 for hierarchical rare-reversal observation.
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

Do not tune v0.5 after its first result. The main test is whether balanced binary CatBoost materially
improves reversal ranking/detection over its unweighted binary control in both 2024 and 2025 while
past-only calibration remains usable. If weighting does not improve the signal, reject class imbalance
as the primary explanation rather than escalating model complexity.

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
