from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from statistics import mean

from medium_trading.crypto_noise_filter import extract_btc_long_noise_samples
from medium_trading.data.bybit_state_history import (
    AccountRatioPoint,
    FundingPoint,
    OpenInterestPoint,
)
from medium_trading.domain import Candle
from medium_trading.strategy.crypto_trend_long import CryptoTrendLongStrategy

FEATURE_NAMES = (
    "m5_return_1_atr",
    "m5_return_3_atr",
    "m5_return_6_atr",
    "m5_return_12_atr",
    "m5_acceleration_3_atr",
    "m5_efficiency_12",
    "m5_bull_ratio_12",
    "m5_body_atr",
    "m5_upper_wick_atr",
    "m5_lower_wick_atr",
    "m5_volume_ratio_20",
    "m5_volume_acceleration",
    "m5_range_6_to_24",
    "m15_return_1_atr",
    "m15_return_4_atr",
    "m15_efficiency_8",
    "m30_return_1_atr",
    "m30_return_4_atr",
    "h1_return_1_atr",
    "h1_return_3_atr",
    "h4_return_1_atr",
    "h4_return_3_atr",
    "h4_ema20_distance_atr",
    "h4_ema20_slope_3_atr",
    "trade_current_r",
    "trade_mfe_so_far_r",
    "trade_mae_so_far_r",
    "trade_retrace_from_mfe_r",
    "trade_time_hours",
    "trade_distance_to_stop_r",
    "trade_distance_to_target_r",
    "oi_change_30m",
    "oi_change_2h",
    "long_ratio",
    "long_ratio_change_2h",
    "funding_rate",
    "oi_available",
    "ratio_available",
    "funding_available",
)

SNAPSHOT_MINUTES = 15
LABEL_HORIZON_HOURS = 8
REGRESSION_HORIZON_HOURS = 2
LOCAL_MOVE_R = 0.5
STRESS_RETRACE_R = 0.15
STRESS_CURRENT_R = -0.05
FIRST_TEST_YEAR = 2024
LAST_TEST_YEAR = 2025
REVERSAL_THRESHOLD = 0.50

CLASSIFIER_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}

REGRESSOR_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


@dataclass(frozen=True, slots=True)
class MarketStateSample:
    trade_entry_time: datetime
    snapshot_time: datetime
    label_end_time: datetime
    features: tuple[float, ...]
    state_class: str
    is_reversal: bool
    future_mfe_2h_r: float
    future_mae_2h_r: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    samples: int
    actual_reversal: int
    predicted_reversal: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    base_reversal_rate: float
    precision: float
    recall: float
    precision_lift: float
    roc_auc: float
    brier_score: float


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    samples: int
    mfe_mae: float
    mfe_naive_mae: float
    mae_mae: float
    mae_naive_mae: float


@dataclass(frozen=True, slots=True)
class MarketStateFold:
    test_year: int
    train_samples: int
    test_samples: int
    train_state_distribution: dict[str, int]
    test_state_distribution: dict[str, int]
    classification: ClassificationMetrics
    regression: RegressionMetrics


@dataclass(frozen=True, slots=True)
class MarketStateEvaluation:
    feature_names: tuple[str, ...]
    classifier_params: dict[str, object]
    regressor_params: dict[str, object]
    reversal_threshold: float
    folds: tuple[MarketStateFold, ...]
    combined_classification: ClassificationMetrics
    combined_regression: RegressionMetrics
    combined_state_distribution: dict[str, int]


def extract_market_state_samples(
    *,
    candles_30m: tuple[Candle, ...],
    candles_5m: tuple[Candle, ...],
    open_interest: tuple[OpenInterestPoint, ...],
    account_ratio: tuple[AccountRatioPoint, ...],
    funding: tuple[FundingPoint, ...],
    symbol: str = "BTCUSDT",
    fee_bps_per_side: float = 5.5,
    slippage_bps_per_side: float = 2.0,
    starting_equity: float = 10_000.0,
    risk_fraction: float = 0.005,
) -> tuple[MarketStateSample, ...]:
    if not candles_5m:
        raise ValueError("market-state extraction requires M5 candles")
    _require_contiguous(candles_5m, timedelta(minutes=5), "M5")

    baseline = extract_btc_long_noise_samples(
        candles=candles_30m,
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    trades = tuple(sample.trade for sample in baseline if sample.economic_pass)
    if not trades:
        raise ValueError("market-state extraction found no economic-pass trades")

    m15 = _aggregate(candles_5m, 15)
    m30 = _aggregate(candles_5m, 30)
    h1 = _aggregate(candles_5m, 60)
    h4 = _aggregate(candles_5m, 240)

    m5_times = [candle.timestamp for candle in candles_5m]
    oi_times = [point.timestamp for point in open_interest]
    ratio_times = [point.timestamp for point in account_ratio]
    funding_times = [point.timestamp for point in funding]

    samples: list[MarketStateSample] = []
    for trade in trades:
        risk_distance = trade.entry - trade.stop
        if risk_distance <= 0:
            continue

        entry_index = bisect_left(m5_times, trade.entry_time)
        if entry_index >= len(candles_5m):
            continue

        exit_limit = min(
            trade.exit_time,
            trade.entry_time + timedelta(hours=24),
            candles_5m[-1].timestamp,
        )
        snapshot_time = trade.entry_time + timedelta(minutes=SNAPSHOT_MINUTES)

        while snapshot_time <= exit_limit:
            if snapshot_time + timedelta(hours=LABEL_HORIZON_HOURS) > (
                candles_5m[-1].timestamp + timedelta(minutes=5)
            ):
                break
            end_index = bisect_left(m5_times, snapshot_time)
            if end_index <= entry_index:
                snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)
                continue

            price = candles_5m[end_index - 1].close
            path = candles_5m[entry_index:end_index]
            mfe_so_far_r = (
                max(candle.high for candle in path) - trade.entry
            ) / risk_distance
            mae_so_far_r = (
                trade.entry - min(candle.low for candle in path)
            ) / risk_distance
            current_r = (price - trade.entry) / risk_distance
            retrace_from_mfe_r = mfe_so_far_r - current_r

            if (
                retrace_from_mfe_r < STRESS_RETRACE_R
                and current_r > STRESS_CURRENT_R
            ):
                snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)
                continue

            features = _features(
                snapshot_time=snapshot_time,
                snapshot_price=price,
                trade_entry_time=trade.entry_time,
                trade_entry=trade.entry,
                trade_stop=trade.stop,
                risk_distance=risk_distance,
                current_r=current_r,
                mfe_so_far_r=mfe_so_far_r,
                mae_so_far_r=mae_so_far_r,
                retrace_from_mfe_r=retrace_from_mfe_r,
                candles_5m=candles_5m,
                m15=m15,
                m30=m30,
                h1=h1,
                h4=h4,
                open_interest=open_interest,
                account_ratio=account_ratio,
                funding=funding,
                oi_times=oi_times,
                ratio_times=ratio_times,
                funding_times=funding_times,
            )
            if features is None:
                snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)
                continue

            state_class = _future_state_class(
                candles_5m=candles_5m,
                m5_times=m5_times,
                snapshot_time=snapshot_time,
                snapshot_price=price,
                risk_distance=risk_distance,
            )
            future_mfe, future_mae = _future_excursions(
                candles_5m=candles_5m,
                m5_times=m5_times,
                snapshot_time=snapshot_time,
                snapshot_price=price,
                risk_distance=risk_distance,
                horizon_hours=REGRESSION_HORIZON_HOURS,
            )
            samples.append(
                MarketStateSample(
                    trade_entry_time=trade.entry_time,
                    snapshot_time=snapshot_time,
                    label_end_time=snapshot_time
                    + timedelta(hours=LABEL_HORIZON_HOURS),
                    features=features,
                    state_class=state_class,
                    is_reversal=state_class == "REVERSAL",
                    future_mfe_2h_r=future_mfe,
                    future_mae_2h_r=future_mae,
                )
            )

            snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)

    return tuple(samples)


def evaluate_market_state_v01(
    samples: Iterable[MarketStateSample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> MarketStateEvaluation:
    classifier_type, regressor_type, metrics_module = _load_ml()
    all_samples = tuple(sorted(samples, key=lambda item: item.snapshot_time))
    if not all_samples:
        raise ValueError("market-state evaluation requires samples")

    folds: list[MarketStateFold] = []
    combined_truth: list[bool] = []
    combined_probabilities: list[float] = []
    combined_mfe_truth: list[float] = []
    combined_mfe_pred: list[float] = []
    combined_mae_truth: list[float] = []
    combined_mae_pred: list[float] = []
    combined_mfe_naive: list[float] = []
    combined_mae_naive: list[float] = []
    combined_test: list[MarketStateSample] = []

    for year in range(first_test_year, last_test_year + 1):
        test_start = datetime(year, 1, 1, tzinfo=UTC)
        test_end = datetime(year + 1, 1, 1, tzinfo=UTC)
        train = tuple(
            sample
            for sample in all_samples
            if sample.label_end_time <= test_start
        )
        test = tuple(
            sample
            for sample in all_samples
            if test_start <= sample.snapshot_time < test_end
        )
        if not test:
            continue
        if len(train) < 500:
            raise ValueError(
                f"not enough pre-{year} market-state samples: "
                f"{len(train)}; need at least 500"
            )

        train_y = [sample.is_reversal for sample in train]
        if len(set(train_y)) < 2:
            raise ValueError(f"pre-{year} reversal labels contain one class")

        classifier = classifier_type(**CLASSIFIER_PARAMS)
        mfe_model = regressor_type(**REGRESSOR_PARAMS)
        mae_model = regressor_type(**REGRESSOR_PARAMS)
        train_x = [sample.features for sample in train]
        test_x = [sample.features for sample in test]

        classifier.fit(train_x, train_y)
        mfe_model.fit(train_x, [sample.future_mfe_2h_r for sample in train])
        mae_model.fit(train_x, [sample.future_mae_2h_r for sample in train])

        probabilities = _positive_probabilities(classifier, test_x)
        mfe_pred = tuple(float(value) for value in mfe_model.predict(test_x))
        mae_pred = tuple(float(value) for value in mae_model.predict(test_x))
        truth = tuple(sample.is_reversal for sample in test)
        mfe_truth = tuple(sample.future_mfe_2h_r for sample in test)
        mae_truth = tuple(sample.future_mae_2h_r for sample in test)
        mfe_naive_value = mean(sample.future_mfe_2h_r for sample in train)
        mae_naive_value = mean(sample.future_mae_2h_r for sample in train)

        folds.append(
            MarketStateFold(
                test_year=year,
                train_samples=len(train),
                test_samples=len(test),
                train_state_distribution=_state_distribution(train),
                test_state_distribution=_state_distribution(test),
                classification=_classification_metrics(
                    truth,
                    probabilities,
                    metrics_module,
                ),
                regression=_regression_metrics(
                    mfe_truth=mfe_truth,
                    mfe_pred=mfe_pred,
                    mae_truth=mae_truth,
                    mae_pred=mae_pred,
                    mfe_naive=tuple(mfe_naive_value for _ in test),
                    mae_naive=tuple(mae_naive_value for _ in test),
                ),
            )
        )

        combined_truth.extend(truth)
        combined_probabilities.extend(probabilities)
        combined_mfe_truth.extend(mfe_truth)
        combined_mfe_pred.extend(mfe_pred)
        combined_mae_truth.extend(mae_truth)
        combined_mae_pred.extend(mae_pred)
        combined_mfe_naive.extend(mfe_naive_value for _ in test)
        combined_mae_naive.extend(mae_naive_value for _ in test)
        combined_test.extend(test)

    if not folds:
        raise ValueError("no market-state folds contained test samples")

    return MarketStateEvaluation(
        feature_names=FEATURE_NAMES,
        classifier_params=dict(CLASSIFIER_PARAMS),
        regressor_params=dict(REGRESSOR_PARAMS),
        reversal_threshold=REVERSAL_THRESHOLD,
        folds=tuple(folds),
        combined_classification=_classification_metrics(
            tuple(combined_truth),
            tuple(combined_probabilities),
            metrics_module,
        ),
        combined_regression=_regression_metrics(
            mfe_truth=tuple(combined_mfe_truth),
            mfe_pred=tuple(combined_mfe_pred),
            mae_truth=tuple(combined_mae_truth),
            mae_pred=tuple(combined_mae_pred),
            mfe_naive=tuple(combined_mfe_naive),
            mae_naive=tuple(combined_mae_naive),
        ),
        combined_state_distribution=_state_distribution(combined_test),
    )


def evaluation_payload(evaluation: MarketStateEvaluation) -> dict[str, object]:
    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "in-trade market-state / noise-vs-reversal feasibility",
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [
                fold.test_year for fold in evaluation.folds
            ],
            "2026_used": False,
            "snapshot_minutes": SNAPSHOT_MINUTES,
            "stress_snapshot_rule": (
                f"retrace_from_mfe >= {STRESS_RETRACE_R:.2f}R "
                f"or current_r <= {STRESS_CURRENT_R:.2f}R"
            ),
        },
        "labels": {
            "local_move_r": LOCAL_MOVE_R,
            "label_horizon_hours": LABEL_HORIZON_HOURS,
            "classes": {
                "TREND_VALID": "+0.5R reached before -0.5R",
                "NOISE_PULLBACK": (
                    "-0.5R reached first, then +0.5R recovered within 8h"
                ),
                "REVERSAL": (
                    "-0.5R reached and +0.5R not recovered within 8h"
                ),
                "STALL": "neither +/-0.5R reached within 8h",
            },
            "binary_classifier": "REVERSAL vs all other states",
            "regression": [
                "future MFE over 2h in original trade R",
                "future MAE over 2h in original trade R",
            ],
        },
        "feature_names": list(evaluation.feature_names),
        "classifier": {
            "type": "HistGradientBoostingClassifier",
            "params": evaluation.classifier_params,
            "reversal_threshold": evaluation.reversal_threshold,
        },
        "regressors": {
            "type": "HistGradientBoostingRegressor",
            "params": evaluation.regressor_params,
        },
        "folds": [
            {
                "test_year": fold.test_year,
                "train_samples": fold.train_samples,
                "test_samples": fold.test_samples,
                "train_state_distribution": fold.train_state_distribution,
                "test_state_distribution": fold.test_state_distribution,
                "classification": asdict(fold.classification),
                "regression": asdict(fold.regression),
            }
            for fold in evaluation.folds
        ],
        "combined": {
            "state_distribution": evaluation.combined_state_distribution,
            "classification": asdict(evaluation.combined_classification),
            "regression": asdict(evaluation.combined_regression),
        },
    }


def _features(
    *,
    snapshot_time: datetime,
    snapshot_price: float,
    trade_entry_time: datetime,
    trade_entry: float,
    trade_stop: float,
    risk_distance: float,
    current_r: float,
    mfe_so_far_r: float,
    mae_so_far_r: float,
    retrace_from_mfe_r: float,
    candles_5m: tuple[Candle, ...],
    m15: tuple[Candle, ...],
    m30: tuple[Candle, ...],
    h1: tuple[Candle, ...],
    h4: tuple[Candle, ...],
    open_interest: tuple[OpenInterestPoint, ...],
    account_ratio: tuple[AccountRatioPoint, ...],
    funding: tuple[FundingPoint, ...],
    oi_times: list[datetime],
    ratio_times: list[datetime],
    funding_times: list[datetime],
) -> tuple[float, ...] | None:
    m5_hist = _history_before(candles_5m, snapshot_time, 60)
    m15_hist = _history_completed(m15, snapshot_time, 15, 16)
    m30_hist = _history_completed(m30, snapshot_time, 30, 10)
    h1_hist = _history_completed(h1, snapshot_time, 60, 8)
    h4_hist = _history_completed(h4, snapshot_time, 240, 24)
    if (
        len(m5_hist) < 51
        or len(m15_hist) < 9
        or len(m30_hist) < 5
        or len(h1_hist) < 4
        or len(h4_hist) < 21
    ):
        return None

    atr5 = _atr(m5_hist, 14)
    atr15 = _atr(m15_hist, 8)
    atr30 = _atr(m30_hist, 4)
    atr1h = _atr(h1_hist, 3)
    atr4h = _atr(h4_hist, 14)
    if min(atr5, atr15, atr30, atr1h, atr4h) <= 0:
        return None

    m5_closes = [candle.close for candle in m5_hist]
    latest = m5_hist[-1]
    upper_wick = latest.high - max(latest.open, latest.close)
    lower_wick = min(latest.open, latest.close) - latest.low

    recent_volume = mean(candle.volume for candle in m5_hist[-5:])
    prior_volume = mean(candle.volume for candle in m5_hist[-10:-5])
    baseline_volume = mean(candle.volume for candle in m5_hist[-21:-1])
    volume_ratio = latest.volume / baseline_volume if baseline_volume > 0 else 1.0
    volume_acceleration = (
        recent_volume / prior_volume if prior_volume > 0 else 1.0
    )

    range_6 = (
        max(candle.high for candle in m5_hist[-6:])
        - min(candle.low for candle in m5_hist[-6:])
    )
    range_24 = (
        max(candle.high for candle in m5_hist[-24:])
        - min(candle.low for candle in m5_hist[-24:])
    )
    range_ratio = range_6 / range_24 if range_24 > 0 else 0.0

    h4_closes = tuple(candle.close for candle in h4_hist)
    ema20 = CryptoTrendLongStrategy._ema_series(h4_closes, 20)

    oi_now, oi_available = _latest_value(
        open_interest,
        oi_times,
        snapshot_time,
        "open_interest",
    )
    oi_30m, _ = _latest_value(
        open_interest,
        oi_times,
        snapshot_time - timedelta(minutes=30),
        "open_interest",
    )
    oi_2h, _ = _latest_value(
        open_interest,
        oi_times,
        snapshot_time - timedelta(hours=2),
        "open_interest",
    )
    oi_change_30m = _fraction_change(oi_now, oi_30m)
    oi_change_2h = _fraction_change(oi_now, oi_2h)

    long_ratio, ratio_available = _latest_value(
        account_ratio,
        ratio_times,
        snapshot_time,
        "buy_ratio",
    )
    long_ratio_2h, _ = _latest_value(
        account_ratio,
        ratio_times,
        snapshot_time - timedelta(hours=2),
        "buy_ratio",
    )
    long_ratio_change = long_ratio - long_ratio_2h

    funding_rate, funding_available = _latest_value(
        funding,
        funding_times,
        snapshot_time,
        "funding_rate",
    )

    target_price = trade_entry + 2.0 * risk_distance
    return (
        (m5_closes[-1] - m5_closes[-2]) / atr5,
        (m5_closes[-1] - m5_closes[-4]) / atr5,
        (m5_closes[-1] - m5_closes[-7]) / atr5,
        (m5_closes[-1] - m5_closes[-13]) / atr5,
        (
            (m5_closes[-1] - m5_closes[-4])
            - (m5_closes[-4] - m5_closes[-7])
        )
        / atr5,
        _efficiency_ratio(m5_closes[-13:]),
        sum(candle.close > candle.open for candle in m5_hist[-12:]) / 12.0,
        (latest.close - latest.open) / atr5,
        upper_wick / atr5,
        lower_wick / atr5,
        volume_ratio,
        volume_acceleration,
        range_ratio,
        (m15_hist[-1].close - m15_hist[-2].close) / atr15,
        (m15_hist[-1].close - m15_hist[-5].close) / atr15,
        _efficiency_ratio([candle.close for candle in m15_hist[-9:]]),
        (m30_hist[-1].close - m30_hist[-2].close) / atr30,
        (m30_hist[-1].close - m30_hist[-5].close) / atr30,
        (h1_hist[-1].close - h1_hist[-2].close) / atr1h,
        (h1_hist[-1].close - h1_hist[-4].close) / atr1h,
        (h4_hist[-1].close - h4_hist[-2].close) / atr4h,
        (h4_hist[-1].close - h4_hist[-4].close) / atr4h,
        (h4_hist[-1].close - ema20[-1]) / atr4h,
        (ema20[-1] - ema20[-4]) / atr4h,
        current_r,
        mfe_so_far_r,
        mae_so_far_r,
        retrace_from_mfe_r,
        (snapshot_time - trade_entry_time).total_seconds() / 3600.0,
        (snapshot_price - trade_stop) / risk_distance,
        (target_price - snapshot_price) / risk_distance,
        oi_change_30m,
        oi_change_2h,
        long_ratio,
        long_ratio_change,
        funding_rate,
        1.0 if oi_available else 0.0,
        1.0 if ratio_available else 0.0,
        1.0 if funding_available else 0.0,
    )


def _future_state_class(
    *,
    candles_5m: tuple[Candle, ...],
    m5_times: list[datetime],
    snapshot_time: datetime,
    snapshot_price: float,
    risk_distance: float,
) -> str:
    start = bisect_left(m5_times, snapshot_time)
    end = bisect_left(
        m5_times,
        snapshot_time + timedelta(hours=LABEL_HORIZON_HOURS),
    )
    up = snapshot_price + LOCAL_MOVE_R * risk_distance
    down = snapshot_price - LOCAL_MOVE_R * risk_distance
    adverse_seen = False

    for candle in candles_5m[start:end]:
        if candle.low <= down:
            adverse_seen = True
        if candle.high >= up:
            return "NOISE_PULLBACK" if adverse_seen else "TREND_VALID"

    return "REVERSAL" if adverse_seen else "STALL"


def _future_excursions(
    *,
    candles_5m: tuple[Candle, ...],
    m5_times: list[datetime],
    snapshot_time: datetime,
    snapshot_price: float,
    risk_distance: float,
    horizon_hours: int,
) -> tuple[float, float]:
    start = bisect_left(m5_times, snapshot_time)
    end = bisect_left(
        m5_times,
        snapshot_time + timedelta(hours=horizon_hours),
    )
    future = candles_5m[start:end]
    if not future:
        return 0.0, 0.0

    mfe = (max(candle.high for candle in future) - snapshot_price) / risk_distance
    mae = (snapshot_price - min(candle.low for candle in future)) / risk_distance
    return mfe, mae


def _aggregate(
    candles: tuple[Candle, ...],
    period_minutes: int,
) -> tuple[Candle, ...]:
    if period_minutes <= 0 or period_minutes % 5 != 0:
        raise ValueError("period_minutes must be a positive multiple of 5")
    expected = period_minutes // 5
    buckets: dict[datetime, list[Candle]] = {}

    for candle in candles:
        timestamp = candle.timestamp.astimezone(UTC)
        epoch_minutes = int(timestamp.timestamp() // 60)
        bucket_epoch = epoch_minutes - (epoch_minutes % period_minutes)
        bucket_start = datetime.fromtimestamp(bucket_epoch * 60, tz=UTC)
        buckets.setdefault(bucket_start, []).append(candle)

    result: list[Candle] = []
    for bucket_start in sorted(buckets):
        group = sorted(buckets[bucket_start], key=lambda item: item.timestamp)
        if len(group) != expected:
            continue
        if any(
            current.timestamp - previous.timestamp != timedelta(minutes=5)
            for previous, current in pairwise(group)
        ):
            continue
        result.append(
            Candle(
                timestamp=bucket_start,
                open=group[0].open,
                high=max(candle.high for candle in group),
                low=min(candle.low for candle in group),
                close=group[-1].close,
                volume=sum(candle.volume for candle in group),
            )
        )
    return tuple(result)


def _history_before(
    candles: tuple[Candle, ...],
    timestamp: datetime,
    count: int,
) -> tuple[Candle, ...]:
    times = [candle.timestamp for candle in candles]
    end = bisect_left(times, timestamp)
    return candles[max(0, end - count) : end]


def _history_completed(
    candles: tuple[Candle, ...],
    timestamp: datetime,
    period_minutes: int,
    count: int,
) -> tuple[Candle, ...]:
    ends = [
        candle.timestamp + timedelta(minutes=period_minutes)
        for candle in candles
    ]
    end = bisect_right(ends, timestamp)
    return candles[max(0, end - count) : end]


def _atr(candles: tuple[Candle, ...], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0
    ranges = []
    for index in range(len(candles) - period, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return sum(ranges) / period


def _efficiency_ratio(closes: list[float]) -> float:
    path = sum(
        abs(closes[index] - closes[index - 1])
        for index in range(1, len(closes))
    )
    return abs(closes[-1] - closes[0]) / path if path > 0 else 0.0


def _latest_value(
    points,
    times: list[datetime],
    timestamp: datetime,
    attribute: str,
) -> tuple[float, bool]:
    index = bisect_right(times, timestamp) - 1
    if index < 0:
        return 0.0, False
    return float(getattr(points[index], attribute)), True


def _fraction_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return current / previous - 1.0


def _classification_metrics(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
    metrics_module,
) -> ClassificationMetrics:
    predicted = tuple(
        probability >= REVERSAL_THRESHOLD
        for probability in probabilities
    )
    tp = sum(a and p for a, p in zip(truth, predicted, strict=True))
    fp = sum((not a) and p for a, p in zip(truth, predicted, strict=True))
    fn = sum(a and (not p) for a, p in zip(truth, predicted, strict=True))
    tn = sum((not a) and (not p) for a, p in zip(truth, predicted, strict=True))
    actual = tp + fn
    selected = tp + fp
    base = actual / len(truth) if truth else 0.0
    precision = tp / selected if selected else 0.0
    recall = tp / actual if actual else 0.0
    auc = (
        float(metrics_module.roc_auc_score(truth, probabilities))
        if len(set(truth)) > 1
        else 0.5
    )
    brier = float(metrics_module.brier_score_loss(truth, probabilities))

    return ClassificationMetrics(
        samples=len(truth),
        actual_reversal=int(actual),
        predicted_reversal=int(selected),
        true_positive=int(tp),
        false_positive=int(fp),
        false_negative=int(fn),
        true_negative=int(tn),
        base_reversal_rate=float(base),
        precision=float(precision),
        recall=float(recall),
        precision_lift=float(precision / base if base > 0 else 0.0),
        roc_auc=auc,
        brier_score=brier,
    )


def _regression_metrics(
    *,
    mfe_truth: tuple[float, ...],
    mfe_pred: tuple[float, ...],
    mae_truth: tuple[float, ...],
    mae_pred: tuple[float, ...],
    mfe_naive: tuple[float, ...],
    mae_naive: tuple[float, ...],
) -> RegressionMetrics:
    return RegressionMetrics(
        samples=len(mfe_truth),
        mfe_mae=_mean_absolute_error(mfe_truth, mfe_pred),
        mfe_naive_mae=_mean_absolute_error(mfe_truth, mfe_naive),
        mae_mae=_mean_absolute_error(mae_truth, mae_pred),
        mae_naive_mae=_mean_absolute_error(mae_truth, mae_naive),
    )


def _mean_absolute_error(
    truth: tuple[float, ...],
    predicted: tuple[float, ...],
) -> float:
    if not truth:
        return 0.0
    return sum(
        abs(actual - guess)
        for actual, guess in zip(truth, predicted, strict=True)
    ) / len(truth)


def _state_distribution(
    samples: Iterable[MarketStateSample],
) -> dict[str, int]:
    counts = Counter(sample.state_class for sample in samples)
    return {
        name: counts.get(name, 0)
        for name in ("TREND_VALID", "NOISE_PULLBACK", "REVERSAL", "STALL")
    }


def _require_contiguous(
    candles: tuple[Candle, ...],
    delta: timedelta,
    label: str,
) -> None:
    for previous, current in pairwise(candles):
        if current.timestamp - previous.timestamp != delta:
            raise ValueError(
                f"{label} history has a gap between "
                f"{previous.timestamp.isoformat()} and "
                f"{current.timestamp.isoformat()}"
            )


def _positive_probabilities(model, features: list[tuple[float, ...]]) -> tuple[float, ...]:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    positive_index = classes.index(True)
    return tuple(float(row[positive_index]) for row in probabilities)


def _load_ml():
    try:
        from sklearn import metrics
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'ML research dependencies are not installed. Run pip install -e ".[ml]".'
        ) from exc

    return HistGradientBoostingClassifier, HistGradientBoostingRegressor, metrics
