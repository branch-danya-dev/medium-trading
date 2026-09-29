from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from statistics import mean, pstdev

from medium_trading.crypto_noise_filter import extract_btc_long_noise_samples
from medium_trading.data.bybit_state_history import (
    AccountRatioPoint,
    FundingPoint,
    OpenInterestPoint,
)
from medium_trading.domain import Candle
from medium_trading.strategy.crypto_trend_long import CryptoTrendLongStrategy

SNAPSHOT_MINUTES = 15
LABEL_HORIZON_HOURS = 8
LOCAL_MOVE_R = 0.5
STRESS_RETRACE_R = 0.15
STRESS_CURRENT_R = -0.05

SWING_LEFT_BARS = 2
SWING_RIGHT_BARS = 2
BREAK_CLOSE_ATR = 0.10
RETEST_TOLERANCE_ATR = 0.15
ACCEPTANCE_WINDOW_BARS = 5
ACCEPTANCE_MIN_CLOSES = 3
FAST_RECLAIM_BARS = 3

FIRST_TEST_YEAR = 2024
LAST_TEST_YEAR = 2025
REVERSAL_THRESHOLD = 0.50

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}

FEATURE_NAMES = (
    "m5_return_3_atr",
    "m5_return_12_atr",
    "m5_efficiency_12",
    "m5_volume_ratio_20",
    "m15_return_4_atr",
    "m30_return_4_atr",
    "h1_return_3_atr",
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
    "distance_to_swing_low_atr",
    "distance_to_swing_high_atr",
    "swing_range_atr",
    "swing_low_age_bars",
    "swing_low_broken_by_low",
    "swing_low_broken_by_close",
    "close_below_swing_low_atr",
    "wick_below_swing_low_atr",
    "body_fraction_below_swing_low",
    "bars_close_below_level_last5",
    "bars_low_below_level_last5",
    "consecutive_closes_below_level",
    "recent_reclaim",
    "recent_sweep",
    "recent_break_displacement_atr",
    "recent_break_volume_zscore",
    "post_break_volume_ratio",
    "recent_retest_happened",
    "recent_retest_held",
    "last_swing_low_lower",
    "last_swing_high_lower",
    "oi_change_30m",
    "oi_change_2h",
    "long_ratio",
    "long_ratio_change_2h",
    "funding_rate",
    "oi_available",
    "ratio_available",
    "funding_available",
)


@dataclass(frozen=True, slots=True)
class StructureContext:
    swing_low: float
    swing_high: float
    previous_swing_low: float
    previous_swing_high: float
    swing_low_index: int
    swing_high_index: int
    atr5: float


@dataclass(frozen=True, slots=True)
class StructuralStateSample:
    trade_entry_time: datetime
    snapshot_time: datetime
    label_end_time: datetime
    features: tuple[float, ...]
    state_class: str

    @property
    def is_clear(self) -> bool:
        return self.state_class in {
            "NOISE",
            "CORRECTION",
            "CONFIRMED_REVERSAL",
        }

    @property
    def is_reversal(self) -> bool:
        return self.state_class == "CONFIRMED_REVERSAL"


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
class StructuralFold:
    test_year: int
    train_samples: int
    test_samples: int
    clear_test_samples: int
    excluded_test_samples: int
    train_state_distribution: dict[str, int]
    test_state_distribution: dict[str, int]
    classification: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class StructuralEvaluation:
    feature_names: tuple[str, ...]
    model_params: dict[str, object]
    reversal_threshold: float
    folds: tuple[StructuralFold, ...]
    combined_state_distribution: dict[str, int]
    combined_classification: ClassificationMetrics
    combined_test_samples: int
    combined_clear_samples: int
    combined_excluded_samples: int


def extract_structural_state_samples(
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
) -> tuple[StructuralStateSample, ...]:
    if not candles_5m:
        raise ValueError("structural-state extraction requires M5 candles")
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
        raise ValueError("structural-state extraction found no economic-pass trades")

    m15 = _aggregate(candles_5m, 15)
    m30 = _aggregate(candles_5m, 30)
    h1 = _aggregate(candles_5m, 60)
    h4 = _aggregate(candles_5m, 240)

    m5_times = [candle.timestamp for candle in candles_5m]
    oi_times = [point.timestamp for point in open_interest]
    ratio_times = [point.timestamp for point in account_ratio]
    funding_times = [point.timestamp for point in funding]

    samples: list[StructuralStateSample] = []
    for trade in trades:
        risk_distance = trade.entry - trade.stop
        if risk_distance <= 0:
            continue

        entry_index = bisect_left(m5_times, trade.entry_time)
        if entry_index >= len(candles_5m):
            continue

        target_price = trade.entry + 2.0 * risk_distance
        exit_limit = min(
            trade.exit_time,
            _first_m5_exit_time(
                candles_5m=candles_5m,
                m5_times=m5_times,
                entry_time=trade.entry_time,
                stop=trade.stop,
                target=target_price,
            ),
            trade.entry_time + timedelta(hours=24),
            candles_5m[-1].timestamp + timedelta(minutes=5),
        )
        snapshot_time = trade.entry_time + timedelta(minutes=SNAPSHOT_MINUTES)

        while snapshot_time < exit_limit:
            if snapshot_time + timedelta(hours=LABEL_HORIZON_HOURS) > (
                candles_5m[-1].timestamp + timedelta(minutes=5)
            ):
                break

            end_index = bisect_left(m5_times, snapshot_time)
            if end_index <= entry_index:
                snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)
                continue

            path = candles_5m[entry_index:end_index]
            price = path[-1].close
            mfe_so_far_r = max(
                0.0,
                (max(candle.high for candle in path) - trade.entry) / risk_distance,
            )
            mae_so_far_r = max(
                0.0,
                (trade.entry - min(candle.low for candle in path)) / risk_distance,
            )
            current_r = (price - trade.entry) / risk_distance
            retrace_from_mfe_r = mfe_so_far_r - current_r

            if (
                retrace_from_mfe_r < STRESS_RETRACE_R
                and current_r > STRESS_CURRENT_R
            ):
                snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)
                continue

            feature_result = _features(
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
            if feature_result is None:
                snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)
                continue
            features, structure = feature_result

            state_class = _structural_outcome_class(
                candles_5m=candles_5m,
                m5_times=m5_times,
                snapshot_time=snapshot_time,
                snapshot_price=price,
                risk_distance=risk_distance,
                structure=structure,
            )
            samples.append(
                StructuralStateSample(
                    trade_entry_time=trade.entry_time,
                    snapshot_time=snapshot_time,
                    label_end_time=snapshot_time
                    + timedelta(hours=LABEL_HORIZON_HOURS),
                    features=features,
                    state_class=state_class,
                )
            )
            snapshot_time += timedelta(minutes=SNAPSHOT_MINUTES)

    return tuple(samples)


def evaluate_structural_state_v02(
    samples: Iterable[StructuralStateSample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> StructuralEvaluation:
    classifier_type, metrics_module = _load_ml()
    all_samples = tuple(sorted(samples, key=lambda item: item.snapshot_time))
    if not all_samples:
        raise ValueError("structural-state evaluation requires samples")

    folds: list[StructuralFold] = []
    combined_test: list[StructuralStateSample] = []
    combined_truth: list[bool] = []
    combined_probabilities: list[float] = []
    combined_clear = 0

    for year in range(first_test_year, last_test_year + 1):
        test_start = datetime(year, 1, 1, tzinfo=UTC)
        test_end = datetime(year + 1, 1, 1, tzinfo=UTC)

        train = tuple(
            sample
            for sample in all_samples
            if sample.label_end_time <= test_start and sample.is_clear
        )
        test = tuple(
            sample
            for sample in all_samples
            if test_start <= sample.snapshot_time < test_end
        )
        clear_test = tuple(sample for sample in test if sample.is_clear)
        if not test:
            continue
        if len(train) < 300:
            raise ValueError(
                f"not enough pre-{year} clear structural samples: "
                f"{len(train)}; need at least 300"
            )
        if len(clear_test) < 100:
            raise ValueError(
                f"not enough {year} clear structural samples: "
                f"{len(clear_test)}; need at least 100"
            )

        train_y = [sample.is_reversal for sample in train]
        if len(set(train_y)) < 2:
            raise ValueError(f"pre-{year} structural labels contain one class")

        classifier = classifier_type(**MODEL_PARAMS)
        classifier.fit(
            [sample.features for sample in train],
            train_y,
        )
        probabilities = _positive_probabilities(
            classifier,
            [sample.features for sample in clear_test],
        )
        truth = tuple(sample.is_reversal for sample in clear_test)

        folds.append(
            StructuralFold(
                test_year=year,
                train_samples=len(train),
                test_samples=len(test),
                clear_test_samples=len(clear_test),
                excluded_test_samples=len(test) - len(clear_test),
                train_state_distribution=_state_distribution(train),
                test_state_distribution=_state_distribution(test),
                classification=_classification_metrics(
                    truth,
                    probabilities,
                    metrics_module,
                ),
            )
        )

        combined_test.extend(test)
        combined_clear += len(clear_test)
        combined_truth.extend(truth)
        combined_probabilities.extend(probabilities)

    if not folds:
        raise ValueError("no structural-state folds contained test samples")

    return StructuralEvaluation(
        feature_names=FEATURE_NAMES,
        model_params=dict(MODEL_PARAMS),
        reversal_threshold=REVERSAL_THRESHOLD,
        folds=tuple(folds),
        combined_state_distribution=_state_distribution(combined_test),
        combined_classification=_classification_metrics(
            tuple(combined_truth),
            tuple(combined_probabilities),
            metrics_module,
        ),
        combined_test_samples=len(combined_test),
        combined_clear_samples=combined_clear,
        combined_excluded_samples=len(combined_test) - combined_clear,
    )


def evaluation_payload(evaluation: StructuralEvaluation) -> dict[str, object]:
    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "explicit market structure noise-vs-reversal feasibility",
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
        "structure": {
            "swing_confirmation": (
                f"{SWING_LEFT_BARS} bars left / {SWING_RIGHT_BARS} bars right"
            ),
            "break_close_atr": BREAK_CLOSE_ATR,
            "retest_tolerance_atr": RETEST_TOLERANCE_ATR,
            "acceptance_window_bars": ACCEPTANCE_WINDOW_BARS,
            "acceptance_min_closes": ACCEPTANCE_MIN_CLOSES,
            "fast_reclaim_bars": FAST_RECLAIM_BARS,
        },
        "labels": {
            "horizon_hours": LABEL_HORIZON_HOURS,
            "local_move_r": LOCAL_MOVE_R,
            "classes": {
                "NOISE": (
                    "sweep/break attempt is quickly reclaimed and the adverse "
                    "move is not structurally accepted"
                ),
                "CORRECTION": (
                    "key swing low remains structurally intact and the market "
                    "recovers +0.5R before a -0.5R continuation"
                ),
                "REVERSAL_CANDIDATE": (
                    "body closes through the key swing low but confirmation "
                    "is incomplete"
                ),
                "CONFIRMED_REVERSAL": (
                    "body break through key swing low plus acceptance/retest "
                    "confirmation and adverse continuation"
                ),
                "AMBIGUOUS": "all remaining unclear structural outcomes",
            },
            "training_policy": (
                "train only NOISE, CORRECTION and CONFIRMED_REVERSAL; "
                "exclude REVERSAL_CANDIDATE and AMBIGUOUS"
            ),
            "binary_classifier": "CONFIRMED_REVERSAL vs NOISE/CORRECTION",
        },
        "feature_names": list(evaluation.feature_names),
        "model": {
            "type": "HistGradientBoostingClassifier",
            "params": evaluation.model_params,
            "reversal_threshold": evaluation.reversal_threshold,
            "threshold_tuning": "none; fixed at 0.50",
        },
        "folds": [
            {
                "test_year": fold.test_year,
                "train_samples": fold.train_samples,
                "test_samples": fold.test_samples,
                "clear_test_samples": fold.clear_test_samples,
                "excluded_test_samples": fold.excluded_test_samples,
                "train_state_distribution": fold.train_state_distribution,
                "test_state_distribution": fold.test_state_distribution,
                "classification": asdict(fold.classification),
            }
            for fold in evaluation.folds
        ],
        "combined": {
            "test_samples": evaluation.combined_test_samples,
            "clear_samples": evaluation.combined_clear_samples,
            "excluded_samples": evaluation.combined_excluded_samples,
            "state_distribution": evaluation.combined_state_distribution,
            "classification": asdict(evaluation.combined_classification),
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
) -> tuple[tuple[float, ...], StructureContext] | None:
    m5_hist = _history_before(candles_5m, snapshot_time, 96)
    m15_hist = _history_completed(m15, snapshot_time, 15, 16)
    m30_hist = _history_completed(m30, snapshot_time, 30, 10)
    h1_hist = _history_completed(h1, snapshot_time, 60, 8)
    h4_hist = _history_completed(h4, snapshot_time, 240, 24)
    if (
        len(m5_hist) < 60
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

    structure = _structure_context(m5_hist, atr5)
    if structure is None:
        return None

    latest = m5_hist[-1]
    m5_closes = [candle.close for candle in m5_hist]
    m5_volume_reference = m5_hist[-21:-1]
    mean_volume = mean(candle.volume for candle in m5_volume_reference)
    volume_ratio = latest.volume / mean_volume if mean_volume > 0 else 1.0

    h4_closes = tuple(candle.close for candle in h4_hist)
    ema20 = CryptoTrendLongStrategy._ema_series(h4_closes, 20)

    structural = _structural_features(m5_hist, structure)

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
    funding_rate, funding_available = _latest_value(
        funding,
        funding_times,
        snapshot_time,
        "funding_rate",
    )

    target_price = trade_entry + 2.0 * risk_distance
    features = (
        (m5_closes[-1] - m5_closes[-4]) / atr5,
        (m5_closes[-1] - m5_closes[-13]) / atr5,
        _efficiency_ratio(m5_closes[-13:]),
        volume_ratio,
        (m15_hist[-1].close - m15_hist[-5].close) / atr15,
        (m30_hist[-1].close - m30_hist[-5].close) / atr30,
        (h1_hist[-1].close - h1_hist[-4].close) / atr1h,
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
        *structural,
        _fraction_change(oi_now, oi_30m),
        _fraction_change(oi_now, oi_2h),
        long_ratio,
        long_ratio - long_ratio_2h,
        funding_rate,
        1.0 if oi_available else 0.0,
        1.0 if ratio_available else 0.0,
        1.0 if funding_available else 0.0,
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("structural feature count mismatch")
    return features, structure


def _structure_context(
    history: tuple[Candle, ...],
    atr5: float,
) -> StructureContext | None:
    lows, highs = _confirmed_swings(history)
    if len(lows) < 2 or len(highs) < 2:
        return None

    low_index, swing_low = lows[-1]
    previous_low = lows[-2][1]
    high_index, swing_high = highs[-1]
    previous_high = highs[-2][1]
    return StructureContext(
        swing_low=swing_low,
        swing_high=swing_high,
        previous_swing_low=previous_low,
        previous_swing_high=previous_high,
        swing_low_index=low_index,
        swing_high_index=high_index,
        atr5=atr5,
    )


def _confirmed_swings(
    history: tuple[Candle, ...],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    lows: list[tuple[int, float]] = []
    highs: list[tuple[int, float]] = []
    for index in range(
        SWING_LEFT_BARS,
        len(history) - SWING_RIGHT_BARS,
    ):
        candle = history[index]
        left = history[index - SWING_LEFT_BARS : index]
        right = history[index + 1 : index + 1 + SWING_RIGHT_BARS]

        if (
            candle.low < min(item.low for item in left)
            and candle.low <= min(item.low for item in right)
        ):
            lows.append((index, candle.low))
        if (
            candle.high > max(item.high for item in left)
            and candle.high >= max(item.high for item in right)
        ):
            highs.append((index, candle.high))
    return lows, highs


def _structural_features(
    history: tuple[Candle, ...],
    structure: StructureContext,
) -> tuple[float, ...]:
    level = structure.swing_low
    atr = structure.atr5
    latest = history[-1]
    recent5 = history[-5:]
    recent12_start = max(structure.swing_low_index + 1, len(history) - 12)
    recent12 = history[recent12_start:]

    break_indexes = [
        index
        for index in range(recent12_start, len(history))
        if history[index].close < level
    ]
    latest_break_index = break_indexes[-1] if break_indexes else None

    recent_sweep = any(candle.low < level for candle in recent12)
    recent_reclaim = recent_sweep and latest.close > level
    broken_by_low = latest.low < level
    broken_by_close = latest.close < level

    close_below = max(0.0, level - latest.close) / atr
    wick_below = max(0.0, level - latest.low) / atr
    body_low = min(latest.open, latest.close)
    body_high = max(latest.open, latest.close)
    body_size = body_high - body_low
    body_below = max(0.0, min(body_high, level) - body_low)
    body_fraction_below = body_below / body_size if body_size > 0 else 0.0

    close_below_count = sum(candle.close < level for candle in recent5)
    low_below_count = sum(candle.low < level for candle in recent5)
    consecutive = 0
    for candle in reversed(history):
        if candle.close >= level:
            break
        consecutive += 1

    break_displacement = 0.0
    break_volume_z = 0.0
    post_break_volume_ratio = 1.0
    retest_happened = False
    retest_held = False
    if latest_break_index is not None:
        break_candle = history[latest_break_index]
        break_displacement = max(0.0, level - break_candle.close) / atr
        volume_start = max(0, latest_break_index - 20)
        reference = [
            candle.volume
            for candle in history[volume_start:latest_break_index]
        ]
        if len(reference) >= 5:
            reference_mean = mean(reference)
            reference_std = pstdev(reference)
            if reference_std > 0:
                break_volume_z = (
                    break_candle.volume - reference_mean
                ) / reference_std

        after_break = history[latest_break_index + 1 :]
        if after_break and break_candle.volume > 0:
            post_break_volume_ratio = (
                mean(candle.volume for candle in after_break)
                / break_candle.volume
            )
        tolerance = RETEST_TOLERANCE_ATR * atr
        for candle in after_break:
            if candle.high >= level - tolerance:
                retest_happened = True
                if candle.close < level:
                    retest_held = True
                    break

    return (
        (latest.close - level) / atr,
        (structure.swing_high - latest.close) / atr,
        (structure.swing_high - structure.swing_low) / atr,
        float(len(history) - 1 - structure.swing_low_index),
        1.0 if broken_by_low else 0.0,
        1.0 if broken_by_close else 0.0,
        close_below,
        wick_below,
        body_fraction_below,
        float(close_below_count),
        float(low_below_count),
        float(consecutive),
        1.0 if recent_reclaim else 0.0,
        1.0 if recent_sweep else 0.0,
        break_displacement,
        break_volume_z,
        post_break_volume_ratio,
        1.0 if retest_happened else 0.0,
        1.0 if retest_held else 0.0,
        1.0 if structure.swing_low < structure.previous_swing_low else 0.0,
        1.0 if structure.swing_high < structure.previous_swing_high else 0.0,
    )


def _structural_outcome_class(
    *,
    candles_5m: tuple[Candle, ...],
    m5_times: list[datetime],
    snapshot_time: datetime,
    snapshot_price: float,
    risk_distance: float,
    structure: StructureContext,
) -> str:
    start = bisect_left(m5_times, snapshot_time)
    end = bisect_left(
        m5_times,
        snapshot_time + timedelta(hours=LABEL_HORIZON_HOURS),
    )
    future = candles_5m[start:end]
    if not future:
        return "AMBIGUOUS"

    level = structure.swing_low
    atr = structure.atr5
    break_level = level - BREAK_CLOSE_ATR * atr
    up_level = snapshot_price + LOCAL_MOVE_R * risk_distance
    down_level = snapshot_price - LOCAL_MOVE_R * risk_distance

    first_sweep = next(
        (index for index, candle in enumerate(future) if candle.low < level),
        None,
    )
    first_break = next(
        (
            index
            for index, candle in enumerate(future)
            if candle.close < break_level
        ),
        None,
    )
    first_up = next(
        (index for index, candle in enumerate(future) if candle.high >= up_level),
        None,
    )
    first_down = next(
        (index for index, candle in enumerate(future) if candle.low <= down_level),
        None,
    )

    if first_break is None:
        if first_sweep is not None:
            reclaim_end = min(
                len(future),
                first_sweep + FAST_RECLAIM_BARS + 1,
            )
            if any(
                candle.close > level
                for candle in future[first_sweep:reclaim_end]
            ):
                return "NOISE"
        if first_up is not None and (
            first_down is None or first_up < first_down
        ):
            return "CORRECTION"
        return "AMBIGUOUS"

    confirmation = future[
        first_break : first_break + ACCEPTANCE_WINDOW_BARS
    ]
    accepted = (
        sum(candle.close < level for candle in confirmation)
        >= ACCEPTANCE_MIN_CLOSES
    )
    tolerance = RETEST_TOLERANCE_ATR * atr
    retest_held = any(
        candle.high >= level - tolerance and candle.close < level
        for candle in confirmation[1:]
    )
    fast_reclaim = any(
        candle.close > level
        for candle in future[
            first_break : first_break + FAST_RECLAIM_BARS + 1
        ]
    )
    adverse_continuation = first_down is not None and (
        first_up is None or first_down < first_up
    )

    if (accepted or retest_held) and adverse_continuation:
        return "CONFIRMED_REVERSAL"
    if fast_reclaim and (
        first_up is not None
        and (first_down is None or first_up < first_down)
    ):
        return "NOISE"
    return "REVERSAL_CANDIDATE"


def _classification_metrics(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
    metrics_module,
) -> ClassificationMetrics:
    predicted = tuple(
        probability >= REVERSAL_THRESHOLD
        for probability in probabilities
    )
    tp = int(sum(a and p for a, p in zip(truth, predicted, strict=True)))
    fp = int(sum((not a) and p for a, p in zip(truth, predicted, strict=True)))
    fn = int(sum(a and (not p) for a, p in zip(truth, predicted, strict=True)))
    tn = int(
        sum((not a) and (not p) for a, p in zip(truth, predicted, strict=True))
    )
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
        actual_reversal=actual,
        predicted_reversal=selected,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        base_reversal_rate=float(base),
        precision=float(precision),
        recall=float(recall),
        precision_lift=float(precision / base if base > 0 else 0.0),
        roc_auc=auc,
        brier_score=brier,
    )


def _state_distribution(
    samples: Iterable[StructuralStateSample],
) -> dict[str, int]:
    counts = Counter(sample.state_class for sample in samples)
    return {
        name: counts.get(name, 0)
        for name in (
            "NOISE",
            "CORRECTION",
            "REVERSAL_CANDIDATE",
            "CONFIRMED_REVERSAL",
            "AMBIGUOUS",
        )
    }


def _aggregate(
    candles: tuple[Candle, ...],
    period_minutes: int,
) -> tuple[Candle, ...]:
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


def _first_m5_exit_time(
    *,
    candles_5m: tuple[Candle, ...],
    m5_times: list[datetime],
    entry_time: datetime,
    stop: float,
    target: float,
) -> datetime:
    start = bisect_left(m5_times, entry_time)
    deadline = entry_time + timedelta(hours=24)
    end = bisect_left(m5_times, deadline)
    for candle in candles_5m[start:end]:
        candle_end = candle.timestamp + timedelta(minutes=5)
        if candle.low <= stop:
            return candle_end
        if candle.high >= target:
            return candle_end
    return deadline


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
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'ML research dependencies are not installed. Run pip install -e ".[ml]".'
        ) from exc

    return HistGradientBoostingClassifier, metrics
