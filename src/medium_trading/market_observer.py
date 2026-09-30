from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from statistics import mean, pstdev

from medium_trading.data.bybit_state_history import (
    AccountRatioPoint,
    FundingPoint,
    OpenInterestPoint,
)
from medium_trading.domain import Candle
from medium_trading.market_state_model import _aggregate

RESEARCH_START = datetime(2023, 1, 1, tzinfo=UTC)
RESEARCH_END = datetime(2026, 1, 1, tzinfo=UTC)
FIRST_TEST_YEAR = 2024
LAST_TEST_YEAR = 2025
LABEL_HORIZON_HOURS = 8
EMBARGO_HOURS = 8

SWING_LEFT_BARS = 2
SWING_RIGHT_BARS = 2
BREAK_CLOSE_ATR = 0.10
ACCEPTANCE_WINDOW_BARS = 5
ACCEPTANCE_MIN_CLOSES = 3

TREND_BARRIER_ATR = 0.75
CORRECTION_BARRIER_ATR = 0.75
REVERSAL_BARRIER_ATR = 1.50

CLEAR_LABELS = ("NOISE", "CORRECTION", "REAL_REVERSAL")
ALL_LABELS = (*CLEAR_LABELS, "AMBIGUOUS")
CATEGORICAL_FEATURES = (
    "trend_side",
    "event_type",
    "session",
    "weekday",
)
NUMERIC_FEATURES = (
    "m5_return_1_atr",
    "m5_return_3_atr",
    "m5_return_12_atr",
    "m5_acceleration_3_atr",
    "m5_efficiency_12",
    "m5_efficiency_36",
    "atr14_to_atr200",
    "realized_vol_12_to_48",
    "range_12_atr",
    "overlap_12",
    "body_signed_atr",
    "wick_with_trend_atr",
    "wick_against_trend_atr",
    "volume_ratio_20",
    "volume_zscore_50",
    "volume_acceleration_5_to_20",
    "m15_return_4_atr",
    "m30_return_4_atr",
    "h1_return_3_atr",
    "h4_return_3_atr",
    "h4_ema20_distance_atr",
    "h4_ema20_slope_3_atr",
    "h1_efficiency_8",
    "h4_efficiency_8",
    "m30_rsi14",
    "m30_rsi_z30",
    "defended_level_distance_atr",
    "opposite_swing_distance_atr",
    "swing_range_atr",
    "defended_swing_age_bars",
    "adverse_wick_through_atr",
    "adverse_close_through_atr",
    "body_fraction_through",
    "adverse_closes_last5",
    "adverse_touches_last5",
    "consecutive_adverse_closes",
    "defended_swing_weaker",
    "opposite_swing_weaker",
    "event_break_displacement_atr",
    "event_volume_zscore",
    "oi_change_30m",
    "oi_change_2h",
    "oi_change_4h",
    "price_oi_alignment_2h",
    "long_ratio",
    "long_ratio_change_2h",
    "funding_rate",
    "oi_available",
    "ratio_available",
    "funding_available",
)
FEATURE_NAMES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
CAT_FEATURE_INDICES = tuple(range(len(CATEGORICAL_FEATURES)))

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "loss_function": "MultiClass",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


class MarketObserverSample:
    __slots__ = (
        "event_time",
        "label_end_time",
        "trend_side",
        "event_type",
        "features",
        "state_class",
        "defended_level",
        "atr5",
    )

    def __init__(
        self,
        *,
        event_time: datetime,
        label_end_time: datetime,
        trend_side: str,
        event_type: str,
        features: tuple[object, ...],
        state_class: str,
        defended_level: float,
        atr5: float,
    ) -> None:
        self.event_time = event_time
        self.label_end_time = label_end_time
        self.trend_side = trend_side
        self.event_type = event_type
        self.features = features
        self.state_class = state_class
        self.defended_level = defended_level
        self.atr5 = atr5

    @property
    def is_clear(self) -> bool:
        return self.state_class in CLEAR_LABELS


class _Structure:
    __slots__ = (
        "defended_index",
        "defended_level",
        "previous_defended",
        "opposite_level",
        "previous_opposite",
    )

    def __init__(
        self,
        *,
        defended_index: int,
        defended_level: float,
        previous_defended: float,
        opposite_level: float,
        previous_opposite: float,
    ) -> None:
        self.defended_index = defended_index
        self.defended_level = defended_level
        self.previous_defended = previous_defended
        self.opposite_level = opposite_level
        self.previous_opposite = previous_opposite


def extract_market_observer_samples(
    *,
    candles_5m: tuple[Candle, ...],
    open_interest: tuple[OpenInterestPoint, ...],
    account_ratio: tuple[AccountRatioPoint, ...],
    funding: tuple[FundingPoint, ...],
) -> tuple[MarketObserverSample, ...]:
    """
    Extract market-only structural disturbance events.

    The observer never sees trade entry/stop/target/PnL state. Every sample is
    anchored to a causal M5 market event while a completed-H4 trend exists.
    """
    if not candles_5m:
        raise ValueError("market observer requires M5 candles")
    _require_contiguous(candles_5m)

    m15 = _aggregate(candles_5m, 15)
    m30 = _aggregate(candles_5m, 30)
    h1 = _aggregate(candles_5m, 60)
    h4 = _aggregate(candles_5m, 240)

    m15_ends = [item.timestamp + timedelta(minutes=15) for item in m15]
    m30_ends = [item.timestamp + timedelta(minutes=30) for item in m30]
    h1_ends = [item.timestamp + timedelta(hours=1) for item in h1]
    h4_ends = [item.timestamp + timedelta(hours=4) for item in h4]
    h4_ema20 = _ema_series([item.close for item in h4], 20)
    h4_regimes = _h4_regimes(h4, h4_ema20)

    lows, highs = _precompute_swings(candles_5m)
    low_cursor = 0
    high_cursor = 0
    usable_lows: list[tuple[int, float]] = []
    usable_highs: list[tuple[int, float]] = []

    oi_times = [point.timestamp for point in open_interest]
    ratio_times = [point.timestamp for point in account_ratio]
    funding_times = [point.timestamp for point in funding]

    sampled_levels: set[tuple[str, int]] = set()
    samples: list[MarketObserverSample] = []

    for index, candle in enumerate(candles_5m):
        event_time = candle.timestamp + timedelta(minutes=5)
        if event_time < RESEARCH_START:
            continue
        if event_time >= RESEARCH_END:
            break
        if event_time + timedelta(hours=LABEL_HORIZON_HOURS) > min(
            RESEARCH_END,
            candles_5m[-1].timestamp + timedelta(minutes=5),
        ):
            break

        while low_cursor < len(lows) and lows[low_cursor][0] <= index:
            _, swing_index, value = lows[low_cursor]
            usable_lows.append((swing_index, value))
            low_cursor += 1
        while high_cursor < len(highs) and highs[high_cursor][0] <= index:
            _, swing_index, value = highs[high_cursor]
            usable_highs.append((swing_index, value))
            high_cursor += 1

        if len(usable_lows) < 2 or len(usable_highs) < 2:
            continue

        h4_index = bisect_right(h4_ends, event_time) - 1
        if h4_index < 20:
            continue
        trend_side = h4_regimes[h4_index]
        if trend_side is None:
            continue

        structure = _structure_for_side(
            trend_side,
            usable_lows,
            usable_highs,
        )
        level_key = (trend_side, structure.defended_index)
        if level_key in sampled_levels:
            continue

        m5_hist = candles_5m[max(0, index + 1 - 240) : index + 1]
        if len(m5_hist) < 201:
            continue
        atr5 = _atr(m5_hist, 14)
        if atr5 <= 0:
            continue

        event_type = _event_type(
            candle=candle,
            trend_side=trend_side,
            level=structure.defended_level,
            atr5=atr5,
        )
        if event_type is None:
            continue

        features = _features(
            candles_5m=candles_5m,
            event_index=index,
            event_time=event_time,
            trend_side=trend_side,
            event_type=event_type,
            structure=structure,
            atr5=atr5,
            m15=m15,
            m30=m30,
            h1=h1,
            h4=h4,
            m15_ends=m15_ends,
            m30_ends=m30_ends,
            h1_ends=h1_ends,
            h4_ends=h4_ends,
            h4_ema20=h4_ema20,
            open_interest=open_interest,
            account_ratio=account_ratio,
            funding=funding,
            oi_times=oi_times,
            ratio_times=ratio_times,
            funding_times=funding_times,
        )
        if features is None:
            continue

        state_class = _market_label(
            candles_5m=candles_5m,
            event_index=index,
            event_time=event_time,
            event_price=candle.close,
            trend_side=trend_side,
            event_type=event_type,
            defended_level=structure.defended_level,
            atr5=atr5,
        )
        sampled_levels.add(level_key)
        samples.append(
            MarketObserverSample(
                event_time=event_time,
                label_end_time=event_time + timedelta(hours=LABEL_HORIZON_HOURS),
                trend_side=trend_side,
                event_type=event_type,
                features=features,
                state_class=state_class,
                defended_level=structure.defended_level,
                atr5=atr5,
            )
        )

    return tuple(samples)


def evaluate_market_observer_v04(
    samples: Iterable[MarketObserverSample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> dict[str, object]:
    CatBoostClassifier, Pool, RandomForestClassifier, metrics = _load_ml()

    all_samples = tuple(sorted(samples, key=lambda item: item.event_time))
    if not all_samples:
        raise ValueError("market observer evaluation requires samples")

    folds: list[dict[str, object]] = []
    combined_truth: list[str] = []
    combined_catboost: list[tuple[float, ...]] = []
    combined_forest: list[tuple[float, ...]] = []
    catboost_importances: list[tuple[float, ...]] = []
    forest_importances: list[tuple[float, ...]] = []

    for year in range(first_test_year, last_test_year + 1):
        test_start = datetime(year, 1, 1, tzinfo=UTC)
        test_end = datetime(year + 1, 1, 1, tzinfo=UTC)
        embargo_cutoff = test_start - timedelta(hours=EMBARGO_HOURS)

        train = tuple(
            sample
            for sample in all_samples
            if sample.is_clear and sample.label_end_time <= embargo_cutoff
        )
        test_all = tuple(
            sample
            for sample in all_samples
            if test_start <= sample.event_time < test_end
        )
        test = tuple(sample for sample in test_all if sample.is_clear)
        if not test_all:
            continue
        if len(train) < 500:
            raise ValueError(
                f"not enough pre-{year} market-observer samples: "
                f"{len(train)}; need at least 500"
            )
        if len(test) < 100:
            raise ValueError(
                f"not enough clear {year} market-observer samples: "
                f"{len(test)}; need at least 100"
            )
        if set(sample.state_class for sample in train) != set(CLEAR_LABELS):
            raise ValueError(f"pre-{year} training data does not contain all clear classes")

        train_x = [list(sample.features) for sample in train]
        test_x = [list(sample.features) for sample in test]
        train_y = [sample.state_class for sample in train]
        truth = tuple(sample.state_class for sample in test)

        train_pool = Pool(
            data=train_x,
            label=train_y,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        test_pool = Pool(
            data=test_x,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        catboost = CatBoostClassifier(**CATBOOST_PARAMS)
        catboost.fit(train_pool)
        cat_probs = _aligned_probabilities(
            catboost.classes_,
            catboost.predict_proba(test_pool),
        )

        forest = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        forest_train_x = [_rf_row(sample.features) for sample in train]
        forest_test_x = [_rf_row(sample.features) for sample in test]
        forest.fit(forest_train_x, train_y)
        forest_probs = _aligned_probabilities(
            forest.classes_,
            forest.predict_proba(forest_test_x),
        )

        cat_importance = tuple(float(value) for value in catboost.get_feature_importance())
        forest_importance = tuple(float(value) for value in forest.feature_importances_)
        catboost_importances.append(cat_importance)
        forest_importances.append(forest_importance)

        folds.append(
            {
                "test_year": year,
                "train_samples": len(train),
                "test_samples": len(test_all),
                "clear_test_samples": len(test),
                "excluded_test_samples": len(test_all) - len(test),
                "train_state_distribution": _state_distribution(train),
                "test_state_distribution": _state_distribution(test_all),
                "test_event_distribution": _event_distribution(test_all),
                "test_trend_distribution": _trend_distribution(test_all),
                "catboost": {
                    "metrics": _model_metrics(truth, cat_probs, metrics),
                    "top_features": _top_features(cat_importance),
                },
                "random_forest": {
                    "metrics": _model_metrics(truth, forest_probs, metrics),
                    "top_features": _top_features(forest_importance),
                },
            }
        )

        combined_truth.extend(truth)
        combined_catboost.extend(cat_probs)
        combined_forest.extend(forest_probs)

    if not folds:
        raise ValueError("no market-observer folds contained test samples")

    combined_test = tuple(
        sample
        for sample in all_samples
        if FIRST_TEST_YEAR <= sample.event_time.year <= LAST_TEST_YEAR
    )
    cat_mean_importance = _mean_importance(catboost_importances)
    forest_mean_importance = _mean_importance(forest_importances)

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "market-only noise/correction/real-reversal observer feasibility",
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [fold["test_year"] for fold in folds],
            "2026_used": False,
            "trade_state_used": False,
            "sampling": "first causal structural disturbance of each defended M5 swing",
            "label_horizon_hours": LABEL_HORIZON_HOURS,
            "embargo_hours": EMBARGO_HOURS,
        },
        "labels": {
            "classes": {
                "NOISE": (
                    "the prevailing trend reaches +0.75 ATR from event price "
                    "before a -0.75 ATR adverse move"
                ),
                "CORRECTION": (
                    "the market first reaches -0.75 ATR against the prevailing "
                    "trend, then recovers +0.75 ATR with-trend inside 8h"
                ),
                "REAL_REVERSAL": (
                    "accepted structural damage continues at least -1.50 ATR "
                    "against trend and the +0.75 ATR trend-recovery barrier is "
                    "not reached inside 8h"
                ),
                "AMBIGUOUS": "all remaining unresolved event outcomes",
            },
            "trend_barrier_atr": TREND_BARRIER_ATR,
            "correction_barrier_atr": CORRECTION_BARRIER_ATR,
            "reversal_barrier_atr": REVERSAL_BARRIER_ATR,
            "acceptance_window_bars": ACCEPTANCE_WINDOW_BARS,
            "acceptance_min_closes": ACCEPTANCE_MIN_CLOSES,
            "training_policy": "train/score clear NOISE/CORRECTION/REAL_REVERSAL only",
        },
        "features": {
            "names": list(FEATURE_NAMES),
            "categorical": list(CATEGORICAL_FEATURES),
            "normalization": (
                "directional price features are signed relative to the prevailing "
                "H4 trend; positive means with-trend"
            ),
            "trade_features_present": False,
        },
        "models": {
            "catboost": {
                "type": "CatBoostClassifier",
                "params": dict(CATBOOST_PARAMS),
            },
            "random_forest": {
                "type": "RandomForestClassifier",
                "params": dict(RANDOM_FOREST_PARAMS),
                "role": "fixed tabular benchmark",
            },
        },
        "folds": folds,
        "combined": {
            "test_samples": len(combined_test),
            "clear_samples": len(combined_truth),
            "excluded_samples": len(combined_test) - len(combined_truth),
            "state_distribution": _state_distribution(combined_test),
            "event_distribution": _event_distribution(combined_test),
            "trend_distribution": _trend_distribution(combined_test),
            "catboost": {
                "metrics": _model_metrics(
                    tuple(combined_truth),
                    tuple(combined_catboost),
                    metrics,
                ),
                "top_features": _top_features(cat_mean_importance),
                "top10_fold_overlap": _top_overlap(catboost_importances, 10),
            },
            "random_forest": {
                "metrics": _model_metrics(
                    tuple(combined_truth),
                    tuple(combined_forest),
                    metrics,
                ),
                "top_features": _top_features(forest_mean_importance),
                "top10_fold_overlap": _top_overlap(forest_importances, 10),
            },
        },
    }


def _precompute_swings(
    candles: tuple[Candle, ...],
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    lows: list[tuple[int, int, float]] = []
    highs: list[tuple[int, int, float]] = []
    for index in range(SWING_LEFT_BARS, len(candles) - SWING_RIGHT_BARS):
        candle = candles[index]
        left = candles[index - SWING_LEFT_BARS : index]
        right = candles[index + 1 : index + 1 + SWING_RIGHT_BARS]
        usable_index = index + SWING_RIGHT_BARS
        if (
            candle.low < min(item.low for item in left)
            and candle.low <= min(item.low for item in right)
        ):
            lows.append((usable_index, index, candle.low))
        if (
            candle.high > max(item.high for item in left)
            and candle.high >= max(item.high for item in right)
        ):
            highs.append((usable_index, index, candle.high))
    return lows, highs


def _structure_for_side(
    trend_side: str,
    lows: list[tuple[int, float]],
    highs: list[tuple[int, float]],
) -> _Structure:
    if trend_side == "BULL":
        return _Structure(
            defended_index=lows[-1][0],
            defended_level=lows[-1][1],
            previous_defended=lows[-2][1],
            opposite_level=highs[-1][1],
            previous_opposite=highs[-2][1],
        )
    return _Structure(
        defended_index=highs[-1][0],
        defended_level=highs[-1][1],
        previous_defended=highs[-2][1],
        opposite_level=lows[-1][1],
        previous_opposite=lows[-2][1],
    )


def _event_type(
    *,
    candle: Candle,
    trend_side: str,
    level: float,
    atr5: float,
) -> str | None:
    if trend_side == "BULL":
        if candle.low >= level:
            return None
        if candle.close > level:
            return "SWEEP_RECLAIM"
        if candle.close <= level - BREAK_CLOSE_ATR * atr5:
            return "BODY_BREAK"
        return None

    if candle.high <= level:
        return None
    if candle.close < level:
        return "SWEEP_RECLAIM"
    if candle.close >= level + BREAK_CLOSE_ATR * atr5:
        return "BODY_BREAK"
    return None


def _market_label(
    *,
    candles_5m: tuple[Candle, ...],
    event_index: int,
    event_time: datetime,
    event_price: float,
    trend_side: str,
    event_type: str,
    defended_level: float,
    atr5: float,
) -> str:
    direction = 1.0 if trend_side == "BULL" else -1.0
    trend_price = event_price + direction * TREND_BARRIER_ATR * atr5
    correction_price = event_price - direction * CORRECTION_BARRIER_ATR * atr5
    reversal_price = event_price - direction * REVERSAL_BARRIER_ATR * atr5

    deadline = event_time + timedelta(hours=LABEL_HORIZON_HOURS)
    future = []
    for candle in candles_5m[event_index + 1 :]:
        if candle.timestamp >= deadline:
            break
        future.append(candle)

    first_trend: int | None = None
    first_correction: int | None = None
    first_reversal: int | None = None
    ambiguous_same_bar = False

    for offset, candle in enumerate(future):
        trend_touch = _touches_trend(candle, trend_side, trend_price)
        correction_touch = _touches_adverse(candle, trend_side, correction_price)
        reversal_touch = _touches_adverse(candle, trend_side, reversal_price)

        if trend_touch and correction_touch and first_trend is None and first_correction is None:
            ambiguous_same_bar = True
            break
        if trend_touch and first_trend is None:
            first_trend = offset
        if correction_touch and first_correction is None:
            first_correction = offset
        if reversal_touch and first_reversal is None:
            first_reversal = offset

    if ambiguous_same_bar:
        return "AMBIGUOUS"
    if first_trend is not None and (
        first_correction is None or first_trend < first_correction
    ):
        return "NOISE"
    if first_correction is not None and first_trend is not None:
        if first_correction < first_trend:
            return "CORRECTION"

    accepted = _accepted_structure(
        candles_5m=candles_5m,
        event_index=event_index,
        event_type=event_type,
        trend_side=trend_side,
        level=defended_level,
    )
    if first_reversal is not None and first_trend is None and accepted:
        return "REAL_REVERSAL"
    return "AMBIGUOUS"


def _accepted_structure(
    *,
    candles_5m: tuple[Candle, ...],
    event_index: int,
    event_type: str,
    trend_side: str,
    level: float,
) -> bool:
    if event_type == "BODY_BREAK":
        window = candles_5m[
            event_index : event_index + ACCEPTANCE_WINDOW_BARS
        ]
    else:
        window = candles_5m[
            event_index + 1 : event_index + 1 + ACCEPTANCE_WINDOW_BARS
        ]
    adverse_closes = sum(
        _close_is_adverse(candle, trend_side, level)
        for candle in window
    )
    return adverse_closes >= ACCEPTANCE_MIN_CLOSES


def _features(
    *,
    candles_5m: tuple[Candle, ...],
    event_index: int,
    event_time: datetime,
    trend_side: str,
    event_type: str,
    structure: _Structure,
    atr5: float,
    m15: tuple[Candle, ...],
    m30: tuple[Candle, ...],
    h1: tuple[Candle, ...],
    h4: tuple[Candle, ...],
    m15_ends: list[datetime],
    m30_ends: list[datetime],
    h1_ends: list[datetime],
    h4_ends: list[datetime],
    h4_ema20: tuple[float, ...],
    open_interest: tuple[OpenInterestPoint, ...],
    account_ratio: tuple[AccountRatioPoint, ...],
    funding: tuple[FundingPoint, ...],
    oi_times: list[datetime],
    ratio_times: list[datetime],
    funding_times: list[datetime],
) -> tuple[object, ...] | None:
    m5_hist = candles_5m[max(0, event_index + 1 - 240) : event_index + 1]
    m15_hist = _completed_history(m15, m15_ends, event_time, 16)
    m30_hist = _completed_history(m30, m30_ends, event_time, 80)
    h1_hist = _completed_history(h1, h1_ends, event_time, 16)
    h4_hist = _completed_history(h4, h4_ends, event_time, 32)
    if (
        len(m5_hist) < 201
        or len(m15_hist) < 9
        or len(m30_hist) < 50
        or len(h1_hist) < 9
        or len(h4_hist) < 21
    ):
        return None

    direction = 1.0 if trend_side == "BULL" else -1.0
    atr15 = _atr(m15_hist, 8)
    atr30 = _atr(m30_hist, 14)
    atr1h = _atr(h1_hist, 8)
    atr4h = _atr(h4_hist, 14)
    atr200 = _atr(m5_hist, 200)
    if min(atr5, atr15, atr30, atr1h, atr4h, atr200) <= 0:
        return None

    latest = m5_hist[-1]
    closes = [item.close for item in m5_hist]
    m5_return_3 = (closes[-1] - closes[-4]) / atr5
    m5_prior_3 = (closes[-4] - closes[-7]) / atr5

    recent12 = m5_hist[-12:]
    range_12 = (
        max(item.high for item in recent12)
        - min(item.low for item in recent12)
    ) / atr5
    overlap_12 = _bar_overlap(recent12)

    body = latest.close - latest.open
    upper_wick = latest.high - max(latest.open, latest.close)
    lower_wick = min(latest.open, latest.close) - latest.low
    wick_with_trend = upper_wick if trend_side == "BULL" else lower_wick
    wick_against_trend = lower_wick if trend_side == "BULL" else upper_wick

    volumes20 = [item.volume for item in m5_hist[-21:-1]]
    volumes50 = [item.volume for item in m5_hist[-51:-1]]
    volume20 = mean(volumes20)
    volume50 = mean(volumes50)
    volume_std50 = pstdev(volumes50)
    volume_ratio20 = latest.volume / volume20 if volume20 > 0 else 1.0
    volume_z50 = (
        (latest.volume - volume50) / volume_std50
        if volume_std50 > 0
        else 0.0
    )
    recent_volume = mean(item.volume for item in m5_hist[-5:])
    prior_volume = mean(item.volume for item in m5_hist[-25:-5])
    volume_acceleration = (
        recent_volume / prior_volume if prior_volume > 0 else 1.0
    )

    m30_closes = [item.close for item in m30_hist]
    rsi_series = _rsi_series(m30_closes, 14)
    rsi14 = rsi_series[-1] if rsi_series else 50.0
    rsi_reference = rsi_series[-31:-1]
    rsi_mean = mean(rsi_reference) if rsi_reference else 50.0
    rsi_std = pstdev(rsi_reference) if len(rsi_reference) >= 2 else 0.0
    rsi_z = (rsi14 - rsi_mean) / rsi_std if rsi_std > 0 else 0.0

    h4_index = bisect_right(h4_ends, event_time) - 1
    if h4_index < 20:
        return None

    defended_distance = (
        direction * (latest.close - structure.defended_level) / atr5
    )
    opposite_distance = (
        direction * (structure.opposite_level - latest.close) / atr5
    )
    swing_range = abs(
        structure.opposite_level - structure.defended_level
    ) / atr5
    defended_age = event_index - structure.defended_index

    if trend_side == "BULL":
        adverse_wick = max(0.0, structure.defended_level - latest.low) / atr5
        adverse_close = max(
            0.0,
            structure.defended_level - latest.close,
        ) / atr5
        defended_weaker = structure.defended_level < structure.previous_defended
        opposite_weaker = structure.opposite_level < structure.previous_opposite
    else:
        adverse_wick = max(0.0, latest.high - structure.defended_level) / atr5
        adverse_close = max(
            0.0,
            latest.close - structure.defended_level,
        ) / atr5
        defended_weaker = structure.defended_level > structure.previous_defended
        opposite_weaker = structure.opposite_level > structure.previous_opposite

    body_fraction = _body_fraction_through(
        latest,
        trend_side,
        structure.defended_level,
    )
    recent5 = m5_hist[-5:]
    adverse_closes = sum(
        _close_is_adverse(item, trend_side, structure.defended_level)
        for item in recent5
    )
    adverse_touches = sum(
        _touches_level_adverse(item, trend_side, structure.defended_level)
        for item in recent5
    )
    consecutive = 0
    for item in reversed(m5_hist):
        if not _close_is_adverse(item, trend_side, structure.defended_level):
            break
        consecutive += 1

    event_break_displacement = adverse_close if event_type == "BODY_BREAK" else 0.0

    oi_now, oi_available = _latest_value(
        open_interest,
        oi_times,
        event_time,
        "open_interest",
    )
    oi_30m, _ = _latest_value(
        open_interest,
        oi_times,
        event_time - timedelta(minutes=30),
        "open_interest",
    )
    oi_2h, _ = _latest_value(
        open_interest,
        oi_times,
        event_time - timedelta(hours=2),
        "open_interest",
    )
    oi_4h, _ = _latest_value(
        open_interest,
        oi_times,
        event_time - timedelta(hours=4),
        "open_interest",
    )
    oi_change_30m = _fraction_change(oi_now, oi_30m)
    oi_change_2h = _fraction_change(oi_now, oi_2h)
    oi_change_4h = _fraction_change(oi_now, oi_4h)

    long_ratio, ratio_available = _latest_value(
        account_ratio,
        ratio_times,
        event_time,
        "buy_ratio",
    )
    long_ratio_2h, _ = _latest_value(
        account_ratio,
        ratio_times,
        event_time - timedelta(hours=2),
        "buy_ratio",
    )
    funding_rate, funding_available = _latest_value(
        funding,
        funding_times,
        event_time,
        "funding_rate",
    )

    price_move_2h = direction * (closes[-1] - closes[-25]) / atr5
    session = _session(event_time.hour)
    weekday = event_time.strftime("%a").upper()

    features: tuple[object, ...] = (
        trend_side,
        event_type,
        session,
        weekday,
        direction * (closes[-1] - closes[-2]) / atr5,
        direction * m5_return_3,
        direction * (closes[-1] - closes[-13]) / atr5,
        direction * (m5_return_3 - m5_prior_3),
        _efficiency_ratio(closes[-13:]),
        _efficiency_ratio(closes[-37:]),
        atr5 / atr200,
        _realized_vol_ratio(closes, 12, 48),
        range_12,
        overlap_12,
        direction * body / atr5,
        wick_with_trend / atr5,
        wick_against_trend / atr5,
        volume_ratio20,
        volume_z50,
        volume_acceleration,
        direction * (m15_hist[-1].close - m15_hist[-5].close) / atr15,
        direction * (m30_hist[-1].close - m30_hist[-5].close) / atr30,
        direction * (h1_hist[-1].close - h1_hist[-4].close) / atr1h,
        direction * (h4_hist[-1].close - h4_hist[-4].close) / atr4h,
        direction * (h4_hist[-1].close - h4_ema20[h4_index]) / atr4h,
        direction * (
            h4_ema20[h4_index] - h4_ema20[h4_index - 3]
        ) / atr4h,
        _efficiency_ratio([item.close for item in h1_hist[-9:]]),
        _efficiency_ratio([item.close for item in h4_hist[-9:]]),
        rsi14,
        rsi_z,
        defended_distance,
        opposite_distance,
        swing_range,
        float(defended_age),
        adverse_wick,
        adverse_close,
        body_fraction,
        float(adverse_closes),
        float(adverse_touches),
        float(consecutive),
        1.0 if defended_weaker else 0.0,
        1.0 if opposite_weaker else 0.0,
        event_break_displacement,
        volume_z50,
        oi_change_30m,
        oi_change_2h,
        oi_change_4h,
        price_move_2h * oi_change_2h,
        long_ratio,
        long_ratio - long_ratio_2h,
        funding_rate,
        1.0 if oi_available else 0.0,
        1.0 if ratio_available else 0.0,
        1.0 if funding_available else 0.0,
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(
            f"market observer feature count mismatch: "
            f"{len(features)} != {len(FEATURE_NAMES)}"
        )
    return features


def _h4_regimes(
    h4: tuple[Candle, ...],
    ema20: tuple[float, ...],
) -> tuple[str | None, ...]:
    regimes: list[str | None] = []
    for index, candle in enumerate(h4):
        if index < 20 or index < 3:
            regimes.append(None)
            continue
        rising = ema20[index] > ema20[index - 3]
        falling = ema20[index] < ema20[index - 3]
        if candle.close > ema20[index] and rising:
            regimes.append("BULL")
        elif candle.close < ema20[index] and falling:
            regimes.append("BEAR")
        else:
            regimes.append(None)
    return tuple(regimes)


def _ema_series(values: list[float], period: int) -> tuple[float, ...]:
    if not values:
        return ()
    alpha = 2.0 / (period + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return tuple(result)


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
        abs(current - previous)
        for previous, current in pairwise(closes)
    )
    return abs(closes[-1] - closes[0]) / path if path > 0 else 0.0


def _realized_vol_ratio(
    closes: list[float],
    short: int,
    long: int,
) -> float:
    returns = [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    if len(returns) < long:
        return 1.0
    short_std = pstdev(returns[-short:])
    long_std = pstdev(returns[-long:])
    return short_std / long_std if long_std > 0 else 1.0


def _bar_overlap(candles: tuple[Candle, ...]) -> float:
    values = []
    for previous, current in pairwise(candles):
        previous_range = previous.high - previous.low
        current_range = current.high - current.low
        denominator = min(previous_range, current_range)
        if denominator <= 0:
            continue
        overlap = max(
            0.0,
            min(previous.high, current.high)
            - max(previous.low, current.low),
        )
        values.append(min(1.0, overlap / denominator))
    return mean(values) if values else 0.0


def _rsi_series(closes: list[float], period: int) -> tuple[float, ...]:
    if len(closes) < period + 1:
        return ()
    result: list[float] = []
    for end in range(period, len(closes)):
        gains = 0.0
        losses = 0.0
        start = end - period + 1
        for index in range(start, end + 1):
            change = closes[index] - closes[index - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            result.append(100.0 if gains > 0 else 50.0)
        else:
            rs = gains / losses
            result.append(100.0 - 100.0 / (1.0 + rs))
    return tuple(result)


def _body_fraction_through(
    candle: Candle,
    trend_side: str,
    level: float,
) -> float:
    body_low = min(candle.open, candle.close)
    body_high = max(candle.open, candle.close)
    size = body_high - body_low
    if size <= 0:
        return 0.0
    if trend_side == "BULL":
        through = max(0.0, min(body_high, level) - body_low)
    else:
        through = max(0.0, body_high - max(body_low, level))
    return min(1.0, through / size)


def _completed_history(
    candles: tuple[Candle, ...],
    ends: list[datetime],
    timestamp: datetime,
    count: int,
) -> tuple[Candle, ...]:
    end = bisect_right(ends, timestamp)
    return candles[max(0, end - count) : end]


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


def _touches_trend(candle: Candle, trend_side: str, price: float) -> bool:
    return candle.high >= price if trend_side == "BULL" else candle.low <= price


def _touches_adverse(candle: Candle, trend_side: str, price: float) -> bool:
    return candle.low <= price if trend_side == "BULL" else candle.high >= price


def _close_is_adverse(candle: Candle, trend_side: str, level: float) -> bool:
    return candle.close < level if trend_side == "BULL" else candle.close > level


def _touches_level_adverse(candle: Candle, trend_side: str, level: float) -> bool:
    return candle.low < level if trend_side == "BULL" else candle.high > level


def _session(hour: int) -> str:
    if hour < 8:
        return "ASIA"
    if hour < 16:
        return "EUROPE"
    return "US"


def _rf_row(features: tuple[object, ...]) -> list[float]:
    trend_map = {"BULL": 1.0, "BEAR": -1.0}
    event_map = {"SWEEP_RECLAIM": 0.0, "BODY_BREAK": 1.0}
    session_map = {"ASIA": 0.0, "EUROPE": 1.0, "US": 2.0}
    weekday_map = {
        "MON": 0.0,
        "TUE": 1.0,
        "WED": 2.0,
        "THU": 3.0,
        "FRI": 4.0,
        "SAT": 5.0,
        "SUN": 6.0,
    }
    categorical = (
        trend_map[str(features[0])],
        event_map[str(features[1])],
        session_map[str(features[2])],
        weekday_map[str(features[3])],
    )
    return [*categorical, *(float(value) for value in features[4:])]


def _aligned_probabilities(
    model_classes,
    probabilities,
) -> tuple[tuple[float, ...], ...]:
    classes = [str(value) for value in model_classes]
    indexes = [classes.index(label) for label in CLEAR_LABELS]
    return tuple(
        tuple(float(row[index]) for index in indexes)
        for row in probabilities
    )


def _model_metrics(
    truth: tuple[str, ...],
    probabilities: tuple[tuple[float, ...], ...],
    metrics,
) -> dict[str, object]:
    predictions = tuple(
        CLEAR_LABELS[
            max(range(len(CLEAR_LABELS)), key=lambda index: row[index])
        ]
        for row in probabilities
    )
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        truth,
        predictions,
        labels=list(CLEAR_LABELS),
        zero_division=0,
    )
    per_class = {}
    for index, label in enumerate(CLEAR_LABELS):
        base = sum(item == label for item in truth) / len(truth)
        class_precision = float(precision[index])
        per_class[label] = {
            "support": int(support[index]),
            "base_rate": float(base),
            "precision": class_precision,
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "precision_lift": float(
                class_precision / base if base > 0 else 0.0
            ),
        }

    reversal_index = CLEAR_LABELS.index("REAL_REVERSAL")
    reversal_truth = tuple(item == "REAL_REVERSAL" for item in truth)
    reversal_probs = tuple(row[reversal_index] for row in probabilities)

    reversal_auc = (
        float(metrics.roc_auc_score(reversal_truth, reversal_probs))
        if len(set(reversal_truth)) > 1
        else 0.5
    )
    reversal_pr_auc = float(
        metrics.average_precision_score(reversal_truth, reversal_probs)
    )
    reversal_brier = float(
        metrics.brier_score_loss(reversal_truth, reversal_probs)
    )
    try:
        multiclass_auc = float(
            metrics.roc_auc_score(
                truth,
                probabilities,
                labels=list(CLEAR_LABELS),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        multiclass_auc = 0.5

    return {
        "samples": len(truth),
        "macro_f1": float(
            metrics.f1_score(
                truth,
                predictions,
                labels=list(CLEAR_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_roc_auc_ovr": multiclass_auc,
        "log_loss": float(
            metrics.log_loss(
                truth,
                probabilities,
                labels=list(CLEAR_LABELS),
            )
        ),
        "per_class": per_class,
        "real_reversal": {
            "roc_auc": reversal_auc,
            "pr_auc": reversal_pr_auc,
            "brier_score": reversal_brier,
            "expected_calibration_error": _calibration_error(
                reversal_truth,
                reversal_probs,
            ),
            "probability_bands": _probability_bands(
                reversal_truth,
                reversal_probs,
            ),
        },
        "confusion_matrix": metrics.confusion_matrix(
            truth,
            predictions,
            labels=list(CLEAR_LABELS),
        ).tolist(),
    }


def _calibration_error(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
    bins: int = 10,
) -> float:
    if not truth:
        return 0.0
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper
            or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = mean(probabilities[index] for index in members)
        observed = mean(1.0 if truth[index] else 0.0 for index in members)
        error += len(members) / len(truth) * abs(confidence - observed)
    return error


def _probability_bands(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
) -> dict[str, dict[str, float | int]]:
    base = sum(truth) / len(truth) if truth else 0.0
    result = {}
    for threshold in (0.50, 0.60, 0.70, 0.80):
        members = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= threshold
        ]
        observed = (
            sum(truth[index] for index in members) / len(members)
            if members
            else 0.0
        )
        result[f">={threshold:.2f}"] = {
            "samples": len(members),
            "observed_reversal_rate": float(observed),
            "lift_vs_base": float(observed / base if base > 0 else 0.0),
        }
    return result


def _top_features(
    importances: tuple[float, ...],
    limit: int = 15,
) -> list[dict[str, float | str]]:
    ranked = sorted(
        zip(FEATURE_NAMES, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return [
        {"feature": name, "importance": float(value)}
        for name, value in ranked
    ]


def _mean_importance(
    importances: list[tuple[float, ...]],
) -> tuple[float, ...]:
    if not importances:
        return tuple(0.0 for _ in FEATURE_NAMES)
    return tuple(
        mean(values)
        for values in zip(*importances, strict=True)
    )


def _top_overlap(
    importances: list[tuple[float, ...]],
    limit: int,
) -> float:
    if len(importances) < 2:
        return 1.0
    sets = []
    for values in importances:
        ranked = sorted(
            range(len(values)),
            key=lambda index: values[index],
            reverse=True,
        )[:limit]
        sets.append(set(ranked))
    overlap = set.intersection(*sets)
    return len(overlap) / limit


def _state_distribution(
    samples: Iterable[MarketObserverSample],
) -> dict[str, int]:
    counts = Counter(sample.state_class for sample in samples)
    return {label: counts.get(label, 0) for label in ALL_LABELS}


def _event_distribution(
    samples: Iterable[MarketObserverSample],
) -> dict[str, int]:
    counts = Counter(sample.event_type for sample in samples)
    return {
        "SWEEP_RECLAIM": counts.get("SWEEP_RECLAIM", 0),
        "BODY_BREAK": counts.get("BODY_BREAK", 0),
    }


def _trend_distribution(
    samples: Iterable[MarketObserverSample],
) -> dict[str, int]:
    counts = Counter(sample.trend_side for sample in samples)
    return {
        "BULL": counts.get("BULL", 0),
        "BEAR": counts.get("BEAR", 0),
    }


def _require_contiguous(candles: tuple[Candle, ...]) -> None:
    for previous, current in pairwise(candles):
        if current.timestamp - previous.timestamp != timedelta(minutes=5):
            raise ValueError(
                "M5 history has a gap between "
                f"{previous.timestamp.isoformat()} and "
                f"{current.timestamp.isoformat()}"
            )


def _load_ml():
    try:
        from catboost import CatBoostClassifier, Pool
        from sklearn import metrics
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Market Observer ML dependencies are not installed. '
            'Run pip install -e ".[ml]".'
        ) from exc
    return CatBoostClassifier, Pool, RandomForestClassifier, metrics
