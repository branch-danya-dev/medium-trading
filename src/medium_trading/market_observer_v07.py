from bisect import bisect_left
from collections.abc import Iterable
from datetime import timedelta
from statistics import mean

from medium_trading.data.bybit_trade_flow import TradeFlowPoint
from medium_trading.domain import Candle
from medium_trading.market_observer import (
    MarketObserverSample,
    _event_distribution,
    _state_distribution,
    _trend_distribution,
)
from medium_trading.market_observer_v05 import (
    REVERSAL_PARAMS,
    _binary_metrics,
    _binary_ranking_metrics,
    _chronological_split,
    _load_ml,
    _mean_importance,
    _top_overlap,
    _validate_split,
)
from medium_trading.market_observer_v06 import (
    FEATURE_NAMES as V06_FEATURE_NAMES,
)
from medium_trading.market_observer_v06 import (
    _count_imbalance,
    _fit_binary_observer,
    _metric_delta,
    _notional_delta_ratio,
    _qty_delta_ratio,
    _top_features_named,
)

CONFIRMATION_BARS = 3
CONFIRMATION_MINUTES = 15
RETEST_TOLERANCE_ATR = 0.15

CONFIRMATION_FEATURE_NAMES = (
    "confirm_adverse_displacement_atr",
    "confirm_max_adverse_extension_atr",
    "confirm_mean_close_beyond_level_atr",
    "confirm_close_beyond_level_count",
    "confirm_touch_beyond_level_count",
    "confirm_final_reclaimed_level",
    "confirm_reclaim_speed",
    "confirm_retest_happened",
    "confirm_retest_held",
    "confirm_efficiency",
    "confirm_range_atr",
    "confirm_adverse_body_fraction",
    "confirm_volume_ratio_vs_pre15m",
    "confirm_taker_delta_ratio_15m",
    "confirm_taker_notional_delta_ratio_15m",
    "confirm_taker_count_imbalance_15m",
    "confirm_trend_aligned_delta_ratio_15m",
    "confirm_delta_shift_vs_pre15m",
    "confirm_flow_persistence",
    "confirm_trade_count_activity_vs_pre2h",
    "confirm_avg_trade_size_imbalance",
)
FEATURE_NAMES = (*V06_FEATURE_NAMES, *CONFIRMATION_FEATURE_NAMES)


def build_confirmed_observer_samples(
    *,
    samples: Iterable[MarketObserverSample],
    candles_5m: tuple[Candle, ...],
    trade_flow: tuple[TradeFlowPoint, ...],
) -> tuple[MarketObserverSample, ...]:
    """
    Move the observer decision point 15 minutes after the original disturbance.

    The original v0.6 feature vector is preserved exactly. New features use only
    the three M5 candles and three M5 trade-flow buckets that have fully completed
    after the disturbance. The original market label is preserved so v0.7 isolates
    whether waiting for confirmation improves discrimination.
    """
    all_samples = tuple(samples)
    if not all_samples:
        raise ValueError("v0.7 requires v0.6 observer samples")
    if not candles_5m:
        raise ValueError("v0.7 requires M5 candles")
    if not trade_flow:
        raise ValueError("v0.7 requires M5 trade flow")

    candle_times = [candle.timestamp for candle in candles_5m]
    flow_times = [point.timestamp for point in trade_flow]
    confirmed: list[MarketObserverSample] = []

    for sample in all_samples:
        if len(sample.features) != len(V06_FEATURE_NAMES):
            raise ValueError(
                "v0.7 input samples must contain the full frozen v0.6 feature set"
            )

        candle_start = bisect_left(candle_times, sample.event_time)
        if (
            candle_start <= 0
            or candle_start >= len(candles_5m)
            or candle_times[candle_start] != sample.event_time
        ):
            raise ValueError(
                "M5 history does not align with disturbance ending at "
                f"{sample.event_time.isoformat()}"
            )
        confirmation = candles_5m[
            candle_start : candle_start + CONFIRMATION_BARS
        ]
        _require_confirmation_candles(confirmation, sample.event_time)
        observation_time = (
            confirmation[-1].timestamp + timedelta(minutes=5)
        )

        flow_start = bisect_left(flow_times, sample.event_time)
        if (
            flow_start < 24
            or flow_start >= len(trade_flow)
            or flow_times[flow_start] != sample.event_time
        ):
            raise ValueError(
                "trade flow does not align with disturbance ending at "
                f"{sample.event_time.isoformat()}"
            )
        post_flow = trade_flow[
            flow_start : flow_start + CONFIRMATION_BARS
        ]
        _require_confirmation_flow(post_flow, sample.event_time)
        pre_flow_2h = trade_flow[flow_start - 24 : flow_start]
        pre_flow_15m = pre_flow_2h[-3:]

        event_candle = candles_5m[candle_start - 1]
        pre_candles = candles_5m[
            max(0, candle_start - 3) : candle_start
        ]
        if len(pre_candles) != 3:
            raise ValueError(
                "v0.7 requires 15 minutes of M5 history before disturbance"
            )

        confirmation_features = _confirmation_features(
            event_candle=event_candle,
            confirmation=confirmation,
            pre_candles=pre_candles,
            post_flow=post_flow,
            pre_flow_15m=pre_flow_15m,
            pre_flow_2h=pre_flow_2h,
            trend_side=sample.trend_side,
            event_type=sample.event_type,
            defended_level=sample.defended_level,
            atr5=sample.atr5,
        )
        confirmed.append(
            MarketObserverSample(
                event_time=observation_time,
                label_end_time=sample.label_end_time,
                trend_side=sample.trend_side,
                event_type=sample.event_type,
                features=(*sample.features, *confirmation_features),
                state_class=sample.state_class,
                defended_level=sample.defended_level,
                atr5=sample.atr5,
            )
        )

    return tuple(confirmed)


def evaluate_market_observer_v07(
    samples: Iterable[MarketObserverSample],
) -> dict[str, object]:
    """
    Compare the frozen v0.6 observer against the same observer after 15 minutes
    of causal structural/flow confirmation.
    """
    (
        CatBoostClassifier,
        Pool,
        LogisticRegression,
        metrics,
    ) = _load_ml()

    all_samples = tuple(sorted(samples, key=lambda item: item.event_time))
    if not all_samples:
        raise ValueError("market observer v0.7 requires samples")
    if any(len(sample.features) != len(FEATURE_NAMES) for sample in all_samples):
        raise ValueError(
            "v0.7 samples must contain v0.6 features plus all confirmation features"
        )

    folds: list[dict[str, object]] = []
    combined_truth: list[bool] = []
    combined_control: list[float] = []
    combined_confirmed: list[float] = []
    control_importances: list[tuple[float, ...]] = []
    confirmed_importances: list[tuple[float, ...]] = []

    for year in (2024, 2025):
        split = _chronological_split(all_samples, year)
        if split is None:
            continue
        fit, calibration, validation, test_all, test = split
        _validate_split(fit, calibration, validation, test, year)

        fit_truth = [
            sample.state_class == "REAL_REVERSAL" for sample in fit
        ]
        calibration_truth = tuple(
            sample.state_class == "REAL_REVERSAL"
            for sample in calibration
        )
        validation_truth = tuple(
            sample.state_class == "REAL_REVERSAL"
            for sample in validation
        )
        test_truth = tuple(
            sample.state_class == "REAL_REVERSAL" for sample in test
        )

        control = _fit_binary_observer(
            CatBoostClassifier=CatBoostClassifier,
            Pool=Pool,
            LogisticRegression=LogisticRegression,
            fit=fit,
            calibration=calibration,
            validation=validation,
            test=test,
            fit_truth=fit_truth,
            calibration_truth=calibration_truth,
            validation_truth=validation_truth,
            feature_names=V06_FEATURE_NAMES,
            feature_limit=len(V06_FEATURE_NAMES),
            metrics=metrics,
        )
        confirmed = _fit_binary_observer(
            CatBoostClassifier=CatBoostClassifier,
            Pool=Pool,
            LogisticRegression=LogisticRegression,
            fit=fit,
            calibration=calibration,
            validation=validation,
            test=test,
            fit_truth=fit_truth,
            calibration_truth=calibration_truth,
            validation_truth=validation_truth,
            feature_names=FEATURE_NAMES,
            feature_limit=None,
            metrics=metrics,
        )

        control_test = _binary_metrics(
            test_truth,
            control["test_probabilities"],
            control["threshold"],
            metrics,
        )
        confirmed_test = _binary_metrics(
            test_truth,
            confirmed["test_probabilities"],
            confirmed["threshold"],
            metrics,
        )
        control_importance = control["feature_importance"]
        confirmed_importance = confirmed["feature_importance"]
        control_importances.append(control_importance)
        confirmed_importances.append(confirmed_importance)

        folds.append(
            {
                "test_year": year,
                "fit_samples": len(fit),
                "calibration_samples": len(calibration),
                "threshold_validation_samples": len(validation),
                "test_samples": len(test_all),
                "clear_test_samples": len(test),
                "excluded_test_samples": len(test_all) - len(test),
                "test_state_distribution": _state_distribution(test_all),
                "test_event_distribution": _event_distribution(test_all),
                "test_trend_distribution": _trend_distribution(test_all),
                "delayed_control_without_confirmation": {
                    "selected_threshold": control["threshold"],
                    "validation": control["validation_metrics"],
                    "test": control_test,
                    "top_features": _top_features_named(
                        control_importance,
                        V06_FEATURE_NAMES,
                    ),
                },
                "confirmed_event_observer": {
                    "selected_threshold": confirmed["threshold"],
                    "validation": confirmed["validation_metrics"],
                    "test": confirmed_test,
                    "top_features": _top_features_named(
                        confirmed_importance,
                        FEATURE_NAMES,
                    ),
                    "confirmation_feature_importance": (
                        _confirmation_feature_importance(
                            confirmed_importance
                        )
                    ),
                },
                "delta": _metric_delta(
                    control_test,
                    confirmed_test,
                ),
            }
        )

        combined_truth.extend(test_truth)
        combined_control.extend(control["test_probabilities"])
        combined_confirmed.extend(confirmed["test_probabilities"])

    if not folds:
        raise ValueError("no market-observer v0.7 folds contained test samples")

    truth = tuple(combined_truth)
    control_combined = _binary_ranking_metrics(
        truth,
        tuple(combined_control),
        metrics,
    )
    confirmed_combined = _binary_ranking_metrics(
        truth,
        tuple(combined_confirmed),
        metrics,
    )
    combined_test = tuple(
        sample
        for sample in all_samples
        if sample.event_time.year in {2024, 2025}
    )
    confirmed_mean_importance = _mean_importance(confirmed_importances)

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": (
                "test whether waiting for causal post-disturbance confirmation "
                "improves real-reversal observation"
            ),
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [2024, 2025],
            "2026_used": False,
            "trade_state_used": False,
            "sampling_anchor_changed": False,
            "prediction_time_shifted": True,
            "prediction_delay_minutes": CONFIRMATION_MINUTES,
            "confirmation_bars": CONFIRMATION_BARS,
            "labels_changed": False,
            "v0_6_features_changed": False,
            "model_architecture_changed": False,
        },
        "confirmation": {
            "window": "3 fully completed M5 bars after structural disturbance",
            "feature_names": list(CONFIRMATION_FEATURE_NAMES),
            "retest_tolerance_atr": RETEST_TOLERANCE_ATR,
            "label_policy": (
                "preserve original disturbance outcome label; no candle or "
                "trade-flow data after the 15-minute observation time enters features"
            ),
        },
        "model": {
            "type": "CatBoostClassifier",
            "params": dict(REVERSAL_PARAMS),
            "target": "REAL_REVERSAL vs TREND_SURVIVES",
            "calibration": "same past-only Platt calibration as v0.5/v0.6",
            "threshold_selection": "same past-only F1 validation as v0.5/v0.6",
        },
        "folds": folds,
        "combined": {
            "test_samples": len(combined_test),
            "clear_samples": len(truth),
            "excluded_samples": len(combined_test) - len(truth),
            "state_distribution": _state_distribution(combined_test),
            "event_distribution": _event_distribution(combined_test),
            "trend_distribution": _trend_distribution(combined_test),
            "delayed_control_without_confirmation": {
                "ranking_and_calibration": control_combined,
                "top10_fold_overlap": _top_overlap(
                    control_importances,
                    10,
                ),
            },
            "confirmed_event_observer": {
                "ranking_and_calibration": confirmed_combined,
                "top_features": _top_features_named(
                    confirmed_mean_importance,
                    FEATURE_NAMES,
                ),
                "confirmation_feature_importance": (
                    _confirmation_feature_importance(
                        confirmed_mean_importance
                    )
                ),
                "top10_fold_overlap": _top_overlap(
                    confirmed_importances,
                    10,
                ),
            },
            "delta": _metric_delta(
                control_combined,
                confirmed_combined,
            ),
        },
    }


def _confirmation_features(
    *,
    event_candle: Candle,
    confirmation: tuple[Candle, ...],
    pre_candles: tuple[Candle, ...],
    post_flow: tuple[TradeFlowPoint, ...],
    pre_flow_15m: tuple[TradeFlowPoint, ...],
    pre_flow_2h: tuple[TradeFlowPoint, ...],
    trend_side: str,
    event_type: str,
    defended_level: float,
    atr5: float,
) -> tuple[float, ...]:
    if atr5 <= 0:
        raise ValueError("v0.7 ATR must be positive")

    direction = 1.0 if trend_side == "BULL" else -1.0
    observation = confirmation[-1]
    adverse_displacement = (
        -direction * (observation.close - event_candle.close) / atr5
    )
    close_beyond_distances = tuple(
        max(0.0, -direction * (candle.close - defended_level)) / atr5
        for candle in confirmation
    )
    touch_beyond_distances = tuple(
        _adverse_touch_distance(
            candle,
            trend_side=trend_side,
            level=defended_level,
        )
        / atr5
        for candle in confirmation
    )
    closes_beyond = tuple(value > 0 for value in close_beyond_distances)
    touches_beyond = tuple(value > 0 for value in touch_beyond_distances)
    final_reclaimed = (
        direction * (observation.close - defended_level) > 0
    )

    reclaim_speed = 0.0
    for index, candle in enumerate(confirmation, start=1):
        if direction * (candle.close - defended_level) > 0:
            reclaim_speed = 1.0 / index
            break

    retest_happened, retest_held = _retest_state(
        confirmation,
        trend_side=trend_side,
        event_type=event_type,
        level=defended_level,
        tolerance=RETEST_TOLERANCE_ATR * atr5,
    )
    path_closes = [event_candle.close, *(candle.close for candle in confirmation)]
    efficiency = _efficiency(path_closes)
    range_atr = (
        max(candle.high for candle in confirmation)
        - min(candle.low for candle in confirmation)
    ) / atr5
    adverse_body_fraction = (
        sum(
            direction * (candle.close - candle.open) < 0
            for candle in confirmation
        )
        / len(confirmation)
    )
    pre_volume = sum(candle.volume for candle in pre_candles)
    post_volume = sum(candle.volume for candle in confirmation)
    volume_ratio = post_volume / pre_volume if pre_volume > 0 else 1.0

    post_delta = _qty_delta_ratio(post_flow)
    post_notional_delta = _notional_delta_ratio(post_flow)
    post_count_imbalance = _count_imbalance(post_flow)
    pre_delta = _qty_delta_ratio(pre_flow_15m)
    flow_persistence = mean(
        1.0
        if direction * (point.buy_qty - point.sell_qty) > 0
        else -1.0
        if direction * (point.buy_qty - point.sell_qty) < 0
        else 0.0
        for point in post_flow
    )
    post_count_per_bar = mean(point.total_count for point in post_flow)
    pre_count_per_bar = mean(point.total_count for point in pre_flow_2h)
    count_activity = (
        post_count_per_bar / pre_count_per_bar
        if pre_count_per_bar > 0
        else 1.0
    )

    buy_qty = sum(point.buy_qty for point in post_flow)
    sell_qty = sum(point.sell_qty for point in post_flow)
    buy_count = sum(point.buy_count for point in post_flow)
    sell_count = sum(point.sell_count for point in post_flow)
    avg_buy = buy_qty / buy_count if buy_count else 0.0
    avg_sell = sell_qty / sell_count if sell_count else 0.0
    avg_total = avg_buy + avg_sell
    avg_trade_size_imbalance = (
        (avg_buy - avg_sell) / avg_total if avg_total > 0 else 0.0
    )

    return (
        adverse_displacement,
        max(touch_beyond_distances),
        mean(close_beyond_distances),
        float(sum(closes_beyond)),
        float(sum(touches_beyond)),
        1.0 if final_reclaimed else 0.0,
        reclaim_speed,
        1.0 if retest_happened else 0.0,
        1.0 if retest_held else 0.0,
        efficiency,
        range_atr,
        adverse_body_fraction,
        volume_ratio,
        post_delta,
        post_notional_delta,
        post_count_imbalance,
        direction * post_delta,
        post_delta - pre_delta,
        flow_persistence,
        count_activity,
        avg_trade_size_imbalance,
    )


def _retest_state(
    confirmation: tuple[Candle, ...],
    *,
    trend_side: str,
    event_type: str,
    level: float,
    tolerance: float,
) -> tuple[bool, bool]:
    if event_type != "BODY_BREAK":
        return False, False

    for candle in confirmation:
        if trend_side == "BULL":
            touched = candle.high >= level - tolerance
            held = touched and candle.close < level
        else:
            touched = candle.low <= level + tolerance
            held = touched and candle.close > level
        if touched:
            return True, held
    return False, False


def _adverse_touch_distance(
    candle: Candle,
    *,
    trend_side: str,
    level: float,
) -> float:
    if trend_side == "BULL":
        return max(0.0, level - candle.low)
    return max(0.0, candle.high - level)


def _efficiency(closes: list[float]) -> float:
    path = sum(
        abs(current - previous)
        for previous, current in zip(closes, closes[1:], strict=True)
    )
    return abs(closes[-1] - closes[0]) / path if path > 0 else 0.0


def _confirmation_feature_importance(
    importances: tuple[float, ...],
) -> list[dict[str, float | str]]:
    start = len(V06_FEATURE_NAMES)
    values = importances[start:]
    ranked = sorted(
        zip(CONFIRMATION_FEATURE_NAMES, values, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": float(value)}
        for name, value in ranked
    ]


def _require_confirmation_candles(
    candles: tuple[Candle, ...],
    event_time,
) -> None:
    if len(candles) != CONFIRMATION_BARS:
        raise ValueError("v0.7 requires exactly three post-event M5 candles")
    for index, candle in enumerate(candles):
        expected = event_time + timedelta(minutes=5 * index)
        if candle.timestamp != expected:
            raise ValueError(
                "post-event M5 confirmation gap: expected "
                f"{expected.isoformat()}, got {candle.timestamp.isoformat()}"
            )


def _require_confirmation_flow(
    points: tuple[TradeFlowPoint, ...],
    event_time,
) -> None:
    if len(points) != CONFIRMATION_BARS:
        raise ValueError("v0.7 requires exactly three post-event flow buckets")
    for index, point in enumerate(points):
        expected = event_time + timedelta(minutes=5 * index)
        if point.timestamp != expected:
            raise ValueError(
                "post-event trade-flow confirmation gap: expected "
                f"{expected.isoformat()}, got {point.timestamp.isoformat()}"
            )
