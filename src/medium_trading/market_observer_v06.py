from bisect import bisect_right
from collections.abc import Iterable
from datetime import timedelta
from statistics import mean

from medium_trading.data.bybit_trade_flow import TradeFlowPoint
from medium_trading.market_observer import (
    CAT_FEATURE_INDICES,
    MarketObserverSample,
    _event_distribution,
    _state_distribution,
    _top_features,
    _trend_distribution,
)
from medium_trading.market_observer import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
)
from medium_trading.market_observer_v05 import (
    REVERSAL_PARAMS,
    _binary_metrics,
    _binary_ranking_metrics,
    _calibrated_probabilities,
    _chronological_split,
    _fit_platt_calibrator,
    _load_ml,
    _mean_importance,
    _select_f1_threshold,
    _top_overlap,
    _validate_split,
)

FLOW_FEATURE_NAMES = (
    "taker_delta_ratio_5m",
    "taker_delta_ratio_15m",
    "taker_delta_ratio_30m",
    "taker_delta_ratio_2h",
    "taker_notional_delta_ratio_5m",
    "taker_notional_delta_ratio_30m",
    "taker_count_imbalance_5m",
    "taker_count_imbalance_30m",
    "trend_aligned_delta_ratio_5m",
    "trend_aligned_delta_ratio_30m",
    "trend_aligned_delta_ratio_2h",
    "delta_acceleration_5m_vs_30m",
    "trade_count_activity_5m_vs_2h",
    "avg_trade_size_imbalance_5m",
    "aggressive_side_persistence_30m",
)
FEATURE_NAMES = (*BASE_FEATURE_NAMES, *FLOW_FEATURE_NAMES)
_MIN_FLOW_BARS = 48


def augment_market_observer_samples_with_trade_flow(
    samples: Iterable[MarketObserverSample],
    trade_flow: tuple[TradeFlowPoint, ...],
) -> tuple[MarketObserverSample, ...]:
    if not trade_flow:
        raise ValueError("v0.6 requires trade-flow data")

    flow_ends = [
        point.timestamp + timedelta(minutes=5)
        for point in trade_flow
    ]
    augmented: list[MarketObserverSample] = []

    for sample in samples:
        end = bisect_right(flow_ends, sample.event_time)
        if end == 0 or flow_ends[end - 1] != sample.event_time:
            raise ValueError(
                "trade-flow data does not contain the completed M5 bucket "
                f"ending at {sample.event_time.isoformat()}"
            )
        if end < _MIN_FLOW_BARS:
            raise ValueError(
                "trade-flow history requires at least 4 hours before "
                f"{sample.event_time.isoformat()}"
            )

        history = trade_flow[end - _MIN_FLOW_BARS : end]
        flow_features = _flow_features(
            history,
            trend_side=sample.trend_side,
        )
        augmented.append(
            MarketObserverSample(
                event_time=sample.event_time,
                label_end_time=sample.label_end_time,
                trend_side=sample.trend_side,
                event_type=sample.event_type,
                features=(*sample.features, *flow_features),
                state_class=sample.state_class,
                defended_level=sample.defended_level,
                atr5=sample.atr5,
            )
        )

    return tuple(augmented)


def evaluate_market_observer_v06(
    samples: Iterable[MarketObserverSample],
) -> dict[str, object]:
    """
    Compare the frozen v0.5 binary reversal observer with an otherwise identical
    model that receives genuine Bybit taker-flow features.
    """
    (
        CatBoostClassifier,
        Pool,
        LogisticRegression,
        metrics,
    ) = _load_ml()

    all_samples = tuple(sorted(samples, key=lambda item: item.event_time))
    if not all_samples:
        raise ValueError("market observer v0.6 requires samples")
    expected_feature_count = len(FEATURE_NAMES)
    if any(len(sample.features) != expected_feature_count for sample in all_samples):
        raise ValueError(
            "v0.6 samples must contain the frozen v0.4 features plus "
            "all trade-flow features"
        )

    folds: list[dict[str, object]] = []
    combined_truth: list[bool] = []
    combined_baseline: list[float] = []
    combined_flow: list[float] = []
    baseline_importances: list[tuple[float, ...]] = []
    flow_importances: list[tuple[float, ...]] = []

    for year in (2024, 2025):
        split = _chronological_split(all_samples, year)
        if split is None:
            continue
        fit, calibration, validation, test_all, test = split
        _validate_split(fit, calibration, validation, test, year)

        truth_fit = [
            sample.state_class == "REAL_REVERSAL" for sample in fit
        ]
        truth_calibration = tuple(
            sample.state_class == "REAL_REVERSAL"
            for sample in calibration
        )
        truth_validation = tuple(
            sample.state_class == "REAL_REVERSAL"
            for sample in validation
        )
        truth_test = tuple(
            sample.state_class == "REAL_REVERSAL" for sample in test
        )

        baseline = _fit_binary_observer(
            CatBoostClassifier=CatBoostClassifier,
            Pool=Pool,
            LogisticRegression=LogisticRegression,
            fit=fit,
            calibration=calibration,
            validation=validation,
            test=test,
            fit_truth=truth_fit,
            calibration_truth=truth_calibration,
            validation_truth=truth_validation,
            feature_names=BASE_FEATURE_NAMES,
            feature_limit=len(BASE_FEATURE_NAMES),
            metrics=metrics,
        )
        enhanced = _fit_binary_observer(
            CatBoostClassifier=CatBoostClassifier,
            Pool=Pool,
            LogisticRegression=LogisticRegression,
            fit=fit,
            calibration=calibration,
            validation=validation,
            test=test,
            fit_truth=truth_fit,
            calibration_truth=truth_calibration,
            validation_truth=truth_validation,
            feature_names=FEATURE_NAMES,
            feature_limit=None,
            metrics=metrics,
        )

        baseline_test = _binary_metrics(
            truth_test,
            baseline["test_probabilities"],
            baseline["threshold"],
            metrics,
        )
        flow_test = _binary_metrics(
            truth_test,
            enhanced["test_probabilities"],
            enhanced["threshold"],
            metrics,
        )
        baseline_importance = baseline["feature_importance"]
        flow_importance = enhanced["feature_importance"]
        baseline_importances.append(baseline_importance)
        flow_importances.append(flow_importance)

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
                "baseline_v0_5_reversal": {
                    "selected_threshold": baseline["threshold"],
                    "validation": baseline["validation_metrics"],
                    "test": baseline_test,
                    "top_features": _top_features(baseline_importance),
                },
                "flow_enhanced_reversal": {
                    "selected_threshold": enhanced["threshold"],
                    "validation": enhanced["validation_metrics"],
                    "test": flow_test,
                    "top_features": _top_features_named(
                        flow_importance,
                        FEATURE_NAMES,
                    ),
                    "flow_feature_importance": _flow_feature_importance(
                        flow_importance
                    ),
                },
                "delta": _metric_delta(baseline_test, flow_test),
            }
        )

        combined_truth.extend(truth_test)
        combined_baseline.extend(baseline["test_probabilities"])
        combined_flow.extend(enhanced["test_probabilities"])

    if not folds:
        raise ValueError("no market-observer v0.6 folds contained test samples")

    truth_combined = tuple(combined_truth)
    baseline_combined = _binary_ranking_metrics(
        truth_combined,
        tuple(combined_baseline),
        metrics,
    )
    flow_combined = _binary_ranking_metrics(
        truth_combined,
        tuple(combined_flow),
        metrics,
    )
    combined_test = tuple(
        sample for sample in all_samples if sample.event_time.year in {2024, 2025}
    )
    flow_mean_importance = _mean_importance(flow_importances)

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "test incremental information from genuine taker trade flow",
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [2024, 2025],
            "2026_used": False,
            "trade_state_used": False,
            "sampling_changed": False,
            "labels_changed": False,
            "base_features_changed": False,
            "model_architecture_changed": False,
            "incremental_input": "Bybit public historical trade tape aggregated causally to M5",
        },
        "trade_flow": {
            "source": "https://public.bybit.com/trading/BTCUSDT/",
            "raw_ticks_persisted": False,
            "aggregation": "completed UTC M5 buckets",
            "side_semantics": "Buy/Sell taker side",
            "feature_names": list(FLOW_FEATURE_NAMES),
        },
        "model": {
            "type": "CatBoostClassifier",
            "params": dict(REVERSAL_PARAMS),
            "target": "REAL_REVERSAL vs TREND_SURVIVES",
            "calibration": "same past-only Platt calibration as v0.5",
            "threshold_selection": "same past-only F1 validation as v0.5",
        },
        "folds": folds,
        "combined": {
            "test_samples": len(combined_test),
            "clear_samples": len(truth_combined),
            "excluded_samples": len(combined_test) - len(truth_combined),
            "state_distribution": _state_distribution(combined_test),
            "event_distribution": _event_distribution(combined_test),
            "trend_distribution": _trend_distribution(combined_test),
            "baseline_v0_5_reversal": {
                "ranking_and_calibration": baseline_combined,
                "top10_fold_overlap": _top_overlap(
                    baseline_importances,
                    10,
                ),
            },
            "flow_enhanced_reversal": {
                "ranking_and_calibration": flow_combined,
                "top_features": _top_features_named(
                    flow_mean_importance,
                    FEATURE_NAMES,
                ),
                "flow_feature_importance": _flow_feature_importance(
                    flow_mean_importance
                ),
                "top10_fold_overlap": _top_overlap(
                    flow_importances,
                    10,
                ),
            },
            "delta": _metric_delta(
                baseline_combined,
                flow_combined,
            ),
        },
    }


def _fit_binary_observer(
    *,
    CatBoostClassifier,
    Pool,
    LogisticRegression,
    fit: tuple[MarketObserverSample, ...],
    calibration: tuple[MarketObserverSample, ...],
    validation: tuple[MarketObserverSample, ...],
    test: tuple[MarketObserverSample, ...],
    fit_truth: list[bool],
    calibration_truth: tuple[bool, ...],
    validation_truth: tuple[bool, ...],
    feature_names: tuple[str, ...],
    feature_limit: int | None,
    metrics,
) -> dict[str, object]:
    fit_x = [_feature_row(sample, feature_limit) for sample in fit]
    calibration_x = [
        _feature_row(sample, feature_limit) for sample in calibration
    ]
    validation_x = [
        _feature_row(sample, feature_limit) for sample in validation
    ]
    test_x = [_feature_row(sample, feature_limit) for sample in test]

    fit_pool = Pool(
        data=fit_x,
        label=fit_truth,
        cat_features=list(CAT_FEATURE_INDICES),
        feature_names=list(feature_names),
    )
    calibration_pool = Pool(
        data=calibration_x,
        label=list(calibration_truth),
        cat_features=list(CAT_FEATURE_INDICES),
        feature_names=list(feature_names),
    )
    validation_pool = Pool(
        data=validation_x,
        label=list(validation_truth),
        cat_features=list(CAT_FEATURE_INDICES),
        feature_names=list(feature_names),
    )
    test_pool = Pool(
        data=test_x,
        cat_features=list(CAT_FEATURE_INDICES),
        feature_names=list(feature_names),
    )

    model = CatBoostClassifier(**REVERSAL_PARAMS)
    model.fit(fit_pool)
    calibrator = _fit_platt_calibrator(
        model=model,
        pool=calibration_pool,
        truth=calibration_truth,
        LogisticRegression=LogisticRegression,
    )
    validation_probabilities = _calibrated_probabilities(
        model=model,
        pool=validation_pool,
        calibrator=calibrator,
    )
    threshold = _select_f1_threshold(
        validation_truth,
        validation_probabilities,
        metrics,
    )
    test_probabilities = _calibrated_probabilities(
        model=model,
        pool=test_pool,
        calibrator=calibrator,
    )
    return {
        "threshold": threshold,
        "validation_metrics": _binary_metrics(
            validation_truth,
            validation_probabilities,
            threshold,
            metrics,
        ),
        "test_probabilities": test_probabilities,
        "feature_importance": tuple(
            float(value) for value in model.get_feature_importance()
        ),
    }


def _feature_row(
    sample: MarketObserverSample,
    feature_limit: int | None,
) -> list[object]:
    features = sample.features
    if feature_limit is not None:
        features = features[:feature_limit]
    return list(features)


def _flow_features(
    history: tuple[TradeFlowPoint, ...],
    *,
    trend_side: str,
) -> tuple[float, ...]:
    current = history[-1:]
    last15 = history[-3:]
    last30 = history[-6:]
    last2h = history[-24:]
    direction = 1.0 if trend_side == "BULL" else -1.0

    delta5 = _qty_delta_ratio(current)
    delta15 = _qty_delta_ratio(last15)
    delta30 = _qty_delta_ratio(last30)
    delta2h = _qty_delta_ratio(last2h)
    notional5 = _notional_delta_ratio(current)
    notional30 = _notional_delta_ratio(last30)
    count5 = _count_imbalance(current)
    count30 = _count_imbalance(last30)

    current_count = sum(point.total_count for point in current)
    mean_2h_count = mean(point.total_count for point in last2h)
    count_activity = (
        current_count / mean_2h_count if mean_2h_count > 0 else 1.0
    )

    point = current[0]
    avg_buy = point.buy_qty / point.buy_count if point.buy_count else 0.0
    avg_sell = point.sell_qty / point.sell_count if point.sell_count else 0.0
    average_sum = avg_buy + avg_sell
    avg_size_imbalance = (
        (avg_buy - avg_sell) / average_sum if average_sum > 0 else 0.0
    )
    persistence = mean(
        1.0
        if (item.buy_qty - item.sell_qty) * direction > 0
        else -1.0
        if (item.buy_qty - item.sell_qty) * direction < 0
        else 0.0
        for item in last30
    )

    return (
        delta5,
        delta15,
        delta30,
        delta2h,
        notional5,
        notional30,
        count5,
        count30,
        direction * delta5,
        direction * delta30,
        direction * delta2h,
        delta5 - delta30,
        count_activity,
        avg_size_imbalance,
        persistence,
    )


def _qty_delta_ratio(points: tuple[TradeFlowPoint, ...]) -> float:
    buy = sum(point.buy_qty for point in points)
    sell = sum(point.sell_qty for point in points)
    total = buy + sell
    return (buy - sell) / total if total > 0 else 0.0


def _notional_delta_ratio(points: tuple[TradeFlowPoint, ...]) -> float:
    buy = sum(point.buy_notional for point in points)
    sell = sum(point.sell_notional for point in points)
    total = buy + sell
    return (buy - sell) / total if total > 0 else 0.0


def _count_imbalance(points: tuple[TradeFlowPoint, ...]) -> float:
    buy = sum(point.buy_count for point in points)
    sell = sum(point.sell_count for point in points)
    total = buy + sell
    return (buy - sell) / total if total > 0 else 0.0


def _top_features_named(
    importances: tuple[float, ...],
    names: tuple[str, ...],
    limit: int = 15,
) -> list[dict[str, float | str]]:
    ranked = sorted(
        zip(names, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return [
        {"feature": name, "importance": float(value)}
        for name, value in ranked
    ]


def _flow_feature_importance(
    importances: tuple[float, ...],
) -> list[dict[str, float | str]]:
    start = len(BASE_FEATURE_NAMES)
    values = importances[start:]
    ranked = sorted(
        zip(FLOW_FEATURE_NAMES, values, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": float(value)}
        for name, value in ranked
    ]


def _metric_delta(
    baseline: dict[str, object],
    enhanced: dict[str, object],
) -> dict[str, float]:
    keys = (
        "roc_auc",
        "pr_auc",
        "pr_auc_lift_vs_base",
        "brier_score",
        "expected_calibration_error",
    )
    return {
        key: float(enhanced[key]) - float(baseline[key])
        for key in keys
    }
