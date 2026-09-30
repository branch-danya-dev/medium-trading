from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from math import ceil
from statistics import mean

from medium_trading.market_observer import (
    CAT_FEATURE_INDICES,
    CLEAR_LABELS,
    FEATURE_NAMES,
    MarketObserverSample,
    _event_distribution,
    _state_distribution,
    _top_features,
    _trend_distribution,
)

FIRST_TEST_YEAR = 2024
LAST_TEST_YEAR = 2025
LABEL_EMBARGO_HOURS = 8
CALIBRATION_DAYS = 60
THRESHOLD_VALIDATION_DAYS = 60

REVERSAL_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
}

UNWEIGHTED_REVERSAL_PARAMS = {
    key: value
    for key, value in REVERSAL_PARAMS.items()
    if key != "auto_class_weights"
}

STATE_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
}

PLATT_PARAMS = {
    "C": 1_000_000.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}


def evaluate_market_observer_v05(
    samples: Iterable[MarketObserverSample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> dict[str, object]:
    """
    Evaluate a hierarchical market observer.

    Stage A estimates P(REAL_REVERSAL) with a balanced binary CatBoost model.
    Stage B estimates P(CORRECTION | NOT_REVERSAL) with a separate CatBoost model.
    Both stages are calibrated only on chronological pre-test data. A hard reversal
    operating point is selected only on a later pre-test validation window and is
    diagnostic; live consumers should use the calibrated probabilities.
    """
    (
        CatBoostClassifier,
        Pool,
        LogisticRegression,
        metrics,
    ) = _load_ml()

    all_samples = tuple(sorted(samples, key=lambda item: item.event_time))
    if not all_samples:
        raise ValueError("market observer v0.5 requires samples")

    folds: list[dict[str, object]] = []
    combined_truth: list[str] = []
    combined_hierarchical: list[tuple[float, float, float]] = []
    combined_weighted_reversal: list[float] = []
    combined_unweighted_reversal: list[float] = []
    combined_thresholds: list[float] = []
    reversal_importances: list[tuple[float, ...]] = []
    state_importances: list[tuple[float, ...]] = []

    for year in range(first_test_year, last_test_year + 1):
        split = _chronological_split(all_samples, year)
        if split is None:
            continue
        fit, calibration, validation, test_all, test = split
        _validate_split(fit, calibration, validation, test, year)

        fit_x = [list(sample.features) for sample in fit]
        calibration_x = [list(sample.features) for sample in calibration]
        validation_x = [list(sample.features) for sample in validation]
        test_x = [list(sample.features) for sample in test]

        fit_reversal = [sample.state_class == "REAL_REVERSAL" for sample in fit]
        calibration_reversal = [
            sample.state_class == "REAL_REVERSAL" for sample in calibration
        ]
        validation_reversal = [
            sample.state_class == "REAL_REVERSAL" for sample in validation
        ]
        test_reversal = tuple(
            sample.state_class == "REAL_REVERSAL" for sample in test
        )

        fit_pool = Pool(
            data=fit_x,
            label=fit_reversal,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        calibration_pool = Pool(
            data=calibration_x,
            label=calibration_reversal,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        validation_pool = Pool(
            data=validation_x,
            label=validation_reversal,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        test_pool = Pool(
            data=test_x,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )

        weighted = CatBoostClassifier(**REVERSAL_PARAMS)
        weighted.fit(fit_pool)
        weighted_calibrator = _fit_platt_calibrator(
            model=weighted,
            pool=calibration_pool,
            truth=tuple(calibration_reversal),
            LogisticRegression=LogisticRegression,
        )
        weighted_validation_probabilities = _calibrated_probabilities(
            model=weighted,
            pool=validation_pool,
            calibrator=weighted_calibrator,
        )
        reversal_threshold = _select_f1_threshold(
            tuple(validation_reversal),
            weighted_validation_probabilities,
            metrics,
        )
        weighted_test_probabilities = _calibrated_probabilities(
            model=weighted,
            pool=test_pool,
            calibrator=weighted_calibrator,
        )

        unweighted = CatBoostClassifier(**UNWEIGHTED_REVERSAL_PARAMS)
        unweighted.fit(fit_pool)
        unweighted_calibrator = _fit_platt_calibrator(
            model=unweighted,
            pool=calibration_pool,
            truth=tuple(calibration_reversal),
            LogisticRegression=LogisticRegression,
        )
        unweighted_validation_probabilities = _calibrated_probabilities(
            model=unweighted,
            pool=validation_pool,
            calibrator=unweighted_calibrator,
        )
        unweighted_threshold = _select_f1_threshold(
            tuple(validation_reversal),
            unweighted_validation_probabilities,
            metrics,
        )
        unweighted_test_probabilities = _calibrated_probabilities(
            model=unweighted,
            pool=test_pool,
            calibrator=unweighted_calibrator,
        )

        fit_non_reversal = tuple(
            sample for sample in fit if sample.state_class != "REAL_REVERSAL"
        )
        calibration_non_reversal = tuple(
            sample
            for sample in calibration
            if sample.state_class != "REAL_REVERSAL"
        )
        validation_non_reversal = tuple(
            sample
            for sample in validation
            if sample.state_class != "REAL_REVERSAL"
        )
        test_non_reversal_indexes = tuple(
            index
            for index, sample in enumerate(test)
            if sample.state_class != "REAL_REVERSAL"
        )

        state_fit_pool = Pool(
            data=[list(sample.features) for sample in fit_non_reversal],
            label=[
                sample.state_class == "CORRECTION"
                for sample in fit_non_reversal
            ],
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        state_calibration_pool = Pool(
            data=[list(sample.features) for sample in calibration_non_reversal],
            label=[
                sample.state_class == "CORRECTION"
                for sample in calibration_non_reversal
            ],
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        state_validation_pool = Pool(
            data=[list(sample.features) for sample in validation_non_reversal],
            label=[
                sample.state_class == "CORRECTION"
                for sample in validation_non_reversal
            ],
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )
        state_test_pool = Pool(
            data=test_x,
            cat_features=list(CAT_FEATURE_INDICES),
            feature_names=list(FEATURE_NAMES),
        )

        state_model = CatBoostClassifier(**STATE_PARAMS)
        state_model.fit(state_fit_pool)
        state_calibrator = _fit_platt_calibrator(
            model=state_model,
            pool=state_calibration_pool,
            truth=tuple(
                sample.state_class == "CORRECTION"
                for sample in calibration_non_reversal
            ),
            LogisticRegression=LogisticRegression,
        )
        state_validation_probabilities = _calibrated_probabilities(
            model=state_model,
            pool=state_validation_pool,
            calibrator=state_calibrator,
        )
        state_test_probabilities = _calibrated_probabilities(
            model=state_model,
            pool=state_test_pool,
            calibrator=state_calibrator,
        )

        hierarchical_probabilities = tuple(
            (
                (1.0 - p_reversal) * (1.0 - p_correction),
                (1.0 - p_reversal) * p_correction,
                p_reversal,
            )
            for p_reversal, p_correction in zip(
                weighted_test_probabilities,
                state_test_probabilities,
                strict=True,
            )
        )
        truth = tuple(sample.state_class for sample in test)

        weighted_importance = tuple(
            float(value) for value in weighted.get_feature_importance()
        )
        state_importance = tuple(
            float(value) for value in state_model.get_feature_importance()
        )
        reversal_importances.append(weighted_importance)
        state_importances.append(state_importance)
        combined_truth.extend(truth)
        combined_hierarchical.extend(hierarchical_probabilities)
        combined_weighted_reversal.extend(weighted_test_probabilities)
        combined_unweighted_reversal.extend(unweighted_test_probabilities)
        combined_thresholds.append(reversal_threshold)

        fold_payload = {
            "test_year": year,
            "fit_samples": len(fit),
            "calibration_samples": len(calibration),
            "threshold_validation_samples": len(validation),
            "test_samples": len(test_all),
            "clear_test_samples": len(test),
            "excluded_test_samples": len(test_all) - len(test),
            "fit_state_distribution": _state_distribution(fit),
            "calibration_state_distribution": _state_distribution(calibration),
            "threshold_validation_state_distribution": _state_distribution(
                validation
            ),
            "test_state_distribution": _state_distribution(test_all),
            "test_event_distribution": _event_distribution(test_all),
            "test_trend_distribution": _trend_distribution(test_all),
            "weighted_reversal": {
                "selected_threshold": reversal_threshold,
                "validation": _binary_metrics(
                    tuple(validation_reversal),
                    weighted_validation_probabilities,
                    reversal_threshold,
                    metrics,
                ),
                "test": _binary_metrics(
                    test_reversal,
                    weighted_test_probabilities,
                    reversal_threshold,
                    metrics,
                ),
                "top_features": _top_features(weighted_importance),
            },
            "unweighted_reversal_control": {
                "selected_threshold": unweighted_threshold,
                "validation": _binary_metrics(
                    tuple(validation_reversal),
                    unweighted_validation_probabilities,
                    unweighted_threshold,
                    metrics,
                ),
                "test": _binary_metrics(
                    test_reversal,
                    unweighted_test_probabilities,
                    unweighted_threshold,
                    metrics,
                ),
            },
            "noise_correction": {
                "validation": _binary_metrics(
                    tuple(
                        sample.state_class == "CORRECTION"
                        for sample in validation_non_reversal
                    ),
                    state_validation_probabilities,
                    0.50,
                    metrics,
                ),
                "test": _binary_metrics(
                    tuple(
                        test[index].state_class == "CORRECTION"
                        for index in test_non_reversal_indexes
                    ),
                    tuple(
                        state_test_probabilities[index]
                        for index in test_non_reversal_indexes
                    ),
                    0.50,
                    metrics,
                ),
                "top_features": _top_features(state_importance),
            },
            "hierarchical": _hierarchical_metrics(
                truth,
                hierarchical_probabilities,
                metrics,
            ),
        }
        folds.append(fold_payload)

    if not folds:
        raise ValueError("no market-observer v0.5 folds contained test samples")

    combined_test = tuple(
        sample
        for sample in all_samples
        if first_test_year <= sample.event_time.year <= last_test_year
    )
    combined_reversal_truth = tuple(
        label == "REAL_REVERSAL" for label in combined_truth
    )
    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": (
                "hierarchical market-only observer with explicit rare-reversal "
                "handling"
            ),
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [
                fold["test_year"] for fold in folds
            ],
            "2026_used": False,
            "trade_state_used": False,
            "sampling": (
                "unchanged v0.4 first causal structural disturbance per defended "
                "M5 swing"
            ),
            "labels_changed": False,
            "features_changed": False,
            "label_embargo_hours": LABEL_EMBARGO_HOURS,
            "calibration_days": CALIBRATION_DAYS,
            "threshold_validation_days": THRESHOLD_VALIDATION_DAYS,
        },
        "architecture": {
            "stage_1": "REAL_REVERSAL vs NOISE+CORRECTION",
            "stage_1_weighting": "CatBoost auto_class_weights=Balanced",
            "stage_1_control": "same binary CatBoost without class weighting",
            "stage_2": "CORRECTION vs NOISE conditional on NOT_REVERSAL",
            "calibration": (
                "Platt-style logistic calibration on a dedicated chronological "
                "pre-test window"
            ),
            "threshold_policy": (
                "diagnostic REAL_REVERSAL threshold maximizes F1 on a later "
                "chronological pre-test validation window; never on test years"
            ),
            "output": (
                "calibrated P(NOISE), P(CORRECTION), P(REAL_REVERSAL); no "
                "BUY/SELL/HOLD/EXIT action"
            ),
        },
        "models": {
            "weighted_reversal": {
                "type": "CatBoostClassifier",
                "params": dict(REVERSAL_PARAMS),
            },
            "unweighted_reversal_control": {
                "type": "CatBoostClassifier",
                "params": dict(UNWEIGHTED_REVERSAL_PARAMS),
            },
            "noise_correction": {
                "type": "CatBoostClassifier",
                "params": dict(STATE_PARAMS),
            },
            "platt_calibrator": {
                "type": "LogisticRegression",
                "params": dict(PLATT_PARAMS),
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
            "weighted_reversal": {
                "fold_thresholds": combined_thresholds,
                "ranking_and_calibration": _binary_ranking_metrics(
                    combined_reversal_truth,
                    tuple(combined_weighted_reversal),
                    metrics,
                ),
                "top_features": _top_features(
                    _mean_importance(reversal_importances)
                ),
                "top10_fold_overlap": _top_overlap(
                    reversal_importances,
                    10,
                ),
            },
            "unweighted_reversal_control": {
                "ranking_and_calibration": _binary_ranking_metrics(
                    combined_reversal_truth,
                    tuple(combined_unweighted_reversal),
                    metrics,
                ),
            },
            "hierarchical": _hierarchical_metrics(
                tuple(combined_truth),
                tuple(combined_hierarchical),
                metrics,
            ),
            "noise_correction": {
                "top_features": _top_features(
                    _mean_importance(state_importances)
                ),
                "top10_fold_overlap": _top_overlap(
                    state_importances,
                    10,
                ),
            },
        },
    }


def _chronological_split(
    all_samples: tuple[MarketObserverSample, ...],
    test_year: int,
) -> tuple[
    tuple[MarketObserverSample, ...],
    tuple[MarketObserverSample, ...],
    tuple[MarketObserverSample, ...],
    tuple[MarketObserverSample, ...],
    tuple[MarketObserverSample, ...],
] | None:
    test_start = datetime(test_year, 1, 1, tzinfo=UTC)
    test_end = datetime(test_year + 1, 1, 1, tzinfo=UTC)
    history_end = test_start - timedelta(hours=LABEL_EMBARGO_HOURS)
    threshold_start = history_end - timedelta(
        days=THRESHOLD_VALIDATION_DAYS
    )
    calibration_start = threshold_start - timedelta(days=CALIBRATION_DAYS)

    fit = tuple(
        sample
        for sample in all_samples
        if sample.is_clear and sample.label_end_time <= calibration_start
    )
    calibration = tuple(
        sample
        for sample in all_samples
        if (
            sample.is_clear
            and calibration_start <= sample.event_time < threshold_start
            and sample.label_end_time <= threshold_start
        )
    )
    validation = tuple(
        sample
        for sample in all_samples
        if (
            sample.is_clear
            and threshold_start <= sample.event_time < history_end
            and sample.label_end_time <= history_end
        )
    )
    test_all = tuple(
        sample
        for sample in all_samples
        if test_start <= sample.event_time < test_end
    )
    if not test_all:
        return None
    test = tuple(sample for sample in test_all if sample.is_clear)
    return fit, calibration, validation, test_all, test


def _validate_split(
    fit: tuple[MarketObserverSample, ...],
    calibration: tuple[MarketObserverSample, ...],
    validation: tuple[MarketObserverSample, ...],
    test: tuple[MarketObserverSample, ...],
    year: int,
) -> None:
    if len(fit) < 300:
        raise ValueError(
            f"not enough pre-{year} fit samples: {len(fit)}; need at least 300"
        )
    if len(calibration) < 60:
        raise ValueError(
            f"not enough pre-{year} calibration samples: "
            f"{len(calibration)}; need at least 60"
        )
    if len(validation) < 60:
        raise ValueError(
            f"not enough pre-{year} threshold-validation samples: "
            f"{len(validation)}; need at least 60"
        )
    if len(test) < 100:
        raise ValueError(
            f"not enough clear {year} test samples: {len(test)}; need at least 100"
        )

    for name, values in (
        ("fit", fit),
        ("calibration", calibration),
        ("threshold validation", validation),
    ):
        classes = {sample.state_class for sample in values}
        if classes != set(CLEAR_LABELS):
            raise ValueError(
                f"pre-{year} {name} data does not contain all clear classes"
            )


def _fit_platt_calibrator(
    *,
    model,
    pool,
    truth: tuple[bool, ...],
    LogisticRegression,
):
    if len(set(truth)) < 2:
        raise ValueError("calibration data must contain both binary classes")
    raw_scores = _raw_scores(model, pool)
    calibrator = LogisticRegression(**PLATT_PARAMS)
    calibrator.fit([[score] for score in raw_scores], truth)
    return calibrator


def _calibrated_probabilities(
    *,
    model,
    pool,
    calibrator,
) -> tuple[float, ...]:
    raw_scores = _raw_scores(model, pool)
    probabilities = calibrator.predict_proba(
        [[score] for score in raw_scores]
    )
    classes = list(calibrator.classes_)
    positive_index = classes.index(True)
    return tuple(float(row[positive_index]) for row in probabilities)


def _raw_scores(model, pool) -> tuple[float, ...]:
    values = model.predict(pool, prediction_type="RawFormulaVal")
    return tuple(float(value) for value in values)


def _select_f1_threshold(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
    metrics,
) -> float:
    precision, recall, thresholds = metrics.precision_recall_curve(
        truth,
        probabilities,
    )
    if len(thresholds) == 0:
        return 0.50

    best_threshold = 0.50
    best_f1 = -1.0
    for index, threshold in enumerate(thresholds):
        denominator = precision[index] + recall[index]
        f1 = (
            2.0 * precision[index] * recall[index] / denominator
            if denominator > 0
            else 0.0
        )
        threshold_value = float(threshold)
        if f1 > best_f1 or (
            f1 == best_f1 and threshold_value > best_threshold
        ):
            best_f1 = f1
            best_threshold = threshold_value
    return best_threshold


def _binary_metrics(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
    threshold: float,
    metrics,
) -> dict[str, object]:
    payload = _binary_ranking_metrics(truth, probabilities, metrics)
    predicted = tuple(
        probability >= threshold for probability in probabilities
    )
    tp = sum(
        actual and guess
        for actual, guess in zip(truth, predicted, strict=True)
    )
    fp = sum(
        (not actual) and guess
        for actual, guess in zip(truth, predicted, strict=True)
    )
    fn = sum(
        actual and (not guess)
        for actual, guess in zip(truth, predicted, strict=True)
    )
    tn = len(truth) - tp - fp - fn
    selected = tp + fp
    actual = tp + fn
    base = actual / len(truth) if truth else 0.0
    precision = tp / selected if selected else 0.0
    recall = tp / actual if actual else 0.0
    f1_denominator = precision + recall
    f1 = (
        2.0 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )
    payload["operating_point"] = {
        "threshold": threshold,
        "selected": selected,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_lift": precision / base if base > 0 else 0.0,
    }
    return payload


def _binary_ranking_metrics(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
    metrics,
) -> dict[str, object]:
    base = sum(truth) / len(truth) if truth else 0.0
    roc_auc = (
        float(metrics.roc_auc_score(truth, probabilities))
        if len(set(truth)) > 1
        else 0.5
    )
    pr_auc = float(
        metrics.average_precision_score(truth, probabilities)
    )
    return {
        "samples": len(truth),
        "base_rate": base,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_lift_vs_base": pr_auc / base if base > 0 else 0.0,
        "brier_score": float(
            metrics.brier_score_loss(truth, probabilities)
        ),
        "log_loss": float(metrics.log_loss(truth, probabilities)),
        "expected_calibration_error": _calibration_error(
            truth,
            probabilities,
        ),
        "probability_quantiles": _probability_quantiles(probabilities),
        "top_fraction_lift": _top_fraction_lift(
            truth,
            probabilities,
        ),
    }


def _hierarchical_metrics(
    truth: tuple[str, ...],
    probabilities: tuple[tuple[float, float, float], ...],
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
    class_ovr_auc = {}
    for index, label in enumerate(CLEAR_LABELS):
        base = sum(item == label for item in truth) / len(truth)
        class_precision = float(precision[index])
        binary_truth = tuple(item == label for item in truth)
        class_probabilities = tuple(row[index] for row in probabilities)
        class_ovr_auc[label] = (
            float(metrics.roc_auc_score(binary_truth, class_probabilities))
            if len(set(binary_truth)) > 1
            else 0.5
        )
        per_class[label] = {
            "support": int(support[index]),
            "base_rate": base,
            "precision": class_precision,
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "precision_lift": (
                class_precision / base if base > 0 else 0.0
            ),
            "roc_auc_ovr": class_ovr_auc[label],
            "pr_auc_ovr": float(
                metrics.average_precision_score(
                    binary_truth,
                    class_probabilities,
                )
            ),
        }

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
        "macro_roc_auc_ovr": mean(class_ovr_auc.values()),
        "log_loss": float(
            metrics.log_loss(
                truth,
                probabilities,
                labels=list(CLEAR_LABELS),
            )
        ),
        "per_class": per_class,
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
            if (
                lower <= probability < upper
                or (bin_index == bins - 1 and probability == 1.0)
            )
        ]
        if not members:
            continue
        confidence = mean(probabilities[index] for index in members)
        observed = mean(1.0 if truth[index] else 0.0 for index in members)
        error += len(members) / len(truth) * abs(confidence - observed)
    return error


def _probability_quantiles(
    probabilities: tuple[float, ...],
) -> dict[str, float]:
    if not probabilities:
        return {}
    ordered = sorted(probabilities)
    return {
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _quantile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _top_fraction_lift(
    truth: tuple[bool, ...],
    probabilities: tuple[float, ...],
) -> dict[str, dict[str, float | int]]:
    if not truth:
        return {}
    base = sum(truth) / len(truth)
    ranking = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )
    result = {}
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, ceil(len(ranking) * fraction))
        members = ranking[:count]
        observed = sum(truth[index] for index in members) / count
        result[f"top_{int(fraction * 100)}pct"] = {
            "samples": count,
            "observed_positive_rate": observed,
            "lift_vs_base": observed / base if base > 0 else 0.0,
            "minimum_probability": probabilities[members[-1]],
        }
    return result


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
    return len(set.intersection(*sets)) / limit


def _load_ml():
    try:
        from catboost import CatBoostClassifier, Pool
        from sklearn import metrics
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Market Observer v0.5 dependencies are not installed. '
            'Run pip install -e ".[ml]".'
        ) from exc
    return CatBoostClassifier, Pool, LogisticRegression, metrics
