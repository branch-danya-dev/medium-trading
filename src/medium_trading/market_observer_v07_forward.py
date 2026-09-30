from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime

from medium_trading.market_observer import MarketObserverSample
from medium_trading.market_observer_v05 import (
    _binary_metrics,
    _binary_ranking_metrics,
    _chronological_split,
    _load_ml,
    _validate_split,
)
from medium_trading.market_observer_v06 import (
    FEATURE_NAMES as V06_FEATURE_NAMES,
)
from medium_trading.market_observer_v06 import (
    _fit_binary_observer,
    _metric_delta,
    _top_features_named,
)
from medium_trading.market_observer_v07 import FEATURE_NAMES
from medium_trading.market_observer_v07_audit import _resolution_summary

FORWARD_START = datetime(2026, 1, 1, tzinfo=UTC)
FORWARD_END = datetime(2026, 9, 30, tzinfo=UTC)
FORWARD_WARMUP_START = date(2025, 11, 1)
FORWARD_LAST_FULL_DAY = date(2026, 9, 29)


def evaluate_market_observer_v07_forward(
    *,
    development_samples: Iterable[MarketObserverSample],
    forward_samples: Iterable[MarketObserverSample],
    resolution_by_time: Mapping[datetime, str | None],
) -> dict[str, object]:
    """
    Run the frozen v0.7 observer on the untouched 2026 forward window.

    Training, Platt calibration and threshold selection are all restricted to
    pre-2026 development samples. The forward test is fixed to
    2026-01-01T00:00Z <= T < 2026-09-30T00:00Z.
    """
    (
        CatBoostClassifier,
        Pool,
        LogisticRegression,
        metrics,
    ) = _load_ml()

    development = tuple(
        sorted(development_samples, key=lambda sample: sample.event_time)
    )
    forward = tuple(
        sorted(forward_samples, key=lambda sample: sample.event_time)
    )
    if not development:
        raise ValueError("v0.7 forward validation requires development samples")
    if not forward:
        raise ValueError("v0.7 forward validation requires 2026 samples")

    if any(sample.event_time >= FORWARD_START for sample in development):
        raise ValueError("development samples must end before 2026-01-01")
    if any(
        sample.event_time < FORWARD_START or sample.event_time >= FORWARD_END
        for sample in forward
    ):
        raise ValueError(
            "forward samples must stay inside the frozen 2026 forward window"
        )
    if any(len(sample.features) != len(FEATURE_NAMES) for sample in development):
        raise ValueError("development samples must contain full v0.7 features")
    if any(len(sample.features) != len(FEATURE_NAMES) for sample in forward):
        raise ValueError("forward samples must contain full v0.7 features")
    for sample in forward:
        if sample.event_time not in resolution_by_time:
            raise ValueError(
                "missing 2026 label-resolution status for "
                f"{sample.event_time.isoformat()}"
            )

    all_samples = tuple(sorted((*development, *forward), key=lambda item: item.event_time))
    split = _chronological_split(all_samples, 2026)
    if split is None:
        raise ValueError("2026 forward split contains no samples")
    fit, calibration, validation, test_all, test = split
    _validate_split(fit, calibration, validation, test, 2026)

    if any(sample.event_time >= FORWARD_START for sample in fit):
        raise ValueError("2026 leaked into model fit")
    if any(sample.event_time >= FORWARD_START for sample in calibration):
        raise ValueError("2026 leaked into probability calibration")
    if any(sample.event_time >= FORWARD_START for sample in validation):
        raise ValueError("2026 leaked into threshold validation")
    if tuple(test_all) != forward:
        raise ValueError(
            "forward split does not exactly match the frozen 2026 sample stream"
        )

    fit_truth = [sample.state_class == "REAL_REVERSAL" for sample in fit]
    calibration_truth = tuple(
        sample.state_class == "REAL_REVERSAL" for sample in calibration
    )
    validation_truth = tuple(
        sample.state_class == "REAL_REVERSAL" for sample in validation
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

    full_control = _binary_metrics(
        test_truth,
        control["test_probabilities"],
        control["threshold"],
        metrics,
    )
    full_confirmed = _binary_metrics(
        test_truth,
        confirmed["test_probabilities"],
        confirmed["threshold"],
        metrics,
    )

    unresolved_indexes = tuple(
        index
        for index, sample in enumerate(test)
        if resolution_by_time[sample.event_time] is None
    )
    if len(unresolved_indexes) < 100:
        raise ValueError(
            "2026 forward validation has fewer than 100 unresolved clear samples"
        )

    unresolved_truth = tuple(test_truth[index] for index in unresolved_indexes)
    unresolved_control_probabilities = tuple(
        control["test_probabilities"][index] for index in unresolved_indexes
    )
    unresolved_confirmed_probabilities = tuple(
        confirmed["test_probabilities"][index] for index in unresolved_indexes
    )

    unresolved_control = _binary_metrics(
        unresolved_truth,
        unresolved_control_probabilities,
        control["threshold"],
        metrics,
    )
    unresolved_confirmed = _binary_metrics(
        unresolved_truth,
        unresolved_confirmed_probabilities,
        confirmed["threshold"],
        metrics,
    )

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "untouched 2026 forward validation of frozen Market Observer v0.7",
            "development_years": [2023, 2024, 2025],
            "forward_start": FORWARD_START.isoformat(),
            "forward_end_exclusive": FORWARD_END.isoformat(),
            "forward_last_full_day": FORWARD_LAST_FULL_DAY.isoformat(),
            "2026_used_for_fit": False,
            "2026_used_for_calibration": False,
            "2026_used_for_threshold_selection": False,
            "model_architecture_changed": False,
            "features_changed": False,
            "labels_changed": False,
            "prediction_delay_minutes": 15,
        },
        "pre_forward_split": {
            "fit_samples": len(fit),
            "calibration_samples": len(calibration),
            "threshold_validation_samples": len(validation),
            "fit_last_label_end": max(sample.label_end_time for sample in fit).isoformat(),
            "calibration_first_event": min(
                sample.event_time for sample in calibration
            ).isoformat(),
            "calibration_last_event": max(
                sample.event_time for sample in calibration
            ).isoformat(),
            "validation_first_event": min(
                sample.event_time for sample in validation
            ).isoformat(),
            "validation_last_event": max(
                sample.event_time for sample in validation
            ).isoformat(),
            "control_threshold": control["threshold"],
            "confirmed_threshold": confirmed["threshold"],
        },
        "forward": {
            "samples": len(test_all),
            "clear_samples": len(test),
            "excluded_samples": len(test_all) - len(test),
            "resolution": _resolution_summary(test, resolution_by_time),
            "full": {
                "control": full_control,
                "confirmed": full_confirmed,
                "delta": _metric_delta(full_control, full_confirmed),
            },
            "unresolved_only": {
                "samples": len(unresolved_indexes),
                "base_rate": sum(unresolved_truth) / len(unresolved_truth),
                "control": unresolved_control,
                "confirmed": unresolved_confirmed,
                "delta": _metric_delta(
                    unresolved_control,
                    unresolved_confirmed,
                ),
            },
            "confirmed_top_features": _top_features_named(
                confirmed["feature_importance"],
                FEATURE_NAMES,
            ),
            "monthly_unresolved": _monthly_metrics(
                test=test,
                indexes=unresolved_indexes,
                truth=test_truth,
                control_probabilities=control["test_probabilities"],
                confirmed_probabilities=confirmed["test_probabilities"],
                control_threshold=control["threshold"],
                confirmed_threshold=confirmed["threshold"],
                metrics=metrics,
            ),
        },
    }


def _monthly_metrics(
    *,
    test: tuple[MarketObserverSample, ...],
    indexes: tuple[int, ...],
    truth: tuple[bool, ...],
    control_probabilities: tuple[float, ...],
    confirmed_probabilities: tuple[float, ...],
    control_threshold: float,
    confirmed_threshold: float,
    metrics,
) -> dict[str, object]:
    months: dict[str, list[int]] = {}
    for index in indexes:
        month = test[index].event_time.strftime("%Y-%m")
        months.setdefault(month, []).append(index)

    payload: dict[str, object] = {}
    for month, month_indexes in sorted(months.items()):
        month_truth = tuple(truth[index] for index in month_indexes)
        month_control = tuple(
            control_probabilities[index] for index in month_indexes
        )
        month_confirmed = tuple(
            confirmed_probabilities[index] for index in month_indexes
        )
        if len(set(month_truth)) < 2:
            payload[month] = {
                "samples": len(month_indexes),
                "base_rate": sum(month_truth) / len(month_truth),
                "scorable": False,
            }
            continue
        payload[month] = {
            "samples": len(month_indexes),
            "base_rate": sum(month_truth) / len(month_truth),
            "scorable": True,
            "control": _binary_metrics(
                month_truth,
                month_control,
                control_threshold,
                metrics,
            ),
            "confirmed": _binary_metrics(
                month_truth,
                month_confirmed,
                confirmed_threshold,
                metrics,
            ),
            "control_ranking": _binary_ranking_metrics(
                month_truth,
                month_control,
                metrics,
            ),
            "confirmed_ranking": _binary_ranking_metrics(
                month_truth,
                month_confirmed,
                metrics,
            ),
        }
    return payload
