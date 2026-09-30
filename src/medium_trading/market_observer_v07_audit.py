from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

from medium_trading.domain import Candle
from medium_trading.market_observer import (
    CORRECTION_BARRIER_ATR,
    TREND_BARRIER_ATR,
    MarketObserverSample,
    _event_distribution,
    _state_distribution,
    _touches_adverse,
    _touches_trend,
    _trend_distribution,
)
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
)
from medium_trading.market_observer_v07 import (
    CONFIRMATION_BARS,
    CONFIRMATION_MINUTES,
    FEATURE_NAMES,
)

_RESOLVED_LABELS = ("NOISE", "CORRECTION", "AMBIGUOUS")


def classify_v07_label_resolution(
    *,
    samples: Iterable[MarketObserverSample],
    candles_5m: tuple[Candle, ...],
) -> dict[datetime, str | None]:
    """
    Determine whether the frozen v0.4 outcome label is already knowable at T+15m.

    NOISE is resolved once the with-trend barrier has been reached before the
    adverse barrier. CORRECTION is resolved once the adverse barrier is reached
    first and the with-trend recovery barrier is subsequently reached. A
    same-bar trend/adverse touch is resolved as AMBIGUOUS. REAL_REVERSAL can
    never be declared resolved at T+15m because its frozen label requires the
    absence of a trend recovery over the full 8-hour label horizon.
    """
    if not candles_5m:
        raise ValueError("v0.7 overlap audit requires M5 candles")

    candle_times = [candle.timestamp for candle in candles_5m]
    result: dict[datetime, str | None] = {}

    for sample in samples:
        prediction_time = sample.event_time
        disturbance_time = prediction_time - timedelta(
            minutes=CONFIRMATION_MINUTES
        )
        event_candle_time = disturbance_time - timedelta(minutes=5)
        event_index = bisect_left(candle_times, event_candle_time)
        if (
            event_index >= len(candles_5m)
            or candle_times[event_index] != event_candle_time
        ):
            raise ValueError(
                "M5 history does not contain disturbance candle starting at "
                f"{event_candle_time.isoformat()}"
            )

        event_candle = candles_5m[event_index]
        direction = 1.0 if sample.trend_side == "BULL" else -1.0
        trend_price = (
            event_candle.close
            + direction * TREND_BARRIER_ATR * sample.atr5
        )
        correction_price = (
            event_candle.close
            - direction * CORRECTION_BARRIER_ATR * sample.atr5
        )

        future = candles_5m[
            event_index + 1 : event_index + 1 + CONFIRMATION_BARS
        ]
        if len(future) != CONFIRMATION_BARS:
            raise ValueError(
                "v0.7 overlap audit requires exactly three post-disturbance bars"
            )
        for index, candle in enumerate(future):
            expected_start = disturbance_time + timedelta(minutes=5 * index)
            if candle.timestamp != expected_start:
                raise ValueError(
                    "post-disturbance M5 gap: expected "
                    f"{expected_start.isoformat()}, got "
                    f"{candle.timestamp.isoformat()}"
                )
            if candle.timestamp + timedelta(minutes=5) > prediction_time:
                raise ValueError(
                    "overlap audit attempted to inspect a candle after prediction"
                )

        result[prediction_time] = _resolved_label_from_prefix(
            future=future,
            trend_side=sample.trend_side,
            trend_price=trend_price,
            correction_price=correction_price,
        )

    return result


def evaluate_market_observer_v07_overlap_audit(
    *,
    samples: Iterable[MarketObserverSample],
    resolution_by_time: Mapping[datetime, str | None],
) -> dict[str, object]:
    """
    Reproduce the frozen v0.7 models and score them on labels unresolved at T+15m.

    Models are not retrained on the unresolved subset. The audit asks whether the
    already-frozen v0.7 predictions still discriminate REAL_REVERSAL among cases
    whose original outcome was not yet fully knowable at prediction time.
    """
    (
        CatBoostClassifier,
        Pool,
        LogisticRegression,
        metrics,
    ) = _load_ml()

    all_samples = tuple(sorted(samples, key=lambda item: item.event_time))
    if not all_samples:
        raise ValueError("v0.7 overlap audit requires samples")
    if any(len(sample.features) != len(FEATURE_NAMES) for sample in all_samples):
        raise ValueError(
            "v0.7 overlap audit requires full confirmed-event feature vectors"
        )

    for sample in all_samples:
        if sample.event_time not in resolution_by_time:
            raise ValueError(
                "missing label-resolution status for "
                f"{sample.event_time.isoformat()}"
            )
        resolved_label = resolution_by_time[sample.event_time]
        if resolved_label is not None:
            if resolved_label not in _RESOLVED_LABELS:
                raise ValueError(
                    f"unsupported early-resolved label {resolved_label!r}"
                )
            if resolved_label != sample.state_class:
                raise ValueError(
                    "early-resolved label disagrees with frozen final label at "
                    f"{sample.event_time.isoformat()}: "
                    f"{resolved_label} != {sample.state_class}"
                )

    folds: list[dict[str, object]] = []
    combined_truth: list[bool] = []
    combined_control: list[float] = []
    combined_confirmed: list[float] = []
    combined_unresolved_truth: list[bool] = []
    combined_unresolved_control: list[float] = []
    combined_unresolved_confirmed: list[float] = []
    combined_clear_samples: list[MarketObserverSample] = []

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

        unresolved_indexes = tuple(
            index
            for index, sample in enumerate(test)
            if resolution_by_time[sample.event_time] is None
        )
        if not unresolved_indexes:
            raise ValueError(f"{year} has no unresolved clear samples at T+15m")

        unresolved_truth = tuple(
            test_truth[index] for index in unresolved_indexes
        )
        unresolved_control = tuple(
            control["test_probabilities"][index]
            for index in unresolved_indexes
        )
        unresolved_confirmed = tuple(
            confirmed["test_probabilities"][index]
            for index in unresolved_indexes
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
        clean_control = _binary_metrics(
            unresolved_truth,
            unresolved_control,
            control["threshold"],
            metrics,
        )
        clean_confirmed = _binary_metrics(
            unresolved_truth,
            unresolved_confirmed,
            confirmed["threshold"],
            metrics,
        )

        resolution = _resolution_summary(test, resolution_by_time)
        folds.append(
            {
                "test_year": year,
                "test_samples": len(test_all),
                "clear_test_samples": len(test),
                "resolution": resolution,
                "full_v0_7": {
                    "control": full_control,
                    "confirmed": full_confirmed,
                    "delta": _metric_delta(
                        full_control,
                        full_confirmed,
                    ),
                },
                "unresolved_only": {
                    "samples": len(unresolved_indexes),
                    "base_rate": (
                        sum(unresolved_truth) / len(unresolved_truth)
                    ),
                    "control": clean_control,
                    "confirmed": clean_confirmed,
                    "delta": _metric_delta(
                        clean_control,
                        clean_confirmed,
                    ),
                },
            }
        )

        combined_truth.extend(test_truth)
        combined_control.extend(control["test_probabilities"])
        combined_confirmed.extend(confirmed["test_probabilities"])
        combined_unresolved_truth.extend(unresolved_truth)
        combined_unresolved_control.extend(unresolved_control)
        combined_unresolved_confirmed.extend(unresolved_confirmed)
        combined_clear_samples.extend(test)

    if not folds:
        raise ValueError("no v0.7 overlap-audit folds contained test samples")

    full_truth = tuple(combined_truth)
    unresolved_truth = tuple(combined_unresolved_truth)
    full_control = _binary_ranking_metrics(
        full_truth,
        tuple(combined_control),
        metrics,
    )
    full_confirmed = _binary_ranking_metrics(
        full_truth,
        tuple(combined_confirmed),
        metrics,
    )
    clean_control = _binary_ranking_metrics(
        unresolved_truth,
        tuple(combined_unresolved_control),
        metrics,
    )
    clean_confirmed = _binary_ranking_metrics(
        unresolved_truth,
        tuple(combined_unresolved_confirmed),
        metrics,
    )

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "audit v0.7 label overlap at the 15-minute prediction time",
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [2024, 2025],
            "2026_used": False,
            "model_retrained_for_clean_subset": False,
            "model_architecture_changed": False,
            "features_changed": False,
            "labels_changed": False,
            "prediction_delay_minutes": CONFIRMATION_MINUTES,
        },
        "resolution_policy": {
            "NOISE": (
                "resolved by T+15 only if the +0.75 ATR with-trend barrier "
                "was reached before the -0.75 ATR adverse barrier"
            ),
            "CORRECTION": (
                "resolved by T+15 only if -0.75 ATR adverse occurred first "
                "and +0.75 ATR with-trend recovery also occurred by T+15"
            ),
            "REAL_REVERSAL": (
                "never considered resolved at T+15 because the frozen label "
                "requires no +0.75 ATR trend recovery over the full 8h horizon"
            ),
            "AMBIGUOUS": (
                "resolved by T+15 only for same-bar trend/adverse barrier touch"
            ),
        },
        "folds": folds,
        "combined": {
            "clear_samples": len(combined_clear_samples),
            "resolution": _resolution_summary(
                tuple(combined_clear_samples),
                resolution_by_time,
            ),
            "full_v0_7": {
                "control": full_control,
                "confirmed": full_confirmed,
                "delta": _metric_delta(full_control, full_confirmed),
            },
            "unresolved_only": {
                "samples": len(unresolved_truth),
                "base_rate": (
                    sum(unresolved_truth) / len(unresolved_truth)
                    if unresolved_truth
                    else 0.0
                ),
                "control": clean_control,
                "confirmed": clean_confirmed,
                "delta": _metric_delta(clean_control, clean_confirmed),
            },
        },
    }


def _resolved_label_from_prefix(
    *,
    future: tuple[Candle, ...],
    trend_side: str,
    trend_price: float,
    correction_price: float,
) -> str | None:
    first_trend: int | None = None
    first_correction: int | None = None

    for offset, candle in enumerate(future):
        trend_touch = _touches_trend(candle, trend_side, trend_price)
        correction_touch = _touches_adverse(
            candle,
            trend_side,
            correction_price,
        )
        if (
            trend_touch
            and correction_touch
            and first_trend is None
            and first_correction is None
        ):
            return "AMBIGUOUS"
        if trend_touch and first_trend is None:
            first_trend = offset
        if correction_touch and first_correction is None:
            first_correction = offset

    if first_trend is not None and (
        first_correction is None or first_trend < first_correction
    ):
        return "NOISE"
    if (
        first_correction is not None
        and first_trend is not None
        and first_correction < first_trend
    ):
        return "CORRECTION"
    return None


def _resolution_summary(
    samples: Iterable[MarketObserverSample],
    resolution_by_time: Mapping[datetime, str | None],
) -> dict[str, object]:
    values = tuple(samples)
    resolved = tuple(
        sample
        for sample in values
        if resolution_by_time[sample.event_time] is not None
    )
    unresolved = tuple(
        sample
        for sample in values
        if resolution_by_time[sample.event_time] is None
    )
    resolved_final_counts = Counter(
        sample.state_class for sample in resolved
    )
    unresolved_final_counts = Counter(
        sample.state_class for sample in unresolved
    )
    return {
        "samples": len(values),
        "resolved_before_prediction": len(resolved),
        "unresolved_at_prediction": len(unresolved),
        "resolved_fraction": (
            len(resolved) / len(values) if values else 0.0
        ),
        "resolved_final_label_distribution": {
            label: resolved_final_counts.get(label, 0)
            for label in ("NOISE", "CORRECTION", "REAL_REVERSAL")
        },
        "unresolved_final_label_distribution": {
            label: unresolved_final_counts.get(label, 0)
            for label in ("NOISE", "CORRECTION", "REAL_REVERSAL")
        },
    }
