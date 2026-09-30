from collections.abc import Iterable

from medium_trading.market_observer import MarketObserverSample
from medium_trading.market_observer_v05 import (
    _chronological_split,
    _load_ml,
    _validate_split,
)
from medium_trading.market_observer_v06 import _fit_binary_observer
from medium_trading.market_observer_v07 import FEATURE_NAMES
from medium_trading.observer_policy import (
    ObserverSnapshot,
    observer_snapshot_from_probability,
)


def build_v07_walk_forward_snapshots(
    samples: Iterable[MarketObserverSample],
    *,
    test_years: tuple[int, ...] = (2024, 2025),
) -> tuple[ObserverSnapshot, ...]:
    """
    Produce causal out-of-time v0.7 observer messages for every structural event.

    Training/calibration/threshold selection still use clear labels only. Once the
    model is frozen for a fold, inference is performed on *all* structural events
    in that test year, including events whose eventual research label is
    AMBIGUOUS. This matches live operation and avoids future-label selection.
    """
    (
        CatBoostClassifier,
        Pool,
        LogisticRegression,
        metrics,
    ) = _load_ml()

    all_samples = tuple(sorted(samples, key=lambda item: item.event_time))
    if not all_samples:
        raise ValueError("observer runtime requires confirmed-event samples")

    snapshots: list[ObserverSnapshot] = []

    for year in test_years:
        split = _chronological_split(all_samples, year)
        if split is None:
            continue
        fit, calibration, validation, test_all, test_clear = split
        _validate_split(
            fit,
            calibration,
            validation,
            test_clear,
            year,
        )

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

        prediction = _fit_binary_observer(
            CatBoostClassifier=CatBoostClassifier,
            Pool=Pool,
            LogisticRegression=LogisticRegression,
            fit=fit,
            calibration=calibration,
            validation=validation,
            test=test_all,
            fit_truth=fit_truth,
            calibration_truth=calibration_truth,
            validation_truth=validation_truth,
            feature_names=FEATURE_NAMES,
            feature_limit=None,
            metrics=metrics,
        )
        probabilities = prediction["test_probabilities"]
        threshold = float(prediction["threshold"])

        for sample, probability in zip(
            test_all,
            probabilities,
            strict=True,
        ):
            snapshots.append(
                observer_snapshot_from_probability(
                    timestamp=sample.event_time,
                    trend_side=sample.trend_side,
                    event_type=sample.event_type,
                    reversal_probability=float(probability),
                    reversal_threshold=threshold,
                )
            )

    snapshots.sort(key=lambda item: item.timestamp)
    return tuple(snapshots)



def build_v07_2026_forward_snapshots(
    *,
    development_samples: Iterable[MarketObserverSample],
    forward_samples: Iterable[MarketObserverSample],
) -> tuple[ObserverSnapshot, ...]:
    """
    Produce frozen v0.7 messages for 2026 trading-policy validation.

    Development samples must be entirely pre-2026. Forward samples must be
    entirely inside 2026. The existing chronological fold logic then fits,
    calibrates and selects the observer threshold only from pre-2026 data while
    inference is emitted for every 2026 structural event.
    """
    development = tuple(development_samples)
    forward = tuple(forward_samples)
    if not development:
        raise ValueError("2026 observer runtime requires development samples")
    if not forward:
        raise ValueError("2026 observer runtime requires forward samples")

    if any(sample.event_time.year >= 2026 for sample in development):
        raise ValueError("2026 observer runtime development data must be pre-2026")
    if any(sample.event_time.year != 2026 for sample in forward):
        raise ValueError("2026 observer runtime forward samples must be in 2026")

    snapshots = build_v07_walk_forward_snapshots(
        (*development, *forward),
        test_years=(2026,),
    )
    if not snapshots:
        raise ValueError("2026 observer runtime produced no snapshots")
    if any(snapshot.timestamp.year != 2026 for snapshot in snapshots):
        raise ValueError("2026 observer runtime emitted non-2026 snapshot")
    return snapshots
