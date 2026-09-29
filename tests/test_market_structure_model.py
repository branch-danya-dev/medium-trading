import json
from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle
from medium_trading.market_structure_model import (
    FEATURE_NAMES,
    StructureContext,
    StructuralStateSample,
    _structural_outcome_class,
    evaluate_structural_state_v02,
    evaluation_payload,
)


def _sample(
    *,
    year: int,
    index: int,
    state_class: str,
) -> StructuralStateSample:
    snapshot = datetime(year, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * index)
    reversal = state_class == "CONFIRMED_REVERSAL"
    signal = 1.0 if reversal else -1.0
    features = (signal,) + (0.0,) * (len(FEATURE_NAMES) - 1)
    return StructuralStateSample(
        trade_entry_time=snapshot - timedelta(hours=1),
        snapshot_time=snapshot,
        label_end_time=snapshot + timedelta(hours=8),
        features=features,
        state_class=state_class,
    )


def _structure() -> StructureContext:
    return StructureContext(
        swing_low=95.0,
        swing_high=110.0,
        previous_swing_low=96.0,
        previous_swing_high=112.0,
        swing_low_index=10,
        swing_high_index=20,
        atr5=10.0,
    )


def test_structural_walk_forward_learns_clear_synthetic_state() -> None:
    samples = []
    for year in (2023, 2024, 2025):
        for index in range(250):
            samples.append(
                _sample(
                    year=year,
                    index=index,
                    state_class="CONFIRMED_REVERSAL",
                )
            )
            samples.append(
                _sample(
                    year=year,
                    index=index + 300,
                    state_class="NOISE",
                )
            )
            samples.append(
                _sample(
                    year=year,
                    index=index + 600,
                    state_class="CORRECTION",
                )
            )
        samples.append(
            _sample(
                year=year,
                index=1000,
                state_class="REVERSAL_CANDIDATE",
            )
        )
        samples.append(
            _sample(
                year=year,
                index=1001,
                state_class="AMBIGUOUS",
            )
        )

    evaluation = evaluate_structural_state_v02(samples)

    assert [fold.test_year for fold in evaluation.folds] == [2024, 2025]
    assert evaluation.combined_classification.roc_auc > 0.95
    assert evaluation.combined_classification.precision > 0.90
    assert evaluation.combined_classification.recall > 0.90
    assert evaluation.combined_classification.precision_lift > 2.5
    assert evaluation.combined_excluded_samples == 4

    payload = evaluation_payload(evaluation)
    assert payload["research_scope"]["2026_used"] is False
    assert (
        payload["labels"]["binary_classifier"]
        == "CONFIRMED_REVERSAL vs NOISE/CORRECTION"
    )
    json.dumps(payload, sort_keys=True)


def test_structural_label_marks_fast_reclaim_as_noise() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 101, 94.5, 96, 10),
        Candle(start + timedelta(minutes=5), 96, 106, 96, 105, 10),
    )
    times = [candle.timestamp for candle in candles]

    state = _structural_outcome_class(
        candles_5m=candles,
        m5_times=times,
        snapshot_time=start,
        snapshot_price=100.0,
        risk_distance=10.0,
        structure=_structure(),
    )

    assert state == "NOISE"


def test_structural_label_marks_accepted_break_as_confirmed_reversal() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 100, 92, 93, 20),
        Candle(start + timedelta(minutes=5), 93, 94, 90, 92, 18),
        Candle(start + timedelta(minutes=10), 92, 93, 89, 91, 17),
        Candle(start + timedelta(minutes=15), 91, 92, 88, 90, 15),
        Candle(start + timedelta(minutes=20), 90, 91, 87, 89, 14),
    )
    times = [candle.timestamp for candle in candles]

    state = _structural_outcome_class(
        candles_5m=candles,
        m5_times=times,
        snapshot_time=start,
        snapshot_price=100.0,
        risk_distance=10.0,
        structure=_structure(),
    )

    assert state == "CONFIRMED_REVERSAL"


def test_structural_label_keeps_unconfirmed_break_as_candidate() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 100, 93, 93, 20),
        Candle(start + timedelta(minutes=5), 93, 97, 93, 96, 10),
        Candle(start + timedelta(minutes=10), 96, 99, 95, 98, 9),
    )
    times = [candle.timestamp for candle in candles]

    state = _structural_outcome_class(
        candles_5m=candles,
        m5_times=times,
        snapshot_time=start,
        snapshot_price=100.0,
        risk_distance=10.0,
        structure=_structure(),
    )

    assert state == "REVERSAL_CANDIDATE"
