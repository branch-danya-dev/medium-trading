import json
from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle
from medium_trading.market_state_model import (
    FEATURE_NAMES,
    MarketStateSample,
    _future_state_class,
    evaluate_market_state_v01,
    evaluation_payload,
)


def _sample(*, year: int, index: int, reversal: bool) -> MarketStateSample:
    snapshot = datetime(year, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * index)
    signal = 1.0 if reversal else -1.0
    features = (signal,) + (0.0,) * (len(FEATURE_NAMES) - 1)
    return MarketStateSample(
        trade_entry_time=snapshot - timedelta(hours=1),
        snapshot_time=snapshot,
        label_end_time=snapshot + timedelta(hours=8),
        features=features,
        state_class="REVERSAL" if reversal else "TREND_VALID",
        is_reversal=reversal,
        future_mfe_2h_r=0.2 if reversal else 1.2,
        future_mae_2h_r=1.2 if reversal else 0.2,
    )


def test_market_state_walk_forward_learns_stable_synthetic_state() -> None:
    samples = []
    for year in (2023, 2024, 2025):
        for index in range(300):
            samples.append(_sample(year=year, index=index, reversal=True))
            samples.append(
                _sample(year=year, index=index + 400, reversal=False)
            )

    evaluation = evaluate_market_state_v01(samples)

    assert [fold.test_year for fold in evaluation.folds] == [2024, 2025]
    assert evaluation.combined_classification.roc_auc > 0.95
    assert evaluation.combined_classification.precision > 0.90
    assert evaluation.combined_classification.recall > 0.90
    assert evaluation.combined_classification.precision_lift > 1.8
    assert (
        evaluation.combined_regression.mfe_mae
        < evaluation.combined_regression.mfe_naive_mae
    )
    assert (
        evaluation.combined_regression.mae_mae
        < evaluation.combined_regression.mae_naive_mae
    )

    payload = evaluation_payload(evaluation)
    assert payload["research_scope"]["2026_used"] is False
    assert payload["labels"]["binary_classifier"] == "REVERSAL vs all other states"
    json.dumps(payload, sort_keys=True)


def test_future_state_class_marks_recovery_after_adverse_as_noise() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 101, 94, 95, 1),
        Candle(start + timedelta(minutes=5), 95, 106, 95, 105, 1),
    )
    times = [candle.timestamp for candle in candles]

    state = _future_state_class(
        candles_5m=candles,
        m5_times=times,
        snapshot_time=start,
        snapshot_price=100.0,
        risk_distance=10.0,
    )

    assert state == "NOISE_PULLBACK"


def test_future_state_class_marks_unrecovered_adverse_move_as_reversal() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 101, 94, 95, 1),
        Candle(start + timedelta(minutes=5), 95, 99, 90, 92, 1),
    )
    times = [candle.timestamp for candle in candles]

    state = _future_state_class(
        candles_5m=candles,
        m5_times=times,
        snapshot_time=start,
        snapshot_price=100.0,
        risk_distance=10.0,
    )

    assert state == "REVERSAL"
