import json
from datetime import UTC, datetime, timedelta

from medium_trading.backtest.model import BacktestTrade
from medium_trading.crypto_noise_filter import (
    FEATURE_NAMES,
    NoiseSample,
    _move_outcome_class,
    _outcome_class,
    evaluate_btc_long_move_filter_v02,
    evaluate_btc_long_noise_filter_v01,
    evaluation_payload,
    move_evaluation_payload,
)
from medium_trading.domain import Candle, Side


def _trade(entry_time: datetime, *, positive: bool) -> BacktestTrade:
    gross_r = 2.0 if positive else -1.0
    cost_r = 0.1
    return BacktestTrade(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(hours=2),
        entry=100.0,
        stop=90.0,
        exit=120.0 if positive else 90.0,
        gross_r=gross_r,
        net_r=gross_r - cost_r,
        cost_r=cost_r,
        exit_reason="target" if positive else "stop",
        fee_r=0.07,
        slippage_r=0.03,
    )


def _sample(
    *,
    year: int,
    index: int,
    clean: bool,
    economic_pass: bool = True,
    real_move: bool | None = None,
    move_class: str | None = None,
) -> NoiseSample:
    entry_time = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    if real_move is None:
        real_move = clean
    feature_signal = 1.0 if real_move else -1.0
    features = (feature_signal,) + (0.0,) * (len(FEATURE_NAMES) - 1)
    ratio = 10.0 if economic_pass else 4.0
    return NoiseSample(
        symbol="BTCUSDT",
        entry_time=entry_time,
        label_end_time=entry_time + timedelta(hours=24),
        features=features,
        trade=_trade(entry_time, positive=clean),
        outcome_class="CLEAN" if clean else "NOISE",
        is_clean=clean,
        expected_cost_r=2.0 / ratio,
        target_to_cost_ratio=ratio,
        move_class=(
            move_class
            if move_class is not None
            else ("DIRECT_MOVE" if real_move else "NO_MOVE")
        ),
        has_real_move=real_move,
    )


def test_walk_forward_noise_filter_learns_simple_clean_signal() -> None:
    samples = []
    for year in (2023, 2024, 2025):
        for index in range(80):
            samples.append(_sample(year=year, index=index, clean=True))
            samples.append(
                _sample(year=year, index=index + 100, clean=False)
            )

    evaluation = evaluate_btc_long_noise_filter_v01(samples)

    assert [fold.test_year for fold in evaluation.folds] == [2024, 2025]
    assert evaluation.combined_economic_gate.trades == 320
    assert 0 < evaluation.combined_ml_filter.trades < 320
    assert evaluation.combined_classification.precision > 0.90
    assert evaluation.combined_classification.recall > 0.90
    assert (
        evaluation.combined_ml_filter.net_r
        > evaluation.combined_economic_gate.net_r
    )

    payload = evaluation_payload(evaluation)
    assert payload["research_scope"]["2026_used"] is False
    assert payload["model"]["probability_threshold"] == 0.5
    assert payload["economic_gate"]["minimum_target_to_cost"] == 8.0
    json.dumps(payload, sort_keys=True)


def test_training_requires_label_to_be_known_before_fold() -> None:
    samples = []
    for index in range(60):
        samples.append(_sample(year=2023, index=index, clean=True))
        samples.append(
            _sample(year=2023, index=index + 100, clean=False)
        )

    crossing = _sample(year=2023, index=8760 - 5, clean=True)
    samples.append(crossing)

    for index in range(60):
        samples.append(_sample(year=2024, index=index, clean=True))
        samples.append(
            _sample(year=2024, index=index + 100, clean=False)
        )

    evaluation = evaluate_btc_long_noise_filter_v01(
        samples,
        first_test_year=2024,
        last_test_year=2024,
    )

    assert evaluation.folds[0].train_samples == 120


def test_outcome_class_marks_clean_before_stop() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 111, 95, 108, 1),
        Candle(start + timedelta(minutes=30), 108, 112, 106, 111, 1),
    )
    trade = _trade(start, positive=True)

    assert _outcome_class(candles=candles, entry_index=0, trade=trade) == "CLEAN"


def test_outcome_class_marks_stop_then_recovery_as_whipsaw() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 105, 89, 92, 1),
        Candle(start + timedelta(minutes=30), 92, 111, 91, 110, 1),
    )
    trade = _trade(start, positive=False)

    assert _outcome_class(candles=candles, entry_index=0, trade=trade) == "WHIPSAW"


def test_outcome_class_marks_no_followthrough_as_noise() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 105, 95, 102, 1),
        Candle(start + timedelta(minutes=30), 102, 107, 96, 103, 1),
    )
    trade = _trade(start, positive=False)

    assert _outcome_class(candles=candles, entry_index=0, trade=trade) == "NOISE"



def test_walk_forward_move_filter_learns_simple_real_move_signal() -> None:
    samples = []
    for year in (2023, 2024, 2025):
        for index in range(80):
            samples.append(
                _sample(
                    year=year,
                    index=index,
                    clean=False,
                    real_move=True,
                    move_class="POST_STOP_MOVE",
                )
            )
            samples.append(
                _sample(
                    year=year,
                    index=index + 100,
                    clean=False,
                    real_move=False,
                    move_class="NO_MOVE",
                )
            )

    evaluation = evaluate_btc_long_move_filter_v02(samples)

    assert [fold.test_year for fold in evaluation.folds] == [2024, 2025]
    assert evaluation.combined_classification.precision > 0.90
    assert evaluation.combined_classification.recall > 0.90
    assert evaluation.combined_classification.precision_lift > 1.5
    assert evaluation.combined_move_distribution["POST_STOP_MOVE"] == 160
    assert evaluation.combined_move_distribution["NO_MOVE"] == 160

    payload = move_evaluation_payload(evaluation)
    assert payload["research_scope"]["2026_used"] is False
    assert payload["model"]["probability_threshold"] == 0.5
    assert payload["label"]["binary_target"] == "REAL_MOVE vs NO_MOVE"
    json.dumps(payload, sort_keys=True)


def test_move_outcome_class_marks_direct_two_r_move() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 121, 95, 118, 1),
        Candle(start + timedelta(minutes=30), 118, 122, 116, 121, 1),
    )
    trade = _trade(start, positive=True)

    assert (
        _move_outcome_class(candles=candles, entry_index=0, trade=trade)
        == "DIRECT_MOVE"
    )


def test_move_outcome_class_marks_post_stop_two_r_move() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 105, 89, 92, 1),
        Candle(start + timedelta(minutes=30), 92, 121, 91, 120, 1),
    )
    trade = _trade(start, positive=False)

    assert (
        _move_outcome_class(candles=candles, entry_index=0, trade=trade)
        == "POST_STOP_MOVE"
    )


def test_move_outcome_class_marks_missing_two_r_as_no_move() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(start, 100, 115, 95, 110, 1),
        Candle(start + timedelta(minutes=30), 110, 119, 106, 118, 1),
    )
    trade = _trade(start, positive=False)

    assert (
        _move_outcome_class(candles=candles, entry_index=0, trade=trade)
        == "NO_MOVE"
    )
