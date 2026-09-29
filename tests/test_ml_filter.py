from datetime import UTC, datetime, timedelta

from medium_trading.ml_filter import (
    FEATURE_NAMES,
    TradeSample,
    evaluate_mean_reversion_ml_filter,
    evaluation_payload,
)


def _sample(
    *,
    year: int,
    index: int,
    positive: bool,
    exit_after_days: int = 0,
) -> TradeSample:
    entry = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    net_r = 1.0 if positive else -1.0
    cost_r = 0.1
    gross_r = net_r + cost_r
    signal_feature = 1.0 if positive else -1.0
    features = (signal_feature,) + (0.0,) * (len(FEATURE_NAMES) - 1)
    return TradeSample(
        symbol="EUR/USD",
        entry_time=entry,
        exit_time=entry + timedelta(days=exit_after_days, hours=1),
        features=features,
        gross_r=gross_r,
        cost_r=cost_r,
        net_r=net_r,
    )


def test_walk_forward_model_can_filter_stable_synthetic_edge() -> None:
    samples = []
    for year in range(2020, 2026):
        for index in range(60):
            samples.append(_sample(year=year, index=index, positive=True))
            samples.append(_sample(year=year, index=index + 100, positive=False))

    evaluation = evaluate_mean_reversion_ml_filter(samples)

    assert [fold.test_year for fold in evaluation.folds] == [2023, 2024, 2025]
    assert evaluation.combined_baseline.net_r == 0.0
    assert evaluation.combined_model.trades > 0
    assert evaluation.combined_model.trades < evaluation.combined_baseline.trades
    assert evaluation.combined_model.net_r > 0
    assert evaluation.combined_model_2x_costs.net_r > 0

    payload = evaluation_payload(evaluation)
    assert payload["model"]["type"] == "HistGradientBoostingRegressor"
    assert payload["model"]["prediction_threshold_r"] == 0.0


def test_training_excludes_labels_not_known_before_fold_start() -> None:
    samples = []
    for year in range(2020, 2023):
        for index in range(50):
            samples.append(_sample(year=year, index=index, positive=True))
            samples.append(_sample(year=year, index=index + 100, positive=False))

    crossing = TradeSample(
        symbol="EUR/USD",
        entry_time=datetime(2022, 12, 31, 20, tzinfo=UTC),
        exit_time=datetime(2023, 1, 2, tzinfo=UTC),
        features=(1.0,) + (0.0,) * (len(FEATURE_NAMES) - 1),
        gross_r=1.1,
        cost_r=0.1,
        net_r=1.0,
    )
    samples.append(crossing)

    for index in range(50):
        samples.append(_sample(year=2023, index=index, positive=True))
        samples.append(_sample(year=2023, index=index + 100, positive=False))

    evaluation = evaluate_mean_reversion_ml_filter(
        samples,
        first_test_year=2023,
        last_test_year=2023,
    )

    assert evaluation.folds[0].train_samples == 300
