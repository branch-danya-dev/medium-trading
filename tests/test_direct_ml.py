from datetime import UTC, datetime, timedelta

from medium_trading.backtest.model import BacktestTrade
from medium_trading.direct_ml import (
    FEATURE_NAMES,
    OpportunitySample,
    _select_best_model_trades,
    evaluate_direct_ml_final,
    evaluate_direct_ml_opportunities,
    evaluation_payload,
    final_evaluation_payload,
)
from medium_trading.domain import Side


def _trade(
    *,
    side: Side,
    entry_time: datetime,
    positive: bool,
) -> BacktestTrade:
    gross_r = 1.1 if positive else -0.9
    cost_r = 0.1
    return BacktestTrade(
        symbol="EUR/USD",
        side=side,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(hours=1),
        entry=1.0,
        stop=0.99 if side is Side.LONG else 1.01,
        exit=1.01 if positive and side is Side.LONG else 0.99,
        gross_r=gross_r,
        net_r=gross_r - cost_r,
        cost_r=cost_r,
        exit_reason="target" if positive else "stop",
    )


def _sample(*, year: int, index: int, long_is_good: bool) -> OpportunitySample:
    decision_time = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=4 * index)
    feature = 1.0 if long_is_good else -1.0
    features = (feature,) + (0.0,) * (len(FEATURE_NAMES) - 1)
    return OpportunitySample(
        symbol="EUR/USD",
        decision_time=decision_time,
        features=features,
        long_trade=_trade(
            side=Side.LONG,
            entry_time=decision_time,
            positive=long_is_good,
        ),
        short_trade=_trade(
            side=Side.SHORT,
            entry_time=decision_time,
            positive=not long_is_good,
        ),
    )


def test_direct_ml_learns_directional_opportunity_on_unseen_year() -> None:
    samples = []
    for year in range(2020, 2024):
        for index in range(200):
            samples.append(
                _sample(
                    year=year,
                    index=index,
                    long_is_good=index % 2 == 0,
                )
            )

    evaluation = evaluate_direct_ml_opportunities(
        samples,
        first_test_year=2023,
        last_test_year=2023,
    )

    fold = evaluation.folds[0]
    assert fold.train_samples == 600
    assert fold.test_samples == 200
    assert fold.selected_samples > 0
    assert fold.model.net_r > 0
    assert fold.model_2x_costs.net_r > 0
    assert fold.model.net_r > fold.always_long.net_r
    assert fold.model.net_r > fold.always_short.net_r

    payload = evaluation_payload(evaluation)
    assert payload["model"]["prediction_threshold_r"] == 0.0
    assert payload["execution"]["stop_atr_multiple"] == 1.5
    assert payload["execution"]["target_atr_multiple"] == 2.0
    assert payload["execution"]["max_holding_m30_bars"] == 48


def test_direct_ml_training_excludes_unknown_labels_at_fold_boundary() -> None:
    samples = []
    for year in range(2020, 2023):
        for index in range(200):
            samples.append(
                _sample(
                    year=year,
                    index=index,
                    long_is_good=index % 2 == 0,
                )
            )

    crossing = _sample(year=2022, index=2190, long_is_good=True)
    crossing = OpportunitySample(
        symbol=crossing.symbol,
        decision_time=datetime(2022, 12, 31, 20, tzinfo=UTC),
        features=crossing.features,
        long_trade=BacktestTrade(
            symbol="EUR/USD",
            side=Side.LONG,
            entry_time=datetime(2022, 12, 31, 20, tzinfo=UTC),
            exit_time=datetime(2023, 1, 2, tzinfo=UTC),
            entry=1.0,
            stop=0.99,
            exit=1.01,
            gross_r=1.1,
            net_r=1.0,
            cost_r=0.1,
            exit_reason="target",
        ),
        short_trade=BacktestTrade(
            symbol="EUR/USD",
            side=Side.SHORT,
            entry_time=datetime(2022, 12, 31, 20, tzinfo=UTC),
            exit_time=datetime(2023, 1, 2, tzinfo=UTC),
            entry=1.0,
            stop=1.01,
            exit=1.01,
            gross_r=-0.9,
            net_r=-1.0,
            cost_r=0.1,
            exit_reason="stop",
        ),
    )
    samples.append(crossing)

    for index in range(200):
        samples.append(
            _sample(
                year=2023,
                index=index,
                long_is_good=index % 2 == 0,
            )
        )

    evaluation = evaluate_direct_ml_opportunities(
        samples,
        first_test_year=2023,
        last_test_year=2023,
    )

    assert evaluation.folds[0].train_samples == 600


def test_best_selector_takes_one_global_opportunity_and_respects_open_trade() -> None:
    first_time = datetime(2025, 1, 1, tzinfo=UTC)
    first_a = _sample(year=2025, index=0, long_is_good=True)
    first_b = OpportunitySample(
        symbol="GBP/USD",
        decision_time=first_time,
        features=first_a.features,
        long_trade=BacktestTrade(
            symbol="GBP/USD",
            side=Side.LONG,
            entry_time=first_time,
            exit_time=first_time + timedelta(hours=6),
            entry=1.0,
            stop=0.99,
            exit=1.01,
            gross_r=1.1,
            net_r=1.0,
            cost_r=0.1,
            exit_reason="target",
        ),
        short_trade=BacktestTrade(
            symbol="GBP/USD",
            side=Side.SHORT,
            entry_time=first_time,
            exit_time=first_time + timedelta(hours=6),
            entry=1.0,
            stop=1.01,
            exit=1.01,
            gross_r=-0.9,
            net_r=-1.0,
            cost_r=0.1,
            exit_reason="stop",
        ),
    )
    blocked = _sample(year=2025, index=1, long_is_good=True)
    after_exit = _sample(year=2025, index=2, long_is_good=True)

    samples = (first_a, first_b, blocked, after_exit)
    selected = _select_best_model_trades(
        samples,
        long_predictions=(0.2, 0.8, 0.9, 0.7),
        short_predictions=(-0.2, -0.1, -0.2, -0.1),
    )

    assert len(selected) == 2
    assert selected[0].symbol == "GBP/USD"
    assert selected[1].entry_time == after_exit.long_trade.entry_time


def test_final_direct_ml_evaluation_exposes_locked_kill_gate() -> None:
    samples = []
    symbols = ("EUR/USD", "GBP/USD")
    for year in range(2020, 2026):
        for index in range(200):
            base = _sample(
                year=year,
                index=index,
                long_is_good=index % 2 == 0,
            )
            for symbol in symbols:
                samples.append(
                    OpportunitySample(
                        symbol=symbol,
                        decision_time=base.decision_time,
                        features=base.features,
                        long_trade=BacktestTrade(
                            symbol=symbol,
                            side=base.long_trade.side,
                            entry_time=base.long_trade.entry_time,
                            exit_time=base.long_trade.exit_time,
                            entry=base.long_trade.entry,
                            stop=base.long_trade.stop,
                            exit=base.long_trade.exit,
                            gross_r=base.long_trade.gross_r,
                            net_r=base.long_trade.net_r,
                            cost_r=base.long_trade.cost_r,
                            exit_reason=base.long_trade.exit_reason,
                        ),
                        short_trade=BacktestTrade(
                            symbol=symbol,
                            side=base.short_trade.side,
                            entry_time=base.short_trade.entry_time,
                            exit_time=base.short_trade.exit_time,
                            entry=base.short_trade.entry,
                            stop=base.short_trade.stop,
                            exit=base.short_trade.exit,
                            gross_r=base.short_trade.gross_r,
                            net_r=base.short_trade.net_r,
                            cost_r=base.short_trade.cost_r,
                            exit_reason=base.short_trade.exit_reason,
                        ),
                    )
                )

    evaluation = evaluate_direct_ml_final(samples)

    payload = final_evaluation_payload(evaluation)
    assert payload["execution"]["selection"] == "single_best_global_non_overlapping"
    assert payload["kill_gate"]["min_trades"] == 500
    assert payload["kill_gate"]["min_net_profit_factor"] == 1.10
    assert payload["kill_gate"]["min_positive_years"] == 2
