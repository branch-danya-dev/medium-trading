from datetime import UTC, datetime, timedelta

import pytest

from medium_trading import market_observer as observer
from medium_trading import market_observer_v05 as observer_v05
from medium_trading import market_observer_v06 as observer_v06
from medium_trading.data.bybit_trade_flow import TradeFlowPoint
from medium_trading.domain import Candle


def _bar(
    index: int,
    *,
    open_: float = 100.0,
    high: float = 102.0,
    low: float = 98.0,
    close: float = 100.0,
) -> Candle:
    return Candle(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC)
        + timedelta(minutes=5 * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def test_observer_feature_schema_has_no_trade_state() -> None:
    joined = " ".join(observer.FEATURE_NAMES)

    for forbidden in (
        "entry",
        "stop",
        "target",
        "current_r",
        "mfe",
        "mae",
        "pnl",
        "risk_distance",
    ):
        assert forbidden not in joined


def test_event_type_is_symmetric_for_bull_and_bear_trends() -> None:
    bullish_sweep = _bar(0, open_=101, high=103, low=99, close=101)
    bullish_break = _bar(1, open_=101, high=102, low=97, close=98)
    bearish_sweep = _bar(2, open_=99, high=101, low=97, close=99)
    bearish_break = _bar(3, open_=99, high=103, low=98, close=102)

    assert observer._event_type(
        candle=bullish_sweep,
        trend_side="BULL",
        level=100,
        atr5=10,
    ) == "SWEEP_RECLAIM"
    assert observer._event_type(
        candle=bullish_break,
        trend_side="BULL",
        level=100,
        atr5=10,
    ) == "BODY_BREAK"
    assert observer._event_type(
        candle=bearish_sweep,
        trend_side="BEAR",
        level=100,
        atr5=10,
    ) == "SWEEP_RECLAIM"
    assert observer._event_type(
        candle=bearish_break,
        trend_side="BEAR",
        level=100,
        atr5=10,
    ) == "BODY_BREAK"


def test_market_label_noise_when_trend_recovers_before_adverse_barrier() -> None:
    candles = (
        _bar(0, open_=100, high=101, low=99, close=100),
        _bar(1, open_=100, high=108, low=99, close=107),
        _bar(2, open_=107, high=109, low=106, close=108),
    )

    label = observer._market_label(
        candles_5m=candles,
        event_index=0,
        event_time=candles[0].timestamp + timedelta(minutes=5),
        event_price=100,
        trend_side="BULL",
        event_type="SWEEP_RECLAIM",
        defended_level=99,
        atr5=10,
    )

    assert label == "NOISE"


def test_market_label_correction_when_adverse_move_recovers_inside_horizon() -> None:
    candles = (
        _bar(0, open_=100, high=101, low=99, close=100),
        _bar(1, open_=100, high=101, low=92, close=93),
        _bar(2, open_=93, high=102, low=93, close=100),
        _bar(3, open_=100, high=108, low=99, close=107),
    )

    label = observer._market_label(
        candles_5m=candles,
        event_index=0,
        event_time=candles[0].timestamp + timedelta(minutes=5),
        event_price=100,
        trend_side="BULL",
        event_type="BODY_BREAK",
        defended_level=101,
        atr5=10,
    )

    assert label == "CORRECTION"


def test_market_label_real_reversal_requires_acceptance_and_continuation() -> None:
    candles = (
        _bar(0, open_=102, high=102, low=99, close=100),
        _bar(1, open_=100, high=100, low=96, close=98),
        _bar(2, open_=98, high=99, low=94, close=96),
        _bar(3, open_=96, high=97, low=84, close=86),
        _bar(4, open_=86, high=90, low=84, close=88),
    )

    label = observer._market_label(
        candles_5m=candles,
        event_index=0,
        event_time=candles[0].timestamp + timedelta(minutes=5),
        event_price=100,
        trend_side="BULL",
        event_type="BODY_BREAK",
        defended_level=101,
        atr5=10,
    )

    assert label == "REAL_REVERSAL"


def test_confirmed_swing_is_not_usable_until_right_bars_close() -> None:
    candles = (
        _bar(0, low=100),
        _bar(1, low=99),
        _bar(2, low=95),
        _bar(3, low=98),
        _bar(4, low=99),
    )

    lows, _ = observer._precompute_swings(candles)

    assert lows == [(4, 2, 95)]


def _synthetic_features(
    label: str,
    index: int,
) -> tuple[object, ...]:
    category_cycle = ("ASIA", "EUROPE", "US")
    weekday_cycle = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
    center = {
        "NOISE": 1.5,
        "CORRECTION": 0.0,
        "REAL_REVERSAL": -1.5,
    }[label]
    numeric = []
    for feature_index in range(len(observer.NUMERIC_FEATURES)):
        drift = ((index + feature_index) % 7 - 3) * 0.01
        numeric.append(center + drift)
    return (
        "BULL" if index % 2 == 0 else "BEAR",
        "SWEEP_RECLAIM" if label == "NOISE" else "BODY_BREAK",
        category_cycle[index % len(category_cycle)],
        weekday_cycle[index % len(weekday_cycle)],
        *numeric,
    )


def _synthetic_samples(
    year: int,
    per_class: int,
) -> tuple[observer.MarketObserverSample, ...]:
    result = []
    offset = 0
    for label in observer.CLEAR_LABELS:
        for index in range(per_class):
            event_time = datetime(year, 1, 1, tzinfo=UTC) + timedelta(
                hours=offset
            )
            result.append(
                observer.MarketObserverSample(
                    event_time=event_time,
                    label_end_time=event_time
                    + timedelta(hours=observer.LABEL_HORIZON_HOURS),
                    trend_side="BULL" if index % 2 == 0 else "BEAR",
                    event_type=(
                        "SWEEP_RECLAIM"
                        if label == "NOISE"
                        else "BODY_BREAK"
                    ),
                    features=_synthetic_features(label, index),
                    state_class=label,
                    defended_level=100.0,
                    atr5=10.0,
                )
            )
            offset += 1
    return tuple(result)


def test_walk_forward_observer_trains_catboost_and_random_forest(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        observer,
        "CATBOOST_PARAMS",
        {
            **observer.CATBOOST_PARAMS,
            "iterations": 25,
            "depth": 4,
            "thread_count": 1,
        },
    )
    monkeypatch.setattr(
        observer,
        "RANDOM_FOREST_PARAMS",
        {
            **observer.RANDOM_FOREST_PARAMS,
            "n_estimators": 40,
            "n_jobs": 1,
        },
    )

    samples = (
        *_synthetic_samples(2023, 180),
        *_synthetic_samples(2024, 60),
        *_synthetic_samples(2025, 60),
    )

    payload = observer.evaluate_market_observer_v04(samples)

    assert [fold["test_year"] for fold in payload["folds"]] == [2024, 2025]
    assert payload["research_scope"]["trade_state_used"] is False
    assert payload["combined"]["clear_samples"] == 360

    for model_name in ("catboost", "random_forest"):
        metrics = payload["combined"][model_name]["metrics"]
        assert metrics["macro_f1"] > 0.70
        assert metrics["per_class"]["REAL_REVERSAL"]["precision"] > 0.70
        assert metrics["real_reversal"]["roc_auc"] > 0.80
        assert metrics["real_reversal"]["probability_bands"]



def _v05_samples(year: int) -> tuple[observer.MarketObserverSample, ...]:
    labels = observer.CLEAR_LABELS
    result = []
    start = datetime(year, 1, 1, tzinfo=UTC)
    for day in range(360):
        for label_index, label in enumerate(labels):
            event_time = start + timedelta(days=day, hours=label_index * 2)
            result.append(
                observer.MarketObserverSample(
                    event_time=event_time,
                    label_end_time=event_time
                    + timedelta(hours=observer.LABEL_HORIZON_HOURS),
                    trend_side="BULL" if day % 2 == 0 else "BEAR",
                    event_type=(
                        "SWEEP_RECLAIM"
                        if label == "NOISE"
                        else "BODY_BREAK"
                    ),
                    features=_synthetic_features(label, day + label_index),
                    state_class=label,
                    defended_level=100.0,
                    atr5=10.0,
                )
            )
    return tuple(result)


def test_v05_hierarchical_observer_uses_past_only_calibration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        observer_v05,
        "REVERSAL_PARAMS",
        {
            **observer_v05.REVERSAL_PARAMS,
            "iterations": 20,
            "depth": 4,
        },
    )
    monkeypatch.setattr(
        observer_v05,
        "UNWEIGHTED_REVERSAL_PARAMS",
        {
            **observer_v05.UNWEIGHTED_REVERSAL_PARAMS,
            "iterations": 20,
            "depth": 4,
        },
    )
    monkeypatch.setattr(
        observer_v05,
        "STATE_PARAMS",
        {
            **observer_v05.STATE_PARAMS,
            "iterations": 20,
            "depth": 4,
        },
    )

    samples = (
        *_v05_samples(2023),
        *_v05_samples(2024),
        *_v05_samples(2025),
    )

    payload = observer_v05.evaluate_market_observer_v05(samples)

    assert payload["research_scope"]["features_changed"] is False
    assert payload["research_scope"]["labels_changed"] is False
    assert payload["architecture"]["stage_1_weighting"] == (
        "CatBoost auto_class_weights=Balanced"
    )

    assert [fold["test_year"] for fold in payload["folds"]] == [2024, 2025]
    for fold in payload["folds"]:
        assert fold["fit_samples"] > 300
        assert fold["calibration_samples"] >= 60
        assert fold["threshold_validation_samples"] >= 60
        assert fold["weighted_reversal"]["selected_threshold"] > 0
        assert (
            fold["weighted_reversal"]["test"]["roc_auc"]
            > 0.80
        )
        assert (
            fold["noise_correction"]["test"]["roc_auc"]
            > 0.80
        )

    combined = payload["combined"]
    assert combined["clear_samples"] == 2160
    assert (
        combined["weighted_reversal"]["ranking_and_calibration"]["roc_auc"]
        > 0.80
    )
    assert combined["hierarchical"]["macro_f1"] > 0.70
    assert (
        combined["hierarchical"]["per_class"]["REAL_REVERSAL"][
            "roc_auc_ovr"
        ]
        > 0.80
    )


def test_v05_reversal_model_is_balanced_but_control_is_not() -> None:
    assert observer_v05.REVERSAL_PARAMS["auto_class_weights"] == "Balanced"
    assert "auto_class_weights" not in observer_v05.UNWEIGHTED_REVERSAL_PARAMS
    assert observer_v05.STATE_PARAMS["loss_function"] == "Logloss"



def test_v06_trade_flow_uses_only_completed_buckets() -> None:
    event_time = datetime(2023, 1, 1, 4, 0, tzinfo=UTC)
    base_sample = observer.MarketObserverSample(
        event_time=event_time,
        label_end_time=event_time + timedelta(hours=8),
        trend_side="BULL",
        event_type="BODY_BREAK",
        features=tuple(0.0 for _ in observer.FEATURE_NAMES),
        state_class="REAL_REVERSAL",
        defended_level=100.0,
        atr5=10.0,
    )
    points = []
    for index in range(49):
        buy = 1.0
        sell = 1.0
        if index == 47:
            buy = 1.0
            sell = 9.0
        if index == 48:
            buy = 100.0
            sell = 0.0
        points.append(
            TradeFlowPoint(
                timestamp=datetime(2023, 1, 1, tzinfo=UTC)
                + timedelta(minutes=5 * index),
                buy_qty=buy,
                sell_qty=sell,
                buy_notional=buy * 100.0,
                sell_notional=sell * 100.0,
                buy_count=int(buy),
                sell_count=int(sell),
            )
        )

    augmented = observer_v06.augment_market_observer_samples_with_trade_flow(
        (base_sample,),
        tuple(points),
    )

    flow_features = augmented[0].features[len(observer.FEATURE_NAMES) :]
    assert flow_features[0] == pytest.approx(-0.8)
    assert flow_features[8] == pytest.approx(-0.8)


def _v06_samples() -> tuple[observer.MarketObserverSample, ...]:
    augmented = []
    for sample in (
        *_v05_samples(2023),
        *_v05_samples(2024),
        *_v05_samples(2025),
    ):
        flow_center = {
            "NOISE": 0.8,
            "CORRECTION": 0.0,
            "REAL_REVERSAL": -0.8,
        }[sample.state_class]
        flow_features = tuple(
            flow_center + (index % 3 - 1) * 0.01
            for index in range(len(observer_v06.FLOW_FEATURE_NAMES))
        )
        augmented.append(
            observer.MarketObserverSample(
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


def test_v06_compares_flow_features_with_frozen_v05_baseline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        observer_v06,
        "REVERSAL_PARAMS",
        {
            **observer_v06.REVERSAL_PARAMS,
            "iterations": 20,
            "depth": 4,
        },
    )

    payload = observer_v06.evaluate_market_observer_v06(_v06_samples())

    assert payload["research_scope"]["labels_changed"] is False
    assert payload["research_scope"]["base_features_changed"] is False
    assert payload["research_scope"]["model_architecture_changed"] is False
    assert [fold["test_year"] for fold in payload["folds"]] == [2024, 2025]

    combined = payload["combined"]
    assert combined["clear_samples"] == 2160
    assert (
        combined["flow_enhanced_reversal"]["ranking_and_calibration"][
            "roc_auc"
        ]
        > 0.80
    )
    assert len(
        combined["flow_enhanced_reversal"]["flow_feature_importance"]
    ) == len(observer_v06.FLOW_FEATURE_NAMES)
