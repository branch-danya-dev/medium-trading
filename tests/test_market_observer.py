from datetime import UTC, datetime, timedelta

from medium_trading import market_observer as observer
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
