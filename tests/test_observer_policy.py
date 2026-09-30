from datetime import UTC, datetime, timedelta

import pytest

from medium_trading import market_observer_v06 as observer_v06
from medium_trading import observer_long_integration as integration
from medium_trading import observer_runtime
from medium_trading.backtest.model import BacktestTrade
from medium_trading.domain import Candle, Side
from medium_trading.market_observer import MarketObserverSample
from medium_trading.market_observer_v07 import FEATURE_NAMES
from medium_trading.observer_long_integration_v2 import _policy_v2_gate
from medium_trading.observer_policy import (
    LongObserverPolicy,
    LongPolicyAction,
    ObserverMarketState,
    observer_snapshot_from_probability,
)
from medium_trading.observer_policy_v2 import LongObserverPolicyV2


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


def test_observer_snapshot_contains_market_state_not_position_state() -> None:
    snapshot = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.70,
        reversal_threshold=0.25,
    )

    assert snapshot.market_state is ObserverMarketState.REAL_REVERSAL_RISK
    assert snapshot.trend_survives_probability == pytest.approx(0.30)
    joined = " ".join(snapshot.__slots__)
    for forbidden in ("position", "entry", "stop", "target", "pnl", "exit"):
        assert forbidden not in joined


def test_long_policy_translates_observer_state_into_trade_action() -> None:
    policy = LongObserverPolicy()
    exit_snapshot = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.40,
        reversal_threshold=0.25,
    )
    hold_snapshot = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, 0, 5, tzinfo=UTC),
        trend_side="BULL",
        event_type="SWEEP_RECLAIM",
        reversal_probability=0.10,
        reversal_threshold=0.25,
    )
    bear_snapshot = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, 0, 10, tzinfo=UTC),
        trend_side="BEAR",
        event_type="BODY_BREAK",
        reversal_probability=0.80,
        reversal_threshold=0.25,
    )

    assert (
        policy.decide_open_long(exit_snapshot).action
        is LongPolicyAction.EXIT_LONG
    )
    assert (
        policy.decide_open_long(hold_snapshot).action
        is LongPolicyAction.HOLD_LONG
    )
    assert (
        policy.decide_open_long(bear_snapshot).action
        is LongPolicyAction.NO_ACTION
    )


def test_observer_exit_executes_at_next_m5_open() -> None:
    candles = (
        _bar(0, open_=100, high=103, low=98, close=101),
        _bar(1, open_=101, high=104, low=99, close=102),
        _bar(2, open_=102, high=103, low=96, close=97),
        _bar(3, open_=95, high=96, low=94, close=95),
        _bar(4, open_=95, high=121, low=95, close=120),
    )
    trade = BacktestTrade(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_time=candles[0].timestamp,
        exit_time=candles[4].timestamp + timedelta(minutes=5),
        entry=100.0,
        stop=90.0,
        exit=120.0,
        gross_r=2.0,
        net_r=1.85,
        cost_r=0.15,
        exit_reason="target",
    )
    snapshot = observer_snapshot_from_probability(
        timestamp=candles[2].timestamp + timedelta(minutes=5),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.50,
        reversal_threshold=0.25,
    )

    managed, decisions, exit_decision = (
        integration._simulate_observer_managed_path(
            candles_5m=candles,
            entry_index=0,
            trade=trade,
            snapshots=(snapshot,),
            snapshot_times=[snapshot.timestamp],
            policy=LongObserverPolicy(),
            fee_bps_per_side=5.5,
            slippage_bps_per_side=2.0,
        )
    )

    assert len(decisions) == 1
    assert exit_decision is not None
    assert managed.exit_reason == "observer_real_reversal"
    assert managed.exit_time == candles[3].timestamp
    assert managed.exit_price == 95.0


def test_stop_gap_has_priority_over_pending_observer_exit() -> None:
    candles = (
        _bar(0, open_=100, high=103, low=98, close=101),
        _bar(1, open_=101, high=102, low=96, close=97),
        _bar(2, open_=89, high=91, low=88, close=90),
    )
    trade = BacktestTrade(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_time=candles[0].timestamp,
        exit_time=candles[2].timestamp + timedelta(minutes=5),
        entry=100.0,
        stop=90.0,
        exit=89.0,
        gross_r=-1.1,
        net_r=-1.25,
        cost_r=0.15,
        exit_reason="stop_gap",
    )
    snapshot = observer_snapshot_from_probability(
        timestamp=candles[1].timestamp + timedelta(minutes=5),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.50,
        reversal_threshold=0.25,
    )

    managed, _, exit_decision = integration._simulate_observer_managed_path(
        candles_5m=candles,
        entry_index=0,
        trade=trade,
        snapshots=(snapshot,),
        snapshot_times=[snapshot.timestamp],
        policy=LongObserverPolicy(),
        fee_bps_per_side=5.5,
        slippage_bps_per_side=2.0,
    )

    assert managed.exit_reason == "stop_gap"
    assert managed.exit_price == 89.0
    assert exit_decision is None


def _features(label: str, index: int) -> tuple[object, ...]:
    center = {
        "NOISE": -1.0,
        "CORRECTION": 0.0,
        "REAL_REVERSAL": 1.0,
        "AMBIGUOUS": 0.2,
    }[label]
    categorical = (
        "BULL" if index % 2 == 0 else "BEAR",
        "BODY_BREAK",
        "ASIA",
        "MON",
    )
    numerics = tuple(
        center + ((index + feature_index) % 5 - 2) * 0.01
        for feature_index in range(len(FEATURE_NAMES) - 4)
    )
    return (*categorical, *numerics)


def _runtime_samples(year: int) -> tuple[MarketObserverSample, ...]:
    result = []
    start = datetime(year, 1, 1, tzinfo=UTC)
    labels = ("NOISE", "CORRECTION", "REAL_REVERSAL")
    for day in range(360):
        for label_index, label in enumerate(labels):
            event_time = start + timedelta(days=day, hours=label_index * 2)
            result.append(
                MarketObserverSample(
                    event_time=event_time,
                    label_end_time=event_time + timedelta(hours=8),
                    trend_side="BULL" if day % 2 == 0 else "BEAR",
                    event_type="BODY_BREAK",
                    features=_features(label, day + label_index),
                    state_class=label,
                    defended_level=100.0,
                    atr5=10.0,
                )
            )

    if year in {2024, 2025}:
        event_time = start + timedelta(days=180, hours=7)
        result.append(
            MarketObserverSample(
                event_time=event_time,
                label_end_time=event_time + timedelta(hours=8),
                trend_side="BULL",
                event_type="SWEEP_RECLAIM",
                features=_features("AMBIGUOUS", 999),
                state_class="AMBIGUOUS",
                defended_level=100.0,
                atr5=10.0,
            )
        )
    return tuple(result)


def test_runtime_emits_snapshots_for_ambiguous_future_labels(
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

    samples = (
        *_runtime_samples(2023),
        *_runtime_samples(2024),
        *_runtime_samples(2025),
    )
    snapshots = observer_runtime.build_v07_walk_forward_snapshots(samples)

    assert len(snapshots) == 2162
    ambiguous_times = {
        sample.event_time
        for sample in samples
        if sample.state_class == "AMBIGUOUS"
    }
    snapshot_times = {snapshot.timestamp for snapshot in snapshots}
    assert ambiguous_times <= snapshot_times
    assert all(snapshot.model_version == "market-observer-v0.7" for snapshot in snapshots)


def test_policy_gate_measures_incremental_value_not_profitability() -> None:
    baseline = {
        "trades": 150,
        "net_r": -20.0,
        "profit_factor": 0.80,
        "max_drawdown": 0.30,
    }
    managed = {
        "trades": 150,
        "net_r": -10.0,
        "profit_factor": 0.90,
        "max_drawdown": 0.25,
    }
    combined = {
        "baseline_m5": baseline,
        "observer_managed_m5": managed,
        "baseline_m5_2x_costs": {"net_r": -40.0},
        "observer_managed_m5_2x_costs": {"net_r": -30.0},
        "delta_net_r": 10.0,
        "policy_diagnostics": {
            "observer_exit_total_delta_net_r": 10.0,
        },
    }
    years = [
        {"year": 2024, "delta_net_r": 4.0},
        {"year": 2025, "delta_net_r": 6.0},
    ]

    gate = integration._policy_value_gate(
        combined=combined,
        years=years,
    )

    assert gate["passes"] is True
    assert "does not declare the underlying strategy profitable" in gate["note"]



def test_v2_first_reversal_arms_warning_and_second_exits() -> None:
    policy = LongObserverPolicyV2()
    first = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.40,
        reversal_threshold=0.25,
    )
    second = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, 0, 20, tzinfo=UTC),
        trend_side="BULL",
        event_type="SWEEP_RECLAIM",
        reversal_probability=0.35,
        reversal_threshold=0.25,
    )

    warning = policy.decide_open_long(first)
    exit_decision = policy.decide_open_long(second)

    assert warning.action is LongPolicyAction.WARNING_LONG
    assert policy.warning is None
    assert exit_decision.action is LongPolicyAction.EXIT_LONG


def test_v2_trend_survives_resets_warning() -> None:
    policy = LongObserverPolicyV2()
    risk = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.40,
        reversal_threshold=0.25,
    )
    survives = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, 0, 20, tzinfo=UTC),
        trend_side="BULL",
        event_type="SWEEP_RECLAIM",
        reversal_probability=0.10,
        reversal_threshold=0.25,
    )
    later_risk = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, 0, 40, tzinfo=UTC),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.50,
        reversal_threshold=0.25,
    )

    assert (
        policy.decide_open_long(risk).action
        is LongPolicyAction.WARNING_LONG
    )
    reset_decision = policy.decide_open_long(survives)
    new_warning = policy.decide_open_long(later_risk)

    assert reset_decision.action is LongPolicyAction.HOLD_LONG
    assert "warning is cleared" in reset_decision.reason
    assert new_warning.action is LongPolicyAction.WARNING_LONG
    assert policy.warning is not None


def test_v2_bear_snapshot_clears_stale_bull_warning() -> None:
    policy = LongObserverPolicyV2()
    risk = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.40,
        reversal_threshold=0.25,
    )
    bear = observer_snapshot_from_probability(
        timestamp=datetime(2025, 1, 1, 0, 20, tzinfo=UTC),
        trend_side="BEAR",
        event_type="BODY_BREAK",
        reversal_probability=0.80,
        reversal_threshold=0.25,
    )

    policy.decide_open_long(risk)
    decision = policy.decide_open_long(bear)

    assert decision.action is LongPolicyAction.NO_ACTION
    assert policy.warning is None
    assert "clears the stale BULL warning" in decision.reason


def test_v2_requires_later_snapshot_for_confirmation() -> None:
    policy = LongObserverPolicyV2()
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    first = observer_snapshot_from_probability(
        timestamp=timestamp,
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.40,
        reversal_threshold=0.25,
    )
    duplicate = observer_snapshot_from_probability(
        timestamp=timestamp,
        trend_side="BULL",
        event_type="SWEEP_RECLAIM",
        reversal_probability=0.45,
        reversal_threshold=0.25,
    )

    policy.decide_open_long(first)
    decision = policy.decide_open_long(duplicate)

    assert decision.action is LongPolicyAction.WARNING_LONG
    assert policy.warning is not None


def test_v2_simulator_does_not_exit_on_first_warning() -> None:
    candles = (
        _bar(0, open_=100, high=103, low=98, close=101),
        _bar(1, open_=101, high=102, low=98, close=100),
        _bar(2, open_=100, high=101, low=97, close=98),
        _bar(3, open_=98, high=99, low=96, close=97),
        _bar(4, open_=97, high=120, low=97, close=119),
    )
    trade = BacktestTrade(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry_time=candles[0].timestamp,
        exit_time=candles[4].timestamp + timedelta(minutes=5),
        entry=100.0,
        stop=90.0,
        exit=120.0,
        gross_r=2.0,
        net_r=1.85,
        cost_r=0.15,
        exit_reason="target",
    )
    warning_snapshot = observer_snapshot_from_probability(
        timestamp=candles[1].timestamp + timedelta(minutes=5),
        trend_side="BULL",
        event_type="BODY_BREAK",
        reversal_probability=0.50,
        reversal_threshold=0.25,
    )

    managed, decisions, exit_decision = (
        integration._simulate_observer_managed_path(
            candles_5m=candles,
            entry_index=0,
            trade=trade,
            snapshots=(warning_snapshot,),
            snapshot_times=[warning_snapshot.timestamp],
            policy=LongObserverPolicyV2(),
            fee_bps_per_side=5.5,
            slippage_bps_per_side=2.0,
        )
    )

    assert decisions[0].action is LongPolicyAction.WARNING_LONG
    assert exit_decision is None
    assert managed.exit_reason == "target"


def test_v2_gate_requires_improvement_over_baseline_and_v1() -> None:
    baseline = {
        "trades": 150,
        "net_r": -20.0,
        "profit_factor": 0.80,
        "max_drawdown": 0.30,
    }
    v1 = {
        "trades": 150,
        "net_r": -15.0,
        "profit_factor": 0.85,
        "max_drawdown": 0.31,
    }
    v2 = {
        "trades": 150,
        "net_r": -10.0,
        "profit_factor": 0.90,
        "max_drawdown": 0.25,
    }
    baseline_2x = {"net_r": -40.0}
    v1_2x = {"net_r": -35.0}
    v2_2x = {"net_r": -30.0}

    v2_combined = {
        "baseline_m5": baseline,
        "observer_managed_m5": v2,
        "baseline_m5_2x_costs": baseline_2x,
        "observer_managed_m5_2x_costs": v2_2x,
        "delta_net_r": 10.0,
        "policy_diagnostics": {
            "observer_exit_total_delta_net_r": 10.0,
            "premature_exits_before_baseline_target": 8,
            "observer_exits": 30,
        },
    }
    v1_combined = {
        "observer_managed_m5": v1,
        "observer_managed_m5_2x_costs": v1_2x,
        "policy_diagnostics": {
            "premature_exits_before_baseline_target": 20,
            "observer_exits": 60,
        },
    }
    v2_years = [
        {"year": 2024, "delta_net_r": 4.0},
        {"year": 2025, "delta_net_r": 6.0},
    ]
    v1_years = [
        {"year": 2024, "delta_net_r": 7.0},
        {"year": 2025, "delta_net_r": -2.0},
    ]

    gate = _policy_v2_gate(
        v2_combined=v2_combined,
        v2_years=v2_years,
        v1_combined=v1_combined,
        v1_years=v1_years,
    )

    assert gate["passes"] is True
    assert all(gate["conditions"].values())
