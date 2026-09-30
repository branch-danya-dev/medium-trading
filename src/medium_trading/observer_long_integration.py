from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from statistics import mean

from medium_trading.backtest.model import BacktestTrade
from medium_trading.crypto_long_baseline import (
    BTC_SLIPPAGE_BPS_PER_SIDE,
    BTC_TAKER_FEE_BPS_PER_SIDE,
)
from medium_trading.crypto_noise_filter import extract_btc_long_noise_samples
from medium_trading.domain import Candle
from medium_trading.market_structure_events import (
    TARGET_R,
    PathResult,
    _hard_exit_on_candle,
    _last_m5_index,
    _path_metrics,
    _path_result,
    _require_contiguous_m5,
    _simulate_hard_path,
    _source_trade_metrics,
)
from medium_trading.observer_policy import (
    LongObserverPolicy,
    LongPolicyAction,
    LongPolicyDecision,
    ObserverSnapshot,
)

DEVELOPMENT_YEARS = (2024, 2025)
POLICY_VERSION = "long-observer-policy-v1"

MIN_POLICY_TRADES = 100


@dataclass(frozen=True, slots=True)
class ObserverTradeComparison:
    source_trade: BacktestTrade
    baseline: PathResult
    managed: PathResult
    decisions: tuple[LongPolicyDecision, ...]
    exit_decision: LongPolicyDecision | None

    @property
    def delta_gross_r(self) -> float:
        return self.managed.gross_r - self.baseline.gross_r

    @property
    def delta_net_r(self) -> float:
        return self.managed.net_r - self.baseline.net_r


def evaluate_long_observer_policy_v1(
    *,
    candles_30m: tuple[Candle, ...],
    candles_5m: tuple[Candle, ...],
    snapshots: tuple[ObserverSnapshot, ...],
    symbol: str = "BTCUSDT",
    fee_bps_per_side: float = BTC_TAKER_FEE_BPS_PER_SIDE,
    slippage_bps_per_side: float = BTC_SLIPPAGE_BPS_PER_SIDE,
    starting_equity: float = 1_000.0,
    risk_fraction: float = 0.005,
) -> dict[str, object]:
    """
    Measure whether frozen Market Observer v0.7 adds value to open-LONG management.

    Entry selection is unchanged. The candidate stream is the existing economic-pass
    Trend LONG v1.1 stream. Observer messages are allowed to manage an already-open
    LONG only; early exits do not introduce replacement trades in v1.
    """
    if not candles_30m:
        raise ValueError("observer policy evaluation requires M30 candles")
    if not candles_5m:
        raise ValueError("observer policy evaluation requires M5 candles")
    _require_contiguous_m5(candles_5m)
    if not snapshots:
        raise ValueError("observer policy evaluation requires observer snapshots")

    snapshot_times = [snapshot.timestamp for snapshot in snapshots]
    if snapshot_times != sorted(snapshot_times):
        raise ValueError("observer snapshots must be sorted by timestamp")

    samples = extract_btc_long_noise_samples(
        candles=candles_30m,
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    source_trades = tuple(
        sample.trade
        for sample in samples
        if (
            sample.economic_pass
            and sample.trade.entry_time.year in DEVELOPMENT_YEARS
        )
    )
    if not source_trades:
        raise ValueError(
            "observer policy evaluation found no 2024-2025 economic-pass trades"
        )

    policy = LongObserverPolicy()
    m5_times = [candle.timestamp for candle in candles_5m]
    comparisons = tuple(
        _simulate_trade_pair(
            candles_5m=candles_5m,
            m5_times=m5_times,
            trade=trade,
            snapshots=snapshots,
            snapshot_times=snapshot_times,
            policy=policy,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        for trade in source_trades
    )

    years = [
        {
            "year": year,
            **_comparison_summary(
                tuple(
                    comparison
                    for comparison in comparisons
                    if comparison.source_trade.entry_time.year == year
                ),
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
            ),
        }
        for year in DEVELOPMENT_YEARS
    ]
    combined = _comparison_summary(
        comparisons,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    combined["policy_value_gate"] = _policy_value_gate(
        combined=combined,
        years=years,
    )

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": (
                "first economic integration test of frozen Market Observer v0.7 "
                "as an independent open-LONG risk observer"
            ),
            "development_years": list(DEVELOPMENT_YEARS),
            "2026_trading_outcomes_used": False,
            "observer_model_changed": False,
            "observer_threshold_tuned_on_pnl": False,
            "entry_strategy_changed": False,
            "replacement_trades": False,
            "execution_resolution": "native M5",
            "candidate_stream": (
                "economic-pass Trend LONG v1.1 executed candidates"
            ),
        },
        "communication_contract": {
            "message": "ObserverSnapshot",
            "observer_knows_position_state": False,
            "observer_emits_trading_action": False,
            "policy": POLICY_VERSION,
            "policy_rules": {
                "BULL + REAL_REVERSAL_RISK": (
                    "EXIT_LONG at next M5 open after the observer snapshot"
                ),
                "BULL + TREND_SURVIVES": "HOLD_LONG",
                "BEAR snapshot": (
                    "NO_ACTION in v1; bearish-trend reversal semantics are not "
                    "reinterpreted for an existing LONG"
                ),
            },
            "entry_blocking": False,
        },
        "execution_rules": {
            "hard_stop": "unchanged Trend LONG v1.1 stop",
            "target_r": TARGET_R,
            "max_holding_hours": 24,
            "observer_exit_execution": "next M5 open after snapshot",
            "same_open_priority": (
                "stop gap / target gap are checked before observer exit"
            ),
            "observer_threshold": (
                "fold-specific v0.7 threshold selected from past-only "
                "observer validation; never tuned on trading PnL"
            ),
        },
        "combined": combined,
        "years": years,
        "trades": [_trade_payload(item) for item in comparisons],
    }


def _simulate_trade_pair(
    *,
    candles_5m: tuple[Candle, ...],
    m5_times: list[datetime],
    trade: BacktestTrade,
    snapshots: tuple[ObserverSnapshot, ...],
    snapshot_times: list[datetime],
    policy: LongObserverPolicy,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> ObserverTradeComparison:
    entry_index = bisect_left(m5_times, trade.entry_time)
    if entry_index >= len(candles_5m) or m5_times[entry_index] != trade.entry_time:
        raise ValueError(
            "M5 history does not contain exact trade entry "
            f"{trade.entry_time.isoformat()}"
        )

    baseline = _simulate_hard_path(
        candles_5m=candles_5m,
        entry_index=entry_index,
        trade=trade,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    managed, decisions, exit_decision = _simulate_observer_managed_path(
        candles_5m=candles_5m,
        entry_index=entry_index,
        trade=trade,
        snapshots=snapshots,
        snapshot_times=snapshot_times,
        policy=policy,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    return ObserverTradeComparison(
        source_trade=trade,
        baseline=baseline,
        managed=managed,
        decisions=decisions,
        exit_decision=exit_decision,
    )


def _simulate_observer_managed_path(
    *,
    candles_5m: tuple[Candle, ...],
    entry_index: int,
    trade: BacktestTrade,
    snapshots: tuple[ObserverSnapshot, ...],
    snapshot_times: list[datetime],
    policy: LongObserverPolicy,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> tuple[
    PathResult,
    tuple[LongPolicyDecision, ...],
    LongPolicyDecision | None,
]:
    risk_distance = trade.entry - trade.stop
    if risk_distance <= 0:
        raise ValueError("observer long policy supports LONG trades only")
    target = trade.entry + TARGET_R * risk_distance
    last_index = _last_m5_index(
        candles_5m,
        entry_index,
        trade.entry_time,
    )

    first_snapshot = bisect_right(snapshot_times, trade.entry_time)
    snapshot_cursor = first_snapshot
    decisions: list[LongPolicyDecision] = []
    pending_exit: LongPolicyDecision | None = None

    for index in range(entry_index, last_index + 1):
        candle = candles_5m[index]

        if pending_exit is not None:
            if candle.open <= trade.stop:
                return (
                    _path_result(
                        trade=trade,
                        exit_time=candle.timestamp,
                        exit_index=index,
                        exit_price=candle.open,
                        exit_reason="stop_gap",
                        fee_bps_per_side=fee_bps_per_side,
                        slippage_bps_per_side=slippage_bps_per_side,
                    ),
                    tuple(decisions),
                    None,
                )
            if candle.open >= target:
                return (
                    _path_result(
                        trade=trade,
                        exit_time=candle.timestamp,
                        exit_index=index,
                        exit_price=target,
                        exit_reason="target",
                        fee_bps_per_side=fee_bps_per_side,
                        slippage_bps_per_side=slippage_bps_per_side,
                    ),
                    tuple(decisions),
                    None,
                )
            return (
                _path_result(
                    trade=trade,
                    exit_time=candle.timestamp,
                    exit_index=index,
                    exit_price=candle.open,
                    exit_reason="observer_real_reversal",
                    fee_bps_per_side=fee_bps_per_side,
                    slippage_bps_per_side=slippage_bps_per_side,
                ),
                tuple(decisions),
                pending_exit,
            )

        hard_exit = _hard_exit_on_candle(
            candle=candle,
            index=index,
            trade=trade,
            target=target,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        if hard_exit is not None:
            return hard_exit, tuple(decisions), None

        candle_end = candle.timestamp + timedelta(minutes=5)
        while (
            snapshot_cursor < len(snapshots)
            and snapshots[snapshot_cursor].timestamp <= candle_end
        ):
            snapshot = snapshots[snapshot_cursor]
            snapshot_cursor += 1
            if snapshot.timestamp <= trade.entry_time:
                continue
            decision = policy.decide_open_long(snapshot)
            decisions.append(decision)
            if decision.action is LongPolicyAction.EXIT_LONG:
                pending_exit = decision
                break

        if pending_exit is not None:
            continue

    last = candles_5m[last_index]
    return (
        _path_result(
            trade=trade,
            exit_time=last.timestamp + timedelta(minutes=5),
            exit_index=last_index,
            exit_price=last.close,
            exit_reason="timeout",
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        ),
        tuple(decisions),
        None,
    )


def _comparison_summary(
    comparisons: tuple[ObserverTradeComparison, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
) -> dict[str, object]:
    source = tuple(item.source_trade for item in comparisons)
    baseline = tuple(item.baseline for item in comparisons)
    managed = tuple(item.managed for item in comparisons)

    baseline_metrics = asdict(
        _path_metrics(
            baseline,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
            cost_multiplier=1.0,
        )
    )
    managed_metrics = asdict(
        _path_metrics(
            managed,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
            cost_multiplier=1.0,
        )
    )
    baseline_2x = asdict(
        _path_metrics(
            baseline,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
            cost_multiplier=2.0,
        )
    )
    managed_2x = asdict(
        _path_metrics(
            managed,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
            cost_multiplier=2.0,
        )
    )
    observer_exits = tuple(
        item for item in comparisons if item.exit_decision is not None
    )
    exit_deltas = tuple(item.delta_net_r for item in observer_exits)
    all_decisions = tuple(
        decision for item in comparisons for decision in item.decisions
    )
    baseline_exit_reasons = Counter(
        item.baseline.exit_reason for item in observer_exits
    )

    return {
        "source_m30": _source_trade_metrics(
            source,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
            cost_multiplier=1.0,
        ),
        "baseline_m5": baseline_metrics,
        "observer_managed_m5": managed_metrics,
        "baseline_m5_2x_costs": baseline_2x,
        "observer_managed_m5_2x_costs": managed_2x,
        "delta_gross_r": managed_metrics["gross_r"] - baseline_metrics["gross_r"],
        "delta_net_r": managed_metrics["net_r"] - baseline_metrics["net_r"],
        "delta_profit_factor": (
            managed_metrics["profit_factor"] - baseline_metrics["profit_factor"]
        ),
        "delta_max_drawdown": (
            managed_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
        ),
        "policy_diagnostics": {
            "snapshots_seen_during_trades": len(all_decisions),
            "hold_long_decisions": sum(
                decision.action is LongPolicyAction.HOLD_LONG
                for decision in all_decisions
            ),
            "warning_long_decisions": sum(
                decision.action is LongPolicyAction.WARNING_LONG
                for decision in all_decisions
            ),
            "exit_long_decisions": sum(
                decision.action is LongPolicyAction.EXIT_LONG
                for decision in all_decisions
            ),
            "no_action_decisions": sum(
                decision.action is LongPolicyAction.NO_ACTION
                for decision in all_decisions
            ),
            "trades_with_no_observer_message": sum(
                not item.decisions for item in comparisons
            ),
            "observer_exits": len(observer_exits),
            "observer_exit_improved": sum(
                value > 0 for value in exit_deltas
            ),
            "observer_exit_worsened": sum(
                value < 0 for value in exit_deltas
            ),
            "observer_exit_unchanged": sum(
                value == 0 for value in exit_deltas
            ),
            "observer_exit_total_delta_net_r": sum(exit_deltas),
            "observer_exit_average_delta_net_r": (
                mean(exit_deltas) if exit_deltas else 0.0
            ),
            "baseline_exit_reasons_for_observer_exits": dict(
                baseline_exit_reasons
            ),
            "avoided_baseline_stops": sum(
                item.baseline.exit_reason in {"stop", "stop_gap"}
                and item.delta_net_r > 0
                for item in observer_exits
            ),
            "premature_exits_before_baseline_target": sum(
                item.baseline.exit_reason == "target"
                and item.delta_net_r < 0
                for item in observer_exits
            ),
            "average_exit_reversal_probability": (
                mean(
                    item.exit_decision.reversal_probability
                    for item in observer_exits
                    if item.exit_decision is not None
                )
                if observer_exits
                else 0.0
            ),
        },
    }


def _policy_value_gate(
    *,
    combined: dict[str, object],
    years: list[dict[str, object]],
) -> dict[str, object]:
    baseline = combined["baseline_m5"]
    managed = combined["observer_managed_m5"]
    baseline_2x = combined["baseline_m5_2x_costs"]
    managed_2x = combined["observer_managed_m5_2x_costs"]
    diagnostics = combined["policy_diagnostics"]

    year_deltas = {
        str(year["year"]): float(year["delta_net_r"])
        for year in years
    }
    conditions = {
        "minimum_policy_trades": int(managed["trades"]) >= MIN_POLICY_TRADES,
        "positive_combined_delta_net_r": float(combined["delta_net_r"]) > 0,
        "managed_profit_factor_exceeds_baseline": (
            float(managed["profit_factor"]) > float(baseline["profit_factor"])
        ),
        "managed_drawdown_not_worse": (
            float(managed["max_drawdown"]) <= float(baseline["max_drawdown"])
        ),
        "positive_delta_in_2024": year_deltas.get("2024", 0.0) > 0,
        "positive_delta_in_2025": year_deltas.get("2025", 0.0) > 0,
        "observer_exits_add_net_value": (
            float(diagnostics["observer_exit_total_delta_net_r"]) > 0
        ),
        "2x_cost_net_r_exceeds_baseline": (
            float(managed_2x["net_r"]) > float(baseline_2x["net_r"])
        ),
    }
    return {
        "passes": all(conditions.values()),
        "conditions": conditions,
        "note": (
            "This gate measures incremental trade-management value only. "
            "It does not declare the underlying strategy profitable."
        ),
        "observed": {
            "trades": managed["trades"],
            "combined_delta_net_r": combined["delta_net_r"],
            "baseline_profit_factor": baseline["profit_factor"],
            "managed_profit_factor": managed["profit_factor"],
            "baseline_max_drawdown": baseline["max_drawdown"],
            "managed_max_drawdown": managed["max_drawdown"],
            "year_delta_net_r": year_deltas,
            "observer_exit_total_delta_net_r": diagnostics[
                "observer_exit_total_delta_net_r"
            ],
            "baseline_2x_net_r": baseline_2x["net_r"],
            "managed_2x_net_r": managed_2x["net_r"],
        },
    }


def _trade_payload(item: ObserverTradeComparison) -> dict[str, object]:
    return {
        "entry_time": item.source_trade.entry_time.isoformat(),
        "entry": item.source_trade.entry,
        "stop": item.source_trade.stop,
        "baseline_m5": _path_payload(item.baseline),
        "observer_managed_m5": _path_payload(item.managed),
        "delta_gross_r": item.delta_gross_r,
        "delta_net_r": item.delta_net_r,
        "exit_decision": (
            _decision_payload(item.exit_decision)
            if item.exit_decision is not None
            else None
        ),
        "observer_decisions": [
            _decision_payload(decision)
            for decision in item.decisions
        ],
    }


def _decision_payload(
    decision: LongPolicyDecision,
) -> dict[str, object]:
    return {
        "timestamp": decision.timestamp.isoformat(),
        "action": decision.action.value,
        "reason": decision.reason,
        "reversal_probability": decision.reversal_probability,
        "reversal_threshold": decision.reversal_threshold,
        "observer_trend_side": decision.observer_trend_side,
        "observer_event_type": decision.observer_event_type,
    }


def _path_payload(path: PathResult) -> dict[str, object]:
    return {
        "exit_time": path.exit_time.isoformat(),
        "exit_price": path.exit_price,
        "exit_reason": path.exit_reason,
        "gross_r": path.gross_r,
        "cost_r": path.cost_r,
        "net_r": path.net_r,
    }
