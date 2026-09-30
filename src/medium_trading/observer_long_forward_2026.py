from collections import defaultdict
from datetime import UTC, datetime

from medium_trading.crypto_long_baseline import (
    BTC_SLIPPAGE_BPS_PER_SIDE,
    BTC_TAKER_FEE_BPS_PER_SIDE,
)
from medium_trading.crypto_noise_filter import extract_btc_long_noise_samples
from medium_trading.domain import Candle
from medium_trading.observer_long_integration import (
    ObserverTradeComparison,
    _comparison_summary,
    _simulate_trade_pair,
)
from medium_trading.observer_policy import ObserverSnapshot
from medium_trading.observer_policy_v2 import LongObserverPolicyV2

TRADING_FORWARD_START = datetime(2026, 1, 1, tzinfo=UTC)
TRADING_FORWARD_END = datetime(2026, 9, 28, tzinfo=UTC)
MIN_FORWARD_TRADES = 30


def evaluate_long_observer_policy_v2_forward_2026(
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
    One-shot 2026 trading validation of frozen Observer -> LONG Policy v2.

    Entry strategy, observer, thresholds and v2 state machine are frozen.
    Entries are allowed from 2026-01-01 through 2026-09-28 exclusive. The
    remaining forward data is reserved for complete 24-hour trade/observer exit
    horizons.
    """
    if not candles_30m:
        raise ValueError("2026 trading forward requires M30 candles")
    if not candles_5m:
        raise ValueError("2026 trading forward requires M5 candles")
    if not snapshots:
        raise ValueError("2026 trading forward requires observer snapshots")
    if any(snapshot.timestamp.year != 2026 for snapshot in snapshots):
        raise ValueError("2026 trading forward received non-2026 snapshots")

    samples = extract_btc_long_noise_samples(
        candles=candles_30m,
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
        trade_start=TRADING_FORWARD_START,
        trade_end=TRADING_FORWARD_END,
    )
    source_trades = tuple(
        sample.trade
        for sample in samples
        if (
            sample.economic_pass
            and TRADING_FORWARD_START <= sample.trade.entry_time
            < TRADING_FORWARD_END
        )
    )
    if not source_trades:
        raise ValueError(
            "2026 trading forward found no economic-pass Trend LONG trades"
        )
    if any(trade.entry_time.year != 2026 for trade in source_trades):
        raise ValueError("2026 trading forward candidate stream leaked other years")

    snapshot_times = [snapshot.timestamp for snapshot in snapshots]
    if snapshot_times != sorted(snapshot_times):
        raise ValueError("2026 observer snapshots must be chronological")
    m5_times = [candle.timestamp for candle in candles_5m]

    comparisons = tuple(
        _simulate_trade_pair(
            candles_5m=candles_5m,
            m5_times=m5_times,
            trade=trade,
            snapshots=snapshots,
            snapshot_times=snapshot_times,
            policy=LongObserverPolicyV2(),
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        for trade in source_trades
    )

    combined = _comparison_summary(
        comparisons,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    equity = _equity_summary(
        comparisons,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    combined["equity"] = equity
    combined["forward_trading_gate"] = _forward_trading_gate(
        combined=combined,
    )

    months: dict[str, list[ObserverTradeComparison]] = defaultdict(list)
    for item in comparisons:
        months[item.source_trade.entry_time.strftime("%Y-%m")].append(item)

    monthly = {
        month: {
            **_comparison_summary(
                tuple(items),
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
            ),
            "equity": _equity_summary(
                tuple(items),
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
            ),
        }
        for month, items in sorted(months.items())
    }

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": (
                "one-shot external trading validation of frozen Trend LONG "
                "bot + Market Observer v0.7 + LongObserverPolicy v2"
            ),
            "entry_start": TRADING_FORWARD_START.isoformat(),
            "entry_end_exclusive": TRADING_FORWARD_END.isoformat(),
            "exit_and_observer_horizon_data_after_entry_end": True,
            "observer_model_changed": False,
            "observer_threshold_tuned_on_2026_pnl": False,
            "policy_changed_from_frozen_v2": False,
            "entry_strategy_changed": False,
            "replacement_trades": False,
            "entry_blocking": False,
            "execution_resolution": "native M5",
            "candidate_stream": (
                "economic-pass Trend LONG v1.1 executed candidates"
            ),
        },
        "assumptions": {
            "starting_equity_usd": starting_equity,
            "risk_fraction": risk_fraction,
            "fee_bps_per_side": fee_bps_per_side,
            "slippage_bps_per_side": slippage_bps_per_side,
            "target_r": 2.0,
            "max_holding_hours": 24,
            "observer_policy": "long-observer-policy-v2",
            "observer_exit_execution": "next M5 open after confirmed exit",
            "same_open_priority": (
                "stop gap / target gap before observer exit"
            ),
            "2x_cost_stress": True,
        },
        "combined": combined,
        "months": monthly,
        "trades": [_trade_payload(item) for item in comparisons],
    }


def _equity_summary(
    comparisons: tuple[ObserverTradeComparison, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
) -> dict[str, float]:
    baseline_net = tuple(item.baseline.net_r for item in comparisons)
    managed_net = tuple(item.managed.net_r for item in comparisons)
    baseline_2x = tuple(
        item.baseline.gross_r - 2.0 * item.baseline.cost_r
        for item in comparisons
    )
    managed_2x = tuple(
        item.managed.gross_r - 2.0 * item.managed.cost_r
        for item in comparisons
    )

    baseline_final = _compound_equity(
        baseline_net,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    managed_final = _compound_equity(
        managed_net,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    baseline_2x_final = _compound_equity(
        baseline_2x,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    managed_2x_final = _compound_equity(
        managed_2x,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )

    return {
        "baseline_final_equity_usd": baseline_final,
        "managed_final_equity_usd": managed_final,
        "baseline_return_fraction": (
            baseline_final / starting_equity - 1.0
        ),
        "managed_return_fraction": (
            managed_final / starting_equity - 1.0
        ),
        "incremental_equity_usd": managed_final - baseline_final,
        "baseline_2x_cost_final_equity_usd": baseline_2x_final,
        "managed_2x_cost_final_equity_usd": managed_2x_final,
        "managed_2x_cost_return_fraction": (
            managed_2x_final / starting_equity - 1.0
        ),
    }


def _compound_equity(
    net_values: tuple[float, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
) -> float:
    equity = starting_equity
    for net_r in net_values:
        equity += equity * risk_fraction * net_r
    return equity


def _forward_trading_gate(
    *,
    combined: dict[str, object],
) -> dict[str, object]:
    baseline = combined["baseline_m5"]
    managed = combined["observer_managed_m5"]
    managed_2x = combined["observer_managed_m5_2x_costs"]
    diagnostics = combined["policy_diagnostics"]

    conditions = {
        "minimum_forward_trades": int(managed["trades"]) >= MIN_FORWARD_TRADES,
        "managed_net_r_positive": float(managed["net_r"]) > 0,
        "managed_profit_factor_above_one": (
            float(managed["profit_factor"]) > 1.0
        ),
        "managed_net_r_exceeds_baseline": (
            float(managed["net_r"]) > float(baseline["net_r"])
        ),
        "managed_profit_factor_exceeds_baseline": (
            float(managed["profit_factor"]) > float(baseline["profit_factor"])
        ),
        "managed_drawdown_not_worse_than_baseline": (
            float(managed["max_drawdown"]) <= float(baseline["max_drawdown"])
        ),
        "confirmed_exits_add_net_value": (
            float(diagnostics["observer_exit_total_delta_net_r"]) > 0
        ),
        "2x_cost_net_r_positive": float(managed_2x["net_r"]) > 0,
        "2x_cost_profit_factor_above_one": (
            float(managed_2x["profit_factor"]) > 1.0
        ),
    }
    return {
        "passes": all(conditions.values()),
        "conditions": conditions,
        "note": (
            "PASS means the frozen bot+observer stack is profitable after the "
            "modeled costs on this external 2026 trading window, improves the "
            "same entry bot, and remains profitable at 2x modeled costs."
        ),
        "observed": {
            "trades": managed["trades"],
            "baseline_net_r": baseline["net_r"],
            "managed_net_r": managed["net_r"],
            "baseline_profit_factor": baseline["profit_factor"],
            "managed_profit_factor": managed["profit_factor"],
            "baseline_max_drawdown": baseline["max_drawdown"],
            "managed_max_drawdown": managed["max_drawdown"],
            "confirmed_exit_total_delta_net_r": diagnostics[
                "observer_exit_total_delta_net_r"
            ],
            "managed_2x_cost_net_r": managed_2x["net_r"],
            "managed_2x_cost_profit_factor": managed_2x["profit_factor"],
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


def _decision_payload(decision) -> dict[str, object]:
    return {
        "timestamp": decision.timestamp.isoformat(),
        "action": decision.action.value,
        "reason": decision.reason,
        "reversal_probability": decision.reversal_probability,
        "reversal_threshold": decision.reversal_threshold,
        "observer_trend_side": decision.observer_trend_side,
        "observer_event_type": decision.observer_event_type,
    }


def _path_payload(path) -> dict[str, object]:
    return {
        "exit_time": path.exit_time.isoformat(),
        "exit_price": path.exit_price,
        "exit_reason": path.exit_reason,
        "gross_r": path.gross_r,
        "cost_r": path.cost_r,
        "net_r": path.net_r,
    }
