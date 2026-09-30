from medium_trading.crypto_long_baseline import (
    BTC_SLIPPAGE_BPS_PER_SIDE,
    BTC_TAKER_FEE_BPS_PER_SIDE,
)
from medium_trading.crypto_noise_filter import extract_btc_long_noise_samples
from medium_trading.domain import Candle
from medium_trading.observer_long_integration import (
    DEVELOPMENT_YEARS,
    MIN_POLICY_TRADES,
    ObserverTradeComparison,
    evaluate_long_observer_policy_v1,
    _comparison_summary,
    _simulate_trade_pair,
)
from medium_trading.observer_policy import ObserverSnapshot
from medium_trading.observer_policy_v2 import LongObserverPolicyV2


POLICY_VERSION = "long-observer-policy-v2"


def evaluate_long_observer_policy_v2(
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
    Compare stateful warning-confirmation policy v2 with baseline and frozen v1.

    Market Observer v0.7, its fold thresholds, entry selection, hard stop, target,
    holding horizon and transaction-cost assumptions are unchanged.
    """
    v1 = evaluate_long_observer_policy_v1(
        candles_30m=candles_30m,
        candles_5m=candles_5m,
        snapshots=snapshots,
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )

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
            "observer policy v2 evaluation found no 2024-2025 "
            "economic-pass trades"
        )

    snapshot_times = [snapshot.timestamp for snapshot in snapshots]
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
    combined["policy_value_gate"] = _policy_v2_gate(
        v2_combined=combined,
        v2_years=years,
        v1_combined=v1["combined"],
        v1_years=v1["years"],
    )

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": (
                "test whether stateful warning confirmation reduces costly "
                "false exits from frozen LONG policy v1"
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
            "state_machine": {
                "CLEAR + BULL REAL_REVERSAL_RISK": "WARNING_LONG",
                "WARNING + later BULL REAL_REVERSAL_RISK": "EXIT_LONG",
                "WARNING + BULL TREND_SURVIVES": (
                    "HOLD_LONG and clear WARNING"
                ),
                "BEAR snapshot": "NO_ACTION and clear stale BULL WARNING",
            },
            "independent_confirmation": (
                "second reversal-risk snapshot must have a later timestamp "
                "than the snapshot that armed WARNING"
            ),
            "entry_blocking": False,
        },
        "execution_rules": v1["execution_rules"],
        "baseline_and_v1_reference": {
            "combined": v1["combined"],
            "years": v1["years"],
        },
        "combined": combined,
        "years": years,
        "trades": [_trade_payload(item) for item in comparisons],
    }


def _policy_v2_gate(
    *,
    v2_combined: dict[str, object],
    v2_years: list[dict[str, object]],
    v1_combined: dict[str, object],
    v1_years: list[dict[str, object]],
) -> dict[str, object]:
    baseline = v2_combined["baseline_m5"]
    v2 = v2_combined["observer_managed_m5"]
    v1 = v1_combined["observer_managed_m5"]
    baseline_2x = v2_combined["baseline_m5_2x_costs"]
    v2_2x = v2_combined["observer_managed_m5_2x_costs"]
    v1_2x = v1_combined["observer_managed_m5_2x_costs"]
    v2_diag = v2_combined["policy_diagnostics"]
    v1_diag = v1_combined["policy_diagnostics"]

    v2_year_delta = {
        str(item["year"]): float(item["delta_net_r"])
        for item in v2_years
    }
    v1_year_delta = {
        str(item["year"]): float(item["delta_net_r"])
        for item in v1_years
    }

    conditions = {
        "minimum_policy_trades": int(v2["trades"]) >= MIN_POLICY_TRADES,
        "positive_combined_delta_net_r": (
            float(v2_combined["delta_net_r"]) > 0
        ),
        "v2_net_r_exceeds_v1": float(v2["net_r"]) > float(v1["net_r"]),
        "v2_profit_factor_exceeds_baseline": (
            float(v2["profit_factor"]) > float(baseline["profit_factor"])
        ),
        "v2_profit_factor_exceeds_v1": (
            float(v2["profit_factor"]) > float(v1["profit_factor"])
        ),
        "v2_drawdown_not_worse_than_baseline": (
            float(v2["max_drawdown"]) <= float(baseline["max_drawdown"])
        ),
        "positive_delta_in_2024": v2_year_delta.get("2024", 0.0) > 0,
        "positive_delta_in_2025": v2_year_delta.get("2025", 0.0) > 0,
        "confirmed_exits_add_net_value": (
            float(v2_diag["observer_exit_total_delta_net_r"]) > 0
        ),
        "v2_2x_cost_net_r_exceeds_baseline": (
            float(v2_2x["net_r"]) > float(baseline_2x["net_r"])
        ),
        "v2_2x_cost_net_r_exceeds_v1": (
            float(v2_2x["net_r"]) > float(v1_2x["net_r"])
        ),
        "fewer_premature_target_exits_than_v1": (
            int(v2_diag["premature_exits_before_baseline_target"])
            < int(v1_diag["premature_exits_before_baseline_target"])
        ),
    }

    return {
        "passes": all(conditions.values()),
        "conditions": conditions,
        "note": (
            "The v2 gate measures whether warning confirmation improves "
            "incremental LONG management versus both baseline and frozen v1. "
            "It does not prove the underlying entry strategy profitable."
        ),
        "observed": {
            "trades": v2["trades"],
            "baseline_net_r": baseline["net_r"],
            "v1_net_r": v1["net_r"],
            "v2_net_r": v2["net_r"],
            "v2_delta_net_r": v2_combined["delta_net_r"],
            "baseline_profit_factor": baseline["profit_factor"],
            "v1_profit_factor": v1["profit_factor"],
            "v2_profit_factor": v2["profit_factor"],
            "baseline_max_drawdown": baseline["max_drawdown"],
            "v1_max_drawdown": v1["max_drawdown"],
            "v2_max_drawdown": v2["max_drawdown"],
            "v1_year_delta_net_r": v1_year_delta,
            "v2_year_delta_net_r": v2_year_delta,
            "v1_observer_exits": v1_diag["observer_exits"],
            "v2_confirmed_exits": v2_diag["observer_exits"],
            "v1_premature_target_exits": v1_diag[
                "premature_exits_before_baseline_target"
            ],
            "v2_premature_target_exits": v2_diag[
                "premature_exits_before_baseline_target"
            ],
            "v2_confirmed_exit_total_delta_net_r": v2_diag[
                "observer_exit_total_delta_net_r"
            ],
            "baseline_2x_net_r": baseline_2x["net_r"],
            "v1_2x_net_r": v1_2x["net_r"],
            "v2_2x_net_r": v2_2x["net_r"],
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
