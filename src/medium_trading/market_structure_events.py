from bisect import bisect_left
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
from medium_trading.market_structure_model import (
    ACCEPTANCE_MIN_CLOSES,
    ACCEPTANCE_WINDOW_BARS,
    BREAK_CLOSE_ATR,
    FAST_RECLAIM_BARS,
    RETEST_TOLERANCE_ATR,
    SWING_LEFT_BARS,
    SWING_RIGHT_BARS,
    _atr,
    _confirmed_swings,
)

MAX_HOLDING_HOURS = 24
TARGET_R = 2.0
SWING_HISTORY_BARS = 96

SWEEP_RECLAIM = "SWEEP_RECLAIM"
BODY_BREAK_UNCONFIRMED = "BODY_BREAK_UNCONFIRMED"
BREAK_ACCEPTED = "BREAK_ACCEPTED"
RETEST_HELD = "RETEST_HELD"
BREAK_RECLAIMED = "BREAK_RECLAIMED"


@dataclass(frozen=True, slots=True)
class StructureEvent:
    kind: str
    signal_time: datetime
    candle_index: int
    level: float
    atr5: float


@dataclass(frozen=True, slots=True)
class PathResult:
    exit_time: datetime
    exit_index: int
    exit_price: float
    exit_reason: str
    gross_r: float
    cost_r: float
    net_r: float


@dataclass(frozen=True, slots=True)
class TradeComparison:
    source_trade: BacktestTrade
    original: PathResult
    managed: PathResult
    events: tuple[StructureEvent, ...]
    break_outcomes: tuple[str, ...]
    sweep_followthrough: tuple[tuple[bool, bool, bool], ...]
    structural_exit_trigger: str | None

    @property
    def delta_gross_r(self) -> float:
        return self.managed.gross_r - self.original.gross_r

    @property
    def delta_net_r(self) -> float:
        return self.managed.net_r - self.original.net_r


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    trades: int
    gross_r: float
    total_cost_r: float
    net_r: float
    average_net_r: float
    gross_profit_factor: float
    profit_factor: float
    win_rate: float
    max_drawdown: float


@dataclass(slots=True)
class _BreakState:
    level: float
    atr5: float
    start_index: int
    closes_below: int


@dataclass(slots=True)
class _SweepState:
    level: float
    atr5: float
    start_index: int


def evaluate_market_structure_events_v03(
    *,
    candles_30m: tuple[Candle, ...],
    candles_5m: tuple[Candle, ...],
    symbol: str = "BTCUSDT",
    fee_bps_per_side: float = BTC_TAKER_FEE_BPS_PER_SIDE,
    slippage_bps_per_side: float = BTC_SLIPPAGE_BPS_PER_SIDE,
    starting_equity: float = 10_000.0,
    risk_fraction: float = 0.005,
) -> dict[str, object]:
    """Evaluate deterministic causal structure management on the fixed candidate stream."""
    if not candles_5m:
        raise ValueError("market-structure event evaluation requires M5 candles")
    _require_contiguous_m5(candles_5m)

    samples = extract_btc_long_noise_samples(
        candles=candles_30m,
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    source_trades = tuple(sample.trade for sample in samples if sample.economic_pass)
    if not source_trades:
        raise ValueError("market-structure event evaluation found no economic-pass trades")

    comparisons = tuple(
        _simulate_trade_pair(
            candles_5m=candles_5m,
            trade=trade,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        for trade in source_trades
    )

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": "deterministic causal event-driven trade-management feasibility",
            "development_years": [2023, 2024, 2025],
            "2026_used": False,
            "candidate_stream": "economic-pass Trend LONG v1.1 executed candidates",
            "replacement_candidates": False,
            "execution_resolution": "native M5",
            "ml_used": False,
        },
        "rules": {
            "swing_confirmation": (
                f"{SWING_LEFT_BARS} bars left / {SWING_RIGHT_BARS} bars right"
            ),
            "break_close_atr": BREAK_CLOSE_ATR,
            "acceptance_window_bars": ACCEPTANCE_WINDOW_BARS,
            "acceptance_min_closes": ACCEPTANCE_MIN_CLOSES,
            "retest_tolerance_atr": RETEST_TOLERANCE_ATR,
            "fast_reclaim_bars": FAST_RECLAIM_BARS,
            "hard_stop": "unchanged Trend LONG v1.1 stop",
            "target_r": TARGET_R,
            "max_holding_hours": MAX_HOLDING_HOURS,
            "event_exit_execution": "next M5 open after the confirming candle",
            "exit_events": [BREAK_ACCEPTED, RETEST_HELD],
            "hold_events": [SWEEP_RECLAIM, BREAK_RECLAIMED],
            "wait_event": BODY_BREAK_UNCONFIRMED,
        },
        "combined": _comparison_summary(
            comparisons,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
        ),
        "years": [
            {
                "year": year,
                **_comparison_summary(
                    tuple(
                        item
                        for item in comparisons
                        if item.source_trade.entry_time.year == year
                    ),
                    starting_equity=starting_equity,
                    risk_fraction=risk_fraction,
                ),
            }
            for year in (2023, 2024, 2025)
        ],
        "trades": [_trade_payload(item) for item in comparisons],
    }


def _simulate_trade_pair(
    *,
    candles_5m: tuple[Candle, ...],
    trade: BacktestTrade,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> TradeComparison:
    times = [candle.timestamp for candle in candles_5m]
    entry_index = bisect_left(times, trade.entry_time)
    if entry_index >= len(candles_5m) or times[entry_index] != trade.entry_time:
        raise ValueError(
            "M5 history does not contain exact trade entry "
            f"{trade.entry_time.isoformat()}"
        )

    original = _simulate_original_path(
        candles_5m=candles_5m,
        entry_index=entry_index,
        trade=trade,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    managed, events, break_outcomes, structural_exit_trigger = _simulate_managed_path(
        candles_5m=candles_5m,
        entry_index=entry_index,
        trade=trade,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )

    sweep_followthrough = tuple(
        _sweep_followthrough(
            candles_5m=candles_5m,
            event=event,
            original=original,
            trade=trade,
        )
        for event in events
        if event.kind == SWEEP_RECLAIM
    )

    return TradeComparison(
        source_trade=trade,
        original=original,
        managed=managed,
        events=events,
        break_outcomes=break_outcomes,
        sweep_followthrough=sweep_followthrough,
        structural_exit_trigger=structural_exit_trigger,
    )


def _simulate_original_path(
    *,
    candles_5m: tuple[Candle, ...],
    entry_index: int,
    trade: BacktestTrade,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> PathResult:
    return _simulate_hard_path(
        candles_5m=candles_5m,
        entry_index=entry_index,
        trade=trade,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )


def _simulate_managed_path(
    *,
    candles_5m: tuple[Candle, ...],
    entry_index: int,
    trade: BacktestTrade,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> tuple[PathResult, tuple[StructureEvent, ...], tuple[str, ...], str | None]:
    risk_distance = trade.entry - trade.stop
    if risk_distance <= 0:
        raise ValueError("v0.3 supports LONG trades with stop below entry only")
    target = trade.entry + TARGET_R * risk_distance
    last_index = _last_m5_index(candles_5m, entry_index, trade.entry_time)

    events: list[StructureEvent] = []
    break_outcomes: list[str] = []
    break_state: _BreakState | None = None
    sweep_state: _SweepState | None = None
    pending_exit: str | None = None

    for index in range(entry_index, last_index + 1):
        candle = candles_5m[index]

        if pending_exit is not None:
            if candle.open <= trade.stop:
                path = _path_result(
                    trade=trade,
                    exit_time=candle.timestamp,
                    exit_index=index,
                    exit_price=candle.open,
                    exit_reason="stop_gap",
                    fee_bps_per_side=fee_bps_per_side,
                    slippage_bps_per_side=slippage_bps_per_side,
                )
                return path, tuple(events), tuple(break_outcomes), None
            if candle.open >= target:
                path = _path_result(
                    trade=trade,
                    exit_time=candle.timestamp,
                    exit_index=index,
                    exit_price=target,
                    exit_reason="target",
                    fee_bps_per_side=fee_bps_per_side,
                    slippage_bps_per_side=slippage_bps_per_side,
                )
                return path, tuple(events), tuple(break_outcomes), None
            path = _path_result(
                trade=trade,
                exit_time=candle.timestamp,
                exit_index=index,
                exit_price=candle.open,
                exit_reason=f"structure_{pending_exit.lower()}",
                fee_bps_per_side=fee_bps_per_side,
                slippage_bps_per_side=slippage_bps_per_side,
            )
            return path, tuple(events), tuple(break_outcomes), pending_exit

        hard_exit = _hard_exit_on_candle(
            candle=candle,
            index=index,
            trade=trade,
            target=target,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        if hard_exit is not None:
            if break_state is not None:
                break_outcomes.append("TRADE_ENDED")
            return hard_exit, tuple(events), tuple(break_outcomes), None

        if break_state is not None:
            (
                break_state,
                pending_exit,
                new_events,
                break_outcome,
            ) = _update_break_state(
                candles_5m=candles_5m,
                index=index,
                state=break_state,
            )
            events.extend(new_events)
            if break_outcome is not None:
                break_outcomes.append(break_outcome)
            if pending_exit is not None:
                continue
            if break_state is not None:
                continue

        if sweep_state is not None:
            elapsed = index - sweep_state.start_index
            break_level = sweep_state.level - BREAK_CLOSE_ATR * sweep_state.atr5
            if candle.close < break_level:
                event = StructureEvent(
                    kind=BODY_BREAK_UNCONFIRMED,
                    signal_time=candle.timestamp + timedelta(minutes=5),
                    candle_index=index,
                    level=sweep_state.level,
                    atr5=sweep_state.atr5,
                )
                events.append(event)
                break_state = _BreakState(
                    level=sweep_state.level,
                    atr5=sweep_state.atr5,
                    start_index=index,
                    closes_below=1,
                )
                sweep_state = None
                continue
            if elapsed <= FAST_RECLAIM_BARS and candle.close > sweep_state.level:
                events.append(
                    StructureEvent(
                        kind=SWEEP_RECLAIM,
                        signal_time=candle.timestamp + timedelta(minutes=5),
                        candle_index=index,
                        level=sweep_state.level,
                        atr5=sweep_state.atr5,
                    )
                )
                sweep_state = None
                continue
            if elapsed >= FAST_RECLAIM_BARS:
                sweep_state = None

        structure = _latest_structure(candles_5m, index)
        if structure is None:
            continue
        level, atr5 = structure
        break_level = level - BREAK_CLOSE_ATR * atr5

        if candle.close < break_level:
            events.append(
                StructureEvent(
                    kind=BODY_BREAK_UNCONFIRMED,
                    signal_time=candle.timestamp + timedelta(minutes=5),
                    candle_index=index,
                    level=level,
                    atr5=atr5,
                )
            )
            break_state = _BreakState(
                level=level,
                atr5=atr5,
                start_index=index,
                closes_below=1,
            )
            continue

        if candle.low < level:
            if candle.close > level:
                events.append(
                    StructureEvent(
                        kind=SWEEP_RECLAIM,
                        signal_time=candle.timestamp + timedelta(minutes=5),
                        candle_index=index,
                        level=level,
                        atr5=atr5,
                    )
                )
            else:
                sweep_state = _SweepState(
                    level=level,
                    atr5=atr5,
                    start_index=index,
                )

    if break_state is not None:
        break_outcomes.append("UNRESOLVED")

    last = candles_5m[last_index]
    path = _path_result(
        trade=trade,
        exit_time=last.timestamp + timedelta(minutes=5),
        exit_index=last_index,
        exit_price=last.close,
        exit_reason="timeout",
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    return path, tuple(events), tuple(break_outcomes), None


def _update_break_state(
    *,
    candles_5m: tuple[Candle, ...],
    index: int,
    state: _BreakState,
) -> tuple[
    _BreakState | None,
    str | None,
    tuple[StructureEvent, ...],
    str | None,
]:
    candle = candles_5m[index]
    elapsed = index - state.start_index
    if elapsed <= 0:
        return state, None, (), None

    if elapsed <= FAST_RECLAIM_BARS and candle.close > state.level:
        event = StructureEvent(
            kind=BREAK_RECLAIMED,
            signal_time=candle.timestamp + timedelta(minutes=5),
            candle_index=index,
            level=state.level,
            atr5=state.atr5,
        )
        return None, None, (event,), BREAK_RECLAIMED

    if elapsed < ACCEPTANCE_WINDOW_BARS and candle.close < state.level:
        state.closes_below += 1

    tolerance = RETEST_TOLERANCE_ATR * state.atr5
    retest_held = (
        elapsed < ACCEPTANCE_WINDOW_BARS
        and candle.high >= state.level - tolerance
        and candle.close < state.level
    )
    accepted = state.closes_below >= ACCEPTANCE_MIN_CLOSES

    new_events: list[StructureEvent] = []
    if accepted:
        new_events.append(
            StructureEvent(
                kind=BREAK_ACCEPTED,
                signal_time=candle.timestamp + timedelta(minutes=5),
                candle_index=index,
                level=state.level,
                atr5=state.atr5,
            )
        )
    if retest_held:
        new_events.append(
            StructureEvent(
                kind=RETEST_HELD,
                signal_time=candle.timestamp + timedelta(minutes=5),
                candle_index=index,
                level=state.level,
                atr5=state.atr5,
            )
        )

    if retest_held:
        return None, RETEST_HELD, tuple(new_events), RETEST_HELD
    if accepted:
        return None, BREAK_ACCEPTED, tuple(new_events), BREAK_ACCEPTED

    if elapsed >= ACCEPTANCE_WINDOW_BARS - 1:
        return None, None, tuple(new_events), "UNRESOLVED"

    return state, None, tuple(new_events), None


def _simulate_hard_path(
    *,
    candles_5m: tuple[Candle, ...],
    entry_index: int,
    trade: BacktestTrade,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> PathResult:
    risk_distance = trade.entry - trade.stop
    if risk_distance <= 0:
        raise ValueError("v0.3 supports LONG trades with stop below entry only")
    target = trade.entry + TARGET_R * risk_distance
    last_index = _last_m5_index(candles_5m, entry_index, trade.entry_time)

    for index in range(entry_index, last_index + 1):
        result = _hard_exit_on_candle(
            candle=candles_5m[index],
            index=index,
            trade=trade,
            target=target,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        if result is not None:
            return result

    last = candles_5m[last_index]
    return _path_result(
        trade=trade,
        exit_time=last.timestamp + timedelta(minutes=5),
        exit_index=last_index,
        exit_price=last.close,
        exit_reason="timeout",
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )


def _hard_exit_on_candle(
    *,
    candle: Candle,
    index: int,
    trade: BacktestTrade,
    target: float,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> PathResult | None:
    if candle.open <= trade.stop:
        return _path_result(
            trade=trade,
            exit_time=candle.timestamp,
            exit_index=index,
            exit_price=candle.open,
            exit_reason="stop_gap",
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
    if candle.open >= target:
        return _path_result(
            trade=trade,
            exit_time=candle.timestamp,
            exit_index=index,
            exit_price=target,
            exit_reason="target",
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )

    # Conservative M5 intrabar ordering: if both are touched, the stop wins.
    if candle.low <= trade.stop:
        return _path_result(
            trade=trade,
            exit_time=candle.timestamp + timedelta(minutes=5),
            exit_index=index,
            exit_price=trade.stop,
            exit_reason="stop",
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
    if candle.high >= target:
        return _path_result(
            trade=trade,
            exit_time=candle.timestamp + timedelta(minutes=5),
            exit_index=index,
            exit_price=target,
            exit_reason="target",
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
    return None


def _path_result(
    *,
    trade: BacktestTrade,
    exit_time: datetime,
    exit_index: int,
    exit_price: float,
    exit_reason: str,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> PathResult:
    risk_distance = trade.entry - trade.stop
    gross_r = (exit_price - trade.entry) / risk_distance
    cost_r = (
        (trade.entry + exit_price)
        * (fee_bps_per_side + slippage_bps_per_side)
        / 10_000.0
        / risk_distance
    )
    return PathResult(
        exit_time=exit_time,
        exit_index=exit_index,
        exit_price=exit_price,
        exit_reason=exit_reason,
        gross_r=gross_r,
        cost_r=cost_r,
        net_r=gross_r - cost_r,
    )


def _latest_structure(
    candles_5m: tuple[Candle, ...],
    current_index: int,
) -> tuple[float, float] | None:
    start = max(0, current_index + 1 - SWING_HISTORY_BARS)
    history = candles_5m[start : current_index + 1]
    if len(history) < 15:
        return None
    atr5 = _atr(history, 14)
    if atr5 <= 0:
        return None
    lows, _ = _confirmed_swings(history)
    if not lows:
        return None
    return lows[-1][1], atr5


def _last_m5_index(
    candles_5m: tuple[Candle, ...],
    entry_index: int,
    entry_time: datetime,
) -> int:
    deadline = entry_time + timedelta(hours=MAX_HOLDING_HOURS)
    index = entry_index
    while index + 1 < len(candles_5m):
        if candles_5m[index + 1].timestamp >= deadline:
            break
        index += 1
    return index


def _sweep_followthrough(
    *,
    candles_5m: tuple[Candle, ...],
    event: StructureEvent,
    original: PathResult,
    trade: BacktestTrade,
) -> tuple[bool, bool, bool]:
    if event.candle_index >= original.exit_index:
        return False, False, False

    risk_distance = trade.entry - trade.stop
    future = candles_5m[event.candle_index + 1 : original.exit_index + 1]
    if not future:
        return False, False, False
    high = max(candle.high for candle in future)
    return (
        high >= trade.entry + 0.5 * risk_distance,
        high >= trade.entry + 1.0 * risk_distance,
        high >= trade.entry + TARGET_R * risk_distance,
    )


def _comparison_summary(
    comparisons: tuple[TradeComparison, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
) -> dict[str, object]:
    source = tuple(item.source_trade for item in comparisons)
    original = tuple(item.original for item in comparisons)
    managed = tuple(item.managed for item in comparisons)

    structural_exits = tuple(
        item for item in comparisons if item.structural_exit_trigger is not None
    )
    delta_net = tuple(item.delta_net_r for item in comparisons)
    delta_gross = tuple(item.delta_gross_r for item in comparisons)

    return {
        "source_m30": _source_trade_metrics(
            source,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
            cost_multiplier=1.0,
        ),
        "original_m5": asdict(
            _path_metrics(
                original,
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
                cost_multiplier=1.0,
            )
        ),
        "managed_m5": asdict(
            _path_metrics(
                managed,
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
                cost_multiplier=1.0,
            )
        ),
        "original_m5_2x_costs": asdict(
            _path_metrics(
                original,
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
                cost_multiplier=2.0,
            )
        ),
        "managed_m5_2x_costs": asdict(
            _path_metrics(
                managed,
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
                cost_multiplier=2.0,
            )
        ),
        "delta_gross_r": sum(delta_gross),
        "delta_net_r": sum(delta_net),
        "average_delta_gross_r": mean(delta_gross) if delta_gross else 0.0,
        "average_delta_net_r": mean(delta_net) if delta_net else 0.0,
        "event_diagnostics": _event_diagnostics(comparisons, structural_exits),
    }


def _event_diagnostics(
    comparisons: tuple[TradeComparison, ...],
    structural_exits: tuple[TradeComparison, ...],
) -> dict[str, object]:
    events = tuple(event for item in comparisons for event in item.events)
    event_counts = Counter(event.kind for event in events)
    break_outcomes = Counter(
        outcome for item in comparisons for outcome in item.break_outcomes
    )
    sweep_outcomes = tuple(
        outcome for item in comparisons for outcome in item.sweep_followthrough
    )

    trigger_stats = {}
    for trigger in (BREAK_ACCEPTED, RETEST_HELD):
        subset = tuple(
            item for item in structural_exits if item.structural_exit_trigger == trigger
        )
        deltas = tuple(item.delta_net_r for item in subset)
        gross_deltas = tuple(item.delta_gross_r for item in subset)
        trigger_stats[trigger] = {
            "trades": len(subset),
            "improved": sum(value > 0 for value in deltas),
            "worsened": sum(value < 0 for value in deltas),
            "unchanged": sum(value == 0 for value in deltas),
            "average_delta_net_r": mean(deltas) if deltas else 0.0,
            "average_delta_gross_r": mean(gross_deltas) if gross_deltas else 0.0,
            "total_delta_net_r": sum(deltas),
        }

    structural_deltas = tuple(item.delta_net_r for item in structural_exits)
    sweep_count = len(sweep_outcomes)
    reached_0_5r = sum(item[0] for item in sweep_outcomes)
    reached_1r = sum(item[1] for item in sweep_outcomes)
    reached_target = sum(item[2] for item in sweep_outcomes)

    return {
        "event_counts": {
            name: event_counts.get(name, 0)
            for name in (
                SWEEP_RECLAIM,
                BODY_BREAK_UNCONFIRMED,
                BREAK_RECLAIMED,
                BREAK_ACCEPTED,
                RETEST_HELD,
            )
        },
        "structural_exits": len(structural_exits),
        "structural_exit_improved": sum(value > 0 for value in structural_deltas),
        "structural_exit_worsened": sum(value < 0 for value in structural_deltas),
        "structural_exit_unchanged": sum(value == 0 for value in structural_deltas),
        "average_delta_net_r_structural_exits": (
            mean(structural_deltas) if structural_deltas else 0.0
        ),
        "exit_trigger_stats": trigger_stats,
        "body_break_outcomes": {
            name: break_outcomes.get(name, 0)
            for name in (
                BREAK_RECLAIMED,
                BREAK_ACCEPTED,
                RETEST_HELD,
                "UNRESOLVED",
                "TRADE_ENDED",
            )
        },
        "sweep_reclaim_followthrough": {
            "events": sweep_count,
            "reached_0_5r": reached_0_5r,
            "reached_1r": reached_1r,
            "reached_target": reached_target,
            "reached_0_5r_rate": reached_0_5r / sweep_count if sweep_count else 0.0,
            "reached_1r_rate": reached_1r / sweep_count if sweep_count else 0.0,
            "reached_target_rate": reached_target / sweep_count if sweep_count else 0.0,
        },
    }


def _path_metrics(
    paths: tuple[PathResult, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
    cost_multiplier: float,
) -> TradeMetrics:
    gross = tuple(path.gross_r for path in paths)
    costs = tuple(path.cost_r * cost_multiplier for path in paths)
    net = tuple(
        gross_r - cost_r
        for gross_r, cost_r in zip(gross, costs, strict=True)
    )
    return TradeMetrics(
        trades=len(paths),
        gross_r=sum(gross),
        total_cost_r=sum(costs),
        net_r=sum(net),
        average_net_r=mean(net) if net else 0.0,
        gross_profit_factor=_profit_factor(gross),
        profit_factor=_profit_factor(net),
        win_rate=sum(value > 0 for value in net) / len(net) if net else 0.0,
        max_drawdown=_max_drawdown(
            net,
            starting_equity=starting_equity,
            risk_fraction=risk_fraction,
        ),
    )


def _source_trade_metrics(
    trades: tuple[BacktestTrade, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
    cost_multiplier: float,
) -> dict[str, object]:
    gross = tuple(trade.gross_r for trade in trades)
    costs = tuple(trade.cost_r * cost_multiplier for trade in trades)
    net = tuple(
        gross_r - cost_r
        for gross_r, cost_r in zip(gross, costs, strict=True)
    )
    return asdict(
        TradeMetrics(
            trades=len(trades),
            gross_r=sum(gross),
            total_cost_r=sum(costs),
            net_r=sum(net),
            average_net_r=mean(net) if net else 0.0,
            gross_profit_factor=_profit_factor(gross),
            profit_factor=_profit_factor(net),
            win_rate=sum(value > 0 for value in net) / len(net) if net else 0.0,
            max_drawdown=_max_drawdown(
                net,
                starting_equity=starting_equity,
                risk_fraction=risk_fraction,
            ),
        )
    )


def _max_drawdown(
    net_values: tuple[float, ...],
    *,
    starting_equity: float,
    risk_fraction: float,
) -> float:
    equity = starting_equity
    peak = equity
    maximum = 0.0
    for net_r in net_values:
        equity += equity * risk_fraction * net_r
        peak = max(peak, equity)
        if peak > 0:
            maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _trade_payload(item: TradeComparison) -> dict[str, object]:
    return {
        "entry_time": item.source_trade.entry_time.isoformat(),
        "entry": item.source_trade.entry,
        "stop": item.source_trade.stop,
        "source_m30": {
            "exit_time": item.source_trade.exit_time.isoformat(),
            "exit_reason": item.source_trade.exit_reason,
            "gross_r": item.source_trade.gross_r,
            "cost_r": item.source_trade.cost_r,
            "net_r": item.source_trade.net_r,
        },
        "original_m5": _path_payload(item.original),
        "managed_m5": _path_payload(item.managed),
        "delta_gross_r": item.delta_gross_r,
        "delta_net_r": item.delta_net_r,
        "structural_exit_trigger": item.structural_exit_trigger,
        "events": [
            {
                "kind": event.kind,
                "signal_time": event.signal_time.isoformat(),
                "level": event.level,
                "atr5": event.atr5,
            }
            for event in item.events
        ],
        "break_outcomes": list(item.break_outcomes),
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


def _profit_factor(values: tuple[float, ...]) -> float:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    return positive / negative if negative else float("inf")


def _require_contiguous_m5(candles: tuple[Candle, ...]) -> None:
    for previous, current in zip(candles, candles[1:], strict=True):
        if current.timestamp - previous.timestamp != timedelta(minutes=5):
            raise ValueError(
                "M5 history has a gap between "
                f"{previous.timestamp.isoformat()} and {current.timestamp.isoformat()}"
            )
