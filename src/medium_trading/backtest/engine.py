from bisect import bisect_right
from datetime import UTC, datetime, timedelta

from medium_trading.domain import Candle, Side, Signal, StrategyContext
from medium_trading.strategy.base import Strategy

from .model import BacktestConfig, BacktestReport, BacktestTrade

_CONTEXT_WINDOW = 256


def pip_size(symbol: str) -> float:
    normalized = symbol.replace("_", "/").upper()
    quote_currency = normalized.split("/")[-1]
    return 0.01 if quote_currency == "JPY" else 0.0001


def aggregate_candles(
    candles: tuple[Candle, ...],
    period_minutes: int,
) -> tuple[Candle, ...]:
    if period_minutes <= 0 or period_minutes % 30 != 0:
        raise ValueError("period_minutes must be a positive multiple of 30")

    expected_count = period_minutes // 30
    buckets: dict[datetime, list[Candle]] = {}

    for candle in candles:
        timestamp = candle.timestamp.astimezone(UTC)
        epoch_minutes = int(timestamp.timestamp() // 60)
        bucket_epoch = epoch_minutes - (epoch_minutes % period_minutes)
        bucket_start = datetime.fromtimestamp(bucket_epoch * 60, tz=UTC)
        buckets.setdefault(bucket_start, []).append(candle)

    result: list[Candle] = []
    for bucket_start in sorted(buckets):
        group = sorted(buckets[bucket_start], key=lambda candle: candle.timestamp)
        if len(group) != expected_count:
            continue
        result.append(
            Candle(
                timestamp=bucket_start,
                open=group[0].open,
                high=max(candle.high for candle in group),
                low=min(candle.low for candle in group),
                close=group[-1].close,
                volume=sum(candle.volume for candle in group),
                spread=sum(candle.spread for candle in group) / len(group),
            )
        )
    return tuple(result)


def run_backtest(
    *,
    symbol: str,
    candles_30m: tuple[Candle, ...],
    strategy: Strategy,
    config: BacktestConfig,
    trade_start: datetime | None = None,
) -> BacktestReport:
    _validate_config(config)
    if len(candles_30m) < 3:
        return _empty_report(symbol, config)

    candles_1h = aggregate_candles(candles_30m, 60)
    candles_4h = aggregate_candles(candles_30m, 240)
    ends_1h = [candle.timestamp + timedelta(hours=1) for candle in candles_1h]
    ends_4h = [candle.timestamp + timedelta(hours=4) for candle in candles_4h]

    equity = config.starting_equity
    peak_equity = equity
    max_drawdown = 0.0
    signal_count = 0
    cost_rejections = 0
    invalidated_before_entry = 0
    trades: list[BacktestTrade] = []

    index = 0
    while index < len(candles_30m) - 1:
        decision_time = candles_30m[index].timestamp + timedelta(minutes=30)
        if trade_start is not None and decision_time < trade_start:
            index += 1
            continue

        one_hour_count = bisect_right(ends_1h, decision_time)
        four_hour_count = bisect_right(ends_4h, decision_time)

        context = StrategyContext(
            symbol=symbol,
            candles_4h=_tail(candles_4h, four_hour_count),
            candles_1h=_tail(candles_1h, one_hour_count),
            candles_30m=_tail(candles_30m, index + 1),
        )
        signal = strategy.evaluate(context)
        if signal is None:
            index += 1
            continue

        signal_count += 1
        entry_index = index + 1
        entry = candles_30m[entry_index].open

        if not _entry_has_valid_stop(signal, entry):
            invalidated_before_entry += 1
            index += 1
            continue

        risk_distance = abs(entry - signal.stop)
        target = _target_price(signal, entry, risk_distance, config.target_r)
        if not _entry_has_valid_target(signal, entry, target):
            invalidated_before_entry += 1
            index += 1
            continue

        effective_cost_pips = (
            config.round_trip_cost_pips * config.cost_stress_multiplier
        )
        expected_move_pips = abs(target - entry) / pip_size(symbol)
        if expected_move_pips / effective_cost_pips < config.minimum_cost_multiple:
            cost_rejections += 1
            index += 1
            continue

        trade, exit_index = _simulate_trade(
            signal=signal,
            symbol=symbol,
            entry=entry,
            target=target,
            entry_index=entry_index,
            candles_30m=candles_30m,
            config=config,
            effective_cost_pips=effective_cost_pips,
        )
        trades.append(trade)

        equity += equity * config.risk_fraction * trade.net_r
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)

        index = exit_index + 1

    positive_net_r = sum(trade.net_r for trade in trades if trade.net_r > 0)
    negative_net_r = abs(sum(trade.net_r for trade in trades if trade.net_r < 0))
    positive_gross_r = sum(trade.gross_r for trade in trades if trade.gross_r > 0)
    negative_gross_r = abs(sum(trade.gross_r for trade in trades if trade.gross_r < 0))

    profit_factor = (
        positive_net_r / negative_net_r if negative_net_r else float("inf")
    )
    gross_profit_factor = (
        positive_gross_r / negative_gross_r if negative_gross_r else float("inf")
    )
    wins = sum(trade.net_r > 0 for trade in trades)
    total_holding_hours = sum(
        (trade.exit_time - trade.entry_time).total_seconds() / 3600
        for trade in trades
    )
    stop_exits = sum(
        trade.exit_reason in {"stop", "stop_gap"} for trade in trades
    )
    target_exits = sum(trade.exit_reason == "target" for trade in trades)
    timeout_exits = sum(trade.exit_reason == "timeout" for trade in trades)

    return BacktestReport(
        symbol=symbol,
        signal_count=signal_count,
        cost_rejections=cost_rejections,
        invalidated_before_entry=invalidated_before_entry,
        trades=tuple(trades),
        starting_equity=config.starting_equity,
        final_equity=equity,
        net_r=sum(trade.net_r for trade in trades),
        gross_r=sum(trade.gross_r for trade in trades),
        total_cost_r=sum(trade.cost_r for trade in trades),
        profit_factor=profit_factor,
        gross_profit_factor=gross_profit_factor,
        win_rate=(wins / len(trades)) if trades else 0.0,
        max_drawdown=max_drawdown,
        average_holding_hours=(total_holding_hours / len(trades)) if trades else 0.0,
        stop_exits=stop_exits,
        target_exits=target_exits,
        timeout_exits=timeout_exits,
    )


def _tail(candles: tuple[Candle, ...], end: int) -> tuple[Candle, ...]:
    start = max(0, end - _CONTEXT_WINDOW)
    return candles[start:end]


def _entry_has_valid_stop(signal: Signal, entry: float) -> bool:
    if signal.side is Side.LONG:
        return signal.stop < entry
    return signal.stop > entry


def _entry_has_valid_target(signal: Signal, entry: float, target: float) -> bool:
    if signal.side is Side.LONG:
        return target > entry
    return target < entry


def _target_price(
    signal: Signal,
    entry: float,
    risk_distance: float,
    target_r: float,
) -> float:
    if signal.target is not None:
        return signal.target
    if signal.side is Side.LONG:
        return entry + target_r * risk_distance
    return entry - target_r * risk_distance


def _simulate_trade(
    *,
    signal: Signal,
    symbol: str,
    entry: float,
    target: float,
    entry_index: int,
    candles_30m: tuple[Candle, ...],
    config: BacktestConfig,
    effective_cost_pips: float,
) -> tuple[BacktestTrade, int]:
    risk_distance = abs(entry - signal.stop)
    last_index = min(
        len(candles_30m) - 1,
        entry_index + config.max_holding_bars - 1,
    )
    exit_price = candles_30m[last_index].close
    exit_reason = "timeout"
    exit_index = last_index

    for current_index in range(entry_index, last_index + 1):
        candle = candles_30m[current_index]

        if signal.side is Side.LONG:
            if candle.open <= signal.stop:
                exit_price = candle.open
                exit_reason = "stop_gap"
                exit_index = current_index
                break
            if candle.open >= target:
                exit_price = target
                exit_reason = "target"
                exit_index = current_index
                break
            stop_hit = candle.low <= signal.stop
            target_hit = candle.high >= target
        else:
            if candle.open >= signal.stop:
                exit_price = candle.open
                exit_reason = "stop_gap"
                exit_index = current_index
                break
            if candle.open <= target:
                exit_price = target
                exit_reason = "target"
                exit_index = current_index
                break
            stop_hit = candle.high >= signal.stop
            target_hit = candle.low <= target

        # Conservative intrabar assumption: if both were touched after the open, stop wins.
        if stop_hit:
            exit_price = signal.stop
            exit_reason = "stop"
            exit_index = current_index
            break
        if target_hit:
            exit_price = target
            exit_reason = "target"
            exit_index = current_index
            break

    gross_r = (
        (exit_price - entry) / risk_distance
        if signal.side is Side.LONG
        else (entry - exit_price) / risk_distance
    )
    cost_r = effective_cost_pips / (risk_distance / pip_size(symbol))
    net_r = gross_r - cost_r

    return (
        BacktestTrade(
            symbol=symbol,
            side=signal.side,
            entry_time=candles_30m[entry_index].timestamp,
            exit_time=candles_30m[exit_index].timestamp + timedelta(minutes=30),
            entry=entry,
            stop=signal.stop,
            exit=exit_price,
            gross_r=gross_r,
            net_r=net_r,
            cost_r=cost_r,
            exit_reason=exit_reason,
        ),
        exit_index,
    )


def _validate_config(config: BacktestConfig) -> None:
    if config.starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    if not 0 < config.risk_fraction < 1:
        raise ValueError("risk_fraction must be between 0 and 1")
    if config.target_r <= 0:
        raise ValueError("target_r must be positive")
    if config.max_holding_bars <= 0:
        raise ValueError("max_holding_bars must be positive")
    if config.round_trip_cost_pips <= 0:
        raise ValueError("round_trip_cost_pips must be positive")
    if config.cost_stress_multiplier <= 0:
        raise ValueError("cost_stress_multiplier must be positive")
    if config.minimum_cost_multiple <= 0:
        raise ValueError("minimum_cost_multiple must be positive")


def _empty_report(symbol: str, config: BacktestConfig) -> BacktestReport:
    return BacktestReport(
        symbol=symbol,
        signal_count=0,
        cost_rejections=0,
        invalidated_before_entry=0,
        trades=(),
        starting_equity=config.starting_equity,
        final_equity=config.starting_equity,
        net_r=0.0,
        gross_r=0.0,
        total_cost_r=0.0,
        profit_factor=0.0,
        gross_profit_factor=0.0,
        win_rate=0.0,
        max_drawdown=0.0,
        average_holding_hours=0.0,
        stop_exits=0,
        target_exits=0,
        timeout_exits=0,
    )
