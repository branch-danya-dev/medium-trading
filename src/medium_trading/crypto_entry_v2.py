from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import inf, sqrt
from random import Random
from statistics import mean, median, stdev

from medium_trading.backtest.engine import (
    aggregate_candles,
    run_backtest,
    simulate_trade,
)
from medium_trading.backtest.model import BacktestConfig, BacktestTrade
from medium_trading.crypto_long_baseline import (
    BTC_MAX_HOLDING_MINUTES,
    BTC_REQUIRED_CONTEXT_BARS,
    BTC_REQUIRED_FUTURE_BARS,
    BTC_SLIPPAGE_BPS_PER_SIDE,
    BTC_TAKER_FEE_BPS_PER_SIDE,
    TRADE_END,
    TRADE_START,
)
from medium_trading.domain import Candle, Side, Signal
from medium_trading.strategy.base import Strategy
from medium_trading.strategy.crypto_trend_long import (
    CryptoTrendLongV11Strategy,
    CryptoTrendLongV2Strategy,
)

ECONOMIC_MIN_TARGET_TO_COST = 8.0
MATCHED_RANDOM_PER_TRADE = 20
BOOTSTRAP_ITERATIONS = 2_000
RANDOM_SEED = 42

MIN_ENTRY_TRADES = 150
MIN_GROSS_PROFIT_FACTOR = 1.10


@dataclass(frozen=True, slots=True)
class EntrySample:
    trade: BacktestTrade
    expected_cost_r: float
    target_to_cost_ratio: float
    stop_fraction: float
    mfe_r_24h: float
    mae_r_24h: float
    reached_1r_before_stop: bool
    reached_2r_before_stop: bool

    @property
    def economic_pass(self) -> bool:
        return self.target_to_cost_ratio >= ECONOMIC_MIN_TARGET_TO_COST


@dataclass(frozen=True, slots=True)
class MatchedControl:
    entry_time: object
    actual_gross_r: float
    random_mean_gross_r: float
    random_mean_net_r: float
    random_reached_1r_rate: float
    random_reached_2r_rate: float
    matches: int
    fallback_used: bool

    @property
    def gross_edge_r(self) -> float:
        return self.actual_gross_r - self.random_mean_gross_r


def evaluate_btc_entry_strategy_v2(
    *,
    candles_30m: tuple[Candle, ...],
    symbol: str = "BTCUSDT",
    fee_bps_per_side: float = BTC_TAKER_FEE_BPS_PER_SIDE,
    slippage_bps_per_side: float = BTC_SLIPPAGE_BPS_PER_SIDE,
    starting_equity: float = 1_000.0,
    risk_fraction: float = 0.005,
) -> dict[str, object]:
    """
    Evaluate whether the LONG entry itself carries stable information.

    2026 is intentionally absent. The one allowed v2 modification is the
    persistent-H4 regime requirement; the M30 pullback/confirmation pattern,
    ATR stop floor, 2R target, 24h holding horizon and execution costs are
    unchanged from v1.1.
    """
    if not candles_30m:
        raise ValueError("entry v2 evaluation requires M30 candles")

    baseline = _extract_entry_samples(
        candles_30m=candles_30m,
        strategy=CryptoTrendLongV11Strategy(),
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )
    candidate = _extract_entry_samples(
        candles_30m=candles_30m,
        strategy=CryptoTrendLongV2Strategy(),
        symbol=symbol,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
    )

    baseline_economic = tuple(item for item in baseline if item.economic_pass)
    candidate_economic = tuple(item for item in candidate if item.economic_pass)
    if not baseline_economic:
        raise ValueError("baseline has no economic-pass trades")
    if not candidate_economic:
        raise ValueError("entry v2 has no economic-pass trades")

    config = _config(
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    baseline_control = _matched_random_control(
        candles_30m=candles_30m,
        samples=baseline_economic,
        regime="baseline",
        symbol=symbol,
        config=config,
    )
    candidate_control = _matched_random_control(
        candles_30m=candles_30m,
        samples=candidate_economic,
        regime="persistent",
        symbol=symbol,
        config=config,
    )

    baseline_summary = _entry_summary(baseline_economic)
    candidate_summary = _entry_summary(candidate_economic)
    baseline_summary["matched_random"] = _matched_summary(baseline_control)
    candidate_summary["matched_random"] = _matched_summary(candidate_control)
    candidate_summary["entry_gate"] = _entry_gate(
        baseline=baseline_summary,
        candidate=candidate_summary,
    )

    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "purpose": (
                "prove or reject stable positive information in the LONG entry "
                "before any further observer/policy work"
            ),
            "development_start": TRADE_START.isoformat(),
            "development_end": TRADE_END.isoformat(),
            "2026_used_for_entry_development": False,
            "observer_used": False,
            "policy_used": False,
            "entry_strategy_v1_1_changed": False,
            "entry_strategy_v2_change": (
                "persistent H4 EMA20 regime only; unchanged M30 setup and "
                "unchanged ATR14 stop floor"
            ),
            "random_control": (
                "trade-level matched random M30 entries in the same calendar "
                "year/month/hour and same causal H4 regime, preserving each "
                "real trade's stop-distance fraction"
            ),
        },
        "assumptions": {
            "fee_bps_per_side": fee_bps_per_side,
            "slippage_bps_per_side": slippage_bps_per_side,
            "target_r": 2.0,
            "max_holding_hours": 24,
            "economic_min_target_to_cost": ECONOMIC_MIN_TARGET_TO_COST,
            "matched_random_per_trade": MATCHED_RANDOM_PER_TRADE,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "random_seed": RANDOM_SEED,
            "funding_modeled": False,
            "funding_note": (
                "funding is deferred until an entry candidate first proves "
                "positive gross and execution-cost edge"
            ),
        },
        "baseline_v1_1": baseline_summary,
        "candidate_v2": candidate_summary,
        "comparison": {
            "trade_delta": (
                candidate_summary["trades"] - baseline_summary["trades"]
            ),
            "gross_r_delta": (
                candidate_summary["gross_r"] - baseline_summary["gross_r"]
            ),
            "mean_gross_r_delta": (
                candidate_summary["mean_gross_r"]
                - baseline_summary["mean_gross_r"]
            ),
            "gross_profit_factor_delta": (
                candidate_summary["gross_profit_factor"]
                - baseline_summary["gross_profit_factor"]
            ),
            "net_r_delta": (
                candidate_summary["net_r"] - baseline_summary["net_r"]
            ),
            "profit_factor_delta": (
                candidate_summary["profit_factor"]
                - baseline_summary["profit_factor"]
            ),
            "2x_cost_net_r_delta": (
                candidate_summary["2x_cost"]["net_r"]
                - baseline_summary["2x_cost"]["net_r"]
            ),
        },
        "candidate_trades": [_sample_payload(item) for item in candidate_economic],
    }


def _extract_entry_samples(
    *,
    candles_30m: tuple[Candle, ...],
    strategy: Strategy,
    symbol: str,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    starting_equity: float,
    risk_fraction: float,
) -> tuple[EntrySample, ...]:
    config = _config(
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    report = run_backtest(
        symbol=symbol,
        candles_30m=candles_30m,
        strategy=strategy,
        config=config,
        trade_start=TRADE_START,
        trade_end=TRADE_END,
    )
    by_time = {
        candle.timestamp: index
        for index, candle in enumerate(candles_30m)
    }

    samples: list[EntrySample] = []
    for trade in report.trades:
        entry_index = by_time.get(trade.entry_time)
        if entry_index is None:
            continue
        risk_distance = trade.entry - trade.stop
        if risk_distance <= 0:
            continue
        expected_cost_r = _expected_target_cost_r(
            trade=trade,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        target_to_cost_ratio = (
            2.0 / expected_cost_r
            if expected_cost_r > 0
            else inf
        )
        mfe_r, mae_r, reached_1r, reached_2r = _path_diagnostics(
            candles_30m=candles_30m,
            entry_index=entry_index,
            entry=trade.entry,
            stop=trade.stop,
        )
        samples.append(
            EntrySample(
                trade=trade,
                expected_cost_r=expected_cost_r,
                target_to_cost_ratio=target_to_cost_ratio,
                stop_fraction=risk_distance / trade.entry,
                mfe_r_24h=mfe_r,
                mae_r_24h=mae_r,
                reached_1r_before_stop=reached_1r,
                reached_2r_before_stop=reached_2r,
            )
        )
    return tuple(samples)


def _config(
    *,
    starting_equity: float,
    risk_fraction: float,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> BacktestConfig:
    return BacktestConfig(
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
        target_r=2.0,
        max_holding_bars=48,
        round_trip_cost_pips=0.000001,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        minimum_cost_multiple=0.01,
        max_holding_minutes=BTC_MAX_HOLDING_MINUTES,
        required_contiguous_context_bars=BTC_REQUIRED_CONTEXT_BARS,
        required_contiguous_future_bars=BTC_REQUIRED_FUTURE_BARS,
    )


def _entry_summary(samples: tuple[EntrySample, ...]) -> dict[str, object]:
    gross = tuple(item.trade.gross_r for item in samples)
    costs = tuple(item.trade.cost_r for item in samples)
    net = tuple(
        gross_r - cost_r
        for gross_r, cost_r in zip(gross, costs, strict=True)
    )
    net_2x = tuple(
        gross_r - 2.0 * cost_r
        for gross_r, cost_r in zip(gross, costs, strict=True)
    )
    years = sorted({item.trade.entry_time.year for item in samples})

    gross_ci = _bootstrap_mean_ci(gross, seed=RANDOM_SEED + 1)
    net_ci = _bootstrap_mean_ci(net, seed=RANDOM_SEED + 2)

    return {
        "trades": len(samples),
        "gross_r": sum(gross),
        "mean_gross_r": mean(gross),
        "gross_mean_bootstrap_95pct": gross_ci,
        "gross_t_stat_vs_zero": _t_stat(gross),
        "gross_profit_factor": _profit_factor(gross),
        "net_r": sum(net),
        "mean_net_r": mean(net),
        "net_mean_bootstrap_95pct": net_ci,
        "profit_factor": _profit_factor(net),
        "win_rate": sum(value > 0 for value in net) / len(net),
        "total_cost_r": sum(costs),
        "average_cost_r": mean(costs),
        "median_cost_r": median(costs),
        "average_stop_fraction": mean(
            item.stop_fraction for item in samples
        ),
        "median_stop_fraction": median(
            item.stop_fraction for item in samples
        ),
        "average_mfe_r_24h": mean(item.mfe_r_24h for item in samples),
        "median_mfe_r_24h": median(item.mfe_r_24h for item in samples),
        "average_mae_r_24h": mean(item.mae_r_24h for item in samples),
        "median_mae_r_24h": median(item.mae_r_24h for item in samples),
        "reached_1r_before_stop_rate": (
            sum(item.reached_1r_before_stop for item in samples) / len(samples)
        ),
        "reached_2r_before_stop_rate": (
            sum(item.reached_2r_before_stop for item in samples) / len(samples)
        ),
        "exit_reasons": dict(Counter(item.trade.exit_reason for item in samples)),
        "2x_cost": {
            "net_r": sum(net_2x),
            "mean_net_r": mean(net_2x),
            "profit_factor": _profit_factor(net_2x),
            "win_rate": sum(value > 0 for value in net_2x) / len(net_2x),
        },
        "years": [
            {
                "year": year,
                **_basic_summary(
                    tuple(
                        item
                        for item in samples
                        if item.trade.entry_time.year == year
                    )
                ),
            }
            for year in years
        ],
        "stop_geometry": _stop_geometry(samples),
    }


def _basic_summary(samples: tuple[EntrySample, ...]) -> dict[str, object]:
    if not samples:
        return {
            "trades": 0,
            "gross_r": 0.0,
            "mean_gross_r": 0.0,
            "gross_profit_factor": 0.0,
            "net_r": 0.0,
            "profit_factor": 0.0,
            "reached_1r_before_stop_rate": 0.0,
            "reached_2r_before_stop_rate": 0.0,
        }
    gross = tuple(item.trade.gross_r for item in samples)
    net = tuple(item.trade.net_r for item in samples)
    return {
        "trades": len(samples),
        "gross_r": sum(gross),
        "mean_gross_r": mean(gross),
        "gross_profit_factor": _profit_factor(gross),
        "net_r": sum(net),
        "profit_factor": _profit_factor(net),
        "reached_1r_before_stop_rate": (
            sum(item.reached_1r_before_stop for item in samples) / len(samples)
        ),
        "reached_2r_before_stop_rate": (
            sum(item.reached_2r_before_stop for item in samples) / len(samples)
        ),
    }


def _stop_geometry(
    samples: tuple[EntrySample, ...],
) -> dict[str, object]:
    bins = (
        ("lt_0_8pct", 0.0, 0.008),
        ("0_8_to_1_0pct", 0.008, 0.010),
        ("1_0_to_1_5pct", 0.010, 0.015),
        ("gte_1_5pct", 0.015, inf),
    )
    result: dict[str, object] = {}
    for name, lower, upper in bins:
        subset = tuple(
            item
            for item in samples
            if lower <= item.stop_fraction < upper
        )
        result[name] = _basic_summary(subset)
    return result


def _matched_random_control(
    *,
    candles_30m: tuple[Candle, ...],
    samples: tuple[EntrySample, ...],
    regime: str,
    symbol: str,
    config: BacktestConfig,
) -> tuple[MatchedControl, ...]:
    by_time = {
        candle.timestamp: index
        for index, candle in enumerate(candles_30m)
    }
    candidate_pools = _random_candidate_pools(
        candles_30m=candles_30m,
        regime=regime,
    )
    actual_times = {item.trade.entry_time for item in samples}
    rng = Random(RANDOM_SEED + (0 if regime == "baseline" else 100))

    controls: list[MatchedControl] = []
    for sample in samples:
        trade = sample.trade
        key = (
            trade.entry_time.year,
            trade.entry_time.month,
            trade.entry_time.hour,
        )
        pool = [
            index
            for index in candidate_pools["exact"].get(key, ())
            if candles_30m[index].timestamp not in actual_times
        ]
        fallback = False
        if not pool:
            fallback = True
            fallback_key = (
                trade.entry_time.year,
                trade.entry_time.hour,
            )
            pool = [
                index
                for index in candidate_pools["fallback"].get(fallback_key, ())
                if candles_30m[index].timestamp not in actual_times
            ]
        if not pool:
            continue

        match_count = min(MATCHED_RANDOM_PER_TRADE, len(pool))
        chosen = rng.sample(pool, match_count)
        random_gross: list[float] = []
        random_net: list[float] = []
        reached_1r: list[bool] = []
        reached_2r: list[bool] = []

        for entry_index in chosen:
            entry = candles_30m[entry_index].open
            risk_distance = entry * sample.stop_fraction
            if risk_distance <= 0 or risk_distance >= entry:
                continue
            stop = entry - risk_distance
            target = entry + 2.0 * risk_distance
            signal = Signal(
                symbol=symbol,
                side=Side.LONG,
                entry=entry,
                stop=stop,
                target=target,
                confidence=1.0,
                strategy="matched_random_control",
                reasons=("matched causal H4 regime random control",),
            )
            random_trade, _ = simulate_trade(
                signal=signal,
                symbol=symbol,
                entry=entry,
                target=target,
                entry_index=entry_index,
                candles_30m=candles_30m,
                config=config,
            )
            _, _, hit_1r, hit_2r = _path_diagnostics(
                candles_30m=candles_30m,
                entry_index=entry_index,
                entry=entry,
                stop=stop,
            )
            random_gross.append(random_trade.gross_r)
            random_net.append(random_trade.net_r)
            reached_1r.append(hit_1r)
            reached_2r.append(hit_2r)

        if not random_gross:
            continue
        controls.append(
            MatchedControl(
                entry_time=trade.entry_time,
                actual_gross_r=trade.gross_r,
                random_mean_gross_r=mean(random_gross),
                random_mean_net_r=mean(random_net),
                random_reached_1r_rate=(
                    sum(reached_1r) / len(reached_1r)
                ),
                random_reached_2r_rate=(
                    sum(reached_2r) / len(reached_2r)
                ),
                matches=len(random_gross),
                fallback_used=fallback,
            )
        )
    return tuple(controls)


def _random_candidate_pools(
    *,
    candles_30m: tuple[Candle, ...],
    regime: str,
) -> dict[str, dict[tuple[int, ...], tuple[int, ...]]]:
    if regime not in {"baseline", "persistent"}:
        raise ValueError("unknown random-control regime")

    candles_4h = aggregate_candles(candles_30m, 240)
    ends_4h = [candle.timestamp for candle in candles_4h]
    ends_4h = [timestamp.replace() for timestamp in ends_4h]
    # aggregate_candles timestamps are bucket starts; the bar is known 4h later.
    from datetime import timedelta

    known_4h = [timestamp + timedelta(hours=4) for timestamp in ends_4h]
    strategy = CryptoTrendLongV2Strategy()
    regime_cache: dict[int, bool] = {}

    exact: dict[tuple[int, ...], list[int]] = defaultdict(list)
    fallback: dict[tuple[int, ...], list[int]] = defaultdict(list)
    last_entry_index = len(candles_30m) - BTC_REQUIRED_FUTURE_BARS

    for index, candle in enumerate(candles_30m[:last_entry_index]):
        if candle.timestamp < TRADE_START or candle.timestamp > TRADE_END:
            continue
        four_hour_count = bisect_right(known_4h, candle.timestamp)
        if four_hour_count not in regime_cache:
            history = candles_4h[max(0, four_hour_count - 256):four_hour_count]
            regime_cache[four_hour_count] = _h4_regime_ok(
                history=history,
                persistent=regime == "persistent",
                strategy=strategy,
            )
        if not regime_cache[four_hour_count]:
            continue
        exact[
            (candle.timestamp.year, candle.timestamp.month, candle.timestamp.hour)
        ].append(index)
        fallback[(candle.timestamp.year, candle.timestamp.hour)].append(index)

    return {
        "exact": {key: tuple(value) for key, value in exact.items()},
        "fallback": {key: tuple(value) for key, value in fallback.items()},
    }


def _h4_regime_ok(
    *,
    history: tuple[Candle, ...],
    persistent: bool,
    strategy: CryptoTrendLongV2Strategy,
) -> bool:
    required = strategy.ema_period + strategy.ema_slope_lookback
    if len(history) < required:
        return False
    ema = strategy._ema_series(
        tuple(candle.close for candle in history),
        strategy.ema_period,
    )
    current = history[-1]
    if current.close <= ema[-1]:
        return False
    if ema[-1] <= ema[-1 - strategy.ema_slope_lookback]:
        return False
    if not persistent:
        return True

    bars = strategy.ema_slope_lookback
    recent_ema = ema[-(bars + 1):]
    if not all(
        current_ema > previous_ema
        for previous_ema, current_ema in zip(
            recent_ema,
            recent_ema[1:],
            strict=True,
        )
    ):
        return False
    if not all(
        candle.close > ema_value
        for candle, ema_value in zip(
            history[-bars:],
            ema[-bars:],
            strict=True,
        )
    ):
        return False
    return history[-1].close > history[-1 - bars].close


def _matched_summary(
    controls: tuple[MatchedControl, ...],
) -> dict[str, object]:
    if not controls:
        return {
            "matched_trades": 0,
            "mean_actual_gross_r": 0.0,
            "mean_random_gross_r": 0.0,
            "mean_gross_edge_r": 0.0,
            "gross_edge_bootstrap_95pct": [0.0, 0.0],
            "fallback_matches": 0,
        }
    actual = tuple(item.actual_gross_r for item in controls)
    random_gross = tuple(item.random_mean_gross_r for item in controls)
    edges = tuple(item.gross_edge_r for item in controls)
    return {
        "matched_trades": len(controls),
        "coverage": len(controls) / len(actual) if actual else 0.0,
        "average_matches_per_trade": mean(item.matches for item in controls),
        "fallback_matches": sum(item.fallback_used for item in controls),
        "mean_actual_gross_r": mean(actual),
        "mean_random_gross_r": mean(random_gross),
        "mean_random_net_r": mean(
            item.random_mean_net_r for item in controls
        ),
        "mean_gross_edge_r": mean(edges),
        "gross_edge_bootstrap_95pct": _bootstrap_mean_ci(
            edges,
            seed=RANDOM_SEED + 3,
        ),
        "random_reached_1r_before_stop_rate": mean(
            item.random_reached_1r_rate for item in controls
        ),
        "random_reached_2r_before_stop_rate": mean(
            item.random_reached_2r_rate for item in controls
        ),
    }


def _entry_gate(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    matched = candidate["matched_random"]
    years = candidate["years"]
    positive_gross_years = sum(
        float(item["gross_r"]) > 0
        for item in years
    )
    edge_ci = matched["gross_edge_bootstrap_95pct"]

    information_conditions = {
        "minimum_trades": int(candidate["trades"]) >= MIN_ENTRY_TRADES,
        "positive_gross_r": float(candidate["gross_r"]) > 0,
        "gross_profit_factor_at_least_1_10": (
            float(candidate["gross_profit_factor"])
            >= MIN_GROSS_PROFIT_FACTOR
        ),
        "positive_gross_in_at_least_2_years": positive_gross_years >= 2,
        "beats_matched_random_mean_gross": (
            float(matched["mean_gross_edge_r"]) > 0
        ),
        "matched_random_edge_bootstrap_lower_bound_positive": (
            float(edge_ci[0]) > 0
        ),
        "reached_1r_before_stop_beats_random": (
            float(candidate["reached_1r_before_stop_rate"])
            > float(matched["random_reached_1r_before_stop_rate"])
        ),
    }
    economic_conditions = {
        "positive_net_r": float(candidate["net_r"]) > 0,
        "profit_factor_above_one": float(candidate["profit_factor"]) > 1.0,
        "positive_2x_cost_net_r": (
            float(candidate["2x_cost"]["net_r"]) > 0
        ),
        "2x_cost_profit_factor_above_one": (
            float(candidate["2x_cost"]["profit_factor"]) > 1.0
        ),
        "net_r_exceeds_v1_1": (
            float(candidate["net_r"]) > float(baseline["net_r"])
        ),
        "gross_profit_factor_exceeds_v1_1": (
            float(candidate["gross_profit_factor"])
            > float(baseline["gross_profit_factor"])
        ),
    }

    all_conditions = {
        **information_conditions,
        **economic_conditions,
    }
    return {
        "passes": all(all_conditions.values()),
        "information_gate_passes": all(information_conditions.values()),
        "economic_gate_passes": all(economic_conditions.values()),
        "conditions": all_conditions,
        "thresholds": {
            "minimum_trades": MIN_ENTRY_TRADES,
            "minimum_gross_profit_factor": MIN_GROSS_PROFIT_FACTOR,
            "positive_gross_years_required": 2,
        },
        "note": (
            "PASS requires evidence that the entry carries information above "
            "matched regime-random timing and that the same frozen setup is "
            "positive after normal and 2x execution costs. Funding is not yet "
            "included and must be added before deployment."
        ),
    }


def _path_diagnostics(
    *,
    candles_30m: tuple[Candle, ...],
    entry_index: int,
    entry: float,
    stop: float,
) -> tuple[float, float, bool, bool]:
    risk_distance = entry - stop
    if risk_distance <= 0:
        return 0.0, 0.0, False, False

    deadline = candles_30m[entry_index].timestamp
    from datetime import timedelta

    deadline += timedelta(hours=24)
    horizon: list[Candle] = []
    for candle in candles_30m[entry_index:]:
        if candle.timestamp >= deadline:
            break
        horizon.append(candle)
    if not horizon:
        return 0.0, 0.0, False, False

    high = max(candle.high for candle in horizon)
    low = min(candle.low for candle in horizon)
    mfe = (high - entry) / risk_distance
    mae = (entry - low) / risk_distance

    one_r = entry + risk_distance
    two_r = entry + 2.0 * risk_distance
    hit_1r = False
    hit_2r = False
    for candle in horizon:
        if candle.open <= stop:
            break
        if candle.open >= two_r:
            hit_1r = True
            hit_2r = True
            break
        if candle.open >= one_r:
            hit_1r = True
        if candle.low <= stop:
            break
        if candle.high >= two_r:
            hit_1r = True
            hit_2r = True
            break
        if candle.high >= one_r:
            hit_1r = True

    return mfe, mae, hit_1r, hit_2r


def _expected_target_cost_r(
    *,
    trade: BacktestTrade,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> float:
    risk_distance = trade.entry - trade.stop
    if risk_distance <= 0:
        return inf
    target = trade.entry + 2.0 * risk_distance
    bps = fee_bps_per_side + slippage_bps_per_side
    price_cost = (trade.entry + target) * bps / 10_000.0
    return price_cost / risk_distance


def _bootstrap_mean_ci(
    values: tuple[float, ...],
    *,
    seed: int,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = Random(seed)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return [means[low_index], means[high_index]]


def _t_stat(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    sigma = stdev(values)
    if sigma == 0:
        return 0.0
    return mean(values) / (sigma / sqrt(len(values)))


def _profit_factor(values: tuple[float, ...]) -> float:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    return positive / negative if negative else inf


def _sample_payload(item: EntrySample) -> dict[str, object]:
    return {
        "entry_time": item.trade.entry_time.isoformat(),
        "exit_time": item.trade.exit_time.isoformat(),
        "entry": item.trade.entry,
        "stop": item.trade.stop,
        "exit": item.trade.exit,
        "exit_reason": item.trade.exit_reason,
        "gross_r": item.trade.gross_r,
        "cost_r": item.trade.cost_r,
        "net_r": item.trade.net_r,
        "expected_cost_r": item.expected_cost_r,
        "target_to_cost_ratio": item.target_to_cost_ratio,
        "stop_fraction": item.stop_fraction,
        "mfe_r_24h": item.mfe_r_24h,
        "mae_r_24h": item.mae_r_24h,
        "reached_1r_before_stop": item.reached_1r_before_stop,
        "reached_2r_before_stop": item.reached_2r_before_stop,
    }
