from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from statistics import median
from zoneinfo import ZoneInfo

from medium_trading.backtest.engine import run_backtest
from medium_trading.backtest.model import BacktestConfig, BacktestReport, BacktestTrade
from medium_trading.domain import Candle
from medium_trading.strategy.base import Strategy

MIN_SESSION_DAYS = 250
MIN_TRADES = 300
MIN_ACTIVE_DAY_RATE = 0.70
MIN_PROFITABLE_ACTIVE_DAY_RATE = 0.45
MIN_NET_PROFIT_FACTOR = 1.15
MIN_2X_PROFIT_FACTOR = 1.00
MIN_POSITIVE_YEAR_FOLDS = 2
MIN_POSITIVE_MONTH_RATE = 0.60
MIN_WORST_DAY_R = -2.0
MAX_BEST_DAY_PROFIT_SHARE = 0.15
MAX_LOSING_DAY_STREAK = 8


@dataclass(frozen=True, slots=True)
class DailyMetrics:
    session_days: int
    active_days: int
    active_day_rate: float
    profitable_days: int
    profitable_active_day_rate: float
    trades: int
    gross_r: float
    total_cost_r: float
    net_r: float
    gross_profit_factor: float
    profit_factor: float
    average_net_r_per_session: float
    median_net_r_per_session: float
    best_day_r: float
    worst_day_r: float
    longest_losing_day_streak: int
    positive_month_rate: float
    best_day_profit_share: float


@dataclass(frozen=True, slots=True)
class DailyYearEvaluation:
    year: int
    ordinary: DailyMetrics
    stressed_2x: DailyMetrics


@dataclass(frozen=True, slots=True)
class DailyStrategyEvaluation:
    symbol: str
    years: tuple[DailyYearEvaluation, ...]
    combined: DailyMetrics
    combined_2x: DailyMetrics
    positive_years: int
    passes_gate: bool


def evaluate_daily_strategy(
    *,
    symbol: str,
    candles: tuple[Candle, ...],
    strategy: Strategy,
    config: BacktestConfig,
    years: tuple[int, ...] = (2023, 2024, 2025),
    timezone: str = "America/New_York",
    required_session_time: tuple[int, int] | None = (9, 30),
    minimum_session_bars: int = 0,
) -> DailyStrategyEvaluation:
    tz = ZoneInfo(timezone)
    yearly: list[DailyYearEvaluation] = []
    ordinary_trades: list[BacktestTrade] = []
    stressed_trades: list[BacktestTrade] = []
    all_sessions = []

    for year in years:
        session_dates = _session_dates(
            candles,
            year=year,
            timezone=tz,
            required_session_time=required_session_time,
            minimum_session_bars=minimum_session_bars,
        )
        if not session_dates:
            continue

        start = datetime(year, 1, 1, tzinfo=tz).astimezone(UTC)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=tz).astimezone(UTC)

        ordinary_report = run_backtest(
            symbol=symbol,
            candles_30m=candles,
            strategy=strategy,
            config=config,
            trade_start=start,
            trade_end=end,
        )
        stressed_report = run_backtest(
            symbol=symbol,
            candles_30m=candles,
            strategy=strategy,
            config=_double_costs(config),
            trade_start=start,
            trade_end=end,
        )

        ordinary_metrics = _daily_metrics(
            ordinary_report,
            session_dates=session_dates,
            timezone=tz,
        )
        stressed_metrics = _daily_metrics(
            stressed_report,
            session_dates=session_dates,
            timezone=tz,
        )
        yearly.append(
            DailyYearEvaluation(
                year=year,
                ordinary=ordinary_metrics,
                stressed_2x=stressed_metrics,
            )
        )
        ordinary_trades.extend(ordinary_report.trades)
        stressed_trades.extend(stressed_report.trades)
        all_sessions.extend(session_dates)

    if not yearly:
        raise ValueError("daily evaluation found no eligible session days")

    combined = _metrics_from_trades(
        ordinary_trades,
        session_dates=tuple(sorted(set(all_sessions))),
        timezone=tz,
    )
    combined_2x = _metrics_from_trades(
        stressed_trades,
        session_dates=tuple(sorted(set(all_sessions))),
        timezone=tz,
    )
    positive_years = sum(item.ordinary.net_r > 0 for item in yearly)
    passes_gate = (
        combined.session_days >= MIN_SESSION_DAYS
        and combined.trades >= MIN_TRADES
        and combined.active_day_rate >= MIN_ACTIVE_DAY_RATE
        and (
            combined.profitable_active_day_rate
            >= MIN_PROFITABLE_ACTIVE_DAY_RATE
        )
        and combined.average_net_r_per_session > 0
        and combined.profit_factor >= MIN_NET_PROFIT_FACTOR
        and combined_2x.profit_factor > MIN_2X_PROFIT_FACTOR
        and positive_years >= MIN_POSITIVE_YEAR_FOLDS
        and combined.positive_month_rate >= MIN_POSITIVE_MONTH_RATE
        and combined.worst_day_r >= MIN_WORST_DAY_R
        and combined.best_day_profit_share < MAX_BEST_DAY_PROFIT_SHARE
        and combined.longest_losing_day_streak <= MAX_LOSING_DAY_STREAK
    )

    return DailyStrategyEvaluation(
        symbol=symbol,
        years=tuple(yearly),
        combined=combined,
        combined_2x=combined_2x,
        positive_years=positive_years,
        passes_gate=passes_gate,
    )


def evaluation_payload(evaluation: DailyStrategyEvaluation) -> dict[str, object]:
    return {
        "symbol": evaluation.symbol,
        "gate": {
            "min_session_days": MIN_SESSION_DAYS,
            "min_trades": MIN_TRADES,
            "min_active_day_rate": MIN_ACTIVE_DAY_RATE,
            "min_profitable_active_day_rate": MIN_PROFITABLE_ACTIVE_DAY_RATE,
            "min_net_profit_factor": MIN_NET_PROFIT_FACTOR,
            "min_2x_profit_factor_exclusive": MIN_2X_PROFIT_FACTOR,
            "min_positive_year_folds": MIN_POSITIVE_YEAR_FOLDS,
            "min_positive_month_rate": MIN_POSITIVE_MONTH_RATE,
            "min_worst_day_r": MIN_WORST_DAY_R,
            "max_best_day_profit_share": MAX_BEST_DAY_PROFIT_SHARE,
            "max_losing_day_streak": MAX_LOSING_DAY_STREAK,
            "positive_years": evaluation.positive_years,
            "passes": evaluation.passes_gate,
        },
        "years": [
            {
                "year": item.year,
                "ordinary": asdict(item.ordinary),
                "stressed_2x": asdict(item.stressed_2x),
            }
            for item in evaluation.years
        ],
        "combined": asdict(evaluation.combined),
        "combined_2x": asdict(evaluation.combined_2x),
    }


def _daily_metrics(
    report: BacktestReport,
    *,
    session_dates,
    timezone: ZoneInfo,
) -> DailyMetrics:
    return _metrics_from_trades(
        report.trades,
        session_dates=session_dates,
        timezone=timezone,
    )


def _metrics_from_trades(
    trades: Iterable[BacktestTrade],
    *,
    session_dates,
    timezone: ZoneInfo,
) -> DailyMetrics:
    items = tuple(trades)
    dates = tuple(sorted(session_dates))
    by_day: dict[object, list[BacktestTrade]] = defaultdict(list)
    for trade in items:
        by_day[trade.entry_time.astimezone(timezone).date()].append(trade)

    daily_net = {
        session_date: sum(trade.net_r for trade in by_day.get(session_date, ()))
        for session_date in dates
    }
    active_days = sum(bool(by_day.get(session_date)) for session_date in dates)
    profitable_days = sum(
        value > 0
        for session_date, value in daily_net.items()
        if by_day.get(session_date)
    )

    gross_values = tuple(trade.gross_r for trade in items)
    net_values = tuple(trade.net_r for trade in items)
    gross_pf = _profit_factor(gross_values)
    net_pf = _profit_factor(net_values)

    day_values = tuple(daily_net[session_date] for session_date in dates)
    positive_day_total = sum(value for value in day_values if value > 0)
    best_day = max(day_values, default=0.0)
    best_day_share = (
        best_day / positive_day_total
        if positive_day_total > 0 and best_day > 0
        else 0.0
    )

    monthly: dict[tuple[int, int], float] = defaultdict(float)
    for session_date, value in daily_net.items():
        monthly[(session_date.year, session_date.month)] += value
    positive_month_rate = (
        sum(value > 0 for value in monthly.values()) / len(monthly)
        if monthly
        else 0.0
    )

    return DailyMetrics(
        session_days=len(dates),
        active_days=active_days,
        active_day_rate=(active_days / len(dates)) if dates else 0.0,
        profitable_days=profitable_days,
        profitable_active_day_rate=(
            profitable_days / active_days
            if active_days
            else 0.0
        ),
        trades=len(items),
        gross_r=sum(gross_values),
        total_cost_r=sum(trade.cost_r for trade in items),
        net_r=sum(net_values),
        gross_profit_factor=gross_pf,
        profit_factor=net_pf,
        average_net_r_per_session=(
            sum(day_values) / len(day_values)
            if day_values
            else 0.0
        ),
        median_net_r_per_session=median(day_values) if day_values else 0.0,
        best_day_r=best_day,
        worst_day_r=min(day_values, default=0.0),
        longest_losing_day_streak=_longest_losing_streak(day_values),
        positive_month_rate=positive_month_rate,
        best_day_profit_share=best_day_share,
    )


def _session_dates(
    candles: tuple[Candle, ...],
    *,
    year: int,
    timezone: ZoneInfo,
    required_session_time: tuple[int, int] | None,
    minimum_session_bars: int,
):
    by_date: dict[object, list[datetime]] = defaultdict(list)
    for candle in candles:
        local = candle.timestamp.astimezone(timezone)
        if local.year == year:
            by_date[local.date()].append(local)

    required = required_session_time
    eligible = []
    for session_date, timestamps in by_date.items():
        if len(timestamps) < minimum_session_bars:
            continue
        if required is not None:
            hour, minute = required
            if not any(
                timestamp.hour == hour and timestamp.minute == minute
                for timestamp in timestamps
            ):
                continue
        eligible.append(session_date)
    return tuple(sorted(eligible))


def _profit_factor(values: Iterable[float]) -> float:
    items = tuple(values)
    positive = sum(value for value in items if value > 0)
    negative = abs(sum(value for value in items if value < 0))
    return positive / negative if negative else float("inf")


def _longest_losing_streak(values: Iterable[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _double_costs(config: BacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        starting_equity=config.starting_equity,
        risk_fraction=config.risk_fraction,
        target_r=config.target_r,
        max_holding_bars=config.max_holding_bars,
        round_trip_cost_pips=config.round_trip_cost_pips,
        cost_stress_multiplier=config.cost_stress_multiplier * 2.0,
        minimum_cost_multiple=config.minimum_cost_multiple,
    )
