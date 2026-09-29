from collections import Counter
from datetime import UTC, datetime
from statistics import mean, median

from medium_trading.backtest.engine import run_backtest
from medium_trading.backtest.model import BacktestConfig, BacktestReport, BacktestTrade
from medium_trading.domain import Candle
from medium_trading.strategy.crypto_trend_long import CryptoTrendLongStrategy

TRADE_START = datetime(2023, 1, 1, tzinfo=UTC)
TRADE_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
MIN_COMPLETE_DAY_BARS = 40
DIAGNOSTIC_HORIZON_BARS = 48


def evaluate_btc_long_baseline(
    *,
    candles: tuple[Candle, ...],
    round_trip_cost_usd: float = 50.0,
    starting_equity: float = 10_000.0,
    risk_fraction: float = 0.005,
) -> dict[str, object]:
    config = BacktestConfig(
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
        target_r=2.0,
        max_holding_bars=48,
        round_trip_cost_pips=round_trip_cost_usd,
        # Diagnostic baseline: do not hide micro/noise setups behind the old 8x
        # expected-move cost gate. Costs are still deducted from every trade.
        minimum_cost_multiple=0.01,
    )
    report = run_backtest(
        symbol="BTC/USD",
        candles_30m=candles,
        strategy=CryptoTrendLongStrategy(),
        config=config,
        trade_start=TRADE_START,
        trade_end=TRADE_END,
    )

    diagnostics = tuple(
        _trade_diagnostic(candles, trade)
        for trade in report.trades
    )
    complete_days = _complete_utc_days(candles)

    return {
        "symbol": "BTC/USD",
        "strategy": "crypto-trend-long",
        "trade_start": TRADE_START.isoformat(),
        "trade_end": TRADE_END.isoformat(),
        "assumptions": {
            "side": "LONG_ONLY",
            "ml": False,
            "noise_filter": False,
            "starting_equity_usd": starting_equity,
            "risk_fraction": risk_fraction,
            "round_trip_cost_usd": round_trip_cost_usd,
            "target_r": 2.0,
            "max_holding_hours": 24,
            "minimum_cost_multiple": 0.01,
        },
        "summary": _report_summary(
            report,
            diagnostics=diagnostics,
            complete_days=complete_days,
        ),
        "years": [
            _year_summary(year, report.trades, diagnostics)
            for year in (2023, 2024, 2025)
        ],
        "trades": list(diagnostics),
    }


def _report_summary(
    report: BacktestReport,
    *,
    diagnostics: tuple[dict[str, object], ...],
    complete_days: int,
) -> dict[str, object]:
    mfe_values = tuple(float(item["mfe_r_24h"]) for item in diagnostics)
    mae_values = tuple(float(item["mae_r_24h"]) for item in diagnostics)
    post_exit_mfe = tuple(
        float(item["post_exit_mfe_r_24h"])
        for item in diagnostics
    )
    net_usd = report.final_equity - report.starting_equity

    return {
        "signals": report.signal_count,
        "cost_rejections": report.cost_rejections,
        "invalidated_before_entry": report.invalidated_before_entry,
        "trades": len(report.trades),
        "gross_r": report.gross_r,
        "net_r": report.net_r,
        "total_cost_r": report.total_cost_r,
        "gross_profit_factor": report.gross_profit_factor,
        "profit_factor": report.profit_factor,
        "win_rate": report.win_rate,
        "max_drawdown": report.max_drawdown,
        "starting_equity_usd": report.starting_equity,
        "final_equity_usd": report.final_equity,
        "net_usd": net_usd,
        "complete_utc_days": complete_days,
        "average_net_usd_per_complete_day": (
            net_usd / complete_days if complete_days else 0.0
        ),
        "average_mfe_r_24h": mean(mfe_values) if mfe_values else 0.0,
        "median_mfe_r_24h": median(mfe_values) if mfe_values else 0.0,
        "average_mae_r_24h": mean(mae_values) if mae_values else 0.0,
        "median_mae_r_24h": median(mae_values) if mae_values else 0.0,
        "reached_0_5r_24h_rate": _threshold_rate(mfe_values, 0.5),
        "reached_1r_24h_rate": _threshold_rate(mfe_values, 1.0),
        "reached_2r_24h_rate": _threshold_rate(mfe_values, 2.0),
        "reached_3r_24h_rate": _threshold_rate(mfe_values, 3.0),
        "stopped_then_reached_1r_after_exit": sum(
            item["exit_reason"] in {"stop", "stop_gap"}
            and float(item["post_exit_mfe_r_24h"]) >= 1.0
            for item in diagnostics
        ),
        "exit_reasons": dict(Counter(trade.exit_reason for trade in report.trades)),
        "average_post_exit_mfe_r_24h": (
            mean(post_exit_mfe) if post_exit_mfe else 0.0
        ),
    }


def _year_summary(
    year: int,
    trades: tuple[BacktestTrade, ...],
    diagnostics: tuple[dict[str, object], ...],
) -> dict[str, object]:
    year_trades = tuple(
        trade for trade in trades if trade.entry_time.year == year
    )
    year_diagnostics = tuple(
        item
        for item in diagnostics
        if datetime.fromisoformat(str(item["entry_time"])).year == year
    )
    gross = tuple(trade.gross_r for trade in year_trades)
    net = tuple(trade.net_r for trade in year_trades)
    mfe = tuple(float(item["mfe_r_24h"]) for item in year_diagnostics)
    return {
        "year": year,
        "trades": len(year_trades),
        "gross_r": sum(gross),
        "net_r": sum(net),
        "total_cost_r": sum(trade.cost_r for trade in year_trades),
        "gross_profit_factor": _profit_factor(gross),
        "profit_factor": _profit_factor(net),
        "win_rate": (
            sum(value > 0 for value in net) / len(net)
            if net
            else 0.0
        ),
        "average_mfe_r_24h": mean(mfe) if mfe else 0.0,
        "reached_1r_24h_rate": _threshold_rate(mfe, 1.0),
        "reached_2r_24h_rate": _threshold_rate(mfe, 2.0),
    }


def _trade_diagnostic(
    candles: tuple[Candle, ...],
    trade: BacktestTrade,
) -> dict[str, object]:
    by_timestamp = {
        candle.timestamp: index
        for index, candle in enumerate(candles)
    }
    entry_index = by_timestamp[trade.entry_time]
    last_index = min(
        len(candles) - 1,
        entry_index + DIAGNOSTIC_HORIZON_BARS - 1,
    )
    horizon = candles[entry_index : last_index + 1]
    risk_distance = abs(trade.entry - trade.stop)

    highest = max(candle.high for candle in horizon)
    lowest = min(candle.low for candle in horizon)
    mfe_r = (highest - trade.entry) / risk_distance
    mae_r = (trade.entry - lowest) / risk_distance

    post_exit = tuple(
        candle
        for candle in horizon
        if candle.timestamp >= trade.exit_time
    )
    post_exit_high = (
        max(candle.high for candle in post_exit)
        if post_exit
        else trade.exit
    )
    post_exit_mfe_r = (post_exit_high - trade.entry) / risk_distance

    return {
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "entry": trade.entry,
        "stop": trade.stop,
        "exit": trade.exit,
        "gross_r": trade.gross_r,
        "cost_r": trade.cost_r,
        "net_r": trade.net_r,
        "exit_reason": trade.exit_reason,
        "mfe_r_24h": mfe_r,
        "mae_r_24h": mae_r,
        "post_exit_mfe_r_24h": post_exit_mfe_r,
        "reached_0_5r_24h": mfe_r >= 0.5,
        "reached_1r_24h": mfe_r >= 1.0,
        "reached_2r_24h": mfe_r >= 2.0,
        "reached_3r_24h": mfe_r >= 3.0,
    }


def _complete_utc_days(candles: tuple[Candle, ...]) -> int:
    counts: Counter[object] = Counter(
        candle.timestamp.date()
        for candle in candles
        if TRADE_START.date() <= candle.timestamp.date() <= TRADE_END.date()
    )
    return sum(count >= MIN_COMPLETE_DAY_BARS for count in counts.values())


def _threshold_rate(values: tuple[float, ...], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value >= threshold for value in values) / len(values)


def _profit_factor(values: tuple[float, ...]) -> float:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    return positive / negative if negative else float("inf")
