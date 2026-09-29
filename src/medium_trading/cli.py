import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from medium_trading.backtest.engine import run_backtest
from medium_trading.backtest.model import BacktestConfig
from medium_trading.data import load_candles, save_candles
from medium_trading.data.oanda import OandaHistoryClient
from medium_trading.strategy import TrendPullbackStrategy


def main() -> None:
    parser = argparse.ArgumentParser(prog="medium-trading")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-oanda")
    download.add_argument("--instrument", required=True, help="Example: EUR_USD")
    download.add_argument("--from", dest="start", required=True)
    download.add_argument("--to", dest="end", required=True)
    download.add_argument("--output", required=True)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--symbol", required=True, help="Example: EUR/USD")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--round-trip-cost-pips", type=float, default=1.2)
    backtest.add_argument("--cost-stress", type=float, default=1.0)
    backtest.add_argument("--risk", type=float, default=0.005)
    backtest.add_argument("--target-r", type=float, default=2.0)

    args = parser.parse_args()
    if args.command == "download-oanda":
        _download_oanda(args)
    elif args.command == "backtest":
        _backtest(args)


def _download_oanda(args: argparse.Namespace) -> None:
    token = os.environ.get("OANDA_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    environment = os.environ.get("OANDA_ENV", "practice")
    if not token or not account_id:
        raise SystemExit("OANDA_TOKEN and OANDA_ACCOUNT_ID must be set")

    client = OandaHistoryClient(
        account_id=account_id,
        token=token,
        environment=environment,
    )
    candles = client.fetch_m30(
        instrument=args.instrument,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    save_candles(args.output, candles)
    print(f"saved {len(candles)} M30 candles to {args.output}")


def _backtest(args: argparse.Namespace) -> None:
    candles = load_candles(Path(args.data))
    report = run_backtest(
        symbol=args.symbol,
        candles_30m=candles,
        strategy=TrendPullbackStrategy(),
        config=BacktestConfig(
            risk_fraction=args.risk,
            target_r=args.target_r,
            round_trip_cost_pips=args.round_trip_cost_pips,
            cost_stress_multiplier=args.cost_stress,
        ),
    )

    profit_factor = (
        "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.3f}"
    )
    print(f"symbol: {report.symbol}")
    print(f"signals: {report.signal_count}")
    print(f"cost rejections: {report.cost_rejections}")
    print(f"trades: {len(report.trades)}")
    print(f"win rate: {report.win_rate:.2%}")
    print(f"net R: {report.net_r:.3f}")
    print(f"profit factor: {profit_factor}")
    print(f"max drawdown: {report.max_drawdown:.2%}")
    print(f"equity: USD {report.starting_equity:.2f} -> USD {report.final_equity:.2f}")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)
