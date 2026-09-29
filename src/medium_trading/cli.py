import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from medium_trading.backtest.engine import run_backtest
from medium_trading.backtest.model import BacktestConfig, BacktestReport
from medium_trading.backtest.validation import SymbolEvaluation, evaluate_symbol
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

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV, e.g. EUR/USD=data/EUR_USD_M30.csv",
    )
    evaluate.add_argument(
        "--cost",
        action="append",
        default=[],
        help="Optional repeatable SYMBOL=PIPS override",
    )
    evaluate.add_argument("--default-cost-pips", type=float, default=1.2)
    evaluate.add_argument("--risk", type=float, default=0.005)
    evaluate.add_argument("--target-r", type=float, default=2.0)
    evaluate.add_argument("--train-fraction", type=float, default=0.60)
    evaluate.add_argument("--validation-fraction", type=float, default=0.20)
    evaluate.add_argument("--json", dest="json_output")

    args = parser.parse_args()
    if args.command == "download-oanda":
        _download_oanda(args)
    elif args.command == "backtest":
        _backtest(args)
    elif args.command == "evaluate":
        _evaluate(args)


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
    _print_report(report)


def _evaluate(args: argparse.Namespace) -> None:
    datasets = _parse_assignments(args.dataset, "dataset")
    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }

    evaluations: list[SymbolEvaluation] = []
    for symbol, filename in datasets.items():
        cost_pips = costs.get(symbol, args.default_cost_pips)
        evaluation = evaluate_symbol(
            symbol=symbol,
            candles=load_candles(filename),
            strategy=TrendPullbackStrategy(),
            config=BacktestConfig(
                risk_fraction=args.risk,
                target_r=args.target_r,
                round_trip_cost_pips=cost_pips,
            ),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
        evaluations.append(evaluation)

    _print_evaluations(evaluations)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                [_evaluation_payload(item) for item in evaluations],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote evaluation report to {output}")


def _print_report(report: BacktestReport) -> None:
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


def _print_evaluations(evaluations: list[SymbolEvaluation]) -> None:
    header = (
        f"{'symbol':<10} {'segment':<14} {'trades':>7} {'netR':>9} "
        f"{'PF':>8} {'win':>8} {'maxDD':>8}"
    )
    print(header)
    print("-" * len(header))

    for evaluation in evaluations:
        rows = (
            ("train", evaluation.train),
            ("validation", evaluation.validation),
            ("oos", evaluation.out_of_sample),
            ("oos_2x_costs", evaluation.out_of_sample_2x_costs),
        )
        for name, report in rows:
            pf = "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.2f}"
            print(
                f"{evaluation.symbol:<10} {name:<14} {len(report.trades):>7} "
                f"{report.net_r:>9.2f} {pf:>8} {report.win_rate:>7.1%} "
                f"{report.max_drawdown:>7.1%}"
            )


def _evaluation_payload(evaluation: SymbolEvaluation) -> dict[str, object]:
    return {
        "symbol": evaluation.symbol,
        "train": _report_payload(evaluation.train),
        "validation": _report_payload(evaluation.validation),
        "out_of_sample": _report_payload(evaluation.out_of_sample),
        "out_of_sample_2x_costs": _report_payload(evaluation.out_of_sample_2x_costs),
    }


def _report_payload(report: BacktestReport) -> dict[str, object]:
    return {
        "signals": report.signal_count,
        "cost_rejections": report.cost_rejections,
        "trades": len(report.trades),
        "win_rate": report.win_rate,
        "net_r": report.net_r,
        "profit_factor": report.profit_factor,
        "max_drawdown": report.max_drawdown,
        "starting_equity": report.starting_equity,
        "final_equity": report.final_equity,
    }


def _parse_assignments(values: list[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--{label} must use SYMBOL=VALUE")
        symbol, assigned = value.split("=", 1)
        symbol = symbol.strip()
        assigned = assigned.strip()
        if not symbol or not assigned:
            raise SystemExit(f"--{label} must use non-empty SYMBOL=VALUE")
        parsed[symbol] = assigned
    return parsed


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)
