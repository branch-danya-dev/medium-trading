import argparse
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

from medium_trading.backtest.engine import pip_size, run_backtest
from medium_trading.backtest.model import BacktestConfig, BacktestReport
from medium_trading.backtest.validation import (
    ForwardEvaluation,
    SymbolEvaluation,
    evaluate_forward_symbol,
    evaluate_symbol,
)
from medium_trading.config import Settings
from medium_trading.crypto_long_baseline import (
    evaluate_btc_long_baseline,
    evaluate_btc_long_v1_1,
    evaluate_btc_long_v1_1_corrected,
)
from medium_trading.crypto_noise_filter import (
    evaluate_btc_long_noise_filter_v01,
    evaluation_payload as crypto_noise_filter_payload,
    extract_btc_long_noise_samples,
)
from medium_trading.daily_evaluation import (
    DailyStrategyEvaluation,
    evaluate_daily_strategy,
)
from medium_trading.daily_evaluation import evaluation_payload as daily_evaluation_payload
from medium_trading.data import import_dukascopy, load_candles, save_candles
from medium_trading.data.bybit_download import download_m30 as download_bybit_m30
from medium_trading.data.dukascopy_download import download_m30
from medium_trading.data.oanda import OandaHistoryClient
from medium_trading.direct_ml import (
    evaluate_direct_ml_final,
    evaluate_direct_ml_opportunities,
    extract_direct_opportunities,
)
from medium_trading.direct_ml import evaluation_payload as direct_ml_payload
from medium_trading.direct_ml import final_evaluation_payload as direct_ml_final_payload
from medium_trading.ml_filter import (
    evaluate_mean_reversion_ml_filter,
    evaluate_mean_reversion_ml_forward,
    extract_mean_reversion_samples,
)
from medium_trading.ml_filter import evaluation_payload as ml_evaluation_payload
from medium_trading.ml_filter import forward_evaluation_payload as ml_forward_payload
from medium_trading.strategy import (
    CryptoDailyVolatilityExpansionStrategy,
    CryptoIntradayMomentumContinuationStrategy,
    CryptoTrendLongStrategy,
    GoldLondonNewYorkBreakoutStrategy,
    GoldNewYorkExhaustionReversalStrategy,
    GoldNewYorkMomentumContinuationStrategy,
    MeanReversionStrategy,
    OpeningRangeBreakoutQualityStrategy,
    OpeningRangeBreakoutStrategy,
    TimeSeriesMomentumStrategy,
    TrendPullbackStrategy,
    VolatilityBreakoutStrategy,
)
from medium_trading.strategy.base import Strategy

_STRATEGY_CHOICES = (
    "trend-pullback",
    "volatility-breakout",
    "time-series-momentum",
    "mean-reversion",
    "opening-range-breakout",
    "opening-range-breakout-quality",
    "gold-london-ny-breakout",
    "gold-ny-momentum-continuation",
    "gold-ny-exhaustion-reversal",
    "crypto-daily-volatility-expansion",
    "crypto-intraday-momentum-continuation",
    "crypto-trend-long",
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="medium-trading")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dukascopy_download = subparsers.add_parser("download-dukascopy")
    dukascopy_download.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Repeatable. Defaults to the four MVP FX pairs.",
    )
    dukascopy_download.add_argument("--from", dest="start", required=True)
    dukascopy_download.add_argument("--to", dest="end", required=True)
    dukascopy_download.add_argument("--side", choices=("BID", "ASK"), default="BID")
    dukascopy_download.add_argument("--workers", type=int, default=4)
    dukascopy_download.add_argument("--output-dir", default="data")

    bybit_download = subparsers.add_parser("download-bybit")
    bybit_download.add_argument("--symbol", default="BTCUSDT")
    bybit_download.add_argument(
        "--category",
        choices=("linear", "inverse", "spot"),
        default="linear",
    )
    bybit_download.add_argument("--from", dest="start", required=True)
    bybit_download.add_argument("--to", dest="end", required=True)
    bybit_download.add_argument(
        "--base-url",
        default="https://api.bybit.com/v5/market/kline",
    )
    bybit_download.add_argument("--output-dir", default="data/bybit")

    dukascopy_import = subparsers.add_parser("import-dukascopy")
    dukascopy_import.add_argument("--symbol", required=True, help="Example: EUR/USD")
    dukascopy_import.add_argument("--input", required=True)
    dukascopy_import.add_argument("--output", required=True)

    download = subparsers.add_parser("download-oanda")
    download.add_argument("--instrument", required=True, help="Example: EUR_USD")
    download.add_argument("--from", dest="start", required=True)
    download.add_argument("--to", dest="end", required=True)
    download.add_argument("--output", required=True)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--symbol", required=True, help="Example: EUR/USD")
    backtest.add_argument("--data", required=True)
    backtest.add_argument(
        "--strategy",
        choices=_STRATEGY_CHOICES,
        default="mean-reversion",
    )
    backtest.add_argument("--round-trip-cost-pips", type=float, default=1.2)
    backtest.add_argument("--cost-stress", type=float, default=1.0)
    backtest.add_argument("--risk", type=float, default=0.005)
    backtest.add_argument("--target-r", type=float)
    backtest.add_argument("--max-holding-bars", type=int)

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
    evaluate.add_argument(
        "--strategy",
        choices=_STRATEGY_CHOICES,
        default="mean-reversion",
    )
    evaluate.add_argument("--default-cost-pips", type=float, default=1.2)
    evaluate.add_argument("--risk", type=float, default=0.005)
    evaluate.add_argument("--target-r", type=float)
    evaluate.add_argument("--max-holding-bars", type=int)
    evaluate.add_argument("--train-fraction", type=float, default=0.60)
    evaluate.add_argument("--validation-fraction", type=float, default=0.20)
    evaluate.add_argument("--json", dest="json_output")

    forward = subparsers.add_parser("forward-evaluate")
    forward.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV with warmup before --trade-start",
    )
    forward.add_argument(
        "--cost",
        action="append",
        default=[],
        help="Optional repeatable SYMBOL=PIPS override",
    )
    forward.add_argument(
        "--strategy",
        choices=_STRATEGY_CHOICES,
        default="mean-reversion",
    )
    forward.add_argument("--trade-start", required=True)
    forward.add_argument("--trade-end", required=True)
    forward.add_argument("--default-cost-pips", type=float, default=1.2)
    forward.add_argument("--risk", type=float, default=0.005)
    forward.add_argument("--target-r", type=float)
    forward.add_argument("--max-holding-bars", type=int)
    forward.add_argument("--json", dest="json_output")

    ml_evaluate = subparsers.add_parser("ml-evaluate")
    ml_evaluate.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV using frozen 2020-2025 research history",
    )
    ml_evaluate.add_argument(
        "--cost",
        action="append",
        default=[],
        help="Optional repeatable SYMBOL=PIPS override",
    )
    ml_evaluate.add_argument("--default-cost-pips", type=float, default=1.2)
    ml_evaluate.add_argument("--risk", type=float, default=0.005)
    ml_evaluate.add_argument("--json", dest="json_output")

    ml_forward = subparsers.add_parser("ml-forward-evaluate")
    ml_forward.add_argument(
        "--train-dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV for pre-forward training history",
    )
    ml_forward.add_argument(
        "--forward-dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV containing warmup, forward window and exit horizon",
    )
    ml_forward.add_argument(
        "--cost",
        action="append",
        default=[],
        help="Optional repeatable SYMBOL=PIPS override",
    )
    ml_forward.add_argument("--trade-start", required=True)
    ml_forward.add_argument("--trade-end", required=True)
    ml_forward.add_argument("--default-cost-pips", type=float, default=1.2)
    ml_forward.add_argument("--risk", type=float, default=0.005)
    ml_forward.add_argument("--json", dest="json_output")

    direct_ml = subparsers.add_parser("direct-ml-evaluate")
    direct_ml.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV using frozen 2020-2025 research history",
    )
    direct_ml.add_argument(
        "--cost",
        action="append",
        default=[],
        help="Optional repeatable SYMBOL=PIPS override",
    )
    direct_ml.add_argument("--default-cost-pips", type=float, default=1.2)
    direct_ml.add_argument("--risk", type=float, default=0.005)
    direct_ml.add_argument("--json", dest="json_output")

    direct_ml_final = subparsers.add_parser("direct-ml-final-evaluate")
    direct_ml_final.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV using the frozen direct-ML research history",
    )
    direct_ml_final.add_argument(
        "--cost",
        action="append",
        default=[],
        help="Optional repeatable SYMBOL=PIPS override",
    )
    direct_ml_final.add_argument("--default-cost-pips", type=float, default=1.2)
    direct_ml_final.add_argument("--risk", type=float, default=0.005)
    direct_ml_final.add_argument("--json", dest="json_output")

    daily_evaluate = subparsers.add_parser("daily-evaluate")
    daily_evaluate.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable SYMBOL=CSV for daily-income research",
    )
    daily_evaluate.add_argument(
        "--cost",
        action="append",
        default=[],
        help=(
            "Repeatable SYMBOL=COST; FX uses pips, index CFDs use index "
            "price points, BTC/USD and ETH/USD use whole USD price points"
        ),
    )
    daily_evaluate.add_argument(
        "--strategy",
        choices=(
            "opening-range-breakout",
            "opening-range-breakout-quality",
            "gold-london-ny-breakout",
            "gold-ny-momentum-continuation",
            "gold-ny-exhaustion-reversal",
            "crypto-daily-volatility-expansion",
            "crypto-intraday-momentum-continuation",
        ),
        default="opening-range-breakout",
    )
    daily_evaluate.add_argument("--default-cost", type=float, default=1.0)
    daily_evaluate.add_argument("--risk", type=float, default=0.005)
    daily_evaluate.add_argument("--json", dest="json_output")

    btc_long = subparsers.add_parser("btc-long-evaluate")
    btc_long.add_argument(
        "--data",
        required=True,
        help="BTC/USD M30 CSV, e.g. data/crypto/BTC_USD_M30.csv",
    )
    btc_long.add_argument("--cost-usd", type=float, default=50.0)
    btc_long.add_argument("--starting-equity", type=float, default=10_000.0)
    btc_long.add_argument("--risk", type=float, default=0.005)
    btc_long.add_argument("--json", dest="json_output")

    btc_long_v1_1 = subparsers.add_parser("btc-long-v1-1-evaluate")
    btc_long_v1_1.add_argument(
        "--data",
        required=True,
        help="BTC/USD M30 CSV, e.g. data/crypto/BTC_USD_M30.csv",
    )
    btc_long_v1_1.add_argument("--cost-usd", type=float, default=50.0)
    btc_long_v1_1.add_argument("--starting-equity", type=float, default=10_000.0)
    btc_long_v1_1.add_argument("--risk", type=float, default=0.005)
    btc_long_v1_1.add_argument("--json", dest="json_output")

    btc_long_v1_1_corrected = subparsers.add_parser(
        "btc-long-v1-1-corrected-evaluate"
    )
    btc_long_v1_1_corrected.add_argument(
        "--data",
        required=True,
        help="BTC M30 CSV, e.g. data/bybit/BTCUSDT_M30.csv",
    )
    btc_long_v1_1_corrected.add_argument("--symbol", default="BTC/USD")
    btc_long_v1_1_corrected.add_argument(
        "--fee-bps-per-side",
        type=float,
        default=5.5,
    )
    btc_long_v1_1_corrected.add_argument(
        "--slippage-bps-per-side",
        type=float,
        default=2.0,
    )
    btc_long_v1_1_corrected.add_argument(
        "--starting-equity",
        type=float,
        default=10_000.0,
    )
    btc_long_v1_1_corrected.add_argument("--risk", type=float, default=0.005)
    btc_long_v1_1_corrected.add_argument("--json", dest="json_output")

    btc_noise_ml = subparsers.add_parser("btc-long-noise-ml-v0-1-evaluate")
    btc_noise_ml.add_argument(
        "--data",
        required=True,
        help="Bybit BTCUSDT M30 CSV, e.g. data/bybit/BTCUSDT_M30.csv",
    )
    btc_noise_ml.add_argument("--symbol", default="BTCUSDT")
    btc_noise_ml.add_argument("--fee-bps-per-side", type=float, default=5.5)
    btc_noise_ml.add_argument("--slippage-bps-per-side", type=float, default=2.0)
    btc_noise_ml.add_argument("--starting-equity", type=float, default=10_000.0)
    btc_noise_ml.add_argument("--risk", type=float, default=0.005)
    btc_noise_ml.add_argument("--json", dest="json_output")

    args = parser.parse_args()
    if args.command == "download-dukascopy":
        _download_dukascopy(args)
    elif args.command == "download-bybit":
        _download_bybit(args)
    elif args.command == "import-dukascopy":
        _import_dukascopy(args)
    elif args.command == "download-oanda":
        _download_oanda(args)
    elif args.command == "backtest":
        _backtest(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "forward-evaluate":
        _forward_evaluate(args)
    elif args.command == "ml-evaluate":
        _ml_evaluate(args)
    elif args.command == "ml-forward-evaluate":
        _ml_forward_evaluate(args)
    elif args.command == "direct-ml-evaluate":
        _direct_ml_evaluate(args)
    elif args.command == "direct-ml-final-evaluate":
        _direct_ml_final_evaluate(args)
    elif args.command == "daily-evaluate":
        _daily_evaluate(args)
    elif args.command == "btc-long-evaluate":
        _btc_long_evaluate(args)
    elif args.command == "btc-long-v1-1-evaluate":
        _btc_long_v1_1_evaluate(args)
    elif args.command == "btc-long-v1-1-corrected-evaluate":
        _btc_long_v1_1_corrected_evaluate(args)
    elif args.command == "btc-long-noise-ml-v0-1-evaluate":
        _btc_long_noise_ml_v0_1_evaluate(args)


def _download_dukascopy(args: argparse.Namespace) -> None:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    symbols = tuple(args.symbols or Settings().symbols)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        print(f"downloading {symbol}: {start} -> {end} ({args.side})")

        def progress(done: int, total: int) -> None:
            if done == total or done % 100 == 0:
                print(f"  {done}/{total} days")

        result = download_m30(
            symbol=symbol,
            start=start,
            end=end,
            side=args.side,
            workers=args.workers,
            progress=progress,
        )
        output = output_dir / f"{symbol.replace('/', '_')}_M30.csv"
        save_candles(output, result.candles)
        print(
            f"saved {len(result.candles)} M30 candles to {output} "
            f"({result.data_days} data days, {result.empty_days} empty days)"
        )


def _download_bybit(args: argparse.Namespace) -> None:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"downloading Bybit {args.category} {args.symbol}: "
        f"{start} -> {end} (M30)"
    )

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"  {done}/{total} requests")

    result = download_bybit_m30(
        symbol=args.symbol,
        category=args.category,
        start=start,
        end=end,
        progress=progress,
        base_url=args.base_url,
    )
    output = output_dir / f"{result.symbol}_M30.csv"
    save_candles(output, result.candles)
    print(
        f"saved {len(result.candles)} strict M30 candles to {output} "
        f"({result.request_count} requests, no missing intervals)"
    )


def _import_dukascopy(args: argparse.Namespace) -> None:
    result = import_dukascopy(args.input)
    if not result.candles:
        raise SystemExit("Dukascopy file contained no usable data")

    save_candles(args.output, result.candles)
    print(
        f"imported {len(result.candles)} M30 candles from "
        f"{result.source_kind} to {args.output}"
    )
    if result.mean_spread is not None:
        spread_pips = result.mean_spread / pip_size(args.symbol)
        print(f"observed mean bid/ask spread: {spread_pips:.3f} pips")


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
    target_r, max_holding_bars = _strategy_backtest_defaults(args.strategy)
    report = run_backtest(
        symbol=args.symbol,
        candles_30m=candles,
        strategy=_strategy_from_name(args.strategy),
        config=BacktestConfig(
            risk_fraction=args.risk,
            target_r=args.target_r if args.target_r is not None else target_r,
            max_holding_bars=(
                args.max_holding_bars
                if args.max_holding_bars is not None
                else max_holding_bars
            ),
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

    target_r, max_holding_bars = _strategy_backtest_defaults(args.strategy)
    evaluations: list[SymbolEvaluation] = []
    for symbol, filename in datasets.items():
        cost_pips = costs.get(symbol, args.default_cost_pips)
        evaluation = evaluate_symbol(
            symbol=symbol,
            candles=load_candles(filename),
            strategy=_strategy_from_name(args.strategy),
            config=BacktestConfig(
                risk_fraction=args.risk,
                target_r=args.target_r if args.target_r is not None else target_r,
                max_holding_bars=(
                    args.max_holding_bars
                    if args.max_holding_bars is not None
                    else max_holding_bars
                ),
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


def _forward_evaluate(args: argparse.Namespace) -> None:
    datasets = _parse_assignments(args.dataset, "dataset")
    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }
    trade_start = _parse_date(args.trade_start)
    trade_end = _parse_date(args.trade_end)
    target_r, max_holding_bars = _strategy_backtest_defaults(args.strategy)

    evaluations: list[ForwardEvaluation] = []
    for symbol, filename in datasets.items():
        cost_pips = costs.get(symbol, args.default_cost_pips)
        evaluation = evaluate_forward_symbol(
            symbol=symbol,
            candles=load_candles(filename),
            strategy=_strategy_from_name(args.strategy),
            config=BacktestConfig(
                risk_fraction=args.risk,
                target_r=args.target_r if args.target_r is not None else target_r,
                max_holding_bars=(
                    args.max_holding_bars
                    if args.max_holding_bars is not None
                    else max_holding_bars
                ),
                round_trip_cost_pips=cost_pips,
            ),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        evaluations.append(evaluation)

    _print_forward_evaluations(evaluations)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                [_forward_evaluation_payload(item) for item in evaluations],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote forward evaluation report to {output}")


def _ml_evaluate(args: argparse.Namespace) -> None:
    datasets = _parse_assignments(args.dataset, "dataset")
    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }

    _, max_holding_bars = _strategy_backtest_defaults("mean-reversion")
    samples = []
    for symbol, filename in datasets.items():
        config = BacktestConfig(
            risk_fraction=args.risk,
            target_r=2.0,
            max_holding_bars=max_holding_bars,
            round_trip_cost_pips=costs.get(symbol, args.default_cost_pips),
        )
        samples.extend(
            extract_mean_reversion_samples(
                symbol=symbol,
                candles=load_candles(filename),
                config=config,
            )
        )

    evaluation = evaluate_mean_reversion_ml_filter(samples)
    _print_ml_evaluation(evaluation)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                ml_evaluation_payload(evaluation),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote ML evaluation report to {output}")


def _ml_forward_evaluate(args: argparse.Namespace) -> None:
    train_datasets = _parse_assignments(args.train_dataset, "train-dataset")
    forward_datasets = _parse_assignments(args.forward_dataset, "forward-dataset")
    if set(train_datasets) != set(forward_datasets):
        raise SystemExit(
            "--train-dataset and --forward-dataset must contain the same symbols"
        )

    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }
    trade_start = _parse_date(args.trade_start)
    trade_end = _parse_date(args.trade_end)
    _, max_holding_bars = _strategy_backtest_defaults("mean-reversion")

    training_samples = []
    forward_samples = []
    for symbol in train_datasets:
        config = BacktestConfig(
            risk_fraction=args.risk,
            target_r=2.0,
            max_holding_bars=max_holding_bars,
            round_trip_cost_pips=costs.get(symbol, args.default_cost_pips),
        )
        training_samples.extend(
            extract_mean_reversion_samples(
                symbol=symbol,
                candles=load_candles(train_datasets[symbol]),
                config=config,
            )
        )
        forward_samples.extend(
            extract_mean_reversion_samples(
                symbol=symbol,
                candles=load_candles(forward_datasets[symbol]),
                config=config,
                trade_start=trade_start,
                trade_end=trade_end,
            )
        )

    evaluation = evaluate_mean_reversion_ml_forward(
        training_samples,
        forward_samples,
        trade_start=trade_start,
        trade_end=trade_end,
    )
    _print_ml_forward_evaluation(evaluation)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                ml_forward_payload(evaluation),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote ML forward evaluation report to {output}")


def _direct_ml_evaluate(args: argparse.Namespace) -> None:
    datasets = _parse_assignments(args.dataset, "dataset")
    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }

    opportunities = []
    for symbol, filename in datasets.items():
        config = BacktestConfig(
            risk_fraction=args.risk,
            target_r=2.0,
            max_holding_bars=48,
            round_trip_cost_pips=costs.get(symbol, args.default_cost_pips),
        )
        opportunities.extend(
            extract_direct_opportunities(
                symbol=symbol,
                candles=load_candles(filename),
                config=config,
            )
        )

    evaluation = evaluate_direct_ml_opportunities(opportunities)
    _print_direct_ml_evaluation(evaluation)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                direct_ml_payload(evaluation),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote direct ML evaluation report to {output}")


def _direct_ml_final_evaluate(args: argparse.Namespace) -> None:
    datasets = _parse_assignments(args.dataset, "dataset")
    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }

    opportunities = []
    for symbol, filename in datasets.items():
        config = BacktestConfig(
            risk_fraction=args.risk,
            target_r=2.0,
            max_holding_bars=48,
            round_trip_cost_pips=costs.get(symbol, args.default_cost_pips),
        )
        opportunities.extend(
            extract_direct_opportunities(
                symbol=symbol,
                candles=load_candles(filename),
                config=config,
            )
        )

    evaluation = evaluate_direct_ml_final(opportunities)
    _print_direct_ml_final_evaluation(evaluation)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                direct_ml_final_payload(evaluation),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote final direct ML evaluation report to {output}")


def _daily_evaluate(args: argparse.Namespace) -> None:
    datasets = _parse_assignments(args.dataset, "dataset")
    costs = {
        symbol: float(value)
        for symbol, value in _parse_assignments(args.cost, "cost").items()
    }
    target_r, max_holding_bars = _strategy_backtest_defaults(args.strategy)
    timezone, required_session_time = _daily_evaluation_clock(args.strategy)
    minimum_session_bars = _daily_minimum_session_bars(args.strategy)

    evaluations: list[DailyStrategyEvaluation] = []
    for symbol, filename in datasets.items():
        evaluation = evaluate_daily_strategy(
            symbol=symbol,
            candles=load_candles(filename),
            strategy=_strategy_from_name(args.strategy),
            config=BacktestConfig(
                risk_fraction=args.risk,
                target_r=target_r,
                max_holding_bars=max_holding_bars,
                round_trip_cost_pips=costs.get(symbol, args.default_cost),
            ),
            timezone=timezone,
            required_session_time=required_session_time,
            minimum_session_bars=minimum_session_bars,
        )
        evaluations.append(evaluation)

    _print_daily_evaluations(evaluations)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                [daily_evaluation_payload(item) for item in evaluations],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote daily evaluation report to {output}")


def _btc_long_evaluate(args: argparse.Namespace) -> None:
    candles = load_candles(Path(args.data))
    evaluation = evaluate_btc_long_baseline(
        candles=candles,
        round_trip_cost_usd=args.cost_usd,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    summary = evaluation["summary"]
    print("BTC/USD Trend LONG v1")
    print(
        f"trades={summary['trades']} "
        f"grossR={summary['gross_r']:.2f} "
        f"netR={summary['net_r']:.2f} "
        f"gPF={summary['gross_profit_factor']:.2f} "
        f"nPF={summary['profit_factor']:.2f}"
    )
    print(
        f"equity=${summary['starting_equity_usd']:.2f} -> "
        f"${summary['final_equity_usd']:.2f} "
        f"net=${summary['net_usd']:.2f}"
    )
    print(
        f"MFE24 avg={summary['average_mfe_r_24h']:.2f}R "
        f"median={summary['median_mfe_r_24h']:.2f}R "
        f"reached1R={summary['reached_1r_24h_rate']:.1%} "
        f"reached2R={summary['reached_2r_24h_rate']:.1%}"
    )
    print(
        "stopped then reached +1R after exit: "
        f"{summary['stopped_then_reached_1r_after_exit']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote BTC LONG baseline report to {output}")

def _btc_long_v1_1_evaluate(args: argparse.Namespace) -> None:
    candles = load_candles(Path(args.data))
    evaluation = evaluate_btc_long_v1_1(
        candles=candles,
        round_trip_cost_usd=args.cost_usd,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    summary = evaluation["summary"]
    print("BTC/USD Trend LONG v1.1")
    print(
        f"trades={summary['trades']} "
        f"grossR={summary['gross_r']:.2f} "
        f"netR={summary['net_r']:.2f} "
        f"gPF={summary['gross_profit_factor']:.2f} "
        f"nPF={summary['profit_factor']:.2f}"
    )
    print(
        f"cost avg={summary['average_cost_r']:.2f}R "
        f"median={summary['median_cost_r']:.2f}R "
        f"stopATR median={summary['median_stop_distance_atr']:.2f}"
    )
    print(
        "stopped then recovered after exit: "
        f"+1R={summary['stopped_then_reached_1r_after_exit']} "
        f"+2R={summary['stopped_then_reached_2r_after_exit']}"
    )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote BTC LONG v1.1 report to {output}")


def _btc_long_v1_1_corrected_evaluate(args: argparse.Namespace) -> None:
    candles = load_candles(Path(args.data))
    evaluation = evaluate_btc_long_v1_1_corrected(
        candles=candles,
        symbol=args.symbol,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    summary = evaluation["summary"]
    print("BTC/USD Trend LONG v1.1 corrected execution")
    print(
        f"trades={summary['trades']} "
        f"grossR={summary['gross_r']:.2f} "
        f"netR={summary['net_r']:.2f} "
        f"gPF={summary['gross_profit_factor']:.2f} "
        f"nPF={summary['profit_factor']:.2f}"
    )
    print(
        f"fee={summary['total_fee_r']:.2f}R "
        f"slippage={summary['total_slippage_r']:.2f}R "
        f"cost={summary['total_cost_r']:.2f}R "
        f"avg_hold={summary['average_holding_hours']:.2f}h"
    )
    print(
        f"gap_context={summary['gap_context_rejections']} "
        f"gap_signals={summary['gap_signal_rejections']} "
        f"maxDD={summary['max_drawdown']:.1%}"
    )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote corrected BTC LONG v1.1 report to {output}")



def _btc_long_noise_ml_v0_1_evaluate(args: argparse.Namespace) -> None:
    candles = load_candles(Path(args.data))
    samples = extract_btc_long_noise_samples(
        candles=candles,
        symbol=args.symbol,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    evaluation = evaluate_btc_long_noise_filter_v01(samples)
    payload = crypto_noise_filter_payload(evaluation)
    combined = payload["combined"]
    classification = combined["classification"]
    raw = combined["raw"]
    economic = combined["economic_gate"]
    model = combined["ml_filter"]

    print("BTCUSDT LONG ML noise filter v0.1")
    print(
        f"raw trades={raw['trades']} grossR={raw['gross_r']:.2f} "
        f"netR={raw['net_r']:.2f} PF={raw['profit_factor']:.2f}"
    )
    print(
        f"economic trades={economic['trades']} grossR={economic['gross_r']:.2f} "
        f"netR={economic['net_r']:.2f} PF={economic['profit_factor']:.2f}"
    )
    print(
        f"ml trades={model['trades']} grossR={model['gross_r']:.2f} "
        f"netR={model['net_r']:.2f} PF={model['profit_factor']:.2f}"
    )
    print(
        f"clean precision={classification['precision']:.1%} "
        f"recall={classification['recall']:.1%} "
        f"selected={classification['predicted_clean']}/"
        f"{classification['candidates']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote BTC LONG ML noise-filter v0.1 report to {output}")


def _strategy_from_name(name: str) -> Strategy:
    if name == "trend-pullback":
        return TrendPullbackStrategy()
    if name == "volatility-breakout":
        return VolatilityBreakoutStrategy()
    if name == "time-series-momentum":
        return TimeSeriesMomentumStrategy()
    if name == "mean-reversion":
        return MeanReversionStrategy()
    if name == "opening-range-breakout":
        return OpeningRangeBreakoutStrategy()
    if name == "opening-range-breakout-quality":
        return OpeningRangeBreakoutQualityStrategy()
    if name == "gold-london-ny-breakout":
        return GoldLondonNewYorkBreakoutStrategy()
    if name == "gold-ny-momentum-continuation":
        return GoldNewYorkMomentumContinuationStrategy()
    if name == "gold-ny-exhaustion-reversal":
        return GoldNewYorkExhaustionReversalStrategy()
    if name == "crypto-daily-volatility-expansion":
        return CryptoDailyVolatilityExpansionStrategy()
    if name == "crypto-intraday-momentum-continuation":
        return CryptoIntradayMomentumContinuationStrategy()
    if name == "crypto-trend-long":
        return CryptoTrendLongStrategy()
    raise ValueError(f"unsupported strategy: {name}")


def _strategy_backtest_defaults(name: str) -> tuple[float, int]:
    if name == "time-series-momentum":
        return 3.0, 240
    if name == "mean-reversion":
        return 2.0, 192
    if name in {
        "opening-range-breakout",
        "opening-range-breakout-quality",
        "gold-london-ny-breakout",
        "gold-ny-momentum-continuation",
    }:
        return 1.5, 6
    if name == "gold-ny-exhaustion-reversal":
        return 1.25, 4
    if name in {
        "crypto-daily-volatility-expansion",
        "crypto-intraday-momentum-continuation",
    }:
        return 1.5, 8
    if name == "crypto-trend-long":
        return 2.0, 48
    return 2.0, 48


def _daily_evaluation_clock(
    name: str,
) -> tuple[str, tuple[int, int] | None]:
    if name in {
        "crypto-daily-volatility-expansion",
        "crypto-intraday-momentum-continuation",
    }:
        return "UTC", None
    return "America/New_York", (9, 30)


def _daily_minimum_session_bars(name: str) -> int:
    if name in {
        "crypto-daily-volatility-expansion",
        "crypto-intraday-momentum-continuation",
    }:
        return 40
    return 0


def _print_daily_evaluations(
    evaluations: list[DailyStrategyEvaluation],
) -> None:
    header = (
        f"{'symbol':<18} {'year':<8} {'days':>6} {'active':>8} {'trades':>7} "
        f"{'avgR/day':>9} {'netR':>9} {'gPF':>7} {'nPF':>7} {'2xPF':>7}"
    )
    print(header)
    print("-" * len(header))

    for evaluation in evaluations:
        stressed_by_year = {
            item.year: item.stressed_2x
            for item in evaluation.years
        }
        for item in evaluation.years:
            stressed = stressed_by_year[item.year]
            print(
                f"{evaluation.symbol:<18} {item.year:<8} "
                f"{item.ordinary.session_days:>6} "
                f"{item.ordinary.active_day_rate:>7.1%} "
                f"{item.ordinary.trades:>7} "
                f"{item.ordinary.average_net_r_per_session:>9.3f} "
                f"{item.ordinary.net_r:>9.2f} "
                f"{item.ordinary.gross_profit_factor:>7.2f} "
                f"{item.ordinary.profit_factor:>7.2f} "
                f"{stressed.profit_factor:>7.2f}"
            )

        status = "PASS" if evaluation.passes_gate else "REJECT"
        print(
            f"{evaluation.symbol:<18} {'combined':<8} "
            f"{evaluation.combined.session_days:>6} "
            f"{evaluation.combined.active_day_rate:>7.1%} "
            f"{evaluation.combined.trades:>7} "
            f"{evaluation.combined.average_net_r_per_session:>9.3f} "
            f"{evaluation.combined.net_r:>9.2f} "
            f"{evaluation.combined.gross_profit_factor:>7.2f} "
            f"{evaluation.combined.profit_factor:>7.2f} "
            f"{evaluation.combined_2x.profit_factor:>7.2f} "
            f"gate={status}"
        )


def _print_report(report: BacktestReport) -> None:
    profit_factor = (
        "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.3f}"
    )
    gross_profit_factor = (
        "inf"
        if report.gross_profit_factor == float("inf")
        else f"{report.gross_profit_factor:.3f}"
    )
    print(f"symbol: {report.symbol}")
    print(f"signals: {report.signal_count}")
    print(f"cost rejections: {report.cost_rejections}")
    print(f"invalidated before entry: {report.invalidated_before_entry}")
    print(f"trades: {len(report.trades)}")
    print(f"win rate: {report.win_rate:.2%}")
    print(f"gross R: {report.gross_r:.3f}")
    print(f"cost drag R: {report.total_cost_r:.3f}")
    print(f"net R: {report.net_r:.3f}")
    print(f"gross profit factor: {gross_profit_factor}")
    print(f"net profit factor: {profit_factor}")
    print(f"average holding: {report.average_holding_hours:.2f}h")
    print(
        "exits: "
        f"stop={report.stop_exits} target={report.target_exits} "
        f"timeout={report.timeout_exits}"
    )
    print(f"max drawdown: {report.max_drawdown:.2%}")
    print(f"equity: USD {report.starting_equity:.2f} -> USD {report.final_equity:.2f}")


def _print_evaluations(evaluations: list[SymbolEvaluation]) -> None:
    header = (
        f"{'symbol':<10} {'segment':<14} {'trades':>7} {'grossR':>9} "
        f"{'costR':>8} {'netR':>9} {'gPF':>7} {'nPF':>7} {'maxDD':>8}"
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
            net_pf = (
                "inf"
                if report.profit_factor == float("inf")
                else f"{report.profit_factor:.2f}"
            )
            gross_pf = (
                "inf"
                if report.gross_profit_factor == float("inf")
                else f"{report.gross_profit_factor:.2f}"
            )
            print(
                f"{evaluation.symbol:<10} {name:<14} {len(report.trades):>7} "
                f"{report.gross_r:>9.2f} {report.total_cost_r:>8.2f} "
                f"{report.net_r:>9.2f} {gross_pf:>7} {net_pf:>7} "
                f"{report.max_drawdown:>7.1%}"
            )


def _print_forward_evaluations(evaluations: list[ForwardEvaluation]) -> None:
    header = (
        f"{'symbol':<10} {'segment':<18} {'trades':>7} {'grossR':>9} "
        f"{'costR':>8} {'netR':>9} {'gPF':>7} {'nPF':>7} {'maxDD':>8}"
    )
    print(header)
    print("-" * len(header))

    for evaluation in evaluations:
        rows = (
            ("forward", evaluation.forward),
            ("forward_2x_costs", evaluation.forward_2x_costs),
        )
        for name, report in rows:
            net_pf = (
                "inf"
                if report.profit_factor == float("inf")
                else f"{report.profit_factor:.2f}"
            )
            gross_pf = (
                "inf"
                if report.gross_profit_factor == float("inf")
                else f"{report.gross_profit_factor:.2f}"
            )
            print(
                f"{evaluation.symbol:<10} {name:<18} {len(report.trades):>7} "
                f"{report.gross_r:>9.2f} {report.total_cost_r:>8.2f} "
                f"{report.net_r:>9.2f} {gross_pf:>7} {net_pf:>7} "
                f"{report.max_drawdown:>7.1%}"
            )


def _print_ml_evaluation(evaluation) -> None:
    header = (
        f"{'year':<6} {'train':>7} {'test':>7} {'select':>7} "
        f"{'baseR':>9} {'mlR':>9} {'ml2xR':>9} {'mlPF':>7} {'ml2xPF':>8}"
    )
    print(header)
    print("-" * len(header))

    for fold in evaluation.folds:
        print(
            f"{fold.test_year:<6} {fold.train_samples:>7} {fold.test_samples:>7} "
            f"{fold.selected_samples:>7} {fold.baseline.net_r:>9.2f} "
            f"{fold.model.net_r:>9.2f} {fold.model_2x_costs.net_r:>9.2f} "
            f"{fold.model.profit_factor:>7.2f} "
            f"{fold.model_2x_costs.profit_factor:>8.2f}"
        )

    print(
        "combined: "
        f"baseline={evaluation.combined_baseline.net_r:.2f}R, "
        f"ml={evaluation.combined_model.net_r:.2f}R, "
        f"ml_2x={evaluation.combined_model_2x_costs.net_r:.2f}R"
    )


def _print_direct_ml_evaluation(evaluation) -> None:
    header = (
        f"{'year':<6} {'train':>8} {'test':>8} {'select':>8} "
        f"{'longR':>9} {'shortR':>9} {'mlR':>9} {'ml2xR':>9} "
        f"{'mlPF':>7} {'ml2xPF':>8}"
    )
    print(header)
    print("-" * len(header))

    for fold in evaluation.folds:
        print(
            f"{fold.test_year:<6} {fold.train_samples:>8} {fold.test_samples:>8} "
            f"{fold.selected_samples:>8} {fold.always_long.net_r:>9.2f} "
            f"{fold.always_short.net_r:>9.2f} {fold.model.net_r:>9.2f} "
            f"{fold.model_2x_costs.net_r:>9.2f} "
            f"{fold.model.profit_factor:>7.2f} "
            f"{fold.model_2x_costs.profit_factor:>8.2f}"
        )

    print(
        "combined: "
        f"always_long={evaluation.combined_always_long.net_r:.2f}R, "
        f"always_short={evaluation.combined_always_short.net_r:.2f}R, "
        f"ml={evaluation.combined_model.net_r:.2f}R, "
        f"ml_2x={evaluation.combined_model_2x_costs.net_r:.2f}R"
    )


def _print_direct_ml_final_evaluation(evaluation) -> None:
    header = (
        f"{'year':<6} {'train':>8} {'test':>8} {'broad':>8} {'best':>8} "
        f"{'bestR':>9} {'best2xR':>9} {'gPF':>7} {'nPF':>7} {'2xPF':>7}"
    )
    print(header)
    print("-" * len(header))

    for fold in evaluation.folds:
        print(
            f"{fold.test_year:<6} {fold.train_samples:>8} {fold.test_samples:>8} "
            f"{fold.broad_selected_samples:>8} {fold.best_selected_samples:>8} "
            f"{fold.best_model.net_r:>9.2f} "
            f"{fold.best_model_2x_costs.net_r:>9.2f} "
            f"{fold.best_model.gross_profit_factor:>7.2f} "
            f"{fold.best_model.profit_factor:>7.2f} "
            f"{fold.best_model_2x_costs.profit_factor:>7.2f}"
        )

    status = "PASS" if evaluation.passes_kill_gate else "REJECT"
    print(
        "combined best: "
        f"trades={evaluation.combined_best_model.trades}, "
        f"grossR={evaluation.combined_best_model.gross_r:.2f}R, "
        f"netR={evaluation.combined_best_model.net_r:.2f}R, "
        f"gPF={evaluation.combined_best_model.gross_profit_factor:.2f}, "
        f"nPF={evaluation.combined_best_model.profit_factor:.2f}, "
        f"2xPF={evaluation.combined_best_model_2x_costs.profit_factor:.2f}, "
        f"positive_years={evaluation.positive_years}/3, "
        f"kill_gate={status}"
    )


def _print_ml_forward_evaluation(evaluation) -> None:
    print(f"train samples: {evaluation.train_samples}")
    print(f"forward samples: {evaluation.test_samples}")
    print(f"selected samples: {evaluation.selected_samples}")
    print(
        "baseline: "
        f"{evaluation.baseline.net_r:.2f}R, PF={evaluation.baseline.profit_factor:.2f}"
    )
    print(
        "model: "
        f"{evaluation.model.net_r:.2f}R, PF={evaluation.model.profit_factor:.2f}"
    )
    print(
        "model 2x costs: "
        f"{evaluation.model_2x_costs.net_r:.2f}R, "
        f"PF={evaluation.model_2x_costs.profit_factor:.2f}"
    )


def _forward_evaluation_payload(evaluation: ForwardEvaluation) -> dict[str, object]:
    return {
        "symbol": evaluation.symbol,
        "forward": _report_payload(evaluation.forward),
        "forward_2x_costs": _report_payload(evaluation.forward_2x_costs),
    }


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
        "invalidated_before_entry": report.invalidated_before_entry,
        "trades": len(report.trades),
        "win_rate": report.win_rate,
        "gross_r": report.gross_r,
        "total_cost_r": report.total_cost_r,
        "net_r": report.net_r,
        "gross_profit_factor": report.gross_profit_factor,
        "profit_factor": report.profit_factor,
        "average_holding_hours": report.average_holding_hours,
        "stop_exits": report.stop_exits,
        "target_exits": report.target_exits,
        "timeout_exits": report.timeout_exits,
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
