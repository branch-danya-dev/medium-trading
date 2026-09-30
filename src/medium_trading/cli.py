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
    evaluate_btc_long_move_filter_v02,
    evaluate_btc_long_noise_filter_v01,
    extract_btc_long_noise_samples,
)
from medium_trading.crypto_noise_filter import (
    evaluation_payload as crypto_noise_filter_payload,
)
from medium_trading.crypto_noise_filter import (
    move_evaluation_payload as crypto_move_filter_payload,
)
from medium_trading.daily_evaluation import (
    DailyStrategyEvaluation,
    evaluate_daily_strategy,
)
from medium_trading.daily_evaluation import evaluation_payload as daily_evaluation_payload
from medium_trading.data import import_dukascopy, load_candles, save_candles
from medium_trading.data.bybit_download import download_m5 as download_bybit_m5
from medium_trading.data.bybit_download import download_m30 as download_bybit_m30
from medium_trading.data.bybit_state_history import (
    download_account_ratio,
    download_funding_history,
    download_open_interest,
    load_account_ratio,
    load_funding,
    load_open_interest,
    save_account_ratio,
    save_funding,
    save_open_interest,
)
from medium_trading.data.bybit_trade_flow import (
    download_trade_flow_range,
    load_trade_flow,
)
from medium_trading.data.dukascopy_download import download_m30
from medium_trading.data.oanda import OandaHistoryClient
from medium_trading.direct_ml import (
    evaluate_direct_ml_final,
    evaluate_direct_ml_opportunities,
    extract_direct_opportunities,
)
from medium_trading.direct_ml import evaluation_payload as direct_ml_payload
from medium_trading.direct_ml import final_evaluation_payload as direct_ml_final_payload
from medium_trading.market_observer import (
    evaluate_market_observer_v04,
    extract_market_observer_samples,
)
from medium_trading.market_observer_v05 import evaluate_market_observer_v05
from medium_trading.market_observer_v06 import (
    augment_market_observer_samples_with_trade_flow,
    evaluate_market_observer_v06,
)
from medium_trading.market_observer_v07 import (
    build_confirmed_observer_samples,
    evaluate_market_observer_v07,
)
from medium_trading.market_observer_v07_audit import (
    classify_v07_label_resolution,
    evaluate_market_observer_v07_overlap_audit,
)
from medium_trading.market_observer_v07_forward import (
    FORWARD_END,
    FORWARD_START,
    evaluate_market_observer_v07_forward,
)
from medium_trading.market_state_model import (
    evaluate_market_state_v01,
    extract_market_state_samples,
)
from medium_trading.market_state_model import (
    evaluation_payload as market_state_payload,
)
from medium_trading.market_structure_events import evaluate_market_structure_events_v03
from medium_trading.market_structure_model import (
    evaluate_structural_state_v02,
    extract_structural_state_samples,
)
from medium_trading.market_structure_model import (
    evaluation_payload as market_structure_payload,
)
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


    bybit_state = subparsers.add_parser("download-bybit-state")
    bybit_state.add_argument("--symbol", default="BTCUSDT")
    bybit_state.add_argument("--from", dest="start", required=True)
    bybit_state.add_argument("--to", dest="end", required=True)
    bybit_state.add_argument("--base-url", default="https://api.bybit.com")
    bybit_state.add_argument("--output-dir", default="data/bybit/state")

    bybit_trade_flow = subparsers.add_parser("download-bybit-trade-flow")
    bybit_trade_flow.add_argument("--symbol", default="BTCUSDT")
    bybit_trade_flow.add_argument("--from", dest="start", required=True)
    bybit_trade_flow.add_argument("--to", dest="end", required=True)
    bybit_trade_flow.add_argument(
        "--base-url",
        default="https://public.bybit.com/trading",
    )
    bybit_trade_flow.add_argument("--workers", type=int, default=2)
    bybit_trade_flow.add_argument("--output-dir", default="data/bybit/flow")

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

    btc_move_ml = subparsers.add_parser("btc-long-move-ml-v0-2-evaluate")
    btc_move_ml.add_argument(
        "--data",
        required=True,
        help="Bybit BTCUSDT M30 CSV, e.g. data/bybit/BTCUSDT_M30.csv",
    )
    btc_move_ml.add_argument("--symbol", default="BTCUSDT")
    btc_move_ml.add_argument("--fee-bps-per-side", type=float, default=5.5)
    btc_move_ml.add_argument("--slippage-bps-per-side", type=float, default=2.0)
    btc_move_ml.add_argument("--starting-equity", type=float, default=10_000.0)
    btc_move_ml.add_argument("--risk", type=float, default=0.005)
    btc_move_ml.add_argument("--json", dest="json_output")


    market_state = subparsers.add_parser("btc-market-state-v0-1-evaluate")
    market_state.add_argument(
        "--m30",
        required=True,
        help="Native Bybit BTCUSDT M30 CSV",
    )
    market_state.add_argument(
        "--m5",
        required=True,
        help="Native Bybit BTCUSDT M5 CSV",
    )
    market_state.add_argument("--open-interest", required=True)
    market_state.add_argument("--account-ratio", required=True)
    market_state.add_argument("--funding", required=True)
    market_state.add_argument("--symbol", default="BTCUSDT")
    market_state.add_argument("--fee-bps-per-side", type=float, default=5.5)
    market_state.add_argument("--slippage-bps-per-side", type=float, default=2.0)
    market_state.add_argument("--starting-equity", type=float, default=10_000.0)
    market_state.add_argument("--risk", type=float, default=0.005)
    market_state.add_argument("--json", dest="json_output")

    market_structure = subparsers.add_parser(
        "btc-market-structure-v0-2-evaluate"
    )
    market_structure.add_argument("--m30", required=True)
    market_structure.add_argument("--m5", required=True)
    market_structure.add_argument("--open-interest", required=True)
    market_structure.add_argument("--account-ratio", required=True)
    market_structure.add_argument("--funding", required=True)
    market_structure.add_argument("--symbol", default="BTCUSDT")
    market_structure.add_argument("--fee-bps-per-side", type=float, default=5.5)
    market_structure.add_argument(
        "--slippage-bps-per-side",
        type=float,
        default=2.0,
    )
    market_structure.add_argument(
        "--starting-equity",
        type=float,
        default=10_000.0,
    )
    market_structure.add_argument("--risk", type=float, default=0.005)
    market_structure.add_argument("--json", dest="json_output")

    market_structure_events = subparsers.add_parser(
        "btc-market-structure-events-v0-3-evaluate"
    )
    market_structure_events.add_argument("--m30", required=True)
    market_structure_events.add_argument("--m5", required=True)
    market_structure_events.add_argument("--symbol", default="BTCUSDT")
    market_structure_events.add_argument(
        "--fee-bps-per-side",
        type=float,
        default=5.5,
    )
    market_structure_events.add_argument(
        "--slippage-bps-per-side",
        type=float,
        default=2.0,
    )
    market_structure_events.add_argument(
        "--starting-equity",
        type=float,
        default=10_000.0,
    )
    market_structure_events.add_argument("--risk", type=float, default=0.005)
    market_structure_events.add_argument("--json", dest="json_output")

    market_observer = subparsers.add_parser(
        "btc-market-observer-v0-4-evaluate"
    )
    market_observer.add_argument("--m5", required=True)
    market_observer.add_argument("--open-interest", required=True)
    market_observer.add_argument("--account-ratio", required=True)
    market_observer.add_argument("--funding", required=True)
    market_observer.add_argument("--json", dest="json_output")

    market_observer_v05 = subparsers.add_parser(
        "btc-market-observer-v0-5-evaluate"
    )
    market_observer_v05.add_argument("--m5", required=True)
    market_observer_v05.add_argument("--open-interest", required=True)
    market_observer_v05.add_argument("--account-ratio", required=True)
    market_observer_v05.add_argument("--funding", required=True)
    market_observer_v05.add_argument("--json", dest="json_output")

    market_observer_v06 = subparsers.add_parser(
        "btc-market-observer-v0-6-evaluate"
    )
    market_observer_v06.add_argument("--m5", required=True)
    market_observer_v06.add_argument("--open-interest", required=True)
    market_observer_v06.add_argument("--account-ratio", required=True)
    market_observer_v06.add_argument("--funding", required=True)
    market_observer_v06.add_argument("--trade-flow", required=True)
    market_observer_v06.add_argument("--json", dest="json_output")

    market_observer_v07 = subparsers.add_parser(
        "btc-market-observer-v0-7-evaluate"
    )
    market_observer_v07.add_argument("--m5", required=True)
    market_observer_v07.add_argument("--open-interest", required=True)
    market_observer_v07.add_argument("--account-ratio", required=True)
    market_observer_v07.add_argument("--funding", required=True)
    market_observer_v07.add_argument("--trade-flow", required=True)
    market_observer_v07.add_argument("--json", dest="json_output")

    market_observer_v07_audit = subparsers.add_parser(
        "btc-market-observer-v0-7-overlap-audit"
    )
    market_observer_v07_audit.add_argument("--m5", required=True)
    market_observer_v07_audit.add_argument("--open-interest", required=True)
    market_observer_v07_audit.add_argument("--account-ratio", required=True)
    market_observer_v07_audit.add_argument("--funding", required=True)
    market_observer_v07_audit.add_argument("--trade-flow", required=True)
    market_observer_v07_audit.add_argument("--json", dest="json_output")

    market_observer_v07_forward = subparsers.add_parser(
        "btc-market-observer-v0-7-forward-evaluate"
    )
    market_observer_v07_forward.add_argument(
        "--dev-m5",
        default="data/bybit/state/BTCUSDT_M5.csv",
    )
    market_observer_v07_forward.add_argument(
        "--dev-open-interest",
        default="data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv",
    )
    market_observer_v07_forward.add_argument(
        "--dev-account-ratio",
        default="data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv",
    )
    market_observer_v07_forward.add_argument(
        "--dev-funding",
        default="data/bybit/state/BTCUSDT_FUNDING.csv",
    )
    market_observer_v07_forward.add_argument(
        "--dev-trade-flow",
        default="data/bybit/flow/BTCUSDT_TRADE_FLOW_M5.csv",
    )
    market_observer_v07_forward.add_argument(
        "--forward-m5",
        default="data/bybit/forward2026/state/BTCUSDT_M5.csv",
    )
    market_observer_v07_forward.add_argument(
        "--forward-open-interest",
        default=(
            "data/bybit/forward2026/state/"
            "BTCUSDT_OPEN_INTEREST_30M.csv"
        ),
    )
    market_observer_v07_forward.add_argument(
        "--forward-account-ratio",
        default=(
            "data/bybit/forward2026/state/"
            "BTCUSDT_ACCOUNT_RATIO_30M.csv"
        ),
    )
    market_observer_v07_forward.add_argument(
        "--forward-funding",
        default="data/bybit/forward2026/state/BTCUSDT_FUNDING.csv",
    )
    market_observer_v07_forward.add_argument(
        "--forward-trade-flow",
        default=(
            "data/bybit/forward2026/flow/"
            "BTCUSDT_TRADE_FLOW_M5.csv"
        ),
    )
    market_observer_v07_forward.add_argument("--json", dest="json_output")

    args = parser.parse_args()
    if args.command == "download-dukascopy":
        _download_dukascopy(args)
    elif args.command == "download-bybit":
        _download_bybit(args)
    elif args.command == "download-bybit-state":
        _download_bybit_state(args)
    elif args.command == "download-bybit-trade-flow":
        _download_bybit_trade_flow(args)
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
    elif args.command == "btc-long-move-ml-v0-2-evaluate":
        _btc_long_move_ml_v0_2_evaluate(args)
    elif args.command == "btc-market-state-v0-1-evaluate":
        _btc_market_state_v0_1_evaluate(args)
    elif args.command == "btc-market-structure-v0-2-evaluate":
        _btc_market_structure_v0_2_evaluate(args)
    elif args.command == "btc-market-structure-events-v0-3-evaluate":
        _btc_market_structure_events_v0_3_evaluate(args)
    elif args.command == "btc-market-observer-v0-4-evaluate":
        _btc_market_observer_v0_4_evaluate(args)
    elif args.command == "btc-market-observer-v0-5-evaluate":
        _btc_market_observer_v0_5_evaluate(args)
    elif args.command == "btc-market-observer-v0-6-evaluate":
        _btc_market_observer_v0_6_evaluate(args)
    elif args.command == "btc-market-observer-v0-7-evaluate":
        _btc_market_observer_v0_7_evaluate(args)
    elif args.command == "btc-market-observer-v0-7-overlap-audit":
        _btc_market_observer_v0_7_overlap_audit(args)
    elif args.command == "btc-market-observer-v0-7-forward-evaluate":
        _btc_market_observer_v0_7_forward_evaluate(args)


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



def _download_bybit_state(args: argparse.Namespace) -> None:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = args.symbol.strip().upper()
    base_url = args.base_url.rstrip("/")

    print(f"downloading Bybit market-state history {symbol}: {start} -> {end}")

    def kline_progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"  M5 klines {done}/{total} requests")

    m5 = download_bybit_m5(
        symbol=symbol,
        category="linear",
        start=start,
        end=end,
        progress=kline_progress,
        base_url=f"{base_url}/v5/market/kline",
    )
    m5_path = output_dir / f"{symbol}_M5.csv"
    save_candles(m5_path, m5.candles)
    print(f"saved {len(m5.candles)} M5 candles to {m5_path}")

    def state_progress(label: str):
        def report(done: int) -> None:
            if done == 1 or done % 25 == 0:
                print(f"  {label}: {done} requests")
        return report

    oi = download_open_interest(
        symbol=symbol,
        start=start,
        end=end,
        interval="30min",
        base_url=base_url,
        progress=state_progress("open interest"),
    )
    oi_path = output_dir / f"{symbol}_OPEN_INTEREST_30M.csv"
    save_open_interest(oi_path, oi)
    print(f"saved {len(oi)} open-interest points to {oi_path}")

    ratio = download_account_ratio(
        symbol=symbol,
        start=start,
        end=end,
        period="30min",
        base_url=base_url,
        progress=state_progress("account ratio"),
    )
    ratio_path = output_dir / f"{symbol}_ACCOUNT_RATIO_30M.csv"
    save_account_ratio(ratio_path, ratio)
    print(f"saved {len(ratio)} account-ratio points to {ratio_path}")

    funding = download_funding_history(
        symbol=symbol,
        start=start,
        end=end,
        base_url=base_url,
        progress=state_progress("funding"),
    )
    funding_path = output_dir / f"{symbol}_FUNDING.csv"
    save_funding(funding_path, funding)
    print(f"saved {len(funding)} funding points to {funding_path}")


def _download_bybit_trade_flow(args: argparse.Namespace) -> None:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output_dir)

    print(
        f"downloading Bybit public trade flow {args.symbol}: "
        f"{start} -> {end}"
    )

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"  {done}/{total} archive days aggregated")

    points = download_trade_flow_range(
        symbol=args.symbol,
        start=start,
        end=end,
        output_dir=output_dir,
        base_url=args.base_url,
        workers=args.workers,
        progress=progress,
    )
    output = output_dir / f"{args.symbol.upper()}_TRADE_FLOW_M5.csv"
    print(f"saved {len(points)} M5 trade-flow rows to {output}")


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



def _btc_long_move_ml_v0_2_evaluate(args: argparse.Namespace) -> None:
    candles = load_candles(Path(args.data))
    samples = extract_btc_long_noise_samples(
        candles=candles,
        symbol=args.symbol,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    evaluation = evaluate_btc_long_move_filter_v02(samples)
    payload = crypto_move_filter_payload(evaluation)
    combined = payload["combined"]
    classification = combined["classification"]
    economic = combined["economic_gate"]
    model = combined["ml_filter"]

    print("BTCUSDT LONG real-move ML filter v0.2")
    print(
        f"economic trades={economic['trades']} grossR={economic['gross_r']:.2f} "
        f"netR={economic['net_r']:.2f} PF={economic['profit_factor']:.2f}"
    )
    print(
        f"ml trades={model['trades']} grossR={model['gross_r']:.2f} "
        f"netR={model['net_r']:.2f} PF={model['profit_factor']:.2f}"
    )
    print(
        f"move precision={classification['precision']:.1%} "
        f"base={classification['base_move_rate']:.1%} "
        f"lift={classification['precision_lift']:.2f}x "
        f"recall={classification['recall']:.1%} "
        f"selected={classification['predicted_move']}/"
        f"{classification['candidates']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote BTC LONG real-move ML v0.2 report to {output}")



def _btc_market_state_v0_1_evaluate(args: argparse.Namespace) -> None:
    candles_30m = load_candles(Path(args.m30))
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))

    samples = extract_market_state_samples(
        candles_30m=candles_30m,
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
        symbol=args.symbol,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    evaluation = evaluate_market_state_v01(samples)
    payload = market_state_payload(evaluation)
    combined = payload["combined"]
    classification = combined["classification"]
    regression = combined["regression"]

    print("BTCUSDT Market State Model v0.1")
    print(
        f"samples={classification['samples']} "
        f"reversal_base={classification['base_reversal_rate']:.1%} "
        f"precision={classification['precision']:.1%} "
        f"recall={classification['recall']:.1%} "
        f"lift={classification['precision_lift']:.2f}x "
        f"AUC={classification['roc_auc']:.3f}"
    )
    print(
        f"MFE2h MAE={regression['mfe_mae']:.3f}R "
        f"naive={regression['mfe_naive_mae']:.3f}R | "
        f"MAE2h MAE={regression['mae_mae']:.3f}R "
        f"naive={regression['mae_naive_mae']:.3f}R"
    )
    print(f"state distribution={combined['state_distribution']}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market State Model v0.1 report to {output}")


def _btc_market_structure_v0_2_evaluate(args: argparse.Namespace) -> None:
    candles_30m = load_candles(Path(args.m30))
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))

    samples = extract_structural_state_samples(
        candles_30m=candles_30m,
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
        symbol=args.symbol,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    evaluation = evaluate_structural_state_v02(samples)
    payload = market_structure_payload(evaluation)
    combined = payload["combined"]
    classification = combined["classification"]

    print("BTCUSDT Market Structure Model v0.2")
    print(
        f"samples={combined['test_samples']} "
        f"clear={combined['clear_samples']} "
        f"excluded={combined['excluded_samples']} "
        f"reversal_base={classification['base_reversal_rate']:.1%}"
    )
    print(
        f"precision={classification['precision']:.1%} "
        f"recall={classification['recall']:.1%} "
        f"lift={classification['precision_lift']:.2f}x "
        f"AUC={classification['roc_auc']:.3f} "
        f"Brier={classification['brier_score']:.3f}"
    )
    print(f"state distribution={combined['state_distribution']}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market Structure Model v0.2 report to {output}")


def _btc_market_structure_events_v0_3_evaluate(args: argparse.Namespace) -> None:
    candles_30m = load_candles(Path(args.m30))
    candles_5m = load_candles(Path(args.m5))
    payload = evaluate_market_structure_events_v03(
        candles_30m=candles_30m,
        candles_5m=candles_5m,
        symbol=args.symbol,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        starting_equity=args.starting_equity,
        risk_fraction=args.risk,
    )
    combined = payload["combined"]
    original = combined["original_m5"]
    managed = combined["managed_m5"]
    events = combined["event_diagnostics"]

    print("BTCUSDT Market Structure Event Model v0.3")
    print(
        f"trades={original['trades']} "
        f"original_net={original['net_r']:.2f}R "
        f"managed_net={managed['net_r']:.2f}R "
        f"delta={combined['delta_net_r']:+.2f}R"
    )
    print(
        f"original_PF={original['profit_factor']:.3f} "
        f"managed_PF={managed['profit_factor']:.3f} "
        f"original_DD={original['max_drawdown']:.1%} "
        f"managed_DD={managed['max_drawdown']:.1%}"
    )
    print(
        f"structural_exits={events['structural_exits']} "
        f"improved={events['structural_exit_improved']} "
        f"worsened={events['structural_exit_worsened']}"
    )
    print(f"events={events['event_counts']}")
    print(f"break outcomes={events['body_break_outcomes']}")
    print(f"sweep followthrough={events['sweep_reclaim_followthrough']}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market Structure Event Model v0.3 report to {output}")


def _btc_market_observer_v0_4_evaluate(args: argparse.Namespace) -> None:
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))

    samples = extract_market_observer_samples(
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
    )
    payload = evaluate_market_observer_v04(samples)
    combined = payload["combined"]
    catboost = combined["catboost"]["metrics"]
    forest = combined["random_forest"]["metrics"]
    cat_reversal = catboost["per_class"]["REAL_REVERSAL"]
    forest_reversal = forest["per_class"]["REAL_REVERSAL"]

    print("BTCUSDT Market Observer v0.4")
    print(
        f"samples={combined['test_samples']} "
        f"clear={combined['clear_samples']} "
        f"excluded={combined['excluded_samples']} "
        f"states={combined['state_distribution']}"
    )
    print(
        f"CatBoost macro_F1={catboost['macro_f1']:.3f} "
        f"reversal_precision={cat_reversal['precision']:.1%} "
        f"reversal_recall={cat_reversal['recall']:.1%} "
        f"reversal_lift={cat_reversal['precision_lift']:.2f}x "
        f"PR_AUC={catboost['real_reversal']['pr_auc']:.3f} "
        f"ROC_AUC={catboost['real_reversal']['roc_auc']:.3f}"
    )
    print(
        f"RandomForest macro_F1={forest['macro_f1']:.3f} "
        f"reversal_precision={forest_reversal['precision']:.1%} "
        f"reversal_recall={forest_reversal['recall']:.1%} "
        f"reversal_lift={forest_reversal['precision_lift']:.2f}x "
        f"PR_AUC={forest['real_reversal']['pr_auc']:.3f} "
        f"ROC_AUC={forest['real_reversal']['roc_auc']:.3f}"
    )
    print(f"events={combined['event_distribution']}")
    print(f"trends={combined['trend_distribution']}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market Observer v0.4 report to {output}")


def _btc_market_observer_v0_5_evaluate(args: argparse.Namespace) -> None:
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))

    samples = extract_market_observer_samples(
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
    )
    payload = evaluate_market_observer_v05(samples)
    combined = payload["combined"]
    weighted = combined["weighted_reversal"]["ranking_and_calibration"]
    unweighted = combined["unweighted_reversal_control"][
        "ranking_and_calibration"
    ]
    hierarchy = combined["hierarchical"]
    reversal = hierarchy["per_class"]["REAL_REVERSAL"]

    print("BTCUSDT Market Observer v0.5")
    print(
        f"samples={combined['test_samples']} "
        f"clear={combined['clear_samples']} "
        f"excluded={combined['excluded_samples']} "
        f"states={combined['state_distribution']}"
    )
    print(
        f"weighted reversal ROC_AUC={weighted['roc_auc']:.3f} "
        f"PR_AUC={weighted['pr_auc']:.3f} "
        f"PR_lift={weighted['pr_auc_lift_vs_base']:.2f}x "
        f"Brier={weighted['brier_score']:.3f} "
        f"ECE={weighted['expected_calibration_error']:.3f}"
    )
    print(
        f"unweighted control ROC_AUC={unweighted['roc_auc']:.3f} "
        f"PR_AUC={unweighted['pr_auc']:.3f} "
        f"PR_lift={unweighted['pr_auc_lift_vs_base']:.2f}x"
    )
    print(
        f"hierarchical macro_F1={hierarchy['macro_f1']:.3f} "
        f"reversal_precision={reversal['precision']:.1%} "
        f"reversal_recall={reversal['recall']:.1%} "
        f"reversal_lift={reversal['precision_lift']:.2f}x "
        f"macro_AUC={hierarchy['macro_roc_auc_ovr']:.3f}"
    )
    print(
        f"weighted top fractions={weighted['top_fraction_lift']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market Observer v0.5 report to {output}")


def _btc_market_observer_v0_6_evaluate(args: argparse.Namespace) -> None:
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))
    trade_flow = load_trade_flow(Path(args.trade_flow))

    base_samples = extract_market_observer_samples(
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
    )
    samples = augment_market_observer_samples_with_trade_flow(
        base_samples,
        trade_flow,
    )
    payload = evaluate_market_observer_v06(samples)
    combined = payload["combined"]
    baseline = combined["baseline_v0_5_reversal"]["ranking_and_calibration"]
    enhanced = combined["flow_enhanced_reversal"]["ranking_and_calibration"]
    delta = combined["delta"]

    print("BTCUSDT Market Observer v0.6")
    print(
        f"samples={combined['test_samples']} "
        f"clear={combined['clear_samples']} "
        f"excluded={combined['excluded_samples']} "
        f"states={combined['state_distribution']}"
    )
    print(
        f"baseline ROC_AUC={baseline['roc_auc']:.3f} "
        f"PR_AUC={baseline['pr_auc']:.3f} "
        f"PR_lift={baseline['pr_auc_lift_vs_base']:.2f}x"
    )
    print(
        f"trade-flow ROC_AUC={enhanced['roc_auc']:.3f} "
        f"PR_AUC={enhanced['pr_auc']:.3f} "
        f"PR_lift={enhanced['pr_auc_lift_vs_base']:.2f}x "
        f"Brier={enhanced['brier_score']:.3f} "
        f"ECE={enhanced['expected_calibration_error']:.3f}"
    )
    print(
        f"delta ROC_AUC={delta['roc_auc']:+.3f} "
        f"PR_AUC={delta['pr_auc']:+.3f} "
        f"PR_lift={delta['pr_auc_lift_vs_base']:+.2f}x"
    )
    print(
        "flow feature importance="
        f"{combined['flow_enhanced_reversal']['flow_feature_importance']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market Observer v0.6 report to {output}")


def _btc_market_observer_v0_7_evaluate(args: argparse.Namespace) -> None:
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))
    trade_flow = load_trade_flow(Path(args.trade_flow))

    base_samples = extract_market_observer_samples(
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
    )
    flow_samples = augment_market_observer_samples_with_trade_flow(
        base_samples,
        trade_flow,
    )
    samples = build_confirmed_observer_samples(
        samples=flow_samples,
        candles_5m=candles_5m,
        trade_flow=trade_flow,
    )
    payload = evaluate_market_observer_v07(samples)
    combined = payload["combined"]
    control = combined["delayed_control_without_confirmation"][
        "ranking_and_calibration"
    ]
    confirmed = combined["confirmed_event_observer"][
        "ranking_and_calibration"
    ]
    delta = combined["delta"]

    print("BTCUSDT Market Observer v0.7")
    print(
        f"samples={combined['test_samples']} "
        f"clear={combined['clear_samples']} "
        f"excluded={combined['excluded_samples']} "
        f"states={combined['state_distribution']}"
    )
    print(
        f"control ROC_AUC={control['roc_auc']:.3f} "
        f"PR_AUC={control['pr_auc']:.3f} "
        f"PR_lift={control['pr_auc_lift_vs_base']:.2f}x"
    )
    print(
        f"confirmed ROC_AUC={confirmed['roc_auc']:.3f} "
        f"PR_AUC={confirmed['pr_auc']:.3f} "
        f"PR_lift={confirmed['pr_auc_lift_vs_base']:.2f}x "
        f"Brier={confirmed['brier_score']:.3f} "
        f"ECE={confirmed['expected_calibration_error']:.3f}"
    )
    print(
        f"delta ROC_AUC={delta['roc_auc']:+.3f} "
        f"PR_AUC={delta['pr_auc']:+.3f} "
        f"PR_lift={delta['pr_auc_lift_vs_base']:+.2f}x"
    )
    print(
        "confirmation feature importance="
        f"{combined['confirmed_event_observer']['confirmation_feature_importance']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote Market Observer v0.7 report to {output}")


def _btc_market_observer_v0_7_overlap_audit(
    args: argparse.Namespace,
) -> None:
    candles_5m = load_candles(Path(args.m5))
    open_interest = load_open_interest(Path(args.open_interest))
    account_ratio = load_account_ratio(Path(args.account_ratio))
    funding = load_funding(Path(args.funding))
    trade_flow = load_trade_flow(Path(args.trade_flow))

    base_samples = extract_market_observer_samples(
        candles_5m=candles_5m,
        open_interest=open_interest,
        account_ratio=account_ratio,
        funding=funding,
    )
    flow_samples = augment_market_observer_samples_with_trade_flow(
        base_samples,
        trade_flow,
    )
    samples = build_confirmed_observer_samples(
        samples=flow_samples,
        candles_5m=candles_5m,
        trade_flow=trade_flow,
    )
    resolution = classify_v07_label_resolution(
        samples=samples,
        candles_5m=candles_5m,
    )
    payload = evaluate_market_observer_v07_overlap_audit(
        samples=samples,
        resolution_by_time=resolution,
    )
    combined = payload["combined"]
    resolution_summary = combined["resolution"]
    full = combined["full_v0_7"]["confirmed"]
    clean = combined["unresolved_only"]["confirmed"]

    print("BTCUSDT Market Observer v0.7 label-overlap audit")
    print(
        f"clear={combined['clear_samples']} "
        f"resolved_before_T15="
        f"{resolution_summary['resolved_before_prediction']} "
        f"({resolution_summary['resolved_fraction']:.1%}) "
        f"unresolved={resolution_summary['unresolved_at_prediction']}"
    )
    print(
        f"full ROC_AUC={full['roc_auc']:.3f} "
        f"PR_AUC={full['pr_auc']:.3f} "
        f"PR_lift={full['pr_auc_lift_vs_base']:.2f}x"
    )
    print(
        f"unresolved-only ROC_AUC={clean['roc_auc']:.3f} "
        f"PR_AUC={clean['pr_auc']:.3f} "
        f"PR_lift={clean['pr_auc_lift_vs_base']:.2f}x "
        f"base={clean['base_rate']:.1%}"
    )
    print(
        "resolved labels="
        f"{resolution_summary['resolved_final_label_distribution']}"
    )
    print(
        "unresolved labels="
        f"{resolution_summary['unresolved_final_label_distribution']}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote v0.7 label-overlap audit to {output}")


def _btc_market_observer_v0_7_forward_evaluate(
    args: argparse.Namespace,
) -> None:
    dev_candles = load_candles(Path(args.dev_m5))
    dev_open_interest = load_open_interest(Path(args.dev_open_interest))
    dev_account_ratio = load_account_ratio(Path(args.dev_account_ratio))
    dev_funding = load_funding(Path(args.dev_funding))
    dev_trade_flow = load_trade_flow(Path(args.dev_trade_flow))

    dev_base = extract_market_observer_samples(
        candles_5m=dev_candles,
        open_interest=dev_open_interest,
        account_ratio=dev_account_ratio,
        funding=dev_funding,
    )
    dev_flow = augment_market_observer_samples_with_trade_flow(
        dev_base,
        dev_trade_flow,
    )
    development = build_confirmed_observer_samples(
        samples=dev_flow,
        candles_5m=dev_candles,
        trade_flow=dev_trade_flow,
    )

    forward_candles = load_candles(Path(args.forward_m5))
    forward_open_interest = load_open_interest(
        Path(args.forward_open_interest)
    )
    forward_account_ratio = load_account_ratio(
        Path(args.forward_account_ratio)
    )
    forward_funding = load_funding(Path(args.forward_funding))
    forward_trade_flow = load_trade_flow(Path(args.forward_trade_flow))

    forward_base = extract_market_observer_samples(
        candles_5m=forward_candles,
        open_interest=forward_open_interest,
        account_ratio=forward_account_ratio,
        funding=forward_funding,
        research_start=FORWARD_START,
        research_end=FORWARD_END,
    )
    forward_flow = augment_market_observer_samples_with_trade_flow(
        forward_base,
        forward_trade_flow,
    )
    forward = build_confirmed_observer_samples(
        samples=forward_flow,
        candles_5m=forward_candles,
        trade_flow=forward_trade_flow,
    )
    resolution = classify_v07_label_resolution(
        samples=forward,
        candles_5m=forward_candles,
    )

    payload = evaluate_market_observer_v07_forward(
        development_samples=development,
        forward_samples=forward,
        resolution_by_time=resolution,
    )
    result = payload["forward"]
    full = result["full"]["confirmed"]
    clean = result["unresolved_only"]["confirmed"]
    resolution_summary = result["resolution"]

    print("BTCUSDT Market Observer v0.7 frozen 2026 forward")
    print(
        f"samples={result['samples']} "
        f"clear={result['clear_samples']} "
        f"excluded={result['excluded_samples']}"
    )
    print(
        f"resolved_before_T15="
        f"{resolution_summary['resolved_before_prediction']} "
        f"({resolution_summary['resolved_fraction']:.1%}) "
        f"unresolved={resolution_summary['unresolved_at_prediction']}"
    )
    print(
        f"full ROC_AUC={full['roc_auc']:.3f} "
        f"PR_AUC={full['pr_auc']:.3f} "
        f"PR_lift={full['pr_auc_lift_vs_base']:.2f}x"
    )
    print(
        f"unresolved-only ROC_AUC={clean['roc_auc']:.3f} "
        f"PR_AUC={clean['pr_auc']:.3f} "
        f"PR_lift={clean['pr_auc_lift_vs_base']:.2f}x "
        f"base={clean['base_rate']:.1%}"
    )
    print(
        f"frozen thresholds: control="
        f"{payload['pre_forward_split']['control_threshold']:.6f}, "
        f"confirmed="
        f"{payload['pre_forward_split']['confirmed_threshold']:.6f}"
    )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote v0.7 frozen 2026 forward report to {output}")


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
