import sys

from medium_trading import cli


def test_evaluate_parser_exposes_optional_strategy_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "evaluate",
            "--strategy",
            "time-series-momentum",
            "--dataset",
            "EUR/USD=data/EUR_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "time-series-momentum"
    assert args.target_r is None
    assert args.max_holding_bars is None


def test_time_series_momentum_backtest_defaults() -> None:
    assert cli._strategy_backtest_defaults("time-series-momentum") == (3.0, 240)


def test_mean_reversion_backtest_defaults() -> None:
    assert cli._strategy_backtest_defaults("mean-reversion") == (2.0, 192)


def test_forward_evaluate_parser_requires_fixed_window(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_forward_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "forward-evaluate",
            "--strategy",
            "mean-reversion",
            "--dataset",
            "EUR/USD=data/forward_2026/EUR_USD_M30.csv",
            "--trade-start",
            "2026-01-02",
            "--trade-end",
            "2026-09-18",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "mean-reversion"
    assert args.trade_start == "2026-01-02"
    assert args.trade_end == "2026-09-18"
    assert args.target_r is None
    assert args.max_holding_bars is None


def test_ml_evaluate_parser_uses_fixed_research_command(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_ml_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "ml-evaluate",
            "--dataset",
            "EUR/USD=data/EUR_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.default_cost_pips == 1.2
    assert args.risk == 0.005


def test_ml_forward_parser_uses_separate_train_and_forward_data(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_ml_forward_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "ml-forward-evaluate",
            "--train-dataset",
            "EUR/USD=data/EUR_USD_M30.csv",
            "--forward-dataset",
            "EUR/USD=data/forward_2026/EUR_USD_M30.csv",
            "--trade-start",
            "2026-01-02",
            "--trade-end",
            "2026-09-18",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.trade_start == "2026-01-02"
    assert args.trade_end == "2026-09-18"
    assert args.default_cost_pips == 1.2


def test_direct_ml_parser_uses_frozen_research_command(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_direct_ml_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "direct-ml-evaluate",
            "--dataset",
            "EUR/USD=data/EUR_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.default_cost_pips == 1.2
    assert args.risk == 0.005


def test_direct_ml_final_parser_uses_frozen_research_command(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_direct_ml_final_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "direct-ml-final-evaluate",
            "--dataset",
            "EUR/USD=data/EUR_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.default_cost_pips == 1.2
    assert args.risk == 0.005


def test_opening_range_breakout_backtest_defaults() -> None:
    assert cli._strategy_backtest_defaults("opening-range-breakout") == (1.5, 6)


def test_daily_evaluate_parser_uses_opening_range_breakout(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "opening-range-breakout",
            "--dataset",
            "USA500.IDX/USD=data/index/USA500.IDX_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "opening-range-breakout"
    assert args.default_cost == 1.0
    assert args.risk == 0.005


def test_opening_range_breakout_quality_defaults() -> None:
    assert cli._strategy_backtest_defaults(
        "opening-range-breakout-quality"
    ) == (1.5, 6)


def test_daily_evaluate_parser_accepts_orb_quality(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "opening-range-breakout-quality",
            "--dataset",
            "USATECH.IDX/USD=data/index/USATECH.IDX_USD_M30.csv",
        ],
    )

    cli.main()

    assert captured[0].strategy == "opening-range-breakout-quality"


def test_gold_london_ny_breakout_defaults() -> None:
    assert cli._strategy_backtest_defaults(
        "gold-london-ny-breakout"
    ) == (1.5, 6)


def test_daily_evaluate_parser_accepts_gold_breakout(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "gold-london-ny-breakout",
            "--dataset",
            "XAU/USD=data/gold/XAU_USD_M30.csv",
            "--cost",
            "XAU/USD=80",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "gold-london-ny-breakout"
    assert args.cost == ["XAU/USD=80"]


def test_gold_ny_momentum_continuation_defaults() -> None:
    assert cli._strategy_backtest_defaults(
        "gold-ny-momentum-continuation"
    ) == (1.5, 6)


def test_daily_evaluate_parser_accepts_gold_momentum(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "gold-ny-momentum-continuation",
            "--dataset",
            "XAU/USD=data/gold/XAU_USD_M30.csv",
            "--cost",
            "XAU/USD=80",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "gold-ny-momentum-continuation"
    assert args.cost == ["XAU/USD=80"]


def test_gold_ny_exhaustion_reversal_defaults() -> None:
    assert cli._strategy_backtest_defaults(
        "gold-ny-exhaustion-reversal"
    ) == (1.25, 4)


def test_daily_evaluate_parser_accepts_gold_exhaustion_reversal(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "gold-ny-exhaustion-reversal",
            "--dataset",
            "XAU/USD=data/gold/XAU_USD_M30.csv",
            "--cost",
            "XAU/USD=80",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "gold-ny-exhaustion-reversal"
    assert args.cost == ["XAU/USD=80"]


def test_crypto_daily_volatility_expansion_defaults() -> None:
    assert cli._strategy_backtest_defaults(
        "crypto-daily-volatility-expansion"
    ) == (1.5, 8)


def test_crypto_daily_evaluation_uses_quality_filtered_utc_sessions() -> None:
    assert cli._daily_evaluation_clock(
        "crypto-daily-volatility-expansion"
    ) == ("UTC", None)
    assert cli._daily_minimum_session_bars(
        "crypto-daily-volatility-expansion"
    ) == 40


def test_daily_evaluate_parser_accepts_crypto_volatility_expansion(
    monkeypatch,
) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "crypto-daily-volatility-expansion",
            "--dataset",
            "BTC/USD=data/crypto/BTC_USD_M30.csv",
            "--cost",
            "BTC/USD=50",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "crypto-daily-volatility-expansion"
    assert args.cost == ["BTC/USD=50"]


def test_crypto_intraday_momentum_continuation_defaults() -> None:
    assert cli._strategy_backtest_defaults(
        "crypto-intraday-momentum-continuation"
    ) == (1.5, 8)


def test_crypto_momentum_uses_quality_filtered_utc_sessions() -> None:
    assert cli._daily_evaluation_clock(
        "crypto-intraday-momentum-continuation"
    ) == ("UTC", None)
    assert cli._daily_minimum_session_bars(
        "crypto-intraday-momentum-continuation"
    ) == 40


def test_daily_evaluate_parser_accepts_crypto_momentum(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_daily_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "daily-evaluate",
            "--strategy",
            "crypto-intraday-momentum-continuation",
            "--dataset",
            "BTC/USD=data/crypto/BTC_USD_M30.csv",
            "--cost",
            "BTC/USD=50",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.strategy == "crypto-intraday-momentum-continuation"
    assert args.cost == ["BTC/USD=50"]


def test_crypto_trend_long_defaults() -> None:
    assert cli._strategy_backtest_defaults("crypto-trend-long") == (2.0, 48)


def test_btc_long_evaluate_parser_uses_fixed_baseline_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_long_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-long-evaluate",
            "--data",
            "data/crypto/BTC_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.data == "data/crypto/BTC_USD_M30.csv"
    assert args.cost_usd == 50.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_btc_long_v1_1_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_long_v1_1_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-long-v1-1-evaluate",
            "--data",
            "data/crypto/BTC_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.cost_usd == 50.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_btc_long_v1_1_corrected_parser_uses_execution_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_long_v1_1_corrected_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-long-v1-1-corrected-evaluate",
            "--data",
            "data/crypto/BTC_USD_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.fee_bps_per_side == 5.5
    assert args.slippage_bps_per_side == 2.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_download_bybit_parser_uses_btcusdt_linear_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_download_bybit", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "download-bybit",
            "--from",
            "2023-01-01",
            "--to",
            "2025-12-31",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.category == "linear"
    assert args.base_url == "https://api.bybit.com/v5/market/kline"
    assert args.output_dir == "data/bybit"


def test_corrected_btc_parser_accepts_exchange_symbol(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_long_v1_1_corrected_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-long-v1-1-corrected-evaluate",
            "--data",
            "data/bybit/BTCUSDT_M30.csv",
            "--symbol",
            "BTCUSDT",
        ],
    )

    cli.main()

    assert captured[0].symbol == "BTCUSDT"



def test_btc_noise_ml_v0_1_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_long_noise_ml_v0_1_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-long-noise-ml-v0-1-evaluate",
            "--data",
            "data/bybit/BTCUSDT_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.fee_bps_per_side == 5.5
    assert args.slippage_bps_per_side == 2.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_btc_move_ml_v0_2_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_long_move_ml_v0_2_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-long-move-ml-v0-2-evaluate",
            "--data",
            "data/bybit/BTCUSDT_M30.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.fee_bps_per_side == 5.5
    assert args.slippage_bps_per_side == 2.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_download_bybit_state_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_download_bybit_state", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "download-bybit-state",
            "--from",
            "2023-01-01",
            "--to",
            "2025-12-31",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.base_url == "https://api.bybit.com"
    assert args.output_dir == "data/bybit/state"


def test_market_state_v0_1_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(cli, "_btc_market_state_v0_1_evaluate", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-market-state-v0-1-evaluate",
            "--m30",
            "data/bybit/BTCUSDT_M30.csv",
            "--m5",
            "data/bybit/state/BTCUSDT_M5.csv",
            "--open-interest",
            "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv",
            "--account-ratio",
            "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv",
            "--funding",
            "data/bybit/state/BTCUSDT_FUNDING.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.fee_bps_per_side == 5.5
    assert args.slippage_bps_per_side == 2.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_market_structure_v0_2_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(
        cli,
        "_btc_market_structure_v0_2_evaluate",
        captured.append,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-market-structure-v0-2-evaluate",
            "--m30",
            "data/bybit/BTCUSDT_M30.csv",
            "--m5",
            "data/bybit/state/BTCUSDT_M5.csv",
            "--open-interest",
            "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv",
            "--account-ratio",
            "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv",
            "--funding",
            "data/bybit/state/BTCUSDT_FUNDING.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.fee_bps_per_side == 5.5
    assert args.slippage_bps_per_side == 2.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_market_structure_events_v0_3_parser_uses_frozen_defaults(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(
        cli,
        "_btc_market_structure_events_v0_3_evaluate",
        captured.append,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-market-structure-events-v0-3-evaluate",
            "--m30",
            "data/bybit/BTCUSDT_M30.csv",
            "--m5",
            "data/bybit/state/BTCUSDT_M5.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.symbol == "BTCUSDT"
    assert args.fee_bps_per_side == 5.5
    assert args.slippage_bps_per_side == 2.0
    assert args.starting_equity == 10_000.0
    assert args.risk == 0.005



def test_market_observer_v0_4_parser_uses_market_only_inputs(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(
        cli,
        "_btc_market_observer_v0_4_evaluate",
        captured.append,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medium-trading",
            "btc-market-observer-v0-4-evaluate",
            "--m5",
            "data/bybit/state/BTCUSDT_M5.csv",
            "--open-interest",
            "data/bybit/state/BTCUSDT_OPEN_INTEREST_30M.csv",
            "--account-ratio",
            "data/bybit/state/BTCUSDT_ACCOUNT_RATIO_30M.csv",
            "--funding",
            "data/bybit/state/BTCUSDT_FUNDING.csv",
        ],
    )

    cli.main()

    args = captured[0]
    assert args.m5.endswith("BTCUSDT_M5.csv")
    assert args.open_interest.endswith("BTCUSDT_OPEN_INTEREST_30M.csv")
    assert args.account_ratio.endswith("BTCUSDT_ACCOUNT_RATIO_30M.csv")
    assert args.funding.endswith("BTCUSDT_FUNDING.csv")
    assert not hasattr(args, "starting_equity")
    assert not hasattr(args, "risk")
