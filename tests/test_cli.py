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


def test_crypto_daily_evaluation_uses_utc_sessions() -> None:
    assert cli._daily_evaluation_clock(
        "crypto-daily-volatility-expansion"
    ) == ("UTC", (0, 0))


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
