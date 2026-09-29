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
