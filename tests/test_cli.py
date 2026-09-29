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
