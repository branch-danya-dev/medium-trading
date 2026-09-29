from dataclasses import dataclass
from datetime import datetime

from medium_trading.domain import Candle
from medium_trading.strategy.base import Strategy

from .engine import run_backtest
from .model import BacktestConfig, BacktestReport


@dataclass(frozen=True, slots=True)
class DataSplit:
    name: str
    candles: tuple[Candle, ...]
    trade_start: datetime | None


@dataclass(frozen=True, slots=True)
class SymbolEvaluation:
    symbol: str
    train: BacktestReport
    validation: BacktestReport
    out_of_sample: BacktestReport
    out_of_sample_2x_costs: BacktestReport


def chronological_splits(
    candles: tuple[Candle, ...],
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    warmup_bars: int = 500,
) -> tuple[DataSplit, DataSplit, DataSplit]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave an out-of-sample segment")
    if warmup_bars < 0:
        raise ValueError("warmup_bars cannot be negative")
    if len(candles) < 100:
        raise ValueError("dataset is too small for chronological validation")

    train_end = int(len(candles) * train_fraction)
    validation_end = int(len(candles) * (train_fraction + validation_fraction))

    if train_end <= 0 or validation_end <= train_end or validation_end >= len(candles):
        raise ValueError("invalid split boundaries")

    train = DataSplit(
        name="train",
        candles=candles[:train_end],
        trade_start=None,
    )

    validation_warmup_start = max(0, train_end - warmup_bars)
    validation = DataSplit(
        name="validation",
        candles=candles[validation_warmup_start:validation_end],
        trade_start=candles[train_end].timestamp,
    )

    test_warmup_start = max(0, validation_end - warmup_bars)
    out_of_sample = DataSplit(
        name="out_of_sample",
        candles=candles[test_warmup_start:],
        trade_start=candles[validation_end].timestamp,
    )

    return train, validation, out_of_sample


def evaluate_symbol(
    *,
    symbol: str,
    candles: tuple[Candle, ...],
    strategy: Strategy,
    config: BacktestConfig,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    warmup_bars: int = 500,
) -> SymbolEvaluation:
    train, validation, out_of_sample = chronological_splits(
        candles,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        warmup_bars=warmup_bars,
    )

    train_report = run_backtest(
        symbol=symbol,
        candles_30m=train.candles,
        strategy=strategy,
        config=config,
        trade_start=train.trade_start,
    )
    validation_report = run_backtest(
        symbol=symbol,
        candles_30m=validation.candles,
        strategy=strategy,
        config=config,
        trade_start=validation.trade_start,
    )
    out_of_sample_report = run_backtest(
        symbol=symbol,
        candles_30m=out_of_sample.candles,
        strategy=strategy,
        config=config,
        trade_start=out_of_sample.trade_start,
    )

    stressed_config = BacktestConfig(
        starting_equity=config.starting_equity,
        risk_fraction=config.risk_fraction,
        target_r=config.target_r,
        max_holding_bars=config.max_holding_bars,
        round_trip_cost_pips=config.round_trip_cost_pips,
        cost_stress_multiplier=config.cost_stress_multiplier * 2.0,
        minimum_cost_multiple=config.minimum_cost_multiple,
    )
    stressed_report = run_backtest(
        symbol=symbol,
        candles_30m=out_of_sample.candles,
        strategy=strategy,
        config=stressed_config,
        trade_start=out_of_sample.trade_start,
    )

    return SymbolEvaluation(
        symbol=symbol,
        train=train_report,
        validation=validation_report,
        out_of_sample=out_of_sample_report,
        out_of_sample_2x_costs=stressed_report,
    )
