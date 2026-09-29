from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import sqrt
from typing import Iterable

from medium_trading.backtest.engine import aggregate_candles, run_backtest
from medium_trading.backtest.model import BacktestConfig, BacktestTrade
from medium_trading.domain import Candle
from medium_trading.strategy import MeanReversionStrategy

FEATURE_NAMES = (
    "side",
    "abs_z_score",
    "efficiency_ratio_30",
    "atr_fraction",
    "target_distance_atr",
    "reward_to_risk",
    "cost_r",
    "return_1_atr",
    "return_3_atr",
    "return_6_atr",
    "range_10_atr",
    "body_atr",
)

FIRST_TEST_YEAR = 2023
LAST_TEST_YEAR = 2025
PREDICTION_THRESHOLD_R = 0.0

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


@dataclass(frozen=True, slots=True)
class TradeSample:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    features: tuple[float, ...]
    gross_r: float
    cost_r: float
    net_r: float


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    trades: int
    gross_r: float
    total_cost_r: float
    net_r: float
    profit_factor: float
    win_rate: float


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    test_year: int
    train_samples: int
    test_samples: int
    selected_samples: int
    baseline: TradeMetrics
    baseline_2x_costs: TradeMetrics
    model: TradeMetrics
    model_2x_costs: TradeMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluation:
    feature_names: tuple[str, ...]
    model_params: dict[str, object]
    prediction_threshold_r: float
    folds: tuple[FoldEvaluation, ...]
    combined_baseline: TradeMetrics
    combined_baseline_2x_costs: TradeMetrics
    combined_model: TradeMetrics
    combined_model_2x_costs: TradeMetrics
    per_symbol_model: dict[str, TradeMetrics]
    per_symbol_model_2x_costs: dict[str, TradeMetrics]


def extract_mean_reversion_samples(
    *,
    symbol: str,
    candles: tuple[Candle, ...],
    config: BacktestConfig,
) -> tuple[TradeSample, ...]:
    strategy = MeanReversionStrategy()
    report = run_backtest(
        symbol=symbol,
        candles_30m=candles,
        strategy=strategy,
        config=config,
    )
    candles_4h = aggregate_candles(candles, 240)
    ends_4h = [candle.timestamp + timedelta(hours=4) for candle in candles_4h]
    safe_entry_end = candles[-1].timestamp - timedelta(
        minutes=30 * config.max_holding_bars
    )

    samples: list[TradeSample] = []
    for trade in report.trades:
        if trade.entry_time > safe_entry_end:
            continue

        four_hour_count = bisect_right(ends_4h, trade.entry_time)
        history = candles_4h[:four_hour_count]
        features = _features_for_trade(history, trade)
        if features is None:
            continue

        samples.append(
            TradeSample(
                symbol=symbol,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                features=features,
                gross_r=trade.gross_r,
                cost_r=trade.cost_r,
                net_r=trade.net_r,
            )
        )

    return tuple(samples)


def evaluate_mean_reversion_ml_filter(
    samples: Iterable[TradeSample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> WalkForwardEvaluation:
    sklearn = _load_sklearn()
    all_samples = tuple(sorted(samples, key=lambda sample: sample.entry_time))
    if not all_samples:
        raise ValueError("ML evaluation requires at least one trade sample")
    if last_test_year < first_test_year:
        raise ValueError("last_test_year cannot be before first_test_year")

    folds: list[FoldEvaluation] = []
    combined_test: list[TradeSample] = []
    combined_selected: list[TradeSample] = []

    for year in range(first_test_year, last_test_year + 1):
        test_start = datetime(year, 1, 1, tzinfo=UTC)
        test_end = datetime(year + 1, 1, 1, tzinfo=UTC)

        # Training labels must be fully known before the test window begins.
        train = tuple(
            sample for sample in all_samples if sample.exit_time < test_start
        )
        test = tuple(
            sample
            for sample in all_samples
            if test_start <= sample.entry_time < test_end
        )
        if not test:
            continue
        if len(train) < 100:
            raise ValueError(
                f"not enough pre-{year} training samples: {len(train)}; need at least 100"
            )

        model = sklearn.HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(
            [sample.features for sample in train],
            [sample.net_r for sample in train],
        )
        predictions = model.predict([sample.features for sample in test])
        selected = tuple(
            sample
            for sample, prediction in zip(test, predictions, strict=True)
            if prediction > PREDICTION_THRESHOLD_R
        )

        folds.append(
            FoldEvaluation(
                test_year=year,
                train_samples=len(train),
                test_samples=len(test),
                selected_samples=len(selected),
                baseline=_metrics(test),
                baseline_2x_costs=_metrics(test, cost_multiplier=2.0),
                model=_metrics(selected),
                model_2x_costs=_metrics(selected, cost_multiplier=2.0),
            )
        )
        combined_test.extend(test)
        combined_selected.extend(selected)

    if not folds:
        raise ValueError("no walk-forward folds contained test samples")

    symbols = sorted({sample.symbol for sample in combined_test})
    return WalkForwardEvaluation(
        feature_names=FEATURE_NAMES,
        model_params=dict(MODEL_PARAMS),
        prediction_threshold_r=PREDICTION_THRESHOLD_R,
        folds=tuple(folds),
        combined_baseline=_metrics(combined_test),
        combined_baseline_2x_costs=_metrics(combined_test, cost_multiplier=2.0),
        combined_model=_metrics(combined_selected),
        combined_model_2x_costs=_metrics(
            combined_selected,
            cost_multiplier=2.0,
        ),
        per_symbol_model={
            symbol: _metrics(
                sample for sample in combined_selected if sample.symbol == symbol
            )
            for symbol in symbols
        },
        per_symbol_model_2x_costs={
            symbol: _metrics(
                (
                    sample
                    for sample in combined_selected
                    if sample.symbol == symbol
                ),
                cost_multiplier=2.0,
            )
            for symbol in symbols
        },
    )


def evaluation_payload(evaluation: WalkForwardEvaluation) -> dict[str, object]:
    return {
        "feature_names": list(evaluation.feature_names),
        "model": {
            "type": "HistGradientBoostingRegressor",
            "params": evaluation.model_params,
            "prediction_threshold_r": evaluation.prediction_threshold_r,
        },
        "folds": [
            {
                "test_year": fold.test_year,
                "train_samples": fold.train_samples,
                "test_samples": fold.test_samples,
                "selected_samples": fold.selected_samples,
                "baseline": asdict(fold.baseline),
                "baseline_2x_costs": asdict(fold.baseline_2x_costs),
                "model": asdict(fold.model),
                "model_2x_costs": asdict(fold.model_2x_costs),
            }
            for fold in evaluation.folds
        ],
        "combined": {
            "baseline": asdict(evaluation.combined_baseline),
            "baseline_2x_costs": asdict(evaluation.combined_baseline_2x_costs),
            "model": asdict(evaluation.combined_model),
            "model_2x_costs": asdict(evaluation.combined_model_2x_costs),
        },
        "per_symbol_model": {
            symbol: asdict(metrics)
            for symbol, metrics in evaluation.per_symbol_model.items()
        },
        "per_symbol_model_2x_costs": {
            symbol: asdict(metrics)
            for symbol, metrics in evaluation.per_symbol_model_2x_costs.items()
        },
    }


def _features_for_trade(
    candles_4h: tuple[Candle, ...],
    trade: BacktestTrade,
) -> tuple[float, ...] | None:
    if len(candles_4h) < 31:
        return None

    current = candles_4h[-1]
    prior_mean_closes = [candle.close for candle in candles_4h[-21:-1]]
    mean = sum(prior_mean_closes) / len(prior_mean_closes)
    variance = sum(
        (value - mean) ** 2 for value in prior_mean_closes
    ) / len(prior_mean_closes)
    stddev = sqrt(variance)
    if stddev <= 0:
        return None

    regime_closes = [candle.close for candle in candles_4h[-31:]]
    path = sum(
        abs(regime_closes[index] - regime_closes[index - 1])
        for index in range(1, len(regime_closes))
    )
    efficiency_ratio = (
        abs(regime_closes[-1] - regime_closes[0]) / path
        if path > 0
        else 0.0
    )

    atr = _atr(candles_4h, 14)
    if atr <= 0 or current.close <= 0:
        return None

    z_score = (current.close - mean) / stddev
    target_distance = abs(mean - trade.entry)
    risk_distance = abs(trade.entry - trade.stop)
    if risk_distance <= 0:
        return None

    closes = [candle.close for candle in candles_4h]
    recent_10 = candles_4h[-10:]
    return (
        1.0 if trade.side.value == "long" else -1.0,
        abs(z_score),
        efficiency_ratio,
        atr / current.close,
        target_distance / atr,
        target_distance / risk_distance,
        trade.cost_r,
        (closes[-1] - closes[-2]) / atr,
        (closes[-1] - closes[-4]) / atr,
        (closes[-1] - closes[-7]) / atr,
        (
            max(candle.high for candle in recent_10)
            - min(candle.low for candle in recent_10)
        )
        / atr,
        (current.close - current.open) / atr,
    )


def _atr(candles: tuple[Candle, ...], period: int) -> float:
    ranges = []
    for index in range(len(candles) - period, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return sum(ranges) / period


def _metrics(
    samples: Iterable[TradeSample],
    *,
    cost_multiplier: float = 1.0,
) -> TradeMetrics:
    items = tuple(samples)
    net_values = tuple(
        sample.gross_r - cost_multiplier * sample.cost_r
        for sample in items
    )
    positive = sum(value for value in net_values if value > 0)
    negative = abs(sum(value for value in net_values if value < 0))
    profit_factor = positive / negative if negative else float("inf")

    return TradeMetrics(
        trades=len(items),
        gross_r=sum(sample.gross_r for sample in items),
        total_cost_r=sum(sample.cost_r for sample in items) * cost_multiplier,
        net_r=sum(net_values),
        profit_factor=profit_factor,
        win_rate=(
            sum(value > 0 for value in net_values) / len(items)
            if items
            else 0.0
        ),
    )


def _load_sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - exercised by installation path
        raise RuntimeError(
            'ML research dependencies are not installed. Run pip install -e ".[ml]".'
        ) from exc

    class Sklearn:
        pass

    Sklearn.HistGradientBoostingRegressor = HistGradientBoostingRegressor
    return Sklearn
