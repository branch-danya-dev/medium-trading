from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import sqrt

from medium_trading.backtest.engine import (
    aggregate_candles,
    pip_size,
    simulate_trade,
)
from medium_trading.backtest.model import BacktestConfig, BacktestTrade
from medium_trading.domain import Candle, Side, Signal

FEATURE_NAMES = (
    "return_1_atr",
    "return_3_atr",
    "return_6_atr",
    "return_12_atr",
    "return_30_atr",
    "z_score_20",
    "z_score_50",
    "efficiency_ratio_10",
    "efficiency_ratio_30",
    "atr_fraction",
    "atr_ratio_14_50",
    "range_10_atr",
    "range_30_atr",
    "body_atr",
    "candle_range_atr",
    "range_position_20",
)

STOP_ATR_MULTIPLE = 1.5
TARGET_ATR_MULTIPLE = 2.0
MAX_HOLDING_M30_BARS = 48
PREDICTION_THRESHOLD_R = 0.0
FIRST_TEST_YEAR = 2023
LAST_TEST_YEAR = 2025

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


@dataclass(frozen=True, slots=True)
class OpportunitySample:
    symbol: str
    decision_time: datetime
    features: tuple[float, ...]
    long_trade: BacktestTrade
    short_trade: BacktestTrade

    @property
    def label_end_time(self) -> datetime:
        return max(self.long_trade.exit_time, self.short_trade.exit_time)


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    trades: int
    gross_r: float
    total_cost_r: float
    net_r: float
    profit_factor: float
    win_rate: float


@dataclass(frozen=True, slots=True)
class DirectFoldEvaluation:
    test_year: int
    train_samples: int
    test_samples: int
    selected_samples: int
    no_trade: TradeMetrics
    always_long: TradeMetrics
    always_short: TradeMetrics
    model: TradeMetrics
    model_2x_costs: TradeMetrics


@dataclass(frozen=True, slots=True)
class DirectWalkForwardEvaluation:
    feature_names: tuple[str, ...]
    model_params: dict[str, object]
    prediction_threshold_r: float
    stop_atr_multiple: float
    target_atr_multiple: float
    max_holding_m30_bars: int
    folds: tuple[DirectFoldEvaluation, ...]
    combined_always_long: TradeMetrics
    combined_always_short: TradeMetrics
    combined_model: TradeMetrics
    combined_model_2x_costs: TradeMetrics
    per_symbol_model: dict[str, TradeMetrics]
    per_symbol_model_2x_costs: dict[str, TradeMetrics]


def extract_direct_opportunities(
    *,
    symbol: str,
    candles: tuple[Candle, ...],
    config: BacktestConfig,
) -> tuple[OpportunitySample, ...]:
    if len(candles) < MAX_HOLDING_M30_BARS + 2:
        return ()

    candles_4h = aggregate_candles(candles, 240)
    m30_timestamps = [candle.timestamp for candle in candles]
    effective_cost_pips = (
        config.round_trip_cost_pips * config.cost_stress_multiplier
    )

    opportunities: list[OpportunitySample] = []
    for index in range(50, len(candles_4h)):
        history = candles_4h[max(0, index - 50) : index + 1]
        current = history[-1]
        decision_time = current.timestamp + timedelta(hours=4)

        entry_index = bisect_left(m30_timestamps, decision_time)
        if entry_index >= len(candles):
            continue
        if entry_index + MAX_HOLDING_M30_BARS > len(candles):
            continue

        features = _features(history)
        if features is None:
            continue

        atr_14 = _atr(history, 14)
        if atr_14 <= 0:
            continue

        target_distance = TARGET_ATR_MULTIPLE * atr_14
        expected_move_pips = target_distance / pip_size(symbol)
        if expected_move_pips / effective_cost_pips < config.minimum_cost_multiple:
            continue

        entry = candles[entry_index].open
        stop_distance = STOP_ATR_MULTIPLE * atr_14
        long_signal = Signal(
            symbol=symbol,
            side=Side.LONG,
            entry=entry,
            stop=entry - stop_distance,
            target=entry + target_distance,
            confidence=1.0,
            strategy="direct_ml_candidate",
            reasons=("direct 4h ML candidate",),
        )
        short_signal = Signal(
            symbol=symbol,
            side=Side.SHORT,
            entry=entry,
            stop=entry + stop_distance,
            target=entry - target_distance,
            confidence=1.0,
            strategy="direct_ml_candidate",
            reasons=("direct 4h ML candidate",),
        )

        direct_config = BacktestConfig(
            starting_equity=config.starting_equity,
            risk_fraction=config.risk_fraction,
            target_r=config.target_r,
            max_holding_bars=MAX_HOLDING_M30_BARS,
            round_trip_cost_pips=config.round_trip_cost_pips,
            cost_stress_multiplier=config.cost_stress_multiplier,
            minimum_cost_multiple=config.minimum_cost_multiple,
        )
        long_trade, _ = simulate_trade(
            signal=long_signal,
            symbol=symbol,
            entry=entry,
            target=long_signal.target,
            entry_index=entry_index,
            candles_30m=candles,
            config=direct_config,
            effective_cost_pips=effective_cost_pips,
        )
        short_trade, _ = simulate_trade(
            signal=short_signal,
            symbol=symbol,
            entry=entry,
            target=short_signal.target,
            entry_index=entry_index,
            candles_30m=candles,
            config=direct_config,
            effective_cost_pips=effective_cost_pips,
        )

        opportunities.append(
            OpportunitySample(
                symbol=symbol,
                decision_time=decision_time,
                features=features,
                long_trade=long_trade,
                short_trade=short_trade,
            )
        )

    return tuple(opportunities)


def evaluate_direct_ml_opportunities(
    samples: Iterable[OpportunitySample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> DirectWalkForwardEvaluation:
    regressor = _load_regressor()
    all_samples = tuple(sorted(samples, key=lambda sample: sample.decision_time))
    if not all_samples:
        raise ValueError("direct ML evaluation requires opportunity samples")
    if last_test_year < first_test_year:
        raise ValueError("last_test_year cannot be before first_test_year")

    folds: list[DirectFoldEvaluation] = []
    combined_long: list[BacktestTrade] = []
    combined_short: list[BacktestTrade] = []
    combined_model: list[BacktestTrade] = []

    for year in range(first_test_year, last_test_year + 1):
        test_start = datetime(year, 1, 1, tzinfo=UTC)
        test_end = datetime(year + 1, 1, 1, tzinfo=UTC)

        train = tuple(
            sample for sample in all_samples if sample.label_end_time < test_start
        )
        test = tuple(
            sample
            for sample in all_samples
            if test_start <= sample.decision_time < test_end
        )
        if not test:
            continue
        if len(train) < 500:
            raise ValueError(
                f"not enough pre-{year} direct ML samples: {len(train)}; need at least 500"
            )

        long_model = regressor(**MODEL_PARAMS)
        short_model = regressor(**MODEL_PARAMS)
        train_features = [sample.features for sample in train]
        long_model.fit(
            train_features,
            [sample.long_trade.net_r for sample in train],
        )
        short_model.fit(
            train_features,
            [sample.short_trade.net_r for sample in train],
        )

        test_features = [sample.features for sample in test]
        long_predictions = long_model.predict(test_features)
        short_predictions = short_model.predict(test_features)

        selected = _select_model_trades(
            test,
            long_predictions=long_predictions,
            short_predictions=short_predictions,
        )
        always_long = _select_fixed_direction(test, Side.LONG)
        always_short = _select_fixed_direction(test, Side.SHORT)

        folds.append(
            DirectFoldEvaluation(
                test_year=year,
                train_samples=len(train),
                test_samples=len(test),
                selected_samples=len(selected),
                no_trade=_metrics(()),
                always_long=_metrics(always_long),
                always_short=_metrics(always_short),
                model=_metrics(selected),
                model_2x_costs=_metrics(selected, cost_multiplier=2.0),
            )
        )
        combined_model.extend(selected)
        combined_long.extend(always_long)
        combined_short.extend(always_short)

    if not folds:
        raise ValueError("no direct ML walk-forward folds contained test samples")

    symbols = sorted({trade.symbol for trade in combined_model})
    return DirectWalkForwardEvaluation(
        feature_names=FEATURE_NAMES,
        model_params=dict(MODEL_PARAMS),
        prediction_threshold_r=PREDICTION_THRESHOLD_R,
        stop_atr_multiple=STOP_ATR_MULTIPLE,
        target_atr_multiple=TARGET_ATR_MULTIPLE,
        max_holding_m30_bars=MAX_HOLDING_M30_BARS,
        folds=tuple(folds),
        combined_always_long=_metrics(combined_long),
        combined_always_short=_metrics(combined_short),
        combined_model=_metrics(combined_model),
        combined_model_2x_costs=_metrics(
            combined_model,
            cost_multiplier=2.0,
        ),
        per_symbol_model={
            symbol: _metrics(
                trade for trade in combined_model if trade.symbol == symbol
            )
            for symbol in symbols
        },
        per_symbol_model_2x_costs={
            symbol: _metrics(
                (trade for trade in combined_model if trade.symbol == symbol),
                cost_multiplier=2.0,
            )
            for symbol in symbols
        },
    )


def evaluation_payload(
    evaluation: DirectWalkForwardEvaluation,
) -> dict[str, object]:
    return {
        "feature_names": list(evaluation.feature_names),
        "model": {
            "type": "HistGradientBoostingRegressor",
            "params": evaluation.model_params,
            "prediction_threshold_r": evaluation.prediction_threshold_r,
        },
        "execution": {
            "stop_atr_multiple": evaluation.stop_atr_multiple,
            "target_atr_multiple": evaluation.target_atr_multiple,
            "max_holding_m30_bars": evaluation.max_holding_m30_bars,
        },
        "folds": [
            {
                "test_year": fold.test_year,
                "train_samples": fold.train_samples,
                "test_samples": fold.test_samples,
                "selected_samples": fold.selected_samples,
                "no_trade": asdict(fold.no_trade),
                "always_long": asdict(fold.always_long),
                "always_short": asdict(fold.always_short),
                "model": asdict(fold.model),
                "model_2x_costs": asdict(fold.model_2x_costs),
            }
            for fold in evaluation.folds
        ],
        "combined": {
            "always_long": asdict(evaluation.combined_always_long),
            "always_short": asdict(evaluation.combined_always_short),
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


def _select_model_trades(
    samples: tuple[OpportunitySample, ...],
    *,
    long_predictions,
    short_predictions,
) -> tuple[BacktestTrade, ...]:
    selected: list[BacktestTrade] = []
    blocked_until: dict[str, datetime] = {}

    for sample, long_prediction, short_prediction in zip(
        samples,
        long_predictions,
        short_predictions,
        strict=True,
    ):
        if sample.long_trade.entry_time < blocked_until.get(
            sample.symbol,
            datetime.min.replace(tzinfo=UTC),
        ):
            continue

        best_prediction = max(long_prediction, short_prediction)
        if best_prediction <= PREDICTION_THRESHOLD_R:
            continue

        trade = (
            sample.long_trade
            if long_prediction >= short_prediction
            else sample.short_trade
        )
        selected.append(trade)
        blocked_until[sample.symbol] = trade.exit_time

    return tuple(selected)


def _select_fixed_direction(
    samples: tuple[OpportunitySample, ...],
    side: Side,
) -> tuple[BacktestTrade, ...]:
    selected: list[BacktestTrade] = []
    blocked_until: dict[str, datetime] = {}

    for sample in samples:
        trade = sample.long_trade if side is Side.LONG else sample.short_trade
        if trade.entry_time < blocked_until.get(
            sample.symbol,
            datetime.min.replace(tzinfo=UTC),
        ):
            continue
        selected.append(trade)
        blocked_until[sample.symbol] = trade.exit_time

    return tuple(selected)


def _features(candles_4h: tuple[Candle, ...]) -> tuple[float, ...] | None:
    if len(candles_4h) < 51:
        return None

    current = candles_4h[-1]
    atr_14 = _atr(candles_4h, 14)
    atr_50 = _atr(candles_4h, 50)
    if atr_14 <= 0 or atr_50 <= 0 or current.close <= 0:
        return None

    closes = [candle.close for candle in candles_4h]
    z_20 = _z_score(closes[-21:-1], current.close)
    z_50 = _z_score(closes[-51:-1], current.close)
    if z_20 is None or z_50 is None:
        return None

    range_20 = candles_4h[-20:]
    range_low = min(candle.low for candle in range_20)
    range_high = max(candle.high for candle in range_20)
    range_width = range_high - range_low
    range_position = (
        ((current.close - range_low) / range_width) * 2.0 - 1.0
        if range_width > 0
        else 0.0
    )

    return (
        (closes[-1] - closes[-2]) / atr_14,
        (closes[-1] - closes[-4]) / atr_14,
        (closes[-1] - closes[-7]) / atr_14,
        (closes[-1] - closes[-13]) / atr_14,
        (closes[-1] - closes[-31]) / atr_14,
        z_20,
        z_50,
        _efficiency_ratio(closes[-11:]),
        _efficiency_ratio(closes[-31:]),
        atr_14 / current.close,
        atr_14 / atr_50,
        _range_in_atr(candles_4h[-10:], atr_14),
        _range_in_atr(candles_4h[-30:], atr_14),
        (current.close - current.open) / atr_14,
        (current.high - current.low) / atr_14,
        range_position,
    )


def _atr(candles: tuple[Candle, ...], period: int) -> float:
    if len(candles) < period + 1:
        raise ValueError("not enough candles for ATR")

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


def _z_score(values: list[float], current: float) -> float | None:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    stddev = sqrt(variance)
    if stddev <= 0:
        return None
    return (current - mean) / stddev


def _efficiency_ratio(closes: list[float]) -> float:
    path = sum(
        abs(closes[index] - closes[index - 1])
        for index in range(1, len(closes))
    )
    if path <= 0:
        return 0.0
    return abs(closes[-1] - closes[0]) / path


def _range_in_atr(candles: tuple[Candle, ...], atr: float) -> float:
    return (
        max(candle.high for candle in candles)
        - min(candle.low for candle in candles)
    ) / atr


def _metrics(
    trades: Iterable[BacktestTrade],
    *,
    cost_multiplier: float = 1.0,
) -> TradeMetrics:
    items = tuple(trades)
    net_values = tuple(
        trade.gross_r - cost_multiplier * trade.cost_r
        for trade in items
    )
    positive = sum(value for value in net_values if value > 0)
    negative = abs(sum(value for value in net_values if value < 0))
    profit_factor = positive / negative if negative else float("inf")

    return TradeMetrics(
        trades=len(items),
        gross_r=sum(trade.gross_r for trade in items),
        total_cost_r=sum(trade.cost_r for trade in items) * cost_multiplier,
        net_r=sum(net_values),
        profit_factor=profit_factor,
        win_rate=(
            sum(value > 0 for value in net_values) / len(items)
            if items
            else 0.0
        ),
    )


def _load_regressor():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'ML research dependencies are not installed. Run pip install -e ".[ml]".'
        ) from exc

    return HistGradientBoostingRegressor
