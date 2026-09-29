from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import inf
from statistics import mean

from medium_trading.backtest.engine import aggregate_candles, run_backtest
from medium_trading.backtest.model import BacktestConfig, BacktestTrade
from medium_trading.crypto_long_baseline import (
    BTC_MAX_HOLDING_MINUTES,
    BTC_REQUIRED_CONTEXT_BARS,
    BTC_REQUIRED_FUTURE_BARS,
    BTC_SLIPPAGE_BPS_PER_SIDE,
    BTC_TAKER_FEE_BPS_PER_SIDE,
    TRADE_END,
    TRADE_START,
)
from medium_trading.domain import Candle
from medium_trading.strategy.crypto_trend_long import (
    CryptoTrendLongStrategy,
    CryptoTrendLongV11Strategy,
)

FEATURE_NAMES = (
    "return_1_atr",
    "return_2_atr",
    "return_4_atr",
    "return_8_atr",
    "atr_fraction",
    "atr_ratio_14_50",
    "efficiency_ratio_8",
    "bullish_ratio_8",
    "confirmation_body_atr",
    "confirmation_range_atr",
    "pullback_body_atr",
    "pullback_range_atr",
    "volume_ratio_20",
    "stop_distance_atr",
    "ema20_distance_4h_atr",
    "ema20_slope_3_4h_atr",
)

FIRST_TEST_YEAR = 2024
LAST_TEST_YEAR = 2025
CLEAN_HORIZON_HOURS = 8
LABEL_HORIZON_HOURS = 24
CLEAN_TARGET_R = 1.0
ECONOMIC_MIN_TARGET_TO_COST = 8.0
PROBABILITY_THRESHOLD = 0.50

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


@dataclass(frozen=True, slots=True)
class NoiseSample:
    symbol: str
    entry_time: datetime
    label_end_time: datetime
    features: tuple[float, ...]
    trade: BacktestTrade
    outcome_class: str
    is_clean: bool
    expected_cost_r: float
    target_to_cost_ratio: float

    @property
    def economic_pass(self) -> bool:
        return self.target_to_cost_ratio >= ECONOMIC_MIN_TARGET_TO_COST


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    trades: int
    gross_r: float
    total_cost_r: float
    net_r: float
    gross_profit_factor: float
    profit_factor: float
    win_rate: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    candidates: int
    predicted_clean: int
    actual_clean: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class NoiseFoldEvaluation:
    test_year: int
    train_samples: int
    train_class_distribution: dict[str, int]
    test_samples: int
    economic_samples: int
    selected_samples: int
    test_class_distribution: dict[str, int]
    classification: ClassificationMetrics
    raw: TradeMetrics
    economic_gate: TradeMetrics
    economic_gate_2x_costs: TradeMetrics
    ml_filter: TradeMetrics
    ml_filter_2x_costs: TradeMetrics


@dataclass(frozen=True, slots=True)
class NoiseFilterEvaluation:
    feature_names: tuple[str, ...]
    model_params: dict[str, object]
    probability_threshold: float
    clean_horizon_hours: int
    label_horizon_hours: int
    clean_target_r: float
    economic_min_target_to_cost: float
    folds: tuple[NoiseFoldEvaluation, ...]
    combined_classification: ClassificationMetrics
    combined_class_distribution: dict[str, int]
    combined_raw: TradeMetrics
    combined_economic_gate: TradeMetrics
    combined_economic_gate_2x_costs: TradeMetrics
    combined_ml_filter: TradeMetrics
    combined_ml_filter_2x_costs: TradeMetrics


def extract_btc_long_noise_samples(
    *,
    candles: tuple[Candle, ...],
    symbol: str = "BTCUSDT",
    fee_bps_per_side: float = BTC_TAKER_FEE_BPS_PER_SIDE,
    slippage_bps_per_side: float = BTC_SLIPPAGE_BPS_PER_SIDE,
    starting_equity: float = 10_000.0,
    risk_fraction: float = 0.005,
) -> tuple[NoiseSample, ...]:
    """Extract the fixed v1.1 executed candidate stream for ML feasibility research."""
    config = BacktestConfig(
        starting_equity=starting_equity,
        risk_fraction=risk_fraction,
        target_r=2.0,
        max_holding_bars=48,
        round_trip_cost_pips=0.000001,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        minimum_cost_multiple=0.01,
        max_holding_minutes=BTC_MAX_HOLDING_MINUTES,
        required_contiguous_context_bars=BTC_REQUIRED_CONTEXT_BARS,
        required_contiguous_future_bars=BTC_REQUIRED_FUTURE_BARS,
    )
    report = run_backtest(
        symbol=symbol,
        candles_30m=candles,
        strategy=CryptoTrendLongV11Strategy(),
        config=config,
        trade_start=TRADE_START,
        trade_end=TRADE_END,
    )

    candles_4h = aggregate_candles(candles, 240)
    ends_4h = [candle.timestamp + timedelta(hours=4) for candle in candles_4h]
    by_timestamp = {
        candle.timestamp: index
        for index, candle in enumerate(candles)
    }

    samples: list[NoiseSample] = []
    for trade in report.trades:
        entry_index = by_timestamp.get(trade.entry_time)
        if entry_index is None:
            continue

        features = _features_for_entry(
            candles=candles,
            candles_4h=candles_4h,
            ends_4h=ends_4h,
            entry_index=entry_index,
            trade=trade,
        )
        if features is None:
            continue

        outcome_class = _outcome_class(
            candles=candles,
            entry_index=entry_index,
            trade=trade,
        )
        expected_cost_r = _expected_target_cost_r(
            trade=trade,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        target_to_cost_ratio = (
            2.0 / expected_cost_r
            if expected_cost_r > 0
            else inf
        )

        samples.append(
            NoiseSample(
                symbol=symbol,
                entry_time=trade.entry_time,
                label_end_time=trade.entry_time
                + timedelta(hours=LABEL_HORIZON_HOURS),
                features=features,
                trade=trade,
                outcome_class=outcome_class,
                is_clean=outcome_class == "CLEAN",
                expected_cost_r=expected_cost_r,
                target_to_cost_ratio=target_to_cost_ratio,
            )
        )

    return tuple(samples)


def evaluate_btc_long_noise_filter_v01(
    samples: Iterable[NoiseSample],
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> NoiseFilterEvaluation:
    classifier_type = _load_classifier()
    all_samples = tuple(sorted(samples, key=lambda item: item.entry_time))
    if not all_samples:
        raise ValueError("noise-filter evaluation requires samples")
    if last_test_year < first_test_year:
        raise ValueError("last_test_year cannot be before first_test_year")

    folds: list[NoiseFoldEvaluation] = []
    combined_test: list[NoiseSample] = []
    combined_economic: list[NoiseSample] = []
    combined_selected: list[NoiseSample] = []
    combined_truth: list[bool] = []
    combined_predicted: list[bool] = []

    for year in range(first_test_year, last_test_year + 1):
        test_start = datetime(year, 1, 1, tzinfo=UTC)
        test_end = datetime(year + 1, 1, 1, tzinfo=UTC)

        # The full 24-hour outcome label must be known before a sample enters training.
        train = tuple(
            sample
            for sample in all_samples
            if sample.label_end_time <= test_start and sample.economic_pass
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
                f"not enough pre-{year} economic-pass samples: "
                f"{len(train)}; need at least 100"
            )

        economic = tuple(sample for sample in test if sample.economic_pass)
        if not economic:
            raise ValueError(f"{year} has no economic-pass samples")

        labels = [sample.is_clean for sample in train]
        if len(set(labels)) < 2:
            raise ValueError(f"pre-{year} training labels contain only one class")

        model = classifier_type(**MODEL_PARAMS)
        model.fit(
            [sample.features for sample in train],
            labels,
        )
        probabilities = _clean_probabilities(
            model,
            [sample.features for sample in economic],
        )
        predicted = tuple(
            probability >= PROBABILITY_THRESHOLD
            for probability in probabilities
        )
        selected = tuple(
            sample
            for sample, keep in zip(economic, predicted, strict=True)
            if keep
        )
        truth = tuple(sample.is_clean for sample in economic)

        folds.append(
            NoiseFoldEvaluation(
                test_year=year,
                train_samples=len(train),
                train_class_distribution=_class_distribution(train),
                test_samples=len(test),
                economic_samples=len(economic),
                selected_samples=len(selected),
                test_class_distribution=_class_distribution(test),
                classification=_classification_metrics(truth, predicted),
                raw=_trade_metrics(test),
                economic_gate=_trade_metrics(economic),
                economic_gate_2x_costs=_trade_metrics(
                    economic,
                    cost_multiplier=2.0,
                ),
                ml_filter=_trade_metrics(selected),
                ml_filter_2x_costs=_trade_metrics(
                    selected,
                    cost_multiplier=2.0,
                ),
            )
        )

        combined_test.extend(test)
        combined_economic.extend(economic)
        combined_selected.extend(selected)
        combined_truth.extend(truth)
        combined_predicted.extend(predicted)

    if not folds:
        raise ValueError("no walk-forward folds contained test samples")

    return NoiseFilterEvaluation(
        feature_names=FEATURE_NAMES,
        model_params=dict(MODEL_PARAMS),
        probability_threshold=PROBABILITY_THRESHOLD,
        clean_horizon_hours=CLEAN_HORIZON_HOURS,
        label_horizon_hours=LABEL_HORIZON_HOURS,
        clean_target_r=CLEAN_TARGET_R,
        economic_min_target_to_cost=ECONOMIC_MIN_TARGET_TO_COST,
        folds=tuple(folds),
        combined_classification=_classification_metrics(
            tuple(combined_truth),
            tuple(combined_predicted),
        ),
        combined_class_distribution=_class_distribution(combined_test),
        combined_raw=_trade_metrics(combined_test),
        combined_economic_gate=_trade_metrics(combined_economic),
        combined_economic_gate_2x_costs=_trade_metrics(
            combined_economic,
            cost_multiplier=2.0,
        ),
        combined_ml_filter=_trade_metrics(combined_selected),
        combined_ml_filter_2x_costs=_trade_metrics(
            combined_selected,
            cost_multiplier=2.0,
        ),
    )


def evaluation_payload(
    evaluation: NoiseFilterEvaluation,
) -> dict[str, object]:
    return {
        "research_scope": {
            "market": "Bybit BTCUSDT linear perpetual",
            "side": "LONG_ONLY",
            "strategy": "crypto-trend-long-v1.1-corrected",
            "development_years": [2023, 2024, 2025],
            "walk_forward_test_years": [
                fold.test_year for fold in evaluation.folds
            ],
            "2026_used": False,
            "candidate_stream": (
                "fixed executed v1.1 baseline trades; filtering does not "
                "introduce replacement candidates in v0.1"
            ),
        },
        "label": {
            "positive": (
                f"+{evaluation.clean_target_r:.1f}R before stop and within "
                f"{evaluation.clean_horizon_hours}h"
            ),
            "diagnostic_classes": {
                "CLEAN": "positive label",
                "WHIPSAW": (
                    "stop touched before +1R, then +1R reached within 24h"
                ),
                "NOISE": (
                    "all remaining cases, including no +1R within 24h or "
                    "continuation too slow for the 8h clean label"
                ),
            },
            "full_label_horizon_hours": evaluation.label_horizon_hours,
        },
        "economic_gate": {
            "rule": "2R target distance / expected round-trip execution cost >= 8",
            "minimum_target_to_cost": evaluation.economic_min_target_to_cost,
            "equivalent_max_expected_cost_r": (
                2.0 / evaluation.economic_min_target_to_cost
            ),
            "purpose": "deterministic economics; not an ML feature",
        },
        "feature_names": list(evaluation.feature_names),
        "model": {
            "type": "HistGradientBoostingClassifier",
            "params": evaluation.model_params,
            "probability_threshold": evaluation.probability_threshold,
            "threshold_tuning": "none; fixed at 0.50 for v0.1 feasibility",
        },
        "folds": [
            {
                "test_year": fold.test_year,
                "train_samples": fold.train_samples,
                "train_class_distribution": fold.train_class_distribution,
                "test_samples": fold.test_samples,
                "economic_samples": fold.economic_samples,
                "selected_samples": fold.selected_samples,
                "test_class_distribution": fold.test_class_distribution,
                "classification": asdict(fold.classification),
                "raw": asdict(fold.raw),
                "economic_gate": asdict(fold.economic_gate),
                "economic_gate_2x_costs": asdict(
                    fold.economic_gate_2x_costs
                ),
                "ml_filter": asdict(fold.ml_filter),
                "ml_filter_2x_costs": asdict(fold.ml_filter_2x_costs),
            }
            for fold in evaluation.folds
        ],
        "combined": {
            "class_distribution": evaluation.combined_class_distribution,
            "classification": asdict(evaluation.combined_classification),
            "raw": asdict(evaluation.combined_raw),
            "economic_gate": asdict(evaluation.combined_economic_gate),
            "economic_gate_2x_costs": asdict(
                evaluation.combined_economic_gate_2x_costs
            ),
            "ml_filter": asdict(evaluation.combined_ml_filter),
            "ml_filter_2x_costs": asdict(
                evaluation.combined_ml_filter_2x_costs
            ),
        },
    }


def _features_for_entry(
    *,
    candles: tuple[Candle, ...],
    candles_4h: tuple[Candle, ...],
    ends_4h: list[datetime],
    entry_index: int,
    trade: BacktestTrade,
) -> tuple[float, ...] | None:
    if entry_index < 51:
        return None

    history = candles[entry_index - 51 : entry_index]
    atr14 = _atr(history, 14)
    atr50 = _atr(history, 50)
    confirmation = candles[entry_index - 1]
    pullback = candles[entry_index - 2]
    if atr14 <= 0 or atr50 <= 0 or confirmation.close <= 0:
        return None

    closes = [candle.close for candle in history]
    recent_8 = history[-8:]
    volume_reference = history[-21:-1]
    mean_volume = mean(candle.volume for candle in volume_reference)
    volume_ratio = (
        confirmation.volume / mean_volume
        if mean_volume > 0
        else 1.0
    )

    four_hour_count = bisect_right(ends_4h, trade.entry_time)
    four_hour_history = candles_4h[:four_hour_count]
    if len(four_hour_history) < 21:
        return None

    closes_4h = tuple(candle.close for candle in four_hour_history)
    ema_values = CryptoTrendLongStrategy._ema_series(closes_4h, 20)
    atr4h = _atr(four_hour_history, 14)
    if atr4h <= 0 or len(ema_values) < 4:
        return None

    risk_distance = trade.entry - trade.stop
    if risk_distance <= 0:
        return None

    return (
        (closes[-1] - closes[-2]) / atr14,
        (closes[-1] - closes[-3]) / atr14,
        (closes[-1] - closes[-5]) / atr14,
        (closes[-1] - closes[-9]) / atr14,
        atr14 / confirmation.close,
        atr14 / atr50,
        _efficiency_ratio(closes[-9:]),
        sum(candle.close > candle.open for candle in recent_8) / 8.0,
        (confirmation.close - confirmation.open) / atr14,
        (confirmation.high - confirmation.low) / atr14,
        (pullback.close - pullback.open) / atr14,
        (pullback.high - pullback.low) / atr14,
        volume_ratio,
        risk_distance / atr14,
        (four_hour_history[-1].close - ema_values[-1]) / atr4h,
        (ema_values[-1] - ema_values[-4]) / atr4h,
    )


def _outcome_class(
    *,
    candles: tuple[Candle, ...],
    entry_index: int,
    trade: BacktestTrade,
) -> str:
    risk_distance = trade.entry - trade.stop
    target = trade.entry + CLEAN_TARGET_R * risk_distance
    clean_bars = CLEAN_HORIZON_HOURS * 2
    label_bars = LABEL_HORIZON_HOURS * 2
    horizon = candles[entry_index : entry_index + label_bars]
    adverse_seen = False

    for offset, candle in enumerate(horizon):
        # Conservative same-bar ordering: adverse excursion wins ties.
        if candle.low <= trade.stop:
            adverse_seen = True
        if candle.high >= target:
            if adverse_seen:
                return "WHIPSAW"
            if offset < clean_bars:
                return "CLEAN"
            return "NOISE"

    return "NOISE"


def _expected_target_cost_r(
    *,
    trade: BacktestTrade,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> float:
    risk_distance = trade.entry - trade.stop
    if risk_distance <= 0:
        return inf

    target = trade.entry + 2.0 * risk_distance
    bps_per_side = fee_bps_per_side + slippage_bps_per_side
    expected_round_trip_price_cost = (
        (trade.entry + target) * bps_per_side / 10_000.0
    )
    return expected_round_trip_price_cost / risk_distance


def _trade_metrics(
    samples: Iterable[NoiseSample],
    *,
    cost_multiplier: float = 1.0,
) -> TradeMetrics:
    items = tuple(samples)
    gross_values = tuple(item.trade.gross_r for item in items)
    net_values = tuple(
        item.trade.gross_r - item.trade.cost_r * cost_multiplier
        for item in items
    )

    gross_positive = sum(value for value in gross_values if value > 0)
    gross_negative = abs(sum(value for value in gross_values if value < 0))
    net_positive = sum(value for value in net_values if value > 0)
    net_negative = abs(sum(value for value in net_values if value < 0))

    return TradeMetrics(
        trades=len(items),
        gross_r=sum(gross_values),
        total_cost_r=sum(item.trade.cost_r for item in items) * cost_multiplier,
        net_r=sum(net_values),
        gross_profit_factor=(
            gross_positive / gross_negative if gross_negative else inf
        ),
        profit_factor=(
            net_positive / net_negative if net_negative else inf
        ),
        win_rate=(
            sum(value > 0 for value in net_values) / len(items)
            if items
            else 0.0
        ),
    )


def _classification_metrics(
    truth: tuple[bool, ...],
    predicted: tuple[bool, ...],
) -> ClassificationMetrics:
    if len(truth) != len(predicted):
        raise ValueError("classification arrays must have equal length")

    true_positive = sum(
        actual and guess
        for actual, guess in zip(truth, predicted, strict=True)
    )
    false_positive = sum(
        not actual and guess
        for actual, guess in zip(truth, predicted, strict=True)
    )
    false_negative = sum(
        actual and not guess
        for actual, guess in zip(truth, predicted, strict=True)
    )
    true_negative = sum(
        not actual and not guess
        for actual, guess in zip(truth, predicted, strict=True)
    )

    return ClassificationMetrics(
        candidates=len(truth),
        predicted_clean=true_positive + false_positive,
        actual_clean=true_positive + false_negative,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=(
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        recall=(
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
    )


def _class_distribution(samples: Iterable[NoiseSample]) -> dict[str, int]:
    counts = Counter(sample.outcome_class for sample in samples)
    return {
        class_name: counts.get(class_name, 0)
        for class_name in ("CLEAN", "WHIPSAW", "NOISE")
    }


def _atr(candles: tuple[Candle, ...], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0

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


def _efficiency_ratio(closes: list[float]) -> float:
    path = sum(
        abs(closes[index] - closes[index - 1])
        for index in range(1, len(closes))
    )
    if path <= 0:
        return 0.0
    return abs(closes[-1] - closes[0]) / path


def _clean_probabilities(model, features: list[tuple[float, ...]]):
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    clean_index = classes.index(True)
    return probabilities[:, clean_index]


def _load_classifier():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'ML research dependencies are not installed. Run pip install -e ".[ml]".'
        ) from exc

    return HistGradientBoostingClassifier
