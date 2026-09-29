from collections.abc import Sequence
from datetime import timedelta
from math import sqrt

from medium_trading.domain import Candle, Side, Signal, StrategyContext


def _true_range(candle: Candle, previous_close: float) -> float:
    return max(
        candle.high - candle.low,
        abs(candle.high - previous_close),
        abs(candle.low - previous_close),
    )


def _atr(candles: Sequence[Candle], period: int) -> float:
    if len(candles) < period + 1:
        raise ValueError("not enough candles for ATR")

    ranges = [
        _true_range(candles[index], candles[index - 1].close)
        for index in range(len(candles) - period, len(candles))
    ]
    return sum(ranges) / period


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot calculate mean of empty values")
    return sum(values) / len(values)


def _stddev(values: Sequence[float], mean: float) -> float:
    if not values:
        raise ValueError("cannot calculate stddev of empty values")
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance)


def _efficiency_ratio(closes: Sequence[float]) -> float:
    if len(closes) < 2:
        raise ValueError("not enough closes for efficiency ratio")

    path = sum(
        abs(closes[index] - closes[index - 1])
        for index in range(1, len(closes))
    )
    if path <= 0:
        return 0.0
    return abs(closes[-1] - closes[0]) / path


class MeanReversionStrategy:
    """Fixed 4H range mean-reversion baseline.

    Parameters are locked before the first historical evaluation. Do not tune
    them on out-of-sample results.
    """

    name = "mean_reversion"
    mean_period = 20
    regime_lookback = 30
    max_efficiency_ratio = 0.35
    z_threshold = 2.0
    atr_period = 14
    stop_atr_multiple = 1.5

    def evaluate(self, context: StrategyContext) -> Signal | None:
        required_4h = max(
            self.mean_period + 1,
            self.regime_lookback + 1,
            self.atr_period + 1,
        )
        if len(context.candles_4h) < required_4h:
            return None
        if not context.candles_30m:
            return None

        last_4h = context.candles_4h[-1]
        decision_time = context.candles_30m[-1].timestamp + timedelta(minutes=30)

        # Evaluate only once, immediately after a complete 4H candle closes.
        if last_4h.timestamp + timedelta(hours=4) != decision_time:
            return None

        prior_mean_closes = [
            candle.close
            for candle in context.candles_4h[-(self.mean_period + 1) : -1]
        ]
        mean = _mean(prior_mean_closes)
        stddev = _stddev(prior_mean_closes, mean)
        if stddev <= 0:
            return None

        regime_closes = [
            candle.close
            for candle in context.candles_4h[-(self.regime_lookback + 1) :]
        ]
        efficiency_ratio = _efficiency_ratio(regime_closes)
        if efficiency_ratio > self.max_efficiency_ratio:
            return None

        atr_4h = _atr(context.candles_4h, self.atr_period)
        if atr_4h <= 0:
            return None

        z_score = (last_4h.close - mean) / stddev
        if abs(z_score) < self.z_threshold:
            return None

        confidence = min(
            0.95,
            0.60 + 0.10 * (abs(z_score) - self.z_threshold),
        )

        if z_score <= -self.z_threshold:
            return Signal(
                symbol=context.symbol,
                side=Side.LONG,
                entry=last_4h.close,
                stop=last_4h.close - self.stop_atr_multiple * atr_4h,
                target=mean,
                confidence=confidence,
                strategy=self.name,
                reasons=(
                    "4h range regime",
                    "4h close at least 2 standard deviations below prior mean",
                    "target fixed at prior 20-bar mean",
                ),
            )

        return Signal(
            symbol=context.symbol,
            side=Side.SHORT,
            entry=last_4h.close,
            stop=last_4h.close + self.stop_atr_multiple * atr_4h,
            target=mean,
            confidence=confidence,
            strategy=self.name,
            reasons=(
                "4h range regime",
                "4h close at least 2 standard deviations above prior mean",
                "target fixed at prior 20-bar mean",
            ),
        )
