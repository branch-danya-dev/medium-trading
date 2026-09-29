from collections.abc import Sequence
from datetime import timedelta

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


class TimeSeriesMomentumStrategy:
    """Fixed 4H time-series momentum baseline.

    Parameters are locked before the first historical evaluation. Do not tune
    them on out-of-sample results.
    """

    name = "time_series_momentum"
    momentum_lookback = 30
    atr_period = 14
    stop_atr_multiple = 2.0

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if len(context.candles_4h) < self.momentum_lookback + 1:
            return None
        if len(context.candles_30m) < 1:
            return None

        last_4h = context.candles_4h[-1]
        decision_time = context.candles_30m[-1].timestamp + timedelta(minutes=30)

        # Act only once, exactly when a new 4H candle has closed.
        if last_4h.timestamp + timedelta(hours=4) != decision_time:
            return None

        atr_4h = _atr(context.candles_4h, self.atr_period)
        if atr_4h <= 0:
            return None

        reference_close = context.candles_4h[-(self.momentum_lookback + 1)].close
        momentum = last_4h.close - reference_close
        if momentum == 0:
            return None

        normalized_momentum = abs(momentum) / atr_4h
        confidence = min(0.95, 0.60 + 0.05 * normalized_momentum)

        if momentum > 0:
            side = Side.LONG
            stop = last_4h.close - self.stop_atr_multiple * atr_4h
        else:
            side = Side.SHORT
            stop = last_4h.close + self.stop_atr_multiple * atr_4h

        return Signal(
            symbol=context.symbol,
            side=side,
            entry=last_4h.close,
            stop=stop,
            confidence=confidence,
            strategy=self.name,
            reasons=(
                "4h 30-bar time-series momentum",
                "4h ATR14 volatility-normalized stop",
                "signal only on completed 4h candle",
            ),
        )
