from collections.abc import Sequence

from medium_trading.domain import Candle, Side, Signal, StrategyContext


def _ema(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        raise ValueError("not enough values for EMA")

    alpha = 2.0 / (period + 1.0)
    result = sum(values[:period]) / period
    for value in values[period:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


class TrendPullbackStrategy:
    """Small baseline strategy. It is intentionally simple and not yet validated."""

    name = "trend_pullback"

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if len(context.candles_4h) < 21:
            return None
        if len(context.candles_1h) < 8:
            return None
        if len(context.candles_30m) < 4:
            return None

        closes_4h = [candle.close for candle in context.candles_4h]
        fast_4h = _ema(closes_4h, 8)
        slow_4h = _ema(closes_4h, 21)
        last_4h = context.candles_4h[-1]

        closes_1h = [candle.close for candle in context.candles_1h]
        fast_1h = _ema(closes_1h, 8)
        recent_1h = context.candles_1h[-3:]

        previous_30m = context.candles_30m[-2]
        last_30m = context.candles_30m[-1]
        recent_30m = context.candles_30m[-4:]

        if fast_4h > slow_4h and last_4h.close > slow_4h:
            pullback_seen = min(candle.low for candle in recent_1h) <= fast_1h
            trigger = last_30m.close > previous_30m.high
            stop = min(candle.low for candle in recent_30m)
            if pullback_seen and trigger and stop < last_30m.close:
                return self._signal(context.symbol, Side.LONG, last_30m, stop, fast_4h, slow_4h)

        if fast_4h < slow_4h and last_4h.close < slow_4h:
            pullback_seen = max(candle.high for candle in recent_1h) >= fast_1h
            trigger = last_30m.close < previous_30m.low
            stop = max(candle.high for candle in recent_30m)
            if pullback_seen and trigger and stop > last_30m.close:
                return self._signal(context.symbol, Side.SHORT, last_30m, stop, fast_4h, slow_4h)

        return None

    def _signal(
        self,
        symbol: str,
        side: Side,
        candle: Candle,
        stop: float,
        fast_ema: float,
        slow_ema: float,
    ) -> Signal:
        denominator = max(abs(slow_ema), 1e-12)
        trend_strength = abs(fast_ema - slow_ema) / denominator
        confidence = min(0.95, 0.60 + 10.0 * trend_strength)

        return Signal(
            symbol=symbol,
            side=side,
            entry=candle.close,
            stop=stop,
            confidence=confidence,
            strategy=self.name,
            reasons=("4h trend", "1h pullback", "30m continuation trigger"),
        )
