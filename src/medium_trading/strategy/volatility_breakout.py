from collections.abc import Sequence

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


class VolatilityBreakoutStrategy:
    """Fixed baseline for the second strategy research gate.

    The parameters are intentionally small and fixed before the first historical
    evaluation. Do not tune them on out-of-sample results.
    """

    name = "volatility_breakout"
    channel_period = 20
    atr_period = 14
    stop_atr_multiple = 1.5

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if len(context.candles_4h) < self.channel_period:
            return None
        if len(context.candles_1h) < self.channel_period:
            return None
        if len(context.candles_30m) < self.atr_period + 2:
            return None

        regime_4h = context.candles_4h[-self.channel_period :]
        high_4h = max(candle.high for candle in regime_4h)
        low_4h = min(candle.low for candle in regime_4h)
        midpoint_4h = (high_4h + low_4h) / 2.0
        last_4h = regime_4h[-1]

        channel_1h = context.candles_1h[-self.channel_period :]
        upper = max(candle.high for candle in channel_1h)
        lower = min(candle.low for candle in channel_1h)

        last_30m = context.candles_30m[-1]
        previous_30m = context.candles_30m[-2]
        atr_30m = _atr(context.candles_30m, self.atr_period)
        trigger_range = _true_range(last_30m, previous_30m.close)
        range_expansion = trigger_range >= atr_30m

        if not range_expansion or atr_30m <= 0:
            return None

        if last_4h.close > midpoint_4h and last_30m.close > upper:
            stop = last_30m.close - self.stop_atr_multiple * atr_30m
            return self._signal(
                context.symbol,
                Side.LONG,
                last_30m,
                stop,
                breakout_distance=last_30m.close - upper,
                atr=atr_30m,
            )

        if last_4h.close < midpoint_4h and last_30m.close < lower:
            stop = last_30m.close + self.stop_atr_multiple * atr_30m
            return self._signal(
                context.symbol,
                Side.SHORT,
                last_30m,
                stop,
                breakout_distance=lower - last_30m.close,
                atr=atr_30m,
            )

        return None

    def _signal(
        self,
        symbol: str,
        side: Side,
        candle: Candle,
        stop: float,
        *,
        breakout_distance: float,
        atr: float,
    ) -> Signal:
        normalized_breakout = breakout_distance / atr if atr > 0 else 0.0
        confidence = min(0.95, 0.60 + 0.10 * normalized_breakout)

        return Signal(
            symbol=symbol,
            side=side,
            entry=candle.close,
            stop=stop,
            confidence=confidence,
            strategy=self.name,
            reasons=(
                "4h directional regime",
                "1h 20-bar Donchian breakout",
                "30m range expansion",
            ),
        )
