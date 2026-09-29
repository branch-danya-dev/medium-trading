from dataclasses import replace

from medium_trading.domain import Candle, Side, Signal, StrategyContext


class CryptoTrendLongStrategy:
    """Frozen BTC/USD LONG-only trend baseline without ML/noise filtering."""

    name = "crypto_trend_long"
    ema_period = 20
    ema_slope_lookback = 3

    def evaluate(self, context: StrategyContext) -> Signal | None:
        candles_4h = context.candles_4h
        candles_30m = context.candles_30m
        required_4h = self.ema_period + self.ema_slope_lookback
        if len(candles_4h) < required_4h or len(candles_30m) < 2:
            return None

        ema_values = self._ema_series(
            tuple(candle.close for candle in candles_4h),
            self.ema_period,
        )
        current_ema = ema_values[-1]
        prior_ema = ema_values[-1 - self.ema_slope_lookback]
        current_4h = candles_4h[-1]

        if current_4h.close <= current_ema:
            return None
        if current_ema <= prior_ema:
            return None

        pullback = candles_30m[-2]
        confirmation = candles_30m[-1]

        if pullback.close >= pullback.open:
            return None
        if confirmation.close <= confirmation.open:
            return None
        if confirmation.close <= pullback.high:
            return None

        stop = pullback.low
        if stop >= confirmation.close:
            return None

        return Signal(
            symbol=context.symbol,
            side=Side.LONG,
            entry=confirmation.close,
            stop=stop,
            confidence=1.0,
            strategy=self.name,
            reasons=(
                "4h close above EMA20",
                "EMA20 rising versus 3 completed 4h bars ago",
                "bearish M30 pullback",
                "bullish M30 confirmation closes above pullback high",
                "LONG only; no ML or noise filter",
            ),
        )

    @staticmethod
    def _ema_series(values: tuple[float, ...], period: int) -> tuple[float, ...]:
        if not values:
            return ()
        alpha = 2.0 / (period + 1)
        ema = values[0]
        result = [ema]
        for value in values[1:]:
            ema = alpha * value + (1.0 - alpha) * ema
            result.append(ema)
        return tuple(result)



class CryptoTrendLongV11Strategy(CryptoTrendLongStrategy):
    """BTC Trend LONG v1.1: same setup, stop cannot be tighter than ATR14."""

    name = "crypto_trend_long_v1_1"
    atr_period = 14

    def evaluate(self, context: StrategyContext) -> Signal | None:
        signal = super().evaluate(context)
        if signal is None:
            return None

        atr = self._atr(context.candles_30m, self.atr_period)
        if atr <= 0:
            return None

        return replace(
            signal,
            strategy=self.name,
            minimum_stop_distance=atr,
            reasons=signal.reasons
            + (
                "stop floor: at least 1.0 x M30 ATR14 from actual next-open entry",
            ),
        )

    @staticmethod
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
