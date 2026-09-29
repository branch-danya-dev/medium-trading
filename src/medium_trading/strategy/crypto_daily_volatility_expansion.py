from datetime import time, timedelta

from medium_trading.domain import Candle, Side, Signal, StrategyContext


class CryptoDailyVolatilityExpansionStrategy:
    """Frozen BTC/USD UTC-day volatility-expansion baseline."""

    name = "crypto_daily_volatility_expansion"
    reference_times = (
        time(0, 0),
        time(0, 30),
        time(1, 0),
        time(1, 30),
        time(2, 0),
        time(2, 30),
        time(3, 0),
        time(3, 30),
    )
    breakout_start = time(4, 0)
    breakout_end = time(18, 0)
    minimum_extension_atr = 0.10

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.candles_30m:
            return None

        current = context.candles_30m[-1]
        decision_time = current.timestamp + timedelta(minutes=30)
        if not (
            self.breakout_start < decision_time.time() <= self.breakout_end
        ):
            return None

        session_date = decision_time.date()
        reference = self._reference_state(
            context.candles_30m,
            session_date=session_date,
        )
        if reference is None:
            return None

        range_high, range_low, midpoint, atr_14 = reference
        extension = atr_14 * self.minimum_extension_atr

        candidates: list[tuple[Candle, Side]] = []
        for candle in context.candles_30m:
            candle_decision = candle.timestamp + timedelta(minutes=30)
            if candle_decision.date() != session_date:
                continue
            if not (
                self.breakout_start
                < candle_decision.time()
                <= self.breakout_end
            ):
                continue

            if candle.close >= range_high + extension:
                candidates.append((candle, Side.LONG))
            elif candle.close <= range_low - extension:
                candidates.append((candle, Side.SHORT))

        if not candidates:
            return None

        first_breakout, side = candidates[0]
        if first_breakout.timestamp != current.timestamp:
            return None

        stop = midpoint
        if side is Side.LONG and stop >= current.close:
            return None
        if side is Side.SHORT and stop <= current.close:
            return None

        return Signal(
            symbol=context.symbol,
            side=side,
            entry=current.close,
            stop=stop,
            confidence=1.0,
            strategy=self.name,
            reasons=(
                "00:00-04:00 UTC reference range",
                "first close extends >= 0.10 ATR14 beyond reference range",
                "reference midpoint invalidation",
                "one volatility-expansion attempt per UTC day",
            ),
        )

    def _reference_state(
        self,
        candles: tuple[Candle, ...],
        *,
        session_date,
    ) -> tuple[float, float, float, float] | None:
        by_time: dict[time, tuple[int, Candle]] = {}
        for index, candle in enumerate(candles):
            if candle.timestamp.date() != session_date:
                continue
            if candle.timestamp.time() in self.reference_times:
                by_time[candle.timestamp.time()] = (index, candle)

        if set(by_time) != set(self.reference_times):
            return None

        ordered = tuple(by_time[item] for item in self.reference_times)
        end_index = ordered[-1][0]
        if end_index < 14:
            return None

        atr_14 = self._atr(candles[: end_index + 1], 14)
        if atr_14 <= 0:
            return None

        reference_candles = tuple(item[1] for item in ordered)
        range_high = max(candle.high for candle in reference_candles)
        range_low = min(candle.low for candle in reference_candles)
        if range_high <= range_low:
            return None

        return (
            range_high,
            range_low,
            (range_high + range_low) / 2,
            atr_14,
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
