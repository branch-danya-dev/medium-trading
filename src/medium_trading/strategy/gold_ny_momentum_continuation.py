from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from medium_trading.domain import Candle, Side, Signal, StrategyContext

_NEW_YORK = ZoneInfo("America/New_York")


class GoldNewYorkMomentumContinuationStrategy:
    """Frozen XAU/USD pre-NY impulse, pullback and continuation baseline."""

    name = "gold_ny_momentum_continuation"

    impulse_times = (
        time(5, 0),
        time(5, 30),
        time(6, 0),
        time(6, 30),
        time(7, 0),
        time(7, 30),
        time(8, 0),
    )
    pullback_start = time(8, 30)
    trade_end = time(11, 30)
    minimum_directional_bars = 5
    impulse_atr_multiple = 1.0
    maximum_retracement_fraction = 0.50

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.candles_30m:
            return None

        current = context.candles_30m[-1]
        decision_time = current.timestamp + timedelta(minutes=30)
        decision_ny = decision_time.astimezone(_NEW_YORK)
        if not (time(9, 0) <= decision_ny.time() <= self.trade_end):
            return None

        session_date = decision_ny.date()
        impulse = self._impulse_state(
            context.candles_30m,
            session_date=session_date,
        )
        if impulse is None:
            return None

        side, impulse_start, impulse_end, midpoint, impulse_last_timestamp = impulse
        post_impulse = tuple(
            candle
            for candle in context.candles_30m
            if candle.timestamp > impulse_last_timestamp
            and candle.timestamp.astimezone(_NEW_YORK).date() == session_date
            and self.pullback_start
            <= candle.timestamp.astimezone(_NEW_YORK).time()
            < self.trade_end
        )
        if not post_impulse:
            return None

        pullback: Candle | None = None
        pullback_index = -1
        for index, candle in enumerate(post_impulse):
            if self._is_counter_direction(candle, side):
                pullback = candle
                pullback_index = index
                break

        if pullback is None:
            return None

        if side is Side.LONG and pullback.low < midpoint:
            return None
        if side is Side.SHORT and pullback.high > midpoint:
            return None

        continuation: Candle | None = None
        for candle in post_impulse[pullback_index + 1 :]:
            if side is Side.LONG:
                threshold = max(pullback.high, impulse_end)
                if candle.close > candle.open and candle.close > threshold:
                    continuation = candle
                    break
            else:
                threshold = min(pullback.low, impulse_end)
                if candle.close < candle.open and candle.close < threshold:
                    continuation = candle
                    break

        if continuation is None or continuation.timestamp != current.timestamp:
            return None

        stop = pullback.low if side is Side.LONG else pullback.high
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
                "05:00-08:30 New York impulse >= 1.0 ATR14",
                "at least 5 of 7 impulse candles agree with direction",
                "first counter-direction pullback retraces no more than 50%",
                "first continuation close resumes beyond pullback and impulse end",
                "one continuation attempt per session",
            ),
        )

    def _impulse_state(
        self,
        candles: tuple[Candle, ...],
        *,
        session_date,
    ) -> tuple[Side, float, float, float, datetime] | None:
        by_time: dict[time, tuple[int, Candle]] = {}
        for index, candle in enumerate(candles):
            local = candle.timestamp.astimezone(_NEW_YORK)
            if local.date() != session_date:
                continue
            if local.time() in self.impulse_times:
                by_time[local.time()] = (index, candle)

        if set(by_time) != set(self.impulse_times):
            return None

        ordered = tuple(by_time[item] for item in self.impulse_times)
        end_index = ordered[-1][0]
        if end_index < 14:
            return None

        atr_14 = self._atr(candles[: end_index + 1], 14)
        if atr_14 <= 0:
            return None

        impulse_candles = tuple(item[1] for item in ordered)
        impulse_start = impulse_candles[0].open
        impulse_end = impulse_candles[-1].close
        move = impulse_end - impulse_start
        if abs(move) < self.impulse_atr_multiple * atr_14:
            return None

        side = Side.LONG if move > 0 else Side.SHORT
        directional = sum(
            self._is_directional(candle, side)
            for candle in impulse_candles
        )
        if directional < self.minimum_directional_bars:
            return None

        midpoint = impulse_end - move * self.maximum_retracement_fraction
        return (
            side,
            impulse_start,
            impulse_end,
            midpoint,
            impulse_candles[-1].timestamp,
        )

    @staticmethod
    def _is_directional(candle: Candle, side: Side) -> bool:
        if side is Side.LONG:
            return candle.close > candle.open
        return candle.close < candle.open

    @staticmethod
    def _is_counter_direction(candle: Candle, side: Side) -> bool:
        if side is Side.LONG:
            return candle.close < candle.open
        return candle.close > candle.open

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
