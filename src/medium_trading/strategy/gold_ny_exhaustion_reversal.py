from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from medium_trading.domain import Candle, Side, Signal, StrategyContext

_NEW_YORK = ZoneInfo("America/New_York")


class GoldNewYorkExhaustionReversalStrategy:
    """Frozen XAU/USD New York opening-exhaustion reversal baseline."""

    name = "gold_ny_exhaustion_reversal"

    reference_times = (time(8, 30), time(9, 0), time(9, 30))
    reversal_start = time(10, 0)
    decision_cutoff = time(12, 0)
    minimum_move_atr = 0.75
    reversal_band_fraction = 0.50

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.candles_30m:
            return None

        current = context.candles_30m[-1]
        decision_time = current.timestamp + timedelta(minutes=30)
        decision_ny = decision_time.astimezone(_NEW_YORK)
        if not (
            time(10, 30)
            <= decision_ny.time()
            <= self.decision_cutoff
        ):
            return None

        session_date = decision_ny.date()
        reference = self._reference_state(
            context.candles_30m,
            session_date=session_date,
        )
        if reference is None:
            return None

        (
            impulse_side,
            impulse_start,
            impulse_end,
            midpoint,
            impulse_extreme,
            last_reference_timestamp,
        ) = reference

        reversal_candidates = tuple(
            candle
            for candle in context.candles_30m
            if candle.timestamp > last_reference_timestamp
            and candle.timestamp.astimezone(_NEW_YORK).date() == session_date
            and self.reversal_start
            <= candle.timestamp.astimezone(_NEW_YORK).time()
            < self.decision_cutoff
            and self._is_counter_direction(candle, impulse_side)
        )
        if not reversal_candidates:
            return None

        first_reversal = reversal_candidates[0]
        if first_reversal.timestamp != current.timestamp:
            return None

        if not self._closes_inside_last_half(
            first_reversal.close,
            impulse_side=impulse_side,
            impulse_start=impulse_start,
            impulse_end=impulse_end,
            midpoint=midpoint,
        ):
            return None

        side = Side.SHORT if impulse_side is Side.LONG else Side.LONG
        stop = impulse_extreme
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
                "08:30-10:00 New York move >= 0.75 ATR14",
                "first counter-direction M30 reversal candle",
                "reversal closes inside final 50% of opening impulse",
                "one reversal attempt per session",
            ),
        )

    def _reference_state(
        self,
        candles: tuple[Candle, ...],
        *,
        session_date,
    ) -> tuple[Side, float, float, float, float, datetime] | None:
        by_time: dict[time, tuple[int, Candle]] = {}
        for index, candle in enumerate(candles):
            local = candle.timestamp.astimezone(_NEW_YORK)
            if local.date() != session_date:
                continue
            if local.time() in self.reference_times:
                by_time[local.time()] = (index, candle)

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
        impulse_start = reference_candles[0].open
        impulse_end = reference_candles[-1].close
        move = impulse_end - impulse_start
        if abs(move) < self.minimum_move_atr * atr_14:
            return None

        impulse_side = Side.LONG if move > 0 else Side.SHORT
        midpoint = impulse_start + move * self.reversal_band_fraction
        impulse_extreme = (
            max(candle.high for candle in reference_candles)
            if impulse_side is Side.LONG
            else min(candle.low for candle in reference_candles)
        )

        return (
            impulse_side,
            impulse_start,
            impulse_end,
            midpoint,
            impulse_extreme,
            reference_candles[-1].timestamp,
        )

    @staticmethod
    def _is_counter_direction(candle: Candle, impulse_side: Side) -> bool:
        if impulse_side is Side.LONG:
            return candle.close < candle.open
        return candle.close > candle.open

    @staticmethod
    def _closes_inside_last_half(
        close: float,
        *,
        impulse_side: Side,
        impulse_start: float,
        impulse_end: float,
        midpoint: float,
    ) -> bool:
        if impulse_side is Side.LONG:
            return max(impulse_start, midpoint) <= close <= impulse_end
        return impulse_end <= close <= min(impulse_start, midpoint)

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
