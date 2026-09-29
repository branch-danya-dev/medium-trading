from datetime import time, timedelta
from zoneinfo import ZoneInfo

from medium_trading.domain import Side, Signal, StrategyContext

_NEW_YORK = ZoneInfo("America/New_York")


class OpeningRangeBreakoutStrategy:
    """Frozen 30-minute US cash-session opening-range breakout baseline."""

    name = "opening_range_breakout"
    opening_range_start = time(9, 30)
    breakout_start = time(10, 0)
    breakout_end = time(13, 0)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.candles_30m:
            return None

        current = context.candles_30m[-1]
        current_local = current.timestamp.astimezone(_NEW_YORK)
        decision_local = (
            current.timestamp + timedelta(minutes=30)
        ).astimezone(_NEW_YORK)

        if decision_local.date() != current_local.date():
            return None
        if not self.breakout_start < decision_local.time() <= self.breakout_end:
            return None

        session_date = current_local.date()
        session_candles = [
            candle
            for candle in context.candles_30m
            if candle.timestamp.astimezone(_NEW_YORK).date() == session_date
        ]
        opening_range = next(
            (
                candle
                for candle in session_candles
                if candle.timestamp.astimezone(_NEW_YORK).time()
                == self.opening_range_start
            ),
            None,
        )
        if opening_range is None:
            return None

        breakout_candidates = []
        for candle in session_candles:
            local_time = candle.timestamp.astimezone(_NEW_YORK).time()
            close_time = (
                candle.timestamp + timedelta(minutes=30)
            ).astimezone(_NEW_YORK).time()
            if local_time < self.breakout_start or close_time > self.breakout_end:
                continue
            if candle.close > opening_range.high:
                breakout_candidates.append((candle, Side.LONG))
            elif candle.close < opening_range.low:
                breakout_candidates.append((candle, Side.SHORT))

        if not breakout_candidates:
            return None

        first_breakout, side = breakout_candidates[0]
        if first_breakout.timestamp != current.timestamp:
            return None

        stop = (
            opening_range.low
            if side is Side.LONG
            else opening_range.high
        )
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
                "30m US cash opening range",
                "first close outside opening range",
                "one breakout attempt per session",
            ),
        )
