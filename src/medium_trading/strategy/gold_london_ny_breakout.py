from datetime import time, timedelta
from zoneinfo import ZoneInfo

from medium_trading.domain import Candle, Side, Signal, StrategyContext

_LONDON = ZoneInfo("Europe/London")
_NEW_YORK = ZoneInfo("America/New_York")


class GoldLondonNewYorkBreakoutStrategy:
    """Frozen XAU/USD London-reference to New York breakout baseline."""

    name = "gold_london_ny_breakout"
    london_range_start = time(8, 0)
    london_range_end = time(10, 30)
    new_york_trade_start = time(8, 30)
    new_york_trade_end = time(11, 0)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.candles_30m:
            return None

        current = context.candles_30m[-1]
        decision_time = current.timestamp + timedelta(minutes=30)
        decision_ny = decision_time.astimezone(_NEW_YORK)
        if not (
            self.new_york_trade_start
            <= decision_ny.time()
            <= self.new_york_trade_end
        ):
            return None

        session_date = decision_ny.date()
        london_range = self._london_range(
            context.candles_30m,
            session_date=session_date,
        )
        if london_range is None:
            return None
        range_high, range_low = london_range
        if range_high <= range_low:
            return None

        breakout_candidates: list[tuple[Candle, Side]] = []
        for candle in context.candles_30m:
            candle_decision = candle.timestamp + timedelta(minutes=30)
            candle_ny = candle_decision.astimezone(_NEW_YORK)
            if candle_ny.date() != session_date:
                continue
            if not (
                self.new_york_trade_start
                <= candle_ny.time()
                <= self.new_york_trade_end
            ):
                continue

            if candle.close > range_high:
                breakout_candidates.append((candle, Side.LONG))
            elif candle.close < range_low:
                breakout_candidates.append((candle, Side.SHORT))

        if not breakout_candidates:
            return None

        first_breakout, side = breakout_candidates[0]
        if first_breakout.timestamp != current.timestamp:
            return None

        stop = range_low if side is Side.LONG else range_high
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
                "08:00-10:30 London reference range",
                "first New York window close outside London range",
                "one breakout attempt per session",
            ),
        )

    def _london_range(
        self,
        candles: tuple[Candle, ...],
        *,
        session_date,
    ) -> tuple[float, float] | None:
        expected_times = {
            time(8, 0),
            time(8, 30),
            time(9, 0),
            time(9, 30),
            time(10, 0),
        }
        range_candles: dict[time, Candle] = {}

        for candle in candles:
            local = candle.timestamp.astimezone(_LONDON)
            if local.date() != session_date:
                continue
            if local.time() in expected_times:
                range_candles[local.time()] = candle

        if set(range_candles) != expected_times:
            return None

        ordered = tuple(range_candles[item] for item in sorted(expected_times))
        return (
            max(candle.high for candle in ordered),
            min(candle.low for candle in ordered),
        )
