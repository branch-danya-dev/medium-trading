from datetime import time, timedelta

from medium_trading.domain import Candle, Side, Signal, StrategyContext


class CryptoIntradayMomentumContinuationStrategy:
    """Frozen BTC/USD rolling intraday momentum-continuation baseline."""

    name = "crypto_intraday_momentum_continuation"
    decision_start = time(2, 0)
    decision_end = time(22, 0)
    trend_bars = 8
    minimum_directional_bars = 5
    minimum_trend_atr = 0.75
    maximum_retracement_fraction = 0.50

    def evaluate(self, context: StrategyContext) -> Signal | None:
        candles = context.candles_30m
        if len(candles) < 15:
            return None

        current = candles[-1]
        decision_time = current.timestamp + timedelta(minutes=30)
        if not self.decision_start <= decision_time.time() <= self.decision_end:
            return None

        session_date = decision_time.date()
        first_candidate = self._first_candidate_for_day(
            candles,
            session_date=session_date,
        )
        if first_candidate is None:
            return None

        candidate, side, stop = first_candidate
        if candidate.timestamp != current.timestamp:
            return None

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
                "rolling 4h move >= 0.75 ATR14",
                "at least 5 of 8 trend candles align with direction",
                "one shallow counter-direction M30 pullback",
                "next candle resumes through pullback boundary",
                "one momentum-continuation attempt per UTC day",
            ),
        )

    def _first_candidate_for_day(
        self,
        candles: tuple[Candle, ...],
        *,
        session_date,
    ) -> tuple[Candle, Side, float] | None:
        for index in range(14, len(candles)):
            current = candles[index]
            decision_time = current.timestamp + timedelta(minutes=30)
            if decision_time.date() != session_date:
                continue
            if not self.decision_start <= decision_time.time() <= self.decision_end:
                continue

            candidate = self._candidate_at(candles, index)
            if candidate is not None:
                return candidate
        return None

    def _candidate_at(
        self,
        candles: tuple[Candle, ...],
        index: int,
    ) -> tuple[Candle, Side, float] | None:
        if index < self.trend_bars + 1:
            return None

        continuity_start = max(0, index - 14)
        if not self._is_contiguous(candles[continuity_start : index + 1]):
            return None

        atr_14 = self._atr(candles[: index + 1], 14)
        if atr_14 <= 0:
            return None

        pullback = candles[index - 1]
        current = candles[index]
        trend = candles[
            index - self.trend_bars - 1 : index - 1
        ]
        if len(trend) != self.trend_bars:
            return None

        trend_start = trend[0].open
        trend_end = trend[-1].close
        move = trend_end - trend_start
        if abs(move) < self.minimum_trend_atr * atr_14:
            return None

        side = Side.LONG if move > 0 else Side.SHORT
        directional = sum(
            self._is_directional(candle, side)
            for candle in trend
        )
        if directional < self.minimum_directional_bars:
            return None

        midpoint = trend_end - move * self.maximum_retracement_fraction
        if side is Side.LONG:
            if pullback.close >= pullback.open:
                return None
            if pullback.low < midpoint:
                return None
            if current.close <= current.open or current.close <= pullback.high:
                return None
            return current, side, pullback.low

        if pullback.close <= pullback.open:
            return None
        if pullback.high > midpoint:
            return None
        if current.close >= current.open or current.close >= pullback.low:
            return None
        return current, side, pullback.high

    @staticmethod
    def _is_contiguous(candles: tuple[Candle, ...]) -> bool:
        return all(
            current.timestamp - previous.timestamp == timedelta(minutes=30)
            for previous, current in zip(candles, candles[1:])
        )

    @staticmethod
    def _is_directional(candle: Candle, side: Side) -> bool:
        if side is Side.LONG:
            return candle.close > candle.open
        return candle.close < candle.open

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
