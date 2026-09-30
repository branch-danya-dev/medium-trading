from dataclasses import dataclass
from datetime import datetime

from medium_trading.observer_policy import (
    LongPolicyAction,
    LongPolicyDecision,
    ObserverMarketState,
    ObserverSnapshot,
)


@dataclass(frozen=True, slots=True)
class LongObserverWarningState:
    armed_at: datetime
    event_type: str
    reversal_probability: float


class LongObserverPolicyV2:
    """
    Stateful confirmation policy for an already-open LONG.

    One BULL REAL_REVERSAL_RISK message only arms WARNING. A later independent
    BULL REAL_REVERSAL_RISK confirms EXIT_LONG. Any BULL TREND_SURVIVES message
    clears the warning. BEAR messages remain NO_ACTION and also clear stale BULL
    warning state because the original bullish episode context no longer applies.
    """

    version = "long-observer-policy-v2"

    def __init__(self) -> None:
        self._warning: LongObserverWarningState | None = None

    @property
    def warning(self) -> LongObserverWarningState | None:
        return self._warning

    def reset(self) -> None:
        self._warning = None

    def decide_open_long(
        self,
        snapshot: ObserverSnapshot,
    ) -> LongPolicyDecision:
        if snapshot.trend_side != "BULL":
            had_warning = self._warning is not None
            self.reset()
            return LongPolicyDecision(
                timestamp=snapshot.timestamp,
                action=LongPolicyAction.NO_ACTION,
                reason=(
                    "observer snapshot describes a BEAR trend; v2 LONG policy "
                    "does not reinterpret bearish-trend reversal probabilities"
                    + (
                        " and clears the stale BULL warning"
                        if had_warning
                        else ""
                    )
                ),
                reversal_probability=snapshot.reversal_probability,
                reversal_threshold=snapshot.reversal_threshold,
                observer_trend_side=snapshot.trend_side,
                observer_event_type=snapshot.event_type,
            )

        if snapshot.market_state is ObserverMarketState.TREND_SURVIVES:
            had_warning = self._warning is not None
            self.reset()
            return LongPolicyDecision(
                timestamp=snapshot.timestamp,
                action=LongPolicyAction.HOLD_LONG,
                reason=(
                    "BULL trend structural disturbance is classified as "
                    "TREND_SURVIVES"
                    + (
                        "; active reversal warning is cleared"
                        if had_warning
                        else ""
                    )
                ),
                reversal_probability=snapshot.reversal_probability,
                reversal_threshold=snapshot.reversal_threshold,
                observer_trend_side=snapshot.trend_side,
                observer_event_type=snapshot.event_type,
            )

        if self._warning is None:
            self._warning = LongObserverWarningState(
                armed_at=snapshot.timestamp,
                event_type=snapshot.event_type,
                reversal_probability=snapshot.reversal_probability,
            )
            return LongPolicyDecision(
                timestamp=snapshot.timestamp,
                action=LongPolicyAction.WARNING_LONG,
                reason=(
                    "first BULL REAL_REVERSAL_RISK snapshot arms WARNING; "
                    "a later independent reversal-risk snapshot is required "
                    "before EXIT_LONG"
                ),
                reversal_probability=snapshot.reversal_probability,
                reversal_threshold=snapshot.reversal_threshold,
                observer_trend_side=snapshot.trend_side,
                observer_event_type=snapshot.event_type,
            )

        if snapshot.timestamp <= self._warning.armed_at:
            return LongPolicyDecision(
                timestamp=snapshot.timestamp,
                action=LongPolicyAction.WARNING_LONG,
                reason=(
                    "duplicate/non-later reversal-risk snapshot cannot confirm "
                    "the active warning"
                ),
                reversal_probability=snapshot.reversal_probability,
                reversal_threshold=snapshot.reversal_threshold,
                observer_trend_side=snapshot.trend_side,
                observer_event_type=snapshot.event_type,
            )

        self.reset()
        return LongPolicyDecision(
            timestamp=snapshot.timestamp,
            action=LongPolicyAction.EXIT_LONG,
            reason=(
                "second independent BULL REAL_REVERSAL_RISK snapshot confirms "
                "the active warning"
            ),
            reversal_probability=snapshot.reversal_probability,
            reversal_threshold=snapshot.reversal_threshold,
            observer_trend_side=snapshot.trend_side,
            observer_event_type=snapshot.event_type,
        )
