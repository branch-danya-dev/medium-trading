from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ObserverMarketState(str, Enum):
    TREND_SURVIVES = "trend_survives"
    REAL_REVERSAL_RISK = "real_reversal_risk"


class LongPolicyAction(str, Enum):
    NO_ACTION = "no_action"
    HOLD_LONG = "hold_long"
    EXIT_LONG = "exit_long"


@dataclass(frozen=True, slots=True)
class ObserverSnapshot:
    """
    Transport-neutral message emitted by Market Observer v0.7.

    The snapshot contains only market interpretation. It has no position,
    entry, stop, target, PnL or trading-action fields.
    """

    timestamp: datetime
    model_version: str
    trend_side: str
    event_type: str
    reversal_probability: float
    trend_survives_probability: float
    reversal_threshold: float
    market_state: ObserverMarketState

    def __post_init__(self) -> None:
        if not 0.0 <= self.reversal_probability <= 1.0:
            raise ValueError("reversal_probability must be between 0 and 1")
        if not 0.0 <= self.trend_survives_probability <= 1.0:
            raise ValueError(
                "trend_survives_probability must be between 0 and 1"
            )
        if abs(
            self.reversal_probability + self.trend_survives_probability - 1.0
        ) > 1e-9:
            raise ValueError(
                "observer state probabilities must sum to 1"
            )
        if not 0.0 <= self.reversal_threshold <= 1.0:
            raise ValueError("reversal_threshold must be between 0 and 1")
        if self.trend_side not in {"BULL", "BEAR"}:
            raise ValueError("trend_side must be BULL or BEAR")


@dataclass(frozen=True, slots=True)
class LongPolicyDecision:
    """
    Trading-policy interpretation of one observer snapshot.

    This is deliberately outside the observer model. The policy sees an open
    LONG position conceptually; the observer does not.
    """

    timestamp: datetime
    action: LongPolicyAction
    reason: str
    reversal_probability: float
    reversal_threshold: float
    observer_trend_side: str
    observer_event_type: str


class LongObserverPolicy:
    """
    Frozen v1 policy for an already-open LONG.

    A BULL-trend disturbance classified as REAL_REVERSAL_RISK exits the LONG
    at the next executable market price. A BULL TREND_SURVIVES snapshot keeps
    the position open. BEAR-side snapshots are not translated into a LONG
    action in v1 because their target is reversal of a bearish trend.
    """

    version = "long-observer-policy-v1"

    def decide_open_long(
        self,
        snapshot: ObserverSnapshot,
    ) -> LongPolicyDecision:
        if snapshot.trend_side != "BULL":
            return LongPolicyDecision(
                timestamp=snapshot.timestamp,
                action=LongPolicyAction.NO_ACTION,
                reason=(
                    "observer snapshot describes a BEAR trend; v1 LONG policy "
                    "does not reinterpret bearish-trend reversal probabilities"
                ),
                reversal_probability=snapshot.reversal_probability,
                reversal_threshold=snapshot.reversal_threshold,
                observer_trend_side=snapshot.trend_side,
                observer_event_type=snapshot.event_type,
            )

        if snapshot.market_state is ObserverMarketState.REAL_REVERSAL_RISK:
            return LongPolicyDecision(
                timestamp=snapshot.timestamp,
                action=LongPolicyAction.EXIT_LONG,
                reason=(
                    "BULL trend structural disturbance is classified as "
                    "REAL_REVERSAL_RISK"
                ),
                reversal_probability=snapshot.reversal_probability,
                reversal_threshold=snapshot.reversal_threshold,
                observer_trend_side=snapshot.trend_side,
                observer_event_type=snapshot.event_type,
            )

        return LongPolicyDecision(
            timestamp=snapshot.timestamp,
            action=LongPolicyAction.HOLD_LONG,
            reason=(
                "BULL trend structural disturbance is classified as "
                "TREND_SURVIVES"
            ),
            reversal_probability=snapshot.reversal_probability,
            reversal_threshold=snapshot.reversal_threshold,
            observer_trend_side=snapshot.trend_side,
            observer_event_type=snapshot.event_type,
        )


def observer_snapshot_from_probability(
    *,
    timestamp: datetime,
    trend_side: str,
    event_type: str,
    reversal_probability: float,
    reversal_threshold: float,
    model_version: str = "market-observer-v0.7",
) -> ObserverSnapshot:
    state = (
        ObserverMarketState.REAL_REVERSAL_RISK
        if reversal_probability >= reversal_threshold
        else ObserverMarketState.TREND_SURVIVES
    )
    return ObserverSnapshot(
        timestamp=timestamp,
        model_version=model_version,
        trend_side=trend_side,
        event_type=event_type,
        reversal_probability=reversal_probability,
        trend_survives_probability=1.0 - reversal_probability,
        reversal_threshold=reversal_threshold,
        market_state=state,
    )
