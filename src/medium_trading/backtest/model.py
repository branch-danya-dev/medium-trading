from dataclasses import dataclass
from datetime import datetime

from medium_trading.domain import Side


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    starting_equity: float = 1_000.0
    risk_fraction: float = 0.005
    target_r: float = 2.0
    max_holding_bars: int = 48
    round_trip_cost_pips: float = 1.2
    fee_bps_per_side: float | None = None
    slippage_bps_per_side: float = 0.0
    cost_stress_multiplier: float = 1.0
    minimum_cost_multiple: float = 8.0
    max_holding_minutes: int | None = None
    required_contiguous_context_bars: int = 0
    required_contiguous_future_bars: int = 0


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    symbol: str
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry: float
    stop: float
    exit: float
    gross_r: float
    net_r: float
    cost_r: float
    exit_reason: str
    fee_r: float = 0.0
    slippage_r: float = 0.0


@dataclass(frozen=True, slots=True)
class BacktestReport:
    symbol: str
    signal_count: int
    cost_rejections: int
    invalidated_before_entry: int
    trades: tuple[BacktestTrade, ...]
    starting_equity: float
    final_equity: float
    net_r: float
    gross_r: float
    total_cost_r: float
    profit_factor: float
    gross_profit_factor: float
    win_rate: float
    max_drawdown: float
    average_holding_hours: float
    stop_exits: int
    target_exits: int
    timeout_exits: int
    gap_context_rejections: int = 0
    gap_signal_rejections: int = 0
