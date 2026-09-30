from datetime import UTC, datetime, timedelta

from medium_trading.crypto_entry_v2 import (
    MatchedControl,
    _entry_gate,
    _h4_regime_ok,
    _matched_summary,
    _path_diagnostics,
)
from medium_trading.domain import Candle
from medium_trading.strategy.crypto_trend_long import CryptoTrendLongV2Strategy


def _h4_history(*, disrupted: bool) -> tuple[Candle, ...]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = []
    price = 90_000.0
    for index in range(24):
        close = price + 200.0
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=4 * index),
                open=price,
                high=max(price, close) + 50.0,
                low=min(price, close) - 50.0,
                close=close,
            )
        )
        price = close

    if disrupted:
        candle = candles[-2]
        candles[-2] = Candle(
            timestamp=candle.timestamp,
            open=94_400.0,
            high=94_500.0,
            low=91_900.0,
            close=92_000.0,
        )
    return tuple(candles)


def test_persistent_h4_regime_rejects_temporary_recovery() -> None:
    strategy = CryptoTrendLongV2Strategy()
    history = _h4_history(disrupted=True)

    assert _h4_regime_ok(
        history=history,
        persistent=False,
        strategy=strategy,
    )
    assert not _h4_regime_ok(
        history=history,
        persistent=True,
        strategy=strategy,
    )


def test_path_diagnostics_use_conservative_stop_first_ordering() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = (
        Candle(
            timestamp=start,
            open=100.0,
            high=125.0,
            low=85.0,
            close=105.0,
        ),
        Candle(
            timestamp=start + timedelta(minutes=30),
            open=105.0,
            high=130.0,
            low=100.0,
            close=120.0,
        ),
    )

    mfe, mae, hit_1r, hit_2r = _path_diagnostics(
        candles_30m=candles,
        entry_index=0,
        entry=100.0,
        stop=90.0,
    )

    assert mfe == 3.0
    assert mae == 1.5
    assert hit_1r is False
    assert hit_2r is False


def test_matched_summary_reports_real_coverage() -> None:
    controls = (
        MatchedControl(
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
            actual_gross_r=0.5,
            random_mean_gross_r=0.1,
            random_mean_net_r=-0.1,
            random_reached_1r_rate=0.4,
            random_reached_2r_rate=0.2,
            matches=10,
            fallback_used=False,
        ),
        MatchedControl(
            entry_time=datetime(2025, 1, 2, tzinfo=UTC),
            actual_gross_r=-1.0,
            random_mean_gross_r=-0.2,
            random_mean_net_r=-0.4,
            random_reached_1r_rate=0.3,
            random_reached_2r_rate=0.1,
            matches=8,
            fallback_used=True,
        ),
    )

    summary = _matched_summary(controls, requested_trades=4)

    assert summary["matched_trades"] == 2
    assert summary["requested_trades"] == 4
    assert summary["coverage"] == 0.5
    assert summary["fallback_matches"] == 1


def test_entry_gate_requires_information_and_economic_edge() -> None:
    baseline = {
        "net_r": 5.0,
        "gross_profit_factor": 1.05,
    }
    candidate = {
        "trades": 200,
        "gross_r": 40.0,
        "gross_profit_factor": 1.25,
        "net_r": 20.0,
        "profit_factor": 1.15,
        "reached_1r_before_stop_rate": 0.55,
        "2x_cost": {
            "net_r": 5.0,
            "profit_factor": 1.05,
        },
        "years": [
            {"year": 2023, "gross_r": 10.0},
            {"year": 2024, "gross_r": 15.0},
            {"year": 2025, "gross_r": 15.0},
        ],
        "matched_random": {
            "coverage": 0.98,
            "average_matches_per_trade": 12.0,
            "mean_gross_edge_r": 0.15,
            "gross_edge_bootstrap_95pct": [0.03, 0.27],
            "random_reached_1r_before_stop_rate": 0.45,
        },
    }

    gate = _entry_gate(baseline=baseline, candidate=candidate)

    assert gate["passes"] is True
    assert gate["information_gate_passes"] is True
    assert gate["economic_gate_passes"] is True

    candidate["matched_random"]["gross_edge_bootstrap_95pct"] = [-0.02, 0.25]
    failed = _entry_gate(baseline=baseline, candidate=candidate)

    assert failed["passes"] is False
    assert failed["information_gate_passes"] is False
    assert (
        failed["conditions"][
            "matched_random_edge_bootstrap_lower_bound_positive"
        ]
        is False
    )
