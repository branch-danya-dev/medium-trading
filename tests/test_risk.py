import pytest

from medium_trading.trading.risk import calculate_quantity


def test_calculate_quantity_uses_risk_budget() -> None:
    quantity = calculate_quantity(
        equity=1_000.0,
        risk_fraction=0.005,
        loss_per_unit_at_stop=0.001,
    )

    assert quantity == 5_000.0


def test_calculate_quantity_rejects_invalid_risk() -> None:
    with pytest.raises(ValueError):
        calculate_quantity(
            equity=1_000.0,
            risk_fraction=0.0,
            loss_per_unit_at_stop=0.001,
        )
