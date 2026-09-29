import math


def calculate_quantity(
    *,
    equity: float,
    risk_fraction: float,
    loss_per_unit_at_stop: float,
    min_quantity: float = 1.0,
    quantity_step: float = 1.0,
) -> float:
    if equity <= 0:
        raise ValueError("equity must be positive")
    if not 0 < risk_fraction < 1:
        raise ValueError("risk_fraction must be between 0 and 1")
    if loss_per_unit_at_stop <= 0:
        raise ValueError("loss_per_unit_at_stop must be positive")
    if min_quantity <= 0 or quantity_step <= 0:
        raise ValueError("quantity constraints must be positive")

    risk_budget = equity * risk_fraction
    raw_quantity = risk_budget / loss_per_unit_at_stop
    quantity = math.floor(raw_quantity / quantity_step) * quantity_step

    if quantity < min_quantity:
        return 0.0
    return quantity
