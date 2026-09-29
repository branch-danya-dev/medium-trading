def cost_multiple(expected_gross_move: float, all_in_round_trip_cost: float) -> float:
    if expected_gross_move < 0:
        raise ValueError("expected_gross_move cannot be negative")
    if all_in_round_trip_cost <= 0:
        raise ValueError("all_in_round_trip_cost must be positive")
    return expected_gross_move / all_in_round_trip_cost


def is_cost_viable(
    *,
    expected_gross_move: float,
    all_in_round_trip_cost: float,
    minimum_multiple: float = 8.0,
) -> bool:
    if minimum_multiple <= 0:
        raise ValueError("minimum_multiple must be positive")
    return cost_multiple(expected_gross_move, all_in_round_trip_cost) >= minimum_multiple
