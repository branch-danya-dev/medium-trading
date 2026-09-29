from medium_trading.trading.costs import is_cost_viable


def test_cost_gate_requires_configured_margin() -> None:
    assert is_cost_viable(
        expected_gross_move=80.0,
        all_in_round_trip_cost=10.0,
    )
    assert not is_cost_viable(
        expected_gross_move=79.0,
        all_in_round_trip_cost=10.0,
    )
