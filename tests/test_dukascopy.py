from pathlib import Path

import pytest

from medium_trading.data.dukascopy import import_dukascopy


def test_imports_dukascopy_m30_candle_csv(tmp_path: Path) -> None:
    source = tmp_path / "candles.csv"
    source.write_text(
        "Gmt time;Open;High;Low;Close;Volume\n"
        "01.01.2024 00:00:00.000;1.1000;1.1010;1.0990;1.1005;10\n"
        "01.01.2024 00:30:00.000;1.1005;1.1020;1.1000;1.1015;12\n",
        encoding="utf-8",
    )

    result = import_dukascopy(source)

    assert result.source_kind == "candles"
    assert result.mean_spread is None
    assert len(result.candles) == 2
    assert result.candles[1].close == pytest.approx(1.1015)


def test_imports_tick_csv_and_aggregates_to_m30(tmp_path: Path) -> None:
    source = tmp_path / "ticks.csv"
    source.write_text(
        "timestamp,ask,bid,ask_volume,bid_volume\n"
        "2024-01-01T00:00:00Z,1.1002,1.1000,2,3\n"
        "2024-01-01T00:10:00Z,1.1004,1.1002,1,1\n"
        "2024-01-01T00:30:00Z,1.1012,1.1010,4,5\n",
        encoding="utf-8",
    )

    result = import_dukascopy(source)

    assert result.source_kind == "ticks"
    assert len(result.candles) == 2
    first = result.candles[0]
    assert first.open == pytest.approx(1.1001)
    assert first.close == pytest.approx(1.1003)
    assert first.high == pytest.approx(1.1003)
    assert first.low == pytest.approx(1.1001)
    assert first.volume == pytest.approx(7.0)
    assert first.spread == pytest.approx(0.0002)
    assert result.mean_spread == pytest.approx(0.0002)
