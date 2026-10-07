"""constraints.txt must say exactly what obd_core.NUMERIC_PINS says."""
from pathlib import Path

import obd_core as core


def test_constraints_file_matches_numeric_pins():
    lines = Path(__file__).resolve().parents[1].joinpath("constraints.txt").read_text().splitlines()
    pins = dict(line.split("==") for line in lines if line.strip() and not line.startswith("#"))
    assert pins == core.NUMERIC_PINS


def test_running_environment_matches_pins():
    assert core.pin_mismatches() == {}
