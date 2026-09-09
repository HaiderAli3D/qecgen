"""Calibration histories must preserve timing, units and genuinely distinct estimates."""

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from research.realism.calibration_analysis import (
    Measurement,
    analyse,
    backend_cutoffs,
    normalize_value,
    prepare_measurements,
    score_forecasts,
)

START = datetime(2026, 2, 1, tzinfo=UTC)
KEY = ("fixture", "T1", "T1", 0, None)


def row(day: int, value: float = 100.0, **changes: Any) -> dict[str, Any]:
    result = {
        "backend": "fixture",
        "property_family": "T1",
        "property": "T1",
        "qubit_a": 0,
        "qubit_b": None,
        "value": value,
        "unit": "us",
        "calibrated_time": START + timedelta(days=day),
        "observed_time": START + timedelta(days=day, minutes=1),
        "is_failure_ceiling": False,
        "is_new_measurement": None,
    }
    result.update(changes)
    return result


def test_equal_values_at_distinct_times_are_retained_without_new_flag() -> None:
    series, audit = prepare_measurements([row(0), row(1), row(1)])
    assert len(series[KEY]) == 2
    assert audit["duplicate_identity_same_value"] == 1


def test_conflicting_same_measurement_is_quarantined() -> None:
    series, audit = prepare_measurements([row(0), row(0, 101), row(0), row(1)])
    assert [measurement.calibrated for measurement in series[KEY]] == [START + timedelta(days=1)]
    assert audit["contradictory_duplicate_identities"] == 1


def test_missing_coherence_unit_requires_documented_observation_interval() -> None:
    assert normalize_value("T1", None, 0.0001, START) == (
        100.0,
        "documented_null_seconds_to_microseconds",
    )
    assert normalize_value("T1", None, 0.0001, START + timedelta(days=200)) == (
        None,
        "unresolved_coherence_unit",
    )
    assert normalize_value("T1", "ns", 100, START)[0] is None


@pytest.mark.parametrize("seconds,microseconds", [(0.0001, 100), (0.0001201, 120.1)])
def test_duplicate_after_unit_transition_is_merged_after_normalization(
    seconds: float, microseconds: float
) -> None:
    series, audit = prepare_measurements(
        [
            row(0, seconds, unit=None),
            row(0, microseconds, observed_time=datetime(2026, 6, 1, tzinfo=UTC)),
        ]
    )
    assert len(series[KEY]) == 1
    assert audit["duplicate_identity_same_value"] == 1


def test_roundoff_tolerance_does_not_hide_same_unit_contradiction() -> None:
    series, audit = prepare_measurements([row(0, 120.1), row(0, math.nextafter(120.1, math.inf))])
    assert not series
    assert audit["contradictory_duplicate_identities"] == 1


def test_persistence_never_uses_unobserved_previous_calibration() -> None:
    measurements = [
        Measurement(KEY, START, START + timedelta(minutes=1), 1),
        Measurement(KEY, START + timedelta(days=1), START + timedelta(days=10), 99),
        Measurement(KEY, START + timedelta(days=2), START + timedelta(days=2, minutes=1), 2),
    ]
    result = score_forecasts(measurements, measurements[2:], 5)
    assert result["persistence"]["mae"] == 1
    assert result["training_mean"]["mae"] == 3


def test_online_forecast_updates_only_after_prior_target_is_observed() -> None:
    measurements = [
        Measurement(
            KEY, START + timedelta(days=day), START + timedelta(days=day, minutes=1), float(day)
        )
        for day in range(4)
    ]
    result = score_forecasts(measurements, measurements[1:], 0)
    assert result["persistence"]["mae"] == 1
    assert result["training_mean"]["mae"] == 2


def test_backend_cutoffs_shared_across_series() -> None:
    series, _ = prepare_measurements([row(day, qubit_a=day % 2) for day in range(20)])
    assert backend_cutoffs(series)["fixture"] == (
        START + timedelta(days=11),
        START + timedelta(days=15),
    )


def test_training_mean_excludes_old_but_late_observed_measurement() -> None:
    inputs = [row(day, float(day + 1)) for day in range(30)]
    inputs[0] = row(0, 99999, observed_time=START + timedelta(days=25))
    series, _ = prepare_measurements(inputs)
    summary, records = analyse(series)
    assert summary["historical_rows_unavailable_at_training_cutoff"] == 2
    # Day 17 is calibrated at the boundary but observed one minute after it.
    assert records[0]["training"]["mean"] == pytest.approx(sum(range(2, 18)) / 16)
    assert (
        records[0]["training"]["last_calibrated_utc"]
        < records[0]["validation"]["first_calibrated_utc"]
    )
