"""Describe IBM calibration changes without treating API polls as shot experiments.

The two forecasts answer an intentionally limited question: does the last calibration
already available predict the next calibration better than a frozen training mean?
They neither identify a physical cause nor validate a Google surface-code model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_INPUT = Path("data/realism/raw/calibration-drift/train-00000-of-00001.parquet")
EXPECTED_SHA256 = "fe29c6aba4e09d372a10e787419f7a74589c28d903b3384f970ecbb0b4c2dc65"
FAMILIES = ("T1", "T2", "sx_error", "cz_error", "readout_error")
MISSING_UNIT_START = datetime.fromisoformat("2026-01-31T17:05:03.496046+00:00")
MISSING_UNIT_END = datetime.fromisoformat("2026-05-07T08:27:06.049534+00:00")
UNIT_SOURCE = (
    "https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift/blob/"
    "d7180c868a7d692675123518be4ad8f9a6266c66/2026-05-20-report.md"
)
type SeriesKey = tuple[str, str, str, int, int | None]


@dataclass(frozen=True)
class Measurement:
    key: SeriesKey
    calibrated: datetime
    observed: datetime
    value: float
    normalization: str = ""


def normalize_value(
    family: str, unit: str | None, value: float, observed: datetime
) -> tuple[float | None, str]:
    """Only the documented, pinned historical null-unit interval permits conversion.

    The source report explicitly identifies seconds in that interval. A small number
    by itself is never evidence for seconds, and later unknown units stay unresolved.
    """
    if not math.isfinite(value):
        return None, "nonfinite_value"
    if family in {"T1", "T2"}:
        if value <= 0:
            return None, "nonpositive_coherence"
        if unit == "us":
            return value, "explicit_microseconds"
        if unit is None and MISSING_UNIT_START <= observed <= MISSING_UNIT_END:
            return value * 1_000_000, "documented_null_seconds_to_microseconds"
        return None, "unresolved_coherence_unit"
    if not 0 <= value <= 1:
        return None, "error_outside_probability_range"
    if unit not in {None, ""}:
        return None, "unresolved_error_unit"
    return value, "dimensionless_error"


def prepare_measurements(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[SeriesKey, list[Measurement]], dict[str, int]]:
    """Keep equal values at different times; quarantine contradictory duplicate identities.

    ``is_new_measurement`` is deliberately not consulted: almost five million source
    rows omit it, and an equal estimate at a new calibration time remains an observation.
    """
    identities: dict[tuple[SeriesKey, datetime], Measurement] = {}
    conflicts: set[tuple[SeriesKey, datetime]] = set()
    counts: Counter[str] = Counter(
        {
            "unresolved_coherence_unit": 0,
            "unresolved_error_unit": 0,
            "duplicate_identity_same_value": 0,
            "contradictory_duplicate_identities": 0,
        }
    )
    for row in rows:
        counts["selected_raw_rows"] += 1
        if row["is_failure_ceiling"]:
            counts["excluded_failure_ceiling"] += 1
            continue
        calibrated, observed = row["calibrated_time"], row["observed_time"]
        if not isinstance(calibrated, datetime) or not isinstance(observed, datetime):
            counts["missing_timestamp"] += 1
            continue
        if calibrated.tzinfo is None or observed.tzinfo is None or observed < calibrated:
            counts["invalid_timestamp_order_or_timezone"] += 1
            continue
        value, status = normalize_value(
            row["property_family"], row["unit"], float(row["value"]), observed
        )
        counts[status] += 1
        if value is None:
            continue
        pair = row["qubit_b"]
        second = None if pair is None or math.isnan(float(pair)) else int(pair)
        key: SeriesKey = (
            row["backend"],
            row["property_family"],
            row["property"],
            int(row["qubit_a"]),
            second,
        )
        identity = key, calibrated
        if identity in conflicts:
            counts["additional_conflicting_identity_rows"] += 1
            continue
        old = identities.get(identity)
        if old is None:
            identities[identity] = Measurement(key, calibrated, observed, value, status)
        elif old.value == value or (
            {old.normalization, status}
            == {"explicit_microseconds", "documented_null_seconds_to_microseconds"}
            and abs(old.value - value) <= 2 * max(math.ulp(old.value), math.ulp(value))
        ):
            # Scaling a binary float to microseconds can add a rounding bit. Permit
            # two ULPs only across that documented conversion, never within one unit.
            counts["duplicate_identity_same_value"] += 1
            identities[identity] = replace(old, observed=min(old.observed, observed))
        else:
            # There is no source-backed rule for choosing a corrected value here.
            counts["contradictory_duplicate_identities"] += 1
            conflicts.add(identity)
            del identities[identity]
    series: dict[SeriesKey, list[Measurement]] = defaultdict(list)
    for measurement in identities.values():
        series[measurement.key].append(measurement)
    for measurements in series.values():
        measurements.sort(key=lambda item: item.calibrated)
    counts["retained_measurement_identities"] = sum(map(len, series.values()))
    counts["retained_series"] = len(series)
    return dict(series), dict(sorted(counts.items()))


def backend_cutoffs(
    series: Mapping[SeriesKey, Sequence[Measurement]],
) -> dict[str, tuple[datetime, datetime]]:
    """One backend timeline prevents one qubit's future entering another's training set."""
    times: dict[str, set[datetime]] = defaultdict(set)
    for key, measurements in series.items():
        times[key[0]].update(item.calibrated for item in measurements)
    cutoffs = {}
    for backend, unique in times.items():
        ordered = sorted(unique)
        if len(ordered) < 5:
            continue
        cutoffs[backend] = (
            ordered[int(len(ordered) * 0.6) - 1],
            ordered[int(len(ordered) * 0.8) - 1],
        )
    return cutoffs


def autocorrelation(values: Sequence[float], lag: int) -> float | None:
    """This is Pearson correlation at a measurement-index lag, not an hourly ACF."""
    if len(values) - lag < 3:
        return None
    first, second = np.asarray(values[:-lag]), np.asarray(values[lag:])
    if float(np.std(first)) == 0 or float(np.std(second)) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def describe(measurements: Sequence[Measurement]) -> dict[str, Any]:
    values = [row.value for row in measurements]
    if not values:
        return {"count": 0}
    average = float(np.mean(values))
    gaps = [
        (right.calibrated - left.calibrated).total_seconds()
        for left, right in pairwise(measurements)
    ]
    return {
        "count": len(values),
        "first_calibrated_utc": measurements[0].calibrated.isoformat(),
        "last_calibrated_utc": measurements[-1].calibrated.isoformat(),
        "mean": average,
        "coefficient_of_variation": (
            float(np.std(values, ddof=1)) / average if average > 0 and len(values) > 1 else None
        ),
        "measurement_index_autocorrelation": {
            str(lag): autocorrelation(values, lag) for lag in (1, 2, 5)
        },
        "adjacent_calibration_gap_seconds_median": float(np.median(gaps)) if gaps else None,
    }


def score_forecasts(
    all_measurements: Sequence[Measurement],
    targets: Sequence[Measurement],
    training_mean: float,
) -> dict[str, Any]:
    """Rolling persistence may see earlier held-out outcomes only once they were available.

    This is one-step online forecasting, not a fixed multi-step forecast. Availability
    is checked against each target's calibration time, stricter than its API pull time.
    Both predictors are scored on exactly the same targets; the training mean stays fixed.
    """
    arrivals = sorted(all_measurements, key=lambda row: row.observed)
    index = 0
    latest: Measurement | None = None
    persistence_errors: list[float] = []
    mean_errors: list[float] = []
    prior_ages: list[float] = []
    skipped = 0
    for target in targets:
        while index < len(arrivals) and arrivals[index].observed < target.calibrated:
            candidate = arrivals[index]
            if latest is None or candidate.calibrated > latest.calibrated:
                latest = candidate
            index += 1
        if latest is None:
            skipped += 1
            continue
        persistence_errors.append(latest.value - target.value)
        mean_errors.append(training_mean - target.value)
        prior_ages.append((target.calibrated - latest.calibrated).total_seconds())
    result: dict[str, Any] = {
        "scored_measurements": len(persistence_errors),
        "skipped_without_available_prior": skipped,
    }
    if not persistence_errors:
        return result
    for name, errors in (("persistence", persistence_errors), ("training_mean", mean_errors)):
        result[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        }
    result["prior_calibration_age_seconds_median"] = float(np.median(prior_ages))
    denominator = result["training_mean"]["mae"]
    result["mae_ratio_persistence_to_mean"] = (
        result["persistence"]["mae"] / denominator if denominator > 0 else None
    )
    return result


def quantiles(values: Iterable[float | None]) -> dict[str, float] | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    if not present:
        return None
    return dict(
        zip(
            ("p10", "median", "p90"), map(float, np.quantile(present, [0.1, 0.5, 0.9])), strict=True
        )
    )


def analyse(
    series: Mapping[SeriesKey, Sequence[Measurement]], minimum_training: int = 10
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cutoffs = backend_cutoffs(series)
    records: list[dict[str, Any]] = []
    insufficient = 0
    unavailable_training = 0
    for key, measurements in sorted(series.items(), key=lambda item: repr(item[0])):
        if key[0] not in cutoffs:
            insufficient += 1
            continue
        training_end, validation_end = cutoffs[key[0]]
        training = [
            row
            for row in measurements
            if row.calibrated <= training_end and row.observed <= training_end
        ]
        unavailable_training += sum(
            row.calibrated <= training_end < row.observed for row in measurements
        )
        validation = [
            row for row in measurements if training_end < row.calibrated <= validation_end
        ]
        held_out = [row for row in measurements if row.calibrated > validation_end]
        if len(training) < minimum_training or min(len(validation), len(held_out)) < 3:
            insufficient += 1
            continue
        average = float(np.mean([row.value for row in training]))
        records.append(
            {
                "backend": key[0],
                "family": key[1],
                "property": key[2],
                "qubit_a": key[3],
                "qubit_b": key[4],
                "training": describe(training),
                "validation": describe(validation),
                "held_out": describe(held_out),
                "validation_forecast": score_forecasts(measurements, validation, average),
                "held_out_forecast": score_forecasts(measurements, held_out, average),
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["backend"], record["family"])].append(record)
    groups = []
    for (backend, family), members in sorted(grouped.items()):
        group: dict[str, Any] = {
            "backend": backend,
            "family": family,
            "series_count": len(members),
            "unit": "us" if family in {"T1", "T2"} else "dimensionless",
        }
        training_means = [record["training"]["mean"] for record in members]
        grand_mean = float(np.mean(training_means))
        group["between_series_training_mean_cv"] = (
            float(np.std(training_means, ddof=1)) / grand_mean
            if grand_mean > 0 and len(training_means) > 1
            else None
        )
        for split in ("training", "validation", "held_out"):
            group[split] = {
                "measurements": sum(record[split]["count"] for record in members),
                "within_series_cv_quantiles": quantiles(
                    record[split]["coefficient_of_variation"] for record in members
                ),
                "measurement_index_autocorrelation_quantiles": {
                    str(lag): quantiles(
                        record[split]["measurement_index_autocorrelation"][str(lag)]
                        for record in members
                    )
                    for lag in (1, 2, 5)
                },
            }
        for split in ("validation_forecast", "held_out_forecast"):
            members_scored = [
                record[split] for record in members if record[split]["scored_measurements"]
            ]
            count = sum(record["scored_measurements"] for record in members_scored)
            group[split] = {
                "scored_measurements": count,
                "series_improved_mae": sum(
                    record["persistence"]["mae"] < record["training_mean"]["mae"]
                    for record in members_scored
                ),
                "series_worse_mae": sum(
                    record["persistence"]["mae"] > record["training_mean"]["mae"]
                    for record in members_scored
                ),
                "series_equal_mae": sum(
                    record["persistence"]["mae"] == record["training_mean"]["mae"]
                    for record in members_scored
                ),
            }
            if count:
                for name in ("persistence", "training_mean"):
                    group[split][name] = {
                        "pooled_mae": sum(
                            record[name]["mae"] * record["scored_measurements"]
                            for record in members_scored
                        )
                        / count,
                        "pooled_rmse": math.sqrt(
                            sum(
                                record[name]["rmse"] ** 2 * record["scored_measurements"]
                                for record in members_scored
                            )
                            / count
                        ),
                    }
                group[split]["series_mae_ratio_quantiles"] = quantiles(
                    record["mae_ratio_persistence_to_mean"] for record in members_scored
                )
        groups.append(group)
    return {
        "backend_cutoffs": {
            backend: {
                "training_end_utc": dates[0].isoformat(),
                "validation_end_utc": dates[1].isoformat(),
            }
            for backend, dates in sorted(cutoffs.items())
        },
        "retained_series": len(records),
        "excluded_insufficient_series": insufficient,
        "historical_rows_unavailable_at_training_cutoff": unavailable_training,
        "groups": groups,
    }, records


def parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    # An allowlist makes it impossible to accidentally fit weather or restricted SN data.
    columns = [
        "backend",
        "property_family",
        "property",
        "qubit_a",
        "qubit_b",
        "value",
        "unit",
        "observed_time",
        "calibrated_time",
        "is_failure_ceiling",
    ]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        selected = batch.filter(
            pc.is_in(batch.column("property_family"), value_set=pa.array(FAMILIES))
        )
        yield from selected.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument(
        "--output", type=Path, default=Path("data/realism/results/calibration_drift_summary.json")
    )
    args = parser.parse_args()
    with args.input.open("rb") as source:
        actual_sha = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_sha != args.expected_sha256:
        raise ValueError("Source SHA-256 differs from the pinned analysis input")
    series, audit = prepare_measurements(parquet_rows(args.input))
    summary, records = analyse(series)
    summary.update(
        {
            "schema_version": 1,
            "input": str(args.input),
            "input_sha256": actual_sha,
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "software_versions": {"numpy": np.__version__, "pyarrow": pa.__version__},
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "preprocessing_counts": audit,
            "config": {
                "families": FAMILIES,
                "split": [0.6, 0.2, 0.2],
                "split_unit": "distinct calibration timestamp within backend",
                "minimum_training": 10,
                "minimum_validation_and_held_out": 3,
                "stochastic_operations": False,
                "unit_transition_source": UNIT_SOURCE,
                "documented_missing_unit_observation_start": MISSING_UNIT_START.isoformat(),
                "documented_missing_unit_observation_end": MISSING_UNIT_END.isoformat(),
            },
            "interpretation": [
                "Descriptive IBM calibration forecasting, not decoder transfer or "
                "mechanism validation.",
                "Calibration values and timestamps are API reports, not independent experiments.",
                "Equal values at distinct times are retained; repeated identities merge.",
                "Freshness timestamps may restamp equal API values; report persistence alone "
                "does not establish independently measured physical persistence.",
                "Contradictory identities are quarantined; unknown units stay unresolved.",
                "Device series are not independent; CZ directions may share calibration results.",
                "Autocorrelation lags count measurements, not elapsed time; sampling is irregular.",
                "Persistence is online: earlier held-out outcomes enter only after observation.",
                "Training mean is frozen; no validation or held-out hyperparameter selection.",
                "Weather columns are unread; IBM parameters are not assigned to Google.",
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    series_path = args.output.with_name(args.output.stem + "_series.json")
    summary["series_results"] = str(series_path)
    for path, content in ((args.output, summary), (series_path, records)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    print(json.dumps({"summary": str(args.output), "series": len(records), "audit": audit}))


if __name__ == "__main__":
    main()
