"""End-to-end and statistical checks. Slow by design; opt in with -m slow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymatching
import pytest

from qecgen.circuits import NoiseModel, build_circuit
from qecgen.dataset import StructureLevel
from qecgen.environments import build_single_environment
from qecgen.exporters import get_exporter
from qecgen.qa import (
    clopper_pearson,
    estimate_logical_error_rate,
    intervals_increasing,
)
from qecgen.sampling import unpack_bits


class TestClopperPearson:
    def test_contains_point_estimate(self) -> None:
        interval = clopper_pearson(30, 1000)
        assert interval.low < interval.point < interval.high

    def test_zero_successes_has_zero_lower_bound(self) -> None:
        interval = clopper_pearson(0, 500)
        assert interval.low == 0.0
        assert interval.high > 0.0

    def test_all_successes_has_unit_upper_bound(self) -> None:
        interval = clopper_pearson(500, 500)
        assert interval.high == 1.0
        assert interval.low < 1.0

    def test_never_extends_below_zero(self) -> None:
        """A Wald interval would; this is why the exact form is used."""
        assert clopper_pearson(1, 100_000).low >= 0.0

    def test_narrows_with_more_trials(self) -> None:
        narrow = clopper_pearson(1000, 100_000)
        wide = clopper_pearson(10, 1000)
        assert (narrow.high - narrow.low) < (wide.high - wide.low)

    def test_rejects_impossible_counts(self) -> None:
        with pytest.raises(ValueError):
            clopper_pearson(10, 5)
        with pytest.raises(ValueError):
            clopper_pearson(1, 0)

    def test_intervals_increasing_requires_separation(self) -> None:
        a = clopper_pearson(10, 10_000)
        b = clopper_pearson(500, 10_000)
        assert intervals_increasing([a, b])
        assert not intervals_increasing([b, a])

    def test_overlapping_intervals_are_not_increasing(self) -> None:
        """Honest answer when two points are statistically indistinguishable."""
        a = clopper_pearson(100, 1000)
        b = clopper_pearson(105, 1000)
        assert not intervals_increasing([a, b])


@pytest.mark.slow
def test_end_to_end_d3_p001_through_pymatching() -> None:
    """d=3, p=0.01, 20k shots must land in a plausible interval."""
    dataset = build_single_environment(
        distance=3,
        p=0.01,
        shots=20_000,
        seed=1234,
        chunk_size=5000,
        structure_level=StructureLevel.DEM,
    )
    import stim

    dem = stim.DetectorErrorModel(dataset.meta.environments[0].dem)
    matching = pymatching.Matching.from_detector_error_model(dem)

    predictions = matching.decode_batch(
        dataset.detectors, bit_packed_shots=True, bit_packed_predictions=True
    )
    actual = dataset.unpacked_observables()
    predicted = unpack_bits(predictions, dataset.meta.n_observables)
    errors = int(np.any(predicted != actual, axis=1).sum())

    interval = clopper_pearson(errors, dataset.n_shots)
    # Broad plausibility band, not a threshold assertion: d=3 at p=0.01 under
    # uniform circuit-level noise sits well inside this.
    assert 0.01 < interval.point < 0.25, f"implausible logical error rate {interval}"
    assert interval.low <= interval.point <= interval.high


@pytest.mark.slow
def test_logical_error_rate_rises_with_p() -> None:
    estimates = [
        estimate_logical_error_rate(
            distance=3, p=p, seed=7, max_shots=40_000, target_errors=300, chunk_size=10_000
        )
        for p in (0.003, 0.008, 0.02)
    ]
    assert intervals_increasing([e.interval for e in estimates]), [str(e) for e in estimates]


@pytest.mark.slow
def test_detection_event_rate_rises_with_p() -> None:
    from qecgen.qa import detection_event_rate

    rates = [
        detection_event_rate(build_circuit(3, p)[0], shots=4000, seed=11)
        for p in (0.001, 0.005, 0.015)
    ]
    assert rates == sorted(rates), rates


@pytest.mark.slow
def test_distance_helps_at_low_p() -> None:
    """Below threshold, larger distance must reduce the logical error rate."""
    estimates = {
        d: estimate_logical_error_rate(
            distance=d,
            p=0.002,
            seed=5,
            max_shots=200_000,
            target_errors=60,
            chunk_size=25_000,
        )
        for d in (3, 5)
    }
    d3, d5 = estimates[3].interval, estimates[5].interval
    assert d5.strictly_below(d3), f"d=3 {d3} vs d=5 {d5}"


@pytest.mark.slow
def test_full_pipeline_generate_export_validate(tmp_path: Path) -> None:
    from qecgen.validate import validate_dataset

    dataset = build_single_environment(
        distance=5,
        p=0.004,
        shots=5000,
        seed=99,
        chunk_size=1000,
        noise_model=NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
        structure_level=StructureLevel.FULL,
    )
    exporter = get_exporter("hdf5")
    path = tmp_path / "pipeline.h5"
    exporter.write(dataset, path, StructureLevel.FULL)
    restored = exporter.read(path)

    report = validate_dataset(restored)
    assert report.ok, str(report)
    assert restored.compute_content_hash() == dataset.meta.content_hash

    # Circuit and DEM text are provenance, not manifest fields: they live in a separate
    # group so a decoder reading the manifest cannot recover a frozen-prior test set's
    # own error model. At FULL they must still be recoverable for auditing.
    import h5py

    with h5py.File(path, "r") as handle:
        manifest = str(handle.attrs["manifest"])
        provenance = json.loads(str(handle["provenance"].attrs["environments"]))
    # Substring-match on real circuit content, not the word "circuit" -- the noise
    # model name "stim_uniform_circuit_level" contains it legitimately.
    circuit_text = provenance["environments"][0]["circuit"]
    body = [ln.strip() for ln in circuit_text.splitlines() if ln.strip().startswith("CX")]
    assert body, "expected the circuit text to contain gate instructions"
    for line in body[:5]:
        assert line not in manifest
    assert provenance["environments"][0]["circuit"].strip()
    assert provenance["environments"][0]["dem"].strip()
