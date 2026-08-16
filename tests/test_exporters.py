"""Round-trip every registered exporter.

No warnings are suppressed anywhere in this file, deliberately: the project runs with
``filterwarnings = ["error::DeprecationWarning"]`` precisely so a dependency bump that
deprecates an exporter code path fails HERE, in the third-party-heavy tests, rather
than surfacing in production. A blanket ``simplefilter("ignore")`` around write/read
used to defeat that policy in exactly the code it was written for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qecgen.dataset import DriftCondition, InMemoryDataset, StructureLevel
from qecgen.environments import DriftAxis, build_multi_environment, build_single_environment
from qecgen.exporters import EXPORTERS, StreamingHDF5Writer, get_exporter
from qecgen.exporters.base import Exporter
from qecgen.validate import validate_dataset

STRUCTURE_PRESERVING = {name for name, e in EXPORTERS.items() if e.structure_round_trip}
"""Derived from the registry rather than hardcoded.

A literal set here is a second source of truth that starts lying the moment a format
gains or loses structure round-tripping.
"""


def _dataset_at(level: StructureLevel) -> InMemoryDataset:
    """Build the shared fixture at a given structure level.

    Datasets must be built at the level they are written at: the exporter now refuses
    a structure_level that disagrees with the manifest, because a file describing
    itself incorrectly is indistinguishable downstream from a corrupt one.
    """
    return build_multi_environment(
        distance=3,
        error_rates=[0.005, 0.01],
        shots_per_env=120,
        seed=11,
        chunk_size=64,
        emit_mechanisms=True,
        structure_level=level,
    )


@pytest.fixture(scope="module")
def dataset() -> InMemoryDataset:
    return build_multi_environment(
        distance=3,
        error_rates=[0.005, 0.01],
        shots_per_env=120,
        seed=11,
        chunk_size=64,
        emit_mechanisms=True,
        structure_level=StructureLevel.DEM,
    )


@pytest.mark.parametrize("name", sorted(EXPORTERS))
def test_roundtrip_arrays_and_manifest(name: str, dataset: InMemoryDataset, tmp_path: Path) -> None:
    exporter = get_exporter(name)
    path = tmp_path / f"rt{exporter.extension}"
    exporter.write(dataset, path, StructureLevel.DEM)
    restored = exporter.read(path)

    assert np.array_equal(dataset.detectors, restored.detectors)
    assert np.array_equal(dataset.observables, restored.observables)
    assert dataset.environment_ids is not None
    assert restored.environment_ids is not None
    assert np.array_equal(dataset.environment_ids, restored.environment_ids)
    assert dataset.mechanisms is not None
    assert restored.mechanisms is not None
    assert np.array_equal(dataset.mechanisms, restored.mechanisms)

    original = dataset.meta.to_json_dict()
    round_tripped = restored.meta.to_json_dict()
    # Formats that cannot rebuild structure downgrade the level they record, so that
    # the manifest never claims more than the payload holds. Every other field must
    # survive untouched.
    expected_level = StructureLevel.DEM if exporter.structure_round_trip else StructureLevel.NONE
    assert restored.meta.structure_level is expected_level
    original.pop("structure_level")
    round_tripped.pop("structure_level")
    assert original == round_tripped


@pytest.mark.parametrize("name", sorted(EXPORTERS))
def test_content_hash_survives_roundtrip(
    name: str, dataset: InMemoryDataset, tmp_path: Path
) -> None:
    exporter = get_exporter(name)
    path = tmp_path / f"h{exporter.extension}"
    exporter.write(dataset, path, StructureLevel.DEM)
    restored = exporter.read(path)
    assert restored.compute_content_hash() == dataset.meta.content_hash


@pytest.mark.parametrize("name", sorted(STRUCTURE_PRESERVING))
def test_structure_roundtrip(name: str, dataset: InMemoryDataset, tmp_path: Path) -> None:
    exporter = get_exporter(name)
    path = tmp_path / f"s{exporter.extension}"
    exporter.write(dataset, path, StructureLevel.DEM)
    restored = exporter.read(path)

    assert dataset.structure is not None
    assert restored.structure is not None
    assert restored.structure.n_mechanisms == dataset.structure.n_mechanisms
    assert np.array_equal(restored.structure.h.toarray(), dataset.structure.h.toarray())
    assert np.array_equal(restored.structure.l.toarray(), dataset.structure.l.toarray())
    # array_equal, not allclose. The stated contract is exact equality, and allclose is
    # precisely the assertion that would pass on a subtly lossy encoder -- a 1e-12 float
    # drift is invisible to it. equal_nan also fixes a latent bug: allclose returns False
    # for NaN vs NaN, so this would have failed on any ragged-coordinate circuit.
    assert np.array_equal(restored.structure.priors, dataset.structure.priors, equal_nan=True)
    assert restored.structure.components == dataset.structure.components
    assert np.array_equal(
        restored.structure.detector_coords, dataset.structure.detector_coords, equal_nan=True
    )


@pytest.mark.parametrize("name", sorted(STRUCTURE_PRESERVING))
def test_coords_only_level_omits_matrices(name: str, tmp_path: Path) -> None:
    exporter = get_exporter(name)
    path = tmp_path / f"c{exporter.extension}"
    dataset = _dataset_at(StructureLevel.COORDS)
    exporter.write(dataset, path, StructureLevel.COORDS)
    restored = exporter.read(path)
    assert restored.structure is not None
    assert restored.structure.detector_coords.shape[0] == restored.meta.n_detectors
    assert restored.structure.h.nnz == 0


@pytest.mark.parametrize("name", sorted(EXPORTERS))
def test_structure_none_writes_no_structure(name: str, tmp_path: Path) -> None:
    exporter = get_exporter(name)
    path = tmp_path / f"n{exporter.extension}"
    dataset = _dataset_at(StructureLevel.NONE)
    exporter.write(dataset, path, StructureLevel.NONE)
    restored = exporter.read(path)
    assert restored.structure is None


@pytest.mark.parametrize("name", sorted(EXPORTERS))
def test_restored_dataset_validates(name: str, dataset: InMemoryDataset, tmp_path: Path) -> None:
    exporter = get_exporter(name)
    path = tmp_path / f"v{exporter.extension}"
    exporter.write(dataset, path, StructureLevel.DEM)
    restored = exporter.read(path)
    report = validate_dataset(restored)
    assert report.ok, str(report)


@pytest.mark.parametrize("name", sorted(EXPORTERS))
def test_full_level_either_carries_provenance_or_does_not_claim_it(
    name: str, tmp_path: Path
) -> None:
    """`full` must mean the same thing in every format, or say it does not.

    Nothing exercised FULL across the registry before -- every other test here writes at
    DEM, COORDS or NONE -- which is how three formats came to mean three different things
    by it. HDF5 and NPZ wrote the circuit and DEM text; Parquet wrote neither; JSONL wrote
    neither *and still recorded* `structure_level: full`, an over-claim in the one field a
    reader has no way to check against the file.

    Whichever branch a format takes, `read()` must never hand the text back: it is
    provenance, and under FROZEN_PRIOR it is the distribution the condition withholds.
    """
    exporter = get_exporter(name)
    path = tmp_path / f"f{exporter.extension}"
    dataset = _dataset_at(StructureLevel.FULL)
    exporter.write(dataset, path, StructureLevel.FULL)
    restored = exporter.read(path)

    if exporter.carries_provenance:
        assert restored.meta.structure_level is StructureLevel.FULL
    else:
        assert restored.meta.structure_level is not StructureLevel.FULL
        assert b"QUBIT_COORDS" not in path.read_bytes()
    assert restored.meta.environments[0].circuit == ""
    assert restored.meta.environments[0].dem == ""


def test_all_exporters_satisfy_the_protocol() -> None:
    for exporter in EXPORTERS.values():
        assert isinstance(exporter, Exporter)


def test_registry_names_match_format_name() -> None:
    for name, exporter in EXPORTERS.items():
        assert name == exporter.format_name


def test_unknown_format_lists_valid_options() -> None:
    with pytest.raises(ValueError, match="unknown format"):
        get_exporter("nexus")


def test_no_nexus_exporter_is_registered() -> None:
    """The Nexus format is unknown. Nothing here may imply otherwise."""
    assert "nexus" not in EXPORTERS


def test_jsonl_warns_above_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import qecgen.exporters.jsonl as jsonl_module

    monkeypatch.setattr(jsonl_module, "JSONL_SIZE_WARNING_THRESHOLD", 10)
    small = build_single_environment(distance=3, p=0.01, shots=40, seed=1, chunk_size=40)
    with pytest.warns(UserWarning, match="JSONL"):
        jsonl_module.JSONLExporter().write(small, tmp_path / "big.jsonl")


class TestStreamingWriter:
    def test_streamed_file_matches_materialised(self, tmp_path: Path) -> None:
        dataset = build_single_environment(distance=3, p=0.008, shots=300, seed=5, chunk_size=64)
        path = tmp_path / "stream.h5"
        writer = StreamingHDF5Writer(path, dataset.meta.n_detectors, dataset.meta.n_observables)
        step = 64
        for start in range(0, dataset.n_shots, step):
            writer.append(
                dataset.detectors[start : start + step],
                dataset.observables[start : start + step],
            )
        assert writer.rows_written == dataset.n_shots
        writer.close(dataset.meta)

        restored = get_exporter("hdf5").read(path)
        assert np.array_equal(restored.detectors, dataset.detectors)
        assert np.array_equal(restored.observables, dataset.observables)
        assert restored.compute_content_hash() == dataset.meta.content_hash

    def test_append_rejects_mismatched_rows(self, tmp_path: Path) -> None:
        writer = StreamingHDF5Writer(tmp_path / "bad.h5", 24, 1)
        with pytest.raises(ValueError, match="correspondence"):
            writer.append(np.zeros((4, 3), np.uint8), np.zeros((3, 1), np.uint8))


def test_drift_condition_is_recorded_in_the_file(tmp_path: Path) -> None:
    from qecgen.environments import build_drift_environments

    datasets = build_drift_environments(
        distance=3,
        train_p=0.005,
        test_values=[0.012],
        shots=64,
        seed=3,
        condition=DriftCondition.FROZEN_PRIOR,
        axis=DriftAxis.P,
        chunk_size=64,
    )
    exporter = get_exporter("hdf5")
    path = tmp_path / "test.h5"
    exporter.write(datasets[1], path, StructureLevel.DEM)
    restored = exporter.read(path)
    assert restored.meta.drift_condition is DriftCondition.FROZEN_PRIOR
    assert restored.meta.structure_source_environment_id == 0
