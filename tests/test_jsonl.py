"""JSONL layout, bit-string convention and structure encoding.

The round-trip contract is already enforced by the parametrised tests in
`test_exporters.py` now that jsonl declares `structure_round_trip`. What lives here is
everything a round-trip *cannot* detect -- most importantly the bit-string convention,
where reversing on write and again on read is perfectly self-consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qecgen.dataset import StructureLevel
from qecgen.environments import build_single_environment
from qecgen.exporters import get_exporter
from qecgen.exporters.jsonl import MANIFEST_KEY, STRUCTURE_KEY


def _write(tmp_path: Path, level: StructureLevel, shots: int = 40) -> Path:
    dataset = build_single_environment(
        distance=3, p=0.01, shots=shots, seed=0, structure_level=level
    )
    path = tmp_path / "d.jsonl"
    get_exporter("jsonl").write(dataset, path, level)
    return path


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").strip().splitlines()


# --- bit order: absolute assertions, never round-trip ------------------------------


def test_bit_string_index_zero_is_the_first_character(tmp_path: Path) -> None:
    """A round-trip cannot detect a reversal, so this is asserted against the packed form."""
    dataset = build_single_environment(distance=3, p=0.01, shots=8, seed=0)
    path = tmp_path / "d.jsonl"
    get_exporter("jsonl").write(dataset, path)

    unpacked = dataset.unpacked_detectors()
    records = [json.loads(line) for line in _lines(path)[1:]]
    for shot, record in enumerate(records):
        text = record["detectors"]
        assert len(text) == dataset.meta.n_detectors
        for index in range(dataset.meta.n_detectors):
            assert (text[index] == "1") == bool(unpacked[shot, index])


def test_bit_string_repacks_to_the_stored_bytes(tmp_path: Path) -> None:
    """Ties the text form directly to the packed form, little-endian."""
    dataset = build_single_environment(distance=3, p=0.01, shots=8, seed=0)
    path = tmp_path / "d.jsonl"
    get_exporter("jsonl").write(dataset, path)

    for shot, line in enumerate(_lines(path)[1:]):
        text = json.loads(line)["detectors"]
        bits = np.array([c == "1" for c in text], dtype=bool)
        repacked = np.packbits(bits, bitorder="little")
        assert np.array_equal(repacked, dataset.detectors[shot])


def test_observable_string_length_is_n_observables_not_a_byte(tmp_path: Path) -> None:
    """Guards the byte-boundary padding trap: 1 observable gives 1 character, not 8."""
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    record = json.loads(_lines(path)[1])
    assert len(record["observables"]) == 1


# --- line layout -------------------------------------------------------------------


def test_headers_are_the_first_two_lines_and_shots_follow(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.DEM)
    lines = _lines(path)
    assert MANIFEST_KEY in json.loads(lines[0])
    assert STRUCTURE_KEY in json.loads(lines[1])
    assert "shot" in json.loads(lines[2])


def test_no_structure_line_when_level_is_none(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.NONE)
    assert STRUCTURE_KEY not in path.read_text(encoding="utf-8")
    assert get_exporter("jsonl").read(path).structure is None


def test_every_line_is_independently_valid_json(tmp_path: Path) -> None:
    """The streamability property: a reader may parse any line without the others."""
    for line in _lines(_write(tmp_path, StructureLevel.DEM)):
        assert isinstance(json.loads(line), dict)


def test_structure_line_depends_on_distance_not_shot_count(tmp_path: Path) -> None:
    """The size claim, as a test: structure is per-code, not per-shot."""
    small = _lines(_write(tmp_path / "a", StructureLevel.DEM, shots=10))[1]
    large = _lines(_write(tmp_path / "b", StructureLevel.DEM, shots=200))[1]
    assert small == large


def test_structure_is_readable_without_parsing_any_shot(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.DEM, shots=200)
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        payload = json.loads(handle.readline())[STRUCTURE_KEY]
    assert payload["n_mechanisms"] > 0


# --- encoding ----------------------------------------------------------------------


def test_csc_data_values_are_not_stored(tmp_path: Path) -> None:
    """H and L are binary; every stored entry is 1 by construction."""
    path = _write(tmp_path, StructureLevel.DEM)
    payload = json.loads(_lines(path)[1])[STRUCTURE_KEY]
    assert "H_data" not in payload
    assert "L_data" not in payload
    structure = get_exporter("jsonl").read(path).structure
    assert structure is not None
    assert set(np.unique(structure.h.data)) <= {1}


def test_components_are_nested_objects_not_flat_offsets(tmp_path: Path) -> None:
    """JSON has a ragged type; the offset form imports a silent re-segmentation bug."""
    payload = json.loads(_lines(_write(tmp_path, StructureLevel.DEM))[1])[STRUCTURE_KEY]
    assert "component_det_flat" not in payload
    first = payload["components"][0]
    assert set(first) == {
        "parent_mechanism_id",
        "component_index",
        "detectors",
        "observables",
    }


def test_no_nan_or_infinity_tokens_are_written(tmp_path: Path) -> None:
    text = _write(tmp_path, StructureLevel.DEM).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text


def test_priors_round_trip_bit_exactly(tmp_path: Path) -> None:
    """The strongest available float assertion: identical bytes, not approximate equality."""
    dataset = build_single_environment(
        distance=3, p=0.01, shots=20, seed=0, structure_level=StructureLevel.DEM
    )
    path = tmp_path / "d.jsonl"
    get_exporter("jsonl").write(dataset, path, StructureLevel.DEM)
    restored = get_exporter("jsonl").read(path)
    assert restored.structure is not None and dataset.structure is not None
    assert restored.structure.priors.tobytes() == dataset.structure.priors.tobytes()


def test_jsonl_structure_equals_the_hdf5_structure(tmp_path: Path) -> None:
    """Cross-format agreement, not self-agreement.

    A symmetric encode/decode bug round-trips perfectly; comparing against a different
    format's reader is what catches it.
    """
    dataset = build_single_environment(
        distance=3, p=0.01, shots=20, seed=0, structure_level=StructureLevel.DEM
    )
    as_jsonl = tmp_path / "d.jsonl"
    as_hdf5 = tmp_path / "d.h5"
    get_exporter("jsonl").write(dataset, as_jsonl, StructureLevel.DEM)
    get_exporter("hdf5").write(dataset, as_hdf5, StructureLevel.DEM)

    left = get_exporter("jsonl").read(as_jsonl).structure
    right = get_exporter("hdf5").read(as_hdf5).structure
    assert left is not None and right is not None
    assert (left.h != right.h).nnz == 0
    assert (left.l != right.l).nnz == 0
    assert left.priors.tobytes() == right.priors.tobytes()
    assert np.array_equal(left.detector_coords, right.detector_coords, equal_nan=True)
    assert left.components == right.components


# --- refusals ----------------------------------------------------------------------


def test_json_constants_are_rejected_on_read(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    lines = _lines(path)
    lines.insert(1, '{"shot": 99, "detectors": "0", "observables": NaN}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        get_exporter("jsonl").read(path)


def test_manifest_claiming_structure_without_a_structure_line_is_refused(
    tmp_path: Path,
) -> None:
    """The read-side mirror of require_level_agreement."""
    path = _write(tmp_path, StructureLevel.DEM, shots=4)
    lines = _lines(path)
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no __structure__ line"):
        get_exporter("jsonl").read(path)


def test_duplicate_manifest_lines_are_refused(tmp_path: Path) -> None:
    """Catches `cat a.jsonl b.jsonl`."""
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    text = path.read_text(encoding="utf-8")
    path.write_text(text + text, encoding="utf-8")
    with pytest.raises(ValueError, match="second manifest line"):
        get_exporter("jsonl").read(path)


def test_shot_before_the_manifest_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    lines = _lines(path)
    path.write_text("\n".join([lines[1], *lines]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before the manifest"):
        get_exporter("jsonl").read(path)


def test_structure_shape_disagreeing_with_the_manifest_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.DEM, shots=4)
    lines = _lines(path)
    payload = json.loads(lines[1])
    payload[STRUCTURE_KEY]["n_detectors"] += 1
    lines[1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="but the manifest declares"):
        get_exporter("jsonl").read(path)


def test_full_level_leaks_no_circuit_or_dem_text(tmp_path: Path) -> None:
    """The provenance quarantine, enforced by scanning the whole file.

    JSONL writes no provenance at all, which is the strongest form of the split: under
    FROZEN_PRIOR the provenance group holds precisely the distribution the condition
    exists to withhold, and the idiomatic way to read a JSONL file is
    `for line in f: json.loads(line)` -- one loop away from handing it to a decoder.
    Scanning the entire file, not just the manifest, is the right check for a flat format.
    """
    path = _write(tmp_path, StructureLevel.FULL, shots=10)
    text = path.read_text(encoding="utf-8")
    assert "error(" not in text
    assert "QUBIT_COORDS" not in text
    restored = get_exporter("jsonl").read(path)
    assert restored.structure is not None
    assert restored.meta.structure_level is StructureLevel.FULL


def test_files_without_a_structure_line_still_read(tmp_path: Path) -> None:
    """Backward compatibility with files written before __structure__ existed."""
    path = _write(tmp_path, StructureLevel.NONE, shots=6)
    restored = get_exporter("jsonl").read(path)
    assert restored.n_shots == 6
    assert restored.structure is None
