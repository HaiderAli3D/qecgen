"""Foreign-file conventions must fail rather than manufacture decoder targets."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import stim

from research.realism.import_real import (
    blocked_split,
    detector_anchors,
    detector_sequence,
    load_google_cohort,
    load_willow_derived,
    parse_01,
    parse_b8,
)


def test_little_endian_and_padding_are_checked() -> None:
    np.testing.assert_array_equal(parse_b8(bytes([1, 1]), 9), [[1, 1]])
    with pytest.raises(ValueError, match="padding"):
        parse_b8(bytes([0, 128]), 9)
    with pytest.raises(ValueError, match="whole"):
        parse_b8(bytes([1]), 9)


def test_observable_text_is_not_coerced() -> None:
    np.testing.assert_array_equal(parse_01(b"0\r\n1\r\n", 1), [[0], [1]])
    for malformed in (b"true\n", b"\n", b"0\n\n1\n", b"01\n"):
        with pytest.raises(ValueError):
            parse_01(malformed, 1)


def test_import_requires_actual_outcomes_and_matching_rows(tmp_path: Path) -> None:
    archive = tmp_path / "synthetic-format-fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "cohort/circuit_noisy.stim",
            "R 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR(0,0,0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        )
        output.writestr("cohort/detection_events.b8", bytes([0, 1]))
        output.writestr("cohort/obs_flips_predicted_by_belief_matching.01", "0\n1\n")
    with pytest.raises(ValueError, match="actual"):
        load_google_cohort(archive, "cohort")
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("cohort/obs_flips_actual.01", "0\n")
    with pytest.raises(ValueError, match="row"):
        load_google_cohort(archive, "cohort")


def test_split_has_disjoint_rows_and_guard_bands() -> None:
    split = blocked_split(1000, guard=10)
    assert split.train[-1] == 589
    assert split.validation[0] == 610
    assert split.validation[-1] == 789
    assert split.test[0] == 810
    assert len(set(split.train) & set(split.validation)) == 0
    assert len(set(split.validation) & set(split.test)) == 0
    with pytest.raises(ValueError):
        blocked_split(10, guard=10)


def test_sequence_uses_coordinate_identity_and_marks_absent_boundaries() -> None:
    circuit = stim.Circuit(
        "R 0\nM 0\nDETECTOR(2,4,0) rec[-1]\nDETECTOR(1,3,1) rec[-1]\nDETECTOR(2,4,1) rec[-1]"
    )
    bits = np.array([[True, False, True]], dtype=bool)
    sequence, layout = detector_sequence(bits, circuit)
    assert sequence.shape == (1, 2, 4)
    assert layout["positions"] == [[1.0, 3.0], [2.0, 4.0]]
    np.testing.assert_array_equal(sequence[0], [[0, 1, 0, 1], [0, 1, 1, 1]])


def test_final_data_measurements_do_not_become_new_stabilizer_identity() -> None:
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR(9,8,10,8,7,10,8,8,9) rec[-1]")
    assert detector_anchors(circuit) == {0: [8.0, 8.0, 10.0]}


def test_mirror_intake_checks_pins_identity_and_source_order(tmp_path: Path) -> None:
    circuit = tmp_path / "synthetic-format-example.stim"
    circuit.write_text("R 0\nM 0\nDETECTOR(0,0,0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    path = tmp_path / "synthetic-format-example.parquet"
    columns = {
        "shot": [0, 1],
        "distance": [3, 3],
        "basis": ["Z", "Z"],
        "rounds": [1, 1],
        "orientation": ["fixture", "fixture"],
        "detectors": [[False], [True]],
        "observable": [False, True],
    }
    pq.write_table(pa.table(columns), path)
    expected = {
        "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "circuit_sha256": hashlib.sha256(circuit.read_bytes()).hexdigest(),
        "distance": 3,
        "basis": "Z",
        "rounds": 1,
        "orientation": "fixture",
    }
    cohort = load_willow_derived(path, circuit, expected)
    np.testing.assert_array_equal(cohort.observables, [[0], [1]])
    assert cohort.source["original_archive_independently_verified"] is False
    with pytest.raises(ValueError, match="expected basis"):
        load_willow_derived(path, circuit, {**expected, "basis": "X"})
    with pytest.raises(ValueError, match="pinned"):
        load_willow_derived(path, circuit, {**expected, "table_sha256": "0" * 64})
    columns["shot"] = [1, 0]
    pq.write_table(pa.table(columns), path)
    expected["table_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="original row order"):
        load_willow_derived(path, circuit, expected)
