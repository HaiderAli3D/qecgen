"""CSV layout, bit-column convention, provenance quarantine and refusals.

The round-trip contract is already enforced by the parametrised tests in
`test_exporters.py` now that csv is in the registry. What lives here is everything a
round-trip *cannot* detect -- a symmetric encode/decode bug round-trips perfectly -- plus
the refusals, which exist because this is the one format a user is expected to open in a
spreadsheet and save again.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from qecgen.dataset import InMemoryDataset, StructureLevel
from qecgen.environments import build_multi_environment, build_single_environment
from qecgen.exporters import get_exporter
from qecgen.exporters.base import NotAQecgenDatasetError
from qecgen.exporters.csv_table import (
    MAGIC_LINE,
    MANIFEST_KEY,
    PROVENANCE_KEY,
    STRUCTURE_KEY,
    read_manifest_only,
    read_provenance_only,
)


def _write(tmp_path: Path, level: StructureLevel, shots: int = 40) -> Path:
    dataset = build_single_environment(
        distance=3, p=0.01, shots=shots, seed=0, structure_level=level
    )
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path, level)
    return path


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _header_lines(path: Path) -> list[str]:
    return [line for line in _lines(path) if line.startswith("#")]


def _table(path: Path) -> list[list[str]]:
    """The rows an ordinary CSV consumer sees, i.e. with comments filtered."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(line for line in handle if not line.startswith("#"))
        return [row for row in rows if row]


# --- layout and self-description ---------------------------------------------------


def test_first_line_is_the_magic_line(tmp_path: Path) -> None:
    """The magic line is what distinguishes a dataset from a `qecgen sweep` results table."""
    assert _lines(_write(tmp_path, StructureLevel.NONE))[0] == MAGIC_LINE


def test_manifest_is_the_second_line_so_a_listing_reads_two(tmp_path: Path) -> None:
    """Header order is normative: the dataset browser must not have to read the structure
    line, which is 1.5 MB at d=7, to render one row."""
    path = _write(tmp_path, StructureLevel.FULL)
    assert _lines(path)[1].startswith(MANIFEST_KEY + " ")


def test_level_none_writes_the_magic_line_and_the_manifest_and_nothing_else(
    tmp_path: Path,
) -> None:
    headers = _header_lines(_write(tmp_path, StructureLevel.NONE))
    assert len(headers) == 2
    assert headers[0] == MAGIC_LINE
    assert headers[1].startswith(MANIFEST_KEY + " ")


def test_provenance_line_is_written_only_at_full(tmp_path: Path) -> None:
    """CSV is the only text format that carries provenance; `dem` must still not."""
    for level, expected in (
        (StructureLevel.DEM, False),
        (StructureLevel.FULL, True),
    ):
        path = _write(tmp_path / str(level), level)
        keys = [line.split(" ", 1)[0] for line in _header_lines(path)]
        assert (PROVENANCE_KEY in keys) is expected


def test_header_lines_form_a_contiguous_prefix(tmp_path: Path) -> None:
    """Everything after the first non-comment line is data, which is what lets a reader
    stop scanning; a `#` line later in the file is refused rather than skipped."""
    lines = _lines(_write(tmp_path, StructureLevel.FULL))
    first_data = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    assert not any(line.startswith("#") for line in lines[first_data:])


def test_column_header_names_every_bit_position_in_order(tmp_path: Path) -> None:
    dataset = build_single_environment(distance=3, p=0.01, shots=4, seed=0)
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path)
    expected = (
        ["shot"]
        + [f"det_{i}" for i in range(dataset.meta.n_detectors)]
        + [f"obs_{i}" for i in range(dataset.meta.n_observables)]
    )
    assert _table(path)[0] == expected


def test_one_observable_gives_one_column_not_eight(tmp_path: Path) -> None:
    """The byte-boundary padding trap: `observables` is (shots, 1) *bytes*, and emitting a
    column per bit of that byte would invent seven observables."""
    header = _table(_write(tmp_path, StructureLevel.NONE, shots=4))[0]
    assert [c for c in header if c.startswith("obs_")] == ["obs_0"]


def test_environment_id_column_tracks_the_dataset(tmp_path: Path) -> None:
    single = _table(_write(tmp_path / "one", StructureLevel.NONE, shots=4))[0]
    assert "environment_id" not in single

    pooled = build_multi_environment(
        distance=3, error_rates=[0.005, 0.01], shots_per_env=5, seed=1, chunk_size=5
    )
    path = tmp_path / "multi.csv"
    get_exporter("csv").write(pooled, path)
    assert _table(path)[0][1] == "environment_id"


def test_mechanism_columns_appear_only_under_contract_b(tmp_path: Path) -> None:
    plain = _table(_write(tmp_path / "a", StructureLevel.NONE, shots=4))[0]
    assert not any(c.startswith("mech_") for c in plain)

    dataset = build_single_environment(
        distance=3, p=0.01, shots=4, seed=0, emit_mechanisms=True, chunk_size=4
    )
    path = tmp_path / "b.csv"
    get_exporter("csv").write(dataset, path)
    header = _table(path)[0]
    assert sum(c.startswith("mech_") for c in header) == dataset.meta.n_mechanisms


def test_manifest_line_is_the_same_json_the_other_formats_store(tmp_path: Path) -> None:
    dataset = build_single_environment(distance=3, p=0.01, shots=4, seed=0)
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path)
    payload = _lines(path)[1].split(" ", 1)[1]
    assert json.loads(payload) == dataset.meta.to_json_dict()


# --- bit convention: absolute assertions, never round-trip -------------------------


def test_column_det_i_is_bit_i_of_the_packed_row(tmp_path: Path) -> None:
    """A round-trip cannot detect a reversal, so this is asserted against the packed form."""
    dataset = build_single_environment(distance=3, p=0.01, shots=8, seed=0)
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path)

    unpacked = dataset.unpacked_detectors()
    rows = _table(path)[1:]
    for shot, row in enumerate(rows):
        for index in range(dataset.meta.n_detectors):
            assert (row[1 + index] == "1") == bool(unpacked[shot, index])


def test_a_row_repacks_to_the_stored_bytes(tmp_path: Path) -> None:
    """Ties the column form directly to the packed form, little-endian."""
    dataset = build_single_environment(distance=3, p=0.01, shots=8, seed=0)
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path)

    n = dataset.meta.n_detectors
    for shot, row in enumerate(_table(path)[1:]):
        bits = np.array([c == "1" for c in row[1 : 1 + n]], dtype=bool)
        assert np.array_equal(np.packbits(bits, bitorder="little"), dataset.detectors[shot])


def test_shot_column_counts_from_zero_in_file_order(tmp_path: Path) -> None:
    rows = _table(_write(tmp_path, StructureLevel.NONE, shots=6))[1:]
    assert [row[0] for row in rows] == [str(i) for i in range(6)]


# --- encoding and cross-format agreement -------------------------------------------


def test_csv_structure_equals_the_jsonl_structure(tmp_path: Path) -> None:
    """Both read the normative encoding out of `structure_json`; this is what keeps that
    module's single-owner purpose enforceable rather than aspirational."""
    dataset = build_single_environment(
        distance=3, p=0.01, shots=4, seed=0, structure_level=StructureLevel.DEM
    )
    as_csv = tmp_path / "d.csv"
    as_jsonl = tmp_path / "d.jsonl"
    get_exporter("csv").write(dataset, as_csv, StructureLevel.DEM)
    get_exporter("jsonl").write(dataset, as_jsonl, StructureLevel.DEM)

    csv_payload = json.loads(_lines(as_csv)[2].split(" ", 1)[1])
    jsonl_payload = json.loads(Path(as_jsonl).read_text(encoding="utf-8").splitlines()[1])
    assert csv_payload == jsonl_payload["__structure__"]


def test_csv_structure_equals_the_hdf5_structure(tmp_path: Path) -> None:
    """A symmetric encode/decode bug round-trips perfectly; only a different reader
    catches it."""
    dataset = build_single_environment(
        distance=3, p=0.01, shots=4, seed=0, structure_level=StructureLevel.DEM
    )
    as_csv, as_hdf5 = tmp_path / "d.csv", tmp_path / "d.h5"
    get_exporter("csv").write(dataset, as_csv, StructureLevel.DEM)
    get_exporter("hdf5").write(dataset, as_hdf5, StructureLevel.DEM)

    left = get_exporter("csv").read(as_csv).structure
    right = get_exporter("hdf5").read(as_hdf5).structure
    assert left is not None and right is not None
    assert np.array_equal(left.h.toarray(), right.h.toarray())
    assert np.array_equal(left.l.toarray(), right.l.toarray())
    assert np.array_equal(left.priors, right.priors, equal_nan=True)
    assert left.components == right.components


def test_priors_round_trip_bit_exactly(tmp_path: Path) -> None:
    """`sweep.write_csv` formats its floats at %.8g; copying that here would truncate the
    priors, which are the whole content of the frozen-prior condition."""
    dataset = build_single_environment(
        distance=3, p=0.01, shots=4, seed=0, structure_level=StructureLevel.DEM
    )
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path, StructureLevel.DEM)
    restored = get_exporter("csv").read(path)
    assert dataset.structure is not None and restored.structure is not None
    assert np.array_equal(restored.structure.priors, dataset.structure.priors)


def test_no_nan_or_infinity_tokens_are_written(tmp_path: Path) -> None:
    text = _write(tmp_path, StructureLevel.FULL).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text


def test_the_file_uses_bare_line_feeds_and_no_byte_order_mark(tmp_path: Path) -> None:
    """Without `newline=""` the csv module's terminator is translated a second time on
    Windows and every row ends `\\r\\r\\n`, which some parsers read as a blank row between
    every shot."""
    raw = _write(tmp_path, StructureLevel.DEM).read_bytes()
    assert b"\r" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")


# --- provenance quarantine ---------------------------------------------------------


def test_circuit_text_appears_only_on_the_provenance_line(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.FULL, shots=10)
    for line in _lines(path):
        if line.startswith(PROVENANCE_KEY):
            assert "QUBIT_COORDS" in line
        else:
            assert "QUBIT_COORDS" not in line
            assert "error(" not in line


def test_dropping_comment_lines_yields_a_table_with_no_provenance(tmp_path: Path) -> None:
    """`pandas.read_csv(path, comment="#")` is the reason CSV can carry provenance where
    JSONL will not: the step that finds the table is the step that drops the text."""
    path = _write(tmp_path, StructureLevel.FULL, shots=10)
    rows = _table(path)
    assert len(rows) == 11
    assert not any("QUBIT_COORDS" in cell for row in rows for cell in row)


def test_read_does_not_restore_circuit_or_dem_text(tmp_path: Path) -> None:
    """Re-attaching it would hand a frozen-prior test file's own DEM to anything that
    merely called `read()`. HDF5 behaves the same way."""
    restored = get_exporter("csv").read(_write(tmp_path, StructureLevel.FULL))
    assert restored.meta.environments[0].circuit == ""
    assert restored.meta.environments[0].dem == ""


def test_read_provenance_only_returns_the_block(tmp_path: Path) -> None:
    payload = read_provenance_only(_write(tmp_path, StructureLevel.FULL))
    assert payload is not None
    assert "QUBIT_COORDS" in payload["environments"][0]["circuit"]
    assert read_provenance_only(_write(tmp_path / "dem", StructureLevel.DEM)) is None


def test_a_provenance_stripped_file_still_reads(tmp_path: Path) -> None:
    """Deliberately asymmetric with the missing-structure rule. Deleting the provenance
    line is the one edit to a qecgen file that can only *reduce* leakage, and nothing
    decoder-visible is lost, so refusing it would push people toward keeping it."""
    path = _write(tmp_path, StructureLevel.FULL, shots=6)
    kept = [line for line in _lines(path) if not line.startswith(PROVENANCE_KEY)]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    restored = get_exporter("csv").read(path)
    assert restored.n_shots == 6
    assert restored.meta.structure_level is StructureLevel.FULL


def test_read_manifest_only_never_reads_a_shot_row(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.FULL, shots=40)
    assert read_manifest_only(path)["shots"] == 40


# --- refusals ----------------------------------------------------------------------


def test_a_sweep_results_csv_is_refused_by_name(tmp_path: Path) -> None:
    """`.csv` is both a dataset extension and the extension `qecgen sweep` writes its
    threshold table with, so this mistake is one keystroke away."""
    from qecgen.sweep import write_csv

    path = tmp_path / "sweep.csv"
    write_csv([], path)
    with pytest.raises(NotAQecgenDatasetError, match="qecgen sweep"):
        get_exporter("csv").read(path)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(NotAQecgenDatasetError, match="is empty"):
        get_exporter("csv").read(path)


def test_an_unknown_version_is_refused(tmp_path: Path) -> None:
    """A v2 that adds a required header line must not be read as a v1 with an unknown
    line ignored."""
    path = _write(tmp_path, StructureLevel.NONE)
    lines = _lines(path)
    lines[0] = "#qecgen-csv v2"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(NotAQecgenDatasetError, match="does not begin with"):
        get_exporter("csv").read(path)


def test_an_unknown_header_key_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.NONE)
    lines = _lines(path)
    lines.insert(2, '#__future__ {"required": true}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown header key"):
        get_exporter("csv").read(path)


def test_json_constants_are_rejected_on_read(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.DEM)
    lines = _lines(path)
    lines[2] = lines[2].replace('"priors": [', '"priors": [NaN, ', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        get_exporter("csv").read(path)


def test_manifest_claiming_structure_without_a_structure_line_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.DEM)
    kept = [line for line in _lines(path) if not line.startswith(STRUCTURE_KEY)]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="indistinguishable from a corrupt file"):
        get_exporter("csv").read(path)


def test_duplicate_manifest_lines_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.NONE)
    lines = _lines(path)
    lines.insert(2, lines[1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="files were concatenated"):
        get_exporter("csv").read(path)


def test_a_header_line_after_the_table_is_refused(tmp_path: Path) -> None:
    """`cat a.csv b.csv`, and "append the new shots to the end"."""
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    path.write_text(path.read_text(encoding="utf-8") + MAGIC_LINE + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header line appears after the table"):
        get_exporter("csv").read(path)


def test_reordered_rows_are_refused(tmp_path: Path) -> None:
    """The trap this format invites: sorting rows in a spreadsheet severs the per-shot
    correspondence between detectors, environment_id and mechanism labels while leaving a
    perfectly well-formed file behind."""
    path = _write(tmp_path, StructureLevel.NONE, shots=6)
    lines = _lines(path)
    lines[3], lines[4] = lines[4], lines[3]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Rows must stay in the order"):
        get_exporter("csv").read(path)


@pytest.mark.parametrize("replacement", ["TRUE", "", "1.0", "yes"])
def test_a_non_binary_cell_is_refused(tmp_path: Path, replacement: str) -> None:
    """A spreadsheet writes TRUE/FALSE for a boolean-formatted column, and folding an
    empty cell to 0 would invent a shot with no detection events."""
    path = _write(tmp_path / replacement, StructureLevel.NONE, shots=4)
    lines = _lines(path)
    cells = lines[3].split(",")
    cells[1] = replacement
    lines[3] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="neither '0' nor '1'"):
        get_exporter("csv").read(path)


def test_a_short_row_is_refused(tmp_path: Path) -> None:
    """A spreadsheet that dropped trailing empty columns produces exactly this."""
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    lines = _lines(path)
    lines[3] = ",".join(lines[3].split(",")[:-1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields but the header declares"):
        get_exporter("csv").read(path)


def test_a_column_header_disagreeing_with_the_manifest_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, StructureLevel.NONE, shots=4)
    lines = _lines(path)
    header = lines[2].split(",")
    lines[2] = ",".join(header[:-1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column header disagrees with the manifest"):
        get_exporter("csv").read(path)


def test_writing_mechanisms_without_a_mechanism_count_is_refused(tmp_path: Path) -> None:
    """A zero-width block of mechanism columns cannot be told apart from an absent one, so
    the array would read back as None and the content hash would change."""
    dataset = build_single_environment(
        distance=3, p=0.01, shots=4, seed=0, emit_mechanisms=True, chunk_size=4
    )
    broken = dataclasses.replace(dataset.meta, n_mechanisms=None)
    with pytest.raises(ValueError, match="zero-width column block"):
        get_exporter("csv").write(
            InMemoryDataset(
                detectors=dataset.detectors,
                observables=dataset.observables,
                meta=broken,
                mechanisms=dataset.mechanisms,
            ),
            tmp_path / "d.csv",
        )


def test_nonzero_padding_bits_are_refused(tmp_path: Path) -> None:
    """Unpack-then-repack is byte-exact only when the padding bits are zero. Stim zeroes
    them; a hand-built array does not, and the difference never shows in the CSV -- it
    surfaces later as a content_hash that no longer matches."""
    dataset = build_single_environment(distance=3, p=0.01, shots=4, seed=0, rotated=False, rounds=3)
    assert dataset.meta.n_detectors % 8 != 0
    corrupted = dataset.detectors.copy()
    corrupted[0, -1] |= 0b1000_0000
    with pytest.raises(ValueError, match="bits set past the declared width"):
        get_exporter("csv").write(
            InMemoryDataset(
                detectors=corrupted, observables=dataset.observables, meta=dataset.meta
            ),
            tmp_path / "d.csv",
        )


# --- degenerate shapes and the size warning ----------------------------------------


def test_zero_shots_writes_only_the_header_and_reads_back_empty(tmp_path: Path) -> None:
    dataset = build_single_environment(distance=3, p=0.01, shots=0, seed=0, chunk_size=10)
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path)
    assert len(_table(path)) == 1
    restored = get_exporter("csv").read(path)
    assert restored.n_shots == 0
    assert restored.detectors.shape == dataset.detectors.shape


def test_a_zero_shot_pooled_dataset_keeps_its_environment_ids_array(tmp_path: Path) -> None:
    """Presence comes from the column header, not from whether any row carried the field.
    Inferring it from row content turns an empty int32 array into None, which is a
    different content hash."""
    dataset = build_multi_environment(
        distance=3, error_rates=[0.005, 0.01], shots_per_env=0, seed=1, chunk_size=10
    )
    assert dataset.environment_ids is not None
    path = tmp_path / "d.csv"
    get_exporter("csv").write(dataset, path)
    restored = get_exporter("csv").read(path)
    assert restored.environment_ids is not None
    assert restored.compute_content_hash() == dataset.compute_content_hash()


def test_csv_warns_above_the_size_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("qecgen.exporters.csv_table.CSV_SIZE_WARNING_THRESHOLD", 2)
    dataset = build_single_environment(distance=3, p=0.01, shots=4, seed=0)
    with pytest.warns(UserWarning, match="shots as CSV"):
        get_exporter("csv").write(dataset, tmp_path / "d.csv")
