"""One header row, metadata in sidecars, and a reader that needs no arguments.

The format exists because ``pandas.read_csv(path)`` on a ``csv`` dataset returns a single
garbage column named ``#qecgen-csv v1``. These tests pin the thing that fixes that, and
the refusals that stop a one-header-row file being mistaken for any other CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qecgen.dataset import InMemoryDataset, StructureLevel
from qecgen.environments import build_multi_environment, build_single_environment
from qecgen.exporters import NotAQecgenDatasetError, get_exporter, read_manifest
from qecgen.exporters.ml_csv import ML_CSV_SPELLING, MLCSVExporter


def _dataset(**kwargs: object) -> InMemoryDataset:
    params: dict[str, object] = {
        "distance": 3,
        "p": 0.008,
        "shots": 32,
        "seed": 1,
        "chunk_size": 32,
    }
    params.update(kwargs)
    return build_single_environment(**params)  # type: ignore[arg-type]


def _write(dataset: InMemoryDataset, path: Path, level: StructureLevel) -> Path:
    MLCSVExporter().write(dataset, path, level)
    return path


def _sidecar(path: Path) -> dict[str, Any]:
    payload = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def written(tmp_path: Path) -> Path:
    return _write(_dataset(), tmp_path / "d.ml.csv", StructureLevel.NONE)


def test_the_table_has_one_header_row_and_no_comment_lines(written: Path) -> None:
    """The whole point of the format. A `#` line here is the defect it was built to end."""
    lines = written.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("#") for line in lines)
    rows = list(csv.reader(lines))
    assert len(rows) == 1 + 32
    assert len({len(row) for row in rows}) == 1, "ragged rows"


def test_a_naive_reader_with_no_arguments_recovers_the_arrays(written: Path) -> None:
    """`csv.DictReader(open(path))` -- no options, as an ML consumer would.

    On a `csv` dataset the same call yields one column named `#qecgen-csv v1`.
    """
    with written.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert rows[0]["shot"] == "0"
    assert "detector_00" in rows[0]
    assert "target" in rows[0]

    restored = MLCSVExporter().read(written)
    naive = np.array(
        [[int(row[f"detector_{i:02d}"]) for i in range(24)] for row in rows], dtype=bool
    )
    assert np.array_equal(naive, restored.unpacked_detectors())


def test_column_names_are_zero_padded_to_this_files_own_width(tmp_path: Path) -> None:
    """The pad width is a property of the file, which is why the names are published.

    24 detectors need two digits and 120 need three, so a consumer that reproduced the
    width from a remembered example would be wrong at the other distance.
    """
    small = _sidecar(_write(_dataset(), tmp_path / "s.ml.csv", StructureLevel.NONE))
    large = _sidecar(
        _write(_dataset(distance=5, rounds=5), tmp_path / "l.ml.csv", StructureLevel.NONE)
    )
    assert small["columns"]["feature_columns"][0] == "detector_00"
    assert large["columns"]["feature_columns"][0] == "detector_000"


def test_sorting_the_columns_cannot_permute_the_feature_matrix(written: Path) -> None:
    """The reason for padding at all: `sorted(df.columns)` must keep bit order."""
    features = _sidecar(written)["columns"]["feature_columns"]
    assert sorted(features) == features


def test_the_sidecar_names_every_column_so_no_consumer_builds_one(written: Path) -> None:
    columns = _sidecar(written)["columns"]
    header = next(csv.reader(written.read_text(encoding="utf-8").splitlines()))
    # Spelled out rather than derived from COLUMN_ORDER: this is the human-readable
    # statement of the layout, and it must fail if the order changes by accident.
    assert (
        columns["index_columns"]
        + columns["target_columns"]
        + columns["feature_columns"]
        + columns["environment_columns"]
        + columns["mechanism_columns"]
        == header
    )


def test_the_schema_block_uses_this_formats_spelling(written: Path) -> None:
    """The manifest must describe *this* file, not the canonical CSV one.

    `schema_block` used to hardcode `det_`, so an ml_csv manifest would have promised
    columns its own table does not contain -- the exact over-claim that block exists to
    prevent, and the reason the spelling is passed in by the writing format.
    """
    roles = _sidecar(written)["manifest"]["schema"]["roles"]
    entry = roles["detectors"]
    assert entry["csv_prefix"] == "detector_"
    assert entry["csv_pad_width"] == 2

    resolved = [
        f"{entry['csv_prefix']}{i:0{entry['csv_pad_width']}d}" for i in range(entry["width"])
    ]
    assert resolved == _sidecar(written)["columns"]["feature_columns"]


def test_a_missing_sidecar_is_not_a_qecgen_dataset(written: Path) -> None:
    """Provable, not heuristic: a run commits the table and its sidecars in one move."""
    written.with_suffix(".manifest.json").unlink()
    with pytest.raises(NotAQecgenDatasetError):
        MLCSVExporter().read(written)
    with pytest.raises(NotAQecgenDatasetError):
        read_manifest(written)


def test_a_foreign_sidecar_is_not_a_qecgen_dataset(written: Path) -> None:
    written.with_suffix(".manifest.json").write_text('{"format": "someone-else"}', "utf-8")
    with pytest.raises(NotAQecgenDatasetError):
        MLCSVExporter().read(written)


def test_an_unknown_sidecar_version_is_refused(written: Path) -> None:
    """Refused, never read as version 1 with an unknown key ignored."""
    sidecar = written.with_suffix(".manifest.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["version"] = 99
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="understands only"):
        MLCSVExporter().read(written)


def test_a_header_row_disagreeing_with_the_sidecar_is_refused(written: Path) -> None:
    lines = written.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("detector_00", "detector_XX", 1)
    written.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with its sidecar"):
        MLCSVExporter().read(written)


def test_a_sidecar_disagreeing_with_its_own_manifest_is_refused(written: Path) -> None:
    sidecar = written.with_suffix(".manifest.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["columns"]["feature_columns"].pop()
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature columns but its manifest"):
        MLCSVExporter().read(written)


def test_sorted_rows_are_refused(written: Path) -> None:
    """This format's audience is tools that reorder rows, so the check matters more here."""
    lines = written.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    written.write_text("\n".join([header, *reversed(rows)]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Rows must stay in the order"):
        MLCSVExporter().read(written)


@pytest.mark.parametrize("bad", ["TRUE", "", "1.0"])
def test_a_cell_that_is_not_literally_zero_or_one_is_refused(written: Path, bad: str) -> None:
    lines = written.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    cells[1] = bad
    lines[1] = ",".join(cells)
    written.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="neither '0' nor '1'"):
        MLCSVExporter().read(written)


def test_a_comment_line_in_the_table_is_refused(written: Path) -> None:
    """There are no comment lines here, so one means two files were concatenated."""
    lines = written.read_text(encoding="utf-8").splitlines()
    lines.insert(2, "#qecgen-csv v1")
    written.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no comment lines"):
        MLCSVExporter().read(written)


def test_structure_level_without_its_sidecar_is_refused(tmp_path: Path) -> None:
    path = _write(
        _dataset(structure_level=StructureLevel.DEM), tmp_path / "d.ml.csv", StructureLevel.DEM
    )
    path.with_suffix(".structure.json").unlink()
    with pytest.raises(ValueError, match="is missing"):
        MLCSVExporter().read(path)


def test_provenance_never_reaches_the_manifest_sidecar_or_the_table(tmp_path: Path) -> None:
    """`full` puts circuit text in its own file, never in the one a listing opens."""
    path = _write(
        _dataset(structure_level=StructureLevel.FULL), tmp_path / "d.ml.csv", StructureLevel.FULL
    )
    assert b"QUBIT_COORDS" not in path.read_bytes()
    assert b"QUBIT_COORDS" not in path.with_suffix(".manifest.json").read_bytes()
    assert b"QUBIT_COORDS" in path.with_suffix(".provenance.json").read_bytes()


def test_read_never_returns_provenance_text(tmp_path: Path) -> None:
    """One `json.load` away is not close enough: `read` must not take that step."""
    path = _write(
        _dataset(structure_level=StructureLevel.FULL), tmp_path / "d.ml.csv", StructureLevel.FULL
    )
    restored = MLCSVExporter().read(path)
    assert restored.meta.environments[0].circuit == ""
    assert restored.meta.environments[0].dem == ""


def test_the_manifest_reads_without_touching_the_structure_sidecar(tmp_path: Path) -> None:
    """The cheap-listing bound. `json.load` has no early exit, so the files are split."""
    path = _write(
        _dataset(structure_level=StructureLevel.DEM), tmp_path / "d.ml.csv", StructureLevel.DEM
    )
    path.with_suffix(".structure.json").unlink()
    assert read_manifest(path)["n_detectors"] == 24


def test_the_environment_column_appears_only_when_pooled(tmp_path: Path) -> None:
    pooled = build_multi_environment(
        distance=3, error_rates=[0.005, 0.01], shots_per_env=16, seed=1, chunk_size=16
    )
    path = _write(pooled, tmp_path / "p.ml.csv", StructureLevel.NONE)
    columns = _sidecar(path)["columns"]
    assert columns["index_columns"] == ["shot"]
    assert columns["environment_columns"] == ["environment_id"]
    restored = MLCSVExporter().read(path)
    assert restored.environment_ids is not None
    assert pooled.environment_ids is not None
    assert np.array_equal(restored.environment_ids, pooled.environment_ids)


def test_mechanism_columns_appear_only_under_contract_b(tmp_path: Path) -> None:
    plain = _sidecar(_write(_dataset(), tmp_path / "a.ml.csv", StructureLevel.NONE))
    assert plain["columns"]["mechanism_columns"] == []

    path = _write(_dataset(emit_mechanisms=True), tmp_path / "b.ml.csv", StructureLevel.NONE)
    mech = _sidecar(path)["columns"]["mechanism_columns"]
    assert mech[0] == "mechanism_000"
    restored = MLCSVExporter().read(path)
    assert restored.mechanisms is not None


def test_the_registry_resolves_the_two_part_extension() -> None:
    """`Path.suffix` is `.csv` here, which would silently route to the wrong reader."""
    from qecgen.exporters import infer_format

    assert infer_format(Path("d.ml.csv")) == "ml_csv"
    assert infer_format(Path("d.csv")) == "csv"
    assert get_exporter("ml_csv").extension == ".ml.csv"


def test_the_spelling_differs_from_the_csv_one() -> None:
    """Two formats, two rules, deliberately. Neither is safe to copy into the other."""
    from qecgen.dataset import DETECTOR_PREFIX

    assert ML_CSV_SPELLING.detector != DETECTOR_PREFIX
    assert ML_CSV_SPELLING.pad is True


def test_a_rows_cells_land_under_their_own_header(tmp_path: Path) -> None:
    """Each cell must sit under the column name that describes it.

    The only check here that compares the file's *body* against its *header*. Every other
    assertion goes through `MLCSVExporter.read`, which shares its layout with `write`, so a
    symmetric mistake in both is invisible to them: the round trip stays green while the
    detector bits sit under the target's name. Read by NAME, never by index, and compared
    against the in-memory arrays rather than against anything the reader produced.
    """
    dataset = _dataset(p=0.05, shots=48, seed=5)
    path = _write(dataset, tmp_path / "d.ml.csv", StructureLevel.NONE)
    columns = _sidecar(path)["columns"]

    detectors = dataset.unpacked_detectors()
    observables = dataset.unpacked_observables()
    assert observables.any(), "fixture must have at least one flipped observable to be a test"

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for i, row in enumerate(rows):
        for j, name in enumerate(columns["feature_columns"]):
            assert row[name] == ("1" if detectors[i][j] else "0"), (name, i)
        for j, name in enumerate(columns["target_columns"]):
            assert row[name] == ("1" if observables[i][j] else "0"), (name, i)


def test_a_sidecar_from_a_previous_layout_explains_itself(tmp_path: Path) -> None:
    """An older sidecar must reach the header comparison, not crash inside a helper.

    Before the column order changed there was no `environment_columns` block, so indexing
    the sidecar for one raised a bare KeyError out of `_blocks` -- a traceback where the
    reader has a message that names the real cause. The file is not damaged; it is a
    previous layout.
    """
    pooled = build_multi_environment(
        distance=3, error_rates=[0.005, 0.01], shots_per_env=16, seed=1, chunk_size=16
    )
    path = _write(pooled, tmp_path / "p.ml.csv", StructureLevel.NONE)

    sidecar = path.with_suffix(".manifest.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["columns"]["environment_columns"], "a pooled file must carry the block"
    del payload["columns"]["environment_columns"]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="previous layout"):
        MLCSVExporter().read(path)
