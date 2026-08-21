"""A CSV with exactly one header row, and its metadata in JSON sidecars.

``csv`` puts its manifest in ``#``-prefixed lines above the table. That is deliberate --
the step that locates the data is the step that drops the provenance -- but it is the
wrong default for a consumer whose reader is ``pandas.read_csv`` with no arguments: that
call returns a single garbage column named ``#qecgen-csv v1``. This format exists for
that consumer.

Four files, committed together::

    d.ml.csv              one header row, then one row per shot. No '#' line anywhere.
    d.ml.manifest.json    the manifest, plus this file's literal column names
    d.ml.structure.json   iff structure_level != none
    d.ml.provenance.json  iff structure_level == full

**The manifest sidecar is this format's magic line.** With no in-band marker, a bare
``.ml.csv`` is indistinguishable by shape from any other one-header-row CSV, so
"the named sidecar is not beside this table" is what makes a file provably not ours. That
signal is sound because :func:`qecgen.run.staged` commits the whole set in one two-phase
move: a table without its sidecar is not a state this tool can publish, so it can only be
a foreign file. See :class:`NotAQecgenDatasetError`.

**Provenance is a separate file, not a key in the manifest sidecar.** ``csv`` may share a
line with its provenance because a reader that does not filter ``#`` never finds the table
at all. That defence does not transfer here: the idiomatic sidecar reader is
``json.load(open(sidecar))``, a complete and intended reader, so sharing the object would
hand a frozen-prior test file's own DEM back in one call. A separate path is a stronger
wall than a comment line, and comparable to HDF5's ``/provenance`` group.

**Structure is a separate file too**, for a different reason: ``json.load`` has no early
exit. ``csv`` bounds a cheap manifest read by line order -- manifest on line 2, stop before
the structure line, which is 1.5 MB at d=7. Splitting the files restores that bound, so
listing a directory never parses the structure payloads.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from qecgen.dataset import (
    ENVIRONMENT_COLUMN,
    SHOT_COLUMN,
    ColumnSpelling,
    DatasetMeta,
    InMemoryDataset,
    StructureLevel,
)
from qecgen.exporters.base import (
    NotAQecgenDatasetError,
    recorded_structure_level,
    require_level_agreement,
)
from qecgen.exporters.bit_columns import (
    bit_cells,
    bits_from_cells,
    require_row_in_order,
    require_zero_padding,
    warn_if_large,
)
from qecgen.exporters.structure_json import (
    load_json_object,
    repack,
    structure_from_json,
    structure_to_json,
)
from qecgen.sampling import unpack_bits

__all__ = [
    "ML_CSV_SPELLING",
    "SIDECAR_FORMAT",
    "SIDECAR_VERSION",
    "MLCSVExporter",
    "read_manifest_only",
    "read_provenance_only",
]

ML_CSV_SPELLING = ColumnSpelling(
    detector="detector_", observable="observable_", mechanism="mechanism_", pad=True
)
"""Spelled out and zero-padded, unlike ``csv``'s ``det_0``.

Both rules are right for their own consumer and neither is safe to copy across.
``csv_table._expected_columns`` rejects padding because the width is distance-dependent,
which makes a padded name a second encoding of the same index for a script to hardcode.
That objection is correct and still stands -- it is overruled here only because this
format's consumer is a dataframe rather than a spreadsheet: ``sorted(df.columns)``, and
every column sort hiding inside a join or a feature-store schema, puts ``detector_10``
before ``detector_2`` and permutes the feature matrix silently, leaving a model that
trains, converges and means nothing.

The objection is answered rather than dismissed: the sidecar publishes the literal,
ordered column names, so no consumer ever reproduces the pad width. A script that reads
those lists is correct at every distance; one that builds ``f"detector_{i:02d}"`` is the
hardcoding ``csv``'s docstring warns about, and breaks at 100 detectors exactly as
predicted.
"""

SIDECAR_FORMAT = "qecgen-ml-csv"
"""Marker in the manifest sidecar. What makes a file provably ours."""

SIDECAR_VERSION = 1
"""Refused when unknown, never read as this version with an unknown key ignored.

Same reasoning as ``csv``'s magic line carrying ``v1``: a future revision that adds a
*required* key must be refused by an old reader rather than half-understood.
"""

_MANIFEST_SUFFIX = ".manifest.json"
_STRUCTURE_SUFFIX = ".structure.json"
_PROVENANCE_SUFFIX = ".provenance.json"
_LINE_TERMINATOR = "\n"


def _companion(path: Path, suffix: str) -> Path:
    """The sidecar beside ``path``.

    ``with_suffix`` replaces only the final suffix, so ``d.ml.csv`` yields
    ``d.ml.manifest.json`` and the whole set shares the stem ``d.ml``. One rule, and it
    still works for a path with no suffix at all.
    """
    return path.with_suffix(suffix)


def _columns_block(
    meta: DatasetMeta, *, has_environment: bool, has_mechanisms: bool
) -> dict[str, Any]:
    """The literal, ordered column names this file carries.

    Split by role rather than given as one flat list, because the split *is* the contract:
    features to targets is Contract A, and ``mechanism_columns`` are Contract B labels a
    consumer must not concatenate into the targets -- that would train against a different
    contract than the file declares. :meth:`MLCSVExporter.read` rebuilds the header by
    concatenating the four in this order, so there is no fifth field to disagree with them.
    """
    index_columns = [SHOT_COLUMN]
    if has_environment:
        index_columns.append(ENVIRONMENT_COLUMN)
    return {
        "index_columns": index_columns,
        "feature_columns": ML_CSV_SPELLING.columns(ML_CSV_SPELLING.detector, meta.n_detectors),
        "target_columns": ML_CSV_SPELLING.columns(ML_CSV_SPELLING.observable, meta.n_observables),
        "mechanism_columns": (
            ML_CSV_SPELLING.columns(ML_CSV_SPELLING.mechanism, meta.n_mechanisms or 0)
            if has_mechanisms
            else []
        ),
        "bit_values": ["0", "1"],
        "row_order": f"{SHOT_COLUMN} equals the zero-based row index; do not sort",
    }


def _header_row(columns: dict[str, Any]) -> list[str]:
    """The single header row, from the sidecar's four ordered lists."""
    return [
        *columns["index_columns"],
        *columns["feature_columns"],
        *columns["target_columns"],
        *columns["mechanism_columns"],
    ]


def _load_sidecar(path: Path, suffix: str, *, required: bool) -> dict[str, Any] | None:
    """Read one companion, refusing a foreign or future one."""
    companion = _companion(path, suffix)
    if not companion.exists():
        if not required:
            return None
        raise NotAQecgenDatasetError(
            f"{path.name} has no {companion.name} beside it, so it is a plain CSV rather "
            f"than a qecgen dataset. This format keeps its manifest in that sidecar, and "
            f"a run commits the whole set together, so a table without one was not "
            f"written by qecgen."
        )
    payload = load_json_object(companion.read_text(encoding="utf-8"), str(companion))
    if suffix == _MANIFEST_SUFFIX:
        if payload.get("format") != SIDECAR_FORMAT:
            raise NotAQecgenDatasetError(
                f"{companion.name} is not a {SIDECAR_FORMAT} sidecar (format="
                f"{payload.get('format')!r}), so {path.name} is not a qecgen dataset."
            )
        version = payload.get("version")
        if version != SIDECAR_VERSION:
            raise ValueError(
                f"{companion.name} declares version {version!r}, but this reader "
                f"understands only {SIDECAR_VERSION}. Refused rather than read as this "
                f"version, because a later revision may add a required key."
            )
    return payload


def read_manifest_only(path: Path) -> dict[str, Any]:
    """The manifest without reading a single shot row.

    The cheap path for ``qecgen inspect`` and the dataset browser. It opens only the
    manifest sidecar, so listing a directory never touches the table or the structure
    payload -- the bound ``csv`` gets from line ordering, this format gets from having
    the metadata in separate files.
    """
    payload = _load_sidecar(path, _MANIFEST_SUFFIX, required=True)
    assert payload is not None
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{_companion(path, _MANIFEST_SUFFIX).name} has no manifest object")
    return manifest


def read_provenance_only(path: Path) -> dict[str, Any] | None:
    """The provenance block, for ``qecgen inspect --show-text`` and nothing else.

    Deliberately not called by :meth:`MLCSVExporter.read`: returning it there would hand
    a frozen-prior test file's own DEM back to anything that merely read the dataset.
    """
    return _load_sidecar(path, _PROVENANCE_SUFFIX, required=False)


class MLCSVExporter:
    """One header row, one row per shot, metadata in sidecars."""

    @property
    def format_name(self) -> str:
        return "ml_csv"

    @property
    def extension(self) -> str:
        """Two-part, so the table still opens in a spreadsheet on a double click.

        ``.csv`` is already claimed twice over -- by the dataset CSV and by ``qecgen
        sweep``'s results table -- and a third meaning would force content sniffing on a
        format that deliberately has nothing in-band to sniff. ``infer_format`` matches the
        longest registered extension a name ends with, so ``.ml.csv`` beats ``.csv``;
        ``Path.suffix`` is never a valid way to compare it.
        """
        return ".ml.csv"

    @property
    def streaming(self) -> bool:
        return False

    @property
    def structure_round_trip(self) -> bool:
        return True

    @property
    def carries_provenance(self) -> bool:
        return True

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        require_level_agreement(dataset, structure_level)
        meta = dataset.meta
        has_environment = dataset.environment_ids is not None
        has_mechanisms = dataset.mechanisms is not None
        if has_mechanisms and meta.n_mechanisms is None:
            raise ValueError(
                "mechanisms are present but the manifest declares no n_mechanisms; a "
                "zero-width column block cannot be told apart from an absent one on read"
            )

        columns = _columns_block(
            meta, has_environment=has_environment, has_mechanisms=has_mechanisms
        )
        header = _header_row(columns)
        warn_if_large(dataset.n_shots, len(header), self.format_name)

        require_zero_padding(dataset.detectors, meta.n_detectors, "detectors")
        require_zero_padding(dataset.observables, meta.n_observables, "observables")
        detector_cells = bit_cells(unpack_bits(dataset.detectors, meta.n_detectors))
        observable_cells = bit_cells(unpack_bits(dataset.observables, meta.n_observables))
        mechanism_cells: np.ndarray | None = None
        if dataset.mechanisms is not None and meta.n_mechanisms is not None:
            require_zero_padding(dataset.mechanisms, meta.n_mechanisms, "mechanisms")
            mechanism_cells = bit_cells(unpack_bits(dataset.mechanisms, meta.n_mechanisms))

        recorded = dataclasses.replace(
            meta, structure_level=recorded_structure_level(self, structure_level)
        )

        # newline="" is mandatory: csv.writer emits its own terminator, and the text layer
        # would otherwise turn it into \r\n on Windows and break the byte comparison the
        # round-trip tests make.
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator=_LINE_TERMINATOR)
            writer.writerow(header)
            for i in range(dataset.n_shots):
                row = [str(i)]
                if dataset.environment_ids is not None:
                    row.append(str(int(dataset.environment_ids[i])))
                row.extend(detector_cells[i])
                row.extend(observable_cells[i])
                if mechanism_cells is not None:
                    row.extend(mechanism_cells[i])
                writer.writerow(row)

        sidecar = {
            "format": SIDECAR_FORMAT,
            "version": SIDECAR_VERSION,
            "columns": columns,
            "manifest": recorded.to_json_dict(ML_CSV_SPELLING),
        }
        _write_json(_companion(path, _MANIFEST_SUFFIX), sidecar)

        if structure_level is not StructureLevel.NONE and dataset.structure is not None:
            _write_json(
                _companion(path, _STRUCTURE_SUFFIX),
                structure_to_json(dataset.structure, structure_level),
            )
        if structure_level is StructureLevel.FULL:
            _write_json(_companion(path, _PROVENANCE_SUFFIX), meta.provenance_dict())

    def read(self, path: Path) -> InMemoryDataset:
        manifest_payload = _load_sidecar(path, _MANIFEST_SUFFIX, required=True)
        assert manifest_payload is not None
        raw_manifest = manifest_payload.get("manifest")
        if not isinstance(raw_manifest, dict):
            raise ValueError(f"{path}: the sidecar carries no manifest object")
        meta = DatasetMeta.from_json_dict(raw_manifest)
        columns = manifest_payload.get("columns")
        if not isinstance(columns, dict):
            raise ValueError(f"{path}: the sidecar carries no columns block")

        expected = _header_row(columns)
        _require_sidecar_matches_manifest(path, columns, meta)

        det_rows: list[np.ndarray] = []
        obs_rows: list[np.ndarray] = []
        mech_rows: list[np.ndarray] = []
        has_environment = ENVIRONMENT_COLUMN in columns["index_columns"]
        has_mechanisms = bool(columns["mechanism_columns"])
        first_bit = len(columns["index_columns"])
        first_target = first_bit + len(columns["feature_columns"])
        first_mechanism = first_target + len(columns["target_columns"])
        environment_ids: list[int] = []

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                raise NotAQecgenDatasetError(
                    f"{path.name} is empty, so it is not a qecgen dataset."
                ) from None
            if header != expected:
                raise ValueError(
                    f"{path}: the header row disagrees with its sidecar. The file has "
                    f"{len(header)} columns and the sidecar names {len(expected)}"
                    f"{_first_difference(header, expected)}. Resolving that in favour of "
                    f"either one would be a guess about which half is corrupt."
                )
            for row in reader:
                if not row:
                    continue
                where = f"{path}:{reader.line_num}"
                if row and row[0].startswith("#"):
                    raise ValueError(
                        f"{where}: this format has no comment lines, so a line beginning "
                        f"'#' means a qecgen csv file was concatenated in or a header "
                        f"block was pasted on."
                    )
                if len(row) != len(expected):
                    raise ValueError(
                        f"{where}: the row has {len(row)} fields but the header declares "
                        f"{len(expected)}; padding it would fabricate bits"
                    )
                require_row_in_order(row[0], len(det_rows), where, SHOT_COLUMN)
                if has_environment:
                    environment_ids.append(_parse_environment_id(row[1], where))
                det_rows.append(bits_from_cells(row[first_bit:first_target], where, "detector"))
                obs_rows.append(
                    bits_from_cells(row[first_target:first_mechanism], where, "observable")
                )
                if has_mechanisms:
                    mech_rows.append(bits_from_cells(row[first_mechanism:], where, "mechanism"))

        structure_payload = _load_sidecar(path, _STRUCTURE_SUFFIX, required=False)
        if meta.structure_level is not StructureLevel.NONE and structure_payload is None:
            raise ValueError(
                f"{path}: the manifest records structure_level={meta.structure_level} but "
                f"{_companion(path, _STRUCTURE_SUFFIX).name} is missing. Downstream that "
                "is indistinguishable from a corrupt file, so it is refused rather than "
                "silently downgraded."
            )

        return InMemoryDataset(
            detectors=repack(det_rows, meta.n_detectors),
            observables=repack(obs_rows, meta.n_observables),
            meta=meta,
            environment_ids=(
                np.asarray(environment_ids, dtype=np.int32) if has_environment else None
            ),
            mechanisms=(repack(mech_rows, meta.n_mechanisms or 0) if has_mechanisms else None),
            structure=(
                structure_from_json(structure_payload, meta)
                if structure_payload is not None
                else None
            ),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """``allow_nan=False`` so a non-finite value can never become the bare token NaN."""
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def _first_difference(found: list[str], expected: list[str]) -> str:
    for index, (a, b) in enumerate(zip(found, expected, strict=False)):
        if a != b:
            return f", first differing at index {index} ({a!r} vs {b!r})"
    return ""


def _parse_environment_id(cell: str, where: str) -> int:
    try:
        return int(cell)
    except ValueError:
        raise ValueError(f"{where}: {ENVIRONMENT_COLUMN} {cell!r} is not an integer") from None


def _require_sidecar_matches_manifest(
    path: Path, columns: dict[str, Any], meta: DatasetMeta
) -> None:
    """Refuse a sidecar whose column lists contradict its own manifest.

    The two live in one file and describe one table, so a disagreement is corruption in
    one of them and there is no way to tell which. Checked before the table is opened,
    so a mismatch never costs a full parse.
    """
    for name, found, declared in (
        ("feature", len(columns["feature_columns"]), meta.n_detectors),
        ("target", len(columns["target_columns"]), meta.n_observables),
        ("mechanism", len(columns["mechanism_columns"]), meta.n_mechanisms or 0),
    ):
        if found != declared:
            raise ValueError(
                f"{path}: the sidecar names {found} {name} columns but its manifest "
                f"declares {declared}."
            )
