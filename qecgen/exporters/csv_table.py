"""CSV exporter. The spreadsheet-facing format, and the only text format carrying provenance.

Layout: a block of ``#``-prefixed header lines, then a column header row, then one row per
shot::

    #qecgen-csv v1
    #__manifest__ {...}
    #__structure__ {...}          <- iff structure_level != none
    #__provenance__ {...}         <- iff structure_level == full
    shot,environment_id,det_0,...,det_23,obs_0,mech_0,...
    0,0,1,0,...

``pandas.read_csv(path, comment="#")`` and ``numpy.genfromtxt(path, comments="#")`` read
the table with no other argument.

**Every bit is its own column**, unpacked from the little-endian packing with an explicit
``bitorder="little"``. That is the point of the format, and it also sidesteps the trap the
JSONL exporter carries: there is no bit *string* here, so there is no ``int(s, 2)`` to
reverse the detector index with.

**The header block is not CSV, deliberately.** Each line is ``#key`` then one space then
one JSON object, written and read with plain ``readline``. The tempting alternative — a
quoted two-field CSV record, which would put the manifest in a single spreadsheet cell —
is unusable: the structure payload measures 63 KB at d=3, 434 KB at d=5 and 1.5 MB at
d=7, and ``csv.field_size_limit()`` defaults to 131072. A reader would have to raise a
*process-global* limit before the file would open at all. Handing a header line to
``csv.reader`` also splits the manifest across some two hundred fields that cannot be
rejoined except by guessing where the JSON started.

**Header order is normative.** The magic line is first so a foreign ``.csv`` is refused by
name rather than by a confusing complaint about a missing column; the manifest is second
so a listing reads two lines and never touches the structure line. ``qecgen inspect`` and
the UI's dataset browser both depend on that.

**Provenance is written at ``full``, which no other text format does.** The hazard JSONL
quarantines against does not arise here: a reader that does not filter ``#`` lines does
not find the table at all, so the step that locates the data is the same step that drops
the provenance. The cost, stated rather than glossed: circuit and DEM text contain no
commas, so a bare ``list(csv.reader(open(p)))`` over a ``full`` file at d>=5 hits ``field
larger than field limit`` on the provenance line — 207 KB of DEM text at d=7 is a single
field. Filter the comments, as every CSV reader must anyway. And prefer HDF5 for a
``full`` file a decoder will be pointed at: the wall between provenance and payload here
is a ``#``, where HDF5's is a group the reader has to open on purpose.

**A ``.csv`` without the magic line is not a dataset.** ``qecgen sweep`` writes its
threshold-results table with this extension too, so :class:`NotAQecgenDatasetError` — not
a corruption error — is what a reader gets for one.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from qecgen.dataset import DatasetMeta, InMemoryDataset, StructureLevel
from qecgen.exporters.base import (
    NotAQecgenDatasetError,
    recorded_structure_level,
    require_level_agreement,
)
from qecgen.exporters.structure_json import (
    load_json_object,
    repack,
    structure_from_json,
    structure_to_json,
)
from qecgen.sampling import unpack_bits

__all__ = [
    "CSV_SIZE_WARNING_THRESHOLD",
    "MAGIC_LINE",
    "MANIFEST_KEY",
    "PROVENANCE_KEY",
    "STRUCTURE_KEY",
    "CSVExporter",
    "read_manifest_only",
    "read_provenance_only",
]

CSV_SIZE_WARNING_THRESHOLD = 100_000
"""Shot count above which CSV becomes an actively bad idea."""

MAGIC_LINE = "#qecgen-csv v1"
"""First line of every file, compared literally.

Carries a version because a future revision that adds a *required* header line must be
refused by an old reader rather than read as this one with an unknown line ignored.
"""

COMMENT_PREFIX = "#"
MANIFEST_KEY = "#__manifest__"
STRUCTURE_KEY = "#__structure__"
PROVENANCE_KEY = "#__provenance__"
HEADER_KEYS = (MANIFEST_KEY, STRUCTURE_KEY, PROVENANCE_KEY)

SHOT_COLUMN = "shot"
ENVIRONMENT_COLUMN = "environment_id"
DETECTOR_PREFIX = "det_"
OBSERVABLE_PREFIX = "obs_"
MECHANISM_PREFIX = "mech_"

_LINE_TERMINATOR = "\n"
_BIT = {"0": False, "1": True}


def _expected_columns(
    meta: DatasetMeta, *, has_environment: bool, has_mechanisms: bool
) -> list[str]:
    """The column header this manifest implies.

    One builder for both directions: :meth:`CSVExporter.write` emits it and
    :meth:`CSVExporter.read` rebuilds it and demands equality, so the two cannot drift
    into a file whose columns and manifest describe different widths.

    Indices are plain decimal with no zero padding. The column *order* is authoritative
    and is checked exactly, so a padded name would be a second, distance-dependent
    encoding of the same index for downstream scripts to hardcode.
    """
    columns = [SHOT_COLUMN]
    if has_environment:
        columns.append(ENVIRONMENT_COLUMN)
    columns.extend(f"{DETECTOR_PREFIX}{i}" for i in range(meta.n_detectors))
    columns.extend(f"{OBSERVABLE_PREFIX}{i}" for i in range(meta.n_observables))
    if has_mechanisms:
        columns.extend(f"{MECHANISM_PREFIX}{i}" for i in range(meta.n_mechanisms or 0))
    return columns


def _require_zero_padding(array: np.ndarray, n_bits: int, name: str) -> None:
    """Refuse packed input whose padding bits are set.

    CSV stores ``n_bits`` columns, so a bit past the declared width has nowhere to go:
    unpacking drops it and repacking recreates it as zero. Stim zeroes its padding
    (measured: no padding bit set across 500 shots of an unrotated d=3 circuit, which
    pads 36 detectors into 5 bytes), so this never fires on generated data. On a
    hand-built or foreign array it would otherwise round-trip to *different bytes*,
    invisibly — the CSV looks identical either way, and the damage only surfaces later
    as a ``content_hash`` that no longer matches.
    """
    remainder = n_bits % 8
    if remainder == 0 or array.size == 0:
        return
    if bool(np.any(array[:, -1] >> remainder)):
        raise ValueError(
            f"{name} has bits set past the declared width of {n_bits}; CSV writes one "
            f"column per bit and cannot represent them, so the file would read back as "
            f"different bytes than were written"
        )


def _bit_cells(bits: np.ndarray) -> np.ndarray:
    """Render a bool array as an array of ``"0"``/``"1"`` cells."""
    return np.where(bits, "1", "0")


def _bits_from_cells(cells: list[str], where: str, column: str) -> np.ndarray:
    """Parse a row slice of ``0``/``1`` cells into a bool row.

    Strict by literal comparison. A spreadsheet re-saves a 0/1 column formatted as
    boolean into ``TRUE``/``FALSE``, and folding an empty cell to ``0`` would fabricate
    "no detection event" for a shot that had one. Both are refused rather than coerced.
    """
    try:
        return np.fromiter((_BIT[cell] for cell in cells), dtype=bool, count=len(cells))
    except KeyError as exc:
        raise ValueError(
            f"{where}: {column} cell {exc.args[0]!r} is neither '0' nor '1'. Values are "
            f"compared literally: a spreadsheet writes TRUE/FALSE for a boolean-formatted "
            f"column, and an empty cell folded to 0 would invent a shot with no detection "
            f"events"
        ) from None


def _require_magic(path: Path, first_line: str) -> None:
    """Refuse a ``.csv`` this tool did not write, before anything else is parsed."""
    if not first_line:
        raise NotAQecgenDatasetError(
            f"{path.name} is empty, so it is not a qecgen dataset. Every dataset starts "
            f"with the line {MAGIC_LINE!r}."
        )
    found = first_line.rstrip("\r\n")
    if found != MAGIC_LINE:
        raise NotAQecgenDatasetError(
            f"{path.name} does not begin with {MAGIC_LINE!r}, so it is a plain CSV rather "
            f"than a qecgen dataset (first line: {found[:60]!r}). `qecgen sweep` writes "
            f"its threshold-results table with this extension too."
        )


def _header_payload(path: Path, key: str) -> str | None:
    """The raw JSON payload of one header line, without reading a shot row.

    Stops at the first line that is not a comment, so this costs two ``readline`` calls
    for the manifest however large the file is. That bound is why the dataset browser can
    list a directory of million-shot CSVs.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        _require_magic(path, handle.readline())
        for line in handle:
            if not line.startswith(COMMENT_PREFIX):
                return None
            found, _, payload = line.rstrip("\r\n").partition(" ")
            if found == key:
                return payload
    return None


def read_manifest_only(path: Path) -> dict[str, object]:
    """Read a manifest without loading any shots. Used by ``qecgen inspect`` and the UI.

    Reading the whole file to print a manifest would be worst-case here: CSV is text with
    one column per detector, so it is the most expensive format to materialise and the
    cheapest to peek at.
    """
    payload = _header_payload(path, MANIFEST_KEY)
    if payload is None:
        raise NotAQecgenDatasetError(
            f"{path.name} has the qecgen CSV magic line but no {MANIFEST_KEY} header"
        )
    return dict(load_json_object(payload, f"{path}:{MANIFEST_KEY}"))


def read_provenance_only(path: Path) -> dict[str, Any] | None:
    """Read the provenance block, or None when the file carries none.

    Separate from :func:`read_manifest_only` on purpose. Provenance is not
    decoder-visible, and a manifest loader that returned it would put a frozen-prior test
    file's own DEM one call away from every harness that reads manifests.
    """
    payload = _header_payload(path, PROVENANCE_KEY)
    if payload is None:
        return None
    return dict(load_json_object(payload, f"{path}:{PROVENANCE_KEY}"))


class CSVExporter:
    """One row per shot, one column per bit. Readable anywhere; large on disk."""

    @property
    def format_name(self) -> str:
        """Format name for ``--format``."""
        return "csv"

    @property
    def extension(self) -> str:
        """File extension. Shared with ``qecgen sweep``, hence :data:`MAGIC_LINE`."""
        return ".csv"

    @property
    def streaming(self) -> bool:
        """Appendable in principle, but not wired as a streaming writer here.

        ``environments.stream_single_environment`` constructs a ``StreamingHDF5Writer``
        directly rather than resolving one from the registry, so declaring True here
        would route a CSV run into an HDF5 writer and produce an ``.csv`` file full of
        HDF5 bytes.
        """
        return False

    @property
    def structure_round_trip(self) -> bool:
        """Structure round-trips exactly, on the ``#__structure__`` header line."""
        return True

    @property
    def carries_provenance(self) -> bool:
        """Yes, on its own header line above the table every reader must skip."""
        return True

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        """Write the header block, the column header, then one row per shot."""
        require_level_agreement(dataset, structure_level)
        if dataset.mechanisms is not None and not dataset.meta.n_mechanisms:
            # A zero-width block of mechanism columns is indistinguishable from no block
            # at all, so read() would hand back `mechanisms=None` and the content hash
            # would change. Reachable: a p=0 environment has a DEM with no errors.
            raise ValueError(
                "dataset carries mechanisms but the manifest declares "
                f"n_mechanisms={dataset.meta.n_mechanisms!r}; a zero-width column block "
                "cannot be told apart from an absent one on read"
            )
        if dataset.n_shots > CSV_SIZE_WARNING_THRESHOLD:
            bit_columns = dataset.meta.n_detectors + dataset.meta.n_observables
            bit_columns += dataset.meta.n_mechanisms or 0
            warnings.warn(
                f"Writing {dataset.n_shots:,} shots as CSV. This format stores one "
                f"character per bit plus a separator, so expect roughly "
                f"{dataset.n_shots * bit_columns * 2 / 1e9:.1f} GB and very slow reads. "
                f"Use hdf5 for anything above {CSV_SIZE_WARNING_THRESHOLD:,} shots.",
                stacklevel=2,
            )

        _require_zero_padding(dataset.detectors, dataset.meta.n_detectors, "detectors")
        _require_zero_padding(dataset.observables, dataset.meta.n_observables, "observables")
        detector_cells = _bit_cells(unpack_bits(dataset.detectors, dataset.meta.n_detectors))
        observable_cells = _bit_cells(unpack_bits(dataset.observables, dataset.meta.n_observables))
        mechanism_cells = None
        if dataset.mechanisms is not None:
            n_mechanisms = dataset.meta.n_mechanisms or 0
            _require_zero_padding(dataset.mechanisms, n_mechanisms, "mechanisms")
            mechanism_cells = _bit_cells(unpack_bits(dataset.mechanisms, n_mechanisms))

        # Only the persisted copy is downgraded; the caller's dataset is untouched.
        recorded = dataclasses.replace(
            dataset.meta, structure_level=recorded_structure_level(self, structure_level)
        )
        environment_ids = dataset.environment_ids
        columns = _expected_columns(
            recorded,
            has_environment=environment_ids is not None,
            has_mechanisms=mechanism_cells is not None,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" is mandatory even though the header lines are written with an
        # explicit "\n": without it the csv module's terminator is translated a second
        # time on Windows and every data row ends "\r\r\n", which several parsers read as
        # a blank row between every shot.
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(MAGIC_LINE + _LINE_TERMINATOR)
            handle.write(f"{MANIFEST_KEY} {recorded.to_json()}{_LINE_TERMINATOR}")
            if structure_level is not StructureLevel.NONE and dataset.structure is not None:
                payload = structure_to_json(dataset.structure, structure_level)
                encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
                handle.write(f"{STRUCTURE_KEY} {encoded}{_LINE_TERMINATOR}")
            if structure_level is StructureLevel.FULL:
                handle.write(f"{PROVENANCE_KEY} {dataset.meta.provenance_json()}{_LINE_TERMINATOR}")

            writer = csv.writer(handle, lineterminator=_LINE_TERMINATOR)
            writer.writerow(columns)
            for i in range(dataset.n_shots):
                row = [str(i)]
                if environment_ids is not None:
                    row.append(str(int(environment_ids[i])))
                row.extend(detector_cells[i].tolist())
                row.extend(observable_cells[i].tolist())
                if mechanism_cells is not None:
                    row.extend(mechanism_cells[i].tolist())
                writer.writerow(row)

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset written by :meth:`write`.

        Dispatches on the header *key* rather than the line number, so a ``dem``-level
        file and a ``full`` one — three header lines against four — take the same path
        with no line arithmetic.
        """
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            _require_magic(path, handle.readline())
            meta, structure_payload, header_count, line = _read_header_block(path, handle)
            columns = next(csv.reader([line.rstrip("\r\n")]), [])
            if not columns:
                raise ValueError(
                    f"{path}: the header block is present but the column header row is "
                    "missing, so the file was cut short before any shots were written"
                )
            _require_column_agreement(path, columns, meta)
            offsets = _column_offsets(columns, meta)

            det_rows: list[np.ndarray] = []
            obs_rows: list[np.ndarray] = []
            mech_rows: list[np.ndarray] = []
            env_rows: list[int] = []
            # header_count is the line number of the last `#` line; the column header
            # follows it, so the reader's first row is two lines further on.
            first_data_line = header_count + 1
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                where = f"{path}:{first_data_line + reader.line_num}"
                _require_data_row(where, row, columns)
                if row[0] != str(len(det_rows)):
                    raise ValueError(
                        f"{where}: {SHOT_COLUMN} is {row[0]!r} but this is row "
                        f"{len(det_rows)}. Rows must stay in the order they were written: "
                        f"sorting them in a spreadsheet severs the correspondence between "
                        f"a shot's detectors, its {ENVIRONMENT_COLUMN} and its mechanism "
                        f"labels while leaving a perfectly well-formed file"
                    )
                if offsets.environment is not None:
                    env_rows.append(_parse_environment_id(where, row[offsets.environment]))
                det_cells = row[offsets.detectors : offsets.observables]
                det_rows.append(_bits_from_cells(det_cells, where, "detector"))
                obs_rows.append(
                    _bits_from_cells(
                        row[offsets.observables : offsets.mechanisms], where, "observable"
                    )
                )
                if offsets.has_mechanisms:
                    mech_rows.append(
                        _bits_from_cells(row[offsets.mechanisms :], where, "mechanism")
                    )

        if meta.structure_level is not StructureLevel.NONE and structure_payload is None:
            raise ValueError(
                f"{path}: the manifest records structure_level={meta.structure_level} but "
                f"the file has no {STRUCTURE_KEY} line. Downstream that is "
                "indistinguishable from a corrupt file, so it is refused rather than "
                "silently downgraded."
            )

        return InMemoryDataset(
            detectors=repack(det_rows, meta.n_detectors),
            observables=repack(obs_rows, meta.n_observables),
            meta=meta,
            environment_ids=(
                np.asarray(env_rows, dtype=np.int32) if offsets.environment is not None else None
            ),
            mechanisms=(
                repack(mech_rows, meta.n_mechanisms or 0) if offsets.has_mechanisms else None
            ),
            structure=(
                None if structure_payload is None else structure_from_json(structure_payload, meta)
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _Offsets:
    """Where each block of columns starts, resolved once instead of per row."""

    environment: int | None
    detectors: int
    observables: int
    mechanisms: int
    has_mechanisms: bool


def _column_offsets(columns: list[str], meta: DatasetMeta) -> _Offsets:
    """Slice boundaries for one data row, derived from the validated column header."""
    has_environment = len(columns) > 1 and columns[1] == ENVIRONMENT_COLUMN
    detectors = 2 if has_environment else 1
    observables = detectors + meta.n_detectors
    mechanisms = observables + meta.n_observables
    return _Offsets(
        environment=1 if has_environment else None,
        detectors=detectors,
        observables=observables,
        mechanisms=mechanisms,
        has_mechanisms=len(columns) > mechanisms,
    )


def _read_header_block(
    path: Path, handle: TextIO
) -> tuple[DatasetMeta, dict[str, Any] | None, int, str]:
    """Consume the ``#`` header lines and return the first line that is not one.

    Returning that line rather than seeking back to it: ``tell``/``seek`` on a text
    handle deals in opaque cookies, and the one line already in hand is the column
    header the caller wants next.
    """
    meta: DatasetMeta | None = None
    structure_payload: dict[str, Any] | None = None
    seen: set[str] = set()
    line_no = 1
    line = handle.readline()
    while line.startswith(COMMENT_PREFIX):
        line_no += 1
        where = f"{path}:{line_no}"
        key, _, payload = line.rstrip("\r\n").partition(" ")
        if key not in HEADER_KEYS:
            raise ValueError(
                f"{where}: unknown header key {key!r}. Ignoring one would be how a future "
                f"required payload gets silently dropped by an old reader. Known keys: "
                f"{', '.join(HEADER_KEYS)}"
            )
        if key in seen:
            raise ValueError(f"{where}: a second {key} line; files were concatenated")
        seen.add(key)
        if key == MANIFEST_KEY:
            meta = DatasetMeta.from_json_dict(load_json_object(payload, where))
        elif key == STRUCTURE_KEY:
            if meta is None:
                raise ValueError(f"{where}: structure appears before the manifest")
            structure_payload = load_json_object(payload, where)
        # PROVENANCE_KEY is read past and never parsed. Returning it would hand a
        # frozen-prior test file's own DEM back to anything that merely called read().
        line = handle.readline()

    if meta is None:
        raise ValueError(f"{path}: no {MANIFEST_KEY} header line found")
    return meta, structure_payload, line_no, line


def _require_column_agreement(path: Path, columns: list[str], meta: DatasetMeta) -> None:
    """Refuse a column header that disagrees with the manifest.

    The redundancy is deliberate. A column set that disagrees with the manifest means the
    two describe different widths, and resolving that in favour of either one is a guess
    about which of them is the corrupt half.
    """
    has_environment = len(columns) > 1 and columns[1] == ENVIRONMENT_COLUMN
    has_mechanisms = any(column.startswith(MECHANISM_PREFIX) for column in columns)
    if has_mechanisms and meta.n_mechanisms is None:
        raise ValueError(
            f"{path}: the column header carries {MECHANISM_PREFIX}* columns but the "
            "manifest declares no n_mechanisms, so their width is unknown"
        )
    expected = _expected_columns(
        meta, has_environment=has_environment, has_mechanisms=has_mechanisms
    )
    if columns == expected:
        return
    divergence = next(
        (i for i, (a, b) in enumerate(zip(columns, expected, strict=False)) if a != b),
        min(len(columns), len(expected)),
    )
    raise ValueError(
        f"{path}: the column header disagrees with the manifest. It has {len(columns)} "
        f"columns, the manifest implies {len(expected)}, and they first differ at index "
        f"{divergence} (file {columns[divergence : divergence + 1]}, manifest "
        f"{expected[divergence : divergence + 1]})"
    )


def _require_data_row(where: str, row: list[str], columns: list[str]) -> None:
    """Refuse a row that is not a shot, or is not the right width."""
    if row[0].startswith(COMMENT_PREFIX):
        raise ValueError(
            f"{where}: a header line appears after the table began. Two files were "
            "concatenated, or shots were appended to a finished one; either way the "
            "manifest no longer describes the rows beneath it"
        )
    if len(row) != len(columns):
        raise ValueError(
            f"{where}: the row has {len(row)} fields but the header declares "
            f"{len(columns)}. A spreadsheet that dropped trailing empty columns produces "
            f"exactly this, and padding it would fabricate bits"
        )


def _parse_environment_id(where: str, cell: str) -> int:
    """Parse one ``environment_id`` cell, naming the file on failure."""
    try:
        return int(cell)
    except ValueError:
        raise ValueError(f"{where}: {ENVIRONMENT_COLUMN} {cell!r} is not an integer") from None
