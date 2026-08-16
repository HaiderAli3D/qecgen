"""Exporter registry.

The CLI discovers formats through :data:`EXPORTERS`. A new format is one module plus
one line here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from qecgen.exporters.base import Exporter, NotAQecgenDatasetError, require_non_empty
from qecgen.exporters.csv_table import CSVExporter
from qecgen.exporters.hdf5 import HDF5Exporter, StreamingHDF5Writer
from qecgen.exporters.jsonl import JSONLExporter
from qecgen.exporters.npz import NPZExporter
from qecgen.exporters.parquet import ParquetExporter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = [
    "EXPORTERS",
    "CSVExporter",
    "Exporter",
    "HDF5Exporter",
    "JSONLExporter",
    "NPZExporter",
    "NotAQecgenDatasetError",
    "ParquetExporter",
    "StreamingHDF5Writer",
    "get_exporter",
    "infer_format",
    "manifest_reader_coverage",
    "provenance_formats",
    "read_manifest",
    "read_provenance",
]

EXPORTERS: dict[str, Exporter] = {
    exporter.format_name: exporter
    for exporter in (
        HDF5Exporter(),
        NPZExporter(),
        ParquetExporter(),
        JSONLExporter(),
        CSVExporter(),
    )
}
"""Registered exporters keyed by ``format_name``."""


def get_exporter(format_name: str) -> Exporter:
    """Look up an exporter, raising a helpful error listing valid names."""
    try:
        return EXPORTERS[format_name]
    except KeyError:
        valid = ", ".join(sorted(EXPORTERS))
        raise ValueError(f"unknown format {format_name!r}; available: {valid}") from None


def infer_format(path: Path) -> str:
    """Map a file extension to a registered format name.

    Lives beside the registry rather than in a front end because the mapping is
    *derived* from :data:`EXPORTERS`: adding a format must stay "one module plus one line
    here", not "plus a second edit in whichever front end needs to infer it".

    Raises:
        ValueError: for an unrecognised suffix, matching :func:`get_exporter` so callers
            are not forced to import a CLI framework to catch it.
    """
    by_extension = {exporter.extension: name for name, exporter in EXPORTERS.items()}
    try:
        return by_extension[path.suffix]
    except KeyError:
        known = ", ".join(sorted(by_extension))
        raise ValueError(
            f"cannot infer format from {path.suffix!r}; state it explicitly. Known: {known}"
        ) from None


# --------------------------------------------------------------------------------------
# Cheap readers: the manifest, and the provenance block, without loading any shots.
#
# These live here, three lines from EXPORTERS, for the reason infer_format states above.
# They were previously two hand-maintained tables in two different front ends -- one in
# `ui/datasets.py` (all five formats) and one in `cli.py` (three) -- and neither front end
# could use the other's: `cli.py` importing `qecgen.ui` would make `qecgen inspect` depend
# on a web stack, and the reverse edge is no better. So the same knowledge was written
# down twice, in different states of completeness, and adding a format meant editing both.
# --------------------------------------------------------------------------------------


def _manifest_hdf5(path: Path) -> dict[str, Any]:
    from qecgen.exporters.hdf5 import read_manifest_only

    return dict(read_manifest_only(path))


def _manifest_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    from qecgen.exporters.npz import require_qecgen_archive

    require_non_empty(path, "a `manifest` member")
    # np.load on a zip is lazy, so this decompresses one member rather than every array.
    with np.load(path, allow_pickle=False) as data:
        require_qecgen_archive(path, list(data.files))
        return dict(json.loads(str(data["manifest"].item())))


def _manifest_parquet(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from qecgen.exporters.parquet import require_qecgen_metadata

    require_non_empty(path, "a qecgen_manifest schema-metadata key")
    metadata = pq.read_schema(path).metadata or {}
    require_qecgen_metadata(path, metadata)
    return dict(json.loads(metadata[b"qecgen_manifest"].decode("utf-8")))


def _manifest_jsonl(path: Path) -> dict[str, Any]:
    from qecgen.exporters.jsonl import require_qecgen_first_line

    require_non_empty(path, "a '__manifest__' record on line 1")
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    require_qecgen_first_line(path, first)
    return dict(json.loads(first)["__manifest__"])


def _manifest_csv(path: Path) -> dict[str, Any]:
    # A bounded scan of the leading `#` lines that stops at the first non-comment line.
    # The manifest is the second line by construction, so this never reaches the
    # structure line -- 1.5 MB at d=7 -- let alone a shot row.
    from qecgen.exporters.csv_table import read_manifest_only

    return dict(read_manifest_only(path))


_MANIFEST_READERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "hdf5": _manifest_hdf5,
    "npz": _manifest_npz,
    "parquet": _manifest_parquet,
    "jsonl": _manifest_jsonl,
    "csv": _manifest_csv,
}
"""One cheap manifest reader per registered format.

Every registered format has one, and :func:`manifest_reader_coverage` is what keeps that
true: a format registered without an entry made every one of its files list as
*unreadable* in the dataset browser, which is corruption's signal on a healthy file.
"""


def _provenance_hdf5(path: Path) -> dict[str, Any] | None:
    import h5py

    with h5py.File(path, "r") as handle:
        if "provenance" not in handle:
            return None
        return dict(json.loads(str(handle["provenance"].attrs["environments"])))


def _provenance_npz(path: Path) -> dict[str, Any] | None:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if "provenance" not in archive.files:
            return None
        return dict(json.loads(str(archive["provenance"].item())))


def _provenance_csv(path: Path) -> dict[str, Any] | None:
    from qecgen.exporters.csv_table import read_provenance_only

    payload = read_provenance_only(path)
    return None if payload is None else dict(payload)


_PROVENANCE_READERS: dict[str, Callable[[Path], dict[str, Any] | None]] = {
    "hdf5": _provenance_hdf5,
    "npz": _provenance_npz,
    "csv": _provenance_csv,
}
"""How to reach the provenance block, for the formats that store one.

Only *how*. **Which** formats store one is :attr:`Exporter.carries_provenance`, and
:func:`provenance_formats` is the single answer any message should be built from — the
hardcoded ``(hdf5, npz)`` that message once contained was a second source of truth one
edit away from lying, and CSV is the edit that would have made it lie.
"""


def provenance_formats() -> tuple[str, ...]:
    """Registered formats that store circuit and DEM text at ``--structure full``."""
    return tuple(sorted(name for name, exp in EXPORTERS.items() if exp.carries_provenance))


def manifest_reader_coverage() -> set[str]:
    """Formats with a cheap manifest reader. Compared against :data:`EXPORTERS` in tests."""
    return set(_MANIFEST_READERS)


def read_manifest(path: Path, format_name: str | None = None) -> dict[str, Any]:
    """The manifest of one dataset file, without loading its shots.

    Materialising a million-shot file to print its settings defeats the purpose, and CSV
    is the worst case of all — text, one column per bit.

    Raises:
        ValueError: for an unregistered extension, or a format with no cheap reader.
        Exception: whatever the underlying library raises for a corrupt file. Callers
            must catch broadly and report, because "corrupt" arrives as ``OSError``,
            ``KeyError``, ``BadZipFile`` or ``ArrowInvalid`` depending on the format.
    """
    resolved = format_name or infer_format(path)
    reader = _MANIFEST_READERS.get(resolved)
    if reader is None:
        raise ValueError(f"no manifest reader for format {resolved!r}")
    return reader(path)


def read_provenance(path: Path, format_name: str | None = None) -> dict[str, Any] | None:
    """The provenance block of one file, or ``None`` when the format does not store one.

    Kept separate from :func:`read_manifest` on purpose, and that separation is part of
    the data contract rather than a tidiness preference: provenance is **not**
    decoder-visible, and folding it into a manifest loader would put a frozen-prior test
    file's own DEM text one attribute away from every decoder that reads manifests.
    """
    resolved = format_name or infer_format(path)
    reader = _PROVENANCE_READERS.get(resolved)
    return None if reader is None else reader(path)
