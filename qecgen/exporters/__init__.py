"""Exporter registry.

The CLI discovers formats through :data:`EXPORTERS`. A new format is one module plus
one line here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qecgen.exporters.base import Exporter
from qecgen.exporters.hdf5 import HDF5Exporter, StreamingHDF5Writer
from qecgen.exporters.jsonl import JSONLExporter
from qecgen.exporters.npz import NPZExporter
from qecgen.exporters.parquet import ParquetExporter

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "EXPORTERS",
    "Exporter",
    "HDF5Exporter",
    "JSONLExporter",
    "NPZExporter",
    "ParquetExporter",
    "StreamingHDF5Writer",
    "get_exporter",
    "infer_format",
]

EXPORTERS: dict[str, Exporter] = {
    exporter.format_name: exporter
    for exporter in (
        HDF5Exporter(),
        NPZExporter(),
        ParquetExporter(),
        JSONLExporter(),
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
