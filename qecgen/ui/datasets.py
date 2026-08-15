"""Browsing datasets that already exist on disk.

Two rules shape this module.

**Never list a file by reading all of it.** Every exporter's ``read()`` materialises the
whole dataset; doing that to render a directory listing would load gigabytes to show a
row. Each format stores its manifest somewhere cheap — an HDF5 root attribute, a Parquet
footer, an NPZ member, a JSONL first line — so listing reads only that.

**Never present a broken file as a dataset, and never hide one.** A worker killed
mid-write leaves something that opens (HDF5 without a manifest attribute) or even reads
cleanly (JSONL, whose manifest is line 1, so a truncation only shows up as a shot count
that disagrees with the rows). Unreadable entries are listed with the reason attached;
they are never silently skipped and never shown as finished. The manifest's ``shots`` is
reported as a *claim*, and ``validate`` is what turns it into a fact.

Circuit and DEM text never appear here. ``DatasetMeta.to_json_dict()`` excludes them and
``provenance_dict()`` is not called anywhere in this package — under ``FROZEN_PRIOR`` that
text is precisely what the condition withholds from a decoder, and a browser rendering it
beside a test file would hand it back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qecgen.dataset import DatasetMeta
from qecgen.exporters import EXPORTERS, get_exporter, infer_format
from qecgen.run import PARTIAL_PREFIX
from qecgen.validate import validate_dataset

__all__ = [
    "DatasetEntry",
    "PathOutsideRootError",
    "full_manifest",
    "list_datasets",
    "read_manifest",
    "resolve_within",
    "validate_at",
]


class PathOutsideRootError(ValueError):
    """A requested path escapes the directory the server is allowed to touch."""


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and require it to sit under ``root``.

    The path arrives from a browser, so ``../../..`` and absolute paths elsewhere on the
    machine both have to be refused. Resolving first and comparing afterwards is what
    makes symlinks and ``..`` segments harmless; comparing the strings first would not.
    """
    root_resolved = root.resolve()
    target = Path(candidate)
    resolved = (target if target.is_absolute() else root_resolved / target).resolve()
    if resolved == root_resolved:
        # The root itself is not a valid target either: the run writers derive their
        # staging directory from `out.parent`, and the root's *parent* is exactly the
        # directory this function exists to keep the server out of. Accepting `.` here
        # meant a run staged its scratch directory outside the confined root — where a
        # killed worker orphaned it forever, because sweep_partials(root) only sweeps
        # below the root.
        raise PathOutsideRootError(
            f"{candidate!r} resolves to the data root itself; name a file or subdirectory below it"
        )
    if root_resolved not in resolved.parents:
        raise PathOutsideRootError(
            f"{candidate!r} resolves outside the data root {root_resolved}; the UI only "
            "reads and writes below that directory"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """One file in the browser listing."""

    path: str
    name: str
    format_name: str
    size_bytes: int
    modified_at: float
    manifest: dict[str, Any] | None
    unreadable: str | None

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {
            "path": self.path,
            "name": self.name,
            "format": self.format_name,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "manifest": self.manifest,
            "unreadable": self.unreadable,
        }


def _manifest_hdf5(path: Path) -> dict[str, Any]:
    from qecgen.exporters.hdf5 import read_manifest_only

    return dict(read_manifest_only(path))


def _manifest_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    # np.load on a zip is lazy, so this decompresses one member rather than every array.
    with np.load(path, allow_pickle=False) as data:
        raw = json.loads(str(data["manifest"].item()))
    return dict(raw)


def _manifest_parquet(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    return dict(json.loads(metadata[b"qecgen_manifest"].decode("utf-8")))


def _manifest_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    payload = json.loads(first)
    return dict(payload["__manifest__"])


_MANIFEST_READERS = {
    "hdf5": _manifest_hdf5,
    "npz": _manifest_npz,
    "parquet": _manifest_parquet,
    "jsonl": _manifest_jsonl,
}


def read_manifest(path: Path) -> dict[str, Any]:
    """The manifest of one dataset file, without loading its shots.

    Raises:
        ValueError: for an unregistered extension, or a format with no cheap reader.
        Exception: whatever the underlying library raises for a corrupt file — the caller
            is expected to catch broadly and report, because "corrupt" arrives as
            ``OSError``, ``KeyError``, ``BadZipFile`` or ``ArrowInvalid`` depending on
            the format.
    """
    format_name = infer_format(path)
    reader = _MANIFEST_READERS.get(format_name)
    if reader is None:  # pragma: no cover - unreachable while the registry is covered
        raise ValueError(f"no manifest reader for format {format_name!r}")
    return reader(path)


def _summarise(raw: dict[str, Any]) -> dict[str, Any]:
    """Trim a manifest to what a listing row needs.

    The full manifest is available per-file; a listing of fifty files does not need fifty
    copies of every environment's channel vector.
    """
    environments = raw.get("environments") or []
    return {
        "distance": raw.get("distance"),
        "rounds": raw.get("rounds"),
        "basis": raw.get("basis"),
        "shots": raw.get("shots"),
        "n_detectors": raw.get("n_detectors"),
        "n_observables": raw.get("n_observables"),
        "contract": raw.get("contract"),
        "structure_level": raw.get("structure_level"),
        "drift_condition": raw.get("drift_condition"),
        "drift_axis": raw.get("drift_axis"),
        "n_environments": len(environments),
        "content_hash": raw.get("content_hash"),
        "generated_at": raw.get("generated_at"),
    }


def list_datasets(root: Path) -> list[DatasetEntry]:
    """Every dataset file under ``root``, newest first.

    Staging directories are skipped: their contents are mid-write by definition, and a
    run that is still going should not have its half-written output offered for download.
    """
    if not root.is_dir():
        return []
    known = {exporter.extension for exporter in EXPORTERS.values()}
    entries: list[DatasetEntry] = []
    for path in root.rglob("*"):
        if path.suffix not in known or not path.is_file():
            continue
        if any(part.startswith(PARTIAL_PREFIX) for part in path.relative_to(root).parts):
            continue
        stat = path.stat()
        manifest: dict[str, Any] | None = None
        unreadable: str | None = None
        try:
            manifest = _summarise(read_manifest(path))
        except Exception as exc:
            unreadable = f"{type(exc).__name__}: {exc}"
        entries.append(
            DatasetEntry(
                path=str(path.relative_to(root)).replace("\\", "/"),
                name=path.name,
                format_name=infer_format(path),
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
                manifest=manifest,
                unreadable=unreadable,
            )
        )
    entries.sort(key=lambda entry: entry.modified_at, reverse=True)
    return entries


def full_manifest(path: Path) -> dict[str, Any]:
    """The complete manifest, round-tripped through :class:`DatasetMeta`.

    Round-tripping rather than returning the raw JSON so the response is exactly the
    decoder-visible view: anything the manifest schema does not carry cannot leak through
    a stray key someone added to a file by hand.
    """
    return DatasetMeta.from_json_dict(read_manifest(path)).to_json_dict()


def validate_at(path: Path) -> dict[str, Any]:
    """Fully read a dataset and run the structural checks over it.

    The expensive endpoint, and the only one that can catch a truncated JSONL — its
    manifest is on line 1, so a file cut short still *reads*, and only the
    ``manifest.shots`` and ``content_hash`` checks notice the rows are missing.
    """
    dataset = get_exporter(infer_format(path)).read(path)
    report = validate_dataset(dataset)
    return {
        "ok": report.ok,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "requirement": check.requirement,
            }
            for check in report.results
        ],
    }
