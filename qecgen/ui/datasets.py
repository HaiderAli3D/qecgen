"""Browsing datasets that already exist on disk.

Two rules shape this module.

**Never list a file by reading all of it.** Every exporter's ``read()`` materialises the
whole dataset; doing that to render a directory listing would load gigabytes to show a
row. Each format stores its manifest somewhere cheap — an HDF5 root attribute, a Parquet
footer, an NPZ member, a JSONL first line, a CSV comment header — so listing reads only
that.

**Never present a broken file as a dataset, and never hide one.** A worker killed
mid-write leaves something that opens (HDF5 with its array datasets but no manifest
attribute) or even reads cleanly (JSONL and CSV, whose manifests sit in the header, so a
truncation only shows up as a shot count that disagrees with the rows). Unreadable entries
are listed with the reason attached; they are never silently skipped and never shown as
finished. The manifest's ``shots`` is reported as a *claim*, and ``validate`` is what turns
it into a fact.

**A file is not a dataset just because it has the extension**, and this applies to all
five formats, not only to CSV. ``qecgen sweep`` writes a threshold-results table as
``.csv``; ``qecgen score`` reads a proposed correction from an ``.npz``; and ``.parquet``,
``.jsonl`` and ``.h5`` are general-purpose formats a data root may hold for any reason at
all. Every one of those is listed as *not a qecgen dataset* rather than as unreadable.
That is not a third way of hiding a file — it is still listed, sized and dated — and it is
what keeps the corruption flag meaning corruption. A flag that fires on every healthy
foreign file is one a reader learns to skip, which is precisely how the flag that means
"a worker died mid-write" stops being seen.

Each format's signal is **provable** rather than heuristic; the cases are enumerated on
:class:`~qecgen.exporters.base.NotAQecgenDatasetError`. The one worth repeating here is
HDF5, because it is the only format where the missing manifest is genuinely ambiguous:
``StreamingHDF5Writer.abort()`` leaves exactly that. The array datasets are what separate
the two — created on the first append, while the manifest is written only on close — so a
manifest-less HDF5 *with* ``detectors`` is an interrupted run and stays corruption, and
one without is somebody else's file.

Circuit and DEM text never appear here. ``DatasetMeta.to_json_dict()`` excludes them and
``provenance_dict()`` is not called anywhere in this package — under ``FROZEN_PRIOR`` that
text is precisely what the condition withholds from a decoder, and a browser rendering it
beside a test file would hand it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qecgen.dataset import DatasetMeta
from qecgen.exporters import (
    EXPORTERS,
    NotAQecgenDatasetError,
    get_exporter,
    infer_format,
    read_manifest,
)
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
    not_a_dataset: str | None
    """Set when the file carries a dataset extension but this tool did not write it.

    Distinct from ``unreadable`` on purpose; see the module docstring.
    """

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
            "not_a_dataset": self.not_a_dataset,
        }


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
        not_a_dataset: str | None = None
        try:
            manifest = _summarise(read_manifest(path))
        except NotAQecgenDatasetError as exc:
            # Intact, just not ours. Flagging a good sweep results table as corruption
            # trains the reader to skip the flag, which is the failure the flag exists
            # to prevent.
            not_a_dataset = str(exc)
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
                not_a_dataset=not_a_dataset,
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

    The expensive endpoint, and the only one that can catch a truncated JSONL or CSV —
    both put the manifest in the header, so a file cut short still *reads*, and only the
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
