"""Crash-safe chunk store for the residual pipeline: resume from verified chunks or refuse.

A residual build decodes hundreds of thousands of shots and must survive a crash halfway
without ever appending to a file that carries a final name. Each stage therefore lands in
its own chunk file under ``data/residual/.checkpoints/<name>/``, and ``checkpoint.json``
indexes what is complete together with the *identity* of the run that produced it. Three
traps shape the layout:

* Chunk files use the unregistered extension ``.chk``. ``ui/datasets.list_datasets`` walks
  the whole data root and reads any registered extension as a dataset, so a checkpoint
  written as ``.npz`` would be listed as an interrupted qecgen run. ``np.savez`` given a
  *path* appends ``.npz`` on its own (the ``NPZExporter`` trap in ``CLAUDE.md``), so the
  writer passes an open handle, which numpy leaves alone.
* ``checkpoint.json`` is rewritten through a sibling temp file and ``os.replace``, and only
  *after* the chunk file it indexes has itself been replaced into place. The index can
  therefore lag the files (a chunk is recomputed) but never lead them: an indexed chunk
  with no file is external damage and is refused rather than repaired.
* Chunks from different configurations, sources, decoders, schemas, seeds or software
  versions must never be merged. The identity is compared field by field on open and the
  error names every differing field, because "checkpoint mismatch" alone sends the user
  hunting through five hashes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

INDEX_FILENAME = "checkpoint.json"
CHUNK_SUFFIX = ".chk"
INDEX_FORMAT = "qecgen-residual-checkpoint"
INDEX_FORMAT_VERSION = 1

_RAW_KEYS = ("detectors", "observables")
_FEATURE_KEYS = ("features", "pm_guess", "pm_weight", "truth", "pm_wrong")


class CheckpointError(Exception):
    """Base class for every refusal the chunk store raises."""


class CheckpointIdentityError(CheckpointError):
    """The stored checkpoint was produced by a different run; names every differing field."""

    def __init__(self, root: Path, fields: tuple[str, ...], details: dict[str, Any]) -> None:
        self.root = root
        self.fields = fields
        self.details = details
        rendered = "; ".join(
            f"{name}: stored={details[name]['stored']!r} current={details[name]['current']!r}"
            for name in fields
        )
        super().__init__(
            f"checkpoint at {root} belongs to a different run "
            f"(differing fields: {', '.join(fields)}). {rendered}. "
            "Discard it explicitly (build --fresh) or fix the configuration; "
            "chunks from different runs are never merged."
        )


class ChunkIntegrityError(CheckpointError):
    """A chunk file is missing or its bytes no longer match the recorded sha256."""


class ChunkRangeError(CheckpointError):
    """A chunk's index or row range conflicts with what the index already records."""


@dataclass(frozen=True)
class CheckpointIdentity:
    """Everything that must be equal before two chunk sets may be combined.

    A plain frozen dataclass with no dependency on the other residual modules so that a
    resume check can run before any circuit, DEM or matcher is rebuilt.
    """

    config_hash: str
    source_hash: str
    decoder_dem_sha256: str
    graph_digest: str
    schema_version: int
    versions: dict[str, str]
    seed: int | None
    chunk_size: int
    shots: int

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["versions"] = dict(sorted(self.versions.items()))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointIdentity:
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - names)
        missing = sorted(names - set(payload))
        if unknown or missing:
            raise CheckpointError(
                f"checkpoint identity keys do not match: unknown={unknown} missing={missing}"
            )
        seed = payload["seed"]
        return cls(
            config_hash=str(payload["config_hash"]),
            source_hash=str(payload["source_hash"]),
            decoder_dem_sha256=str(payload["decoder_dem_sha256"]),
            graph_digest=str(payload["graph_digest"]),
            schema_version=int(payload["schema_version"]),
            versions={str(k): str(v) for k, v in payload["versions"].items()},
            seed=None if seed is None else int(seed),
            chunk_size=int(payload["chunk_size"]),
            shots=int(payload["shots"]),
        )

    def differing_fields(self, other: CheckpointIdentity) -> tuple[str, ...]:
        """Field names, in declaration order, whose values differ between the two."""
        return tuple(
            field.name
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        )


@dataclass(frozen=True)
class ChunkRecord:
    """One completed chunk: its row range and the sha256 of each stage's file."""

    index: int
    row_start: int
    row_count: int
    raw_sha256: str
    feat_sha256: str | None
    pm_wrong_count: int | None

    @property
    def row_end(self) -> int:
        return self.row_start + self.row_count

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChunkRecord:
        feat = payload["feat_sha256"]
        count = payload["pm_wrong_count"]
        return cls(
            index=int(payload["index"]),
            row_start=int(payload["row_start"]),
            row_count=int(payload["row_count"]),
            raw_sha256=str(payload["raw_sha256"]),
            feat_sha256=None if feat is None else str(feat),
            pm_wrong_count=None if count is None else int(count),
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _require_binary(name: str, values: np.ndarray) -> np.ndarray:
    out = np.asarray(values)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if out.size and not np.isin(out, (0, 1)).all():
        raise ValueError(f"{name} must be binary (0/1)")
    return out.astype(np.uint8)


class ChunkStore:
    """Per-dataset chunk files plus an atomically rewritten index.

    ``root`` is ``data/residual/.checkpoints/<name>``. Opening a root that holds an index
    written under a different identity raises ``CheckpointIdentityError`` before any chunk
    is touched; the caller decides whether to ``discard()``.
    """

    def __init__(self, root: Path, identity: CheckpointIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self._records: dict[int, ChunkRecord] = {}
        self._packed_widths: dict[str, int] | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self._load()

    # -- paths --------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    def raw_path(self, index: int) -> Path:
        return self.root / f"raw_chunk_{index:05d}{CHUNK_SUFFIX}"

    def feat_path(self, index: int) -> Path:
        return self.root / f"feat_chunk_{index:05d}{CHUNK_SUFFIX}"

    # -- index ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("format") != INDEX_FORMAT:
            raise CheckpointError(
                f"{self.index_path} is not a residual checkpoint index "
                f"(format={payload.get('format')!r})"
            )
        if payload.get("format_version") != INDEX_FORMAT_VERSION:
            raise CheckpointError(
                f"{self.index_path} has index format version "
                f"{payload.get('format_version')!r}; this code reads {INDEX_FORMAT_VERSION}"
            )
        stored = CheckpointIdentity.from_dict(payload["identity"])
        differing = stored.differing_fields(self.identity)
        if differing:
            details = {
                name: {
                    "stored": getattr(stored, name),
                    "current": getattr(self.identity, name),
                }
                for name in differing
            }
            raise CheckpointIdentityError(self.root, differing, details)
        widths = payload.get("packed_widths")
        self._packed_widths = None if widths is None else {k: int(v) for k, v in widths.items()}
        records = [ChunkRecord.from_dict(item) for item in payload["chunks"]]
        self._records = {}
        for record in records:
            self._check_range(record.index, record.row_start, record.row_count)
            # An indexed chunk whose file is gone cannot be a crash artifact: the index is
            # only rewritten after the file is in place. Refuse instead of "recomputing",
            # because whatever removed it may have touched the others too.
            if not self.raw_path(record.index).exists():
                raise ChunkIntegrityError(
                    f"index lists chunk {record.index} but {self.raw_path(record.index)} "
                    "is missing; the checkpoint directory was modified outside the pipeline"
                )
            if record.feat_sha256 is not None and not self.feat_path(record.index).exists():
                raise ChunkIntegrityError(
                    f"index lists features for chunk {record.index} but "
                    f"{self.feat_path(record.index)} is missing"
                )
            self._records[record.index] = record

    def _write_index(self) -> None:
        payload = {
            "format": INDEX_FORMAT,
            "format_version": INDEX_FORMAT_VERSION,
            "identity": self.identity.to_dict(),
            "packed_widths": self._packed_widths,
            "chunks": [self._records[i].to_dict() for i in sorted(self._records)],
            "aggregate": self.aggregate(),
            "updated_at": _now(),
        }
        data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        _atomic_write(self.index_path, data)

    # -- queries ----------------------------------------------------------------

    def completed_raw(self) -> dict[int, ChunkRecord]:
        return dict(self._records)

    def completed_features(self) -> dict[int, ChunkRecord]:
        return {i: r for i, r in self._records.items() if r.feat_sha256 is not None}

    def aggregate(self) -> dict[str, int]:
        """Partial totals a resumed run can print before it recomputes anything."""
        feature_records = [r for r in self._records.values() if r.feat_sha256 is not None]
        return {
            "raw_rows": sum(r.row_count for r in self._records.values()),
            "feature_rows": sum(r.row_count for r in feature_records),
            "pm_wrong_total": sum(r.pm_wrong_count or 0 for r in feature_records),
        }

    def contiguous_completed(self, *, stage: Literal["raw", "features"] = "raw") -> tuple[int, ...]:
        """Indices ``0..k`` completed for ``stage`` with no gap in indices or rows.

        Row contiguity is asserted rather than assumed: a resumed run that changed
        ``checkpoint_rows`` would otherwise stitch chunks that skip or repeat rows while
        the index sequence still looks complete.
        """
        records = self._records if stage == "raw" else self.completed_features()
        indices: list[int] = []
        expected_start = 0
        for index in range(len(records)):
            record = records.get(index)
            if record is None:
                break
            if record.row_start != expected_start:
                raise ChunkRangeError(
                    f"chunk {index} starts at row {record.row_start} but the preceding "
                    f"chunks end at row {expected_start}; ranges are not contiguous"
                )
            indices.append(index)
            expected_start = record.row_end
        return tuple(indices)

    # -- range checks -------------------------------------------------------------

    def _check_range(self, index: int, row_start: int, row_count: int) -> None:
        if index < 0:
            raise ChunkRangeError(f"chunk index must be >= 0, got {index}")
        if row_start < 0:
            raise ChunkRangeError(f"row_start must be >= 0, got {row_start}")
        row_end = row_start + row_count
        existing = self._records.get(index)
        if existing is not None and (existing.row_start, existing.row_count) != (
            row_start,
            row_count,
        ):
            raise ChunkRangeError(
                f"chunk index {index} already covers rows "
                f"[{existing.row_start}, {existing.row_end}); refusing to record it as "
                f"[{row_start}, {row_end})"
            )
        for other in self._records.values():
            if other.index == index:
                continue
            if row_start < other.row_end and other.row_start < row_end:
                raise ChunkRangeError(
                    f"chunk {index} rows [{row_start}, {row_end}) overlap chunk "
                    f"{other.index} rows [{other.row_start}, {other.row_end})"
                )

    # -- raw chunks ---------------------------------------------------------------

    def write_raw(
        self, index: int, row_start: int, detectors: np.ndarray, observables: np.ndarray
    ) -> ChunkRecord:
        detectors = np.asarray(detectors)
        observables = np.asarray(observables)
        for name, array in (("detectors", detectors), ("observables", observables)):
            if array.dtype != np.uint8:
                raise ValueError(f"{name} must be packed uint8, got dtype {array.dtype}")
            if array.ndim != 2:
                raise ValueError(f"{name} must be 2-D (rows, bytes), got shape {array.shape}")
        if detectors.shape[0] != observables.shape[0]:
            raise ValueError(
                f"detectors has {detectors.shape[0]} rows but observables has "
                f"{observables.shape[0]} rows"
            )
        row_count = int(detectors.shape[0])
        if row_count == 0:
            raise ValueError("a raw chunk must hold at least one row; zero rows refused")
        widths = {"detectors": int(detectors.shape[1]), "observables": int(observables.shape[1])}
        if self._packed_widths is not None and widths != self._packed_widths:
            raise ValueError(
                f"packed width {widths} differs from the store's {self._packed_widths}; "
                "chunks of different widths cannot belong to one dataset"
            )
        self._check_range(index, row_start, row_count)

        digest = _write_npz(
            self.raw_path(index),
            {
                "detectors": detectors,
                "observables": observables,
                "row_start": np.asarray(row_start, dtype=np.int64),
                "index": np.asarray(index, dtype=np.int64),
            },
        )
        # A rewritten raw chunk invalidates any features derived from the previous bytes.
        stale = self.feat_path(index)
        if stale.exists():
            stale.unlink()
        record = ChunkRecord(
            index=index,
            row_start=row_start,
            row_count=row_count,
            raw_sha256=digest,
            feat_sha256=None,
            pm_wrong_count=None,
        )
        self._records[index] = record
        self._packed_widths = widths
        self._write_index()
        return record

    def read_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self._records.get(index)
        if record is None:
            raise ChunkRangeError(f"no raw chunk {index} is recorded in {self.index_path}")
        arrays = _read_npz(self.raw_path(index), record.raw_sha256, _RAW_KEYS)
        detectors = np.asarray(arrays["detectors"], dtype=np.uint8)
        observables = np.asarray(arrays["observables"], dtype=np.uint8)
        if detectors.shape[0] != record.row_count or observables.shape[0] != record.row_count:
            raise ChunkIntegrityError(
                f"{self.raw_path(index)} holds {detectors.shape[0]}/{observables.shape[0]} "
                f"rows but the index records {record.row_count}"
            )
        return detectors, observables

    # -- feature chunks -----------------------------------------------------------

    def write_features(
        self,
        index: int,
        features: np.ndarray,
        pm_guess: np.ndarray,
        pm_weight: np.ndarray,
        truth: np.ndarray,
        pm_wrong: np.ndarray,
    ) -> ChunkRecord:
        record = self._records.get(index)
        if record is None:
            raise ChunkRangeError(
                f"no raw chunk {index} is recorded; features are derived from a raw chunk "
                "and cannot precede it"
            )
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D (rows, columns), got shape {features.shape}")
        pm_weight = np.asarray(pm_weight, dtype=np.float64)
        if pm_weight.ndim != 1:
            raise ValueError(f"pm_weight must be 1-D, got shape {pm_weight.shape}")
        guess = _require_binary("pm_guess", pm_guess)
        truth_bits = _require_binary("truth", truth)
        wrong = _require_binary("pm_wrong", pm_wrong)
        lengths = {
            "features": features.shape[0],
            "pm_guess": guess.shape[0],
            "pm_weight": pm_weight.shape[0],
            "truth": truth_bits.shape[0],
            "pm_wrong": wrong.shape[0],
        }
        if any(n != record.row_count for n in lengths.values()):
            raise ValueError(
                f"feature arrays must have {record.row_count} rows like raw chunk {index}, "
                f"got {lengths}"
            )
        if not np.array_equal(wrong, (guess != truth_bits).astype(np.uint8)):
            raise ValueError("pm_wrong must equal (pm_guess != truth) for every row")
        if not np.isfinite(features).all() or not np.isfinite(pm_weight).all():
            raise ValueError("features and pm_weight must be finite")

        digest = _write_npz(
            self.feat_path(index),
            {
                "features": features,
                "pm_guess": guess,
                "pm_weight": pm_weight,
                "truth": truth_bits,
                "pm_wrong": wrong,
                "row_start": np.asarray(record.row_start, dtype=np.int64),
                "index": np.asarray(index, dtype=np.int64),
            },
        )
        updated = dataclasses.replace(
            record, feat_sha256=digest, pm_wrong_count=int(wrong.sum(dtype=np.int64))
        )
        self._records[index] = updated
        self._write_index()
        return updated

    def read_features(self, index: int) -> dict[str, np.ndarray]:
        record = self._records.get(index)
        if record is None or record.feat_sha256 is None:
            raise ChunkRangeError(f"no feature chunk {index} is recorded in {self.index_path}")
        arrays = _read_npz(self.feat_path(index), record.feat_sha256, _FEATURE_KEYS)
        out: dict[str, np.ndarray] = {
            "features": np.asarray(arrays["features"], dtype=np.float64),
            "pm_guess": np.asarray(arrays["pm_guess"], dtype=np.uint8),
            "pm_weight": np.asarray(arrays["pm_weight"], dtype=np.float64),
            "truth": np.asarray(arrays["truth"], dtype=np.uint8),
            "pm_wrong": np.asarray(arrays["pm_wrong"], dtype=np.uint8),
        }
        if any(v.shape[0] != record.row_count for v in out.values()):
            raise ChunkIntegrityError(
                f"{self.feat_path(index)} row counts disagree with the index ({record.row_count})"
            )
        return out

    # -- discard --------------------------------------------------------------------

    def discard(self) -> None:
        """Delete this store's own files only.

        ``inventory.json`` and ``pilot.json`` share the directory and are records of
        *other* stages; wiping the directory would destroy the pilot verdict that gates
        the very build that is being restarted.
        """
        for path in self.root.iterdir():
            if path.is_file() and (
                path.name == INDEX_FILENAME
                or path.suffix == CHUNK_SUFFIX
                or path.name.endswith(".tmp")
            ):
                path.unlink()
        self._records = {}
        self._packed_widths = None


# -- file helpers ---------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to a sibling temp file, fsync, then ``os.replace`` onto ``path``.

    The temp file is removed if anything fails before the replace, so a failed commit
    leaves the previous version intact and nothing half-written beside it.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Save ``arrays`` in npz format to ``path`` (extension preserved) and return its sha256.

    ``np.savez`` is handed an open handle, never a path: given a path it appends ``.npz``
    and the subsequent ``os.replace`` would target a file that does not exist.
    """
    # Typed as Any because numpy's ``savez`` stub matches a ``**dict[str, ndarray]`` unpack
    # against its ``allow_pickle: bool`` keyword; the value is passed explicitly anyway, and
    # nothing here is an object array.
    payload: dict[str, Any] = dict(arrays)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, allow_pickle=False, **payload)
            fh.flush()
            os.fsync(fh.fileno())
        digest = _sha256(tmp.read_bytes())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return digest


def _read_npz(path: Path, expected_sha256: str, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Load a chunk file after verifying its bytes against the recorded sha256."""
    if not path.exists():
        raise ChunkIntegrityError(f"{path} is missing")
    data = path.read_bytes()
    actual = _sha256(data)
    if actual != expected_sha256:
        raise ChunkIntegrityError(
            f"{path} sha256 {actual} does not match the recorded {expected_sha256}; "
            "the chunk was modified after it was indexed"
        )
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        present = set(npz.files)
        missing = [k for k in keys if k not in present]
        if missing:
            raise ChunkIntegrityError(f"{path} lacks arrays {missing}")
        return {k: np.asarray(npz[k]) for k in keys}
