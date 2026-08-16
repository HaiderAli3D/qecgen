"""JSON Lines exporter. Human-readable, and deliberately the least efficient option.

Layout: the **first line** is ``{"__manifest__": {...}}``. When the dataset carries
structure, the **second line** is ``{"__structure__": {...}}``. Every subsequent line is
one shot::

    {"shot": 0, "detectors": "010010...", "observables": "1", "environment_id": 3}

Detector and observable strings are written most-significant-index-last, i.e.
``s[i]`` is the value of detector ``i``. That ordering is chosen to be readable, and it
is *not* the on-disk bit packing: unpacking is done with ``bitorder="little"`` before
the string is formed, so the two representations agree bit for bit. **Do not use
``int(s, 2)``** to read one: that treats the leftmost character as most-significant and
therefore reverses the detector index across the whole file.

Structure lives on a header line rather than a sidecar file because it is
**decoder-visible** (``DATA_CONTRACT.md`` puts ``dem/`` in the "may a decoder read it:
yes" row). In a sidecar, a ``.jsonl`` whose manifest records ``structure_level: dem``
could arrive without it, and ``read`` could then only lie (return no structure under a
manifest claiming some) or mutate a manifest field on read. Both are exactly what
``require_level_agreement`` exists to prevent. A reader still gets structure without
touching a single shot -- two ``readline`` calls -- and the line depends only on distance,
never on shot count: measured 63 KB at d=3, 434 KB at d=5, 1.5 MB at d=7.

**Provenance is deliberately not written, at any level, and the manifest says so.** The
idiomatic way to read this format is ``for line in f: json.loads(line)`` -- a complete,
working reader that would ingest every header line along with the shots. Under
``FROZEN_PRIOR`` the provenance block holds precisely the distribution the condition
exists to withhold, so putting it one loop away from a decoder is a hazard no other
format has in this shape. Asking for ``--structure full`` therefore produces a file whose
recorded ``structure_level`` is ``dem``: the payload is identical either way, and
recording ``full`` over a file with no circuit text in it would be the over-claim
:func:`recorded_structure_level` exists to prevent. Use ``hdf5``, ``npz`` or ``csv`` when
the circuit and DEM text have to travel with the shots.

Reader note: Go's ``bufio.Scanner`` defaults to a 64 KB token cap and fails on the
structure line of *every* d>=3 file. The fix is one call,
``scanner.Buffer(make([]byte, 0, 64<<10), 64<<20)``. Python, jq, Node and Java need
nothing.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from qecgen.dataset import DatasetMeta, InMemoryDataset, StructureLevel
from qecgen.exporters.base import recorded_structure_level, require_level_agreement
from qecgen.exporters.structure_json import (
    load_json_object,
    repack,
    structure_from_json,
    structure_to_json,
)

__all__ = [
    "JSONL_SIZE_WARNING_THRESHOLD",
    "MANIFEST_KEY",
    "STRUCTURE_KEY",
    "JSONLExporter",
]

JSONL_SIZE_WARNING_THRESHOLD = 100_000
"""Shot count above which JSONL becomes an actively bad idea."""

MANIFEST_KEY = "__manifest__"
STRUCTURE_KEY = "__structure__"
RESERVED_KEYS = frozenset({MANIFEST_KEY, STRUCTURE_KEY})
"""Top-level keys that mark a header record.

``read`` dispatches on key presence rather than line number, which gives backward
compatibility with existing manifest-then-shots files for free.
"""


class JSONLExporter:
    """One JSON object per shot. Use for inspection and fixtures, not for bulk data."""

    @property
    def format_name(self) -> str:
        """Format name for ``--format``."""
        return "jsonl"

    @property
    def extension(self) -> str:
        """File extension."""
        return ".jsonl"

    @property
    def streaming(self) -> bool:
        """Appendable in principle, but not wired as a streaming writer here."""
        return False

    @property
    def structure_round_trip(self) -> bool:
        """Structure round-trips exactly, on the ``__structure__`` header line."""
        return True

    @property
    def carries_provenance(self) -> bool:
        """No. A line-iterating reader would hand it straight to a decoder.

        See the module docstring: the consequence is that ``--structure full`` records
        ``dem``, which is the honest description of what the file holds.
        """
        return False

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        """Write one JSON object per shot, preceded by the header lines."""
        require_level_agreement(dataset, structure_level)
        if dataset.n_shots > JSONL_SIZE_WARNING_THRESHOLD:
            warnings.warn(
                f"Writing {dataset.n_shots:,} shots as JSONL. This format stores one "
                f"decimal digit per bit plus JSON punctuation, so expect roughly "
                f"{dataset.n_shots * dataset.meta.n_detectors / 1e9:.1f} GB and very slow "
                f"reads. Use hdf5 for anything above "
                f"{JSONL_SIZE_WARNING_THRESHOLD:,} shots.",
                stacklevel=2,
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        # Only the persisted copy is downgraded; the caller's dataset is untouched.
        recorded = dataclasses.replace(
            dataset.meta, structure_level=recorded_structure_level(self, structure_level)
        )

        detectors = dataset.unpacked_detectors()
        observables = dataset.unpacked_observables()
        mechanisms = (
            np.unpackbits(
                dataset.mechanisms,
                axis=1,
                count=dataset.meta.n_mechanisms or 0,
                bitorder="little",
            ).astype(bool)
            if dataset.mechanisms is not None and dataset.meta.n_mechanisms
            else None
        )

        with path.open("w", encoding="utf-8", newline="\n") as handle:
            header = {MANIFEST_KEY: recorded.to_json_dict()}
            handle.write(json.dumps(header, sort_keys=True, allow_nan=False))
            handle.write("\n")
            if structure_level is not StructureLevel.NONE and dataset.structure is not None:
                structure_line = {
                    STRUCTURE_KEY: structure_to_json(dataset.structure, structure_level)
                }
                handle.write(json.dumps(structure_line, sort_keys=True, allow_nan=False))
                handle.write("\n")
            for i in range(dataset.n_shots):
                record: dict[str, object] = {
                    "shot": i,
                    "detectors": _bits_to_string(detectors[i]),
                    "observables": _bits_to_string(observables[i]),
                }
                if dataset.environment_ids is not None:
                    record["environment_id"] = int(dataset.environment_ids[i])
                if mechanisms is not None:
                    record["mechanisms"] = _bits_to_string(mechanisms[i])
                handle.write(json.dumps(record))
                handle.write("\n")

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset written by :meth:`write`.

        Dispatches on reserved-key presence rather than line number, so a file written
        before ``__structure__`` existed reads with no special-casing.
        """
        meta: DatasetMeta | None = None
        structure_payload: dict[str, Any] | None = None
        det_rows: list[np.ndarray] = []
        obs_rows: list[np.ndarray] = []
        env_rows: list[int] = []
        mech_rows: list[np.ndarray] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{line_no}"
                record = load_json_object(line, where)

                if MANIFEST_KEY in record:
                    if meta is not None:
                        raise ValueError(
                            f"{where}: a second manifest line; files were concatenated"
                        )
                    manifest = record[MANIFEST_KEY]
                    if not isinstance(manifest, dict):
                        raise ValueError(f"{where}: {MANIFEST_KEY} must be an object")
                    meta = DatasetMeta.from_json_dict(manifest)
                    continue

                if STRUCTURE_KEY in record:
                    if meta is None:
                        raise ValueError(f"{where}: structure appears before the manifest")
                    if structure_payload is not None:
                        raise ValueError(f"{where}: a second structure line")
                    payload = record[STRUCTURE_KEY]
                    if not isinstance(payload, dict):
                        raise ValueError(f"{where}: {STRUCTURE_KEY} must be an object")
                    structure_payload = payload
                    continue

                if meta is None:
                    raise ValueError(f"{where}: a shot appears before the manifest")
                det_rows.append(_string_to_bits(str(record["detectors"])))
                obs_rows.append(_string_to_bits(str(record["observables"])))
                if "environment_id" in record:
                    env_rows.append(int(record["environment_id"]))
                if "mechanisms" in record:
                    mech_rows.append(_string_to_bits(str(record["mechanisms"])))

        if meta is None:
            raise ValueError(f"{path}: no {MANIFEST_KEY} line found")
        if meta.structure_level is not StructureLevel.NONE and structure_payload is None:
            raise ValueError(
                f"{path}: the manifest records structure_level={meta.structure_level} but "
                f"the file has no {STRUCTURE_KEY} line. Downstream that is "
                "indistinguishable from a corrupt file, so it is refused rather than "
                "silently downgraded."
            )

        detectors = repack(det_rows, meta.n_detectors)
        observables = repack(obs_rows, meta.n_observables)
        mechanisms = repack(mech_rows, meta.n_mechanisms or 0) if mech_rows else None
        environment_ids = np.asarray(env_rows, dtype=np.int32) if env_rows else None
        return InMemoryDataset(
            detectors=detectors,
            observables=observables,
            meta=meta,
            environment_ids=environment_ids,
            mechanisms=mechanisms,
            structure=(
                None if structure_payload is None else structure_from_json(structure_payload, meta)
            ),
        )


def _bits_to_string(bits: np.ndarray) -> str:
    """Render a bool row as a ``0``/``1`` string, index-ordered."""
    return "".join("1" if b else "0" for b in bits)


def _string_to_bits(text: str) -> np.ndarray:
    """Parse a ``0``/``1`` string back into a bool row.

    Strict: any other character is an error rather than being folded to zero, which
    would turn a corrupt or foreign file into plausible-looking all-zero shots.
    """
    raw = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    valid = (raw == ord("0")) | (raw == ord("1"))
    if not valid.all():
        bad = sorted({chr(c) for c in raw[~valid]})
        raise ValueError(f"bit string contains non-0/1 characters {bad}: {text[:40]!r}")
    return np.asarray(raw == ord("1"))
