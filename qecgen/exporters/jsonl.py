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

Reader note: Go's ``bufio.Scanner`` defaults to a 64 KB token cap and fails on the
structure line of *every* d>=3 file. The fix is one call,
``scanner.Buffer(make([]byte, 0, 64<<10), 64<<20)``. Python, jq, Node and Java need
nothing.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from scipy.sparse import csc_matrix

from qecgen.dataset import DatasetMeta, InMemoryDataset, StructureLevel
from qecgen.dem import DemComponent, DemStructure
from qecgen.exporters.base import require_level_agreement
from qecgen.sampling import packed_width

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


def _reject_json_constant(name: str) -> NoReturn:
    """Refuse ``NaN``/``Infinity`` tokens on the way in.

    ``json.loads`` *accepts* them by default even though they are not valid JSON, so a
    foreign or hand-edited file would otherwise seed NaN silently into ``priors``. Same
    posture as ``dataset._strict_bool``.
    """
    raise ValueError(
        f"file contains the non-standard JSON constant {name!r}. qecgen writes strict "
        "JSON (allow_nan=False) and encodes a missing coordinate as null, so this file "
        "was not written by qecgen or has been edited."
    )


def _load_json_object(line: str, where: str) -> dict[str, Any]:
    """Parse one line into a dict, the single choke point for ``Any`` from ``json``.

    Everything downstream sees ``object``, so mypy forces narrowing rather than merely
    permitting it.
    """
    raw = json.loads(line, parse_constant=_reject_json_constant)
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(raw).__name__}")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{where}: JSON object keys must all be strings")
    return dict(raw)


def _int_list(payload: dict[str, Any], key: str) -> list[int]:
    """Narrow a JSON array to ``list[int]``, rejecting bools.

    ``isinstance(True, int)`` is True, so without an explicit bool guard a JSON
    ``[true, false]`` would become CSC indices ``[1, 0]`` and produce a well-formed,
    wrong matrix.
    """
    values = payload[key]
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON array, got {type(values).__name__}")
    out: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must contain integers, found {value!r}")
        out.append(value)
    return out


def _float_list(payload: dict[str, Any], key: str) -> list[float]:
    """Narrow a JSON array to ``list[float]``. Accepts int (JSON ``0`` means ``0.0``)."""
    values = payload[key]
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON array, got {type(values).__name__}")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key} must contain numbers, found {value!r}")
        out.append(float(value))
    return out


def _coords_to_json(coords: np.ndarray) -> list[list[float | None]]:
    """Encode detector coordinates, mapping NaN padding to JSON ``null``.

    ``dem.parse_dem`` pads ragged coordinates with NaN. ``allow_nan=False`` would refuse
    to write correct data, and ``allow_nan=True`` would emit a bare ``NaN`` token that is
    not valid JSON and is rejected by Go and serde_json. ``null`` is valid, exact, and
    self-documenting: this detector has no coordinate on this axis.
    """
    out: list[list[float | None]] = []
    for row in np.asarray(coords, dtype=np.float64):
        encoded: list[float | None] = []
        for value in row:
            number = float(value)
            if np.isnan(number):
                encoded.append(None)
            elif np.isinf(number):
                raise ValueError(
                    "detector coordinate is infinite, which has no legitimate meaning "
                    "here and cannot be written as standard JSON"
                )
            else:
                encoded.append(number)
        out.append(encoded)
    return out


def _coords_from_json(value: Any, n_detectors: int, coord_dim: int) -> np.ndarray:
    """Decode coordinates, mapping ``null`` back to NaN."""
    if not isinstance(value, list):
        raise ValueError("detector_coords must be a JSON array")
    rows = [[np.nan if item is None else float(item) for item in row] for row in value]
    # np.asarray([]) gives shape (0,), not (0, 0), so reshape explicitly.
    return np.array(rows, dtype=np.float64).reshape(n_detectors, coord_dim)


def _structure_to_json(structure: DemStructure, level: StructureLevel) -> dict[str, Any]:
    """Encode a :class:`DemStructure` for the ``__structure__`` header line."""
    payload: dict[str, Any] = {
        "level": str(level),
        "n_detectors": structure.n_detectors,
        "n_observables": structure.n_observables,
        "n_mechanisms": structure.n_mechanisms,
        "coord_dim": structure.coord_dim,
        "detector_coords": _coords_to_json(structure.detector_coords),
    }
    if level is StructureLevel.COORDS:
        return payload

    for name, matrix in (("H", structure.h), ("L", structure.l)):
        csc = matrix.tocsc()
        # `data` is omitted exactly as HDF5 omits it: H and L are binary incidence
        # matrices, so every stored entry is 1 by construction. Writing tens of thousands
        # of literal 1s would also create a second place where that invariant is stated,
        # which could then disagree with indices/indptr.
        payload[f"{name}_indices"] = [int(v) for v in csc.indices]
        payload[f"{name}_indptr"] = [int(v) for v in csc.indptr]
        payload[f"{name}_shape"] = [int(csc.shape[0]), int(csc.shape[1])]

    payload["priors"] = [float(v) for v in structure.priors]
    # Components nest naturally here. HDF5 and NPZ flatten them into value/offset pairs
    # only because those formats have no ragged type; JSON does, and the offset form
    # carries a specific silent failure -- an off-by-one re-segments every component after
    # it while still round-tripping, because the read side is symmetrically wrong.
    payload["components"] = [
        {
            "parent_mechanism_id": component.parent_mechanism_id,
            "component_index": component.component_index,
            "detectors": list(component.detectors),
            "observables": list(component.observables),
        }
        for component in structure.components
    ]
    return payload


def _structure_from_json(payload: dict[str, Any], meta: DatasetMeta) -> DemStructure:
    """Decode the ``__structure__`` header line, cross-checking it against the manifest."""
    n_detectors = int(payload["n_detectors"])
    n_observables = int(payload["n_observables"])
    n_mechanisms = int(payload["n_mechanisms"])
    coord_dim = int(payload["coord_dim"])
    if n_detectors != meta.n_detectors or n_observables != meta.n_observables:
        raise ValueError(
            f"structure declares {n_detectors} detectors and {n_observables} observables, "
            f"but the manifest declares {meta.n_detectors} and {meta.n_observables}. The "
            "redundancy is deliberate so that a disagreement is detectable rather than "
            "silently resolved in favour of one of them."
        )
    coords = _coords_from_json(payload["detector_coords"], n_detectors, coord_dim)

    if "H_indices" not in payload:
        return DemStructure(
            h=csc_matrix((n_detectors, n_mechanisms), dtype=np.uint8),
            l=csc_matrix((n_observables, n_mechanisms), dtype=np.uint8),
            priors=np.zeros(0, dtype=np.float64),
            components=(),
            detector_coords=coords,
            n_detectors=n_detectors,
            n_observables=n_observables,
            n_mechanisms=n_mechanisms,
            coord_dim=coord_dim,
            has_matrices=False,
            metadata={"structure_level": "coords"},
        )

    matrices: dict[str, csc_matrix] = {}
    for name, rows in (("H", n_detectors), ("L", n_observables)):
        indices = np.asarray(_int_list(payload, f"{name}_indices"), dtype=np.int32)
        indptr = np.asarray(_int_list(payload, f"{name}_indptr"), dtype=np.int64)
        shape = _int_list(payload, f"{name}_shape")
        if shape != [rows, n_mechanisms]:
            raise ValueError(
                f"{name}_shape is {shape}, which disagrees with the declared "
                f"({rows}, {n_mechanisms})"
            )
        matrices[name] = csc_matrix(
            (np.ones(indices.shape[0], dtype=np.uint8), indices, indptr),
            shape=(rows, n_mechanisms),
            dtype=np.uint8,
        )

    raw_components = payload["components"]
    if not isinstance(raw_components, list):
        raise ValueError("components must be a JSON array")
    components = tuple(
        DemComponent(
            parent_mechanism_id=int(item["parent_mechanism_id"]),
            component_index=int(item["component_index"]),
            detectors=tuple(_int_list(item, "detectors")),
            observables=tuple(_int_list(item, "observables")),
        )
        for item in raw_components
    )

    return DemStructure(
        h=matrices["H"],
        l=matrices["L"],
        priors=np.asarray(_float_list(payload, "priors"), dtype=np.float64),
        components=components,
        detector_coords=coords,
        n_detectors=n_detectors,
        n_observables=n_observables,
        n_mechanisms=n_mechanisms,
        coord_dim=coord_dim,
        metadata={"n_components": len(components)},
    )


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

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        """Write one JSON object per shot, preceded by the header lines."""
        require_level_agreement(dataset, structure_level)
        if structure_level is not StructureLevel.NONE and dataset.structure is None:
            raise ValueError(
                f"structure_level={structure_level} was requested but the dataset carries "
                "no structure, so the manifest would claim structure the file cannot hold"
            )
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
            header = {MANIFEST_KEY: dataset.meta.to_json_dict()}
            handle.write(json.dumps(header, sort_keys=True, allow_nan=False))
            handle.write("\n")
            if structure_level is not StructureLevel.NONE and dataset.structure is not None:
                structure_line = {
                    STRUCTURE_KEY: _structure_to_json(dataset.structure, structure_level)
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
                record = _load_json_object(line, where)

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

        detectors = _repack(det_rows, meta.n_detectors)
        observables = _repack(obs_rows, meta.n_observables)
        mechanisms = _repack(mech_rows, meta.n_mechanisms or 0) if mech_rows else None
        environment_ids = np.asarray(env_rows, dtype=np.int32) if env_rows else None
        return InMemoryDataset(
            detectors=detectors,
            observables=observables,
            meta=meta,
            environment_ids=environment_ids,
            mechanisms=mechanisms,
            structure=(
                None if structure_payload is None else _structure_from_json(structure_payload, meta)
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


def _repack(rows: list[np.ndarray], n_bits: int) -> np.ndarray:
    """Re-pack unpacked bool rows into little-endian bytes.

    Every row must carry exactly ``n_bits``. A uniformly short row would repack with
    fabricated zero bits in the trailing positions — invisible whenever the byte count
    happens to match — and an over-long one would be silently truncated. Both turn a
    corrupt or foreign file into plausible-looking data, the same failure class
    ``_string_to_bits`` refuses for characters.
    """
    if not rows:
        return np.zeros((0, packed_width(n_bits)), dtype=np.uint8)
    widths = {int(row.shape[0]) for row in rows}
    if widths != {n_bits}:
        raise ValueError(
            f"shot bit strings carry {sorted(widths)} bits but the manifest declares "
            f"{n_bits}; refusing to zero-fill or truncate"
        )
    return np.packbits(np.stack(rows, axis=0), axis=1, bitorder="little")
