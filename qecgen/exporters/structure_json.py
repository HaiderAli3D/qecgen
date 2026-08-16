"""The normative JSON encoding of a :class:`DemStructure`, shared by the text formats.

``DATA_CONTRACT.md`` calls this encoding **normative, not incidental**: the exact key
names, CSC ``data`` omitted because every stored entry is 1 by construction, ``components``
*nested* rather than flattened into value/offset pairs, and NaN coordinates encoded as
``null`` with ``allow_nan=False`` on write and a rejecting ``parse_constant`` on read.

It lives here rather than in ``jsonl.py``, where it was written, because two formats now
depend on it byte for byte. Left in one exporter it would have that exporter as its de
facto owner, and a JSONL-local edit would silently change the CSV files on disk.
``parquet.py`` shows what happens without a single owner: its private copy has already
drifted — no ``*_shape`` keys, coordinates via ``.tolist()``, and ``json.dumps`` at the
default ``allow_nan=True``, so a ragged-coordinate circuit would write bare ``NaN`` tokens
into its metadata. That copy is deliberately left alone; it is written but never read
back, and unifying it would change the bytes of existing Parquet files for no functional
gain.

Not in ``base.py``: that module is the ``Exporter`` protocol plus the guards every
exporter calls, and pulling scipy and :class:`DemStructure` into it would make the
protocol module a utility dump.
"""

from __future__ import annotations

import json
from typing import Any, NoReturn

import numpy as np
from scipy.sparse import csc_matrix

from qecgen.dataset import DatasetMeta, StructureLevel
from qecgen.dem import DemComponent, DemStructure
from qecgen.sampling import packed_width

__all__ = [
    "coords_from_json",
    "coords_to_json",
    "float_list",
    "int_list",
    "load_json_object",
    "reject_json_constant",
    "repack",
    "structure_from_json",
    "structure_to_json",
]


def reject_json_constant(name: str) -> NoReturn:
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


def load_json_object(line: str, where: str) -> dict[str, Any]:
    """Parse one line into a dict, the single choke point for ``Any`` from ``json``.

    Everything downstream sees ``object``, so mypy forces narrowing rather than merely
    permitting it.
    """
    raw = json.loads(line, parse_constant=reject_json_constant)
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(raw).__name__}")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{where}: JSON object keys must all be strings")
    return dict(raw)


def int_list(payload: dict[str, Any], key: str) -> list[int]:
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


def float_list(payload: dict[str, Any], key: str) -> list[float]:
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


def coords_to_json(coords: np.ndarray) -> list[list[float | None]]:
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


def coords_from_json(value: Any, n_detectors: int, coord_dim: int) -> np.ndarray:
    """Decode coordinates, mapping ``null`` back to NaN."""
    if not isinstance(value, list):
        raise ValueError("detector_coords must be a JSON array")
    rows = [[np.nan if item is None else float(item) for item in row] for row in value]
    # np.asarray([]) gives shape (0,), not (0, 0), so reshape explicitly.
    return np.array(rows, dtype=np.float64).reshape(n_detectors, coord_dim)


def structure_to_json(structure: DemStructure, level: StructureLevel) -> dict[str, Any]:
    """Encode a :class:`DemStructure` for a header record."""
    payload: dict[str, Any] = {
        "level": str(level),
        "n_detectors": structure.n_detectors,
        "n_observables": structure.n_observables,
        "n_mechanisms": structure.n_mechanisms,
        "coord_dim": structure.coord_dim,
        "detector_coords": coords_to_json(structure.detector_coords),
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


def structure_from_json(payload: dict[str, Any], meta: DatasetMeta) -> DemStructure:
    """Decode a structure header record, cross-checking it against the manifest."""
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
    coords = coords_from_json(payload["detector_coords"], n_detectors, coord_dim)

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
        indices = np.asarray(int_list(payload, f"{name}_indices"), dtype=np.int32)
        indptr = np.asarray(int_list(payload, f"{name}_indptr"), dtype=np.int64)
        shape = int_list(payload, f"{name}_shape")
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
            detectors=tuple(int_list(item, "detectors")),
            observables=tuple(int_list(item, "observables")),
        )
        for item in raw_components
    )

    return DemStructure(
        h=matrices["H"],
        l=matrices["L"],
        priors=np.asarray(float_list(payload, "priors"), dtype=np.float64),
        components=components,
        detector_coords=coords,
        n_detectors=n_detectors,
        n_observables=n_observables,
        n_mechanisms=n_mechanisms,
        coord_dim=coord_dim,
        metadata={"n_components": len(components)},
    )


def repack(rows: list[np.ndarray], n_bits: int) -> np.ndarray:
    """Re-pack unpacked bool rows into little-endian bytes.

    Every row must carry exactly ``n_bits``. A uniformly short row would repack with
    fabricated zero bits in the trailing positions — invisible whenever the byte count
    happens to match — and an over-long one would be silently truncated. Both turn a
    corrupt or foreign file into plausible-looking data, the same failure class the
    per-format bit parsers refuse for characters.
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
