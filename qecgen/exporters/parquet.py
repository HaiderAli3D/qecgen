"""Parquet exporter. One row per shot, manifest in the file's key-value metadata.

Packed detector and observable bytes are stored as fixed-width binary columns rather
than as lists of booleans: it keeps the little-endian packing intact end to end, and
avoids a 8x size blow-up. Structural information (DEM, coordinates) does not fit the
one-row-per-shot model and is stored in the key-value metadata as JSON when requested.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qecgen.dataset import DatasetMeta, InMemoryDataset, StructureLevel
from qecgen.dem import DemStructure
from qecgen.exporters.base import (
    NotAQecgenDatasetError,
    recorded_structure_level,
    require_level_agreement,
    require_non_empty,
)

__all__ = ["ParquetExporter", "require_qecgen_metadata"]

_MANIFEST_KEY = b"qecgen_manifest"


def require_qecgen_metadata(path: Path, metadata: dict[bytes, bytes]) -> None:
    """Refuse a ``.parquet`` this tool did not write, from its schema metadata.

    Parquet is a general-purpose columnar format, so a data root can hold plenty of
    intact ``.parquet`` files that are simply somebody else's — and indexing the metadata
    for a key that is not there raised a bare ``KeyError``, which the dataset browser
    reports as corruption.

    Truncation cannot reach this branch: the Parquet footer is written last, so a file cut
    short fails to open at all rather than opening without its metadata.
    """
    if _MANIFEST_KEY in metadata:
        return
    found = ", ".join(sorted(key.decode("utf-8", "replace") for key in metadata)[:6]) or "none"
    raise NotAQecgenDatasetError(
        f"{path.name} is a Parquet file whose schema metadata has no "
        f"{_MANIFEST_KEY.decode()!r} key, so this tool did not write it (keys: {found})."
    )


_STRUCTURE_KEY = b"qecgen_structure"


def _structure_to_json(structure: DemStructure, level: StructureLevel) -> str:
    """Serialise structure for the key-value metadata block."""
    payload: dict[str, object] = {
        "n_detectors": structure.n_detectors,
        "n_observables": structure.n_observables,
        "n_mechanisms": structure.n_mechanisms,
        "coord_dim": structure.coord_dim,
        "detector_coords": structure.detector_coords.tolist(),
    }
    if level is not StructureLevel.COORDS:
        h = structure.h.tocsc()
        ell = structure.l.tocsc()
        payload["H_indices"] = h.indices.tolist()
        payload["H_indptr"] = h.indptr.tolist()
        payload["L_indices"] = ell.indices.tolist()
        payload["L_indptr"] = ell.indptr.tolist()
        payload["priors"] = structure.priors.tolist()
        payload["components"] = [
            {
                "parent_mechanism_id": c.parent_mechanism_id,
                "component_index": c.component_index,
                "detectors": list(c.detectors),
                "observables": list(c.observables),
            }
            for c in structure.components
        ]
    return json.dumps(payload)


class ParquetExporter:
    """One row per shot. Good for columnar tooling; no streaming append here."""

    @property
    def format_name(self) -> str:
        """Format name for ``--format``."""
        return "parquet"

    @property
    def extension(self) -> str:
        """File extension."""
        return ".parquet"

    @property
    def streaming(self) -> bool:
        """Not wired for incremental append in this iteration."""
        return False

    @property
    def structure_round_trip(self) -> bool:
        """Structure is recorded in key-value metadata but not reconstructed on read."""
        return False

    @property
    def carries_provenance(self) -> bool:
        """No. Nothing is written at ``FULL`` that is not written at ``DEM``."""
        return False

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        """Write one row per shot with the manifest in key-value metadata."""
        require_level_agreement(dataset, structure_level)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, pa.Array] = {
            "detectors": pa.array([row.tobytes() for row in dataset.detectors], type=pa.binary()),
            "observables": pa.array(
                [row.tobytes() for row in dataset.observables], type=pa.binary()
            ),
        }
        if dataset.environment_ids is not None:
            columns["environment_id"] = pa.array(
                dataset.environment_ids.astype(np.int32), type=pa.int32()
            )
        if dataset.mechanisms is not None:
            columns["mechanisms"] = pa.array(
                [row.tobytes() for row in dataset.mechanisms], type=pa.binary()
            )

        # Parquet records structure but does not reconstruct it on read, so the
        # persisted manifest declares the level it can actually reproduce. A manifest
        # that claims more than the payload holds is worse than one that claims less.
        recorded = dataclasses.replace(
            dataset.meta, structure_level=recorded_structure_level(self, structure_level)
        )
        metadata: dict[bytes, bytes] = {_MANIFEST_KEY: recorded.to_json().encode("utf-8")}
        if dataset.structure is not None and structure_level is not StructureLevel.NONE:
            metadata[_STRUCTURE_KEY] = _structure_to_json(
                dataset.structure, structure_level
            ).encode("utf-8")

        table = pa.table(columns)
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path, compression="zstd")

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset written by :meth:`write`.

        Structural information is not reconstructed into a :class:`DemStructure` here;
        Parquet is a shot-table format and the round-trip contract covers arrays and
        manifest. Use HDF5 or NPZ when the structure must survive a round trip.
        """
        require_non_empty(path, f"a {_MANIFEST_KEY.decode()} schema-metadata key")
        table = pq.read_table(path)
        require_qecgen_metadata(path, table.schema.metadata or {})
        schema_metadata = table.schema.metadata or {}
        meta = DatasetMeta.from_json(schema_metadata[_MANIFEST_KEY].decode("utf-8"))

        detectors = _binary_column_to_array(table, "detectors")
        observables = _binary_column_to_array(table, "observables")
        environment_ids = (
            np.asarray(
                table.column("environment_id").to_numpy(zero_copy_only=False), dtype=np.int32
            )
            if "environment_id" in table.column_names
            else None
        )
        mechanisms = (
            _binary_column_to_array(table, "mechanisms")
            if "mechanisms" in table.column_names
            else None
        )
        return InMemoryDataset(
            detectors=detectors,
            observables=observables,
            meta=meta,
            environment_ids=environment_ids,
            mechanisms=mechanisms,
            structure=None,
        )


def _binary_column_to_array(table: pa.Table, name: str) -> np.ndarray:
    """Convert a binary column of equal-length rows back to a 2D uint8 array."""
    rows = table.column(name).to_pylist()
    if not rows:
        return np.zeros((0, 0), dtype=np.uint8)
    width = len(rows[0])
    # Rows of unequal length would reshape into a different number of shots, silently
    # re-segmenting the dataset across shot boundaries.
    bad = {len(r) for r in rows} - {width}
    if bad:
        raise ValueError(
            f"column {name!r} has rows of differing widths {sorted(bad | {width})}; "
            "the shot boundaries cannot be recovered"
        )
    flat = np.frombuffer(b"".join(rows), dtype=np.uint8)
    return flat.reshape(len(rows), width)
