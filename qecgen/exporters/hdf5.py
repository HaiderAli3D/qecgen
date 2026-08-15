"""HDF5 exporter. Primary format, and the only streaming-capable one.

Layout::

    /detectors        uint8  (shots, ceil(n_detectors/8))   little-endian bit-packed
    /observables      uint8  (shots, ceil(n_observables/8))
    /environment_ids  int32  (shots,)                       optional
    /mechanisms       uint8  (shots, ceil(n_mechanisms/8))  optional, Contract B
    /dem/             group                                 present per --structure
        H_indices, H_indptr, H_shape                        CSC of (n_detectors, n_mechanisms)
        L_indices, L_indptr, L_shape                        CSC of (n_observables, n_mechanisms)
        priors                float64 (n_mechanisms,)
        detector_coords       float64 (n_detectors, coord_dim)
        component_parent      int32   (n_components,)       parent_mechanism_id
        component_index       int32   (n_components,)
        component_det_flat    int32   ragged detector ids, concatenated
        component_det_offset  int64   (n_components + 1,)
        component_obs_flat    int32
        component_obs_offset  int64   (n_components + 1,)

The manifest is stored as a JSON string in the root attribute ``manifest``, with a few
scalars mirrored as separate attributes so ``h5dump -A`` is readable without parsing.

CSC data values are not stored: ``H`` and ``L`` are binary incidence matrices, so every
stored entry is 1 by construction. Storing a vector of ones would be pure overhead.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np
from scipy.sparse import csc_matrix

from qecgen.dataset import DatasetMeta, InMemoryDataset, StructureLevel
from qecgen.dem import DemComponent, DemStructure
from qecgen.exporters.base import require_level_agreement
from qecgen.sampling import packed_width

__all__ = ["HDF5Exporter", "StreamingHDF5Writer"]

_COMPRESSION = "gzip"
_COMPRESSION_LEVEL = 4


def _write_structure(handle: h5py.File, structure: DemStructure, level: StructureLevel) -> None:
    """Write the ``dem/`` group at the requested detail level."""
    if level is StructureLevel.NONE:
        return

    group = handle.create_group("dem")
    group.attrs["n_detectors"] = structure.n_detectors
    group.attrs["n_observables"] = structure.n_observables
    group.attrs["n_mechanisms"] = structure.n_mechanisms
    group.attrs["coord_dim"] = structure.coord_dim

    group.create_dataset(
        "detector_coords",
        data=structure.detector_coords,
        compression=_COMPRESSION,
        compression_opts=_COMPRESSION_LEVEL,
    )
    if level is StructureLevel.COORDS:
        return

    for name, matrix in (("H", structure.h), ("L", structure.l)):
        csc = matrix.tocsc()
        group.create_dataset(f"{name}_indices", data=csc.indices.astype(np.int32))
        group.create_dataset(f"{name}_indptr", data=csc.indptr.astype(np.int64))
        group.create_dataset(f"{name}_shape", data=np.asarray(csc.shape, dtype=np.int64))

    group.create_dataset("priors", data=structure.priors)

    parents = np.asarray([c.parent_mechanism_id for c in structure.components], dtype=np.int32)
    indices = np.asarray([c.component_index for c in structure.components], dtype=np.int32)
    group.create_dataset("component_parent", data=parents)
    group.create_dataset("component_index", data=indices)

    for kind in ("det", "obs"):
        flat: list[int] = []
        offsets: list[int] = [0]
        for component in structure.components:
            values = component.detectors if kind == "det" else component.observables
            flat.extend(values)
            offsets.append(len(flat))
        group.create_dataset(f"component_{kind}_flat", data=np.asarray(flat, dtype=np.int32))
        group.create_dataset(f"component_{kind}_offset", data=np.asarray(offsets, dtype=np.int64))


def _read_structure(handle: h5py.File) -> DemStructure | None:
    """Read back the ``dem/`` group, tolerating coords-only files."""
    if "dem" not in handle:
        return None
    group = handle["dem"]
    n_detectors = int(group.attrs["n_detectors"])
    n_observables = int(group.attrs["n_observables"])
    n_mechanisms = int(group.attrs["n_mechanisms"])
    coord_dim = int(group.attrs["coord_dim"])
    coords = np.asarray(group["detector_coords"])

    if "H_indices" not in group:
        empty = csc_matrix((n_detectors, n_mechanisms), dtype=np.uint8)
        return DemStructure(
            h=empty,
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
    for name in ("H", "L"):
        indices = np.asarray(group[f"{name}_indices"])
        indptr = np.asarray(group[f"{name}_indptr"])
        shape = tuple(int(v) for v in np.asarray(group[f"{name}_shape"]))
        data = np.ones(indices.shape[0], dtype=np.uint8)
        matrices[name] = csc_matrix((data, indices, indptr), shape=shape, dtype=np.uint8)

    parents = np.asarray(group["component_parent"])
    indices_arr = np.asarray(group["component_index"])
    det_flat = np.asarray(group["component_det_flat"])
    det_off = np.asarray(group["component_det_offset"])
    obs_flat = np.asarray(group["component_obs_flat"])
    obs_off = np.asarray(group["component_obs_offset"])

    components = tuple(
        DemComponent(
            parent_mechanism_id=int(parents[i]),
            component_index=int(indices_arr[i]),
            detectors=tuple(int(v) for v in det_flat[det_off[i] : det_off[i + 1]]),
            observables=tuple(int(v) for v in obs_flat[obs_off[i] : obs_off[i + 1]]),
        )
        for i in range(parents.shape[0])
    )

    return DemStructure(
        h=matrices["H"],
        l=matrices["L"],
        priors=np.asarray(group["priors"]),
        components=components,
        detector_coords=coords,
        n_detectors=n_detectors,
        n_observables=n_observables,
        n_mechanisms=n_mechanisms,
        coord_dim=coord_dim,
        metadata={"n_components": len(components)},
    )


def _write_manifest(handle: h5py.File, meta: DatasetMeta, level: StructureLevel) -> None:
    """Store the manifest as JSON plus a few readable scalar attributes.

    The manifest is **decoder-visible** and never carries circuit or DEM text. At
    ``FULL`` the text goes into a separate ``/provenance`` group instead, so a decoder
    that reads the manifest cannot recover a frozen-prior test set's own error model.
    """
    handle.attrs["manifest"] = meta.to_json()
    if level is StructureLevel.FULL:
        group = handle.create_group("provenance")
        group.attrs["environments"] = meta.provenance_json()
        group.attrs["warning"] = (
            "PROVENANCE ONLY - a decoder must not read this group. Under frozen_prior "
            "it contains the test environment's own DEM."
        )
    handle.attrs["bit_order"] = meta.bit_order
    handle.attrs["contract"] = str(meta.contract)
    handle.attrs["n_detectors"] = meta.n_detectors
    handle.attrs["n_observables"] = meta.n_observables
    handle.attrs["shots"] = meta.shots
    handle.attrs["distance"] = meta.distance
    handle.attrs["rounds"] = meta.rounds
    handle.attrs["chunk_size"] = meta.chunk_size
    handle.attrs["drift_condition"] = str(meta.drift_condition)
    handle.attrs["qecgen_version"] = meta.versions.get("qecgen", "unknown")


class HDF5Exporter:
    """Materialising HDF5 reader/writer."""

    @property
    def format_name(self) -> str:
        """Format name for ``--format``."""
        return "hdf5"

    @property
    def extension(self) -> str:
        """File extension."""
        return ".h5"

    @property
    def streaming(self) -> bool:
        """HDF5 is the only streaming-capable format."""
        return True

    @property
    def structure_round_trip(self) -> bool:
        """Full structure survives a round trip."""
        return True

    def write(
        self,
        dataset: InMemoryDataset,
        path: Path,
        structure_level: StructureLevel = StructureLevel.NONE,
    ) -> None:
        """Write a dataset to an HDF5 file."""
        require_level_agreement(dataset, structure_level)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            for name, array in (
                ("detectors", dataset.detectors),
                ("observables", dataset.observables),
                ("environment_ids", dataset.environment_ids),
                ("mechanisms", dataset.mechanisms),
            ):
                if array is None:
                    continue
                handle.create_dataset(
                    name,
                    data=array,
                    compression=_COMPRESSION,
                    compression_opts=_COMPRESSION_LEVEL,
                )
            if dataset.structure is not None:
                _write_structure(handle, dataset.structure, structure_level)
            _write_manifest(handle, dataset.meta, structure_level)

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset written by :meth:`write`."""
        with h5py.File(path, "r") as handle:
            meta = DatasetMeta.from_json(str(handle.attrs["manifest"]))
            detectors = np.asarray(handle["detectors"])
            observables = np.asarray(handle["observables"])
            environment_ids = (
                np.asarray(handle["environment_ids"]) if "environment_ids" in handle else None
            )
            mechanisms = np.asarray(handle["mechanisms"]) if "mechanisms" in handle else None
            structure = _read_structure(handle)
        return InMemoryDataset(
            detectors=detectors,
            observables=observables,
            meta=meta,
            environment_ids=environment_ids,
            mechanisms=mechanisms,
            structure=structure,
        )


class StreamingHDF5Writer:
    """Append chunks to an HDF5 file without ever holding all shots in memory.

    Datasets are created with an unlimited first dimension and resized per chunk. The
    manifest is written on :meth:`close`, so a file that was never closed is detectably
    incomplete: it has no ``manifest`` attribute.
    """

    def __init__(self, path: Path, n_detectors: int, n_observables: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._handle = h5py.File(path, "w")
        self._n_detectors = n_detectors
        self._n_observables = n_observables
        self._rows = 0
        self._created: set[str] = set()
        self._columns: set[str] | None = None
        """Column set fixed by the first append; None until then."""

    def _ensure(self, name: str, sample: np.ndarray) -> None:
        if name in self._created:
            return
        shape = (0,) if sample.ndim == 1 else (0, sample.shape[1])
        maxshape: tuple[int | None, ...] = (None,) if sample.ndim == 1 else (None, sample.shape[1])
        chunk = (
            (min(8192, max(1, sample.shape[0])),)
            if sample.ndim == 1
            else (min(8192, max(1, sample.shape[0])), sample.shape[1])
        )
        self._handle.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=sample.dtype,
            chunks=chunk,
            compression=_COMPRESSION,
            compression_opts=_COMPRESSION_LEVEL,
        )
        self._created.add(name)

    def append(
        self,
        detectors: np.ndarray,
        observables: np.ndarray,
        environment_ids: np.ndarray | None = None,
        mechanisms: np.ndarray | None = None,
    ) -> None:
        """Append one chunk. All arrays must have the same number of rows.

        The set of columns is fixed by the **first** append. Adding a column later
        would leave the earlier rows zero-filled — fabricated labels indistinguishable
        from real ones — and omitting one later would leave the columns at unequal
        lengths. Both are refused rather than silently produced.
        """
        rows = detectors.shape[0]
        for name, array in (
            ("observables", observables),
            ("environment_ids", environment_ids),
            ("mechanisms", mechanisms),
        ):
            if array is not None and array.shape[0] != rows:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows but detectors has {rows}; "
                    "per-shot correspondence would be broken"
                )

        supplied = {
            name
            for name, array in (
                ("detectors", detectors),
                ("observables", observables),
                ("environment_ids", environment_ids),
                ("mechanisms", mechanisms),
            )
            if array is not None
        }
        if self._columns is None:
            self._columns = supplied
        elif supplied != self._columns:
            missing = sorted(self._columns - supplied)
            extra = sorted(supplied - self._columns)
            raise ValueError(
                f"chunk column set {sorted(supplied)} differs from the set fixed by the "
                f"first append {sorted(self._columns)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
                + ". Introducing a column mid-stream would zero-fill every earlier row."
            )

        for name, array in (
            ("detectors", detectors),
            ("observables", observables),
            ("environment_ids", environment_ids),
            ("mechanisms", mechanisms),
        ):
            if array is None:
                continue
            self._ensure(name, array)
            handle = self._handle[name]
            if handle.dtype != array.dtype:
                raise ValueError(
                    f"{name} chunk dtype {array.dtype} differs from {handle.dtype} "
                    "fixed by the first append"
                )
            if array.ndim > 1 and handle.shape[1] != array.shape[1]:
                raise ValueError(
                    f"{name} chunk width {array.shape[1]} differs from "
                    f"{handle.shape[1]} fixed by the first append"
                )
            handle.resize(self._rows + rows, axis=0)
            handle[self._rows : self._rows + rows] = array
        self._rows += rows

    @property
    def rows_written(self) -> int:
        """Shots appended so far."""
        return self._rows

    def close(self, meta: DatasetMeta, structure: DemStructure | None = None) -> None:
        """Write the manifest and structure, then close the file."""
        if "detectors" not in self._created:
            # Zero shots: append never ran, so no array datasets exist and read()
            # would fail on a file whose manifest promises them. Create the empty
            # columns the manifest describes — the same shapes the materialised
            # path writes for zero shots.
            self._ensure("detectors", np.zeros((0, packed_width(meta.n_detectors)), dtype=np.uint8))
            self._ensure(
                "observables", np.zeros((0, packed_width(meta.n_observables)), dtype=np.uint8)
            )
            if meta.n_mechanisms is not None:
                self._ensure(
                    "mechanisms", np.zeros((0, packed_width(meta.n_mechanisms)), dtype=np.uint8)
                )
        if structure is not None:
            _write_structure(self._handle, structure, meta.structure_level)
        _write_manifest(self._handle, meta, meta.structure_level)
        self._handle.close()

    def abort(self) -> None:
        """Close the handle without writing a manifest.

        A file with no ``manifest`` attribute is detectably incomplete, which is the
        correct outcome when generation failed part way through: better an obviously
        broken file than a plausible-looking truncated one.
        """
        with contextlib.suppress(OSError, ValueError):
            self._handle.close()

    def __enter__(self) -> StreamingHDF5Writer:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the raw handle if an exception escaped before :meth:`close`."""
        if self._handle:
            # Already-closed is the expected case when close() ran normally.
            with contextlib.suppress(OSError, ValueError):
                self._handle.close()


def read_manifest_only(path: Path) -> dict[str, object]:
    """Read a manifest without loading any shots. Used by ``qecgen inspect``."""
    with h5py.File(path, "r") as handle:
        raw = json.loads(str(handle.attrs["manifest"]))
    return dict(raw)
