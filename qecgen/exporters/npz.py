"""NPZ exporter. ``numpy.savez_compressed`` with the manifest as a JSON string."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix

from qecgen.dataset import DatasetMeta, InMemoryDataset, StructureLevel
from qecgen.dem import DemComponent, DemStructure
from qecgen.exporters.base import require_level_agreement

__all__ = ["NPZExporter"]


def _with_npz_suffix(path: Path) -> Path:
    """Normalise the ``.npz`` suffix that numpy would otherwise append unreported.

    ``np.savez_compressed("out")`` silently writes ``out.npz``, so the caller is told
    one path and a different file appears on disk.
    """
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


class NPZExporter:
    """Single-file NumPy archive. Materialising only; NPZ cannot be appended to."""

    @property
    def format_name(self) -> str:
        """Format name for ``--format``."""
        return "npz"

    @property
    def extension(self) -> str:
        """File extension."""
        return ".npz"

    @property
    def streaming(self) -> bool:
        """NPZ cannot be written incrementally."""
        return False

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
        """Write a dataset to a compressed ``.npz`` archive."""
        require_level_agreement(dataset, structure_level)
        path = _with_npz_suffix(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Typed as Any because numpy declares savez_compressed as
        # (file, *args, allow_pickle=..., **kwds); a narrower value type makes mypy
        # match this unpack against allow_pickle. Round-trip is covered by tests.
        arrays: dict[str, Any] = {
            "detectors": dataset.detectors,
            "observables": dataset.observables,
        }
        if dataset.environment_ids is not None:
            arrays["environment_ids"] = dataset.environment_ids
        if dataset.mechanisms is not None:
            arrays["mechanisms"] = dataset.mechanisms

        structure = dataset.structure
        if structure is not None and structure_level is not StructureLevel.NONE:
            arrays["dem_detector_coords"] = structure.detector_coords
            arrays["dem_dims"] = np.asarray(
                [
                    structure.n_detectors,
                    structure.n_observables,
                    structure.n_mechanisms,
                    structure.coord_dim,
                ],
                dtype=np.int64,
            )
            if structure_level is not StructureLevel.COORDS:
                for name, matrix in (("H", structure.h), ("L", structure.l)):
                    csc = matrix.tocsc()
                    arrays[f"dem_{name}_indices"] = csc.indices.astype(np.int32)
                    arrays[f"dem_{name}_indptr"] = csc.indptr.astype(np.int64)
                    arrays[f"dem_{name}_shape"] = np.asarray(csc.shape, dtype=np.int64)
                arrays["dem_priors"] = structure.priors
                arrays["dem_component_parent"] = np.asarray(
                    [c.parent_mechanism_id for c in structure.components], dtype=np.int32
                )
                arrays["dem_component_index"] = np.asarray(
                    [c.component_index for c in structure.components], dtype=np.int32
                )
                for kind in ("det", "obs"):
                    flat: list[int] = []
                    offsets: list[int] = [0]
                    for component in structure.components:
                        values = component.detectors if kind == "det" else component.observables
                        flat.extend(values)
                        offsets.append(len(flat))
                    arrays[f"dem_component_{kind}_flat"] = np.asarray(flat, dtype=np.int32)
                    arrays[f"dem_component_{kind}_offset"] = np.asarray(offsets, dtype=np.int64)

        arrays["manifest"] = np.asarray(dataset.meta.to_json())
        if structure_level is StructureLevel.FULL:
            # Provenance is a separate array, never folded into the manifest.
            arrays["provenance"] = np.asarray(dataset.meta.provenance_json())
        np.savez_compressed(path, **arrays)

    def read(self, path: Path) -> InMemoryDataset:
        """Read a dataset written by :meth:`write`."""
        with np.load(path, allow_pickle=False) as handle:
            data: dict[str, Any] = {key: handle[key] for key in handle.files}
        meta = DatasetMeta.from_json(str(data["manifest"].item()))
        structure = _structure_from_arrays(data)
        return InMemoryDataset(
            detectors=data["detectors"],
            observables=data["observables"],
            meta=meta,
            environment_ids=data.get("environment_ids"),
            mechanisms=data.get("mechanisms"),
            structure=structure,
        )


def _structure_from_arrays(data: dict[str, Any]) -> DemStructure | None:
    """Rebuild a :class:`DemStructure` from the flat NPZ arrays, if present."""
    if "dem_dims" not in data:
        return None
    n_detectors, n_observables, n_mechanisms, coord_dim = (int(v) for v in data["dem_dims"])
    coords = data["dem_detector_coords"]

    if "dem_H_indices" not in data:
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
    for name in ("H", "L"):
        indices = data[f"dem_{name}_indices"]
        indptr = data[f"dem_{name}_indptr"]
        shape = tuple(int(v) for v in data[f"dem_{name}_shape"])
        matrices[name] = csc_matrix(
            (np.ones(indices.shape[0], dtype=np.uint8), indices, indptr),
            shape=shape,
            dtype=np.uint8,
        )

    parents = data["dem_component_parent"]
    indices_arr = data["dem_component_index"]
    det_flat, det_off = data["dem_component_det_flat"], data["dem_component_det_offset"]
    obs_flat, obs_off = data["dem_component_obs_flat"], data["dem_component_obs_offset"]
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
        priors=data["dem_priors"],
        components=components,
        detector_coords=coords,
        n_detectors=n_detectors,
        n_observables=n_observables,
        n_mechanisms=n_mechanisms,
        coord_dim=coord_dim,
        metadata={"n_components": len(components)},
    )
