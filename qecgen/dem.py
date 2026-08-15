"""Detector error model parsing into sparse matrices.

The central invariant of this module: **one column per independent ``error(...)``
instruction**.

Stim's decomposed DEM writes correlated mechanisms as graphlike components joined by
``^`` separators. Those components share a single probability and belong to a single
physical mechanism. Emitting each component as its own column with the parent
probability duplicated would silently substitute a different noise model for the one
Stim simulated. Measured on d=3, r=3 circuit-level p=0.01: 286 ``error`` instructions
decompose into 556 components, so the naive reading inflates the column count by 94%.

Components are preserved in a parallel structure keyed by ``parent_mechanism_id`` so the
correlation structure survives export.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import stim
from scipy.sparse import csc_matrix

__all__ = ["DemComponent", "DemStructure", "parse_dem"]


@dataclass(frozen=True, slots=True)
class DemComponent:
    """One graphlike component of a (possibly decomposed) mechanism."""

    parent_mechanism_id: int
    """Index of the ``error(...)`` instruction this component came from.

    Multiple components share this id. The parent's probability belongs to the
    mechanism as a whole and must not be duplicated per component.
    """

    component_index: int
    """Position of this component within its parent, 0-based."""

    detectors: tuple[int, ...]
    """Detector ids this component touches. Graphlike components have 1 or 2."""

    observables: tuple[int, ...]
    """Observable ids this component flips."""


@dataclass(frozen=True)
class DemStructure:
    """Sparse representation of a decomposed detector error model."""

    h: csc_matrix
    """Detector incidence, ``(n_detectors, n_mechanisms)``, uint8, values in {0, 1}."""

    l: csc_matrix  # noqa: E741 - matches the standard H / L naming in the QEC literature
    """Observable incidence, ``(n_observables, n_mechanisms)``, uint8."""

    priors: np.ndarray
    """float64 ``(n_mechanisms,)`` probabilities, one per ``error(...)`` instruction."""

    components: tuple[DemComponent, ...]
    """Flattened components across all mechanisms, carrying ``parent_mechanism_id``."""

    detector_coords: np.ndarray
    """float64 ``(n_detectors, coord_dim)``. For the surface code this is (x, y, t).

    Exportable independently of the full DEM via ``--structure coords``: it is the
    natural graph structure to hand a learning-based decoder without also handing it
    the full error model.
    """

    n_detectors: int
    n_observables: int
    n_mechanisms: int
    coord_dim: int

    has_matrices: bool = True
    """Whether ``h``, ``l``, ``priors`` and ``components`` are populated.

    False for a structure read back from a ``--structure coords`` file, which carries
    detector coordinates and nothing else. Validation branches on this: demanding one
    prior per mechanism from a coords-only file would fail on correct data, and a
    validator that cries wolf gets ignored.

    This is a plain bool rather than a ``StructureLevel`` because :mod:`qecgen.dataset`
    imports this module, so the dependency cannot run the other way.
    """

    metadata: dict[str, object] = field(default_factory=dict)
    """Diagnostics: component counts, weight histograms, cancellation counts."""

    def component_weight_histogram(self) -> dict[int, int]:
        """Detector-count histogram over **components** (expected: only 1 and 2)."""
        return dict(Counter(len(c.detectors) for c in self.components))

    def mechanism_weight_histogram(self) -> dict[int, int]:
        """Detector-count histogram over **whole mechanisms** (may exceed 2).

        A mechanism with 2 components legitimately touches up to 4 detectors, so this
        distribution is reported, not asserted on. See ``DATA_CONTRACT.md``.
        """
        counts = np.asarray((self.h != 0).sum(axis=0)).ravel()
        return dict(Counter(int(v) for v in counts))


def parse_dem(dem: stim.DetectorErrorModel, circuit: stim.Circuit | None = None) -> DemStructure:
    """Parse a decomposed DEM into sparse ``H`` / ``L`` plus component structure.

    Args:
        dem: A detector error model, which should have been produced with
            ``decompose_errors=True``.
        circuit: Optional source circuit, used for detector coordinates via
            ``circuit.get_detector_coordinates()``. Falls back to the DEM's own
            coordinates when omitted.

    Returns:
        A :class:`DemStructure` with exactly one column per ``error(...)`` instruction.
    """
    flat = dem.flattened()

    priors: list[float] = []
    components: list[DemComponent] = []
    h_rows: list[int] = []
    h_cols: list[int] = []
    l_rows: list[int] = []
    l_cols: list[int] = []

    n_cancelled_detectors = 0
    n_cancelled_observables = 0
    n_with_separator = 0

    mechanism_id = 0
    for instruction in flat:
        if instruction.type != "error":
            continue

        args = instruction.args_copy()
        if not args:
            raise ValueError(f"error instruction {mechanism_id} carries no probability")
        priors.append(float(args[0]))

        # Walk targets, splitting on separators. Detector and observable ids are
        # accumulated with XOR semantics: a target repeated within one mechanism
        # cancels mod 2. Using a plain append here would corrupt H.
        comp_dets: list[list[int]] = [[]]
        comp_obs: list[list[int]] = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                comp_dets.append([])
                comp_obs.append([])
            elif target.is_relative_detector_id():
                comp_dets[-1].append(target.val)
            elif target.is_logical_observable_id():
                comp_obs[-1].append(target.val)
            else:  # pragma: no cover - stim has no other DemTarget kind here
                raise ValueError(f"unrecognised DEM target {target!r}")

        if len(comp_dets) > 1:
            n_with_separator += 1

        mech_dets: set[int] = set()
        mech_obs: set[int] = set()
        for index, (dets, obs) in enumerate(zip(comp_dets, comp_obs, strict=True)):
            unique_dets = _xor_reduce(dets)
            unique_obs = _xor_reduce(obs)
            n_cancelled_detectors += len(dets) - len(unique_dets)
            n_cancelled_observables += len(obs) - len(unique_obs)
            components.append(
                DemComponent(
                    parent_mechanism_id=mechanism_id,
                    component_index=index,
                    detectors=tuple(sorted(unique_dets)),
                    observables=tuple(sorted(unique_obs)),
                )
            )
            mech_dets ^= unique_dets
            mech_obs ^= unique_obs

        for d in mech_dets:
            h_rows.append(d)
            h_cols.append(mechanism_id)
        for o in mech_obs:
            l_rows.append(o)
            l_cols.append(mechanism_id)

        mechanism_id += 1

    n_mechanisms = mechanism_id
    n_detectors = dem.num_detectors
    n_observables = dem.num_observables

    h = csc_matrix(
        (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
        shape=(n_detectors, n_mechanisms),
        dtype=np.uint8,
    )
    ell = csc_matrix(
        (np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
        shape=(n_observables, n_mechanisms),
        dtype=np.uint8,
    )

    coords, coord_dim = _detector_coords(dem, circuit, n_detectors)

    return DemStructure(
        h=h,
        l=ell,
        priors=np.asarray(priors, dtype=np.float64),
        components=tuple(components),
        detector_coords=coords,
        n_detectors=n_detectors,
        n_observables=n_observables,
        n_mechanisms=n_mechanisms,
        coord_dim=coord_dim,
        metadata={
            "n_components": len(components),
            "n_mechanisms_with_separator": n_with_separator,
            "n_cancelled_detector_targets": n_cancelled_detectors,
            "n_cancelled_observable_targets": n_cancelled_observables,
        },
    )


def _xor_reduce(values: list[int]) -> set[int]:
    """Reduce a target list mod 2: a value appearing twice cancels."""
    out: set[int] = set()
    for v in values:
        out ^= {v}
    return out


def _detector_coords(
    dem: stim.DetectorErrorModel,
    circuit: stim.Circuit | None,
    n_detectors: int,
) -> tuple[np.ndarray, int]:
    """Collect detector coordinates, padding ragged entries with NaN."""
    raw: dict[int, list[float]]
    if circuit is not None:
        raw = circuit.get_detector_coordinates()
    else:
        raw = dem.get_detector_coordinates()

    if not raw:
        return np.zeros((n_detectors, 0), dtype=np.float64), 0

    coord_dim = max((len(v) for v in raw.values()), default=0)
    coords = np.full((n_detectors, coord_dim), np.nan, dtype=np.float64)
    for det_id, values in raw.items():
        if 0 <= det_id < n_detectors:
            coords[det_id, : len(values)] = values
    return coords, coord_dim
