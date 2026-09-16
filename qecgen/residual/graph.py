"""Time slices and matching-graph node sets behind the residual features.

Two of the schema's inputs are structural rather than per-shot: which detectors share a
time slice, and which detectors sit next to a boundary edge or a logical edge of the
matching graph. Both are derived here once per dataset and frozen.

**Node sets come from the decomposed DEM, not from ``matching.edges()``.** PyMatching
merges parallel edges and keeps only the *first-inserted* edge's ``fault_ids`` (measured
on pymatching 2.4.0: ``error(0.1) D0 D1 L0`` followed by ``error(0.2) D0 D1`` yields one
edge with ``fault_ids={0}``; the reverse order yields ``fault_ids=set()``). A logical-edge
set read off ``edges()`` would therefore depend on instruction order, which is exactly
the kind of silent drift a feature schema must not carry. Walking the DEM's graphlike
components instead is order-independent, and a pair carrying components with *different*
observable sets is refused outright rather than resolved by whichever came first.

``matching.edges()`` is still consulted, but only as a cross-check: its collapsed pair set
and boundary-node set must equal the DEM-derived ones. That guards the one place where
the two readings can legitimately disagree - PyMatching applies its "at most two
detectors" rule to the *raw* target count while :func:`qecgen.dem.parse_dem` XOR-reduces
repeated targets first - and a DEM Stim would never emit is refused rather than decoded
with a graph the features do not describe.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pymatching
import stim
from scipy.sparse import csr_array

from qecgen.dem import parse_dem
from qecgen.hardware import detector_anchors

__all__ = [
    "EdgeRecord",
    "MatchingGraphSummary",
    "TimeSlices",
    "summarise_graph",
    "time_slices",
]


@dataclass(frozen=True)
class TimeSlices:
    """Detector -> ordered time slice, derived from the latest time coordinate.

    A detector's slice is the rank of its *latest* time coordinate among the sorted
    distinct latest-time values. Stim-generated circuits carry one ``(x, y, t)`` triple
    per detector; the Willow circuit carries several, with the final slice mixing data-qubit
    measurements at ``t`` with the last stabiliser at ``t - 1``. Taking the earliest time
    there would file a final-boundary detector under the previous round, so the final
    slice would be mis-sized and ``n_fired_final_round`` would count the wrong detectors.
    """

    slice_of_detector: np.ndarray
    """int32 ``(n_detectors,)``: slice index per detector."""

    times: tuple[float, ...]
    """Sorted distinct latest-time coordinate values, one per slice."""

    sizes: np.ndarray
    """int32 1-D ``(n_slices,)``: detectors per slice."""

    membership: csr_array
    """``(n_detectors, n_slices)`` one-hot uint8 sparse *array*.

    The array API is deliberate: ``csr_matrix`` products return ``np.matrix``, whose
    ``.sum`` keeps two dimensions and whose ``*`` is a matrix product, and its use raises
    deprecation warnings the suite turns into errors.
    """

    @property
    def n_slices(self) -> int:
        return len(self.times)


def time_slices(circuit: stim.Circuit) -> TimeSlices:
    """Group a circuit's detectors into ordered time slices.

    Every detector must carry finite coordinates. A coordinate-less detector has no
    slice, and silently assigning it one (say, slice 0) would move it into the first
    round and change ``n_fired_first_round`` for every shot in which it fires.
    """
    if circuit.num_detectors == 0:
        raise ValueError("time slices need at least one detector; the circuit has none")
    coordinates = circuit.get_detector_coordinates()
    missing = [d for d in range(circuit.num_detectors) if not coordinates.get(d)]
    if missing:
        raise ValueError(
            f"{len(missing)} detector(s) carry no coordinates (first: D{missing[0]}); "
            "time slices require a time coordinate for every detector"
        )
    anchors = detector_anchors(circuit)
    latest = np.asarray([anchors[d][2] for d in range(circuit.num_detectors)], dtype=np.float64)
    times = tuple(float(t) for t in np.unique(latest))
    slice_of_detector = np.searchsorted(np.asarray(times), latest).astype(np.int32)
    n_slices = len(times)
    sizes = np.bincount(slice_of_detector, minlength=n_slices).astype(np.int32)
    membership = csr_array(
        (
            np.ones(circuit.num_detectors, dtype=np.uint8),
            (np.arange(circuit.num_detectors), slice_of_detector),
        ),
        shape=(circuit.num_detectors, n_slices),
        dtype=np.uint8,
    )
    return TimeSlices(
        slice_of_detector=slice_of_detector,
        times=times,
        sizes=np.asarray(sizes, dtype=np.int32).ravel(),
        membership=membership,
    )


@dataclass(frozen=True)
class EdgeRecord:
    """One graphlike DEM component: an edge of the matching graph before collapsing.

    ``v`` is ``None`` for a boundary edge. Parallel components on the same pair each keep
    their own record, so the observable set of every component survives even though the
    adjacency matrix collapses them to one entry.
    """

    u: int
    v: int | None
    observables: frozenset[int]
    probability: float


@dataclass(frozen=True)
class MatchingGraphSummary:
    """Frozen node sets and adjacency of one decoder's matching graph."""

    n_detectors: int

    adjacency: csr_array
    """``(D, D)`` symmetric 0/1 uint8; non-boundary, parallel edges collapsed, no loops."""

    boundary_adjacent: np.ndarray
    """bool ``(D,)``: endpoints of 1-detector components."""

    logical_adjacent: np.ndarray
    """bool ``(D,)``: endpoints of components whose observable set contains 0."""

    edge_records: tuple[EdgeRecord, ...]
    """DEM-derived components; fault information kept apart from the collapsed adjacency."""

    matching_edge_weights: tuple[tuple[int, int | None, float], ...]
    """``(u, v, weight)`` from ``matching.edges()``, recorded for provenance only."""

    n_components: int
    n_boundary_components: int
    n_logical_components: int
    n_collapsed_pairs: int
    n_conflicting_pairs: int
    n_ignored_components: int
    """Components with more than two detectors, which PyMatching does not decode."""

    def digest(self) -> str:
        """sha256 over the sorted edge records: the identity a checkpoint is keyed on.

        Probabilities are rounded to 1e-12 so that a DEM re-serialised through Stim, which
        prints shortest round-trip floats, hashes the same as the object it came from.
        """
        rows = sorted(
            (r.u, -1 if r.v is None else r.v, sorted(r.observables), round(r.probability, 12))
            for r in self.edge_records
        )
        payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def summarise_graph(
    dem: stim.DetectorErrorModel, matching: pymatching.Matching, n_detectors: int
) -> MatchingGraphSummary:
    """Derive the node sets from ``dem`` and cross-check them against ``matching``.

    Raises ``ValueError`` on every condition under which the features would be computed
    against a graph other than the one PyMatching decodes with: a detector-count
    mismatch, a node id outside the detector range, more or fewer than one fault id, an
    empty logical-edge set, a pair carrying components with different observable sets,
    or a pair/boundary set that ``matching.edges()`` does not reproduce.
    """
    if dem.num_detectors != n_detectors:
        raise ValueError(
            f"DEM declares {dem.num_detectors} detectors but the dataset has {n_detectors}"
        )
    if matching.num_detectors != n_detectors:
        raise ValueError(
            f"matching has {matching.num_detectors} detectors but the dataset has {n_detectors}"
        )
    if matching.num_fault_ids != 1:
        raise ValueError(
            f"matching carries {matching.num_fault_ids} fault ids; the residual schema is "
            "defined for exactly one logical observable"
        )

    structure = parse_dem(dem)
    records: list[EdgeRecord] = []
    n_ignored = 0
    for component in structure.components:
        # parse_dem has already XOR-reduced repeated targets; the size rule is enforced
        # here again rather than trusted, because it is what makes a record an edge.
        if len(component.detectors) == 0 or len(component.detectors) > 2:
            n_ignored += 1
            continue
        if any(d >= n_detectors for d in component.detectors):
            raise ValueError(
                f"DEM component touches detector {max(component.detectors)} "
                f"but only {n_detectors} detectors exist"
            )
        u = component.detectors[0]
        v = component.detectors[1] if len(component.detectors) == 2 else None
        records.append(
            EdgeRecord(
                u=u,
                v=v,
                observables=frozenset(component.observables),
                probability=float(structure.priors[component.parent_mechanism_id]),
            )
        )

    observable_sets: dict[tuple[int, int | None], set[frozenset[int]]] = defaultdict(set)
    boundary = np.zeros(n_detectors, dtype=np.bool_)
    logical = np.zeros(n_detectors, dtype=np.bool_)
    pairs: set[tuple[int, int]] = set()
    n_boundary_components = 0
    n_logical_components = 0
    for record in records:
        observable_sets[(record.u, record.v)].add(record.observables)
        if record.v is None:
            boundary[record.u] = True
            n_boundary_components += 1
        else:
            pairs.add((record.u, record.v))
        if 0 in record.observables:
            n_logical_components += 1
            logical[record.u] = True
            if record.v is not None:
                logical[record.v] = True

    n_conflicting = sum(1 for sets in observable_sets.values() if len(sets) > 1)
    if n_conflicting:
        raise ValueError(
            f"{n_conflicting} detector pair(s)/boundary node(s) carry components with "
            "conflicting observable sets; PyMatching keeps only the first-inserted fault "
            "ids on merge, so the logical-edge set would depend on instruction order"
        )
    if not logical.any():
        raise ValueError("no graphlike DEM component flips logical observable 0")

    edge_weights = _cross_check_matching_edges(matching, pairs, boundary)

    if pairs:
        rows = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
        cols = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    else:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
    adjacency = csr_array(
        (
            np.ones(2 * len(pairs), dtype=np.uint8),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n_detectors, n_detectors),
        dtype=np.uint8,
    )

    return MatchingGraphSummary(
        n_detectors=n_detectors,
        adjacency=adjacency,
        boundary_adjacent=boundary,
        logical_adjacent=logical,
        edge_records=tuple(records),
        matching_edge_weights=edge_weights,
        n_components=len(records),
        n_boundary_components=n_boundary_components,
        n_logical_components=n_logical_components,
        n_collapsed_pairs=len(pairs),
        n_conflicting_pairs=n_conflicting,
        n_ignored_components=n_ignored,
    )


def _cross_check_matching_edges(
    matching: pymatching.Matching, pairs: set[tuple[int, int]], boundary: np.ndarray
) -> tuple[tuple[int, int | None, float], ...]:
    """Require ``matching.edges()`` to reproduce the DEM-derived pair and boundary sets.

    Returns the per-edge weights for provenance. The weights themselves are not
    validated against the DEM: PyMatching's merge arithmetic is its own business, and the
    features only ever consume the *topology*.
    """
    matching_pairs: set[tuple[int, int]] = set()
    matching_boundary: set[int] = set()
    weights: list[tuple[int, int | None, float]] = []
    for edge in matching.edges():
        u = int(edge[0])
        v = None if edge[1] is None else int(edge[1])
        weight = float(edge[2]["weight"])
        if v is None:
            matching_boundary.add(u)
        else:
            matching_pairs.add((min(u, v), max(u, v)))
        weights.append((u, v, weight))

    dem_boundary = {int(d) for d in np.flatnonzero(boundary)}
    if matching_pairs != pairs:
        raise ValueError(
            "matching.edges() pair set differs from the DEM-derived one: "
            f"{len(matching_pairs ^ pairs)} pair(s) disagree"
        )
    if matching_boundary != dem_boundary:
        raise ValueError(
            "matching.edges() boundary set differs from the DEM-derived one: "
            f"{len(matching_boundary ^ dem_boundary)} node(s) disagree"
        )
    weights.sort(key=lambda w: (w[0], w[1] is None, -1 if w[1] is None else w[1]))
    return tuple(weights)
