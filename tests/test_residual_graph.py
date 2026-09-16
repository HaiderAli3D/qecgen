"""Time slices and matching-graph node sets for the residual features.

Every failure mode here is silent: a wrong slice map or a wrong boundary/logical set
produces a well-formed feature CSV whose columns simply mean something else. The tests
therefore pin the *rules* (latest time coordinate, DEM-derived node sets, order
invariance, refusal on conflict) rather than any single circuit's numbers.
"""

from __future__ import annotations

import numpy as np
import pymatching
import pytest
import stim
from scipy.sparse import csr_array

from qecgen.residual.graph import (
    EdgeRecord,
    MatchingGraphSummary,
    TimeSlices,
    summarise_graph,
    time_slices,
)

# A DEM with two *parallel* components on (0, 1) that carry the same observable set, a
# plain non-logical edge (1, 2), a logical boundary edge at 2 and a plain boundary edge
# at 3. Every instruction is graphlike, so every one becomes a record.
_PARALLEL_SAME_OBS = """
error(0.1) D0 D1 L0
error(0.2) D0 D1 L0
error(0.05) D1 D2
error(0.3) D2 L0
error(0.02) D3
"""


def _matching(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(dem)


def _pairs(summary: MatchingGraphSummary) -> set[tuple[int, int]]:
    rows, cols = summary.adjacency.nonzero()
    return {(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if r < c}


def _summarise(text: str) -> MatchingGraphSummary:
    dem = stim.DetectorErrorModel(text)
    return summarise_graph(dem, _matching(dem), dem.num_detectors)


class TestSummariseGraph:
    def test_parallel_components_with_same_observables_collapse(self) -> None:
        summary = _summarise(_PARALLEL_SAME_OBS)

        assert summary.n_detectors == 4
        assert _pairs(summary) == {(0, 1), (1, 2)}
        assert set(np.flatnonzero(summary.boundary_adjacent).tolist()) == {2, 3}
        assert set(np.flatnonzero(summary.logical_adjacent).tolist()) == {0, 1, 2}
        assert summary.n_components == 5
        assert summary.n_boundary_components == 2
        assert summary.n_logical_components == 3
        assert summary.n_collapsed_pairs == 2
        assert summary.n_conflicting_pairs == 0
        assert len(summary.edge_records) == 5

    def test_adjacency_is_symmetric_binary_uint8_without_self_loops(self) -> None:
        summary = _summarise(_PARALLEL_SAME_OBS)

        assert isinstance(summary.adjacency, csr_array)
        assert summary.adjacency.dtype == np.uint8
        assert summary.adjacency.shape == (4, 4)
        dense = summary.adjacency.toarray()
        assert np.array_equal(dense, dense.T)
        assert set(np.unique(dense).tolist()) <= {0, 1}
        assert int(np.trace(dense)) == 0
        # Parallel components collapse to *one* entry, never a count of 2.
        assert int(dense[0, 1]) == 1

    def test_edge_records_keep_fault_information_per_component(self) -> None:
        """The collapsed adjacency loses the observable sets; the records must not."""
        summary = _summarise(_PARALLEL_SAME_OBS)

        records = set(summary.edge_records)
        assert EdgeRecord(0, 1, frozenset({0}), 0.1) in records
        assert EdgeRecord(0, 1, frozenset({0}), 0.2) in records
        assert EdgeRecord(1, 2, frozenset(), 0.05) in records
        assert EdgeRecord(2, None, frozenset({0}), 0.3) in records
        assert EdgeRecord(3, None, frozenset(), 0.02) in records

    def test_result_is_invariant_under_instruction_order(self) -> None:
        """PyMatching's edges() depends on instruction order; the summary must not."""
        lines = [line for line in _PARALLEL_SAME_OBS.strip().splitlines() if line]
        reference = _summarise("\n".join(lines))
        permuted = _summarise("\n".join(reversed(lines)))

        assert _pairs(permuted) == _pairs(reference)
        assert np.array_equal(permuted.boundary_adjacent, reference.boundary_adjacent)
        assert np.array_equal(permuted.logical_adjacent, reference.logical_adjacent)
        assert set(permuted.edge_records) == set(reference.edge_records)
        assert permuted.digest() == reference.digest()

    def test_digest_is_sha256_hex_and_changes_with_probability(self) -> None:
        reference = _summarise(_PARALLEL_SAME_OBS)
        altered = _summarise(_PARALLEL_SAME_OBS.replace("error(0.02) D3", "error(0.03) D3"))

        assert len(reference.digest()) == 64
        assert int(reference.digest(), 16) >= 0
        assert reference.digest() != altered.digest()

    def test_parallel_components_with_different_observables_are_refused(self) -> None:
        """A merged edge keeps only the first-inserted fault ids; the answer would then
        depend on instruction order, so v1 refuses rather than picking one."""
        dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0\nerror(0.2) D0 D1\n")

        with pytest.raises(ValueError, match="conflicting"):
            summarise_graph(dem, _matching(dem), dem.num_detectors)

    def test_pymatching_keeps_first_inserted_fault_ids_on_merge(self) -> None:
        """Documents the pymatching 2.4.0 rule the refusal above guards against, so a
        future pymatching that merges fault ids differently is noticed here."""
        matching = pymatching.Matching()
        matching.add_edge(0, 1, fault_ids=set())
        matching.add_edge(0, 1, fault_ids={0}, merge_strategy="independent")

        edges = matching.edges()
        assert len(edges) == 1
        assert edges[0][2]["fault_ids"] == set()

    def test_three_detector_component_without_separator_is_ignored(self) -> None:
        """Mirrors PyMatching, which drops hyperedges instead of decoding them."""
        summary = _summarise("error(0.1) D0 D1 D2\nerror(0.1) D0 D1 L0\nerror(0.1) D2\n")

        assert summary.n_components == 2
        assert summary.n_ignored_components == 1
        assert _pairs(summary) == {(0, 1)}
        assert set(np.flatnonzero(summary.boundary_adjacent).tolist()) == {2}
        assert all(len({r.u, r.v}) <= 2 for r in summary.edge_records)

    def test_separated_components_are_split_before_the_size_rule(self) -> None:
        """A 4-detector mechanism written as two ``^``-joined components is two edges."""
        summary = _summarise(
            "error(0.1) D0 D1 ^ D2 D3 L0\nerror(0.1) D1 D2\nerror(0.1) D0\nerror(0.1) D3\n"
        )

        assert summary.n_components == 5
        assert _pairs(summary) == {(0, 1), (2, 3), (1, 2)}
        assert set(np.flatnonzero(summary.logical_adjacent).tolist()) == {2, 3}

    def test_no_logical_edge_is_refused(self) -> None:
        dem = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D1\n")

        with pytest.raises(ValueError):
            summarise_graph(dem, _matching(dem), dem.num_detectors)

    def test_logical_only_on_ignored_hyperedge_is_refused(self) -> None:
        """num_fault_ids is 1 here, yet no graphlike edge flips the observable."""
        dem = stim.DetectorErrorModel("error(0.1) D0 D1 D2 L0\nerror(0.1) D0 D1\nerror(0.1) D2\n")

        with pytest.raises(ValueError, match="logical"):
            summarise_graph(dem, _matching(dem), dem.num_detectors)

    def test_detector_count_mismatch_is_refused(self) -> None:
        dem = stim.DetectorErrorModel(_PARALLEL_SAME_OBS)

        with pytest.raises(ValueError):
            summarise_graph(dem, _matching(dem), 3)
        with pytest.raises(ValueError):
            summarise_graph(dem, _matching(dem), 5)

    def test_repeated_target_disagreement_with_pymatching_is_refused(self) -> None:
        """PyMatching counts raw targets for its size rule while the DEM walk XOR-reduces,
        so ``D0 D0 D1`` is a hyperedge to one and a boundary edge to the other. The
        cross-check against edges() exists to catch exactly this class of drift."""
        dem = stim.DetectorErrorModel(
            "error(0.1) D0 D0 D1 L0\nerror(0.1) D1 D2 L0\nerror(0.1) D2\n"
        )

        with pytest.raises(ValueError, match="edges"):
            summarise_graph(dem, _matching(dem), dem.num_detectors)

    def test_matching_edge_weights_are_recorded_for_provenance(self) -> None:
        summary = _summarise(_PARALLEL_SAME_OBS)

        assert len(summary.matching_edge_weights) == 4
        by_edge = {(u, v): w for u, v, w in summary.matching_edge_weights}
        assert set(by_edge) == {(0, 1), (1, 2), (2, None), (3, None)}
        # Merged parallel edge: p = 0.1 + 0.2 - 2*0.1*0.2 = 0.26.
        assert by_edge[(0, 1)] == pytest.approx(np.log((1 - 0.26) / 0.26))

    def test_real_d3_circuit_summary(
        self, d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel
    ) -> None:
        summary = summarise_graph(d3_dem, _matching(d3_dem), d3_circuit.num_detectors)

        assert summary.n_detectors == 24
        assert summary.n_conflicting_pairs == 0
        assert summary.logical_adjacent.sum() > 0
        assert summary.boundary_adjacent.dtype == np.bool_
        assert summary.logical_adjacent.dtype == np.bool_
        assert summary.boundary_adjacent.shape == (24,)
        assert summary.n_collapsed_pairs == len(_pairs(summary))


class TestTimeSlices:
    def test_rotated_d3_r3_has_four_slices(self) -> None:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )
        slices = time_slices(circuit)

        assert isinstance(slices, TimeSlices)
        assert slices.n_slices == 4
        assert slices.times == (0.0, 1.0, 2.0, 3.0)
        assert slices.sizes.tolist() == [4, 8, 8, 4]
        assert slices.sizes.dtype == np.int32
        assert slices.sizes.ndim == 1
        assert slices.slice_of_detector.dtype == np.int32
        assert slices.slice_of_detector.shape == (24,)

    def test_membership_is_one_hot_csr_array(self) -> None:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )
        slices = time_slices(circuit)

        assert isinstance(slices.membership, csr_array)
        assert slices.membership.dtype == np.uint8
        assert slices.membership.shape == (24, 4)
        dense = slices.membership.toarray()
        assert np.array_equal(dense.sum(axis=1), np.ones(24, dtype=np.int64))
        assert np.array_equal(dense.sum(axis=0), slices.sizes)
        assert np.array_equal(np.argmax(dense, axis=1), slices.slice_of_detector)
        # A sparse-array product must stay an ndarray, never np.matrix.
        counts = slices.membership.T @ np.ones((24, 3), dtype=np.int32)
        assert type(counts) is np.ndarray

    def test_six_tuple_coordinates_use_the_latest_time(self) -> None:
        """Willow detectors carry (x,y,t,x,y,t-1); the slice is the latest time."""
        circuit = stim.Circuit(
            """
            R 0 1 2
            X_ERROR(0.1) 0 1 2
            M 0 1 2
            DETECTOR(1, 1, 1, 1, 1, 0) rec[-1]
            DETECTOR(2, 2, 0) rec[-2]
            DETECTOR(3, 3, 3, 2, 2, 1) rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )
        slices = time_slices(circuit)

        assert slices.times == (0.0, 1.0, 3.0)
        assert slices.slice_of_detector.tolist() == [1, 0, 2]
        assert slices.sizes.tolist() == [1, 1, 1]

    def test_detector_without_coordinates_is_refused(self) -> None:
        circuit = stim.Circuit(
            """
            R 0 1
            X_ERROR(0.1) 0 1
            M 0 1
            DETECTOR(1, 1, 0) rec[-1]
            DETECTOR rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )

        with pytest.raises(ValueError):
            time_slices(circuit)

    def test_circuit_without_any_detector_is_refused(self) -> None:
        circuit = stim.Circuit("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")

        with pytest.raises(ValueError):
            time_slices(circuit)
