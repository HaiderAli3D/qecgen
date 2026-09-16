"""Schema v1 residual features: exact definitions, reference agreement, leakage audit.

The numbers below are worked by hand on a six-detector context. Every feature is a small
formula, and a wrong one (say ``ddof=1`` or a big-endian unpack) still yields a well-formed
CSV, so the tests pin each definition on a shot where the right and wrong answers differ.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pymatching
import pytest
import stim
from scipy.sparse import csr_array

from qecgen.residual.features import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    INTEGER_COLUMNS,
    LABEL_COLUMNS,
    FeatureContext,
    audit_feature_columns,
    audit_feature_inputs,
    extract_features,
    extract_features_reference,
)
from qecgen.residual.graph import (
    EdgeRecord,
    MatchingGraphSummary,
    TimeSlices,
    summarise_graph,
    time_slices,
)
from qecgen.sampling import unpack_bits

_D = 6


def _hand_context() -> FeatureContext:
    """D=6, three slices of two detectors, pairs (0,1),(1,2),(3,4), boundary {0,5},
    logical {2,3}.

    The node sets are chosen for discriminating arithmetic, not for DEM consistency: the
    extractors consume only the arrays, and ``graph.py`` owns their derivation.
    """
    slice_of = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    membership = csr_array(
        (np.ones(_D, dtype=np.uint8), (np.arange(_D), slice_of)), shape=(_D, 3), dtype=np.uint8
    )
    slices = TimeSlices(
        slice_of_detector=slice_of,
        times=(0.0, 1.0, 2.0),
        sizes=np.asarray([2, 2, 2], dtype=np.int32),
        membership=membership,
    )
    pairs = [(0, 1), (1, 2), (3, 4)]
    rows = np.asarray([p[0] for p in pairs] + [p[1] for p in pairs])
    cols = np.asarray([p[1] for p in pairs] + [p[0] for p in pairs])
    adjacency = csr_array(
        (np.ones(2 * len(pairs), dtype=np.uint8), (rows, cols)), shape=(_D, _D), dtype=np.uint8
    )
    boundary = np.zeros(_D, dtype=np.bool_)
    boundary[[0, 5]] = True
    logical = np.zeros(_D, dtype=np.bool_)
    logical[[2, 3]] = True
    graph = MatchingGraphSummary(
        n_detectors=_D,
        adjacency=adjacency,
        boundary_adjacent=boundary,
        logical_adjacent=logical,
        edge_records=tuple(EdgeRecord(u, v, frozenset(), 0.01) for u, v in pairs),
        matching_edge_weights=(),
        n_components=3,
        n_boundary_components=2,
        n_logical_components=1,
        n_collapsed_pairs=3,
        n_conflicting_pairs=0,
        n_ignored_components=0,
    )
    return FeatureContext(n_detectors=_D, slices=slices, graph=graph)


def _pack(bits: str) -> np.ndarray:
    """Little-endian pack of a detector string written detector 0 first."""
    row = np.asarray([[int(c) for c in bits]], dtype=np.uint8)
    return np.packbits(row, axis=1, bitorder="little")


def _row(bits: str, pm_guess: int, pm_weight: float, ctx: FeatureContext) -> dict[str, float]:
    features = extract_features(
        _pack(bits),
        np.asarray([pm_guess], dtype=np.uint8),
        np.asarray([pm_weight], dtype=np.float64),
        ctx,
    )
    assert features.shape == (1, len(FEATURE_COLUMNS))
    assert features.dtype == np.float64
    return dict(zip(FEATURE_COLUMNS, features[0].tolist(), strict=True))


class TestSchema:
    def test_column_lists(self) -> None:
        assert len(FEATURE_COLUMNS) == 24
        assert FEATURE_COLUMNS[0] == "n_fired_total"
        assert FEATURE_COLUMNS[-1] == "pm_guess"
        assert LABEL_COLUMNS == ("truth", "pm_wrong", "run_id", "split")
        assert ALL_COLUMNS == FEATURE_COLUMNS + LABEL_COLUMNS
        assert len(set(ALL_COLUMNS)) == len(ALL_COLUMNS)
        assert set(ALL_COLUMNS) >= INTEGER_COLUMNS
        assert "split" not in INTEGER_COLUMNS
        assert {"pm_guess", "truth", "pm_wrong", "run_id"} <= INTEGER_COLUMNS

    def test_audit_accepts_exact_columns(self) -> None:
        audit_feature_columns(ALL_COLUMNS)
        audit_feature_columns(list(ALL_COLUMNS))

    def test_audit_rejects_reordered_renamed_extra_and_missing(self) -> None:
        columns = list(ALL_COLUMNS)
        reordered = columns[1:] + columns[:1]
        renamed = ["n_fired" if c == "n_fired_total" else c for c in columns]
        with pytest.raises(ValueError):
            audit_feature_columns(reordered)
        with pytest.raises(ValueError):
            audit_feature_columns(renamed)
        with pytest.raises(ValueError):
            audit_feature_columns([*columns, "drift_state"])
        with pytest.raises(ValueError):
            audit_feature_columns(columns[:-1])
        with pytest.raises(ValueError):
            audit_feature_columns(list(FEATURE_COLUMNS))


class TestHandBuiltShots:
    def test_empty_syndrome_is_all_zero_except_weight(self) -> None:
        row = _row("000000", 0, 1.5, _hand_context())

        for name, value in row.items():
            if name in {"pm_weight", "pm_weight_per_fired"}:
                assert value == 1.5, name
            else:
                assert value == 0.0, name

    def test_two_fired_in_first_slice(self) -> None:
        """Also the little-endian negative control: byte 0b00000011 unpacked big-endian
        lands on bits 6 and 7, outside the six detectors, and every count would be 0."""
        row = _row("110000", 1, 3.0, _hand_context())

        assert row["n_fired_total"] == 2
        assert row["frac_fired"] == pytest.approx(2 / 6)
        assert row["n_fired_first_round"] == 2
        assert row["frac_fired_first_round"] == 1.0
        assert row["n_fired_final_round"] == 0
        assert row["frac_fired_final_round"] == 0.0
        assert row["fired_per_round_mean"] == pytest.approx(2 / 3)
        assert row["fired_per_round_max"] == 2
        assert row["fired_per_round_std"] == pytest.approx(math.sqrt(8 / 9))
        assert row["fired_per_round_range"] == 2
        assert row["round_of_max_fired"] == 0.0
        assert row["n_active_rounds"] == 1
        assert row["frac_active_rounds"] == pytest.approx(1 / 3)
        assert row["fired_round_center"] == 0.0
        assert row["fired_round_spread"] == 0.0
        assert row["n_fired_boundary_adjacent"] == 1
        assert row["frac_fired_boundary_adjacent"] == 0.5
        assert row["n_fired_logical_edge_adjacent"] == 0
        assert row["frac_fired_logical_edge_adjacent"] == 0.0
        assert row["n_fired_neighbor_pairs"] == 1
        assert row["n_fired_isolated"] == 0
        assert row["pm_weight"] == 3.0
        assert row["pm_weight_per_fired"] == 1.5
        assert row["pm_guess"] == 1

    def test_three_isolated_fired_across_slices(self) -> None:
        row = _row("101001", 0, 2.0, _hand_context())

        assert row["n_fired_total"] == 3
        assert row["frac_fired"] == 0.5
        assert row["n_fired_first_round"] == 1
        assert row["frac_fired_first_round"] == 0.5
        assert row["n_fired_final_round"] == 1
        assert row["frac_fired_final_round"] == 0.5
        assert row["fired_per_round_mean"] == 1.0
        assert row["fired_per_round_max"] == 1
        assert row["fired_per_round_std"] == 0.0
        assert row["fired_per_round_range"] == 0
        assert row["round_of_max_fired"] == 0.0  # earliest maximum
        assert row["n_active_rounds"] == 3
        assert row["frac_active_rounds"] == 1.0
        # t_norm = (0, 0.5, 1); one fired detector per slice.
        assert row["fired_round_center"] == pytest.approx(0.5)
        assert row["fired_round_spread"] == pytest.approx(math.sqrt(1.25 / 3 - 0.25))
        assert row["n_fired_boundary_adjacent"] == 2
        assert row["frac_fired_boundary_adjacent"] == 1.0
        assert row["n_fired_logical_edge_adjacent"] == 1
        assert row["frac_fired_logical_edge_adjacent"] == 0.5
        assert row["n_fired_neighbor_pairs"] == 0
        assert row["n_fired_isolated"] == 3
        assert row["pm_weight_per_fired"] == pytest.approx(2.0 / 3)
        assert row["pm_guess"] == 0

    def test_round_of_max_takes_the_latest_slice_when_it_is_the_unique_max(self) -> None:
        row = _row("000011", 0, 0.0, _hand_context())

        assert row["round_of_max_fired"] == 1.0
        assert row["fired_round_center"] == 1.0
        assert row["fired_round_spread"] == 0.0
        assert row["n_fired_neighbor_pairs"] == 0  # (4,5) is not an edge
        assert row["n_fired_isolated"] == 2

    def test_reference_matches_on_hand_built_shots(self) -> None:
        ctx = _hand_context()
        for bits, guess, weight in [("000000", 0, 1.5), ("110000", 1, 3.0), ("101001", 0, 2.0)]:
            vectorised = extract_features(
                _pack(bits),
                np.asarray([guess], dtype=np.uint8),
                np.asarray([weight], dtype=np.float64),
                ctx,
            )
            reference = extract_features_reference([int(c) for c in bits], guess, weight, ctx)
            assert len(reference) == len(FEATURE_COLUMNS)
            np.testing.assert_allclose(vectorised[0], np.asarray(reference), rtol=0, atol=1e-12)

    def test_integer_columns_hold_integral_values(self) -> None:
        ctx = _hand_context()
        packed = np.concatenate([_pack("110000"), _pack("101001"), _pack("011110")])
        features = extract_features(
            packed,
            np.asarray([1, 0, 1], dtype=np.uint8),
            np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
            ctx,
        )
        for index, name in enumerate(FEATURE_COLUMNS):
            if name in INTEGER_COLUMNS:
                assert np.array_equal(features[:, index], np.floor(features[:, index])), name

    def test_zero_rows_yield_an_empty_matrix(self) -> None:
        ctx = _hand_context()
        features = extract_features(
            np.zeros((0, 1), dtype=np.uint8),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.float64),
            ctx,
        )
        assert features.shape == (0, len(FEATURE_COLUMNS))

    def test_input_shape_and_value_errors(self) -> None:
        ctx = _hand_context()
        with pytest.raises(ValueError):
            extract_features(
                np.zeros((2, 2), dtype=np.uint8),  # wrong packed width for 6 detectors
                np.zeros(2, dtype=np.uint8),
                np.zeros(2, dtype=np.float64),
                ctx,
            )
        with pytest.raises(ValueError):
            extract_features(
                np.zeros((2, 1), dtype=np.uint8),
                np.zeros(3, dtype=np.uint8),  # row count mismatch
                np.zeros(2, dtype=np.float64),
                ctx,
            )
        with pytest.raises(ValueError):
            extract_features(
                np.zeros((2, 1), dtype=np.uint8),
                np.asarray([0, 2], dtype=np.uint8),  # non-binary guess
                np.zeros(2, dtype=np.float64),
                ctx,
            )
        with pytest.raises(ValueError):
            extract_features_reference([0] * 5, 0, 0.0, ctx)  # wrong bit count


class TestRealCircuitProperty:
    def test_vectorised_equals_reference_on_d3(
        self, d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel
    ) -> None:
        matching = pymatching.Matching.from_detector_error_model(d3_dem)
        n_detectors = d3_circuit.num_detectors
        ctx = FeatureContext(
            n_detectors=n_detectors,
            slices=time_slices(d3_circuit),
            graph=summarise_graph(d3_dem, matching, n_detectors),
        )
        audit_feature_inputs(ctx)
        sampler = d3_circuit.compile_detector_sampler(seed=7)
        packed, _ = sampler.sample(200, bit_packed=True, separate_observables=True)
        packed = np.asarray(packed, dtype=np.uint8)
        predictions, weights = matching.decode_batch(
            packed, return_weights=True, bit_packed_shots=True, bit_packed_predictions=False
        )
        pm_guess = np.asarray(predictions, dtype=np.uint8).reshape(200)
        pm_weight = np.asarray(weights, dtype=np.float64).reshape(200)

        features = extract_features(packed, pm_guess, pm_weight, ctx)
        bits = unpack_bits(packed, n_detectors)
        assert features.shape == (200, 24)
        assert bits.any(), "the sample must contain fired detectors to be a real test"
        assert not bits.all(axis=1).any()
        for i in range(200):
            reference = extract_features_reference(
                [int(b) for b in bits[i]], int(pm_guess[i]), float(pm_weight[i]), ctx
            )
            np.testing.assert_allclose(
                features[i], np.asarray(reference), rtol=1e-12, atol=1e-12, err_msg=f"shot {i}"
            )
        assert np.isfinite(features).all()
        # Every fraction-valued column stays in [0, 1].
        for name in FEATURE_COLUMNS:
            if name.startswith("frac_") or name in {
                "round_of_max_fired",
                "fired_round_center",
                "fired_round_spread",
            }:
                column = features[:, FEATURE_COLUMNS.index(name)]
                assert column.min() >= 0.0 and column.max() <= 1.0, name
        assert features[:, FEATURE_COLUMNS.index("fired_round_spread")].max() <= 0.5

    def test_all_zero_rows_on_the_real_context(
        self, d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel
    ) -> None:
        matching = pymatching.Matching.from_detector_error_model(d3_dem)
        n_detectors = d3_circuit.num_detectors
        ctx = FeatureContext(
            n_detectors=n_detectors,
            slices=time_slices(d3_circuit),
            graph=summarise_graph(d3_dem, matching, n_detectors),
        )
        packed = np.zeros((3, 3), dtype=np.uint8)
        features = extract_features(
            packed, np.zeros(3, dtype=np.uint8), np.zeros(3, dtype=np.float64), ctx
        )
        assert np.array_equal(features, np.zeros((3, 24)))


class TestLeakageAudit:
    def test_context_has_exactly_the_three_fields(self) -> None:
        assert [f.name for f in dataclasses.fields(FeatureContext)] == [
            "n_detectors",
            "slices",
            "graph",
        ]

    def test_context_refuses_truth_field(self) -> None:
        ctx = _hand_context()
        with pytest.raises(TypeError):
            FeatureContext(  # type: ignore[call-arg]
                n_detectors=ctx.n_detectors, slices=ctx.slices, graph=ctx.graph, truth=1
            )

    def test_context_is_frozen(self) -> None:
        ctx = _hand_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.n_detectors = 7  # type: ignore[misc]

    def test_audit_accepts_a_clean_context(self) -> None:
        audit_feature_inputs(_hand_context())

    def test_audit_rejects_a_subclass_adding_an_attribute(self) -> None:
        ctx = _hand_context()

        @dataclass(frozen=True)
        class Leaky(FeatureContext):
            truth: int = 0

        leaky = Leaky(n_detectors=ctx.n_detectors, slices=ctx.slices, graph=ctx.graph, truth=1)
        with pytest.raises(ValueError, match="truth"):
            audit_feature_inputs(leaky)

    def test_audit_rejects_a_smuggled_instance_attribute(self) -> None:
        ctx = _hand_context()
        object.__setattr__(ctx, "hidden_state", np.zeros(3))
        with pytest.raises(ValueError, match="hidden_state"):
            audit_feature_inputs(ctx)

    def test_context_validates_its_structure(self) -> None:
        ctx = _hand_context()
        with pytest.raises(ValueError):
            FeatureContext(n_detectors=5, slices=ctx.slices, graph=ctx.graph)
        no_boundary = dataclasses.replace(ctx.graph, boundary_adjacent=np.zeros(_D, dtype=np.bool_))
        with pytest.raises(ValueError, match="boundary"):
            FeatureContext(n_detectors=_D, slices=ctx.slices, graph=no_boundary)
        no_logical = dataclasses.replace(ctx.graph, logical_adjacent=np.zeros(_D, dtype=np.bool_))
        with pytest.raises(ValueError, match="logical"):
            FeatureContext(n_detectors=_D, slices=ctx.slices, graph=no_logical)
