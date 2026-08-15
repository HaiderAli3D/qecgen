"""DEM parsing: the failure modes here are silent and produce well-formed wrong data."""

from __future__ import annotations

import numpy as np
import stim

from qecgen.circuits import NoiseModel, build_circuit
from qecgen.dem import DemStructure, parse_dem
from qecgen.validate import validate_dem_structure


def _count_error_instructions(dem: stim.DetectorErrorModel) -> int:
    return sum(1 for i in dem.flattened() if i.type == "error")


def _count_components(dem: stim.DetectorErrorModel) -> int:
    total = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        total += 1 + sum(1 for t in instruction.targets_copy() if t.is_separator())
    return total


def test_one_column_per_error_instruction_not_per_component(
    d3_dem: stim.DetectorErrorModel, d3_structure: DemStructure
) -> None:
    """The central invariant. Components must not become independent columns."""
    n_errors = _count_error_instructions(d3_dem)
    n_components = _count_components(d3_dem)

    assert d3_structure.n_mechanisms == n_errors
    assert len(d3_structure.components) == n_components
    # If this ever becomes equal, the parser has collapsed correlated mechanisms
    # into independent ones and is simulating a different noise model.
    assert n_components > n_errors


def test_separator_components_are_preserved_under_their_parent(
    d3_structure: DemStructure,
) -> None:
    by_parent: dict[int, int] = {}
    for component in d3_structure.components:
        by_parent[component.parent_mechanism_id] = (
            by_parent.get(component.parent_mechanism_id, 0) + 1
        )
    assert len(by_parent) == d3_structure.n_mechanisms
    assert max(by_parent.values()) > 1, "expected at least one decomposed mechanism"


def test_probability_is_not_duplicated_across_components(
    d3_dem: stim.DetectorErrorModel, d3_structure: DemStructure
) -> None:
    """One probability per mechanism, regardless of component count."""
    assert d3_structure.priors.shape == (d3_structure.n_mechanisms,)
    stim_probabilities = [i.args_copy()[0] for i in d3_dem.flattened() if i.type == "error"]
    assert np.allclose(d3_structure.priors, stim_probabilities)


def test_components_are_graphlike(d3_structure: DemStructure) -> None:
    """Each decomposed component touches at most two detectors."""
    assert all(len(c.detectors) <= 2 for c in d3_structure.components)


def test_mechanism_weight_legitimately_exceeds_two(d3_structure: DemStructure) -> None:
    """Guards against 're-fixing' the validator to assert weight<=2 on mechanisms.

    A mechanism with two components touches up to four detectors. Asserting the
    graphlike bound at mechanism level would fail on correct data.
    """
    histogram = d3_structure.mechanism_weight_histogram()
    assert max(histogram) > 2


def test_shapes_agree_with_dem(d3_dem: stim.DetectorErrorModel, d3_structure: DemStructure) -> None:
    assert d3_structure.h.shape == (d3_dem.num_detectors, d3_structure.n_mechanisms)
    assert d3_structure.l.shape == (d3_dem.num_observables, d3_structure.n_mechanisms)
    assert d3_structure.n_detectors == d3_dem.num_detectors
    assert d3_structure.n_observables == d3_dem.num_observables


def test_h_is_binary(d3_structure: DemStructure) -> None:
    assert d3_structure.h.dtype == np.uint8
    assert set(np.unique(d3_structure.h.toarray())) <= {0, 1}


def test_detector_coords_are_xyt(d3_structure: DemStructure) -> None:
    assert d3_structure.coord_dim == 3
    assert d3_structure.detector_coords.shape == (d3_structure.n_detectors, 3)
    assert not np.isnan(d3_structure.detector_coords).any()


def test_repeated_detector_cancels_mod_two() -> None:
    """A detector appearing twice in one mechanism must cancel, not accumulate.

    Constructed directly as DEM text because Stim's surface code circuits do not
    happen to produce this case; the parser must still be correct for it.
    """
    dem = stim.DetectorErrorModel("error(0.1) D0 D0 D1\ndetector D0\ndetector D1")
    structure = parse_dem(dem)
    assert structure.n_mechanisms == 1
    column = structure.h.toarray()[:, 0]
    assert column[0] == 0, "D0 appeared twice and must cancel"
    assert column[1] == 1
    assert structure.metadata["n_cancelled_detector_targets"] == 2


def test_observable_targets_land_in_l() -> None:
    dem = stim.DetectorErrorModel("error(0.25) D0 L0\ndetector D0")
    structure = parse_dem(dem)
    assert structure.h.toarray()[0, 0] == 1
    assert structure.l.toarray()[0, 0] == 1
    assert structure.priors[0] == 0.25


def test_structure_passes_its_own_validator(
    d3_structure: DemStructure, d3_dem: stim.DetectorErrorModel
) -> None:
    report = validate_dem_structure(d3_structure, d3_dem)
    assert report.ok, str(report)


def test_code_capacity_dem_parses() -> None:
    circuit, _ = build_circuit(3, 0.01, NoiseModel.CODE_CAPACITY)
    dem = circuit.detector_error_model(decompose_errors=True)
    structure = parse_dem(dem, circuit)
    assert structure.n_mechanisms == _count_error_instructions(dem)
    assert validate_dem_structure(structure, dem).ok
