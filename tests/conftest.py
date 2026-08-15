"""Shared fixtures and markers."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import cast

import pytest
import stim

from qecgen.circuits import NoiseModel, build_circuit
from qecgen.dem import DemStructure, parse_dem


def requires_mwpf[F: Callable[..., None]](test: F) -> F:
    """Gate a test on the optional `decoders` extra, both ways at once.

    The skipif and the registered marker do different jobs and neither substitutes for
    the other. The skipif is what keeps the suite green on a machine without the extra.
    The marker is what makes ``-m "not requires_mwpf"`` deselect this test up front,
    which is the flag ``pyproject.toml`` advertises. Applied with the skipif alone — as
    this was originally — that documented flag matched nothing at all.

    Lives here because two test files need it; the previous copy-in-each drifted the
    moment one docstring was edited.
    """
    skip_without_extra = pytest.mark.skipif(
        importlib.util.find_spec("mwpf") is None,
        reason='mwpf not installed; pip install -e ".[decoders]"',
    )
    # Cast because `MarkDecorator.__call__` is untyped; both marks land on one decorator.
    return cast(F, skip_without_extra(pytest.mark.requires_mwpf(test)))


@pytest.fixture(scope="session")
def d3_circuit() -> stim.Circuit:
    """d=3, 3 rounds, uniform circuit-level p=0.01."""
    circuit, _ = build_circuit(3, 0.01, NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL)
    return circuit


@pytest.fixture(scope="session")
def d3_dem(d3_circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Decomposed DEM of the d=3 circuit."""
    return d3_circuit.detector_error_model(decompose_errors=True)


@pytest.fixture(scope="session")
def d3_structure(d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel) -> DemStructure:
    """Parsed structure of the d=3 DEM."""
    return parse_dem(d3_dem, d3_circuit)
