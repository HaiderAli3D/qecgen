"""The wire format between the server and a worker.

The protocol has two sides and they are hand-mapped field by field, so the failure this
file exists to catch is a spec that gains a field which the encoder never learns about:
the worker then silently constructs it with the default, and the recorded config disagrees
with the run that actually happened. Nothing else in the suite calls `spec_from_json`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import DriftCondition, StructureLevel
from qecgen.environments import DriftAxis
from qecgen.run import (
    DriftSpec,
    GenerateSpec,
    JobSpec,
    MultiEnvSpec,
    QaSpec,
    ScoreSpec,
    SweepSpec,
)
from qecgen.ui.protocol import MODES, encode_line, mode_of, spec_from_json, spec_to_json

# Every field set away from its default, so a dropped field shows up as a changed value
# rather than coincidentally matching.
SPECS: dict[str, JobSpec] = {
    "generate": GenerateSpec(
        distance=5,
        p=0.007,
        shots=321,
        seed=9,
        out=Path("out/g.npz"),
        fmt="npz",
        noise_model=NoiseModel.PHENOMENOLOGICAL,
        rounds=4,
        basis=Basis.X,
        rotated=False,
        structure_level=StructureLevel.DEM,
        emit_mechanisms=True,
        chunk_size=77,
    ),
    "multi-env": MultiEnvSpec(
        distance=5,
        axis_values=(0.3, 0.7),
        shots_per_env=64,
        seed=11,
        out=Path("out/m.h5"),
        fmt="hdf5",
        axis=DriftAxis.XZ_BIAS,
        base_p=0.004,
        noise_model=NoiseModel.PHENOMENOLOGICAL,
        rounds=3,
        basis=Basis.X,
        rotated=False,
        shuffle=False,
        structure_level=StructureLevel.COORDS,
        emit_mechanisms=True,
        chunk_size=55,
    ),
    "drift": DriftSpec(
        distance=7,
        train_p=0.005,
        test_values=(0.008, 0.012),
        shots=128,
        seed=13,
        condition=DriftCondition.ORACLE_CALIBRATED,
        out=Path("out/drift"),
        fmt="jsonl",
        axis=DriftAxis.P,
        noise_model=NoiseModel.PHENOMENOLOGICAL,
        rounds=5,
        basis=Basis.X,
        rotated=False,
        structure_level=StructureLevel.DEM,
        emit_mechanisms=True,
        chunk_size=33,
    ),
    "sweep": SweepSpec(
        distances=(3, 5),
        error_rates=(0.004, 0.009),
        out=Path("out/s.csv"),
        max_errors=42,
        max_shots=4242,
        workers=3,
        decoders=("pymatching", "vacuous"),
        noise_model=NoiseModel.PHENOMENOLOGICAL,
        rounds=6,
        basis=Basis.X,
        rotated=False,
        alpha=0.02,
    ),
    "score": ScoreSpec(
        dataset=Path("data/d.h5"),
        correction=Path("data/c.npy"),
        fmt="hdf5",
        unpacked=True,
        alpha=0.01,
    ),
    "qa": QaSpec(
        dataset=Path("data/d.h5"),
        fmt="hdf5",
        max_shots=1234,
        target_errors=56,
    ),
}


def test_every_mode_has_a_spec_under_test() -> None:
    """Guards the table above, so a new mode cannot be added without a round-trip."""
    assert set(SPECS) == set(MODES)


@pytest.mark.parametrize("mode", MODES)
def test_a_spec_survives_the_round_trip_unchanged(mode: str) -> None:
    spec = SPECS[mode]
    assert mode_of(spec) == mode
    assert spec_from_json(spec_to_json(spec)) == spec


@pytest.mark.parametrize("mode", MODES)
def test_every_field_of_every_spec_reaches_the_wire(mode: str) -> None:
    """The check that actually catches a forgotten field.

    A round-trip alone passes if the encoder drops a field whose value happens to equal
    its default. Comparing the payload's keys against the dataclass's fields fails the
    moment a spec grows one the encoder does not know about.
    """
    payload = spec_to_json(SPECS[mode])
    expected = {field.name for field in dataclasses.fields(SPECS[mode])}
    assert set(payload) - {"mode"} == expected


def test_an_unknown_mode_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        spec_from_json({"mode": "nope"})


def test_encode_line_refuses_non_json_floats() -> None:
    """NaN and Infinity are not JSON; they would decode on the far side as a syntax error
    rather than as the bad number they are."""
    with pytest.raises(ValueError):
        encode_line({"value": float("nan")})
