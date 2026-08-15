"""The JSON contract between the web server and a generation worker.

One implementation of spec (de)serialisation, used from both ends. The server encodes a
:mod:`qecgen.run` spec onto the worker's stdin; the worker decodes it back. Putting this
in a shared module rather than writing it twice is what stops the two sides from drifting
apart on, say, whether ``base_p`` may be null.

Standard library plus :mod:`qecgen.run` only — deliberately no pydantic. The worker must
stay startable without the ``ui`` extra installed, and its import cost is paid once per
job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import DriftCondition, StructureLevel
from qecgen.environments import DriftAxis
from qecgen.run import DriftSpec, GenerateSpec, MultiEnvSpec, RunSpec

__all__ = [
    "MODES",
    "encode_line",
    "mode_of",
    "spec_from_json",
    "spec_to_json",
]

MODES = ("generate", "multi-env", "drift")
"""The run kinds a worker accepts, named as the CLI commands they mirror."""


def mode_of(spec: RunSpec) -> str:
    """The mode name for a spec."""
    match spec:
        case GenerateSpec():
            return "generate"
        case MultiEnvSpec():
            return "multi-env"
        case DriftSpec():
            return "drift"


def spec_to_json(spec: RunSpec) -> dict[str, Any]:
    """Encode a spec as a JSON-safe dict, tagged with its mode.

    Enums are ``StrEnum`` so they encode as their own values; ``Path`` becomes a string
    and is rebuilt on the far side.
    """
    shared: dict[str, Any] = {
        "mode": mode_of(spec),
        "distance": spec.distance,
        "seed": spec.seed,
        "out": str(spec.out),
        "fmt": spec.fmt,
        "noise_model": str(spec.noise_model),
        "rounds": spec.rounds,
        "basis": str(spec.basis),
        "rotated": spec.rotated,
        "structure_level": str(spec.structure_level),
        "emit_mechanisms": spec.emit_mechanisms,
        "chunk_size": spec.chunk_size,
    }
    match spec:
        case GenerateSpec():
            return {**shared, "p": spec.p, "shots": spec.shots}
        case MultiEnvSpec():
            return {
                **shared,
                "axis_values": list(spec.axis_values),
                "shots_per_env": spec.shots_per_env,
                "axis": str(spec.axis),
                "base_p": spec.base_p,
                "shuffle": spec.shuffle,
            }
        case DriftSpec():
            return {
                **shared,
                "train_p": spec.train_p,
                "test_values": list(spec.test_values),
                "shots": spec.shots,
                "condition": str(spec.condition),
                "axis": str(spec.axis),
            }


def spec_from_json(payload: dict[str, Any]) -> RunSpec:
    """Rebuild a spec from :func:`spec_to_json` output.

    Raises:
        ValueError: for an unknown mode or an enum value outside its choice set. Same
            exception type the domain raises for bad input, so a caller mapping
            ``ValueError`` to a 400 needs no special case here.
    """
    mode = payload.get("mode")
    shared: dict[str, Any] = {
        "distance": payload["distance"],
        "seed": payload["seed"],
        "out": Path(payload["out"]),
        "fmt": payload["fmt"],
        "noise_model": NoiseModel(payload["noise_model"]),
        "rounds": payload["rounds"],
        "basis": Basis(payload["basis"]),
        "rotated": payload["rotated"],
        "structure_level": StructureLevel(payload["structure_level"]),
        "emit_mechanisms": payload["emit_mechanisms"],
        "chunk_size": payload["chunk_size"],
    }
    if mode == "generate":
        return GenerateSpec(**shared, p=payload["p"], shots=payload["shots"])
    if mode == "multi-env":
        return MultiEnvSpec(
            **shared,
            axis_values=tuple(payload["axis_values"]),
            shots_per_env=payload["shots_per_env"],
            axis=DriftAxis(payload["axis"]),
            base_p=payload["base_p"],
            shuffle=payload["shuffle"],
        )
    if mode == "drift":
        return DriftSpec(
            **shared,
            train_p=payload["train_p"],
            test_values=tuple(payload["test_values"]),
            shots=payload["shots"],
            condition=DriftCondition(payload["condition"]),
            axis=DriftAxis(payload["axis"]),
        )
    raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")


def encode_line(payload: dict[str, Any]) -> str:
    """One newline-terminated JSON message.

    ``allow_nan=False`` because ``NaN`` and ``Infinity`` are not JSON and would decode as
    a syntax error on the far side rather than as the bad number they are.
    """
    return json.dumps(payload, allow_nan=False) + "\n"
