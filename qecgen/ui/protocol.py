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
import math
from pathlib import Path
from typing import Any

from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import DriftCondition, StructureLevel
from qecgen.environments import DriftAxis
from qecgen.run import (
    DriftSpec,
    GenerateSpec,
    JobSpec,
    MultiEnvSpec,
    QaSpec,
    RunSpec,
    ScoreSpec,
    SweepSpec,
)

__all__ = [
    "ANALYSIS_MODES",
    "GENERATION_MODES",
    "MODES",
    "encode_line",
    "json_safe",
    "mode_of",
    "spec_from_json",
    "spec_to_json",
]

GENERATION_MODES = ("generate", "multi-env", "drift")
"""Modes that produce a dataset, named as the CLI commands they mirror."""

ANALYSIS_MODES = ("score", "qa", "sweep")
"""Modes that read existing files and report on them, producing no dataset."""

MODES = GENERATION_MODES + ANALYSIS_MODES
"""Every mode a worker accepts."""


def mode_of(spec: JobSpec) -> str:
    """The mode name for a spec."""
    match spec:
        case GenerateSpec():
            return "generate"
        case MultiEnvSpec():
            return "multi-env"
        case DriftSpec():
            return "drift"
        case ScoreSpec():
            return "score"
        case QaSpec():
            return "qa"
        case SweepSpec():
            return "sweep"


def _generation_fields(spec: RunSpec) -> dict[str, Any]:
    """The fields every dataset-producing spec shares.

    Only those. This block used to be built for *every* spec on the assumption that a
    spec has a distance, an output format and a structure level — true of the three run
    kinds and of nothing else. An analysis spec has none of them.
    """
    return {
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


def spec_to_json(spec: JobSpec) -> dict[str, Any]:
    """Encode a spec as a JSON-safe dict, tagged with its mode.

    Enums are ``StrEnum`` so they encode as their own values; ``Path`` becomes a string
    and is rebuilt on the far side.
    """
    if isinstance(spec, SweepSpec):
        return {
            "mode": "sweep",
            "distances": list(spec.distances),
            "error_rates": list(spec.error_rates),
            "out": str(spec.out),
            "max_errors": spec.max_errors,
            "max_shots": spec.max_shots,
            "workers": spec.workers,
            "decoders": list(spec.decoders),
            "noise_model": str(spec.noise_model),
            "basis": str(spec.basis),
            "rotated": spec.rotated,
            "rounds": spec.rounds,
            "alpha": spec.alpha,
        }
    if isinstance(spec, QaSpec):
        return {
            "mode": "qa",
            "dataset": str(spec.dataset),
            "fmt": spec.fmt,
            "max_shots": spec.max_shots,
            "target_errors": spec.target_errors,
        }
    if isinstance(spec, ScoreSpec):
        return {
            "mode": "score",
            "dataset": str(spec.dataset),
            "correction": str(spec.correction),
            "fmt": spec.fmt,
            "unpacked": spec.unpacked,
            "alpha": spec.alpha,
        }
    shared: dict[str, Any] = {"mode": mode_of(spec), **_generation_fields(spec)}
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


def spec_from_json(payload: dict[str, Any]) -> JobSpec:
    """Rebuild a spec from :func:`spec_to_json` output.

    An ``if``-chain rather than a ``match``, so mypy cannot check it for exhaustiveness
    the way it checks :func:`mode_of` and :func:`spec_to_json`. ``tests/test_ui_worker.py``
    drives a round trip from ``typing.get_args(JobSpec)`` instead, which covers a new
    member automatically — stronger than the type checker can be here.

    Raises:
        ValueError: for an unknown mode or an enum value outside its choice set. Same
            exception type the domain raises for bad input, so a caller mapping
            ``ValueError`` to a 400 needs no special case here.
    """
    mode = payload.get("mode")
    # Checked before anything is read out of the payload. The "unknown mode" message
    # below used to be unreachable for exactly the payloads that needed it: building the
    # shared block first meant an unrecognised mode raised `KeyError: 'distance'`, so a
    # typo in the one field that selects the shape was reported as a missing field of a
    # shape that was never going to be right.
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    if mode == "sweep":
        return SweepSpec(
            distances=tuple(payload["distances"]),
            error_rates=tuple(payload["error_rates"]),
            out=Path(payload["out"]),
            max_errors=payload["max_errors"],
            max_shots=payload["max_shots"],
            workers=payload["workers"],
            decoders=tuple(payload["decoders"]),
            noise_model=NoiseModel(payload["noise_model"]),
            basis=Basis(payload["basis"]),
            rotated=payload["rotated"],
            rounds=payload["rounds"],
            alpha=payload["alpha"],
        )
    if mode == "qa":
        return QaSpec(
            dataset=Path(payload["dataset"]),
            fmt=payload["fmt"],
            max_shots=payload["max_shots"],
            target_errors=payload["target_errors"],
        )
    if mode == "score":
        return ScoreSpec(
            dataset=Path(payload["dataset"]),
            correction=Path(payload["correction"]),
            fmt=payload["fmt"],
            unpacked=payload["unpacked"],
            alpha=payload["alpha"],
        )
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
    # Reached only when a name is added to MODES without a branch here, which the
    # `get_args(JobSpec)` round trip in tests/test_ui_worker.py is what actually catches.
    raise ValueError(f"mode {mode!r} is registered but this decoder has no branch for it")


def encode_line(payload: dict[str, Any]) -> str:
    """One newline-terminated JSON message.

    ``allow_nan=False`` because ``NaN`` and ``Infinity`` are not JSON and would decode as
    a syntax error on the far side rather than as the bad number they are.
    """
    return json.dumps(payload, allow_nan=False) + "\n"


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively.

    :func:`encode_line` refuses ``NaN`` and ``Infinity``, which is right — they are not
    JSON — but the refusal lands at the worst possible moment. An analysis result is
    emitted in the terminal ``done`` message, *after* the work succeeded and
    :func:`~qecgen.run.staged` committed its files, so a single non-finite number in a
    summary turns a completed run into "the worker exited without reporting a result"
    with its output sitting on disk.

    No source of one is known: ``math.exp`` raises ``OverflowError`` rather than returning
    ``inf``, and every interval in :mod:`qecgen.qa` guards its degenerate cases. But that
    conclusion rests on several separate guards, one of them scipy's, and the cost of
    being wrong is a successful run reported as a failure. ``null`` is also what
    ``write_threshold_json`` already produces for an absent fit, so the two agree.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value
