"""Request models for the run endpoints.

Validation happens twice on purpose. The constraints here are the cheap ones — a distance
below 2, a probability above 1, a format that is not registered — so the browser gets an
immediate, field-attributable error. The domain remains the authority: it owns the rules
that need a built circuit or several environments side by side (``code_capacity`` with
``rounds != 1``, environments that will not stack, a ``FROZEN_PRIOR`` drift study whose
DEMs differ in width), and those surface as ``ValueError`` from the worker.

Mirroring domain rules here would be worse than leaving them out. Two copies of a rule
drift, and the copy the user sees would start disagreeing with the copy that decides what
gets written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import DriftCondition, StructureLevel
from qecgen.environments import DriftAxis
from qecgen.exporters import EXPORTERS
from qecgen.run import DriftSpec, GenerateSpec, JobSpec, MultiEnvSpec, ScoreSpec
from qecgen.sampling import DEFAULT_CHUNK_SIZE

__all__ = [
    "SELECTABLE_DRIFT_CONDITIONS",
    "DriftRequest",
    "GenerateRequest",
    "JobRequest",
    "MultiEnvRequest",
    "RunRequest",
    "ScoreRequest",
]

SELECTABLE_DRIFT_CONDITIONS = tuple(
    condition for condition in DriftCondition if condition is not DriftCondition.NOT_APPLICABLE
)
"""Drift conditions a user may pick.

``NOT_APPLICABLE`` is excluded because ``build_drift_environments`` always refuses it — a
drift study has to state its condition. The CLI still offers it and fails at run time;
there is no reason for a dropdown to repeat that.
"""

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _SharedFields(TypedDict):
    """The fields every spec shares.

    A ``TypedDict`` rather than ``dict[str, object]`` so ``**shared`` into a frozen spec
    stays type-checked. Untyped unpacking would let a renamed or retyped spec field pass
    review here and fail at the first request instead.
    """

    distance: int
    seed: int
    out: Path
    fmt: str
    noise_model: NoiseModel
    rounds: int | None
    basis: Basis
    rotated: bool
    structure_level: StructureLevel
    emit_mechanisms: bool
    chunk_size: int


class _Base(BaseModel):
    """Fields every run kind shares."""

    model_config = ConfigDict(extra="forbid")

    distance: Annotated[int, Field(ge=2)]
    seed: Annotated[int, Field(ge=0)] = 0
    out: Annotated[str, Field(min_length=1)]
    fmt: str = "hdf5"
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    rounds: Annotated[int, Field(ge=1)] | None = None
    basis: Basis = Basis.Z
    rotated: bool = True
    structure_level: StructureLevel = StructureLevel.NONE
    emit_mechanisms: bool = False
    chunk_size: Annotated[int, Field(ge=1)] = DEFAULT_CHUNK_SIZE

    @field_validator("fmt")
    @classmethod
    def _known_format(cls, value: str) -> str:
        if value not in EXPORTERS:
            valid = ", ".join(sorted(EXPORTERS))
            raise ValueError(f"unknown format {value!r}; available: {valid}")
        return value

    def _shared(self, data_root: Path) -> _SharedFields:
        from qecgen.ui.datasets import resolve_within

        return {
            "distance": self.distance,
            "seed": self.seed,
            "out": resolve_within(data_root, self.out),
            "fmt": self.fmt,
            "noise_model": self.noise_model,
            "rounds": self.rounds,
            "basis": self.basis,
            "rotated": self.rotated,
            "structure_level": self.structure_level,
            "emit_mechanisms": self.emit_mechanisms,
            "chunk_size": self.chunk_size,
        }


class GenerateRequest(_Base):
    """A single-environment run."""

    mode: Literal["generate"] = "generate"
    p: Probability
    shots: Annotated[int, Field(ge=1)]

    def to_spec(self, data_root: Path) -> GenerateSpec:
        """Resolve into the domain spec, confining ``out`` to the data root."""
        return GenerateSpec(**self._shared(data_root), p=self.p, shots=self.shots)


class MultiEnvRequest(_Base):
    """A pooled multi-environment run."""

    mode: Literal["multi-env"] = "multi-env"
    axis_values: Annotated[list[float], Field(min_length=1)]
    shots_per_env: Annotated[int, Field(ge=1)]
    axis: DriftAxis = DriftAxis.P
    base_p: Probability | None = None
    shuffle: bool = True

    @model_validator(mode="after")
    def _base_p_for_non_p_axes(self) -> MultiEnvRequest:
        if self.axis is not DriftAxis.P and self.base_p is None:
            raise ValueError(
                f"axis={self.axis} varies something other than the error rate, so base_p "
                "must say what the underlying rate is"
            )
        return self

    def to_spec(self, data_root: Path) -> MultiEnvSpec:
        """Resolve into the domain spec, confining ``out`` to the data root."""
        return MultiEnvSpec(
            **self._shared(data_root),
            axis_values=tuple(self.axis_values),
            shots_per_env=self.shots_per_env,
            axis=self.axis,
            base_p=self.base_p,
            shuffle=self.shuffle,
        )


class DriftRequest(_Base):
    """A drift study. ``out`` names a directory."""

    mode: Literal["drift"] = "drift"
    train_p: Probability
    test_values: Annotated[list[float], Field(min_length=1)]
    shots: Annotated[int, Field(ge=1)]
    condition: DriftCondition = DriftCondition.FROZEN_PRIOR
    axis: DriftAxis = DriftAxis.P
    structure_level: StructureLevel = StructureLevel.DEM

    @field_validator("condition")
    @classmethod
    def _condition_is_stated(cls, value: DriftCondition) -> DriftCondition:
        if value is DriftCondition.NOT_APPLICABLE:
            raise ValueError(
                "a drift study must state its condition: pass frozen_prior or oracle_calibrated"
            )
        return value

    def to_spec(self, data_root: Path) -> DriftSpec:
        """Resolve into the domain spec, confining ``out`` to the data root."""
        return DriftSpec(
            **self._shared(data_root),
            train_p=self.train_p,
            test_values=tuple(self.test_values),
            shots=self.shots,
            condition=self.condition,
            axis=self.axis,
        )


class ScoreRequest(BaseModel):
    """Score a supplied correction against a dataset. Reads only; writes nothing.

    Both paths are confined to the data root like every other path the browser sends.
    There is no upload endpoint on purpose: this server has no authentication, and a
    write path that accepts arbitrary bytes is a different security posture from one
    that only reads what is already there.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["score"] = "score"
    dataset: Annotated[str, Field(min_length=1)]
    correction: Annotated[str, Field(min_length=1)]
    fmt: str | None = None
    unpacked: bool = False
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05

    @field_validator("fmt")
    @classmethod
    def _known_format(cls, value: str | None) -> str | None:
        if value is not None and value not in EXPORTERS:
            valid = ", ".join(sorted(EXPORTERS))
            raise ValueError(f"unknown format {value!r}; available: {valid}")
        return value

    def to_spec(self, data_root: Path) -> ScoreSpec:
        """Resolve into the domain spec, confining both paths to the data root."""
        from qecgen.ui.datasets import resolve_within

        return ScoreSpec(
            dataset=resolve_within(data_root, self.dataset),
            correction=resolve_within(data_root, self.correction),
            fmt=self.fmt,
            unpacked=self.unpacked,
            alpha=self.alpha,
        )


RunRequest = Annotated[
    GenerateRequest | MultiEnvRequest | DriftRequest,
    Field(discriminator="mode"),
]
"""The dataset-producing requests. Kept as its own union because the cost preview and the
streaming decision are questions only these can answer."""

JobRequest = Annotated[
    GenerateRequest | MultiEnvRequest | DriftRequest | ScoreRequest,
    Field(discriminator="mode"),
]
"""One request body for all three run kinds, discriminated on ``mode``.

A discriminated union rather than three endpoints so the browser posts to one URL and
gets one error shape, and so ``mode`` is validated before any of the mode-specific fields
produce confusing messages about the wrong shape.
"""


def to_spec(
    request: GenerateRequest | MultiEnvRequest | DriftRequest | ScoreRequest, data_root: Path
) -> JobSpec:
    """Resolve any job request into its domain spec."""
    return request.to_spec(data_root)
