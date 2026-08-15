"""Multi-environment and drift dataset construction.

Invariant causal learning requires training data drawn from several *distinct*
environments rather than one pooled distribution. This module builds those
environments, keeps per-shot provenance via ``environment_id``, and makes the drift
study's most important choice — where a test file's structure came from — explicit and
recorded rather than accidental.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy as np
import stim

from qecgen.circuits import (
    XZ_BIAS_SCOPE,
    Basis,
    ChannelVector,
    NoiseModel,
    apply_xz_bias,
    build_circuit_from_channels,
    channels_for,
    default_rounds,
)
from qecgen.dataset import (
    Contract,
    DatasetMeta,
    DriftCondition,
    EnvironmentSpec,
    InMemoryDataset,
    StreamingContentHasher,
    StructureLevel,
    content_hash,
    dem_digest,
)
from qecgen.dem import DemStructure, parse_dem
from qecgen.exporters.hdf5 import StreamingHDF5Writer
from qecgen.sampling import DEFAULT_CHUNK_SIZE, iter_chunks, packed_width

__all__ = [
    "AXIS_BUILDERS",
    "DriftAxis",
    "EnvironmentBuild",
    "build_drift_environments",
    "build_environment",
    "build_multi_environment",
    "build_single_environment",
    "derive_seeds",
    "drift_dataset_names",
    "stream_single_environment",
    "unbiased_point",
]


class DriftAxis(enum.StrEnum):
    """Which noise property a drift study varies.

    Sweeping the uniform rate alone mostly tests interpolation within one synthetic
    family. Held-out noise *families* are much stronger evidence of invariance than a
    held-out rate, which is why this is not limited to ``p``.
    """

    P = "p"
    """The single uniform rate. Axis value is the rate itself."""

    XZ_BIAS = "xz_bias"
    """Ratio ``pz / (px + py)`` on single-qubit data noise.

    Axis value is ``eta``; ``0.5`` is the unbiased (depolarising) point. Requires
    rewriting the generated circuit, because ``stim.Circuit.generated`` cannot express
    asymmetric channels. Two-qubit gate noise stays symmetric — see
    :func:`qecgen.circuits.apply_xz_bias`.
    """

    MEASUREMENT_RATIO = "measurement_ratio"
    """Measurement error scaled independently of gate error.

    Axis value multiplies ``before_measure_flip_probability`` relative to ``base_p``.
    """


@dataclass(frozen=True, slots=True)
class EnvironmentBuild:
    """A fully resolved environment: circuit, DEM and the spec describing it."""

    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    spec: EnvironmentSpec


def _axis_p(base_p: float, value: float, noise_model: NoiseModel) -> tuple[ChannelVector, float]:
    return channels_for(noise_model, value), value


def _axis_measurement_ratio(
    base_p: float, value: float, noise_model: NoiseModel
) -> tuple[ChannelVector, float]:
    if noise_model is NoiseModel.CODE_CAPACITY:
        raise ValueError(
            "measurement_ratio cannot be applied to CODE_CAPACITY: that model is "
            "*defined* by perfect measurements and a single round, so adding "
            "measurement noise silently turns it into a phenomenological model while "
            "the manifest still reads 'code_capacity'. Use PHENOMENOLOGICAL instead."
        )
    base = channels_for(noise_model, base_p)
    scaled = ChannelVector(
        after_clifford_depolarization=base.after_clifford_depolarization,
        after_reset_flip_probability=base.after_reset_flip_probability,
        before_measure_flip_probability=base_p * value,
        before_round_data_depolarization=base.before_round_data_depolarization,
    )
    return scaled, base_p


def _axis_xz_bias(
    base_p: float, value: float, noise_model: NoiseModel
) -> tuple[ChannelVector, float]:
    # Channels are unchanged; the asymmetry is introduced by rewriting the circuit,
    # because generated() has no parameter that can express it.
    return channels_for(noise_model, base_p), base_p


AXIS_BUILDERS: dict[
    DriftAxis, Callable[[float, float, NoiseModel], tuple[ChannelVector, float]]
] = {
    DriftAxis.P: _axis_p,
    DriftAxis.XZ_BIAS: _axis_xz_bias,
    DriftAxis.MEASUREMENT_RATIO: _axis_measurement_ratio,
}
"""Axis registry. Adding an axis is one function plus one entry here."""


def _concat_or_empty(parts: list[np.ndarray], width: int) -> np.ndarray:
    """Concatenate chunks, returning a correctly shaped empty array for zero shots.

    ``np.concatenate([])`` raises, so a zero-shot request used to crash even though the
    sampler handles it fine. An empty dataset is a legitimate degenerate case.
    """
    if parts:
        return np.concatenate(parts, axis=0)
    return np.zeros((0, width), dtype=np.uint8)


def _concat_mechanisms(
    parts: list[np.ndarray], emit_mechanisms: bool, n_mechanisms: int
) -> np.ndarray | None:
    """Concatenate mechanism chunks, honouring Contract B for zero shots.

    Zero shots with ``emit_mechanisms=True`` must still yield an (empty) array: a
    ``contract=dem_mechanism`` manifest beside ``mechanisms=None`` is a dataset that
    fails its own validator.
    """
    if parts:
        joined: np.ndarray = np.concatenate(parts, axis=0)
        return joined
    if emit_mechanisms:
        return np.zeros((0, packed_width(n_mechanisms)), dtype=np.uint8)
    return None


def _validate_axis_value(axis: DriftAxis, base_p: float, axis_value: float) -> None:
    """Reject non-finite or out-of-domain axis coordinates before any circuit is built.

    :class:`ChannelVector` catches bad *channel* values, but an axis value can be
    nonsense without ever reaching a channel — ``xz_bias`` is consumed by the circuit
    rewrite rather than by a channel, so a NaN there would slip past unnoticed.
    """
    if not math.isfinite(base_p):
        raise ValueError(f"base_p must be finite, got {base_p!r}")
    if not 0.0 <= base_p <= 1.0:
        raise ValueError(f"base_p must lie in [0, 1], got {base_p!r}")
    if not math.isfinite(axis_value):
        raise ValueError(f"axis value for {axis} must be finite, got {axis_value!r}")

    match axis:
        case DriftAxis.P:
            if not 0.0 <= axis_value <= 1.0:
                raise ValueError(f"axis value for {axis} is a probability, got {axis_value!r}")
        case DriftAxis.XZ_BIAS:
            if axis_value <= 0.0:
                raise ValueError(
                    f"xz_bias eta must be > 0 (0.5 is the unbiased point), got {axis_value!r}"
                )
        case DriftAxis.MEASUREMENT_RATIO:
            if axis_value < 0.0:
                raise ValueError(f"measurement_ratio must be >= 0, got {axis_value!r}")
        case _:
            # A new axis must state its domain here, or every value it takes is
            # accepted unchecked — the fail-open twin of the unbiased_point trap.
            assert_never(axis)


def derive_seeds(seed: int, count: int) -> list[int]:
    """Derive ``count`` independent child seeds from one master seed.

    Uses :class:`numpy.random.SeedSequence` spawning so the children are statistically
    independent and reproducible from the master seed alone. No global RNG state is
    touched anywhere in this package.
    """
    children = np.random.SeedSequence(seed).spawn(count)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def build_environment(
    environment_id: int,
    distance: int,
    base_p: float,
    axis: DriftAxis,
    axis_value: float,
    shots: int,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
) -> EnvironmentBuild:
    """Build one environment's circuit, DEM and spec."""
    try:
        builder = AXIS_BUILDERS[axis]
    except KeyError as exc:  # pragma: no cover - enum keeps this unreachable
        raise ValueError(f"unsupported drift axis {axis!r}") from exc

    _validate_axis_value(axis, base_p, axis_value)
    channels, effective_p = builder(base_p, axis_value, noise_model)
    resolved_rounds = default_rounds(noise_model, distance, rounds)
    circuit = build_circuit_from_channels(distance, channels, resolved_rounds, basis, rotated)

    if axis is DriftAxis.XZ_BIAS:
        circuit = apply_xz_bias(circuit, axis_value)

    dem = circuit.detector_error_model(decompose_errors=True)
    spec = EnvironmentSpec(
        environment_id=environment_id,
        p=effective_p,
        noise_model=noise_model,
        channels=channels,
        circuit=str(circuit),
        dem=str(dem),
        shots=shots,
        axis=str(axis),
        axis_value=axis_value,
    )
    return EnvironmentBuild(circuit=circuit, dem=dem, spec=spec)


def _assert_stackable(builds: Sequence[EnvironmentBuild]) -> tuple[int, int]:
    """Check every environment produces the same detector and observable widths.

    Environments differing in distance or rounds produce different ``n_detectors`` and
    cannot be stacked into one array. Failing loudly here beats silently producing a
    ragged or broadcast-mangled dataset.
    """
    detectors = {b.circuit.num_detectors for b in builds}
    observables = {b.circuit.num_observables for b in builds}
    if len(detectors) != 1 or len(observables) != 1:
        raise ValueError(
            "environments must share detector and observable counts to be stacked; "
            f"got n_detectors={sorted(detectors)}, n_observables={sorted(observables)}. "
            "Environments differing in distance or rounds must be written to separate files."
        )
    return detectors.pop(), observables.pop()


def _assert_mechanisms_stackable(builds: Sequence[EnvironmentBuild]) -> None:
    """Check every environment's DEM has the same mechanism count.

    Mechanism *index* is an artifact of DEM construction order and is only comparable
    across environments that share a topology. Environments built on the ``xz_bias``
    axis go through a different Stim code path (``PAULI_CHANNEL_1`` rather than
    ``DEPOLARIZE1``) and are not guaranteed to agree, so this is checked rather than
    assumed. Concatenating mismatched widths would either fail obscurely or, worse,
    line up columns that mean different things in different environments.
    """
    counts = {b.dem.num_errors for b in builds}
    if len(counts) != 1:
        raise ValueError(
            "cannot pool Contract B mechanism labels across environments with differing "
            f"mechanism counts {sorted(counts)}: column k would mean a different error "
            "mechanism in different environments. Drop --emit-mechanisms or write "
            "separate files."
        )

    # Matching counts are necessary but not sufficient: two DEMs can have the same
    # number of mechanisms in a different order, which would silently permute the
    # meaning of every label column. Compare the mechanism *topology* — the detector
    # and observable targets of each mechanism, ignoring probabilities, which are the
    # only thing that legitimately differs between environments.
    topologies = {_mechanism_topology(b.dem) for b in builds}
    if len(topologies) != 1:
        raise ValueError(
            "cannot pool Contract B mechanism labels across environments whose DEMs "
            "enumerate mechanisms differently: the label columns would not be "
            "comparable even though the counts match. Drop --emit-mechanisms or write "
            "separate files."
        )


def _assert_frozen_mechanisms_compatible(
    train_dem: stim.DetectorErrorModel, test_dem: stim.DetectorErrorModel
) -> None:
    """Refuse FROZEN_PRIOR + mechanisms when label columns would not line up.

    The labels are indexed against the *test* DEM while the shipped structure describes
    the *training* one. Matching counts are necessary but not sufficient — the same rule
    :func:`_assert_mechanisms_stackable` applies to pooling: equal counts in a different
    enumeration order would silently permute the meaning of every label column against
    the shipped training structure, with nothing in the file to detect it.
    """
    if train_dem.num_errors != test_dem.num_errors:
        raise ValueError(
            "cannot combine FROZEN_PRIOR with --emit-mechanisms when the "
            f"training DEM has {train_dem.num_errors} mechanisms and the "
            f"test DEM has {test_dem.num_errors}: the labels are indexed "
            "against the test DEM while the shipped structure describes the "
            "training one, so column k would denote different mechanisms in "
            "each. Drop --emit-mechanisms or use ORACLE_CALIBRATED."
        )
    if _mechanism_topology(train_dem) != _mechanism_topology(test_dem):
        raise ValueError(
            "cannot combine FROZEN_PRIOR with --emit-mechanisms when the "
            "training and test DEMs enumerate their mechanisms differently: the "
            "counts match but column k would still denote different mechanisms "
            "in each. Drop --emit-mechanisms or use ORACLE_CALIBRATED."
        )


def _mechanism_topology(dem: stim.DetectorErrorModel) -> str:
    """Digest of every mechanism's targets, ignoring probabilities.

    Probabilities are exactly what differs between environments; the target structure
    is what must agree for a mechanism index to mean the same thing in each.
    """
    parts: list[str] = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        targets = []
        for target in instruction.targets_copy():
            if target.is_separator():
                targets.append("^")
            elif target.is_relative_detector_id():
                targets.append(f"D{target.val}")
            else:
                targets.append(f"L{target.val}")
        parts.append(" ".join(targets))
    return dem_digest("\n".join(parts))


def build_multi_environment(
    distance: int,
    error_rates: Sequence[float],
    shots_per_env: int,
    seed: int,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    emit_mechanisms: bool = False,
    axis: DriftAxis = DriftAxis.P,
    base_p: float | None = None,
    shuffle: bool = True,
    structure_level: StructureLevel = StructureLevel.NONE,
    progress: Callable[[int], None] | None = None,
) -> InMemoryDataset:
    """Build a pooled dataset spanning several noise environments.

    Each rate gets its own circuit, DEM and :class:`EnvironmentSpec`. Shots carry an
    ``environment_id``. The pooled arrays are then shuffled with a seeded permutation so
    ordering carries no signal — without this a model could read the environment off the
    row index instead of the physics.

    The permutation is applied to **every** per-shot array with the same index vector,
    preserving correspondence between detectors, observables, environment ids and
    mechanism labels.

    Args:
        distance: Code distance, shared by all environments.
        error_rates: One value per environment, interpreted on ``axis``.
        shots_per_env: Shots drawn from each environment.
        seed: Master seed; per-environment and shuffle seeds are derived from it.
        axis: Which noise property ``error_rates`` varies.
        base_p: Base rate for non-``p`` axes. Required unless ``axis`` is ``P``.
        shuffle: Disable only for debugging; ordering then carries environment signal.
        progress: Called once per sampled chunk with that chunk's shot count, summing to
            ``shots_per_env * len(error_rates)``. Sampling only: the shuffle, hash and
            export that follow are not covered, and at these sizes they are not free.
            Raising from it cancels the run before anything is written.

    Returns:
        An :class:`InMemoryDataset` whose manifest lists every environment.
    """
    if not error_rates:
        raise ValueError("error_rates must contain at least one value")
    if axis is not DriftAxis.P and base_p is None:
        raise ValueError(f"axis={axis} requires base_p")
    resolved_base = base_p if base_p is not None else 0.0

    builds = [
        build_environment(
            environment_id=i,
            distance=distance,
            base_p=resolved_base if axis is not DriftAxis.P else value,
            axis=axis,
            axis_value=value,
            shots=shots_per_env,
            noise_model=noise_model,
            rounds=rounds,
            basis=basis,
            rotated=rotated,
        )
        for i, value in enumerate(error_rates)
    ]
    n_detectors, n_observables = _assert_stackable(builds)
    if emit_mechanisms:
        _assert_mechanisms_stackable(builds)

    env_seeds = derive_seeds(seed, len(builds) + 1)
    shuffle_seed = env_seeds[-1]

    det_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    mech_parts: list[np.ndarray] = []
    env_parts: list[np.ndarray] = []

    for build, env_seed in zip(builds, env_seeds[:-1], strict=True):
        for chunk in iter_chunks(
            build.circuit,
            shots_per_env,
            seed=env_seed,
            chunk_size=chunk_size,
            emit_mechanisms=emit_mechanisms,
            dem=build.dem if emit_mechanisms else None,
        ):
            det_parts.append(chunk.detectors)
            obs_parts.append(chunk.observables)
            env_parts.append(np.full(chunk.n_shots, build.spec.environment_id, dtype=np.int32))
            if chunk.mechanisms is not None:
                mech_parts.append(chunk.mechanisms)
            if progress is not None:
                progress(chunk.n_shots)

    detectors = _concat_or_empty(det_parts, packed_width(n_detectors))
    observables = _concat_or_empty(obs_parts, packed_width(n_observables))
    environment_ids = (
        np.concatenate(env_parts, axis=0) if env_parts else np.zeros(0, dtype=np.int32)
    )
    mechanisms = _concat_mechanisms(mech_parts, emit_mechanisms, builds[0].dem.num_errors)

    if shuffle:
        permutation = np.random.default_rng(shuffle_seed).permutation(detectors.shape[0])
        detectors = detectors[permutation]
        observables = observables[permutation]
        environment_ids = environment_ids[permutation]
        if mechanisms is not None:
            mechanisms = mechanisms[permutation]

    structure: DemStructure | None = None
    if structure_level in (StructureLevel.COORDS, StructureLevel.DEM, StructureLevel.FULL):
        structure = parse_dem(builds[0].dem, builds[0].circuit)

    resolved_rounds = default_rounds(noise_model, distance, rounds)
    meta = DatasetMeta(
        distance=distance,
        rounds=resolved_rounds,
        basis=basis,
        rotated=rotated,
        shots=int(detectors.shape[0]),
        seed=seed,
        chunk_size=chunk_size,
        n_detectors=n_detectors,
        n_observables=n_observables,
        environments=tuple(b.spec for b in builds),
        contract=Contract.DEM_MECHANISM if emit_mechanisms else Contract.LOGICAL_FRAME,
        structure_level=structure_level,
        drift_condition=DriftCondition.NOT_APPLICABLE,
        drift_axis=str(axis),
        structure_source_environment_id=0 if structure is not None else None,
        structure_dem_sha=dem_digest(builds[0].spec.dem) if structure is not None else None,
        bias_scope=XZ_BIAS_SCOPE if axis is DriftAxis.XZ_BIAS else None,
        n_mechanisms=builds[0].dem.num_errors if emit_mechanisms else None,
        mechanism_source_environment_id=0 if emit_mechanisms else None,
        shuffle_seed=shuffle_seed if shuffle else None,
        content_hash=content_hash(detectors, observables, environment_ids, mechanisms),
    )

    assert detectors.shape[1] == packed_width(n_detectors)
    return InMemoryDataset(
        detectors=detectors,
        observables=observables,
        meta=meta,
        environment_ids=environment_ids,
        mechanisms=mechanisms,
        structure=structure,
    )


def build_single_environment(
    distance: int,
    p: float,
    shots: int,
    seed: int,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    emit_mechanisms: bool = False,
    structure_level: StructureLevel = StructureLevel.NONE,
    progress: Callable[[int], None] | None = None,
) -> InMemoryDataset:
    """Build a one-environment dataset.

    The manifest still holds a *list* of environments, of length one. There is no
    top-level ``p`` even here, so single- and multi-environment files have exactly the
    same shape and downstream readers need no special case.

    ``progress`` is called once per sampled chunk with that chunk's shot count, matching
    :func:`stream_single_environment`. It covers *sampling only* — a caller driving a
    progress bar must not present a full bar as "done", because concatenation, hashing
    and export still follow. Raising from the callback cancels the run; nothing has been
    written at that point, so no partial file exists.
    """
    build = build_environment(
        environment_id=0,
        distance=distance,
        base_p=p,
        axis=DriftAxis.P,
        axis_value=p,
        shots=shots,
        noise_model=noise_model,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
    )
    structure = (
        parse_dem(build.dem, build.circuit) if structure_level is not StructureLevel.NONE else None
    )
    return _materialise(
        build=build,
        shots=shots,
        seed=seed,
        chunk_size=chunk_size,
        emit_mechanisms=emit_mechanisms,
        structure=structure,
        structure_level=structure_level,
        structure_dem_text=build.spec.dem,
        condition=DriftCondition.NOT_APPLICABLE,
        structure_source_environment_id=0,
        distance=distance,
        noise_model=noise_model,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
        axis=DriftAxis.P,
        progress=progress,
    )


def stream_single_environment(
    path: Path,
    distance: int,
    p: float,
    shots: int,
    seed: int,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    emit_mechanisms: bool = False,
    structure_level: StructureLevel = StructureLevel.NONE,
    progress: Callable[[int], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> DatasetMeta:
    """Generate a single-environment HDF5 file without materialising all shots.

    Sampling was always chunked, but the CLI concatenated every chunk before writing,
    so a large ``--shots`` still needed the whole dataset resident. This path writes
    each chunk straight to disk and holds only one chunk at a time.

    The content hash is accumulated incrementally over the same bytes the materialising
    path would hash, so both routes produce the same digest for the same seed and chunk
    size.

    Returns:
        The finalised manifest, already written into the file.
    """
    build = build_environment(
        environment_id=0,
        distance=distance,
        base_p=p,
        axis=DriftAxis.P,
        axis_value=p,
        shots=shots,
        noise_model=noise_model,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
    )
    structure = (
        parse_dem(build.dem, build.circuit) if structure_level is not StructureLevel.NONE else None
    )

    n_detectors = build.circuit.num_detectors
    n_observables = build.circuit.num_observables
    hasher = StreamingContentHasher(with_mechanisms=emit_mechanisms)

    writer = StreamingHDF5Writer(path, n_detectors, n_observables)
    written = 0
    try:
        for chunk in iter_chunks(
            build.circuit,
            shots,
            seed=seed,
            chunk_size=chunk_size,
            emit_mechanisms=emit_mechanisms,
            dem=build.dem if emit_mechanisms else None,
        ):
            writer.append(chunk.detectors, chunk.observables, None, chunk.mechanisms)
            hasher.update(chunk.detectors, chunk.observables, chunk.mechanisms)
            written += chunk.n_shots
            if progress is not None:
                progress(chunk.n_shots)

        if on_phase is not None:
            # Sampling is done; hash finalisation, structure and manifest remain.
            # Without this the streaming route never reported the "writing" phase
            # run.PhaseHook promises, and a front end showing a full sampling bar
            # looked hung during finalisation.
            on_phase("writing")

        meta = DatasetMeta(
            distance=distance,
            rounds=default_rounds(noise_model, distance, rounds),
            basis=basis,
            rotated=rotated,
            shots=written,
            seed=seed,
            chunk_size=chunk_size,
            n_detectors=n_detectors,
            n_observables=n_observables,
            environments=(build.spec,),
            contract=Contract.DEM_MECHANISM if emit_mechanisms else Contract.LOGICAL_FRAME,
            structure_level=structure_level,
            drift_condition=DriftCondition.NOT_APPLICABLE,
            drift_axis=str(DriftAxis.P),
            structure_source_environment_id=0 if structure is not None else None,
            structure_dem_sha=dem_digest(build.spec.dem) if structure is not None else None,
            n_mechanisms=build.dem.num_errors if emit_mechanisms else None,
            mechanism_source_environment_id=0 if emit_mechanisms else None,
            content_hash=hasher.hexdigest(
                written,
                n_detectors,
                n_observables,
                build.dem.num_errors if emit_mechanisms else None,
            ),
        )
        writer.close(meta, structure)
    except BaseException:
        writer.abort()
        raise
    return meta


def build_drift_environments(
    distance: int,
    train_p: float,
    test_values: Sequence[float],
    shots: int,
    seed: int,
    condition: DriftCondition,
    axis: DriftAxis = DriftAxis.P,
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
    rounds: int | None = None,
    basis: Basis = Basis.Z,
    rotated: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    emit_mechanisms: bool = False,
    structure_level: StructureLevel = StructureLevel.DEM,
    progress: Callable[[int], None] | None = None,
) -> list[InMemoryDataset]:
    """Build one training set plus one test set per drifted value.

    The condition decides which DEM the *test* files ship with:

    * :attr:`DriftCondition.ORACLE_CALIBRATED` — each test file gets structure derived
      from its own environment. The decoder receives the true test-time noise
      distribution, so this measures a ceiling, not generalisation.
    * :attr:`DriftCondition.FROZEN_PRIOR` — every test file gets structure derived from
      the **training** environment. The decoder must genuinely generalise.

    Both are legitimate experiments. Which one a file represents is recorded in its
    manifest, never inferred from which DEM happened to be in scope.

    ``progress`` is called once per sampled chunk across *all* files, so its increments
    sum to ``shots * (1 + len(test_values))`` rather than to one file's worth.

    Returns:
        ``[train, test_0, test_1, ...]``, each a self-describing dataset. Use
        :func:`drift_dataset_names` to name them; the names and the list are a pair.
    """
    if condition is DriftCondition.NOT_APPLICABLE:
        raise ValueError(
            "a drift study must state its condition explicitly: pass FROZEN_PRIOR or "
            "ORACLE_CALIBRATED. NOT_APPLICABLE previously fell through to oracle "
            "behaviour while stamping 'not_applicable' on the file, which is the exact "
            "silent mislabelling the condition exists to prevent."
        )

    train_axis_value = train_p if axis is DriftAxis.P else unbiased_point(axis)
    seeds = derive_seeds(seed, len(test_values) + 1)

    train_build = build_environment(
        environment_id=0,
        distance=distance,
        base_p=train_p,
        axis=axis,
        axis_value=train_axis_value,
        shots=shots,
        noise_model=noise_model,
        rounds=rounds,
        basis=basis,
        rotated=rotated,
    )
    # At structure_level NONE nothing downstream consumes a parsed DEM; parsing every
    # environment anyway costs a full sparse-matrix build per file just to discard it.
    frozen_structure = (
        parse_dem(train_build.dem, train_build.circuit)
        if structure_level is not StructureLevel.NONE
        else None
    )

    datasets = [
        _materialise(
            build=train_build,
            shots=shots,
            seed=seeds[0],
            chunk_size=chunk_size,
            emit_mechanisms=emit_mechanisms,
            structure=frozen_structure,
            structure_level=structure_level,
            structure_dem_text=train_build.spec.dem,
            # The training file is by definition calibrated to its own environment.
            condition=DriftCondition.ORACLE_CALIBRATED,
            structure_source_environment_id=0,
            distance=distance,
            noise_model=noise_model,
            rounds=rounds,
            basis=basis,
            rotated=rotated,
            axis=axis,
            progress=progress,
        )
    ]

    for index, value in enumerate(test_values):
        test_build = build_environment(
            environment_id=index + 1,
            distance=distance,
            base_p=value if axis is DriftAxis.P else train_p,
            axis=axis,
            axis_value=value,
            shots=shots,
            noise_model=noise_model,
            rounds=rounds,
            basis=basis,
            rotated=rotated,
        )
        if condition is DriftCondition.FROZEN_PRIOR:
            if emit_mechanisms:
                _assert_frozen_mechanisms_compatible(train_build.dem, test_build.dem)
            structure = frozen_structure
            source_env = 0
        else:
            structure = (
                parse_dem(test_build.dem, test_build.circuit)
                if structure_level is not StructureLevel.NONE
                else None
            )
            source_env = index + 1

        datasets.append(
            _materialise(
                build=test_build,
                shots=shots,
                seed=seeds[index + 1],
                chunk_size=chunk_size,
                emit_mechanisms=emit_mechanisms,
                structure=structure,
                structure_level=structure_level,
                structure_dem_text=(
                    train_build.spec.dem
                    if condition is DriftCondition.FROZEN_PRIOR
                    else test_build.spec.dem
                ),
                condition=condition,
                structure_source_environment_id=source_env,
                distance=distance,
                noise_model=noise_model,
                rounds=rounds,
                basis=basis,
                rotated=rotated,
                axis=axis,
                progress=progress,
            )
        )
    return datasets


def drift_dataset_names(test_values: Sequence[float]) -> list[str]:
    """Extension-free names for what :func:`build_drift_environments` returns, in order.

    Lives here rather than in a front end because it is a property of the returned
    dataset *set*: the names and the list are a pair, and a second front end must not be
    able to derive one without the other's check.

    Raises:
        ValueError: when two values print identically under ``%g``. ``0.01`` and
            ``0.010`` both render ``test_0.01``, so one file would silently overwrite
            another. Call this **before** building — the check is instant, and the run
            it guards is not.
    """
    names = ["train"] + [f"test_{value:g}" for value in test_values]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"test values collide into the same filename(s) {duplicates} under '%g' "
            "formatting; one file would silently overwrite another. Use distinct values "
            "that differ at the printed precision."
        )
    return names


def unbiased_point(axis: DriftAxis) -> float:
    """The axis value that reproduces the plain noise model.

    Exhaustive over the enum rather than a dict with a ``0.0`` default: a newly added
    axis that silently mapped to 0.0 would hand drift training files a nonsense
    coordinate (0.0 is out of domain for ``xz_bias``) instead of failing loudly at the
    extension point — the same reason ``channels_for`` ends in ``assert_never``.
    """
    match axis:
        case DriftAxis.P:
            raise ValueError("the P axis has no unbiased point: the axis value is the rate itself")
        case DriftAxis.XZ_BIAS:
            return 0.5
        case DriftAxis.MEASUREMENT_RATIO:
            return 1.0
        case _:
            assert_never(axis)


def _materialise(
    *,
    build: EnvironmentBuild,
    shots: int,
    seed: int,
    chunk_size: int,
    emit_mechanisms: bool,
    structure: DemStructure | None,
    structure_level: StructureLevel,
    structure_dem_text: str,
    condition: DriftCondition,
    structure_source_environment_id: int,
    distance: int,
    noise_model: NoiseModel,
    rounds: int | None,
    basis: Basis,
    rotated: bool,
    axis: DriftAxis,
    progress: Callable[[int], None] | None = None,
) -> InMemoryDataset:
    """Sample one environment fully into memory and wrap it with a manifest."""
    det_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    mech_parts: list[np.ndarray] = []
    for chunk in iter_chunks(
        build.circuit,
        shots,
        seed=seed,
        chunk_size=chunk_size,
        emit_mechanisms=emit_mechanisms,
        dem=build.dem if emit_mechanisms else None,
    ):
        det_parts.append(chunk.detectors)
        obs_parts.append(chunk.observables)
        if chunk.mechanisms is not None:
            mech_parts.append(chunk.mechanisms)
        if progress is not None:
            progress(chunk.n_shots)

    n_detectors = build.circuit.num_detectors
    n_observables = build.circuit.num_observables
    detectors = _concat_or_empty(det_parts, packed_width(n_detectors))
    observables = _concat_or_empty(obs_parts, packed_width(n_observables))
    mechanisms = _concat_mechanisms(mech_parts, emit_mechanisms, build.dem.num_errors)

    meta = DatasetMeta(
        distance=distance,
        rounds=default_rounds(noise_model, distance, rounds),
        basis=basis,
        rotated=rotated,
        shots=int(detectors.shape[0]),
        seed=seed,
        chunk_size=chunk_size,
        n_detectors=n_detectors,
        n_observables=n_observables,
        environments=(build.spec,),
        contract=Contract.DEM_MECHANISM if emit_mechanisms else Contract.LOGICAL_FRAME,
        structure_level=structure_level,
        drift_condition=condition,
        drift_axis=str(axis),
        # Only meaningful when structure is actually exported. Stamping it anyway made
        # identical runs produce different manifests depending on which code path
        # (materialised / streamed / multi-env) happened to build them.
        structure_source_environment_id=(
            structure_source_environment_id if structure is not None else None
        ),
        # Digest of the DEM the structure actually came from, which under FROZEN_PRIOR
        # is the training environment rather than this file's own.
        structure_dem_sha=dem_digest(structure_dem_text) if structure is not None else None,
        bias_scope=XZ_BIAS_SCOPE if axis is DriftAxis.XZ_BIAS else None,
        # From the DEM actually sampled, never from the exported structure: under
        # FROZEN_PRIOR those are different objects and the structure's count would
        # misdescribe the label array's width.
        n_mechanisms=build.dem.num_errors if emit_mechanisms else None,
        mechanism_source_environment_id=(build.spec.environment_id if emit_mechanisms else None),
        content_hash=content_hash(detectors, observables, None, mechanisms),
    )
    return InMemoryDataset(
        detectors=detectors,
        observables=observables,
        meta=meta,
        environment_ids=None,
        mechanisms=mechanisms,
        structure=structure,
    )
