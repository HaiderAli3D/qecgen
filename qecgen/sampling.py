"""Chunked, bounded-memory shot generation.

Circuit, DEM and experimental hidden-state samplers all produce **detection events**
and separate logical flips. Dynamic profiles use FlipSimulator and explicitly pack
the feature axis little-endian: its native packed output instead packs the shot axis.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim

from qecgen.noise import (
    ANNOTATIONS,
    RESETS,
    RNG_STREAM_NAMES,
    NoiseProfile,
    _static_segments,
    profile_audit,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ShotChunk",
    "iter_chunks",
    "iter_profile_chunks",
    "packed_width",
    "sample_profile",
    "unpack_bits",
]

DEFAULT_CHUNK_SIZE = 100_000
"""Default shots per sample call.

Chunk size is part of the reproducibility contract: it changes the sequence of
``sample()`` calls against one seeded sampler, and therefore changes the sample stream.
It is recorded in every manifest.
"""


def packed_width(n_bits: int) -> int:
    """Bytes needed to bit-pack ``n_bits``.

    Bit packing pads to a byte boundary, so this is lossy in the other direction: a
    3-byte array could hold anywhere from 17 to 24 detectors. The true width must be
    stored separately, which is why ``n_detectors`` and ``n_observables`` are explicit
    manifest fields.
    """
    return math.ceil(n_bits / 8)


def unpack_bits(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """Unpack a little-endian bit-packed array to bool.

    ``bitorder="little"`` is passed explicitly. NumPy's default is ``"big"``, which is
    the opposite of the convention Stim and PyMatching use; taking the default here
    silently reverses every byte's bits and produces well-formed, wrong data.

    The packed width is checked rather than trusted: ``np.unpackbits(count=...)``
    zero-fills past the end of an under-wide array instead of raising — and on a
    zero-width array returns uninitialized memory — so a truncated or mismatched input
    would otherwise unpack "successfully" into fabricated bits.
    """
    expected = packed_width(n_bits)
    if packed.ndim != 2 or packed.shape[1] != expected:
        raise ValueError(
            f"packed array has shape {packed.shape}; expected (rows, {expected}) for {n_bits} bits"
        )
    return np.unpackbits(packed, axis=1, count=n_bits, bitorder="little").astype(bool)


@dataclass(frozen=True, slots=True)
class ShotChunk:
    """One chunk of shots. Arrays correspond row-for-row."""

    detectors: np.ndarray
    """uint8, shape ``(n, packed_width(n_detectors))``, little-endian bit-packed."""

    observables: np.ndarray
    """uint8, shape ``(n, packed_width(n_observables))``, little-endian bit-packed."""

    mechanisms: np.ndarray | None
    """Contract B labels, or None. uint8, ``(n, packed_width(n_mechanisms))``.

    These are **abstract DEM mechanisms in the decomposed noise model**, not
    gate-level physical Pauli faults. See ``DATA_CONTRACT.md``.
    """

    @property
    def n_shots(self) -> int:
        """Number of shots in this chunk."""
        return int(self.detectors.shape[0])


def _chunk_sizes(shots: int, chunk_size: int) -> Iterator[int]:
    if shots < 0:
        raise ValueError(f"shots must be >= 0, got {shots}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    remaining = shots
    while remaining > 0:
        take = min(chunk_size, remaining)
        yield take
        remaining -= take


def iter_chunks(
    circuit: stim.Circuit,
    shots: int,
    seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    emit_mechanisms: bool = False,
    dem: stim.DetectorErrorModel | None = None,
) -> Iterator[ShotChunk]:
    """Yield chunks of shots, never materialising the full dataset.

    Two distinct sampling paths, because mixing them would break per-shot
    correspondence:

    * ``emit_mechanisms=False`` — detection events and observables come from
      ``circuit.compile_detector_sampler(seed=...)``.
    * ``emit_mechanisms=True`` — **all three** arrays come from
      ``dem.compile_sampler(seed=...)``, which returns detectors, observables and fired
      mechanisms from a single consistent draw. Sampling the circuit and the DEM
      separately would give two independent RNG streams, so the mechanism labels would
      not explain the detection events stored beside them. Sampling the DEM reproduces
      the same detector/observable distribution, so Contract A targets remain valid.

    Args:
        circuit: The circuit to sample.
        shots: Total shots to produce.
        seed: Explicit seed. No global RNG state is used anywhere.
        chunk_size: Shots per ``sample()`` call. Part of the reproducibility contract.
        emit_mechanisms: Also record which DEM mechanisms fired (Contract B).
        dem: Required when ``emit_mechanisms`` is True. Must be the decomposed DEM of
            ``circuit``.

    Yields:
        :class:`ShotChunk` objects whose arrays correspond row-for-row.
    """
    if not emit_mechanisms:
        sampler = circuit.compile_detector_sampler(seed=seed)
        for take in _chunk_sizes(shots, chunk_size):
            dets, obs = sampler.sample(take, separate_observables=True, bit_packed=True)
            yield ShotChunk(detectors=dets, observables=obs, mechanisms=None)
        return

    if dem is None:
        raise ValueError("emit_mechanisms=True requires the circuit's decomposed dem")

    dem_sampler = dem.compile_sampler(seed=seed)
    for take in _chunk_sizes(shots, chunk_size):
        dets, obs, errs = dem_sampler.sample(take, bit_packed=True, return_errors=True)
        if errs is None:  # pragma: no cover - defensive, stim returns errors when asked
            raise RuntimeError("dem sampler returned no error data despite return_errors=True")
        yield ShotChunk(detectors=dets, observables=obs, mechanisms=errs)


def _stream(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("ascii")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(words)))


def _mask_inject(
    sim: stim.FlipSimulator, pauli: str, qubits: list[int], selected: np.ndarray
) -> None:
    mask = np.zeros((sim.num_qubits, sim.batch_size), dtype=np.bool_)
    mask[qubits] = selected
    sim.broadcast_pauli_errors(pauli=pauli, mask=mask)


def iter_profile_chunks(
    ideal: stim.Circuit, profile: NoiseProfile, shots: int, seed: int, chunk_size: int = 100_000
) -> Iterator[ShotChunk]:
    """Preserve acquisition-state trajectories across chunks and ideal reference samples.

    Drift and burst state advance once per acquisition shot, independent of chunk boundaries.
    Leakage-effect state persists across circuit layers, and starts unoccupied each shot.
    That explicit fresh-shot assumption does not model incompletely reset inter-shot leakage.
    Output chunk size still affects RNG consumption by Stim and is part of reproducibility.
    """
    if type(shots) is not int or shots < 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("shots must be nonnegative and chunk_size positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = profile.to_dict()
    segments = _static_segments(ideal, profile)
    names = RNG_STREAM_NAMES
    rng = {name: _stream(seed, name) for name in names}
    drift_state, burst_state = 0.0, False
    noisy = sum(segments, stim.Circuit())
    static_seed = int(rng["stim"].integers(0, 2**63))
    if not profile.dynamic:
        yield from iter_chunks(noisy, shots, static_seed, chunk_size)
        return
    for start in range(0, shots, chunk_size):
        count = min(chunk_size, shots - start)
        drift = config["drift"]
        burst = config["bursts"]
        leakage = config["leakage"]
        drift_p = np.zeros(count)
        burst_active = np.zeros(count, dtype=np.bool_)
        # This loop follows acquisition order, not vectorized layer execution order.
        for shot in range(count):
            if drift.get("enabled", False):
                drift_state = drift["rho"] * drift_state + rng["drift_state"].normal(
                    0, drift["sigma_logit"]
                )
                baseline = drift["baseline_probability"]
                logit = math.log(baseline / (1 - baseline)) + drift_state
                drift_p[shot] = 1 / (1 + math.exp(-max(-700, min(700, logit))))
            if burst.get("enabled", False):
                if burst_state:
                    burst_state = not rng["burst_state"].random() < burst["recovery_probability"]
                else:
                    burst_state = rng["burst_state"].random() < burst["onset_probability"]
                burst_active[shot] = burst_state
        sim = stim.FlipSimulator(
            batch_size=count,
            num_qubits=ideal.num_qubits,
            seed=int(rng["stim"].integers(0, 2**63)),
        )
        leaked = np.zeros((ideal.num_qubits, count), dtype=np.bool_)
        for segment in segments:
            sim.do(segment)
            if not any(i.name not in ANNOTATIONS or i.name == "TICK" for i in segment):
                continue
            if drift.get("enabled", False):
                qs = drift["qubits"]
                selected = rng["drift_effect"].random((len(qs), count)) < drift_p
                _mask_inject(sim, drift["pauli"], qs, selected)
            if burst.get("enabled", False):
                qs = burst["qubits"]
                selected = (
                    rng["burst_effect"].random((len(qs), count)) < burst["effect_probability"]
                ) & burst_active
                _mask_inject(sim, burst["pauli"], qs, selected)
            if leakage.get("enabled", False):
                qs = leakage["qubits"]
                state = leaked[qs]
                state &= rng["leakage"].random(state.shape) >= leakage["recovery_probability"]
                state |= rng["leakage"].random(state.shape) < leakage["entry_probability"]
                reset_qubits = {
                    t.value for i in segment if i.name in RESETS for t in i.targets_copy()
                }
                for index, q in enumerate(qs):
                    if q in reset_qubits:
                        state[index] &= (
                            rng["leakage"].random(count) >= leakage["reset_removal_probability"]
                        )
                leaked[qs] = state
                # Independent X/Z Bernoulli masks model randomized Pauli effects, not qutrits.
                for pauli in ("X", "Z"):
                    selected = state & (
                        rng["leakage"].random(state.shape) < leakage["effect_probability"]
                    )
                    _mask_inject(sim, pauli, qs, selected)
                    for source, target in leakage.get("neighbor_edges", []):
                        selected = leaked[source] & (
                            rng["leakage"].random(count) < leakage["neighbor_effect_probability"]
                        )
                        _mask_inject(sim, pauli, [target], selected[np.newaxis, :])
        # FlipSimulator's packed=True packs the SHOT axis, unlike the dataset contract.
        det = np.packbits(sim.get_detector_flips().T, axis=1, bitorder="little")
        obs = np.packbits(sim.get_observable_flips().T, axis=1, bitorder="little")
        yield ShotChunk(det, obs, None)


@dataclass(frozen=True)
class ProfileSample:
    """Materialized convenience result; large exports should consume the iterator."""

    detectors: np.ndarray
    observables: np.ndarray
    audit: dict[str, Any]


def sample_profile(
    ideal: stim.Circuit,
    profile: NoiseProfile,
    shots: int,
    seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ProfileSample:
    """Use precisely the streaming call sequence so materialization cannot change draws."""
    chunks = list(iter_profile_chunks(ideal, profile, shots, seed, chunk_size))
    detectors = (
        np.concatenate([chunk.detectors for chunk in chunks])
        if chunks
        else np.empty((0, packed_width(ideal.num_detectors)), dtype=np.uint8)
    )
    observables = (
        np.concatenate([chunk.observables for chunk in chunks])
        if chunks
        else np.empty((0, packed_width(ideal.num_observables)), dtype=np.uint8)
    )
    return ProfileSample(
        detectors, observables, profile_audit(ideal, profile, seed, shots, chunk_size)
    )
