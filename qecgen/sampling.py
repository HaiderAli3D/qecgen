"""Chunked, bounded-memory shot generation.

All sampling goes through :meth:`stim.Circuit.compile_detector_sampler` — never
``compile_sampler`` — so what is stored is **detection events**, not raw measurement
records. Every sample call passes ``separate_observables=True`` so observables are never
mixed into the detector array, and ``bit_packed=True`` so packing is done by Stim in its
native little-endian order rather than by ``np.packbits`` (whose default is big-endian).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import stim

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ShotChunk",
    "iter_chunks",
    "packed_width",
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
