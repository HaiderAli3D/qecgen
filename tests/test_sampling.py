"""Sampling: bit order, packed widths, determinism and bounded memory."""

from __future__ import annotations

import numpy as np
import pymatching
import pytest
import stim

from qecgen.dem import DemStructure
from qecgen.sampling import DEFAULT_CHUNK_SIZE, iter_chunks, packed_width, unpack_bits


def test_packed_width_pads_to_byte_boundary() -> None:
    assert packed_width(1) == 1
    assert packed_width(8) == 1
    assert packed_width(9) == 2
    assert packed_width(24) == 3


def test_packed_width_does_not_determine_true_width() -> None:
    """3 bytes could mean 17..24 detectors, which is why n_detectors is explicit."""
    assert len({packed_width(n) for n in range(17, 25)}) == 1


class TestBitOrder:
    """The single most dangerous silent failure in this package."""

    def test_stim_packing_is_little_endian(self, d3_circuit: stim.Circuit) -> None:
        n = d3_circuit.num_detectors
        packed = d3_circuit.compile_detector_sampler(seed=99).sample(
            32, separate_observables=True, bit_packed=True
        )[0]
        plain = d3_circuit.compile_detector_sampler(seed=99).sample(
            32, separate_observables=True, bit_packed=False
        )[0]

        little = np.unpackbits(packed, axis=1, count=n, bitorder="little").astype(bool)
        big = np.unpackbits(packed, axis=1, count=n, bitorder="big").astype(bool)

        assert np.array_equal(little, plain)
        assert not np.array_equal(big, plain), "numpy's default would silently reverse bits"

    def test_unpack_bits_helper_uses_little(self, d3_circuit: stim.Circuit) -> None:
        n = d3_circuit.num_detectors
        packed, _ = d3_circuit.compile_detector_sampler(seed=5).sample(
            16, separate_observables=True, bit_packed=True
        )
        plain, _ = d3_circuit.compile_detector_sampler(seed=5).sample(
            16, separate_observables=True, bit_packed=False
        )
        assert np.array_equal(unpack_bits(packed, n), plain)

    def test_roundtrip_through_pymatching_agrees_packed_and_unpacked(
        self, d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel
    ) -> None:
        """Known-good case: the two paths must give identical predictions."""
        matching = pymatching.Matching.from_detector_error_model(d3_dem)
        n_obs = d3_circuit.num_observables
        n_det = d3_circuit.num_detectors

        dets, _ = d3_circuit.compile_detector_sampler(seed=17).sample(
            256, separate_observables=True, bit_packed=True
        )
        packed_predictions = matching.decode_batch(
            dets, bit_packed_shots=True, bit_packed_predictions=True
        )
        plain_predictions = matching.decode_batch(unpack_bits(dets, n_det))

        assert np.array_equal(
            unpack_bits(packed_predictions, n_obs), plain_predictions.astype(bool)
        )

    def test_repack_roundtrip(self, d3_circuit: stim.Circuit) -> None:
        n = d3_circuit.num_detectors
        packed, _ = d3_circuit.compile_detector_sampler(seed=3).sample(
            64, separate_observables=True, bit_packed=True
        )
        repacked = np.packbits(unpack_bits(packed, n), axis=1, bitorder="little")
        assert np.array_equal(repacked, packed)


class TestChunking:
    def test_chunks_sum_to_requested_shots(self, d3_circuit: stim.Circuit) -> None:
        chunks = list(iter_chunks(d3_circuit, 1000, seed=1, chunk_size=256))
        assert sum(c.n_shots for c in chunks) == 1000
        assert [c.n_shots for c in chunks] == [256, 256, 256, 232]

    def test_no_chunk_exceeds_chunk_size(self, d3_circuit: stim.Circuit) -> None:
        assert all(c.n_shots <= 128 for c in iter_chunks(d3_circuit, 500, seed=1, chunk_size=128))

    def test_zero_shots_yields_nothing(self, d3_circuit: stim.Circuit) -> None:
        assert list(iter_chunks(d3_circuit, 0, seed=1)) == []

    def test_rejects_bad_chunk_size(self, d3_circuit: stim.Circuit) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            list(iter_chunks(d3_circuit, 10, seed=1, chunk_size=0))

    def test_widths_match_manifest_fields(self, d3_circuit: stim.Circuit) -> None:
        chunk = next(iter_chunks(d3_circuit, 10, seed=1, chunk_size=10))
        assert chunk.detectors.shape[1] == packed_width(d3_circuit.num_detectors)
        assert chunk.observables.shape[1] == packed_width(d3_circuit.num_observables)
        assert chunk.detectors.dtype == np.uint8


class TestDeterminism:
    def test_same_seed_and_chunk_size_reproduces(self, d3_circuit: stim.Circuit) -> None:
        a = np.concatenate(
            [c.detectors for c in iter_chunks(d3_circuit, 500, seed=42, chunk_size=100)]
        )
        b = np.concatenate(
            [c.detectors for c in iter_chunks(d3_circuit, 500, seed=42, chunk_size=100)]
        )
        assert np.array_equal(a, b)

    def test_different_seed_differs(self, d3_circuit: stim.Circuit) -> None:
        a = next(iter_chunks(d3_circuit, 200, seed=1, chunk_size=200)).detectors
        b = next(iter_chunks(d3_circuit, 200, seed=2, chunk_size=200)).detectors
        assert not np.array_equal(a, b)

    def test_chunk_size_is_part_of_the_contract(self, d3_circuit: stim.Circuit) -> None:
        """Documented, not a bug: changing chunk size changes the sample stream."""
        a = np.concatenate(
            [c.detectors for c in iter_chunks(d3_circuit, 400, seed=7, chunk_size=400)]
        )
        b = np.concatenate(
            [c.detectors for c in iter_chunks(d3_circuit, 400, seed=7, chunk_size=100)]
        )
        assert a.shape == b.shape
        # Not asserted equal; this documents why chunk_size lives in the manifest.


class TestContractB:
    def test_mechanisms_require_a_dem(self, d3_circuit: stim.Circuit) -> None:
        with pytest.raises(ValueError, match="requires the circuit's decomposed dem"):
            list(iter_chunks(d3_circuit, 10, seed=1, emit_mechanisms=True))

    def test_mechanism_width_matches_mechanism_count(
        self, d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel
    ) -> None:
        chunk = next(
            iter_chunks(d3_circuit, 32, seed=1, chunk_size=32, emit_mechanisms=True, dem=d3_dem)
        )
        assert chunk.mechanisms is not None
        n_mechanisms = sum(1 for i in d3_dem.flattened() if i.type == "error")
        assert chunk.mechanisms.shape[1] == packed_width(n_mechanisms)

    def test_arrays_correspond_row_for_row(
        self, d3_circuit: stim.Circuit, d3_dem: stim.DetectorErrorModel
    ) -> None:
        """All three arrays come from one DEM draw, so rows must align."""
        chunk = next(
            iter_chunks(d3_circuit, 64, seed=2, chunk_size=64, emit_mechanisms=True, dem=d3_dem)
        )
        assert chunk.mechanisms is not None
        assert chunk.detectors.shape[0] == 64
        assert chunk.observables.shape[0] == 64
        assert chunk.mechanisms.shape[0] == 64

    def test_mechanism_labels_explain_the_events_beside_them(
        self,
        d3_circuit: stim.Circuit,
        d3_dem: stim.DetectorErrorModel,
        d3_structure: DemStructure,
    ) -> None:
        """The invariant behind --emit-mechanisms, asserted rather than described.

        All three arrays must come from ONE DEM-sampler draw, so the fired mechanisms
        must *reproduce* the events beside them: ``H @ m == detectors`` and
        ``L @ m == observables`` (mod 2), row for row. Sampling the circuit and the DEM
        separately — two RNG streams — fails this on the first chunk, while the
        row-count check above cannot see it.
        """
        h = d3_structure.h.toarray().astype(np.int64)
        ell = d3_structure.l.toarray().astype(np.int64)
        for chunk in iter_chunks(
            d3_circuit, 400, seed=13, chunk_size=150, emit_mechanisms=True, dem=d3_dem
        ):
            assert chunk.mechanisms is not None
            mechs = unpack_bits(chunk.mechanisms, d3_structure.n_mechanisms).astype(np.int64)
            detectors = unpack_bits(chunk.detectors, d3_structure.n_detectors)
            observables = unpack_bits(chunk.observables, d3_structure.n_observables)
            assert np.array_equal((mechs @ h.T) % 2 == 1, detectors)
            assert np.array_equal((mechs @ ell.T) % 2 == 1, observables)


def test_default_chunk_size_is_100k() -> None:
    assert DEFAULT_CHUNK_SIZE == 100_000
