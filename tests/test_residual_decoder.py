"""Frozen decoder models and chunk decoding for the residual datasets.

Every failure mode here is silent in the artifact: a big-endian unpack, a matcher built
from a DEM other than the published one, or an observable array quietly reduced to one
bit all produce a well-formed feature CSV whose ``pm_guess``/``truth`` columns mean
something else. The tests pin the rules (little-endian everywhere, the text is the model,
exactly one observable, refusal on every mismatch) rather than any circuit's numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pymatching
import pytest
import stim

from qecgen.noise import NoiseProfile, build_noisy_circuit
from qecgen.residual.config import DecoderKind
from qecgen.residual.decoder import (
    DecoderModel,
    MultiObservableError,
    decode_chunk,
    dem_stats,
    from_circuit,
    from_frozen_reference,
    from_official_dem,
    from_static_profile,
    frozen_reference_profile,
    pm_wrong,
    truth_from_packed,
)
from qecgen.sampling import unpack_bits

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _noise_block(name: str) -> dict[str, object]:
    payload = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    noise = payload["noise"]
    assert isinstance(noise, dict)
    return noise


def _ideal_d3() -> stim.Circuit:
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)


def _sample(circuit: stim.Circuit, shots: int, seed: int, *, packed: bool) -> np.ndarray:
    dets, _ = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True, bit_packed=packed
    )
    return np.asarray(dets)


class TestFromCircuit:
    def test_model_fields(self, d3_circuit: stim.Circuit) -> None:
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {"source": "test"})

        assert model.kind is DecoderKind.CIRCUIT_DEM
        assert model.n_detectors == d3_circuit.num_detectors == model.matching.num_detectors
        assert model.n_observables == 1
        assert model.enable_correlations is False
        assert model.dem_sha256 == hashlib.sha256(model.dem_text.encode("utf-8")).hexdigest()
        assert len(model.dem_blake2b128) == 32
        assert model.provenance["source"] == "test"
        assert model.provenance["method"] == "circuit.detector_error_model(decompose_errors=True)"
        assert model.provenance["circuit_sha256"] == (
            hashlib.sha256(str(d3_circuit).encode()).hexdigest()
        )
        assert model.provenance["dem_stats"]["num_errors"] == model.dem.num_errors

    def test_text_is_the_model(self, d3_circuit: stim.Circuit) -> None:
        """The published ``.dem`` text must rebuild *this* matcher, bit for bit.

        Stim prints rounded floats, so a DEM parsed from its own text is not equal to the
        in-memory object it was printed from. The model is therefore built from the text
        that will be published; rebuilding from that text must give identical weights.
        """
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})
        reparsed = stim.DetectorErrorModel(model.dem_text)

        assert str(reparsed) == model.dem_text
        assert reparsed == model.dem

        rebuilt = pymatching.Matching.from_detector_error_model(reparsed)
        packed = _sample(d3_circuit, 50, 7, packed=True)
        guess, weight = decode_chunk(model, packed)
        pred, weight_rebuilt = rebuilt.decode_batch(
            packed, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True
        )
        assert np.array_equal(guess, unpack_bits(pred, 1)[:, 0].astype(np.uint8))
        assert np.array_equal(weight, np.asarray(weight_rebuilt, dtype=np.float64))

    def test_batch_matches_single_shot(self, d3_circuit: stim.Circuit) -> None:
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})
        packed = _sample(d3_circuit, 50, 11, packed=True)
        bits = unpack_bits(packed, d3_circuit.num_detectors)

        guess, weight = decode_chunk(model, packed)

        assert guess.dtype == np.uint8 and guess.shape == (50,)
        assert weight.dtype == np.float64 and weight.shape == (50,)
        for row in range(50):
            single, single_weight = model.matching.decode(bits[row], return_weight=True)
            assert int(guess[row]) == int(np.asarray(single)[0])
            assert abs(float(weight[row]) - float(single_weight)) <= 1e-9

    def test_little_endian_negative_control(self, d3_circuit: stim.Circuit) -> None:
        """The house pattern from ``tests/test_sampling.py``: big-endian must disagree."""
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})
        n = d3_circuit.num_detectors
        packed = _sample(d3_circuit, 64, 23, packed=True)
        plain = _sample(d3_circuit, 64, 23, packed=False)

        guess, weight = decode_chunk(model, packed)
        pred, weight_plain = model.matching.decode_batch(plain, return_weights=True)

        assert np.array_equal(guess, np.asarray(pred)[:, 0].astype(np.uint8))
        assert np.allclose(weight, np.asarray(weight_plain, dtype=np.float64), atol=1e-9)
        big = np.unpackbits(packed, axis=1, count=n, bitorder="big").astype(bool)
        assert not np.array_equal(big, plain), "numpy's default would silently reverse bits"

    def test_empty_syndrome(self, d3_circuit: stim.Circuit) -> None:
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})
        packed = np.zeros((5, (d3_circuit.num_detectors + 7) // 8), dtype=np.uint8)

        guess, weight = decode_chunk(model, packed)

        assert np.array_equal(guess, np.zeros(5, dtype=np.uint8))
        assert np.array_equal(weight, np.zeros(5, dtype=np.float64))

    def test_zero_rows(self, d3_circuit: stim.Circuit) -> None:
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})
        packed = np.zeros((0, (d3_circuit.num_detectors + 7) // 8), dtype=np.uint8)

        guess, weight = decode_chunk(model, packed)

        assert guess.shape == (0,) and guess.dtype == np.uint8
        assert weight.shape == (0,) and weight.dtype == np.float64

    def test_wrong_width_raises(self, d3_circuit: stim.Circuit) -> None:
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})
        width = (d3_circuit.num_detectors + 7) // 8

        with pytest.raises(ValueError, match="packed"):
            decode_chunk(model, np.zeros((4, width + 1), dtype=np.uint8))
        with pytest.raises(ValueError, match="packed"):
            decode_chunk(model, np.zeros((4, width - 1), dtype=np.uint8))
        with pytest.raises(ValueError, match="packed"):
            decode_chunk(model, np.zeros(width, dtype=np.uint8))
        with pytest.raises(ValueError, match="uint8"):
            decode_chunk(model, np.zeros((4, width), dtype=np.int64))

    def test_two_observables_refused_with_schema_proposal(self, d3_circuit: stim.Circuit) -> None:
        """A second observable is stopped, never reduced to one bit (brief, Phase 0)."""
        circuit = d3_circuit.copy()
        last = circuit[len(circuit) - 1]
        assert isinstance(last, stim.CircuitInstruction)
        assert last.name == "OBSERVABLE_INCLUDE"
        circuit.append("OBSERVABLE_INCLUDE", last.targets_copy(), 1)
        assert circuit.num_observables == 2

        with pytest.raises(MultiObservableError) as info:
            from_circuit(circuit, DecoderKind.CIRCUIT_DEM, {})
        message = str(info.value)
        assert "truth_0" in message and "pm_guess_0" in message and "pm_wrong" in message

    def test_enable_correlations_refused(self, d3_circuit: stim.Circuit) -> None:
        model = from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {})

        with pytest.raises(ValueError, match="separate"):
            DecoderModel(
                kind=model.kind,
                dem=model.dem,
                dem_text=model.dem_text,
                dem_sha256=model.dem_sha256,
                dem_blake2b128=model.dem_blake2b128,
                matching=model.matching,
                n_detectors=model.n_detectors,
                n_observables=model.n_observables,
                circuit=model.circuit,
                provenance=model.provenance,
                enable_correlations=True,
            )

    def test_provenance_collision_raises(self, d3_circuit: stim.Circuit) -> None:
        with pytest.raises(ValueError, match="circuit_sha256"):
            from_circuit(d3_circuit, DecoderKind.CIRCUIT_DEM, {"circuit_sha256": "bogus"})


class TestLabels:
    def test_truth_from_packed(self, d3_circuit: stim.Circuit) -> None:
        _, obs_packed = d3_circuit.compile_detector_sampler(seed=3).sample(
            40, separate_observables=True, bit_packed=True
        )
        _, obs_plain = d3_circuit.compile_detector_sampler(seed=3).sample(
            40, separate_observables=True, bit_packed=False
        )

        truth = truth_from_packed(obs_packed, 1)

        assert truth.dtype == np.uint8 and truth.shape == (40,)
        assert np.array_equal(truth, np.asarray(obs_plain)[:, 0].astype(np.uint8))

    def test_truth_from_packed_multi_observable_raises(self) -> None:
        with pytest.raises(MultiObservableError, match="truth_0"):
            truth_from_packed(np.zeros((4, 1), dtype=np.uint8), 2)
        with pytest.raises(ValueError, match="packed"):
            truth_from_packed(np.zeros((4, 2), dtype=np.uint8), 1)

    def test_pm_wrong_truth_table(self) -> None:
        guess = np.array([0, 0, 1, 1], dtype=np.uint8)
        truth = np.array([0, 1, 0, 1], dtype=np.uint8)

        wrong = pm_wrong(guess, truth)

        assert wrong.dtype == np.uint8
        assert wrong.tolist() == [0, 1, 1, 0]

    def test_pm_wrong_refuses_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            pm_wrong(np.zeros(3, dtype=np.uint8), np.zeros(4, dtype=np.uint8))
        with pytest.raises(ValueError, match="binary"):
            pm_wrong(np.array([0, 2], dtype=np.uint8), np.zeros(2, dtype=np.uint8))
        with pytest.raises(ValueError, match="1-D"):
            pm_wrong(np.zeros((2, 1), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8))


class TestDemStats:
    def test_counts(self) -> None:
        dem = stim.DetectorErrorModel(
            """
            error(0.1) D0 D1 ^ D1 D2 L0
            error(0.1) D0 D1 D2
            error(0.1) D0
            error(0.1) D3 D4
            """
        )

        stats = dem_stats(dem)

        assert stats == {
            "num_errors": 4,
            "errors_with_separator": 1,
            "errors_gt2_detectors": 2,
            "errors_gt2_detectors_without_separator": 1,
            "components_gt2_detectors": 1,
        }


class TestOfficialDem:
    def test_round_trip_keeps_exact_bytes(self, d3_circuit: stim.Circuit) -> None:
        text = str(d3_circuit.detector_error_model(decompose_errors=True))
        raw = text.replace("\n", "\r\n").encode("utf-8")

        model = from_official_dem(raw, d3_circuit, {"member": "x/error_model.dem"})

        assert model.kind is DecoderKind.OFFICIAL_DEM
        assert model.dem_text.encode("utf-8") == raw
        assert model.dem_sha256 == hashlib.sha256(raw).hexdigest()
        assert model.provenance["member"] == "x/error_model.dem"
        assert model.provenance["fitted_in_this_pipeline"] is False
        assert model.circuit is d3_circuit
        guess, _ = decode_chunk(model, _sample(d3_circuit, 8, 1, packed=True))
        assert set(guess.tolist()) <= {0, 1}

    def test_mismatched_coordinates_raise(self, d3_circuit: stim.Circuit) -> None:
        raw = str(d3_circuit.detector_error_model(decompose_errors=True)).encode()
        shifted = stim.Circuit("SHIFT_COORDS(0, 0, 1)\n") + d3_circuit
        assert shifted.num_detectors == d3_circuit.num_detectors

        with pytest.raises(ValueError, match="coordinate"):
            from_official_dem(raw, shifted, {})

    def test_detector_count_mismatch_raises(self, d3_circuit: stim.Circuit) -> None:
        raw = str(d3_circuit.detector_error_model(decompose_errors=True)).encode()
        shorter = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=2,
            after_clifford_depolarization=0.01,
        )

        with pytest.raises(ValueError, match="detectors"):
            from_official_dem(raw, shorter, {})

    def test_two_observables_refused(self, d3_circuit: stim.Circuit) -> None:
        raw = b"error(0.1) D0 D1 L0\nerror(0.1) D1 L1\n"

        with pytest.raises(MultiObservableError):
            from_official_dem(raw, d3_circuit, {})


class TestStaticProfile:
    def test_builds_from_example(self) -> None:
        ideal = _ideal_d3()
        profile = NoiseProfile.from_dict(_noise_block("device-static.json"))

        model = from_static_profile(ideal, profile, {"config": "device-static"})

        assert model.kind is DecoderKind.STATIC_PROFILE_DEM
        noisy = build_noisy_circuit(ideal, profile)
        assert model.provenance["circuit_sha256"] == hashlib.sha256(str(noisy).encode()).hexdigest()
        assert model.provenance["dem_stats"]["errors_gt2_detectors_without_separator"] == 0
        assert model.provenance["decompose_errors"] is True
        assert model.provenance["config"] == "device-static"
        assert model.circuit == noisy
        assert model.n_detectors == ideal.num_detectors

    def test_refuses_dynamic_profile(self) -> None:
        profile = NoiseProfile.from_dict(_noise_block("device-dynamic.json"))

        with pytest.raises(ValueError, match="Dynamic profile has no exact static DEM"):
            from_static_profile(_ideal_d3(), profile, {})


class TestFrozenReference:
    def test_frozen_reference_profile(self) -> None:
        raw = _noise_block("device-dynamic.json")
        profile = NoiseProfile.from_dict(raw)
        assert profile.dynamic

        frozen, text = frozen_reference_profile(profile)

        assert frozen.dynamic is False
        config = frozen.to_dict()
        assert config["drift"] == {"enabled": False}
        assert config["bursts"] == {"enabled": False}
        assert config["leakage"] == {"enabled": False}
        expected_spatial = [
            {
                "qubits": [q],
                "paulis": "Z",
                "probability": 0.002,
                "basis": "scenario_assumption",
                "enabled": True,
            }
            for q in (1, 3)
        ]
        assert config["spatial"] == expected_spatial
        original = profile.to_dict()
        for key in (
            "probabilities",
            "qubit_overrides",
            "edge_overrides",
            "coherence",
            "covariates",
        ):
            assert config[key] == original[key]
        for block in ("drift", "bursts", "leakage"):
            assert block in text
        assert "baseline_probability" in text
        assert "stationary" in text

    def test_static_profile_is_refused(self) -> None:
        profile = NoiseProfile.from_dict(_noise_block("device-static.json"))

        with pytest.raises(ValueError, match="dynamic"):
            frozen_reference_profile(profile)
        with pytest.raises(ValueError, match="dynamic"):
            from_frozen_reference(_ideal_d3(), profile, {})

    def test_existing_spatial_terms_are_kept(self) -> None:
        raw = _noise_block("device-dynamic.json")
        raw["spatial"] = [
            {
                "qubits": [1, 3],
                "paulis": "XX",
                "probability": 0.0002,
                "basis": "scenario_assumption",
            }
        ]

        frozen, _ = frozen_reference_profile(NoiseProfile.from_dict(raw))

        spatial = frozen.to_dict()["spatial"]
        assert spatial[0]["qubits"] == [1, 3] and spatial[0]["paulis"] == "XX"
        assert [entry["qubits"] for entry in spatial[1:]] == [[1], [3]]

    def test_from_frozen_reference_model(self) -> None:
        ideal = _ideal_d3()
        profile = NoiseProfile.from_dict(_noise_block("device-dynamic.json"))

        model = from_frozen_reference(ideal, profile, {"config": "device-dynamic"})

        assert model.kind is DecoderKind.FROZEN_REFERENCE_DEM
        frozen, text = frozen_reference_profile(profile)
        assert model.provenance["transformation"] == text
        assert model.provenance["frozen_profile"] == frozen.to_dict()
        assert len(model.provenance["dynamic_profile_sha256"]) == 64
        assert len(model.provenance["frozen_profile_sha256"]) == 64
        assert (
            model.provenance["dynamic_profile_sha256"] != model.provenance["frozen_profile_sha256"]
        )
        assert model.provenance["exact_dem"] is False
        assert model.circuit == build_noisy_circuit(ideal, frozen)
        # The drift qubits' stationary Pauli terms are in the reference circuit (Stim
        # prints CORRELATED_ERROR as E).
        assert "E(0.002) Z1" in str(model.circuit)
        assert "E(0.002) Z3" in str(model.circuit)
        guess, weight = decode_chunk(model, _sample(model.circuit, 16, 5, packed=True))
        assert guess.shape == (16,) and weight.shape == (16,)
