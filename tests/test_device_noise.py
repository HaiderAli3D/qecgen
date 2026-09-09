"""Production safeguards for channel placement, reference frames and temporal state."""

from itertools import islice
from typing import Any

import numpy as np
import pytest
import stim

from qecgen.noise import (
    NoiseProfile,
    build_noisy_circuit,
    coherence_probabilities,
    profile_audit,
)
from qecgen.sampling import iter_profile_chunks, sample_profile


def memory() -> stim.Circuit:
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)


@pytest.mark.parametrize("dynamic", [False, True])
def test_streaming_matches_materialization(dynamic: bool) -> None:
    profile = NoiseProfile.from_dict(dynamics()) if dynamic else NoiseProfile.uniform(0.02)
    chunks = list(iter_profile_chunks(memory(), profile, 133, 42, 19))
    whole = sample_profile(memory(), profile, 133, 42, 19)
    assert [chunk.n_shots for chunk in chunks] == [19] * 7
    np.testing.assert_array_equal(np.concatenate([c.detectors for c in chunks]), whole.detectors)
    np.testing.assert_array_equal(
        np.concatenate([c.observables for c in chunks]), whole.observables
    )
    assert all(chunk.mechanisms is None for chunk in chunks)


def test_large_requested_count_does_not_materialize_all_shots() -> None:
    chunks = iter_profile_chunks(memory(), NoiseProfile.from_dict(dynamics()), 10**12, 12, 19)
    assert [chunk.n_shots for chunk in islice(chunks, 2)] == [19, 19]


def test_burst_acquisition_state_persists_across_chunks() -> None:
    ideal = stim.Circuit("R 0\nTICK\nM 0\nDETECTOR rec[-1]")
    profile = NoiseProfile.from_dict(
        {
            "version": 1,
            "bursts": {
                "enabled": True,
                "qubits": [0],
                "pauli": "X",
                "onset_probability": 0.1,
                "recovery_probability": 0,
                "effect_probability": 1,
            },
        }
    )
    # Once the burst begins, it cannot recover. Restarting hidden state for each
    # output chunk creates impossible returns to zero after that first onset.
    chunks = list(iter_profile_chunks(ideal, profile, 200, 7, 1))
    bits = np.concatenate([chunk.detectors[:, 0] for chunk in chunks])
    assert bits.any()
    assert np.all(np.diff(bits.astype(int)) >= 0)


def test_provenance_is_available_without_sampling_or_hidden_final_state() -> None:
    profile = NoiseProfile.from_dict(dynamics())
    audit = profile_audit(memory(), profile, seed=3, shots=10**12, chunk_size=7)
    assert audit["profile"] == profile.to_dict()
    assert audit["backend"] == "batched_flip_simulator"
    assert "final_acquisition_state" not in audit
    assert audit["noise_schema_version"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"probabilities": []},
        {"qubit_overrides": {1: {}}},
        {"edge_overrides": []},
        {"spatial": {}},
        {"spatial": [{}]},
        {"spatial": [{"enabled": 1}]},
        {"coherence": {"enabled": False, "layer_durations_s": [float("nan")]}},
        {"coherence": {"enabled": False, "qubits": {"0": {"t1_s": 1}}}},
        {"drift": {"enabled": True}},
        {"drift": {"enabled": False, "rho": 2}},
        {"leakage": {"enabled": False, "neighbor_edges": 2}},
        {"covariates": {"transition_frequency_hz": {"01": 4e9}}},
        {"classical_control_policy": []},
        {"label": 17},
    ],
)
def test_malformed_or_disabled_invalid_config_fails_closed(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        NoiseProfile.from_dict({"version": 1, **bad})


def test_constructor_cannot_bypass_profile_validation() -> None:
    with pytest.raises(ValueError):
        NoiseProfile({"version": 999})


def test_present_qubits_do_not_make_an_unused_gate_edge_valid() -> None:
    ideal = stim.Circuit("R 0 1\nM 0 1\nDETECTOR rec[-1]")
    profile = NoiseProfile.from_dict({"version": 1, "edge_overrides": {"0,1": 0.1}})
    with pytest.raises(ValueError, match="unused gate edges"):
        build_noisy_circuit(ideal, profile)


def test_disabled_coherence_does_not_claim_timing_or_read_missing_durations() -> None:
    profile = NoiseProfile.from_dict(
        {"version": 1, "coherence": {"enabled": False, "round_end_layers": [2]}}
    )
    audit = profile_audit(memory(), profile, 1, 5, 5)
    assert audit["round_rates_hz"] == []
    assert audit["physical_duration_s"] is None


def test_spatial_toggle_keeps_configuration_but_removes_channel() -> None:
    ideal = stim.Circuit("R 0\nTICK\nM 0\nDETECTOR rec[-1]")
    raw = {
        "version": 1,
        "spatial": [
            {
                "enabled": False,
                "qubits": [0],
                "paulis": "X",
                "probability": 1,
                "basis": "scenario_assumption",
            }
        ],
    }
    assert build_noisy_circuit(ideal, NoiseProfile.from_dict(raw)) == ideal


def dynamics() -> dict[str, Any]:
    return {
        "version": 1,
        "probabilities": {"measurement": 0.001},
        "drift": {
            "enabled": True,
            "qubits": [1],
            "pauli": "X",
            "baseline_probability": 0.02,
            "rho": 0.99,
            "sigma_logit": 0.1,
        },
        "bursts": {
            "enabled": True,
            "qubits": [1, 3],
            "pauli": "X",
            "onset_probability": 0.1,
            "recovery_probability": 0.2,
            "effect_probability": 0.1,
        },
        "leakage": {
            "enabled": True,
            "qubits": [1],
            "entry_probability": 0.03,
            "recovery_probability": 0.1,
            "reset_removal_probability": 1,
            "effect_probability": 0.2,
            "neighbor_edges": [[1, 3]],
            "neighbor_effect_probability": 0.1,
        },
    }


def test_zero_profile_preserves_ideal_and_nonzero_reference_state() -> None:
    ideal = stim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    profile = NoiseProfile.uniform(0)
    assert build_noisy_circuit(ideal, profile) == ideal
    sample = sample_profile(ideal, profile, 33, seed=12, chunk_size=7)
    assert not sample.detectors.any()
    assert not sample.observables.any()


def test_injected_errors_do_not_disappear_into_reference() -> None:
    ideal = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    profile = NoiseProfile.from_dict({"version": 1, "probabilities": {"measurement": 1}})
    sample = sample_profile(ideal, profile, 17, seed=0)
    assert np.all(sample.detectors == 1)
    assert np.all(sample.observables == 1)


def test_per_qubit_readout_respects_each_header_target() -> None:
    ideal = stim.Circuit(
        "R 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    profile = NoiseProfile.from_dict({"version": 1, "qubit_overrides": {"1": {"measurement": 1}}})
    sample = sample_profile(ideal, profile, 9, seed=1)
    assert np.all(sample.detectors == 2)
    assert np.all(sample.observables == 1)


def test_readout_noise_does_not_flip_reused_postmeasurement_state() -> None:
    ideal = stim.Circuit("R 0\nM 0\nTICK\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
    profile = NoiseProfile.from_dict({"version": 1, "probabilities": {"measurement": 1}})
    assert np.all(sample_profile(ideal, profile, 13, seed=41).detectors == 3)


def test_readout_preserves_inverted_measurement_target() -> None:
    ideal = stim.Circuit("R 0\nM !0\nDETECTOR rec[-1]")
    noisy = build_noisy_circuit(ideal, NoiseProfile.uniform(0))
    assert noisy == ideal


def test_two_qubit_override_and_ideal_structure() -> None:
    ideal = memory()
    profile = NoiseProfile.from_dict({"version": 1, "edge_overrides": {"1,2": 0.07}})
    noisy = build_noisy_circuit(ideal, profile)
    assert noisy.without_noise() == ideal.flattened()
    assert "DEPOLARIZE2(0.07)" in str(noisy)


def test_basis_correct_measurement_and_reset_channels() -> None:
    for reset, measurement in [("R", "M"), ("RX", "MX"), ("RY", "MY")]:
        circuit = stim.Circuit(f"{reset} 0\n{measurement} 0\nDETECTOR rec[-1]")
        profile = NoiseProfile.from_dict({"version": 1, "probabilities": {"reset": 1}})
        assert np.all(sample_profile(circuit, profile, 10, seed=0).detectors == 1)


@pytest.mark.parametrize(
    "field,value", [("measurement", -0.1), ("idle", float("nan")), ("reset", True)]
)
def test_invalid_probability_fails(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        NoiseProfile.from_dict({"version": 1, "probabilities": {field: value}})


def test_unknown_physics_couplings_fail_instead_of_inventing_effects() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        NoiseProfile.from_dict({"version": 1, "humidity_error_coefficient": 0.2})
    first = NoiseProfile.uniform(0.01)
    raw = first.to_dict()
    raw["covariates"] = {
        "transition_frequency_hz": {"1": 5e9},
        "cryostat_temperature_k": 0.015,
        "effective_qubit_temperature_k": 0.03,
        "humidity_relative_fraction": 0.6,
    }
    a, b = (sample_profile(memory(), p, 100, seed=3) for p in (first, NoiseProfile.from_dict(raw)))
    assert np.array_equal(a.detectors, b.detectors)
    assert np.array_equal(a.observables, b.observables)
    assert "recorded only" in b.audit["covariate_response"]


def test_profile_rejects_noise_and_unsupported_record_controls() -> None:
    with pytest.raises(ValueError, match="ideal"):
        build_noisy_circuit(stim.Circuit("X_ERROR(0.1) 0"), NoiseProfile.uniform(0))
    with pytest.raises(ValueError, match="Unsupported"):
        build_noisy_circuit(stim.Circuit("M 0\nCX rec[-1] 1"), NoiseProfile.uniform(0.1))


def test_unused_qubit_is_not_silently_ignored() -> None:
    profile = NoiseProfile.from_dict({"version": 1, "qubit_overrides": {"0": {"measurement": 0.1}}})
    with pytest.raises(ValueError, match="unused"):
        build_noisy_circuit(memory(), profile)


def test_explicit_ideal_sweep_frame_preserves_controls_and_ordinary_gates() -> None:
    ideal = stim.Circuit("R 0 1\nCX sweep[0] 0 0 1\nM 0 1\nDETECTOR rec[-1]")
    raw = NoiseProfile.uniform(0.1).to_dict()
    raw["classical_control_policy"] = "ideal_pauli_frame"
    profile = NoiseProfile.from_dict(raw)
    noisy = build_noisy_circuit(ideal, profile)
    assert noisy.without_noise() == ideal
    assert "CX sweep[0] 0 0 1" in str(noisy)
    assert "DEPOLARIZE2(0.1) 0 1" in str(noisy)
    assert sample_profile(ideal, profile, 17, seed=0).detectors.shape == (17, 1)


def test_ideal_sweep_policy_still_refuses_record_controls() -> None:
    profile = NoiseProfile.from_dict(
        {"version": 1, "classical_control_policy": "ideal_pauli_frame"}
    )
    with pytest.raises(ValueError, match="Unsupported"):
        build_noisy_circuit(stim.Circuit("M 0\nCX rec[-1] 1"), profile)


def test_overlapping_pair_instruction_does_not_move_noise_past_a_later_gate() -> None:
    with pytest.raises(ValueError, match="Overlapping"):
        build_noisy_circuit(stim.Circuit("R 0 1 2\nCX 0 1 1 2"), NoiseProfile.uniform(0.1))


def test_coherence_complete_positive_and_rejects_incompatible_times() -> None:
    assert coherence_probabilities(0, 1, 1) == (0, 0, 0)
    p = coherence_probabilities(0.1, 1, 2)
    assert all(value >= 0 for value in p)
    assert sum(p) <= 1
    with pytest.raises(ValueError, match="T2"):
        coherence_probabilities(0.1, 1, 3)


def test_coherence_requires_timing_and_excludes_double_counting() -> None:
    profile: dict[str, Any] = {
        "version": 1,
        "probabilities": {"one_qubit_gate": 0.01},
        "coherence": {
            "enabled": True,
            "t2_protocol": "exponential_ramsey",
            "qubits": {"0": {"t1_s": 1, "t2_s": 1}},
            "layer_durations_s": [0.1],
        },
    }
    with pytest.raises(ValueError, match="residual"):
        NoiseProfile.from_dict(profile)
    profile["coherence"]["residual_gate_errors_exclude_decoherence"] = True
    model = NoiseProfile.from_dict(profile)
    noisy = build_noisy_circuit(stim.Circuit("R 0"), model)
    assert "PAULI_CHANNEL_1" in str(noisy)
    with pytest.raises(ValueError, match="duration"):
        build_noisy_circuit(stim.Circuit("R 0\nTICK\nM 0"), model)


def test_final_measurement_layer_coherence_is_not_discarded() -> None:
    ideal = stim.Circuit("R 0\nX 0\nTICK\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    profile = NoiseProfile.from_dict(
        {
            "version": 1,
            "coherence": {
                "enabled": True,
                "t2_protocol": "exponential_ramsey",
                "qubits": {"0": {"t1_s": 1, "t2_s": 1}},
                "layer_durations_s": [0, 100],
            },
        }
    )
    sample = sample_profile(ideal, profile, 1000, seed=191)
    assert 0.4 < sample.observables.mean() < 0.6
    noisy = str(build_noisy_circuit(ideal, profile))
    assert noisy.index("PAULI_CHANNEL_1") < noisy.index("M 0")


def test_per_qubit_override_also_requires_residual_error_acknowledgement() -> None:
    with pytest.raises(ValueError, match="residual"):
        NoiseProfile.from_dict(
            {
                "version": 1,
                "qubit_overrides": {"0": {"idle": 0.01}},
                "coherence": {"enabled": True, "t2_protocol": "exponential_ramsey"},
            }
        )


def test_static_and_dynamic_reproducibility_and_contract() -> None:
    for profile in (NoiseProfile.uniform(0.02), NoiseProfile.from_dict(dynamics())):
        a, b = (sample_profile(memory(), profile, 37, seed=812, chunk_size=16) for _ in range(2))
        assert np.array_equal(a.detectors, b.detectors)
        assert np.array_equal(a.observables, b.observables)
        assert a.audit == b.audit
        assert a.detectors.shape == (37, 3)
        assert a.observables.shape == (37, 1)
        assert a.detectors.dtype == np.uint8
        assert a.audit["contract"] == "A"
        assert not a.audit["contains_mechanism_targets"]


def test_dynamic_state_does_not_restart_at_chunk_boundaries() -> None:
    profile = NoiseProfile.from_dict(dynamics())
    a = sample_profile(memory(), profile, 113, seed=10, chunk_size=8)
    b = sample_profile(memory(), profile, 113, seed=10, chunk_size=57)
    assert a.audit["state_scope"] == b.audit["state_scope"]
    assert a.audit["state_scope"]["leakage"] == "circuit_layers_reset_each_shot"
    with pytest.raises(ValueError, match="Dynamic"):
        build_noisy_circuit(memory(), profile)


def test_burst_effect_is_injected_and_not_absorbed_into_reference() -> None:
    ideal = stim.Circuit("R 0\nX 0\nTICK\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    config: dict[str, Any] = {
        "version": 1,
        "bursts": {
            "enabled": True,
            "qubits": [0],
            "pauli": "X",
            "onset_probability": 1,
            "recovery_probability": 0,
            "effect_probability": 1,
        },
    }
    sample = sample_profile(ideal, NoiseProfile.from_dict(config), 13, seed=5, chunk_size=3)
    assert np.all(sample.detectors == 1)
    assert np.all(sample.observables == 1)
    config["bursts"]["effect_probability"] = 0
    sample = sample_profile(ideal, NoiseProfile.from_dict(config), 13, seed=5, chunk_size=3)
    assert not sample.detectors.any()
    assert not sample.observables.any()


def test_dynamic_effects_do_not_skip_empty_idle_ticks() -> None:
    ideal = stim.Circuit("R 0\nTICK\nTICK\nM 0\nDETECTOR rec[-1]")
    profile = NoiseProfile.from_dict(
        {
            "version": 1,
            "bursts": {
                "enabled": True,
                "qubits": [0],
                "pauli": "X",
                "onset_probability": 1,
                "recovery_probability": 0,
                "effect_probability": 1,
            },
        }
    )
    # One X after reset plus one X in the pure idle layer cancel before measurement.
    assert not sample_profile(ideal, profile, 7, seed=1).detectors.any()


def test_leakage_neighbor_requires_occupied_source() -> None:
    profile = dynamics()
    profile["leakage"]["neighbor_edges"] = [[3, 1]]
    with pytest.raises(ValueError, match="source"):
        NoiseProfile.from_dict(profile)


def test_spatial_event_is_shared_not_two_independent_draws() -> None:
    circuit = stim.Circuit("R 0 1\nTICK\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
    profile = NoiseProfile.from_dict(
        {
            "version": 1,
            "spatial": [
                {
                    "qubits": [0, 1],
                    "paulis": "XX",
                    "probability": 0.2,
                    "basis": "scenario_assumption",
                }
            ],
        }
    )
    values = sample_profile(circuit, profile, 1000, seed=51).detectors[:, 0]
    assert set(values) == {0, 3}


def test_zero_shots_preserves_true_width_metadata() -> None:
    result = sample_profile(memory(), NoiseProfile.uniform(0), 0, seed=1)
    assert result.detectors.shape == (0, 3)
    assert result.audit["n_detectors"] == 24


def test_profile_round_trip_does_not_share_mutable_input() -> None:
    config = dynamics()
    profile = NoiseProfile.from_dict(config)
    saved = profile.to_dict()
    config["drift"]["rho"] = 0
    saved["drift"]["rho"] = 0
    assert profile.to_dict()["drift"]["rho"] == 0.99
