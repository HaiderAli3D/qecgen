"""Circuit construction and noise model conventions."""

from __future__ import annotations

import math

import pytest
import stim

from qecgen.circuits import (
    Basis,
    ChannelVector,
    NoiseModel,
    apply_xz_bias,
    build_circuit,
    channels_for,
    code_task,
)


def test_code_capacity_sets_only_data_depolarization() -> None:
    _, channels = build_circuit(3, 0.01, NoiseModel.CODE_CAPACITY)
    assert channels.before_round_data_depolarization == 0.01
    assert channels.after_clifford_depolarization == 0.0
    assert channels.after_reset_flip_probability == 0.0
    assert channels.before_measure_flip_probability == 0.0


def test_phenomenological_sets_data_and_measurement_only() -> None:
    _, channels = build_circuit(3, 0.01, NoiseModel.PHENOMENOLOGICAL)
    assert channels.before_round_data_depolarization == 0.01
    assert channels.before_measure_flip_probability == 0.01
    assert channels.after_clifford_depolarization == 0.0
    assert channels.after_reset_flip_probability == 0.0


def test_stim_uniform_sets_all_four_channels() -> None:
    _, channels = build_circuit(3, 0.01, NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL)
    assert set(channels.as_dict().values()) == {0.01}


def test_code_capacity_forces_one_round() -> None:
    circuit, _ = build_circuit(3, 0.01, NoiseModel.CODE_CAPACITY)
    dem = circuit.detector_error_model(decompose_errors=True)
    # d=3 rotated memory with a single round has 8 detectors, not 24.
    assert dem.num_detectors == 8


def test_code_capacity_rejects_conflicting_rounds() -> None:
    """Silently overriding rounds would make the manifest disagree with the file."""
    with pytest.raises(ValueError, match="rounds=1"):
        build_circuit(3, 0.01, NoiseModel.CODE_CAPACITY, rounds=5)


def test_code_capacity_accepts_explicit_rounds_one() -> None:
    circuit, _ = build_circuit(3, 0.01, NoiseModel.CODE_CAPACITY, rounds=1)
    assert circuit.num_detectors == 8


def test_rounds_defaults_to_distance() -> None:
    circuit, _ = build_circuit(5, 0.001)
    reference = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.001,
        before_measure_flip_probability=0.001,
        before_round_data_depolarization=0.001,
    )
    assert circuit.num_detectors == reference.num_detectors


@pytest.mark.parametrize("basis", [Basis.Z, Basis.X])
@pytest.mark.parametrize("rotated", [True, False])
def test_all_layout_combinations_build_and_decompose(basis: Basis, rotated: bool) -> None:
    circuit, _ = build_circuit(3, 0.005, basis=basis, rotated=rotated)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors > 0
    assert dem.num_observables == 1


def test_code_task_strings() -> None:
    assert code_task(Basis.Z, True) == "surface_code:rotated_memory_z"
    assert code_task(Basis.X, False) == "surface_code:unrotated_memory_x"


def test_invalid_p_rejected() -> None:
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        build_circuit(3, 1.5)


def test_channel_vector_is_noiseless() -> None:
    assert ChannelVector().is_noiseless
    assert not channels_for(NoiseModel.CODE_CAPACITY, 0.01).is_noiseless
    assert channels_for(NoiseModel.CODE_CAPACITY, 0.0).is_noiseless


def test_zero_noise_produces_no_detection_events() -> None:
    circuit, channels = build_circuit(3, 0.0)
    assert channels.is_noiseless
    sampler = circuit.compile_detector_sampler(seed=1)
    dets, obs = sampler.sample(256, separate_observables=True, bit_packed=True)
    assert not dets.any()
    assert not obs.any()


class TestXZBias:
    """The bias axis requires rewriting the circuit; generated() cannot express it."""

    def test_replaces_depolarize1_with_pauli_channel_1(self) -> None:
        circuit, _ = build_circuit(3, 0.01)
        assert "DEPOLARIZE1" in str(circuit)
        biased = apply_xz_bias(circuit, 10.0)
        assert "DEPOLARIZE1" not in str(biased)
        assert "PAULI_CHANNEL_1" in str(biased)

    def test_preserves_repeat_blocks(self) -> None:
        circuit, _ = build_circuit(5, 0.01)
        assert any(isinstance(i, stim.CircuitRepeatBlock) for i in circuit)
        biased = apply_xz_bias(circuit, 3.0)
        assert any(isinstance(i, stim.CircuitRepeatBlock) for i in biased)

    def test_leaves_two_qubit_noise_symmetric(self) -> None:
        """Documented limitation: DEPOLARIZE2 is not biased."""
        circuit, _ = build_circuit(3, 0.01)
        biased = apply_xz_bias(circuit, 10.0)
        assert "DEPOLARIZE2" in str(biased)

    def test_eta_half_is_the_depolarizing_point(self) -> None:
        """px = py = pz = p/3 at eta=0.5, verified on an isolated channel."""
        base = stim.Circuit("R 0 1\nDEPOLARIZE1(0.01) 0\nM 0 1\nDETECTOR rec[-1]")
        biased = apply_xz_bias(base, 0.5)
        args = next(i.gate_args_copy() for i in biased if i.name == "PAULI_CHANNEL_1")
        assert args == pytest.approx([0.01 / 3, 0.01 / 3, 0.01 / 3])

    def test_bias_changes_the_error_model(self) -> None:
        circuit, _ = build_circuit(3, 0.01)
        plain = circuit.detector_error_model(decompose_errors=True)
        biased = apply_xz_bias(circuit, 20.0).detector_error_model(decompose_errors=True)
        assert str(plain) != str(biased)
        assert plain.num_detectors == biased.num_detectors

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite_eta(self, bad: float) -> None:
        """NaN passes an `eta <= 0` guard (every NaN comparison is False) and +inf
        turns pz into inf/inf = NaN; both then failed deep inside stim with an opaque
        message, or succeeded silently on a circuit with no DEPOLARIZE1."""
        circuit, _ = build_circuit(3, 0.01)
        with pytest.raises(ValueError, match="finite positive"):
            apply_xz_bias(circuit, bad)
