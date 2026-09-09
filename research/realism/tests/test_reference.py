"""Reference tests use closed-form channels and independent classical histories."""

import itertools
import json
import math

import numpy as np
import pytest

from research.realism.reference import (
    apply_channel,
    channel_kraus,
    detection_observable_distribution,
    repeated_parity_distribution,
    report_reference,
    total_variation,
)


@pytest.mark.parametrize("approximation", ["exact", "pauli_twirl"])
@pytest.mark.parametrize(
    "duration,t1,t2", [(0.0, 20.0, 30.0), (1.0, 20.0, 40.0), (100.0, 2.0, 1.0)]
)
def test_channel_preserves_trace_and_positivity(
    approximation: str, duration: float, t1: float, t2: float
) -> None:
    operators = channel_kraus(
        duration=duration,
        t1=t1,
        t2=t2,
        approximation=approximation,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        sum(operator.conj().T @ operator for operator in operators), np.eye(2)
    )
    vector = np.array([1, 2j], dtype=np.complex128) / math.sqrt(5)
    initial = np.asarray(np.outer(vector, vector.conj()), dtype=np.complex128)
    evolved = apply_channel(initial, operators)
    assert np.trace(evolved).real == pytest.approx(1)
    np.testing.assert_allclose(evolved, evolved.conj().T)
    assert np.linalg.eigvalsh(evolved).min() >= -1e-14
    if duration == 0:
        np.testing.assert_allclose(evolved, initial)


def test_exact_channel_retains_ground_preference_and_correct_transverse_decay() -> None:
    gamma = -math.expm1(-1 / 20)
    exact = channel_kraus(duration=1, t1=20, t2=30)
    twirl = channel_kraus(duration=1, t1=20, t2=30, approximation="pauli_twirl")
    ground = np.diag(np.array([1, 0], dtype=np.complex128))
    excited = np.diag(np.array([0, 1], dtype=np.complex128))
    plus = np.full((2, 2), 0.5, dtype=np.complex128)
    assert apply_channel(ground, exact)[1, 1] == 0
    assert apply_channel(ground, twirl)[1, 1] == pytest.approx(gamma / 2)
    assert apply_channel(excited, exact)[0, 0] == pytest.approx(gamma)
    assert apply_channel(excited, twirl)[0, 0] == pytest.approx(gamma / 2)
    for channel in (exact, twirl):
        assert apply_channel(plus, channel)[0, 1] == pytest.approx(0.5 * math.exp(-1 / 30))


@pytest.mark.parametrize("duration,t1,t2", [(-1, 2, 2), (1, 0, 2), (1, 2, 5), (math.nan, 2, 2)])
def test_incompatible_channel_parameters_are_refused(duration: float, t1: float, t2: float) -> None:
    with pytest.raises(ValueError):
        channel_kraus(duration=duration, t1=t1, t2=t2)


@pytest.mark.parametrize("approximation", ["exact", "pauli_twirl"])
@pytest.mark.parametrize("initial_data", ["00", "11"])
def test_joint_repeated_measurements_match_independent_classical_histories(
    approximation: str, initial_data: str
) -> None:
    gamma = -math.expm1(-1 / 20)
    expected: dict[str, float] = {}
    for outcomes in itertools.product((0, 1), repeat=6):
        previous = [int(initial_data[0]), int(initial_data[1])]
        probability = 1.0
        parities = []
        for round_index in range(3):
            current = outcomes[2 * round_index : 2 * round_index + 2]
            for old, new in zip(previous, current, strict=True):
                if approximation == "pauli_twirl":
                    probability *= gamma / 2 if old != new else 1 - gamma / 2
                elif old:
                    probability *= gamma if new == 0 else 1 - gamma
                else:
                    probability *= float(new == 0)
            parities.append(current[0] ^ current[1])
            previous = list(current)
        key = "".join(str(bit) for bit in [*parities, *previous])
        expected[key] = expected.get(key, 0) + probability
    observed = repeated_parity_distribution(
        duration=1,
        t1=20,
        t2=30,
        initial_data=initial_data,  # type: ignore[arg-type]
        approximation=approximation,  # type: ignore[arg-type]
    )
    assert sum(observed.values()) == pytest.approx(1)
    assert total_variation(observed, expected) < 1e-14


def test_zero_noise_has_no_detection_or_observable_flips() -> None:
    for initial in ("00", "11"):
        for approximation in ("exact", "pauli_twirl"):
            joint = repeated_parity_distribution(
                duration=0,
                t1=20,
                t2=30,
                initial_data=initial,
                approximation=approximation,
            )
            assert detection_observable_distribution(joint, initial_data=initial) == {"00000": 1}


def test_detection_mapping_orders_nonzero_transitions_and_uses_preparation_reference() -> None:
    # Ancilla readouts 0,1,0 then data 1,0 imply detections 0,1,1,1.
    # Ancilla readouts 1,0,1 then data 1,1 imply detections 1,1,1,1.
    joint = {"01010": 0.3, "10111": 0.7}
    assert detection_observable_distribution(joint, initial_data="00") == {
        "01111": 0.3,
        "11111": 0.7,
    }
    assert detection_observable_distribution(joint, initial_data="11") == {
        "01110": 0.3,
        "11110": 0.7,
    }


@pytest.mark.parametrize("rounds", [0, 9, True])
def test_exact_enumeration_rejects_invalid_or_excessive_rounds(rounds: int) -> None:
    with pytest.raises(ValueError, match="rounds"):
        repeated_parity_distribution(duration=1, t1=20, t2=30, rounds=rounds)


def test_report_exposes_discrepancy_instead_of_treating_twirl_as_exact() -> None:
    report = report_reference()
    assert "not measured" in str(report["parameter_status"])
    serialized = json.dumps(report, allow_nan=False)
    assert "joint_detection_observable_total_variation" in serialized
    ground = repeated_parity_distribution(duration=1, t1=20, t2=30, initial_data="00")
    twirl = repeated_parity_distribution(
        duration=1, t1=20, t2=30, initial_data="00", approximation="pauli_twirl"
    )
    assert ground == {"00000": 1}
    assert total_variation(ground, twirl) > 0.1
