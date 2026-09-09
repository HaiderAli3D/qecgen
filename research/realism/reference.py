"""Small exact-channel references that expose what a Pauli approximation removes.

This is exact density-matrix evolution for the stated two-level channel and ideal
three-qubit circuit, not exact transmon physics. Ghosh, Fowler and Geller,
https://arxiv.org/pdf/1210.5799, Eqs. (1)-(10), supply the relaxation/dephasing
channel and its Pauli twirl. All report defaults are illustrative mathematical
inputs, not measured device parameters. No samples or fault targets are produced.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

type Matrix = NDArray[np.complex128]
type Approximation = Literal["exact", "pauli_twirl"]


class ChannelParameters(TypedDict):
    duration: float
    t1: float
    t2: float


_IDENTITY: Matrix = np.eye(2, dtype=np.complex128)
_X: Matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y: Matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z: Matrix = np.diag(np.array([1, -1], dtype=np.complex128))


def channel_kraus(
    *, duration: float, t1: float, t2: float, approximation: Approximation = "exact"
) -> tuple[Matrix, ...]:
    """Keep relaxation's ground-state preference in the reference, not in the twirl.

    Times must have matching units. The model assumes exponential, zero-temperature
    amplitude damping with independent Markovian pure dephasing. T2 > 2*T1 is
    incompatible with these assumptions and is rejected, not silently repaired.
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(t1) or not math.isfinite(t2) or min(t1, t2) <= 0:
        raise ValueError("T1 and T2 must be finite and positive")
    if t2 > 2 * t1:
        raise ValueError("T2 must not exceed 2*T1 for this channel")
    gamma = -math.expm1(-duration / t1)
    if approximation == "exact":
        phase_flip = -0.5 * math.expm1(-duration * (1 / t2 - 1 / (2 * t1)))
        amplitude = (
            np.diag(np.array([1, math.sqrt(1 - gamma)], dtype=np.complex128)),
            np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128),
        )
        phase = (math.sqrt(1 - phase_flip) * _IDENTITY, math.sqrt(phase_flip) * _Z)
        return tuple(phase_operator @ damping for phase_operator in phase for damping in amplitude)
    if approximation == "pauli_twirl":
        px = py = gamma / 4
        # At T2=2*T1 and very short durations cancellation can introduce roundoff.
        pz = max(0.0, -math.expm1(-duration / t2) / 2 - px)
        probabilities = (1 - px - py - pz, px, py, pz)
        return tuple(
            math.sqrt(probability) * operator
            for probability, operator in zip(probabilities, (_IDENTITY, _X, _Y, _Z), strict=True)
        )
    raise ValueError(f"Unknown approximation: {approximation}")


def _embed(operator: Matrix, qubit: int, qubits: int) -> Matrix:
    # Basis index bit q is qubit q: little-endian, also for the reference matrices.
    result: Matrix = np.ones((1, 1), dtype=np.complex128)
    for index in reversed(range(qubits)):
        result = np.asarray(
            np.kron(result, operator if index == qubit else _IDENTITY), dtype=np.complex128
        )
    return result


def apply_channel(state: Matrix, operators: tuple[Matrix, ...], *, qubit: int = 0) -> Matrix:
    """Retain unnormalised branch weights so repeated measurements obey Born's rule."""
    dimension = state.shape[0]
    if state.ndim != 2 or state.shape != (dimension, dimension):
        raise ValueError("state must be a square matrix")
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("state dimension must be a power of two, at least two")
    qubits = dimension.bit_length() - 1
    if not 0 <= qubit < qubits:
        raise ValueError("qubit index is outside the state")
    result = np.zeros_like(state)
    for operator in operators:
        if operator.shape != (2, 2):
            raise ValueError("channel operators must be two by two")
        embedded = _embed(operator, qubit, qubits)
        result += embedded @ state @ embedded.conj().T
    return result


def _cnot(*, control: int, target: int, qubits: int) -> Matrix:
    dimension = 1 << qubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in range(dimension):
        destination = index ^ (1 << target) if index & (1 << control) else index
        result[destination, index] = 1
    return result


def repeated_parity_distribution(
    *,
    duration: float,
    t1: float,
    t2: float,
    rounds: int = 3,
    initial_data: Literal["00", "11"] = "11",
    approximation: Approximation = "exact",
) -> dict[str, float]:
    """Enumerate joint parity-readout and final-data outcomes of a tiny memory.

    Each round idles both data qubits through the channel, performs ideal CNOTs
    data0->ancilla and data1->ancilla, then measures and resets the ancilla. Noise
    acts on data only; all preparation, gates, readout and reset are ideal. Keys
    list parity measurements in time order, followed by final data0 and data1.
    The bound prevents mistaking exponential enumeration for a bulk generator.
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 8:
        raise ValueError("rounds must be an integer from one to eight")
    if initial_data not in ("00", "11"):
        raise ValueError("initial_data must be 00 or 11")
    operators = channel_kraus(duration=duration, t1=t1, t2=t2, approximation=approximation)
    state = np.zeros((8, 8), dtype=np.complex128)
    initial_index = 3 if initial_data == "11" else 0
    state[initial_index, initial_index] = 1
    gates = _cnot(control=1, target=2, qubits=3) @ _cnot(control=0, target=2, qubits=3)
    reset_x = _embed(_X, 2, 3)
    projectors = tuple(
        np.diag(np.array([int((index >> 2) == outcome) for index in range(8)]))
        for outcome in (0, 1)
    )
    branches: dict[str, Matrix] = {"": state}
    for _ in range(rounds):
        next_branches: dict[str, Matrix] = {}
        for prefix, branch in branches.items():
            evolved = apply_channel(apply_channel(branch, operators, qubit=0), operators, qubit=1)
            evolved = gates @ evolved @ gates.conj().T
            for outcome, projector in enumerate(projectors):
                measured = projector @ evolved @ projector
                if np.trace(measured).real > 0:
                    if outcome:
                        measured = reset_x @ measured @ reset_x.conj().T
                    next_branches[f"{prefix}{outcome}"] = measured
        branches = next_branches
    result: dict[str, float] = {}
    for prefix, branch in branches.items():
        for data_index in range(4):
            probability = float(branch[data_index, data_index].real)
            if probability > 0:
                result[f"{prefix}{data_index & 1}{(data_index >> 1) & 1}"] = probability
    return result


def detection_observable_distribution(
    joint: dict[str, float], *, initial_data: Literal["00", "11"]
) -> dict[str, float]:
    """Apply the same ideal reference to both channels rather than comparing raw bits.

    Both allowed initial states have even parity. Keys contain initial parity,
    round-to-round differences, final parity difference, then the data0 logical
    flip relative to preparation. This describes observable outcomes, not faults.
    """
    if initial_data not in ("00", "11"):
        raise ValueError("initial_data must be 00 or 11")
    result: dict[str, float] = {}
    for key, probability in joint.items():
        if len(key) < 3 or any(bit not in "01" for bit in key):
            raise ValueError("joint keys require parity readouts followed by two data bits")
        parity = [int(bit) for bit in key[:-2]]
        data0, data1 = int(key[-2]), int(key[-1])
        detections = [parity[0], *(a ^ b for a, b in pairwise(parity))]
        detections.append(data0 ^ data1 ^ parity[-1])
        observable = data0 ^ int(initial_data[0])
        mapped = "".join(str(bit) for bit in [*detections, observable])
        result[mapped] = result.get(mapped, 0.0) + probability
    return result


def total_variation(first: dict[str, float], second: dict[str, float]) -> float:
    return sum(abs(first.get(key, 0) - second.get(key, 0)) for key in first.keys() | second) / 2


def report_reference() -> dict[str, object]:
    """Report reproducible approximation discrepancies without fitted device claims."""
    parameters: ChannelParameters = {"duration": 1.0, "t1": 20.0, "t2": 30.0}
    exact = channel_kraus(**parameters)
    twirl = channel_kraus(**parameters, approximation="pauli_twirl")
    states = {
        "ground": np.array([1, 0], dtype=np.complex128),
        "excited": np.array([0, 1], dtype=np.complex128),
        "plus_x": np.array([1, 1], dtype=np.complex128) / math.sqrt(2),
        "plus_y": np.array([1, 1j], dtype=np.complex128) / math.sqrt(2),
    }
    single_qubit = {}
    for name, vector in states.items():
        state = np.asarray(np.outer(vector, vector.conj()), dtype=np.complex128)
        expected, approximate = apply_channel(state, exact), apply_channel(state, twirl)
        single_qubit[name] = {
            "exact_excited_probability": float(expected[1, 1].real),
            "pauli_excited_probability": float(approximate[1, 1].real),
            "trace_distance": float(np.abs(np.linalg.eigvalsh(expected - approximate)).sum() / 2),
            "off_diagonal_absolute_difference": float(abs(expected[0, 1] - approximate[0, 1])),
        }
    repeated = {}
    for initial in ("00", "11"):
        distributions = {
            mode: detection_observable_distribution(
                repeated_parity_distribution(
                    **parameters, initial_data=initial, approximation=mode
                ),
                initial_data=initial,
            )
            for mode in ("exact", "pauli_twirl")
        }
        repeated[initial] = {
            "joint_detection_observable_total_variation": total_variation(
                distributions["exact"], distributions["pauli_twirl"]
            ),
            "exact_observable_flip_probability": sum(
                p for key, p in distributions["exact"].items() if key[-1] == "1"
            ),
            "pauli_observable_flip_probability": sum(
                p for key, p in distributions["pauli_twirl"].items() if key[-1] == "1"
            ),
            "distributions": distributions,
        }
    return {
        "source": "https://arxiv.org/pdf/1210.5799",
        "source_equations": "1-10",
        "parameters": {**parameters, "time_unit": "microsecond", "rounds": 3},
        "parameter_status": "illustrative inputs, not measured or fitted hardware values",
        "reference_scope": (
            "Exact two-level zero-temperature amplitude damping and independent Markovian "
            "dephasing; not exact transmon dynamics."
        ),
        "motif": (
            "Two data qubits and one ancilla; per round: data idle, two ideal CNOTs, "
            "ideal ancilla measurement/reset; ideal final data measurement."
        ),
        "distribution_key": "four detection bits in time order, then one logical observable flip",
        "single_qubit": single_qubit,
        "repeated_parity": repeated,
        "limitation": (
            "The Pauli twirl preserves transverse decay but removes ground-state preference. "
            "These discrepancies test that approximation; they do not measure hardware transfer."
        ),
    }
