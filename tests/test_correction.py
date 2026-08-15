"""Scoring a supplied physical Pauli correction.

The correctness argument is not "the formula looks right". It is that the fast bitwise
scorer agrees with a literal apply-and-measure injection, exhaustively at d=3 and on
random corrections at d=5, in both bases and both layouts.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import stim

from qecgen.correction import (
    CorrectionFrame,
    PauliBasis,
    build_correction_schema,
    estimate_correction_logical_error_rate,
    extract_logical_operators,
    inject_correction,
    logical_effect,
    pack_correction,
    score_corrections,
)
from qecgen.sampling import unpack_bits

CONFIGS = [
    ("surface_code:rotated_memory_z", 3),
    ("surface_code:rotated_memory_x", 3),
    ("surface_code:unrotated_memory_z", 3),
    ("surface_code:unrotated_memory_x", 3),
]


def _noiseless(task: str, distance: int) -> stim.Circuit:
    return stim.Circuit.generated(task, distance=distance, rounds=distance)


def _tail_from_final_readout(circuit: stim.Circuit) -> stim.Circuit:
    """The circuit from its last non-resetting measurement instruction onwards.

    Derived here rather than reusing `final_data_measurement_layer`, so the flow oracle
    does not depend on the same layer logic it is helping to check.
    """
    flat = list(circuit.flattened())
    first = max(
        index
        for index, instruction in enumerate(flat)
        if instruction.name in {"M", "MX", "MY", "MZ"}
    )
    tail = stim.Circuit()
    for instruction in flat[first:]:
        tail.append(instruction)
    return tail


def _noisy(task: str, distance: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated(
        task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


# --- extraction -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "distance", "basis", "support"),
    [
        ("surface_code:rotated_memory_z", 3, PauliBasis.Z, (1, 3, 5)),
        ("surface_code:rotated_memory_x", 3, PauliBasis.X, (1, 8, 15)),
        ("surface_code:unrotated_memory_z", 3, PauliBasis.Z, (0, 2, 4)),
        ("surface_code:rotated_memory_z", 5, PauliBasis.Z, (1, 3, 5, 7, 9)),
    ],
)
def test_support_matches_measured_facts(
    task: str, distance: int, basis: PauliBasis, support: tuple[int, ...]
) -> None:
    """Independently measured against stim before any of this was written."""
    operators = extract_logical_operators(_noiseless(task, distance))
    assert operators.n_observables == 1
    assert operators.stim_support(0) == support
    assert operators.basis_of(0) is basis
    assert operators.weight(0) == distance


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("rotated", [True, False])
@pytest.mark.parametrize("memory", ["z", "x"])
def test_stim_flow_confirms_the_extracted_operator(
    distance: int, rotated: bool, memory: str
) -> None:
    """An independent oracle: stim's own flow machinery, not the rec walk again.

    The circuit is split at the final data layer; the extracted operator must be the one
    the tail measures into the observable.
    """
    layout = "rotated" if rotated else "unrotated"
    circuit = _noiseless(f"surface_code:{layout}_memory_{memory}", distance)
    operators = extract_logical_operators(circuit)
    tail = _tail_from_final_readout(circuit)

    operator = operators.to_pauli_string(0, circuit.num_qubits)
    assert tail.has_flow(
        stim.Flow(
            input=operator,
            output=stim.PauliString(circuit.num_qubits),
            included_observables=[0],
        ),
        unsigned=True,
    )


@pytest.mark.parametrize("distance", [3, 5])
def test_stim_flow_rejects_a_perturbed_operator(distance: int) -> None:
    """Negative control.

    Without this, the flow test above could be passing on a vacuously satisfiable flow.
    """
    circuit = _noiseless("surface_code:rotated_memory_z", distance)
    operators = extract_logical_operators(circuit)
    tail = _tail_from_final_readout(circuit)

    perturbed = operators.to_pauli_string(0, circuit.num_qubits)
    perturbed[operators.stim_support(0)[0]] = "Y"
    assert not tail.has_flow(
        stim.Flow(
            input=perturbed,
            output=stim.PauliString(circuit.num_qubits),
            included_observables=[0],
        ),
        unsigned=True,
    )


@pytest.mark.parametrize(("task", "distance"), [(t, d) for t, d in CONFIGS])
def test_data_qubit_count_matches_the_code_layout(task: str, distance: int) -> None:
    schema = build_correction_schema(_noiseless(task, distance))
    expected = distance**2 if "unrotated" not in task else distance**2 + (distance - 1) ** 2
    assert schema.n_data_qubits == expected


@pytest.mark.parametrize(("task", "distance"), [(t, d) for t, d in CONFIGS])
def test_data_qubits_are_disjoint_from_ancilla_measure_qubits(task: str, distance: int) -> None:
    """The test that catches the annotation-skipping layer bug.

    A rule that skips only annotations merges the final ancilla `MR` layer into the data
    `M` layer -- 17 "data qubits" instead of 9 at rotated d=3. The end-to-end scoring
    tests do NOT catch it, because ancilla entries contribute nothing to the observable.
    """
    circuit = _noiseless(task, distance)
    schema = build_correction_schema(circuit)
    data = {q.stim_qubit for q in schema.data_qubits}
    ancilla = {
        int(target.qubit_value)
        for instruction in circuit.flattened()
        if instruction.name in {"MR", "MRX", "MRY", "MRZ"}
        for target in instruction.targets_copy()
    }
    assert data and ancilla
    assert not (data & ancilla)


def test_every_data_qubit_has_coordinates_and_a_record() -> None:
    schema = build_correction_schema(_noiseless("surface_code:rotated_memory_z", 5))
    assert all(len(q.coords) >= 2 for q in schema.data_qubits)
    records = [q.measurement_record for q in schema.data_qubits]
    assert len(set(records)) == len(records)
    assert [q.index for q in schema.data_qubits] == list(range(schema.n_data_qubits))


def test_schema_digest_is_stable_and_noise_independent() -> None:
    """The FROZEN_PRIOR argument, asserted rather than claimed.

    Logical operator support is a property of the code, not of the noise. If this ever
    stops holding -- say because a drift axis starts varying the code -- the suite says so.
    """
    base = build_correction_schema(_noiseless("surface_code:rotated_memory_z", 3)).digest()
    for p in (0.0, 1e-4, 0.01, 0.05):
        assert (
            build_correction_schema(_noisy("surface_code:rotated_memory_z", 3, p)).digest() == base
        )


def test_refuses_a_pauli_target_observable() -> None:
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        R 0
        M 0
        OBSERVABLE_INCLUDE(0) Z0
    """)
    with pytest.raises(ValueError, match="Pauli target"):
        extract_logical_operators(circuit)


def test_refuses_an_observable_outside_the_final_data_layer() -> None:
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        MR 1
        M 0
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    with pytest.raises(ValueError, match="not in the final data layer"):
        extract_logical_operators(circuit)


def test_refuses_a_repeated_qubit_in_the_final_layer() -> None:
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        R 0
        M 0 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    with pytest.raises(ValueError, match="measured twice"):
        extract_logical_operators(circuit)


def test_duplicate_rec_targets_cancel_mod_2() -> None:
    """XOR accumulation, not append. stim genuinely cancels a repeated record."""
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        R 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-1]
    """)
    operators = extract_logical_operators(circuit)
    assert operators.support(0) == ()
    _, observables = circuit.compile_detector_sampler(seed=1).sample(8, separate_observables=True)
    assert not observables.any()


def test_mixed_basis_observable_is_represented_not_refused() -> None:
    """A mixed-basis observable is an ordinary Pauli; the symplectic pair holds it exactly.

    Refusing here would be a validator that fails on correct data. It is reported through
    `basis_of` returning None, and guarded only behind the opt-in strict flag.
    """
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        MX 0
        M 1
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
    """)
    operators = extract_logical_operators(circuit)
    assert operators.basis_of(0) is None
    assert bool(operators.unpacked_x()[0][0])
    assert bool(operators.unpacked_z()[0][1])
    with pytest.raises(ValueError, match="more than one Pauli basis"):
        extract_logical_operators(circuit, strict_single_basis=True)


# --- the injection oracle ----------------------------------------------------------


def test_a_deterministic_pauli_gate_is_absorbed_by_the_reference_sample() -> None:
    """The most valuable test in this file.

    `compile_detector_sampler` defines observables relative to a reference sample of the
    same circuit, so a deterministic X is folded into the reference and reports nothing.
    An oracle built on deterministic gates passes every test while measuring nothing.
    This test exists to stop anyone "simplifying" `inject_correction` into exactly that.
    """
    circuit = _noiseless("surface_code:rotated_memory_z", 3)
    flat = list(circuit.flattened())
    first = min(index for index, i in enumerate(flat) if i.name == "M" and index > len(flat) - 8)

    def build(gate: str, arg: float | None) -> stim.Circuit:
        out = stim.Circuit()
        for index, instruction in enumerate(flat):
            if index == first:
                if arg is None:
                    out.append(gate, [1])
                else:
                    out.append(gate, [1], arg)
            out.append(instruction)
        return out

    deterministic = build("X", None).compile_detector_sampler(seed=3)
    dets, obs = deterministic.sample(16, separate_observables=True)
    assert not obs.any(), "a deterministic X was absorbed -- this is the trap"
    assert dets.sum() == 0

    as_error = build("X_ERROR", 1.0).compile_detector_sampler(seed=3)
    dets, obs = as_error.sample(16, separate_observables=True)
    assert obs.all()
    assert dets.sum() > 0


@pytest.mark.parametrize(("task", "distance"), [(t, d) for t, d in CONFIGS])
def test_scorer_matches_exhaustive_single_qubit_injection(task: str, distance: int) -> None:
    """Every data qubit x {X, Z, Y}, scored both ways."""
    circuit = _noiseless(task, distance)
    schema = build_correction_schema(circuit)
    operators = extract_logical_operators(circuit)
    n = schema.n_data_qubits

    for index in range(n):
        for want_x, want_z in ((True, False), (False, True), (True, True)):
            cx = np.zeros(n, dtype=bool)
            cz = np.zeros(n, dtype=bool)
            cx[index] = want_x
            cz[index] = want_z

            predicted = unpack_bits(
                logical_effect(
                    pack_correction(cx.reshape(1, n)),
                    pack_correction(cz.reshape(1, n)),
                    operators,
                ),
                operators.n_observables,
            )[0]
            injected = inject_correction(circuit, cx, cz, schema)
            _, observed = injected.compile_detector_sampler(seed=5).sample(
                4, separate_observables=True
            )
            assert np.array_equal(predicted, observed[0]), (
                f"{task} d={distance} qubit index {index} x={want_x} z={want_z}"
            )


@pytest.mark.slow
@pytest.mark.parametrize(("task", "distance"), [("surface_code:rotated_memory_z", 5)])
def test_scorer_matches_random_multiqubit_injection(task: str, distance: int) -> None:
    circuit = _noiseless(task, distance)
    schema = build_correction_schema(circuit)
    operators = extract_logical_operators(circuit)
    n = schema.n_data_qubits
    rng = np.random.default_rng(11)

    for _ in range(40):
        cx = rng.random(n) < 0.3
        cz = rng.random(n) < 0.3
        predicted = unpack_bits(
            logical_effect(
                pack_correction(cx.reshape(1, n)), pack_correction(cz.reshape(1, n)), operators
            ),
            operators.n_observables,
        )[0]
        injected = inject_correction(circuit, cx, cz, schema)
        _, observed = injected.compile_detector_sampler(seed=5).sample(4, separate_observables=True)
        assert np.array_equal(predicted, observed[0])


def test_odd_parity_on_the_logical_support_flips_the_observable() -> None:
    """The measured table, encoded: X on 1 or 3 flips; X on {1,3} does not."""
    circuit = _noiseless("surface_code:rotated_memory_z", 3)
    operators = extract_logical_operators(circuit)
    schema = build_correction_schema(circuit)
    n = schema.n_data_qubits
    support = operators.support(0)

    for size in range(len(support) + 1):
        for subset in itertools.combinations(support, size):
            cx = np.zeros(n, dtype=bool)
            cx[list(subset)] = True
            flip = unpack_bits(
                logical_effect(
                    pack_correction(cx.reshape(1, n)),
                    pack_correction(np.zeros((1, n), dtype=bool)),
                    operators,
                ),
                1,
            )[0][0]
            assert bool(flip) == (len(subset) % 2 == 1)


# --- rates ------------------------------------------------------------------------


def test_identity_correction_equals_the_raw_logical_error_rate() -> None:
    """Exact integer equality, not a statistical comparison."""
    circuit = _noisy("surface_code:rotated_memory_z", 3, 0.01)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    shots = 20_000
    _, observables = circuit.compile_detector_sampler(seed=42).sample(
        shots, separate_observables=True, bit_packed=True
    )
    zeros = pack_correction(np.zeros((shots, n), dtype=bool))
    estimate = estimate_correction_logical_error_rate(zeros, zeros, observables, operators)
    raw = int(np.any(unpack_bits(observables, operators.n_observables), axis=1).sum())
    assert estimate.failures == raw


def test_perfect_correction_gives_zero_failures() -> None:
    """Applying X to one support qubit exactly when the observable flipped."""
    circuit = _noisy("surface_code:rotated_memory_z", 3, 0.01)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    shots = 20_000
    _, observables = circuit.compile_detector_sampler(seed=42).sample(
        shots, separate_observables=True, bit_packed=True
    )
    flipped = unpack_bits(observables, 1)[:, 0]
    cx = np.zeros((shots, n), dtype=bool)
    cx[flipped, operators.support(0)[0]] = True
    estimate = estimate_correction_logical_error_rate(
        pack_correction(cx),
        pack_correction(np.zeros((shots, n), dtype=bool)),
        observables,
        operators,
    )
    assert estimate.failures == 0
    assert estimate.interval.point == 0.0


def test_pymatching_prediction_lifted_to_a_correction_reproduces_contract_a() -> None:
    """Two independent routes to the same verdict, compared as exact integers.

    PyMatching cannot supply a physical correction -- its fault ids are observables, and
    matching-graph edges are spacetime DEM mechanisms, many with no data-qubit support at
    all. What it can supply is the logical action, which is lifted here onto one support
    qubit. The failure count must match qa.py's Contract A count exactly.
    """
    import pymatching

    circuit = _noisy("surface_code:rotated_memory_z", 3, 0.01)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    shots = 20_000

    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    detectors, observables = circuit.compile_detector_sampler(seed=42).sample(
        shots, separate_observables=True, bit_packed=True
    )
    predictions = matching.decode_batch(
        detectors, bit_packed_shots=True, bit_packed_predictions=True
    )
    contract_a_failures = int(
        np.any(
            unpack_bits(predictions, operators.n_observables)
            != unpack_bits(observables, operators.n_observables),
            axis=1,
        ).sum()
    )

    predicted_flip = unpack_bits(predictions, 1)[:, 0]
    cx = np.zeros((shots, n), dtype=bool)
    cx[predicted_flip, operators.support(0)[0]] = True
    estimate = estimate_correction_logical_error_rate(
        pack_correction(cx),
        pack_correction(np.zeros((shots, n), dtype=bool)),
        observables,
        operators,
    )
    assert estimate.failures == contract_a_failures


def test_big_endian_packing_produces_a_different_verdict() -> None:
    """The bit-order convention is load-bearing, not decorative."""
    circuit = _noiseless("surface_code:rotated_memory_z", 5)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    bits = np.zeros((1, n), dtype=bool)
    bits[0, operators.support(0)[0]] = True
    zeros = pack_correction(np.zeros((1, n), dtype=bool))

    little = logical_effect(pack_correction(bits), zeros, operators)
    big = logical_effect(np.packbits(bits, axis=1, bitorder="big"), zeros, operators)
    assert not np.array_equal(little, big)


def test_scorer_refuses_an_unpacked_array_by_width() -> None:
    circuit = _noiseless("surface_code:rotated_memory_z", 3)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    unpacked = np.zeros((4, n), dtype=np.uint8)
    with pytest.raises(ValueError, match="pack_correction"):
        logical_effect(unpacked, unpacked, operators)


def test_scorer_refuses_wrong_dtype_and_row_mismatch() -> None:
    circuit = _noiseless("surface_code:rotated_memory_z", 3)
    operators = extract_logical_operators(circuit)
    width = operators.schema.packed_width
    with pytest.raises(ValueError, match="must be uint8"):
        logical_effect(
            np.zeros((4, width), dtype=bool), np.zeros((4, width), dtype=np.uint8), operators
        )
    with pytest.raises(ValueError, match="row-for-row"):
        logical_effect(
            np.zeros((4, width), dtype=np.uint8), np.zeros((5, width), dtype=np.uint8), operators
        )


def test_packed_and_unpacked_round_trip() -> None:
    circuit = _noiseless("surface_code:rotated_memory_z", 3)
    schema = build_correction_schema(circuit)
    rng = np.random.default_rng(3)
    bits = rng.random((7, schema.n_data_qubits)) < 0.5
    from qecgen.correction import unpack_correction

    assert np.array_equal(unpack_correction(pack_correction(bits), schema), bits)


def test_estimate_carries_the_schema_digest_and_an_interval() -> None:
    circuit = _noisy("surface_code:rotated_memory_z", 3, 0.01)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    shots = 2_000
    _, observables = circuit.compile_detector_sampler(seed=1).sample(
        shots, separate_observables=True, bit_packed=True
    )
    zeros = pack_correction(np.zeros((shots, n), dtype=bool))
    estimate = estimate_correction_logical_error_rate(zeros, zeros, observables, operators)
    assert estimate.schema_digest == operators.schema.digest()
    assert estimate.interval.low <= estimate.interval.point <= estimate.interval.high
    assert estimate.interval.trials == shots


def test_frame_is_recorded_in_the_schema() -> None:
    schema = build_correction_schema(_noiseless("surface_code:rotated_memory_z", 3))
    assert schema.frame is CorrectionFrame.FINAL_DATA_LAYER
    assert schema.bit_order == "little"
    assert "final_data_layer" in schema.to_json()


def test_score_corrections_returns_per_shot_success() -> None:
    circuit = _noisy("surface_code:rotated_memory_z", 3, 0.01)
    operators = extract_logical_operators(circuit)
    n = operators.schema.n_data_qubits
    shots = 500
    _, observables = circuit.compile_detector_sampler(seed=9).sample(
        shots, separate_observables=True, bit_packed=True
    )
    zeros = pack_correction(np.zeros((shots, n), dtype=bool))
    success = score_corrections(zeros, zeros, observables, operators)
    assert success.dtype == np.bool_
    assert success.shape == (shots,)
    assert success.sum() < shots
