"""Scoring a **supplied** physical Pauli correction by its logical effect.

This is not Contract C, and the distinction is load-bearing.

Contract C asks: *given a syndrome, which physical fault occurred?* That inverts a
deliberately many-to-one map, has no unique answer, and remains refused -- see
``DATA_CONTRACT.md``. This module asks the opposite question: *given a correction, what
was its logical effect?* That is a forward evaluation of a deterministic function,
single-valued for every input, with no tie-breaking anywhere. The degeneracy that makes
the inverse ill-posed is exactly what makes the forward map well-posed: every member of an
equivalence class has the same logical effect, so the answer does not depend on which
representative is held.

The client brief's "Nexus outputs a proposed correction. Apply and Measure. If the final
state matches the pristine state, the decoding was a success" is *this* operation. It
needs no ground-truth fault label, because the correction is an **input**, not a target.

The rule, for observable ``k`` with symplectic masks ``(L_x[k], L_z[k])`` over data qubits
and a proposed correction ``(C_x, C_z)``::

    predicted_flip[k] = parity(popcount(C_x & L_z[k]) + popcount(C_z & L_x[k]))
    success(shot)     = all(predicted_flip[k] == true_flip[k] for every k)

That is anticommutation of two Pauli operators. It is exact for a memory experiment,
because the logical observable is fully determined by the final data measurements. A ``Y``
correction sets both ``C_x`` and ``C_z`` on a qubit and falls out of the same expression
with no special case.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim

from qecgen.qa import Interval, clopper_pearson
from qecgen.sampling import packed_width, unpack_bits

__all__ = [
    "CorrectionEncoding",
    "CorrectionFrame",
    "CorrectionSchema",
    "CorrectionScoreEstimate",
    "DataQubit",
    "LogicalOperators",
    "PauliBasis",
    "build_correction_schema",
    "estimate_correction_logical_error_rate",
    "extract_logical_operators",
    "final_data_measurement_layer",
    "inject_correction",
    "logical_effect",
    "pack_correction",
    "score_corrections",
    "unpack_correction",
]

# The resetting family (MR/MRX/MRY/MRZ) is deliberately NOT in this set: excluding it
# is what stops the final ancilla layer merging into the data layer. Record *counting*
# does not enumerate names at all — it uses stim's per-instruction num_measurements.
_NON_RESETTING_MEASUREMENTS = frozenset({"M", "MX", "MY", "MZ"})
_ANNOTATIONS = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "SHIFT_COORDS", "QUBIT_COORDS"})

_PARITY: np.ndarray = np.array([bin(i).count("1") & 1 for i in range(256)], dtype=np.uint8)
"""Bit-parity of each byte value.

uint8 rather than bool so a fancy-index result can be reduced with ``bitwise_xor`` and
assigned straight into a uint8 output column.
"""


class PauliBasis(enum.StrEnum):
    """Basis a single data qubit is measured in at the final layer.

    Deliberately **not** :class:`qecgen.circuits.Basis`. That enum is the *memory
    experiment* basis and has exactly two members; a Y member there would be meaningless.
    Here Y is real: stim emits ``MY``/``MRY``, and a Y-basis record contributes both an X
    and a Z term to a logical operator's symplectic masks.
    """

    X = "x"
    Y = "y"
    Z = "z"


class CorrectionFrame(enum.StrEnum):
    """Where in the circuit a proposed correction is defined to act."""

    FINAL_DATA_LAYER = "final_data_layer"
    """Immediately before the final non-resetting data measurement layer.

    Single-member today. It is an enum rather than a bare string because the alternative
    frames are real and out of scope -- mid-circuit Pauli frames (which propagate through
    the remaining CX layers, so their final-layer residue is not the operator written),
    measurement-result corrections (not a Pauli on a data qubit at all), and non-Pauli
    corrections (no defined commutation relation with a Pauli observable). A file that
    silently omitted the frame would be unscoreable by anyone who assumed a different one.
    """


class CorrectionEncoding(enum.StrEnum):
    """How X/Y/Z are laid out in the correction arrays."""

    DUAL_MASK_PACKED = "dual_mask_packed"
    """Two parallel little-endian bit-packed uint8 arrays, x and z. ``Y == x & z``.

    Chosen over a 2-bit-per-qubit layout because this *is* the symplectic representation
    the scoring rule consumes: the formula reads straight off the two arrays, where a
    2-bit encoding would need a de-interleave on every shot. It also introduces no new
    bit-layout convention, and leaves no free choice (such as a codepoint order) left to
    get wrong. Cost is identical: two bits per qubit either way.
    """


@dataclass(frozen=True, slots=True)
class DataQubit:
    """One data qubit of the final measurement layer."""

    index: int
    """Compact ``0 .. n_data - 1`` position; the bit position in a correction array."""
    stim_qubit: int
    """Index in the circuit. Sparse, and NOT equal to :attr:`index`."""
    coords: tuple[float, ...]
    basis: PauliBasis
    """Basis of this qubit's final measurement."""
    measurement_record: int
    """Absolute index into the circuit's measurement stream."""


@dataclass(frozen=True)
class CorrectionSchema:
    """Everything a third party needs to emit a conforming correction.

    A property of the **code** -- distance, layout, basis, rounds -- not of the noise.
    """

    data_qubits: tuple[DataQubit, ...]
    frame: CorrectionFrame = CorrectionFrame.FINAL_DATA_LAYER
    encoding: CorrectionEncoding = CorrectionEncoding.DUAL_MASK_PACKED
    bit_order: str = "little"

    @property
    def n_data_qubits(self) -> int:
        """Number of data qubits, and the width of an unpacked correction row."""
        return len(self.data_qubits)

    @property
    def packed_width(self) -> int:
        """Bytes per packed correction row."""
        return packed_width(self.n_data_qubits)

    @property
    def single_basis(self) -> PauliBasis | None:
        """The one basis of the final layer, or None when the layer is mixed."""
        bases = {qubit.basis for qubit in self.data_qubits}
        return next(iter(bases)) if len(bases) == 1 else None

    def stim_qubit_ids(self) -> np.ndarray:
        """int32 ``(n_data,)`` map from compact index to stim qubit index."""
        return np.array([q.stim_qubit for q in self.data_qubits], dtype=np.int32)

    def coords_array(self) -> np.ndarray:
        """float64 ``(n_data, coord_dim)``."""
        if not self.data_qubits:
            return np.zeros((0, 0), dtype=np.float64)
        width = max(len(q.coords) for q in self.data_qubits)
        out = np.zeros((self.n_data_qubits, width), dtype=np.float64)
        for row, qubit in enumerate(self.data_qubits):
            out[row, : len(qubit.coords)] = qubit.coords
        return out

    def basis_codes(self) -> np.ndarray:
        """uint8 ``(n_data,)``: X=0, Y=1, Z=2."""
        order = {PauliBasis.X: 0, PauliBasis.Y: 1, PauliBasis.Z: 2}
        return np.array([order[q.basis] for q in self.data_qubits], dtype=np.uint8)

    def measurement_records(self) -> np.ndarray:
        """int64 ``(n_data,)`` position of each data qubit in the measurement stream."""
        return np.array([q.measurement_record for q in self.data_qubits], dtype=np.int64)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialisable form, with keys sorted for a stable digest."""
        return {
            "frame": str(self.frame),
            "encoding": str(self.encoding),
            "bit_order": self.bit_order,
            "n_data_qubits": self.n_data_qubits,
            "single_basis": None if self.single_basis is None else str(self.single_basis),
            "data_qubits": [
                {
                    "index": q.index,
                    "stim_qubit": q.stim_qubit,
                    "coords": list(q.coords),
                    "basis": str(q.basis),
                    "measurement_record": q.measurement_record,
                }
                for q in self.data_qubits
            ],
        }

    def to_json(self) -> str:
        """Canonical JSON. ``allow_nan=False``: a bare NaN token is not valid JSON."""
        return json.dumps(self.to_json_dict(), sort_keys=True, allow_nan=False)

    def digest(self) -> str:
        """Stable content digest of the schema.

        A correction score without one is unfalsifiable: the same correction array scored
        under a different qubit ordering gives a different, equally plausible rate.
        """
        return hashlib.blake2b(self.to_json().encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class LogicalOperators:
    """Symplectic (X, Z) masks of every ``OBSERVABLE_INCLUDE``, over data-qubit index."""

    schema: CorrectionSchema
    x_mask: np.ndarray
    """uint8 ``(n_observables, packed_width)``, little-endian."""
    z_mask: np.ndarray
    """uint8 ``(n_observables, packed_width)``, little-endian."""
    n_observables: int

    def unpacked_x(self) -> np.ndarray:
        """bool ``(n_observables, n_data_qubits)``."""
        return unpack_bits(self.x_mask, self.schema.n_data_qubits)

    def unpacked_z(self) -> np.ndarray:
        """bool ``(n_observables, n_data_qubits)``."""
        return unpack_bits(self.z_mask, self.schema.n_data_qubits)

    def support(self, observable: int) -> tuple[int, ...]:
        """Compact data-qubit indices this observable acts on."""
        touched = self.unpacked_x()[observable] | self.unpacked_z()[observable]
        return tuple(int(i) for i in np.flatnonzero(touched))

    def stim_support(self, observable: int) -> tuple[int, ...]:
        """Stim qubit indices this observable acts on."""
        ids = self.schema.stim_qubit_ids()
        return tuple(int(ids[i]) for i in self.support(observable))

    def weight(self, observable: int) -> int:
        """Number of data qubits in this observable's support."""
        return len(self.support(observable))

    def basis_of(self, observable: int) -> PauliBasis | None:
        """The single basis of this observable, or None when it is mixed.

        A mixed-basis observable is a perfectly ordinary Pauli operator and the symplectic
        pair represents it exactly, so this reports rather than refuses. Refusing here
        would be a validator that fails on correct data.
        """
        xs = self.unpacked_x()[observable]
        zs = self.unpacked_z()[observable]
        if np.any(xs & zs):
            return PauliBasis.Y if not np.any(xs ^ zs) else None
        if np.any(xs) and np.any(zs):
            return None
        if np.any(xs):
            return PauliBasis.X
        if np.any(zs):
            return PauliBasis.Z
        return None

    def to_pauli_string(self, observable: int, n_qubits: int) -> stim.PauliString:
        """Full-width :class:`stim.PauliString`, for :meth:`stim.Circuit.has_flow` oracles.

        Verification only. This is the bridge that lets a test hand the extracted operator
        to stim's own flow machinery instead of trusting the rec-walking extraction.
        """
        result = stim.PauliString(n_qubits)
        ids = self.schema.stim_qubit_ids()
        xs = self.unpacked_x()[observable]
        zs = self.unpacked_z()[observable]
        for index in range(self.schema.n_data_qubits):
            has_x, has_z = bool(xs[index]), bool(zs[index])
            if not (has_x or has_z):
                continue
            result[int(ids[index])] = "Y" if (has_x and has_z) else ("X" if has_x else "Z")
        return result


def final_data_measurement_layer(circuit: stim.Circuit) -> tuple[int, ...]:
    """Indices, into ``circuit.flattened()``, of the final data measurement layer.

    The layer is the maximal trailing run of **non-resetting** measurement instructions
    (``M``/``MX``/``MY``/``MZ``), skipping only pure annotations.

    Keying on non-resetting measurements is load-bearing, not stylistic. In a
    stim-generated surface code the final ancilla layer is ``MR ...`` and is separated
    from the final data layer ``M ...`` only by ``DETECTOR`` and ``SHIFT_COORDS``
    annotations. A rule that merely skips annotations merges the two and reports 17 "data
    qubits" for a rotated d=3 code instead of 9 -- and the end-to-end scoring tests still
    pass, because ancilla entries contribute nothing to the observable. Data qubits are
    never reset at the end of a memory experiment; ancillas always are. That is the
    discriminator.
    """
    flat = list(circuit.flattened())
    indices: list[int] = []
    for position in range(len(flat) - 1, -1, -1):
        name = flat[position].name
        if name in _ANNOTATIONS:
            continue
        if name in _NON_RESETTING_MEASUREMENTS:
            indices.append(position)
            continue
        break
    return tuple(sorted(indices))


def build_correction_schema(circuit: stim.Circuit) -> CorrectionSchema:
    """Derive the physical correction schema from a circuit.

    Raises:
        ValueError: when the final measurement layer is ``MPP`` (one measurement record
            spanning several qubits, so a per-qubit schema is undefined), when no final
            data measurement layer exists, when the layer repeats a qubit (the
            record-to-qubit map would be ambiguous), or when it contains a qubit with
            no ``QUBIT_COORDS`` entry (the schema would ship an unlabelled qubit).
    """
    flat = list(circuit.flattened())
    layer = final_data_measurement_layer(circuit)
    if not layer:
        # The layer walk breaks at the first non-annotation instruction that is not a
        # per-qubit non-resetting measurement, so a circuit whose data qubits are
        # measured via MPP arrives here with an EMPTY layer — a previous version
        # looked for MPP *inside* the layer, which is unreachable by construction.
        # Inspect the instruction the walk stopped on to give the specific diagnostic.
        stop = len(flat) - 1
        while stop >= 0 and flat[stop].name in _ANNOTATIONS:
            stop -= 1
        if stop >= 0 and flat[stop].name == "MPP":
            raise ValueError(
                "the final measurement layer is MPP: one measurement record spans "
                "several qubits, so a per-qubit correction schema is undefined."
            )
        raise ValueError(
            "no final data measurement layer found. This module scores memory "
            "experiments, which end by measuring every data qubit; a circuit that does "
            "not is outside the FINAL_DATA_LAYER frame."
        )

    # Absolute measurement-record index of every measurement in the circuit, so the layer
    # can be tied back to the stream that OBSERVABLE_INCLUDE's rec[-k] targets index.
    # Counting uses stim's own per-instruction record accounting rather than a name
    # allowlist: MPP, MXX/MYY/MZZ, MPAD and the heralded channels all append to the
    # measurement record, and any of them before the final layer silently shifted every
    # exported measurement_record index under allowlist counting.
    record = 0
    record_of: dict[int, int] = {}
    for position, instruction in enumerate(flat):
        if position in layer:
            record_of[position] = record
        record += instruction.num_measurements

    basis_of_gate = {
        "M": PauliBasis.Z,
        "MZ": PauliBasis.Z,
        "MX": PauliBasis.X,
        "MY": PauliBasis.Y,
    }
    coords = circuit.get_final_qubit_coordinates()

    entries: list[tuple[int, PauliBasis, int]] = []
    for position in layer:
        instruction = flat[position]
        basis = basis_of_gate[instruction.name]
        for offset, target in enumerate(instruction.targets_copy()):
            entries.append((int(target.qubit_value), basis, record_of[position] + offset))

    seen: set[int] = set()
    for stim_qubit, _, _ in entries:
        if stim_qubit in seen:
            raise ValueError(
                f"qubit {stim_qubit} is measured twice in the final data layer, so the "
                "measurement-record to qubit map is ambiguous."
            )
        seen.add(stim_qubit)

    missing = sorted(q for q, _, _ in entries if q not in coords)
    if missing:
        raise ValueError(
            f"data qubits {missing} have no QUBIT_COORDS entry, so the exported schema "
            "would ship unlabelled qubits that a third party cannot verify."
        )

    # Compact indexing in ascending stim qubit order. A rotated d=3 circuit allocates 26
    # qubit slots for 9 data qubits; a (shots, 26) correction array would leave 17 columns
    # with no defined meaning, which must then either be asserted zero (a validation
    # burden pushed onto the client) or silently ignored (worse). Compact indexing makes
    # the illegal state unrepresentable.
    entries.sort(key=lambda entry: entry[0])
    return CorrectionSchema(
        data_qubits=tuple(
            DataQubit(
                index=index,
                stim_qubit=stim_qubit,
                coords=tuple(float(c) for c in coords[stim_qubit]),
                basis=basis,
                measurement_record=measurement_record,
            )
            for index, (stim_qubit, basis, measurement_record) in enumerate(entries)
        )
    )


def extract_logical_operators(
    circuit: stim.Circuit,
    strict_single_basis: bool = False,
) -> LogicalOperators:
    """Recover every logical observable as a Pauli operator over data qubits.

    Walks ``circuit.flattened()``, builds a measurement-record index to (gate, qubit) map,
    then resolves each ``OBSERVABLE_INCLUDE``'s ``rec[-k]`` targets against it.

    Accumulation is **XOR**, not append: a detector or measurement repeated within one
    ``OBSERVABLE_INCLUDE`` cancels mod 2, which stim genuinely does.

    Args:
        circuit: A memory-experiment circuit.
        strict_single_basis: Refuse a mixed-basis observable. Off by default because a
            mixed-basis observable is exactly representable in the symplectic pair and the
            scoring rule is unchanged, so refusing would fail on correct data. The dataset
            export path passes True, since for every stim surface-code memory circuit a
            single basis is guaranteed and a violation means the circuit is not what the
            manifest claims.

    Raises:
        ValueError: for zero observables; a Pauli-target ``OBSERVABLE_INCLUDE`` (which
            names a mid-circuit time slice, outside this frame); a ``rec`` target that
            resolves outside the final data layer (so the observable is not a data-qubit
            operator at this time slice); or, with ``strict_single_basis``, a mixed basis.
    """
    if circuit.num_observables == 0:
        raise ValueError("circuit declares no observables, so there is nothing to score against")

    schema = build_correction_schema(circuit)
    index_of_record = {q.measurement_record: q.index for q in schema.data_qubits}
    basis_of_record = {q.measurement_record: q.basis for q in schema.data_qubits}

    flat = list(circuit.flattened())

    width = schema.packed_width
    x_bits = np.zeros((circuit.num_observables, schema.n_data_qubits), dtype=bool)
    z_bits = np.zeros((circuit.num_observables, schema.n_data_qubits), dtype=bool)

    # rec[-k] is resolved at the instruction's own position, exactly as stim resolves
    # it. Resolving against the end-of-circuit total silently mapped a mid-circuit
    # OBSERVABLE_INCLUDE onto whichever data qubit happened to own that absolute
    # index, and the out-of-layer ValueError below never fired for it.
    records_so_far = 0
    for instruction in flat:
        if instruction.name != "OBSERVABLE_INCLUDE":
            records_so_far += instruction.num_measurements
            continue
        observable = int(instruction.gate_args_copy()[0])
        for target in instruction.targets_copy():
            if not target.is_measurement_record_target:
                raise ValueError(
                    f"OBSERVABLE_INCLUDE({observable}) carries a Pauli target rather than "
                    "a rec[-k] target. That names a mid-circuit time slice, which is "
                    f"outside the {CorrectionFrame.FINAL_DATA_LAYER} frame this module scores."
                )
            record = records_so_far + target.value
            if record not in index_of_record:
                raise ValueError(
                    f"OBSERVABLE_INCLUDE({observable}) references measurement record "
                    f"{record}, which is not in the final data layer. The observable is "
                    "therefore not a data-qubit operator at this time slice, so a "
                    "data-qubit correction cannot be scored against it."
                )
            index = index_of_record[record]
            basis = basis_of_record[record]
            # XOR, never append: a repeated record cancels mod 2, which stim does too.
            if basis in (PauliBasis.X, PauliBasis.Y):
                x_bits[observable, index] ^= True
            if basis in (PauliBasis.Z, PauliBasis.Y):
                z_bits[observable, index] ^= True

    operators = LogicalOperators(
        schema=schema,
        x_mask=np.packbits(x_bits, axis=1, bitorder="little").reshape(
            circuit.num_observables, width
        ),
        z_mask=np.packbits(z_bits, axis=1, bitorder="little").reshape(
            circuit.num_observables, width
        ),
        n_observables=circuit.num_observables,
    )

    if strict_single_basis:
        mixed = [k for k in range(operators.n_observables) if operators.basis_of(k) is None]
        if mixed:
            raise ValueError(
                f"observables {mixed} span more than one Pauli basis. The scoring rule "
                "handles this correctly; strict_single_basis=True was requested, which "
                "means the circuit is not the single-basis memory experiment expected."
            )
    return operators


def pack_correction(bits: np.ndarray) -> np.ndarray:
    """Pack a bool ``(shots, n_data_qubits)`` array to little-endian uint8.

    Mirrors :func:`qecgen.sampling.unpack_bits`. ``bitorder="little"`` is hardcoded:
    NumPy's default is ``"big"``, the opposite of Stim's, and taking the default silently
    maps data qubit 0 onto bit 7 and produces a well-formed, wrong score. This is the one
    function a third party should ever use to pack a correction.
    """
    if bits.ndim != 2:
        raise ValueError(f"expected a 2-D (shots, n_data_qubits) array, got shape {bits.shape}")
    return np.packbits(bits.astype(bool), axis=1, bitorder="little")


def unpack_correction(packed: np.ndarray, schema: CorrectionSchema) -> np.ndarray:
    """Unpack a packed correction to bool ``(shots, n_data_qubits)``."""
    return unpack_bits(packed, schema.n_data_qubits)


def _require_packed(array: np.ndarray, schema: CorrectionSchema, name: str) -> None:
    """Reject the obvious third-party mistake: passing unpacked 0/1 columns.

    A ``(shots, n_data)`` unpacked uint8 array is indistinguishable by dtype from a packed
    one, so the width is the only available signal. Ambiguous only when ``n_data <= 8``,
    which no surface code with ``d >= 3`` reaches (minimum 9 data qubits).
    """
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (shots, {schema.packed_width}), got {array.shape}")
    if array.shape[1] == schema.packed_width:
        return
    if array.shape[1] == schema.n_data_qubits:
        raise ValueError(
            f"{name} has {array.shape[1]} columns, which is n_data_qubits rather than the "
            f"packed width {schema.packed_width}. It looks unpacked; pass it through "
            "qecgen.correction.pack_correction first."
        )
    raise ValueError(
        f"{name} must have {schema.packed_width} columns (packed width for "
        f"{schema.n_data_qubits} data qubits), got {array.shape[1]}"
    )


def logical_effect(
    correction_x: np.ndarray,
    correction_z: np.ndarray,
    operators: LogicalOperators,
) -> np.ndarray:
    """Observable flips induced by each correction.

    Returns uint8 ``(shots, packed_width(n_observables))``, little-endian bit-packed and
    directly comparable with ``InMemoryDataset.observables``.

    This is a deterministic function of a supplied correction. It is **not** an inference
    of a physical fault from a syndrome; see the module docstring and ``DATA_CONTRACT.md``.
    """
    schema = operators.schema
    _require_packed(correction_x, schema, "correction_x")
    _require_packed(correction_z, schema, "correction_z")
    if correction_x.shape[0] != correction_z.shape[0]:
        raise ValueError(
            f"correction_x has {correction_x.shape[0]} rows and correction_z has "
            f"{correction_z.shape[0]}; they must correspond row-for-row"
        )

    shots = correction_x.shape[0]
    flips = np.zeros((shots, operators.n_observables), dtype=bool)
    for observable in range(operators.n_observables):
        # parity(popcount(a)) ^ parity(popcount(b)) == parity(popcount(a ^ b)), so the
        # whole kernel is two ANDs, one XOR, a byte-wise reduce and one LUT lookup.
        overlap = (correction_x & operators.z_mask[observable]) ^ (
            correction_z & operators.x_mask[observable]
        )
        folded = np.bitwise_xor.reduce(overlap, axis=1)
        flips[:, observable] = _PARITY[folded].astype(bool)
    return np.packbits(flips, axis=1, bitorder="little").reshape(
        shots, packed_width(operators.n_observables)
    )


def score_corrections(
    correction_x: np.ndarray,
    correction_z: np.ndarray,
    observables: np.ndarray,
    operators: LogicalOperators,
) -> np.ndarray:
    """Per-shot success. bool ``(shots,)``; True means the decoding succeeded.

    A shot succeeds iff the correction's induced flip equals the true flip on **every**
    observable. Comparison goes through :func:`qecgen.sampling.unpack_bits` on both sides
    rather than comparing packed bytes, mirroring ``qa.py``: padding bits above
    ``n_observables`` are not part of the contract, and a byte compare would make the
    verdict depend on them.
    """
    if observables.shape[0] != correction_x.shape[0]:
        raise ValueError(
            f"observables has {observables.shape[0]} rows and the correction has "
            f"{correction_x.shape[0]}; they must correspond row-for-row"
        )
    # Width is enforced with the same rigour _require_packed applies to the correction
    # arrays. unpack_bits(count=...) would silently truncate an over-wide array — e.g.
    # the detectors array passed by mistake, measured to yield a plausible 0.98
    # "success rate" from detector bit 0 alone — and zero-fill an under-wide one.
    expected_width = packed_width(operators.n_observables)
    if observables.ndim != 2 or observables.shape[1] != expected_width:
        raise ValueError(
            f"observables has shape {observables.shape}; expected "
            f"(shots, {expected_width}) for {operators.n_observables} packed "
            "observables. Wrong-width input would otherwise be silently truncated or "
            "zero-filled into a plausible wrong score."
        )
    if observables.dtype != np.uint8:
        raise ValueError(
            f"observables must be packed uint8, got {observables.dtype}; pass the "
            "packed array a dataset file holds, not an unpacked bool view"
        )
    predicted = unpack_bits(
        logical_effect(correction_x, correction_z, operators), operators.n_observables
    )
    actual = unpack_bits(observables, operators.n_observables)
    success: np.ndarray = ~np.any(predicted != actual, axis=1)
    return success


@dataclass(frozen=True)
class CorrectionScoreEstimate:
    """Logical error rate of a supplied physical correction, with its interval."""

    interval: Interval
    shots: int
    failures: int
    n_observables: int
    n_data_qubits: int
    schema_digest: str
    """Digest of the :class:`CorrectionSchema` this was scored under.

    A rate without it is unfalsifiable: the same correction array scored under a different
    qubit ordering gives a different, equally plausible number.
    """

    def __str__(self) -> str:
        """Render as one summary line."""
        return (
            f"logical={self.interval} n_data={self.n_data_qubits} "
            f"n_obs={self.n_observables} schema={self.schema_digest[:12]}"
        )


def estimate_correction_logical_error_rate(
    correction_x: np.ndarray,
    correction_z: np.ndarray,
    observables: np.ndarray,
    operators: LogicalOperators,
    alpha: float = 0.05,
) -> CorrectionScoreEstimate:
    """Score a batch of corrections and return the rate with its exact interval.

    Uses :func:`qecgen.qa.clopper_pearson`, so this composes with every existing report.
    The house rule holds here too: no bare point estimate is ever returned.
    """
    success = score_corrections(correction_x, correction_z, observables, operators)
    shots = int(success.shape[0])
    failures = int((~success).sum())
    return CorrectionScoreEstimate(
        interval=clopper_pearson(failures, shots, alpha=alpha),
        shots=shots,
        failures=failures,
        n_observables=operators.n_observables,
        n_data_qubits=operators.schema.n_data_qubits,
        schema_digest=operators.schema.digest(),
    )


def inject_correction(
    circuit: stim.Circuit,
    correction_x: np.ndarray,
    correction_z: np.ndarray,
    schema: CorrectionSchema,
) -> stim.Circuit:
    """Literally apply a correction to the circuit, ahead of the final data layer.

    Inserts ``X_ERROR(1)`` / ``Z_ERROR(1)`` immediately before the first instruction of
    the final data measurement layer, and returns a new circuit. This is the client's
    "Apply and Measure" done literally; it exists to prove :func:`logical_effect` correct,
    not to replace it.

    **It must be ``X_ERROR(1)``, not ``X``.** ``compile_detector_sampler`` defines
    detectors and observables relative to a reference sample of the circuit it is compiled
    from, so a deterministic Clifford is folded into the reference and the reported
    observable does not move at all. Measured on a d=3 rotated memory-Z circuit: ``X 1``
    before the final ``M`` gives observable 0 and **zero** detection events, while
    ``X_ERROR(1) 1`` gives observable 1 and one detection event. Noise instructions are
    excluded from the reference, so a probability-1 error channel is the only way to
    inject a correction the detector sampler actually reports. An oracle built on
    deterministic gates passes every test while measuring nothing.

    Args:
        correction_x: bool ``(n_data_qubits,)``, unpacked.
        correction_z: bool ``(n_data_qubits,)``, unpacked.
    """
    layer = final_data_measurement_layer(circuit)
    if not layer:
        raise ValueError("no final data measurement layer to inject before")
    first = layer[0]
    ids = schema.stim_qubit_ids()
    x_targets = [int(ids[i]) for i in np.flatnonzero(np.asarray(correction_x, dtype=bool))]
    z_targets = [int(ids[i]) for i in np.flatnonzero(np.asarray(correction_z, dtype=bool))]

    out = stim.Circuit()
    for position, instruction in enumerate(circuit.flattened()):
        if position == first:
            if x_targets:
                out.append("X_ERROR", x_targets, 1.0)
            if z_targets:
                out.append("Z_ERROR", z_targets, 1.0)
        out.append(instruction)
    return out
